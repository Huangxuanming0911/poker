"""Offline self-play training: evolve AIParams via (1+lambda)-ES.

Usage:
  python train.py --generations 20 --lambda 6 --hands 1500 --workers 4
  python train.py --eval-only --hands 2000

Strategy:
  Each generation, perturb the parent params with Gaussian noise (sigma=0.06)
  to produce `lambda` offspring. Each offspring plays a head-to-head match
  vs the parent. The best offspring replaces the parent IFF its bb/hand
  exceeds SIGNIFICANCE_THRESHOLD (filters out noise-driven "wins").

  Every 5 generations a regression check vs default params catches drift.
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import random
import time
from typing import Tuple

from advanced_ai import AdvancedPokerAI
from ai import BasicPokerAI
from ai_params import AIParams
from engine import Game
from persistence import AI_PARAMS_PATH

DEFAULT_HANDS = 1500
DEFAULT_GENERATIONS = 20
DEFAULT_LAMBDA = 6
SIGMA = 0.06
SIGNIFICANCE_THRESHOLD = 0.05
STARTING_STACK = 5000
RESET_FLOOR = 500


def _run_hands(
    ai_a: AdvancedPokerAI,
    ai_b: AdvancedPokerAI,
    n_hands: int,
) -> Tuple[int, int]:
    game = Game(["A", "B"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a = 0
    hands_played = 0

    def provider(ctx):
        return ai_a.decide(ctx) if ctx.player_index == 0 else ai_b.decide(ctx)

    for _ in range(n_hands):
        game.play_hand(provider)
        hands_played += 1
        if min(p.stack for p in game.players) < RESET_FLOOR:
            net_a += game.players[0].stack - STARTING_STACK
            game = Game(["A", "B"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a += game.players[0].stack - STARTING_STACK
    return net_a, hands_played


def run_match(
    params_a: AIParams,
    params_b: AIParams,
    n_hands: int,
    seed: int,
    training_mode: bool = True,
) -> float:
    """A vs B for n_hands; returns A's bb/hand."""
    random.seed(seed)
    rng = random.Random(seed)
    ai_a = AdvancedPokerAI(
        "A", params_a, seed=rng.randint(0, 2**31 - 1), training_mode=training_mode
    )
    ai_b = AdvancedPokerAI(
        "B", params_b, seed=rng.randint(0, 2**31 - 1), training_mode=training_mode
    )
    net_a, hands = _run_hands(ai_a, ai_b, n_hands)
    big_blind = 20
    return net_a / max(1, hands) / big_blind


def run_match_vs_basic(params: AIParams, n_hands: int, seed: int) -> float:
    """AdvancedAI(params) vs BasicPokerAI; returns Advanced's bb/hand."""
    random.seed(seed)
    rng = random.Random(seed)
    adv = AdvancedPokerAI("Adv", params, seed=rng.randint(0, 2**31 - 1))
    basic = BasicPokerAI("Basic", seed=rng.randint(0, 2**31 - 1))
    game = Game(["Adv", "Basic"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a = 0
    hands = 0

    def provider(ctx):
        return adv.decide(ctx) if ctx.player_index == 0 else basic.decide(ctx)

    for _ in range(n_hands):
        game.play_hand(provider)
        hands += 1
        if min(p.stack for p in game.players) < RESET_FLOOR:
            net_a += game.players[0].stack - STARTING_STACK
            game = Game(["Adv", "Basic"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a += game.players[0].stack - STARTING_STACK
    return net_a / max(1, hands) / 20


def perturb(parent: AIParams, sigma: float, rng: random.Random) -> AIParams:
    new_values = {}
    for f in dataclasses.fields(AIParams):
        v = getattr(parent, f.name)
        new_values[f.name] = v + rng.gauss(0, sigma)
    return AIParams(**new_values).clipped()


def _eval_offspring(args):
    parent, child, n_hands, seed = args
    return run_match(child, parent, n_hands, seed)


def train(generations: int, lambda_: int, n_hands: int, workers: int, out_path: str) -> None:
    parent = AIParams.load_or_default(out_path)
    rng = random.Random(0)

    print(
        f"Training: generations={generations} lambda={lambda_} hands/match={n_hands} workers={workers}"
    )

    baseline = run_match(parent, AIParams(), n_hands * 2, seed=99999)
    print(f"Initial: parent vs default = {baseline:+.3f} bb/hand")

    best_known = baseline

    for gen in range(generations):
        offspring = [perturb(parent, SIGMA, rng) for _ in range(lambda_)]
        seeds = [10_000 * (gen + 1) + i for i in range(lambda_)]
        tasks = [(parent, child, n_hands, seed) for child, seed in zip(offspring, seeds)]

        t0 = time.perf_counter()
        if workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                fitnesses = pool.map(_eval_offspring, tasks)
        else:
            fitnesses = [_eval_offspring(t) for t in tasks]
        elapsed = time.perf_counter() - t0

        best_idx = max(range(lambda_), key=lambda i: fitnesses[i])
        best_fit = fitnesses[best_idx]
        fits_str = " ".join(f"{f:+.2f}" for f in fitnesses)

        if best_fit > SIGNIFICANCE_THRESHOLD:
            parent = offspring[best_idx]
            parent.save(out_path)
            print(f"gen {gen:2d}: best={best_fit:+.3f}  [{fits_str}]  {elapsed:.0f}s  ACCEPT (saved)")
        else:
            print(f"gen {gen:2d}: best={best_fit:+.3f}  [{fits_str}]  {elapsed:.0f}s  reject")

        if (gen + 1) % 5 == 0:
            check = run_match(parent, AIParams(), n_hands * 2, seed=99999 + gen)
            print(f"  regression check (parent vs default) @ gen{gen+1}: {check:+.3f} bb/hand")
            if check + 0.15 < best_known:
                print("  regression detected — reverting to last saved default")
                parent = AIParams.load_or_default(out_path)
            best_known = max(best_known, check)

    parent.save(out_path)
    final = run_match_vs_basic(parent, n_hands * 2, seed=88888)
    print(f"\nFinal: AdvancedAI(trained) vs BasicPokerAI = {final:+.3f} bb/hand")
    print(f"Saved to {out_path}")


def eval_only(n_hands: int, out_path: str, seeds: int) -> None:
    params = AIParams.load_or_default(out_path)
    print(f"Evaluating {seeds} seeds x {n_hands} hands: AdvancedAI(loaded) vs BasicPokerAI")
    results = []
    for s in range(seeds):
        r = run_match_vs_basic(params, n_hands, seed=1000 + s)
        results.append(r)
        print(f"  seed={s}: {r:+.3f} bb/hand")
    mean = sum(results) / len(results)
    print(f"\nMean: {mean:+.3f} bb/hand over {seeds} matches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--lambda", type=int, default=DEFAULT_LAMBDA, dest="lambda_")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for --eval-only")
    parser.add_argument("--out", default=AI_PARAMS_PATH)
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args.hands, args.out, args.seeds)
    else:
        train(args.generations, args.lambda_, args.hands, args.workers, args.out)


if __name__ == "__main__":
    main()
