"""Offline self-play training with CMA-ES + Fictitious Self-Play opponent pool.

Usage:
  python train.py --generations 20 --hands 1500 --workers 4
  python train.py --generations 20 --hands 1500 --workers 4 --cfr
  python train.py --eval-only --hands 2000 --seeds 5

Strategy:
  CMA-ES (Covariance Matrix Adaptation) evolves the 15-dim AIParams vector.
  Each candidate is evaluated by playing head-to-head matches against opponents
  sampled from a Fictitious Self-Play pool (recent best params + default).

  Optional --cfr flag enables CFR-Lite regret collection during training matches,
  using the accumulated regret signal to bias the CMA-ES mean after each generation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="Could not import matplotlib")

import cma

from advanced_ai import AdvancedPokerAI
from ai import BasicPokerAI
from ai_params import AIParams
from engine import Game
from persistence import AI_PARAMS_PATH

DEFAULT_HANDS = 1500
DEFAULT_GENERATIONS = 20
STARTING_STACK = 5000
RESET_FLOOR = 500
OPPONENT_POOL_PATH = os.path.join("data", "opponent_pool.json")


# ----------------------------------------------------------------------
# Opponent Pool (Fictitious Self-Play)
# ----------------------------------------------------------------------

class OpponentPool:
    def __init__(self, max_size: int = 10):
        self.pool: List[List[float]] = [AIParams().to_vector()]
        self.max_size = max_size

    def add(self, params: AIParams) -> None:
        self.pool.append(params.to_vector())
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample(self, rng: random.Random) -> AIParams:
        vec = rng.choice(self.pool)
        return AIParams.from_vector(vec)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"pool": self.pool}, f)

    @classmethod
    def load(cls, path: str) -> "OpponentPool":
        obj = cls()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                obj.pool = data.get("pool", obj.pool)
            except (json.JSONDecodeError, KeyError):
                pass
        return obj

    def __len__(self) -> int:
        return len(self.pool)


# ----------------------------------------------------------------------
# Match runner
# ----------------------------------------------------------------------

def run_match(
    params_a: AIParams,
    params_b: AIParams,
    n_hands: int,
    seed: int,
    training_mode: bool = True,
    cfr_collector=None,
) -> float:
    """A vs B for n_hands; returns A's bb/hand."""
    rng = random.Random(seed)
    ai_a = AdvancedPokerAI(
        "A", params_a, seed=rng.randint(0, 2**31 - 1), training_mode=training_mode
    )
    if cfr_collector is not None:
        ai_a.cfr_collector = cfr_collector
    ai_b = AdvancedPokerAI(
        "B", params_b, seed=rng.randint(0, 2**31 - 1), training_mode=training_mode
    )
    game = Game(["A", "B"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a = 0
    hands = 0

    def provider(ctx):
        return ai_a.decide(ctx) if ctx.player_index == 0 else ai_b.decide(ctx)

    for _ in range(n_hands):
        game.play_hand(provider)
        hands += 1
        if min(p.stack for p in game.players) < RESET_FLOOR:
            net_a += game.players[0].stack - STARTING_STACK
            game = Game(["A", "B"], starting_stack=STARTING_STACK, small_blind=10, big_blind=20)
    net_a += game.players[0].stack - STARTING_STACK
    return net_a / max(1, hands) / 20


def run_match_vs_basic(params: AIParams, n_hands: int, seed: int) -> float:
    """AdvancedAI(params) vs BasicPokerAI; returns Advanced's bb/hand."""
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


# ----------------------------------------------------------------------
# Parallel evaluation helper (for multiprocessing)
# ----------------------------------------------------------------------

def _eval_candidate(args):
    import warnings
    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    candidate_vec, opp_vec, n_hands, seed = args
    candidate = AIParams.from_vector(candidate_vec)
    opp = AIParams.from_vector(opp_vec)
    return run_match(candidate, opp, n_hands, seed, training_mode=True)


# ----------------------------------------------------------------------
# CMA-ES Training
# ----------------------------------------------------------------------

def train(
    generations: int,
    n_hands: int,
    workers: int,
    out_path: str,
    use_cfr: bool = False,
) -> None:
    parent = AIParams.load_or_default(out_path)
    pool = OpponentPool.load(OPPONENT_POOL_PATH)

    x0 = parent.to_vector()
    lo, hi = AIParams.vector_bounds()
    sigma0 = 0.08
    popsize = 12

    opts = cma.CMAOptions()
    opts["maxiter"] = generations
    opts["popsize"] = popsize
    opts["bounds"] = [lo, hi]
    opts["seed"] = 42
    opts["verbose"] = -9
    opts["verb_disp"] = 0

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    print(f"Training: CMA-ES popsize={popsize} generations={generations} "
          f"hands/match={n_hands} workers={workers} cfr={use_cfr}")
    print(f"Opponent pool size: {len(pool)}")
    est_seconds_per_gen = popsize * n_hands * 0.04 / max(1, workers)
    est_minutes = est_seconds_per_gen * generations / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes")
    print()

    cfr_collector = None
    if use_cfr:
        from cfr_lite import CFRCollector
        cfr_collector = CFRCollector()

    gen = 0
    mp_pool = None
    if workers > 1:
        mp_pool = mp.Pool(workers)

    start_time = time.perf_counter()

    try:
        while not es.stop():
            t0 = time.perf_counter()
            solutions = es.ask()

            rng = random.Random(gen * 1000)
            tasks = []
            for sol in solutions:
                opp_params = pool.sample(rng)
                seed = rng.randint(0, 2**31 - 1)
                tasks.append((list(sol), opp_params.to_vector(), n_hands, seed))

            print(f"gen {gen:2d}: evaluating {len(tasks)} candidates × {n_hands} hands ...", end="", flush=True)

            if mp_pool is not None:
                fitnesses = mp_pool.map(_eval_candidate, tasks)
            else:
                fitnesses = [_eval_candidate(t) for t in tasks]

            es.tell(solutions, [-f for f in fitnesses])
            elapsed = time.perf_counter() - t0
            total_elapsed = time.perf_counter() - start_time

            best_this_gen = max(fitnesses)
            worst_this_gen = min(fitnesses)
            mean_this_gen = sum(fitnesses) / len(fitnesses)
            sigma_now = es.sigma

            best_params = AIParams.from_vector(es.result.xbest)

            if use_cfr and cfr_collector is not None:
                gradient = cfr_collector.suggest_param_gradient(best_params)
                if gradient:
                    from dataclasses import replace as dc_replace
                    adjustments = {}
                    for param_name, delta in gradient.items():
                        current = getattr(best_params, param_name)
                        adjustments[param_name] = current + 0.005 * delta
                    adjusted = dc_replace(best_params, **adjustments).clipped()
                    es.mean = adjusted.to_vector()
                    cfr_applied = True
                else:
                    cfr_applied = False
                cfr_collector.reset()
            else:
                cfr_applied = False

            if (gen + 1) % 3 == 0:
                pool.add(best_params)

            best_params.save(out_path)

            # Main log line
            fits_str = " ".join(f"{f:+.2f}" for f in sorted(fitnesses, reverse=True)[:5])
            remaining_gens = generations - gen - 1
            eta_s = (elapsed * remaining_gens) if gen == 0 else (total_elapsed / (gen + 1) * remaining_gens)
            eta_min = eta_s / 60

            print(f"\rgen {gen:2d}: best={best_this_gen:+.3f} worst={worst_this_gen:+.3f} "
                  f"mean={mean_this_gen:+.3f} σ={sigma_now:.4f} "
                  f"pool={len(pool)} {elapsed:.0f}s "
                  f"[ETA {eta_min:.0f}min]"
                  f"{' +CFR' if cfr_applied else ''}")
            print(f"        top5: [{fits_str}]")

            # Every 5 gens: regression check + param snapshot
            if (gen + 1) % 5 == 0:
                check = run_match_vs_basic(best_params, n_hands, seed=99999 + gen)
                print(f"  ── checkpoint: vs BasicPokerAI = {check:+.3f} bb/hand")
                # Print key param changes vs default
                default = AIParams()
                diffs = []
                for f in fields(AIParams):
                    cur = getattr(best_params, f.name)
                    orig = getattr(default, f.name)
                    if abs(cur - orig) > 0.005:
                        diffs.append(f"{f.name}={cur:.3f}({orig:.3f})")
                if diffs:
                    print(f"  ── param drift: {', '.join(diffs[:6])}")
                print()

            gen += 1
    finally:
        if mp_pool is not None:
            mp_pool.close()
            mp_pool.join()

    total_time = time.perf_counter() - start_time
    pool.save(OPPONENT_POOL_PATH)
    final_params = AIParams.from_vector(es.result.xbest)
    final_params.save(out_path)

    print(f"\n{'='*60}")
    print(f"Training complete: {gen} generations in {total_time/60:.1f} minutes")
    print(f"Final σ={es.sigma:.4f}, opponent pool size={len(pool)}")
    print(f"\nFinal evaluation (5 seeds × {n_hands} hands vs BasicPokerAI):")
    results = []
    for s in range(5):
        r = run_match_vs_basic(final_params, n_hands, seed=88000 + s)
        results.append(r)
        print(f"  seed={s}: {r:+.3f} bb/hand")
    print(f"  Mean: {sum(results)/5:+.3f} bb/hand")
    print(f"\nSaved to {out_path}")
    print(f"{'='*60}")


# ----------------------------------------------------------------------
# Eval-only mode
# ----------------------------------------------------------------------

def eval_only(n_hands: int, out_path: str, seeds: int) -> None:
    params = AIParams.load_or_default(out_path)
    print(f"Evaluating {seeds} seeds × {n_hands} hands: AdvancedAI(loaded) vs BasicPokerAI")
    results = []
    for s in range(seeds):
        r = run_match_vs_basic(params, n_hands, seed=1000 + s)
        results.append(r)
        print(f"  seed={s}: {r:+.3f} bb/hand")
    mean = sum(results) / len(results)
    print(f"\nMean: {mean:+.3f} bb/hand over {seeds} matches")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AdvancedPokerAI via CMA-ES self-play")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for --eval-only")
    parser.add_argument("--cfr", action="store_true", help="Enable CFR-Lite regret signal")
    parser.add_argument("--out", default=AI_PARAMS_PATH)
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args.hands, args.out, args.seeds)
    else:
        train(args.generations, args.hands, args.workers, args.out, use_cfr=args.cfr)


if __name__ == "__main__":
    main()
