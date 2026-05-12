"""Hybrid equity estimation for Texas Hold'em.

Dispatch by street:
  preflop  -> 169-bucket precomputed lookup table (effectively O(1))
  flop     -> Monte Carlo over remaining 47 cards
  turn     -> Monte Carlo over remaining 46 cards
  river    -> exact enumeration of C(45,2)=990 opponent hands
"""

from __future__ import annotations

import itertools
import json
import os
import random
from typing import Iterable, List, Optional, Sequence

from engine import Card, create_deck, evaluate_best_hand

PREFLOP_CACHE_FILENAME = "equity_preflop_table.json"
PREFLOP_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), PREFLOP_CACHE_FILENAME)
DEFAULT_PREFLOP_SAMPLES = 1000

_PREFLOP_TABLE: Optional[dict] = None


def hand_key(card_a: Card, card_b: Card) -> str:
    """Canonical 169-bucket label: 'AA', 'AKs', 'AKo', etc. High rank first."""
    if card_a.value() < card_b.value():
        card_a, card_b = card_b, card_a
    if card_a.rank == card_b.rank:
        return card_a.rank + card_b.rank
    suffix = "s" if card_a.suit == card_b.suit else "o"
    return card_a.rank + card_b.rank + suffix


def _representative_hand(key: str) -> tuple[Card, Card]:
    if len(key) == 2:
        return Card(key[0], "♠"), Card(key[1], "♥")
    rank_a, rank_b, kind = key[0], key[1], key[2]
    if kind == "s":
        return Card(rank_a, "♠"), Card(rank_b, "♠")
    return Card(rank_a, "♠"), Card(rank_b, "♥")


def _all_bucket_keys() -> List[str]:
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    keys = []
    for i, hi in enumerate(ranks):
        for j, lo in enumerate(ranks):
            if i == j:
                keys.append(hi + hi)
            elif i < j:
                keys.append(hi + lo + "s")
                keys.append(hi + lo + "o")
    return keys


def _build_preflop_table(samples_per_bucket: int, seed: int) -> dict:
    rng = random.Random(seed)
    deck = create_deck()
    table: dict = {}
    for key in _all_bucket_keys():
        hole_a, hole_b = _representative_hand(key)
        held = {(hole_a.rank, hole_a.suit), (hole_b.rank, hole_b.suit)}
        remaining = [c for c in deck if (c.rank, c.suit) not in held]
        wins = ties = 0
        for _ in range(samples_per_bucket):
            sample = rng.sample(remaining, 7)
            opp = sample[:2]
            board = sample[2:7]
            my_best = evaluate_best_hand([hole_a, hole_b] + board)
            opp_best = evaluate_best_hand(opp + board)
            if my_best > opp_best:
                wins += 1
            elif my_best == opp_best:
                ties += 1
        table[key] = (wins + ties / 2) / samples_per_bucket
    return table


def _load_or_build_preflop_table() -> dict:
    global _PREFLOP_TABLE
    if _PREFLOP_TABLE is not None:
        return _PREFLOP_TABLE
    if os.path.exists(PREFLOP_CACHE_PATH):
        with open(PREFLOP_CACHE_PATH, "r") as f:
            _PREFLOP_TABLE = json.load(f)
        return _PREFLOP_TABLE
    print(
        f"[equity] preflop table missing; building {DEFAULT_PREFLOP_SAMPLES} samples/bucket (~10s)..."
    )
    _PREFLOP_TABLE = _build_preflop_table(DEFAULT_PREFLOP_SAMPLES, seed=0)
    with open(PREFLOP_CACHE_PATH, "w") as f:
        json.dump(_PREFLOP_TABLE, f, indent=2, sort_keys=True)
    print(f"[equity] preflop table saved to {PREFLOP_CACHE_PATH}")
    return _PREFLOP_TABLE


def _preflop_equity(hole: Sequence[Card]) -> float:
    return _load_or_build_preflop_table()[hand_key(hole[0], hole[1])]


def _remaining_deck(known: Iterable[Card]) -> List[Card]:
    known_keys = {(c.rank, c.suit) for c in known}
    return [c for c in create_deck() if (c.rank, c.suit) not in known_keys]


def _mc_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    samples: int,
    rng: random.Random,
) -> float:
    board_remainder = 5 - len(board)
    cards_needed = 2 * num_opponents + board_remainder
    remaining = _remaining_deck(list(hole) + list(board))
    hole_list = list(hole)
    board_list = list(board)
    wins = 0.0
    ties = 0.0
    for _ in range(samples):
        sample = rng.sample(remaining, cards_needed)
        full_board = board_list + sample[:board_remainder]
        opp_offset = board_remainder
        my_best = evaluate_best_hand(hole_list + full_board)
        tied = False
        lost = False
        for opp_i in range(num_opponents):
            opp_hand = sample[opp_offset + 2 * opp_i : opp_offset + 2 * (opp_i + 1)]
            opp_best = evaluate_best_hand(list(opp_hand) + full_board)
            if opp_best > my_best:
                lost = True
                break
            if opp_best == my_best:
                tied = True
        if lost:
            continue
        if tied:
            ties += 1
        else:
            wins += 1
    return (wins + ties / 2) / samples


def _river_exact_equity(hole: Sequence[Card], board: Sequence[Card]) -> float:
    remaining = _remaining_deck(list(hole) + list(board))
    board_list = list(board)
    my_best = evaluate_best_hand(list(hole) + board_list)
    wins = ties = total = 0
    for opp in itertools.combinations(remaining, 2):
        opp_best = evaluate_best_hand(list(opp) + board_list)
        if my_best > opp_best:
            wins += 1
        elif my_best == opp_best:
            ties += 1
        total += 1
    return (wins + ties / 2) / total


def estimate_equity(
    hole: Sequence[Card],
    board: Sequence[Card] = (),
    num_opponents: int = 1,
    mc_samples: int = 500,
    rng: Optional[random.Random] = None,
) -> float:
    """Win+tie equity given known hole cards and current board.

    Heads-up (num_opponents=1) gets the fast paths. Multi-way always uses MC.
    """
    if rng is None:
        rng = random.Random()
    n = len(board)
    if n == 0:
        if num_opponents == 1:
            return _preflop_equity(hole)
        return _mc_equity(hole, board, num_opponents, mc_samples, rng)
    if n in (3, 4):
        return _mc_equity(hole, board, num_opponents, mc_samples, rng)
    if n == 5:
        if num_opponents == 1:
            return _river_exact_equity(hole, board)
        return _mc_equity(hole, board, num_opponents, mc_samples, rng)
    raise ValueError(f"unexpected board length: {n}")


def pick_mc_samples(
    pot: int,
    effective_stack: int,
    big_blind: int,
    is_critical: bool = False,
    training_mode: bool = False,
) -> int:
    """Adaptive sample count for flop/turn MC."""
    if training_mode:
        return 200
    base = 500
    if effective_stack > 0 and pot >= 0.30 * effective_stack:
        base = 2000
    if is_critical:
        base = max(base, 2000)
    return base


if __name__ == "__main__":
    import sys

    rebuild = "--rebuild" in sys.argv
    samples = DEFAULT_PREFLOP_SAMPLES
    for arg in sys.argv[1:]:
        if arg.startswith("--samples="):
            samples = int(arg.split("=", 1)[1])
    if rebuild and os.path.exists(PREFLOP_CACHE_PATH):
        os.remove(PREFLOP_CACHE_PATH)
        print(f"[equity] removed existing cache, rebuilding with {samples} samples/bucket")
        _PREFLOP_TABLE = None
    if rebuild:
        global_table = _build_preflop_table(samples, seed=0)
        with open(PREFLOP_CACHE_PATH, "w") as f:
            json.dump(global_table, f, indent=2, sort_keys=True)
        print(f"[equity] saved to {PREFLOP_CACHE_PATH}")
    else:
        _load_or_build_preflop_table()
    table = _load_or_build_preflop_table()
    print(f"buckets: {len(table)}")
    for k in ["AA", "KK", "AKs", "AKo", "72o", "22"]:
        print(f"  {k}: {table.get(k, '?'):.4f}")
