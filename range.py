"""Opponent hand range modeling + range-weighted equity + fold equity.

Replaces the "vs random opponent" approximation in equity.py with a tightening
distribution that updates on each observed opponent action. Decisions then use
both equity_vs_range and fold_freq_estimate instead of equity-vs-random plus
a flat discount.
"""

from __future__ import annotations

import bisect
import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from engine import Card, create_deck, evaluate_best_hand

HandKey = FrozenSet[Tuple[str, str]]

POSTFLOP_STRENGTH_CACHE_MAX = 100_000

_postflop_strength_cache: "OrderedDict[Tuple[HandKey, Tuple[Tuple[str, str], ...]], float]" = OrderedDict()
_preflop_strength_cache: Dict[HandKey, float] = {}


def _card_key(card: Card) -> Tuple[str, str]:
    return (card.rank, card.suit)


def _hand_to_key(card_a: Card, card_b: Card) -> HandKey:
    return frozenset({_card_key(card_a), _card_key(card_b)})


def _board_key(board: Sequence[Card]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_card_key(c) for c in board))


def _key_to_cards(key: HandKey) -> Tuple[Card, Card]:
    items = sorted(key)
    return Card(items[0][0], items[0][1]), Card(items[1][0], items[1][1])


def preflop_strength(hand: HandKey) -> float:
    """Equity vs random preflop, cached. Pulls from equity.py's 169-bucket table."""
    cached = _preflop_strength_cache.get(hand)
    if cached is not None:
        return cached
    from equity import _load_or_build_preflop_table, hand_key

    a, b = _key_to_cards(hand)
    score = _load_or_build_preflop_table()[hand_key(a, b)]
    _preflop_strength_cache[hand] = score
    return score


def postflop_strength(hand: HandKey, board: Sequence[Card]) -> float:
    """0~1 score for a 2-card hand on the given board, cached.

    Combines hand category (0=high-card, 8=straight-flush) with kicker info.
    """
    key = (hand, _board_key(board))
    cached = _postflop_strength_cache.get(key)
    if cached is not None:
        _postflop_strength_cache.move_to_end(key)
        return cached
    a, b = _key_to_cards(hand)
    value = evaluate_best_hand([a, b] + list(board))
    category = value[0]
    kicker_sum = sum(value[1][:2]) if value[1] else 0
    score = category / 8.0 + (kicker_sum / 28.0) * 0.08
    score = min(1.0, score)
    _postflop_strength_cache[key] = score
    if len(_postflop_strength_cache) > POSTFLOP_STRENGTH_CACHE_MAX:
        _postflop_strength_cache.popitem(last=False)
    return score


def clear_postflop_strength_cache() -> None:
    """Release per-process postflop cache memory between long training matches."""
    _postflop_strength_cache.clear()


@dataclass
class HandRange:
    """Weighted distribution over the opponent's possible hole-card combinations.

    Starts uniform over all combos that don't conflict with `known_cards`.
    Each `apply_filter` multiplicatively reweights based on a score function.
    """

    hands: Dict[HandKey, float] = field(default_factory=dict)

    def __init__(self, known_cards: Iterable[Card] = ()) -> None:
        excluded = {_card_key(c) for c in known_cards}
        available = [c for c in create_deck() if _card_key(c) not in excluded]
        self.hands = {
            _hand_to_key(a, b): 1.0 for a, b in itertools.combinations(available, 2)
        }

    def num_nonzero(self) -> int:
        return sum(1 for w in self.hands.values() if w > 0)

    def total_weight(self) -> float:
        return sum(self.hands.values())

    def apply_filter(
        self,
        score_fn: Callable[[HandKey], float],
        threshold: float,
        retain: float = 1.0,
        drop: float = 0.05,
    ) -> None:
        for k in list(self.hands.keys()):
            w = self.hands[k]
            if w <= 0:
                continue
            self.hands[k] = w * (retain if score_fn(k) >= threshold else drop)

    def remove_cards(self, new_cards: Iterable[Card]) -> None:
        bad = {_card_key(c) for c in new_cards}
        for k in list(self.hands.keys()):
            if self.hands[k] > 0 and any(card in bad for card in k):
                self.hands[k] = 0.0

    def relax(self, factor: float = 1.5) -> None:
        """Raise low-weight combos toward their non-dropped counterparts (looser opponent)."""
        for k, w in self.hands.items():
            if 0 < w < 1.0:
                self.hands[k] = min(1.0, w * factor)

    def tighten(self, factor: float = 0.7) -> None:
        """Drop low-weight combos further (tighter opponent)."""
        for k, w in self.hands.items():
            if 0 < w < 1.0:
                self.hands[k] = w * factor

    def sample(self, rng: random.Random) -> Optional[Tuple[Card, Card]]:
        keys = [k for k, w in self.hands.items() if w > 0]
        if not keys:
            return None
        weights = [self.hands[k] for k in keys]
        cum = []
        acc = 0.0
        for w in weights:
            acc += w
            cum.append(acc)
        if acc <= 0:
            return None
        r = rng.random() * acc
        idx = bisect.bisect_left(cum, r)
        return _key_to_cards(keys[idx])


def update_preflop(
    range_obj: HandRange,
    action: str,
    position_is_button: bool = False,
) -> None:
    """Tighten range based on a preflop action.

    `drop` values are kept low so that the noise floor doesn't dilute the
    tightened range — but non-zero to preserve room for bluffs/light raises.
    """
    if action in ("fold", "check"):
        return
    if action == "call":
        range_obj.apply_filter(preflop_strength, threshold=0.45, retain=1.0, drop=0.20)
    elif action == "raise-2x":
        range_obj.apply_filter(preflop_strength, threshold=0.58, retain=1.0, drop=0.02)
    elif action in ("raise-3x", "all-in"):
        range_obj.apply_filter(preflop_strength, threshold=0.66, retain=1.0, drop=0.005)


def update_postflop(
    range_obj: HandRange,
    action: str,
    board: Sequence[Card],
    street: str,
) -> None:
    """Tighten range based on a postflop action on the given board state."""
    if action == "fold":
        return
    score = lambda h: postflop_strength(h, board)
    if action == "check":
        range_obj.apply_filter(score, threshold=0.18, retain=0.7, drop=1.0)
    elif action == "call":
        range_obj.apply_filter(score, threshold=0.13, retain=1.0, drop=0.25)
    elif action == "raise-2x":
        range_obj.apply_filter(score, threshold=0.20, retain=1.0, drop=0.10)
    elif action in ("raise-3x", "all-in"):
        range_obj.apply_filter(score, threshold=0.30, retain=1.0, drop=0.04)


def community_at_street(community: Sequence[Card], street: str) -> List[Card]:
    if street == "preflop":
        return []
    if street == "flop":
        return list(community[:3])
    if street == "turn":
        return list(community[:4])
    return list(community[:5])


def equity_vs_range(
    my_hole: Sequence[Card],
    board: Sequence[Card],
    opp_range: HandRange,
    samples: int = 300,
    rng: Optional[random.Random] = None,
) -> float:
    """MC equity where opponent hands are sampled from opp_range, not uniformly."""
    if rng is None:
        rng = random.Random()
    if opp_range.total_weight() <= 0:
        return 0.5

    my_known = {_card_key(c) for c in list(my_hole) + list(board)}
    keys = [k for k, w in opp_range.hands.items() if w > 0 and not (k & my_known)]
    weights = [opp_range.hands[k] for k in keys]
    if not keys:
        return 0.5
    cum: List[float] = []
    acc = 0.0
    for w in weights:
        acc += w
        cum.append(acc)
    total_w = acc
    if total_w <= 0:
        return 0.5

    deck = create_deck()
    my_hole_list = list(my_hole)
    board_list = list(board)
    board_remainder = 5 - len(board)

    wins = 0.0
    ties = 0.0
    n = 0
    for _ in range(samples):
        r = rng.random() * total_w
        idx = bisect.bisect_left(cum, r)
        if idx >= len(keys):
            idx = len(keys) - 1
        opp_key = keys[idx]
        if opp_key & my_known:
            continue
        opp_cards = list(_key_to_cards(opp_key))

        if board_remainder > 0:
            opp_card_keys = {_card_key(c) for c in opp_cards}
            remaining = [c for c in deck if _card_key(c) not in my_known and _card_key(c) not in opp_card_keys]
            future = rng.sample(remaining, board_remainder)
        else:
            future = []
        full_board = board_list + future
        my_best = evaluate_best_hand(my_hole_list + full_board)
        opp_best = evaluate_best_hand(opp_cards + full_board)
        if my_best > opp_best:
            wins += 1
        elif my_best == opp_best:
            ties += 1
        n += 1

    if n == 0:
        return 0.5
    return (wins + ties / 2) / n


def fold_freq_estimate(
    opp_range: HandRange,
    board: Sequence[Card],
    my_bet_chips: int,
    pot_chips: int,
    opp_calling_threshold: float = 0.13,
) -> float:
    """Estimate the fraction of opponent's range that folds to a bet of `my_bet_chips`
    into a pot of `pot_chips`. Larger bets push the calling threshold up.

    postflop_strength uses category/8 normalization so 0.13 ≈ "needs at least a pair";
    0.25 ≈ "needs two pair"; 0.375 ≈ "needs a set".
    """
    total = opp_range.total_weight()
    if total <= 0:
        return 0.0
    sizing_factor = 1.0 + 0.6 * (my_bet_chips / max(1, pot_chips))
    threshold = min(0.55, opp_calling_threshold * sizing_factor)

    fold_weight = 0.0
    if not board:
        for hand, w in opp_range.hands.items():
            if w <= 0:
                continue
            if preflop_strength(hand) < threshold:
                fold_weight += w
    else:
        for hand, w in opp_range.hands.items():
            if w <= 0:
                continue
            if postflop_strength(hand, board) < threshold:
                fold_weight += w
    return fold_weight / total
