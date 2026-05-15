"""CFR-Lite: lightweight counterfactual regret collection for training.

At each decision point during training self-play, we estimate the EV of all
available actions using analytic approximations (no full rollout). The regret
(EV_alt - EV_chosen) is accumulated per bucket. After a generation, the regret
pattern suggests which parameters should be adjusted and in which direction.

Bucket key: (street, equity_bucket, pot_odds_bucket, position)
  - street: 4 values (preflop/flop/turn/river)
  - equity_bucket: 4 ranges [0,0.3), [0.3,0.5), [0.5,0.7), [0.7,1.0]
  - pot_odds_bucket: 3 ranges [0,0.2), [0.2,0.4), [0.4,1.0]
  - position: 2 (IP / OOP)
  Total: 4 * 4 * 3 * 2 = 96 buckets
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

from ai_params import AIParams

BucketKey = Tuple[str, int, int, int]

ACTION_TO_PARAM_MAP: Dict[str, List[Tuple[str, float]]] = {
    "raise-3x": [("value_3x_threshold", -1.0)],
    "raise-2x": [("value_2x_threshold", -1.0), ("bluff_execute_freq", +0.5)],
    "call": [("call_ev_threshold", +1.0)],
    "fold": [("call_ev_threshold", -0.5), ("bluff_execute_freq", -0.3)],
    "check": [],
    "all-in": [("short_stack_jam_threshold", -0.8)],
}


def _equity_bucket(equity: float) -> int:
    if equity < 0.3:
        return 0
    if equity < 0.5:
        return 1
    if equity < 0.7:
        return 2
    return 3


def _pot_odds_bucket(pot_odds: float) -> int:
    if pot_odds < 0.2:
        return 0
    if pot_odds < 0.4:
        return 1
    return 2


def make_bucket_key(street: str, equity: float, pot_odds: float, in_position: bool) -> BucketKey:
    return (street, _equity_bucket(equity), _pot_odds_bucket(pot_odds), int(in_position))


class CFRCollector:
    """Collects counterfactual regret during training matches."""

    def __init__(self):
        self.regrets: Dict[BucketKey, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.visit_count: Dict[BucketKey, int] = defaultdict(int)

    def record(
        self,
        bucket_key: BucketKey,
        chosen_action: str,
        action_evs: Dict[str, float],
    ) -> None:
        chosen_ev = action_evs.get(chosen_action, 0.0)
        for alt, ev in action_evs.items():
            if alt == chosen_action:
                continue
            regret = ev - chosen_ev
            if regret > 0:
                self.regrets[bucket_key][alt] += regret
        self.visit_count[bucket_key] += 1

    def reset(self) -> None:
        self.regrets.clear()
        self.visit_count.clear()

    def total_visits(self) -> int:
        return sum(self.visit_count.values())

    def suggest_param_gradient(self, current_params: AIParams) -> Dict[str, float]:
        """Analyze accumulated regret to suggest parameter adjustments.

        For each action with high regret, look up which parameters would make
        that action more likely, and accumulate a signed gradient.
        """
        if self.total_visits() == 0:
            return {}

        gradient: Dict[str, float] = defaultdict(float)
        total_regret = sum(
            r for bucket in self.regrets.values() for r in bucket.values()
        )
        if total_regret <= 0:
            return {}

        for bucket_key, action_regrets in self.regrets.items():
            visits = max(1, self.visit_count[bucket_key])
            for action, regret in action_regrets.items():
                normalized = regret / visits
                mappings = ACTION_TO_PARAM_MAP.get(action, [])
                for param_name, direction in mappings:
                    gradient[param_name] += direction * normalized

        if not gradient:
            return {}
        max_abs = max(abs(v) for v in gradient.values())
        if max_abs > 0:
            for k in gradient:
                gradient[k] /= max_abs

        return dict(gradient)


def estimate_action_evs(
    equity: float,
    pot: int,
    call_amount: int,
    action_targets: Dict[str, int],
    available_actions: List[str],
    fold_freq: float,
    big_blind: int,
) -> Dict[str, float]:
    """Analytic EV approximation for each available action.

    These are rough estimates — good enough to generate directional regret
    signals, not precise enough for actual decision-making.
    """
    evs: Dict[str, float] = {}
    bb = max(1, big_blind)

    for action in available_actions:
        if action == "fold":
            evs[action] = 0.0
        elif action == "check":
            evs[action] = (equity * pot - (1 - equity) * 0) / bb
        elif action == "call":
            pot_after = pot + call_amount
            evs[action] = (equity * pot_after - (1 - equity) * call_amount) / bb
        elif action in ("raise-2x", "raise-3x", "all-in"):
            target = action_targets.get(action, pot)
            raise_cost = target - call_amount
            pot_if_called = pot + target + call_amount
            ev_called = equity * pot_if_called - (1 - equity) * raise_cost
            ev_fold = fold_freq * (pot + call_amount)
            evs[action] = (fold_freq * ev_fold + (1 - fold_freq) * ev_called) / bb
        else:
            evs[action] = 0.0

    return evs
