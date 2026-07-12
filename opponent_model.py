"""Per-opponent statistical profiling for adaptive play.

Stats tracked per opponent (Bayesian-smoothed for cold-start stability):
  VPIP            — Voluntarily Put $ In Pot (preflop, excluding forced blinds)
  PFR             — Preflop Raise frequency
  AF              — Aggression Factor: (bet+raise) / call
  Fold-to-Cbet    — opp folded on flop after preflop aggressor bet
  WTSD            — Went To Showdown frequency

The AI consults these in AdvancedPokerAI._adjusted_params to shift bluff/value
frequencies. Cold-start (< MIN_HANDS) returns defaults unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Iterable, List, Optional, Tuple

from persistence import HandLog, load_profiles, save_profiles

MIN_HANDS_FOR_ADAPTATION = 20
SMOOTHING_ALPHA = 2.0
SMOOTHING_BETA = 2.0

AGGRESSIVE_ACTIONS = {"raise-2x", "raise-3x", "all-in"}
PASSIVE_ACTIONS = {"call"}


@dataclass
class OpponentStats:
    hands_observed: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    aggressive_count: int = 0
    passive_count: int = 0
    cbet_faced_count: int = 0
    cbet_folded_count: int = 0
    showdown_count: int = 0
    saw_flop_count: int = 0

    def vpip(self) -> float:
        return _smooth(self.vpip_count, self.hands_observed)

    def pfr(self) -> float:
        return _smooth(self.pfr_count, self.hands_observed)

    def af(self) -> float:
        if self.passive_count == 0:
            return _smooth(self.aggressive_count, max(self.aggressive_count, 1)) * 3.0
        return self.aggressive_count / self.passive_count

    def fold_to_cbet(self) -> float:
        return _smooth(self.cbet_folded_count, self.cbet_faced_count)

    def wtsd(self) -> float:
        return _smooth(self.showdown_count, self.saw_flop_count)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentStats":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _smooth(positive: int, total: int) -> float:
    return (positive + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA + SMOOTHING_BETA)


class OpponentModel:
    def __init__(self, profile_path: Optional[str] = None) -> None:
        self.profile_path = profile_path
        self._profiles: Dict[str, OpponentStats] = {}
        if profile_path:
            raw = load_profiles(profile_path)
            for name, data in raw.items():
                self._profiles[name] = OpponentStats.from_dict(data)

    def stats_for(self, name: str) -> OpponentStats:
        if name not in self._profiles:
            self._profiles[name] = OpponentStats()
        return self._profiles[name]

    def names(self) -> List[str]:
        return list(self._profiles.keys())

    def observe(self, log: HandLog, self_name: Optional[str] = None) -> None:
        """Update stats for every player that isn't `self_name`."""
        identities = log.player_ids or log.player_names
        for idx, identity in enumerate(identities):
            display_name = log.player_names[idx] if idx < len(log.player_names) else identity
            if identity == self_name or display_name == self_name:
                continue
            stats = self.stats_for(identity)
            _update_from_log(stats, log, idx)

    def save(self) -> None:
        if not self.profile_path:
            return
        payload = {name: stats.to_dict() for name, stats in self._profiles.items()}
        save_profiles(self.profile_path, payload)


def _update_from_log(stats: OpponentStats, log: HandLog, player_idx: int) -> None:
    stats.hands_observed += 1

    preflop_actions = [a for a in log.actions if a[0] == "preflop" and a[1] == player_idx]
    flop_actions = [a for a in log.actions if a[0] == "flop" and a[1] == player_idx]
    nonpreflop_actions = [a for a in log.actions if a[0] != "preflop" and a[1] == player_idx]

    if preflop_actions:
        any_voluntary = any(act not in ("fold", "check") for _, _, act, _ in preflop_actions)
        any_raise = any(act in AGGRESSIVE_ACTIONS for _, _, act, _ in preflop_actions)
        if any_voluntary:
            stats.vpip_count += 1
        if any_raise:
            stats.pfr_count += 1

    for _, _, act, _ in preflop_actions + nonpreflop_actions:
        if act in AGGRESSIVE_ACTIONS:
            stats.aggressive_count += 1
        elif act in PASSIVE_ACTIONS:
            stats.passive_count += 1

    saw_flop = any(a[0] == "flop" and a[1] == player_idx for a in log.actions)
    if saw_flop:
        stats.saw_flop_count += 1

    pfr_idx = _preflop_aggressor_idx(log.actions)
    if pfr_idx is not None and pfr_idx != player_idx:
        opp_flop_bet = any(
            a[0] == "flop" and a[1] == pfr_idx and a[2] in AGGRESSIVE_ACTIONS
            for a in log.actions
        )
        if opp_flop_bet and flop_actions:
            stats.cbet_faced_count += 1
            first_response = next(
                (act for _, _, act, _ in flop_actions),
                None,
            )
            if first_response == "fold":
                stats.cbet_folded_count += 1

    final_stage_reached_flop = log.final_stage in ("flop", "turn", "river")
    showed_down = (
        log.final_stage == "river"
        and not _player_folded(log, player_idx)
        and len([p for p in range(len(log.player_names)) if not _player_folded(log, p)]) >= 2
    )
    if showed_down:
        stats.showdown_count += 1
    _ = final_stage_reached_flop  # currently unused; reserved for richer stats


def _preflop_aggressor_idx(actions: Iterable[Tuple[str, int, str, int]]) -> Optional[int]:
    last_raiser = None
    for street, idx, act, _ in actions:
        if street != "preflop":
            break
        if act in AGGRESSIVE_ACTIONS:
            last_raiser = idx
    return last_raiser


def _player_folded(log: HandLog, player_idx: int) -> bool:
    return any(act == "fold" and idx == player_idx for _, idx, act, _ in log.actions)


def build_hand_log_from_game(
    game,
    starting_stacks: List[int],
    button_index_at_start: int,
    payoffs: Dict[str, int],
    final_stage: str,
    hole_cards_known: Optional[Dict[str, List[str]]] = None,
    player_ids: Optional[List[str]] = None,
) -> HandLog:
    """Snapshot the just-completed hand. Note: game.button_index has been advanced
    by play_hand for the next hand, so the caller must pass the button_index that
    was active at the start of this hand."""
    import time as _time

    return HandLog(
        timestamp=_time.time(),
        player_names=[p.name for p in game.players],
        player_ids=player_ids or [p.name for p in game.players],
        starting_stacks=list(starting_stacks),
        button_index=button_index_at_start,
        small_blind=game.small_blind,
        big_blind=game.big_blind,
        actions=list(game.actions_this_hand),
        community=[str(c) for c in game.community_cards],
        hole_cards=hole_cards_known or {},
        payoffs=dict(payoffs),
        final_stage=final_stage,
    )
