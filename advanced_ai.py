"""AdvancedPokerAI: range-aware decision maker with mixed strategies.

Decision flow:
  1. Replay opponent's actions to construct a tightened HandRange
  2. Estimate equity via Monte Carlo against that weighted range
  3. Compute pot odds, EV(call), SPR, position
  4. Dispatch to mixed-strategy branch (check spot vs facing bet)
  5. For bluff candidates, compute fold equity and bluff iff +EV

Parameterized by AIParams. An optional OpponentModel tightens/loosens the range
heuristics based on observed VPIP/AF.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional, TYPE_CHECKING

from engine import ActionDecision, DecisionContext
from range import (
    HandRange,
    community_at_street,
    equity_vs_range,
    fold_freq_estimate,
    update_postflop,
    update_preflop,
)
from ai_params import AIParams

if TYPE_CHECKING:
    from cfr_lite import CFRCollector
    from opponent_model import OpponentModel


@dataclass
class AdvancedPokerAI:
    name: str
    params: AIParams = field(default_factory=AIParams)
    opponent_model: Optional["OpponentModel"] = None
    seed: Optional[int] = None
    training_mode: bool = False
    cfr_collector: Optional["CFRCollector"] = None
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def decide(self, context: DecisionContext) -> ActionDecision:
        params = self._adjusted_params(context)
        available = set(context.available_actions)
        big_blind = max(1, context.big_blind)
        effective_stack = self._effective_stack(context)
        pot = max(1, context.pot)
        in_position = self._in_position(context)

        opp_range = self._build_opp_range(context, params)
        samples = 150 if self.training_mode else 300
        equity = equity_vs_range(
            my_hole=context.hole_cards,
            board=context.community_cards,
            opp_range=opp_range,
            samples=samples,
            rng=self.rng,
        )

        spr = effective_stack / pot

        decision = self._core_decide(
            context, equity, opp_range, params, in_position, available, big_blind, spr
        )

        if self.cfr_collector is not None and self.training_mode:
            self._record_cfr(context, equity, in_position, decision, opp_range, params)

        return decision

    def _core_decide(self, context, equity, opp_range, params, in_position, available, big_blind, spr):
        if context.street == "preflop":
            decision = self._decide_preflop(
                context, equity, opp_range, params, in_position, available
            )
            if decision is not None:
                return decision

        if (
            "all-in" in available
            and self._effective_stack(context) <= big_blind * 8
            and equity >= params.short_stack_jam_threshold
        ):
            return ActionDecision("all-in")

        if (
            spr <= params.spr_commit_threshold
            and equity >= params.value_3x_threshold
            and "all-in" in available
            and self._effective_stack(context) <= big_blind * 25
        ):
            return ActionDecision("all-in")

        if context.call_amount == 0:
            return self._choose_open_action(
                context, equity, opp_range, params, in_position, available
            )
        return self._choose_facing_bet_action(
            context, equity, opp_range, params, in_position, available
        )

    def _record_cfr(self, context, equity, in_position, decision, opp_range, params):
        from cfr_lite import make_bucket_key, estimate_action_evs
        pot_odds = context.call_amount / max(1, context.pot + context.call_amount)
        bucket = make_bucket_key(context.street, equity, pot_odds, in_position)
        ff = 0.0
        if context.community_cards:
            ff = fold_freq_estimate(
                opp_range, context.community_cards,
                my_bet_chips=max(1, context.pot // 2),
                pot_chips=max(1, context.pot),
                opp_calling_threshold=params.opp_calling_threshold,
            )
        action_evs = estimate_action_evs(
            equity=equity,
            pot=context.pot,
            call_amount=context.call_amount,
            action_targets=dict(context.action_targets),
            available_actions=list(context.available_actions),
            fold_freq=ff,
            big_blind=context.big_blind,
        )
        self.cfr_collector.record(bucket, decision.action, action_evs)

    def _build_opp_range(self, context: DecisionContext, params: AIParams) -> HandRange:
        known = list(context.hole_cards) + list(context.community_cards)
        range_obj = HandRange(known_cards=known)
        opp_idx = self._opp_player_idx(context)
        if opp_idx is None:
            return range_obj

        for street, idx, action, _amount in context.actions_this_hand:
            if idx != opp_idx:
                continue
            if street == "preflop":
                opp_is_button = idx == context.button_index
                update_preflop(range_obj, action, position_is_button=opp_is_button)
            else:
                board_at_action = community_at_street(context.community_cards, street)
                update_postflop(range_obj, action, board_at_action, street)

        if self.opponent_model is not None and context.opponents:
            stats = self.opponent_model.stats_for(context.opponents[0].name)
            if stats.hands_observed >= 20:
                if stats.vpip() > 0.7:
                    range_obj.relax(factor=1.4)
                elif stats.vpip() < 0.2:
                    range_obj.tighten(factor=0.75)
        return range_obj

    def _opp_player_idx(self, context: DecisionContext) -> Optional[int]:
        n_players = len(context.opponents) + 1
        for idx in range(n_players):
            if idx != context.player_index:
                return idx
        return None

    def _adjusted_params(self, context: DecisionContext) -> AIParams:
        if self.opponent_model is None or not context.opponents:
            return self.params
        stats = self.opponent_model.stats_for(context.opponents[0].name)
        if stats.hands_observed < 20:
            return self.params
        p = replace(self.params)
        if stats.fold_to_cbet() > 0.6:
            p = replace(p, bluff_execute_freq=min(1.0, p.bluff_execute_freq * 1.3))
        if stats.vpip() > 0.7:
            p = replace(
                p,
                bluff_execute_freq=p.bluff_execute_freq * 0.4,
                value_2x_threshold=max(0.40, p.value_2x_threshold - 0.05),
                value_3x_threshold=max(0.60, p.value_3x_threshold - 0.05),
                opp_calling_threshold=min(0.25, p.opp_calling_threshold + 0.05),
            )
        if stats.af() > 2.5:
            p = replace(p, slowplay_freq=min(0.30, p.slowplay_freq * 1.5))
        if stats.vpip() < 0.2:
            p = replace(
                p,
                bluff_execute_freq=p.bluff_execute_freq * 1.2,
                opp_calling_threshold=max(0.05, p.opp_calling_threshold - 0.03),
            )
        return p.clipped()

    def _effective_stack(self, context: DecisionContext) -> int:
        if not context.opponents:
            return context.player.stack
        opp_stack = max(o.stack + o.current_bet for o in context.opponents)
        return min(context.player.stack + context.player.current_bet, opp_stack)

    def _in_position(self, context: DecisionContext) -> bool:
        return context.player_index == context.button_index

    def _decide_preflop(
        self,
        context: DecisionContext,
        equity: float,
        opp_range: HandRange,
        params: AIParams,
        in_position: bool,
        available: set,
    ) -> Optional[ActionDecision]:
        if context.call_amount == 0:
            return None
        threshold = params.sb_open_threshold if in_position else params.bb_defend_threshold
        if in_position:
            threshold -= 0.03
        if equity < threshold:
            if (
                "raise-2x" in available
                and self.rng.random()
                < params.bluff_base_freq * (params.in_position_aggression if in_position else 1.0)
            ):
                return ActionDecision("raise-2x")
            return ActionDecision("fold")
        return None

    def _choose_open_action(
        self,
        context: DecisionContext,
        equity: float,
        opp_range: HandRange,
        params: AIParams,
        in_position: bool,
        available: set,
    ) -> ActionDecision:
        if equity >= params.value_3x_threshold:
            if self.rng.random() < params.slowplay_freq:
                return ActionDecision("check")
            return self._pick_raise(params, available, prefer_large=True)

        if equity >= params.value_2x_threshold:
            return self._pick_raise(params, available, prefer_large=False)

        bluff_action = self._should_bluff_raise(
            context, equity, opp_range, params, available
        )
        if bluff_action is not None:
            return ActionDecision(bluff_action)

        return ActionDecision("check") if "check" in available else ActionDecision("fold")

    def _choose_facing_bet_action(
        self,
        context: DecisionContext,
        equity: float,
        opp_range: HandRange,
        params: AIParams,
        in_position: bool,
        available: set,
    ) -> ActionDecision:
        big_blind = max(1, context.big_blind)
        call_amount = context.call_amount
        pot_after_call = context.pot + call_amount
        pot_odds = call_amount / max(1, pot_after_call)
        ev_call_bb = (equity * pot_after_call - (1 - equity) * call_amount) / big_blind

        if equity >= params.value_3x_threshold:
            if self.rng.random() < params.slowplay_freq:
                return ActionDecision("call")
            return self._pick_raise(params, available, prefer_large=True)

        if equity >= params.value_2x_threshold and equity > pot_odds + 0.05:
            return self._pick_raise(params, available, prefer_large=False)

        if equity > pot_odds:
            return ActionDecision("call")

        if ev_call_bb >= params.call_ev_threshold and call_amount <= big_blind * 3:
            return ActionDecision("call")

        if (
            context.street != "river"
            and in_position
            and call_amount <= big_blind * 5
        ):
            bluff_action = self._should_bluff_raise(
                context, equity, opp_range, params, available
            )
            if bluff_action is not None:
                return ActionDecision(bluff_action)

        return ActionDecision("fold")

    def _should_bluff_raise(
        self,
        context: DecisionContext,
        equity: float,
        opp_range: HandRange,
        params: AIParams,
        available: set,
    ) -> Optional[str]:
        """Decide whether to bluff-raise based on fold equity EV math.

        Returns the action label ('raise-2x' / 'raise-3x') to execute, or None.
        Only fires when equity is weak (we're not value-betting) AND we have a
        plausible fold-equity story.
        """
        if equity > 0.50:
            return None
        if not context.community_cards:
            return None  # preflop bluffs handled in _decide_preflop

        candidates = [s for s in ("raise-2x", "raise-3x") if s in available]
        if not candidates:
            return None

        big_blind = max(1, context.big_blind)
        best_action: Optional[str] = None
        best_ev = float("-inf")

        for sizing in candidates:
            target = context.action_targets.get(sizing)
            if target is None:
                continue
            bet_chips = target - context.player.current_bet
            if bet_chips <= 0:
                continue
            pot_chips = context.pot + context.call_amount
            ff = fold_freq_estimate(
                opp_range=opp_range,
                board=context.community_cards,
                my_bet_chips=bet_chips,
                pot_chips=pot_chips,
                opp_calling_threshold=params.opp_calling_threshold,
            )
            ev_chips = ff * pot_chips - (1 - ff) * bet_chips
            ev_bb = ev_chips / big_blind
            if ev_bb > best_ev:
                best_ev = ev_bb
                best_action = sizing

        if best_action is None or best_ev <= params.min_bluff_ev_bb:
            return None
        if self.rng.random() >= params.bluff_execute_freq:
            return None
        return best_action

    def _pick_raise(self, params: AIParams, available: set, prefer_large: bool) -> ActionDecision:
        has_3x = "raise-3x" in available
        has_2x = "raise-2x" in available
        if not has_2x and not has_3x:
            if "call" in available:
                return ActionDecision("call")
            if "check" in available:
                return ActionDecision("check")
            return ActionDecision("fold")
        if prefer_large:
            if has_3x:
                if self.rng.random() < params.sizing_mix_freq and has_2x:
                    return ActionDecision("raise-2x")
                return ActionDecision("raise-3x")
            return ActionDecision("raise-2x")
        if has_2x:
            if self.rng.random() < params.sizing_mix_freq * 0.5 and has_3x:
                return ActionDecision("raise-3x")
            return ActionDecision("raise-2x")
        return ActionDecision("raise-3x")
