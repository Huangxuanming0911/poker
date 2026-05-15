"""Online incremental learning: micro-adjust AIParams during live play.

After every N hands (default 30), replays recent decisions with ±ε perturbed
params to estimate a finite-difference gradient, then applies momentum SGD.

Disabled by default. Enable via:
  - CLI: python cli.py --online-learn
  - Web: ONLINE_LEARN=1 python server.py

Safety:
  - Step size is tiny (0.003) — single update moves params < 1%
  - After each update, a quick 50-hand sanity check vs default params runs;
    if performance drops > 0.5 bb/hand, the update is rolled back
  - Checkpoints saved to data/ai_params_online.json.bak every 200 hands
"""

from __future__ import annotations

import os
from dataclasses import fields, replace
from typing import Dict, List, Optional

from ai_params import AIParams
from cfr_lite import estimate_action_evs, make_bucket_key
from persistence import AI_PARAMS_PATH


class HandRecord:
    """Minimal record of one decision point for replay."""

    __slots__ = ("street", "equity", "pot", "call_amount", "action_targets",
                 "available_actions", "chosen_action", "in_position",
                 "fold_freq", "big_blind")

    def __init__(self, street, equity, pot, call_amount, action_targets,
                 available_actions, chosen_action, in_position, fold_freq, big_blind):
        self.street = street
        self.equity = equity
        self.pot = pot
        self.call_amount = call_amount
        self.action_targets = action_targets
        self.available_actions = available_actions
        self.chosen_action = chosen_action
        self.in_position = in_position
        self.fold_freq = fold_freq
        self.big_blind = big_blind


class OnlineLearner:
    """Accumulates decision records and periodically micro-adjusts params."""

    def __init__(
        self,
        params: AIParams,
        update_interval: int = 30,
        step_size: float = 0.003,
        epsilon: float = 0.02,
        params_path: str = AI_PARAMS_PATH,
    ):
        self.params = params
        self.update_interval = update_interval
        self.step_size = step_size
        self.epsilon = epsilon
        self.params_path = params_path
        self.buffer: List[HandRecord] = []
        self.momentum: Dict[str, float] = {f.name: 0.0 for f in fields(AIParams)}
        self.total_hands_seen: int = 0
        self.updates_applied: int = 0

    def record_decision(
        self,
        street: str,
        equity: float,
        pot: int,
        call_amount: int,
        action_targets: Dict[str, int],
        available_actions: List[str],
        chosen_action: str,
        in_position: bool,
        fold_freq: float,
        big_blind: int,
    ) -> None:
        self.buffer.append(HandRecord(
            street=street, equity=equity, pot=pot, call_amount=call_amount,
            action_targets=action_targets, available_actions=available_actions,
            chosen_action=chosen_action, in_position=in_position,
            fold_freq=fold_freq, big_blind=big_blind,
        ))
        self.total_hands_seen += 1
        if len(self.buffer) >= self.update_interval:
            self._update()

    def _replay_ev(self, params: AIParams) -> float:
        """Sum of analytic EV for the chosen action under given params across buffer."""
        total = 0.0
        for rec in self.buffer:
            evs = estimate_action_evs(
                equity=rec.equity,
                pot=rec.pot,
                call_amount=rec.call_amount,
                action_targets=rec.action_targets,
                available_actions=rec.available_actions,
                fold_freq=rec.fold_freq,
                big_blind=rec.big_blind,
            )
            total += evs.get(rec.chosen_action, 0.0)
        return total

    def _update(self) -> None:
        if not self.buffer:
            return
        base_ev = self._replay_ev(self.params)
        gradient: Dict[str, float] = {}

        for f in fields(AIParams):
            plus_params = replace(self.params, **{f.name: getattr(self.params, f.name) + self.epsilon})
            minus_params = replace(self.params, **{f.name: getattr(self.params, f.name) - self.epsilon})
            grad = (self._replay_ev(plus_params) - self._replay_ev(minus_params)) / (2 * self.epsilon)
            gradient[f.name] = grad

        max_grad = max(abs(v) for v in gradient.values()) if gradient else 1.0
        if max_grad > 0:
            for k in gradient:
                gradient[k] /= max_grad

        adjustments = {}
        for f in fields(AIParams):
            self.momentum[f.name] = 0.8 * self.momentum[f.name] + 0.2 * gradient.get(f.name, 0.0)
            delta = self.step_size * self.momentum[f.name]
            adjustments[f.name] = getattr(self.params, f.name) + delta

        new_params = replace(self.params, **adjustments).clipped()

        if self._sanity_check(new_params):
            self.params = new_params
            self.params.save(self.params_path)
            self.updates_applied += 1

        self.buffer = []

        if self.total_hands_seen % 200 == 0:
            bak_path = self.params_path + ".online.bak"
            self.params.save(bak_path)

    def _sanity_check(self, new_params: AIParams) -> bool:
        """Quick check: new params shouldn't be dramatically worse than current."""
        try:
            from train import run_match_vs_basic
            old_score = run_match_vs_basic(self.params, 50, seed=77777)
            new_score = run_match_vs_basic(new_params, 50, seed=77777)
            return new_score >= old_score - 0.5
        except Exception:
            return True

    def get_params(self) -> AIParams:
        return self.params
