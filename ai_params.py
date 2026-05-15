"""Tunable parameters for AdvancedPokerAI.

The (1+lambda)-ES trainer perturbs these floats; the AI loads them at startup.
All fields are scalars in [0, 1] or small positive floats so perturbation+clip
is straightforward.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Tuple

SCHEMA_VERSION = 1


@dataclass
class AIParams:
    # Equity threshold above which we strongly prefer a 3x raise as value
    value_3x_threshold: float = 0.72
    # Equity threshold above which a 2x raise is value (assuming below 3x cutoff)
    value_2x_threshold: float = 0.55
    # Base probability of bluffing preflop (no fold-equity model preflop)
    bluff_base_freq: float = 0.06
    # Probability of slowplaying when we have monster equity (trap)
    slowplay_freq: float = 0.12
    # Minimum EV (in big blinds) we require to make a marginal call
    call_ev_threshold: float = -0.15
    # Opponent calling threshold (postflop_strength below which opp folds, before sizing scaling)
    opp_calling_threshold: float = 0.13
    # Minimum EV (in BBs) required to execute a fold-equity-driven bluff
    min_bluff_ev_bb: float = 0.10
    # Even if bluff is +EV, only execute this fraction of the time (balance)
    bluff_execute_freq: float = 0.70
    # Multiplier on raise frequency when in position
    in_position_aggression: float = 1.20
    # Min equity to open-raise as small blind (preflop). Below this we fold rather than limp.
    # HU SB is getting 3:1 pot odds, so this should be near pot_odds (0.25).
    sb_open_threshold: float = 0.30
    # Min equity for big blind to defend vs an open. BB is also getting good odds.
    bb_defend_threshold: float = 0.28
    # Equity above which short stack (low SPR) should jam all-in
    short_stack_jam_threshold: float = 0.62
    # SPR below which we consider ourselves "committed" to the pot
    spr_commit_threshold: float = 3.0
    # Probability of mixing in a 3x raise when we'd normally 2x (for sizing balance)
    sizing_mix_freq: float = 0.20
    # Base scale for randomness across mixed strategies (0=deterministic, 1=high noise)
    mixed_strategy_noise: float = 1.0

    # Hard clip ranges per field — keeps trainer perturbations sane.
    @staticmethod
    def clip_ranges() -> Dict[str, Tuple[float, float]]:
        return {
            "value_3x_threshold": (0.55, 0.90),
            "value_2x_threshold": (0.35, 0.70),
            "bluff_base_freq": (0.00, 0.20),
            "slowplay_freq": (0.00, 0.30),
            "call_ev_threshold": (-1.5, 0.5),
            "in_position_aggression": (0.80, 1.60),
            "sb_open_threshold": (0.22, 0.50),
            "bb_defend_threshold": (0.20, 0.45),
            "short_stack_jam_threshold": (0.50, 0.80),
            "spr_commit_threshold": (1.5, 6.0),
            "sizing_mix_freq": (0.00, 0.50),
            "mixed_strategy_noise": (0.0, 2.0),
            "opp_calling_threshold": (0.05, 0.30),
            "min_bluff_ev_bb": (-0.5, 1.0),
            "bluff_execute_freq": (0.30, 1.0),
        }

    def clipped(self) -> "AIParams":
        ranges = self.clip_ranges()
        kwargs = {}
        for f in fields(self):
            lo, hi = ranges[f.name]
            kwargs[f.name] = max(lo, min(hi, getattr(self, f.name)))
        return replace(self, **kwargs)

    def to_vector(self) -> List[float]:
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def from_vector(cls, vec) -> "AIParams":
        kwargs = {f.name: float(v) for f, v in zip(fields(cls), vec)}
        return cls(**kwargs).clipped()

    @staticmethod
    def vector_bounds() -> Tuple[List[float], List[float]]:
        ranges = AIParams.clip_ranges()
        lo = [ranges[f.name][0] for f in fields(AIParams)]
        hi = [ranges[f.name][1] for f in fields(AIParams)]
        return (lo, hi)

    def to_dict(self) -> dict:
        return {"version": SCHEMA_VERSION, "params": asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "AIParams":
        version = data.get("version", 0)
        if version != SCHEMA_VERSION:
            # Future: handle migration. For now, fall back to defaults if mismatch.
            return cls()
        params_dict = data.get("params", {})
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in params_dict.items() if k in known}
        return cls(**filtered)

    @classmethod
    def load_or_default(cls, path: str) -> "AIParams":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as src:
                content = src.read()
            with open(path + ".bak", "w") as dst:
                dst.write(content)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
