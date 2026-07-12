"""Persistence helpers: HandLog schema + jsonl writer + opponent profile loader."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HAND_HISTORY_PATH = os.path.join(DATA_DIR, "hand_history.jsonl")
SESSION_HISTORY_DIR = os.path.join(DATA_DIR, "sessions")
OPPONENT_PROFILES_PATH = os.path.join(DATA_DIR, "opponent_profiles.json")
AI_PARAMS_PATH = os.path.join(DATA_DIR, "ai_params.json")


@dataclass
class HandLog:
    """One complete hand's observable record. Used for both training history and opponent modeling."""

    timestamp: float
    player_names: List[str]
    starting_stacks: List[int]
    button_index: int
    small_blind: int
    big_blind: int
    actions: List[Tuple[str, int, str, int]]
    community: List[str]
    player_ids: List[str] = field(default_factory=list)
    hole_cards: Dict[str, List[str]] = field(default_factory=dict)
    payoffs: Dict[str, int] = field(default_factory=dict)
    final_stage: str = "river"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HandLog":
        if "player_ids" not in data:
            data = dict(data)
            data["player_ids"] = list(data.get("player_names", []))
        return cls(**data)


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)


def session_history_path(game_id: str) -> str:
    ensure_data_dir()
    safe = "".join(ch for ch in game_id if ch.isalnum() or ch in ("-", "_"))[:64]
    return os.path.join(SESSION_HISTORY_DIR, f"{safe}.jsonl")


def append_hand_log(path: str, log: HandLog) -> None:
    ensure_data_dir()
    with open(path, "a") as f:
        f.write(json.dumps(log.to_dict()) + "\n")


def read_hand_logs(path: str, max_lines: Optional[int] = None) -> List[HandLog]:
    if not os.path.exists(path):
        return []
    logs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            logs.append(HandLog.from_dict(json.loads(line)))
    return logs


def save_profiles(path: str, profiles: dict) -> None:
    ensure_data_dir()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profiles, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_profiles(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}
