"""Flask web server wrapping engine.Game with a threaded session per game.

Architecture:
  Each GameSession owns a daemon thread that runs Game.play_hand. The thread
  blocks on `action_q` whenever it needs human input; the HTTP layer pushes
  ActionDecision via submit_human() and reads the next state from `state_q`.

  This lets a sync engine (engine.Game) speak HTTP request/response without
  refactoring the engine.

Run:
  python server.py
  → http://localhost:5000
"""

from __future__ import annotations

import os
import threading
import uuid
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory

from advanced_ai import AdvancedPokerAI
from ai import BasicPokerAI
from ai_params import AIParams
from engine import ActionDecision, Card, DecisionContext, Game, evaluate_best_hand, RANK_LABELS
from opponent_model import OpponentModel, build_hand_log_from_game
from persistence import (
    AI_PARAMS_PATH,
    HAND_HISTORY_PATH,
    OPPONENT_PROFILES_PATH,
    append_hand_log,
)

SUIT_TO_LETTER = {"♠": "S", "♥": "H", "♦": "D", "♣": "C"}
LETTER_TO_SUIT = {v: k for k, v in SUIT_TO_LETTER.items()}

DEFAULT_STARTING_STACK = 1000
DEFAULT_SMALL_BLIND = 10
DEFAULT_BIG_BLIND = 20

_SESSIONS: Dict[str, "GameSession"] = {}
_SESSIONS_LOCK = threading.Lock()

HAND_CATEGORY_KEYS = {
    0: "hand_cat_high_card",
    1: "hand_cat_pair",
    2: "hand_cat_two_pair",
    3: "hand_cat_three_kind",
    4: "hand_cat_straight",
    5: "hand_cat_flush",
    6: "hand_cat_full_house",
    7: "hand_cat_four_kind",
    8: "hand_cat_straight_flush",
}


def _format_hand_descriptor(hand_value):
    """Serialize an evaluate_best_hand result into i18n-friendly parts.
    Returns {"key": <i18n key>, "high": <high rank label or empty>}."""
    category, details = hand_value
    high_rank_label = RANK_LABELS.get(details[0], "") if details else ""
    if category == 4 or category == 8:
        # straight or straight flush — show just the high
        return {"key": HAND_CATEGORY_KEYS[category], "high": high_rank_label}
    if category in (1, 2, 3, 6, 7):
        return {"key": HAND_CATEGORY_KEYS[category], "high": high_rank_label}
    return {"key": HAND_CATEGORY_KEYS[category], "high": ""}


def _card_to_dict(card: Card) -> Dict[str, str]:
    return {"rank": card.rank, "suit": SUIT_TO_LETTER.get(card.suit, card.suit)}


def _cards_to_list(cards) -> List[Dict[str, str]]:
    return [_card_to_dict(c) for c in cards]


class GameSession:
    """Owns a Game + AI players + a daemon thread driving the play loop.

    Communicates with the HTTP layer through two Queues:
      action_q: HTTP layer pushes signals ('start_hand', 'stop') and
                ActionDecision instances when a human acts.
      state_q:  thread pushes serialized state snapshots whenever the
                outside world needs to react (human's turn, hand done).
    """

    def __init__(self, game_id: str, mode: str, names: List[str], ai_level: str = "advanced"):
        if len(names) != 2:
            raise ValueError("only heads-up (2 players) supported")
        if mode not in ("pve", "pvp"):
            raise ValueError(f"unknown mode: {mode}")

        self.game_id = game_id
        self.mode = mode
        self.names = list(names)
        self.starting_stack = DEFAULT_STARTING_STACK
        self.small_blind = DEFAULT_SMALL_BLIND
        self.big_blind = DEFAULT_BIG_BLIND
        self.game = Game(
            self.names,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

        self.human_indices = {0, 1} if mode == "pvp" else {0}
        self.ai_players: Dict[int, Any] = {}
        self.opp_model: Optional[OpponentModel] = None

        if mode == "pve":
            params = AIParams.load_or_default(AI_PARAMS_PATH)
            self.opp_model = OpponentModel(profile_path=OPPONENT_PROFILES_PATH)
            if ai_level == "basic":
                self.ai_players[1] = BasicPokerAI(name=names[1])
            else:
                self.ai_players[1] = AdvancedPokerAI(
                    name=names[1], params=params, opponent_model=self.opp_model
                )

        self.action_q: Queue = Queue()
        self.state_q: Queue = Queue()
        self.recent_actions: List[Dict[str, Any]] = []
        self.last_winner_info: Optional[List[Dict[str, Any]]] = None
        self.last_showdown_holes: Optional[Dict[str, List[Dict[str, str]]]] = None
        self.last_result_stage: Optional[str] = None
        self.last_hand_descriptions: Optional[Dict[str, str]] = None
        self.stopped = False
        self.broken = False
        self._cached_state: Optional[Dict[str, Any]] = None
        self._stacks_before_hand: List[int] = [self.starting_stack, self.starting_stack]
        self._button_before_hand: int = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public HTTP-facing API
    # ------------------------------------------------------------------

    def start_next_hand(self) -> None:
        self.action_q.put("start_hand")

    def submit_human(self, action: str, amount: Optional[int] = None) -> None:
        self.action_q.put(ActionDecision(action, amount))

    def wait_for_state(self, timeout: float = 20.0) -> Dict[str, Any]:
        try:
            state = self.state_q.get(timeout=timeout)
        except Empty:
            self.broken = True
            raise TimeoutError("game thread did not produce a state in time")
        self._cached_state = state
        return state

    def get_cached_state(self) -> Optional[Dict[str, Any]]:
        return self._cached_state

    def stop(self) -> None:
        self.stopped = True
        self.action_q.put("stop")

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self.stopped:
            sig = self.action_q.get()
            if sig == "stop":
                break
            if sig != "start_hand":
                continue
            self.recent_actions = []
            self.last_winner_info = None
            self.last_showdown_holes = None
            self.last_hand_descriptions = None
            self._stacks_before_hand = [p.stack for p in self.game.players]
            self._button_before_hand = self.game.button_index
            try:
                result = self.game.play_hand(self._provider)
            except Exception as exc:  # noqa: BLE001
                self.broken = True
                self.state_q.put(
                    {
                        "phase": "error",
                        "message": f"game thread crashed: {exc}",
                    }
                )
                break
            self._record_hand_result(result)
            self._observe_for_model(result)
            self.state_q.put(self._snapshot(hand_done=True))

    def _provider(self, ctx: DecisionContext) -> ActionDecision:
        if ctx.player_index in self.human_indices:
            self.state_q.put(self._snapshot(needs_human=True, ctx=ctx))
            self.recent_actions = []
            decision = self.action_q.get()
            if isinstance(decision, str):
                raise RuntimeError(f"expected ActionDecision, got control signal {decision!r}")
            self.recent_actions.append(
                {
                    "player_idx": ctx.player_index,
                    "name": self.game.players[ctx.player_index].name,
                    "action": decision.action,
                    "street": ctx.street,
                    "is_ai": False,
                }
            )
            return decision
        ai = self.ai_players[ctx.player_index]
        decision = ai.decide(ctx)
        self.recent_actions.append(
            {
                "player_idx": ctx.player_index,
                "name": self.game.players[ctx.player_index].name,
                "action": decision.action,
                "street": ctx.street,
                "is_ai": True,
            }
        )
        return decision

    # ------------------------------------------------------------------
    # Hand observation / logging
    # ------------------------------------------------------------------

    def _record_hand_result(self, result: Dict[str, Any]) -> None:
        results = result.get("results", {})
        winner_info = []
        for player, amount in results.items():
            if amount > 0:
                winner_info.append({"name": player.name, "amount": amount})
        self.last_winner_info = winner_info
        self.last_result_stage = result.get("stage")

        showdown_holes: Dict[str, List[Dict[str, str]]] = {}
        descriptions: Dict[str, Dict[str, str]] = {}
        community = list(self.game.community_cards)
        for player in self.game.players:
            if not player.folded and player.hole_cards:
                showdown_holes[player.name] = _cards_to_list(player.hole_cards)
                if len(community) >= 5:
                    hand_value = evaluate_best_hand(list(player.hole_cards) + community)
                    descriptions[player.name] = _format_hand_descriptor(hand_value)
        self.last_showdown_holes = showdown_holes or None
        self.last_hand_descriptions = descriptions or None

    def _observe_for_model(self, result: Dict[str, Any]) -> None:
        if self.opp_model is None:
            return
        payoffs = {
            p.name: p.stack - self._stacks_before_hand[i]
            for i, p in enumerate(self.game.players)
        }
        log = build_hand_log_from_game(
            game=self.game,
            starting_stacks=self._stacks_before_hand,
            button_index_at_start=self._button_before_hand,
            payoffs=payoffs,
            final_stage=result.get("stage", "river"),
        )
        ai_idx = next(iter(self.ai_players))
        self.opp_model.observe(log, self_name=self.game.players[ai_idx].name)
        try:
            append_hand_log(HAND_HISTORY_PATH, log)
        except OSError:
            pass

    def save_persistent(self) -> None:
        if self.opp_model is not None:
            try:
                self.opp_model.save()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _snapshot(
        self,
        needs_human: bool = False,
        ctx: Optional[DecisionContext] = None,
        hand_done: bool = False,
    ) -> Dict[str, Any]:
        players_payload: List[Dict[str, Any]] = []
        current_player_idx = ctx.player_index if ctx is not None else -1

        for idx, player in enumerate(self.game.players):
            expose_hole = self._should_expose_hole_cards(
                idx=idx,
                needs_human=needs_human,
                current_player_idx=current_player_idx,
                hand_done=hand_done,
                player_folded=player.folded,
            )
            hole_cards = _cards_to_list(player.hole_cards) if (expose_hole and player.hole_cards) else None
            players_payload.append(
                {
                    "idx": idx,
                    "name": player.name,
                    "stack": player.stack,
                    "current_bet": player.current_bet,
                    "folded": player.folded,
                    "all_in": player.all_in,
                    "hole_cards": hole_cards,
                    "is_button": idx == self.game.button_index,
                    "is_human": idx in self.human_indices,
                }
            )

        snapshot: Dict[str, Any] = {
            "phase": "hand_done" if hand_done else ("human_turn" if needs_human else "ai_turn"),
            "street": ctx.street if ctx else (self.last_result_stage or "preflop"),
            "community": _cards_to_list(self.game.community_cards),
            "pot": self.game.current_pot(),
            "small_blind": self.game.small_blind,
            "big_blind": self.game.big_blind,
            "players": players_payload,
            "current_player_idx": current_player_idx if needs_human else -1,
            "available_actions": list(ctx.available_actions) if ctx else [],
            "action_targets": dict(ctx.action_targets) if ctx else {},
            "recent_actions": list(self.recent_actions),
            "winner_info": self.last_winner_info if hand_done else None,
            "showdown_holes": self.last_showdown_holes if hand_done else None,
            "hand_descriptions": self.last_hand_descriptions if hand_done else None,
            "mode": self.mode,
            "human_indices": sorted(self.human_indices),
            "game_over": self._is_game_over(),
        }
        return snapshot

    def _should_expose_hole_cards(
        self,
        idx: int,
        needs_human: bool,
        current_player_idx: int,
        hand_done: bool,
        player_folded: bool,
    ) -> bool:
        if hand_done:
            return not player_folded
        if needs_human:
            if self.mode == "pvp":
                return idx == current_player_idx
            return idx in self.human_indices  # PVE: always expose human's cards
        return False

    def _is_game_over(self) -> bool:
        alive = [p for p in self.game.players if p.stack >= self.big_blind]
        return len(alive) <= 1

    def reset_stacks(self) -> None:
        for player in self.game.players:
            player.stack = self.starting_stack
            player.folded = False
            player.all_in = False


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")


def _get_session(game_id: str) -> GameSession:
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(game_id)
    if s is None:
        raise KeyError(f"unknown game_id: {game_id}")
    if s.broken:
        raise RuntimeError("session is broken; create a new game")
    return s


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/new_game")
def new_game():
    body = request.get_json(force=True) or {}
    mode = body.get("mode", "pve")
    names = body.get("names")
    if not names or len(names) != 2:
        return jsonify({"error": "names must be a 2-element list"}), 400
    ai_level = body.get("ai_level", "advanced")

    game_id = uuid.uuid4().hex[:12]
    try:
        session = GameSession(game_id, mode=mode, names=names, ai_level=ai_level)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    with _SESSIONS_LOCK:
        _SESSIONS[game_id] = session
    session.start_next_hand()
    try:
        state = session.wait_for_state(timeout=30)
    except TimeoutError:
        return jsonify({"error": "game did not start in time"}), 503
    return jsonify({"game_id": game_id, "state": state})


@app.post("/api/action")
def action():
    body = request.get_json(force=True) or {}
    game_id = body.get("game_id")
    action_label = body.get("action")
    amount = body.get("amount")
    if not game_id or not action_label:
        return jsonify({"error": "game_id and action are required"}), 400
    try:
        session = _get_session(game_id)
    except (KeyError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 404 if isinstance(exc, KeyError) else 410
    session.submit_human(action_label, amount)
    try:
        state = session.wait_for_state(timeout=30)
    except TimeoutError:
        return jsonify({"error": "game thread timed out"}), 503
    return jsonify({"state": state})


@app.post("/api/next_hand")
def next_hand():
    body = request.get_json(force=True) or {}
    game_id = body.get("game_id")
    if not game_id:
        return jsonify({"error": "game_id required"}), 400
    try:
        session = _get_session(game_id)
    except (KeyError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 404 if isinstance(exc, KeyError) else 410

    if session._is_game_over():
        session.reset_stacks()
    session.start_next_hand()
    try:
        state = session.wait_for_state(timeout=30)
    except TimeoutError:
        return jsonify({"error": "game thread timed out"}), 503
    return jsonify({"state": state})


@app.post("/api/end_game")
def end_game():
    body = request.get_json(force=True) or {}
    game_id = body.get("game_id")
    if not game_id:
        return jsonify({"ok": True})
    with _SESSIONS_LOCK:
        session = _SESSIONS.pop(game_id, None)
    if session:
        session.save_persistent()
        session.stop()
    return jsonify({"ok": True})


@app.get("/api/state")
def get_state():
    game_id = request.args.get("game_id")
    if not game_id:
        return jsonify({"error": "game_id required"}), 400
    try:
        session = _get_session(game_id)
    except (KeyError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 404 if isinstance(exc, KeyError) else 410
    cached = session.get_cached_state()
    if cached is None:
        return jsonify({"error": "no state yet"}), 425
    return jsonify({"state": cached})


if __name__ == "__main__":
    print("Texas Hold'em web UI starting at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
