import atexit
import os
from typing import List, Optional

from advanced_ai import AdvancedPokerAI
from ai import BasicPokerAI, PokerAI
from ai_params import AIParams
from engine import ActionDecision, DecisionContext, Game, Player, describe_hand_value, evaluate_best_hand
from opponent_model import OpponentModel, build_hand_log_from_game
from online_learner import OnlineLearner
from persistence import AI_PARAMS_PATH, HAND_HISTORY_PATH, OPPONENT_PROFILES_PATH, append_hand_log

ACTION_ALIASES = {
    "fold": {"fold", "f"},
    "check": {"check", "c"},
    "call": {"call", "ca"},
    "raise-2x": {"raise-2x", "raise2", "r2", "2x"},
    "raise-3x": {"raise-3x", "raise3", "r3", "3x"},
    "all-in": {"all-in", "allin", "a"},
}


def format_cards(cards: List[str]) -> str:
    return " ".join(str(card) for card in cards)


def show_table(game: Game, reveal_all: bool = False, current_player: Optional[Player] = None) -> None:
    print("\n=== 当前桌面 ===")
    print(f"公共牌: {' '.join(str(card) for card in game.community_cards) if game.community_cards else '(暂无)'}")
    print(f"底池: {game.current_pot()} 盲注: {game.small_blind}/{game.big_blind}")
    for player in game.players:
        status = []
        if player.folded:
            status.append("已弃牌")
        if player.all_in:
            status.append("全下")
        if player.current_bet > 0:
            status.append(f"当前注 {player.current_bet}")
        if player is current_player:
            status.append("当前行动")
        if player.stack == 0 and not player.folded:
            status.append("已破产")
        status_text = ", ".join(status) if status else "等待"
        hole = "" if player.folded or not reveal_all else " " + " ".join(str(card) for card in player.hole_cards)
        if not reveal_all and current_player is not None and player is current_player:
            hole = " " + " ".join(str(card) for card in player.hole_cards)
        print(f"{player.name}: 筹码 {player.stack}, {status_text}{hole}")
    print("===============\n")


def parse_mode(raw: str) -> Optional[str]:
    raw = raw.strip().lower()
    if raw in {"1", "pvp"}:
        return "pvp"
    if raw in {"2", "pve", ""}:
        return "pve"
    return None


def describe_action(action: str, context: DecisionContext) -> str:
    if action == "fold":
        return "fold / 弃牌"
    if action == "check":
        return "check / 过牌"
    if action == "call":
        target = context.action_targets.get("call", context.player.current_bet)
        return f"call / 跟注 {context.call_amount}，总注到 {target}"
    if action == "raise-2x":
        target = context.action_targets.get("raise-2x", context.highest_bet)
        if context.street == "preflop" and context.highest_bet <= context.big_blind:
            return f"raise-2x / 2.5BB 开池到 {target}"
        if context.highest_bet == 0:
            return f"raise-2x / 约 66% pot 下注到 {target}"
        if context.street == "preflop":
            return f"raise-2x / 约 2.2x 再加注到 {target}"
        return f"raise-2x / 标准加注到 {target}"
    if action == "raise-3x":
        target = context.action_targets.get("raise-3x", context.highest_bet)
        if context.street == "preflop" and context.highest_bet <= context.big_blind:
            return f"raise-3x / 3.5BB 开池到 {target}"
        if context.highest_bet == 0:
            return f"raise-3x / 100% pot 下注到 {target}"
        if context.street == "preflop":
            return f"raise-3x / 约 3x 再加注到 {target}"
        return f"raise-3x / 重加注到 {target}"
    if action == "all-in":
        target = context.action_targets.get("all-in", context.player.current_bet + context.player.stack)
        return f"all-in / 全下到 {target}"
    return action


def parse_action_choice(raw: str, options: List[str]) -> Optional[str]:
    raw = raw.strip().lower()
    if raw.isdigit():
        option_index = int(raw) - 1
        if 0 <= option_index < len(options):
            return options[option_index]

    for action in options:
        if raw == action or raw in ACTION_ALIASES.get(action, set()):
            return action
    return None


def prompt_player_action(player: Player, context: DecisionContext) -> ActionDecision:
    print(f"{player.name} 的手牌: {' '.join(str(card) for card in player.hole_cards)}")
    print(f"当前轮次: {context.street}  当前最高注: {context.highest_bet}, 你已下: {player.current_bet}, 需跟注: {context.call_amount}")
    print("可用动作:")
    for index, action in enumerate(context.available_actions, start=1):
        print(f"{index}. {describe_action(action, context)}")
    while True:
        raw = input(f"{player.name} 请选择动作: ").strip()
        action = parse_action_choice(raw, list(context.available_actions))
        if action is None:
            print("无效动作，请输入编号或动作别名。")
            continue
        return ActionDecision(action)


def summarize_hand(game: Game, results: dict) -> None:
    print("\n=== 摊牌结果 ===")
    print(f"公共牌: {' '.join(str(card) for card in game.community_cards)}")
    winners = [player for player in results["results"].keys()]
    for player in game.players:
        if not player.folded:
            hand_cards = player.hole_cards + game.community_cards
            if len(hand_cards) >= 5:
                hand_value = evaluate_best_hand(hand_cards)
                detail = f"牌力: {describe_hand_value(hand_value)}"
            else:
                detail = "牌力: (未摊牌)"
            print(f"{player.name} {detail}  手牌: {' '.join(str(card) for card in player.hole_cards)}")
    for player, amount in results["results"].items():
        print(f"{player.name} 赢得 {amount} 筹码")
    print("===============\n")


def run_game(online_learn: bool = False) -> None:
    print("欢迎来到德州扑克 CLI 小程序！")
    if online_learn:
        print("  [在线学习已启用：每 30 手自动微调 AI 参数]")
    while True:
        mode = parse_mode(input("请选择模式：1. PVP  2. PVE（默认 2）: "))
        if mode is not None:
            break
        print("请输入 1/PVP 或 2/PVE。")

    ai_players: dict[int, PokerAI] = {}
    opp_model: Optional[OpponentModel] = None
    learner: Optional[OnlineLearner] = None
    player_ids: List[str] = []
    if mode == "pve":
        human_name = input("请输入你的名称（默认 玩家1）: ").strip() or "玩家1"
        ai_name = input("请输入 AI 名称（默认 AI）: ").strip() or "AI"
        ai_level_raw = input("请选择 AI 等级：1. 高级（推荐）  2. 基础（默认 1）: ").strip()
        names = [human_name, ai_name]
        player_ids = ["human_0", "ai_basic" if ai_level_raw in {"2", "basic", "basic-ai"} else "ai_advanced"]
        if ai_level_raw in {"2", "basic", "basic-ai"}:
            ai_players[1] = BasicPokerAI(name=ai_name)
        else:
            params = AIParams.load_or_default(AI_PARAMS_PATH)
            opp_model = OpponentModel(profile_path=OPPONENT_PROFILES_PATH)
            ai_players[1] = AdvancedPokerAI(
                name=ai_name,
                params=params,
                opponent_model=opp_model,
                opponent_identity=player_ids[0],
            )
            atexit.register(opp_model.save)
            if online_learn:
                learner = OnlineLearner(params=params)
    else:
        names = []
        for index in range(2):
            name = input(f"请输入玩家 {index + 1} 名称（默认 玩家{index + 1}）: ").strip() or f"玩家{index + 1}"
            names.append(name)
        player_ids = ["human_0", "human_1"]

    starting_stack = 1000
    game = Game(names, starting_stack=starting_stack, small_blind=10, big_blind=20)

    def action_provider(context: DecisionContext) -> ActionDecision:
        player = game.players[context.player_index]
        ai_player = ai_players.get(context.player_index)
        if ai_player is not None:
            decision = ai_player.decide(context)
            print(f"{player.name} 选择: {describe_action(decision.action, context)}")
            if learner is not None and isinstance(ai_player, AdvancedPokerAI):
                from range import fold_freq_estimate, equity_vs_range
                opp_range = ai_player._build_opp_range(context, ai_player.params)
                eq = equity_vs_range(context.hole_cards, context.community_cards, opp_range, samples=100, rng=ai_player.rng)
                ff = fold_freq_estimate(opp_range, context.community_cards, max(1, context.pot // 2), max(1, context.pot), ai_player.params.opp_calling_threshold) if context.community_cards else 0.0
                learner.record_decision(
                    street=context.street, equity=eq, pot=context.pot,
                    call_amount=context.call_amount,
                    action_targets=dict(context.action_targets),
                    available_actions=list(context.available_actions),
                    chosen_action=decision.action,
                    in_position=(context.player_index == context.button_index),
                    fold_freq=ff, big_blind=context.big_blind,
                )
            return decision

        show_table(game, reveal_all=False, current_player=player)
        return prompt_player_action(player, context)

    while True:
        button_before = game.button_index
        stacks_before = [p.stack for p in game.players]
        result = game.play_hand(action_provider)
        summarize_hand(game, result)

        payoffs = {p.name: p.stack - stacks_before[i] for i, p in enumerate(game.players)}
        hand_log = build_hand_log_from_game(
            game,
            starting_stacks=stacks_before,
            button_index_at_start=button_before,
            payoffs=payoffs,
            final_stage=result["stage"],
            player_ids=player_ids,
        )
        if opp_model is not None:
            opp_model.observe(hand_log, self_name=player_ids[1])
        try:
            append_hand_log(HAND_HISTORY_PATH, hand_log)
        except OSError:
            pass

        print("当前筹码:")
        for player in game.players:
            print(f"{player.name}: {player.stack}")

        alive_players = [player for player in game.players if player.stack > 0]
        if len(alive_players) <= 1:
            winner = max(game.players, key=lambda p: p.stack)
            print(f"游戏结束，{winner.name} 获胜！")
            choice = input("对方破产，是否重新开局？(Y/n): ").strip().lower()
            if choice.startswith("n"):
                print("游戏结束，谢谢游玩！")
                break
            game = Game(names, starting_stack=starting_stack, small_blind=10, big_blind=20)
            continue

        if input("继续下一手？(Y/n): ").strip().lower().startswith("n"):
            print("游戏结束，谢谢游玩！")
            break


if __name__ == "__main__":
    import sys
    online_learn = "--online-learn" in sys.argv
    run_game(online_learn=online_learn)
