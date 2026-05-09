from typing import List, Tuple, Optional

from engine import Game, Player, evaluate_best_hand, describe_hand_value


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


def parse_action(raw: str) -> Tuple[str, Optional[int]]:
    raw = raw.strip().lower()
    if raw in {"fold", "f"}:
        return "fold", None
    if raw in {"check", "c"}:
        return "check", None
    if raw in {"call", "ca"}:
        return "call", None
    if raw in {"all-in", "allin", "a"}:
        return "all-in", None
    if raw.startswith("bet "):
        amount_str = raw[4:].strip()
        if amount_str.isdigit():
            return "bet", int(amount_str)
    if raw.startswith("raise "):
        amount_str = raw[6:].strip()
        if amount_str.isdigit():
            return "raise", int(amount_str)
    return raw, None


def prompt_player_action(player: Player, options: List[str], call_amount: int, min_raise: int, highest_bet: int) -> Tuple[str, Optional[int]]:
    print(f"{player.name} 的手牌: {' '.join(str(card) for card in player.hole_cards)}")
    print(f"当前最高注: {highest_bet}, 你已下: {player.current_bet}, 需跟注: {call_amount}")
    print(f"可用动作: {', '.join(options)}")
    if "bet" in options:
        print(f"请输入 bet X 来下注，最低 {min_raise}")
    if "raise" in options:
        print(f"请输入 raise X 来加注，最低加到 {highest_bet + min_raise}")
    while True:
        raw = input(f"{player.name} 请选择动作: ").strip()
        action, amount = parse_action(raw)
        if action not in options:
            print("无效动作，请选择合法动作。")
            continue
        if action in {"bet", "raise"} and amount is None:
            print("请输入正确的金额，例如 bet 100 或 raise 200。")
            continue
        return action, amount


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


def run_game() -> None:
    print("欢迎来到德州扑克 CLI 小程序！")
    names: List[str] = []
    for index in range(2):
        name = input(f"请输入玩家 {index + 1} 名称（默认 Player{index + 1}）: ").strip() or f"玩家{index + 1}"
        names.append(name)

    game = Game(names, starting_stack=1000, small_blind=10, big_blind=20)

    def action_provider(player: Player, options: List[str], call_amount: int, min_raise: int, highest_bet: int) -> Tuple[str, Optional[int]]:
        show_table(game, reveal_all=False, current_player=player)
        action, amount = prompt_player_action(player, options, call_amount, min_raise, highest_bet)
        return action, amount

    while True:
        result = game.play_hand(action_provider)
        summarize_hand(game, result)
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
            game = Game(names, starting_stack=1000, small_blind=10, big_blind=20)
            continue

        if input("继续下一手？(Y/n): ").strip().lower().startswith("n"):
            print("游戏结束，谢谢游玩！")
            break


if __name__ == "__main__":
    run_game()
