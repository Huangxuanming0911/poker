import random
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["♠", "♥", "♦", "♣"]
RANK_VALUE = {rank: index for index, rank in enumerate(RANKS, start=2)}
RANK_LABELS = {value: rank for rank, value in RANK_VALUE.items()}

ActionProvider = Callable[["Player", List[str], int, int, int], Tuple[str, Optional[int]]]


@dataclass
class Card:
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def value(self) -> int:
        return RANK_VALUE[self.rank]


@dataclass(eq=False)
class Player:
    name: str
    stack: int
    hole_cards: List[Card] = field(default_factory=list)
    current_bet: int = 0
    total_bet: int = 0
    folded: bool = False
    all_in: bool = False
    acted: bool = False

    __hash__ = object.__hash__

    def reset_for_new_hand(self) -> None:
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet = 0
        self.folded = False
        self.all_in = False
        self.acted = False

    def bet_to(self, target_bet: int) -> int:
        if target_bet <= self.current_bet:
            return 0
        amount = target_bet - self.current_bet
        if amount >= self.stack:
            amount = self.stack
            self.all_in = True
        self.stack -= amount
        self.current_bet += amount
        self.total_bet += amount
        return amount

    def take_action(self, amount: int) -> int:
        if amount >= self.stack:
            amount = self.stack
            self.all_in = True
        self.stack -= amount
        self.current_bet += amount
        self.total_bet += amount
        return amount

    def is_active(self) -> bool:
        return not self.folded and self.stack > 0

    def is_still_playing(self) -> bool:
        return not self.folded


def create_deck() -> List[Card]:
    return [Card(rank, suit) for rank in RANKS for suit in SUITS]


def shuffle_deck() -> List[Card]:
    deck = create_deck()
    random.shuffle(deck)
    return deck


def sorted_rank_values(cards: List[Card]) -> List[int]:
    return sorted((card.value() for card in cards), reverse=True)


def find_straight(ranks: List[int]) -> Optional[int]:
    unique_ranks = sorted(set(ranks), reverse=True)
    if unique_ranks[0] == 14:
        unique_ranks.append(1)
    for i in range(len(unique_ranks) - 4):
        window = unique_ranks[i : i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return window[0]
    return None


def evaluate_5card_hand(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    ranks = [card.value() for card in cards]
    suits = [card.suit for card in cards]
    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    sorted_ranks = sorted(ranks, reverse=True)
    flush_suit = next((s for s, count in Counter(suits).items() if count >= 5), None)
    is_flush = flush_suit is not None
    flush_ranks = sorted([card.value() for card in cards if card.suit == flush_suit], reverse=True) if is_flush else []
    straight_high = find_straight(sorted_ranks)
    straight_flush_high = None

    if is_flush:
        flush_cards = [card for card in cards if card.suit == flush_suit]
        flush_ranks = sorted((card.value() for card in flush_cards), reverse=True)
        straight_flush_high = find_straight(flush_ranks)

    if straight_flush_high:
        return (8, (straight_flush_high,))

    if counts_sorted[0][1] == 4:
        four = counts_sorted[0][0]
        kicker = max(r for r in sorted_ranks if r != four)
        return (7, (four, kicker))

    if counts_sorted[0][1] == 3 and counts_sorted[1][1] >= 2:
        three = counts_sorted[0][0]
        pair = counts_sorted[1][0]
        return (6, (three, pair))

    if is_flush:
        return (5, tuple(flush_ranks[:5]))

    if straight_high:
        return (4, (straight_high,))

    if counts_sorted[0][1] == 3:
        three = counts_sorted[0][0]
        kickers = tuple(r for r in sorted_ranks if r != three)[:2]
        return (3, (three,) + kickers)

    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        top_pair = counts_sorted[0][0]
        second_pair = counts_sorted[1][0]
        kicker = max(r for r in sorted_ranks if r != top_pair and r != second_pair)
        return (2, (top_pair, second_pair, kicker))

    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = tuple(r for r in sorted_ranks if r != pair)[:3]
        return (1, (pair,) + kickers)

    return (0, tuple(sorted_ranks[:5]))


def evaluate_best_hand(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    if len(cards) < 5:
        ranks = tuple(sorted((card.value() for card in cards), reverse=True))
        return (0, ranks)
    best_rank = (-1, ())
    for combo in itertools.combinations(cards, 5):
        rank = evaluate_5card_hand(list(combo))
        if rank > best_rank:
            best_rank = rank
    return best_rank


def compare_hands(cards_a: List[Card], cards_b: List[Card]) -> int:
    rank_a = evaluate_best_hand(cards_a)
    rank_b = evaluate_best_hand(cards_b)
    if rank_a > rank_b:
        return 1
    if rank_a < rank_b:
        return -1
    return 0


def describe_hand_value(hand_value: Tuple[int, Tuple[int, ...]]) -> str:
    categories = {
        8: "同花顺",
        7: "四条",
        6: "葫芦",
        5: "同花",
        4: "顺子",
        3: "三条",
        2: "两对",
        1: "一对",
        0: "高牌",
    }
    category = categories[hand_value[0]]
    detail_values = hand_value[1]
    detail_labels = [RANK_LABELS.get(value, str(value)) for value in detail_values]

    if hand_value[0] == 8:
        return f"同花顺 {detail_labels[0]}"
    if hand_value[0] == 7:
        return f"四条 {detail_labels[0]}，踢脚 {detail_labels[1]}"
    if hand_value[0] == 6:
        return f"葫芦 {detail_labels[0]} 搭 {detail_labels[1]}"
    if hand_value[0] == 5:
        return f"同花 {' '.join(detail_labels)}"
    if hand_value[0] == 4:
        return f"顺子 {detail_labels[0]}"
    if hand_value[0] == 3:
        return f"三条 {detail_labels[0]}，踢脚 {' '.join(detail_labels[1:])}"
    if hand_value[0] == 2:
        return f"两对 {detail_labels[0]} 和 {detail_labels[1]}，踢脚 {detail_labels[2]}"
    if hand_value[0] == 1:
        return f"一对 {detail_labels[0]}，踢脚 {' '.join(detail_labels[1:])}"
    return f"高牌 {' '.join(detail_labels)}"


class Game:
    def __init__(self, player_names: List[str], starting_stack: int = 1000, small_blind: int = 10, big_blind: int = 20):
        self.players = [Player(name, starting_stack) for name in player_names]
        self.button_index = 0
        self.deck: List[Card] = []
        self.community_cards: List[Card] = []
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.minimum_raise = big_blind
        self.highest_bet = 0

    def alive_players(self) -> List[Player]:
        return [player for player in self.players if player.stack > 0 or not player.folded]

    def active_players(self) -> List[Player]:
        return [player for player in self.players if not player.folded and not player.all_in]

    def playing_players(self) -> List[Player]:
        return [player for player in self.players if not player.folded]

    def current_pot(self) -> int:
        return sum(player.total_bet for player in self.players)

    def next_index(self, index: int) -> int:
        return (index + 1) % len(self.players)

    def reset_round_bets(self) -> None:
        for player in self.players:
            player.current_bet = 0
            player.acted = False
        self.highest_bet = 0

    def reset_for_new_hand(self) -> None:
        self.deck = shuffle_deck()
        self.community_cards = []
        for player in self.players:
            player.reset_for_new_hand()
        self.highest_bet = 0

    def post_blinds(self) -> None:
        if len(self.players) == 2:
            sb_index = self.button_index
            bb_index = self.next_index(sb_index)
        else:
            sb_index = self.next_index(self.button_index)
            bb_index = self.next_index(sb_index)
        small_blind_player = self.players[sb_index]
        big_blind_player = self.players[bb_index]
        small_blind_player.bet_to(self.small_blind)
        big_blind_player.bet_to(self.big_blind)
        self.highest_bet = max(p.current_bet for p in self.players)

    def deal_hole_cards(self) -> None:
        for _ in range(2):
            for player in self.players:
                player.hole_cards.append(self.deck.pop())

    def deal_flop(self) -> None:
        self.deck.pop()
        self.community_cards.extend(self.deck.pop() for _ in range(3))

    def deal_turn(self) -> None:
        self.deck.pop()
        self.community_cards.append(self.deck.pop())

    def deal_river(self) -> None:
        self.deck.pop()
        self.community_cards.append(self.deck.pop())

    def determine_best_players(self, candidates: List[Player]) -> List[Player]:
        best_value = None
        winners: List[Player] = []
        for player in candidates:
            value = evaluate_best_hand(player.hole_cards + self.community_cards)
            if best_value is None or value > best_value:
                best_value = value
                winners = [player]
            elif value == best_value:
                winners.append(player)
        return winners

    def split_pot(self) -> Dict[Player, int]:
        contributions = {player: player.total_bet for player in self.players}
        shares: Dict[Player, int] = defaultdict(int)
        while any(amount > 0 for amount in contributions.values()):
            active_bets = [amount for amount in contributions.values() if amount > 0]
            if not active_bets:
                break
            cap = min(active_bets)
            pot_players = [player for player, amount in contributions.items() if amount >= cap]
            pot_amount = cap * len(pot_players)
            for player in pot_players:
                contributions[player] -= cap
            eligible = [player for player in pot_players if not player.folded]
            if not eligible:
                continue
            winners = self.determine_best_players(eligible)
            share = pot_amount // len(winners)
            for winner in winners:
                shares[winner] += share
            remainder = pot_amount - share * len(winners)
            if remainder:
                shares[winners[0]] += remainder
        return shares

    def is_single_player_remaining(self) -> bool:
        return sum(1 for player in self.players if not player.folded) <= 1

    def get_action_options(self, player: Player) -> List[str]:
        call_amount = self.highest_bet - player.current_bet
        options = ["fold"]
        if call_amount == 0:
            options.append("check")
            if player.stack > 0:
                options.append("bet")
                options.append("all-in")
        else:
            if player.stack > 0:
                options.append("call")
                options.append("all-in")
                if player.stack > call_amount + self.minimum_raise - 1:
                    options.append("raise")
        return options

    def validate_action(self, player: Player, action: str, amount: Optional[int]) -> Tuple[bool, Optional[str]]:
        call_amount = self.highest_bet - player.current_bet
        if action == "fold":
            return True, None
        if action == "check":
            if call_amount != 0:
                return False, "当前需跟注金额不为 0，不能过牌。"
            return True, None
        if action == "call":
            if call_amount == 0:
                return False, "当前无需跟注。"
            if player.stack == 0:
                return False, "没有筹码可用于跟注。"
            return True, None
        if action == "all-in":
            if player.stack == 0:
                return False, "当前已经没有筹码。"
            return True, None
        if action == "bet":
            if self.highest_bet != 0:
                return False, "只有在当前还没有下注时才可下注。"
            if amount is None or amount < self.minimum_raise:
                return False, f"下注金额至少为 {self.minimum_raise}。"
            if amount > player.stack:
                return False, "筹码不足，无法下注该金额。"
            return True, None
        if action == "raise":
            if self.highest_bet == 0:
                return False, "当前还没有下注，不能加注，请使用下注。"
            if amount is None:
                return False, "请输入加注金额。"
            if amount <= self.highest_bet:
                return False, "加注金额必须高于当前最高注。"
            min_raise_amount = self.highest_bet + self.minimum_raise
            if amount < min_raise_amount:
                return False, f"至少加注到 {min_raise_amount}。"
            if amount > player.current_bet + player.stack:
                return False, "筹码不足，无法加注该金额。"
            return True, None
        return False, "未知动作。"

    def betting_round(self, start_index: int, action_provider: ActionProvider) -> None:
        if self.is_single_player_remaining():
            return
        for player in self.players:
            if not player.folded and not player.all_in:
                player.acted = False
        self.highest_bet = max(player.current_bet for player in self.players)
        pending = [player for player in self.players if not player.folded and not player.all_in]
        current_index = start_index

        while pending and not self.is_single_player_remaining():
            player = self.players[current_index]
            current_index = self.next_index(current_index)
            if player.folded or player.all_in:
                continue
            if player not in pending and player.current_bet == self.highest_bet:
                continue

            call_amount = self.highest_bet - player.current_bet
            options = self.get_action_options(player)
            action, amount = action_provider(player, options, call_amount, self.minimum_raise, self.highest_bet)
            valid, message = self.validate_action(player, action, amount)
            if not valid:
                raise ValueError(message or "无效动作。")

            player.acted = True
            if action == "fold":
                player.folded = True
                if player in pending:
                    pending.remove(player)
                continue

            if action == "check":
                if player in pending:
                    pending.remove(player)
                continue

            if action == "call":
                if call_amount >= player.stack:
                    player.bet_to(player.current_bet + player.stack)
                else:
                    player.bet_to(self.highest_bet)
                if player in pending:
                    pending.remove(player)
                continue

            if action == "all-in":
                player.bet_to(player.current_bet + player.stack)
                if player.current_bet > self.highest_bet:
                    self.highest_bet = player.current_bet
                    pending = [p for p in self.players if not p.folded and not p.all_in and p.current_bet < self.highest_bet]
                    if player in pending:
                        pending.remove(player)
                else:
                    if player in pending:
                        pending.remove(player)
                continue

            if action in ("bet", "raise"):
                assert amount is not None
                player.bet_to(amount)
                if player.current_bet > self.highest_bet:
                    self.highest_bet = player.current_bet
                    pending = [p for p in self.players if not p.folded and not p.all_in and p.current_bet < self.highest_bet]
                    if player in pending:
                        pending.remove(player)
                else:
                    if player in pending:
                        pending.remove(player)
                continue

        for player in self.players:
            player.acted = False

    def play_hand(self, action_provider: ActionProvider) -> Dict[str, object]:
        self.reset_for_new_hand()
        self.post_blinds()
        self.deal_hole_cards()

        if len(self.players) == 2:
            first_to_act = self.button_index
        else:
            first_to_act = self.next_index(self.next_index(self.button_index))
        self.betting_round(first_to_act, action_provider)
        if self.is_single_player_remaining():
            results = self.split_pot()
            for player, amount in results.items():
                player.stack += amount
            self.button_index = self.next_index(self.button_index)
            return {
                "stage": "preflop",
                "results": results,
                "community": self.community_cards,
            }

        self.deal_flop()
        self.reset_round_bets()
        middle_first = self.next_index(self.button_index)
        self.betting_round(middle_first, action_provider)
        if self.is_single_player_remaining():
            results = self.split_pot()
            for player, amount in results.items():
                player.stack += amount
            self.button_index = self.next_index(self.button_index)
            return {
                "stage": "flop",
                "results": results,
                "community": self.community_cards,
            }

        self.deal_turn()
        self.reset_round_bets()
        self.betting_round(middle_first, action_provider)
        if self.is_single_player_remaining():
            results = self.split_pot()
            for player, amount in results.items():
                player.stack += amount
            self.button_index = self.next_index(self.button_index)
            return {
                "stage": "turn",
                "results": results,
                "community": self.community_cards,
            }

        self.deal_river()
        self.reset_round_bets()
        self.betting_round(middle_first, action_provider)

        results = self.split_pot()
        for player, amount in results.items():
            player.stack += amount
        self.button_index = self.next_index(self.button_index)
        return {
            "stage": "river",
            "results": results,
            "community": self.community_cards,
        }
