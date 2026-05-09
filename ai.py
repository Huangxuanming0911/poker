import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from engine import ActionDecision, DecisionContext, evaluate_best_hand


class PokerAI(Protocol):
    def decide(self, context: DecisionContext) -> ActionDecision:
        ...


@dataclass
class BasicPokerAI:
    name: str = "AI"
    seed: Optional[int] = None
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def decide(self, context: DecisionContext) -> ActionDecision:
        available = set(context.available_actions)
        strength = self.estimate_strength(context)
        pot_odds = context.call_amount / max(1, context.pot + context.call_amount)

        if "all-in" in available and context.player.stack <= context.big_blind * 4 and strength >= 0.72:
            return ActionDecision("all-in")

        if "check" in available:
            if strength >= 0.82 and "raise-3x" in available and self.rng.random() < 0.75:
                return ActionDecision("raise-3x")
            if strength >= 0.62 and "raise-2x" in available and self.rng.random() < 0.55:
                return ActionDecision("raise-2x")
            return ActionDecision("check")

        if strength < max(0.18, pot_odds * 0.9):
            return ActionDecision("fold")

        if strength >= 0.84 and "raise-3x" in available and self.rng.random() < 0.65:
            return ActionDecision("raise-3x")
        if strength >= 0.68 and "raise-2x" in available and self.rng.random() < 0.50:
            return ActionDecision("raise-2x")
        if strength >= pot_odds or context.call_amount <= context.big_blind:
            return ActionDecision("call")
        return ActionDecision("fold")

    def estimate_strength(self, context: DecisionContext) -> float:
        if not context.community_cards:
            return self.estimate_preflop_strength(context)

        hand_value = evaluate_best_hand(list(context.hole_cards) + list(context.community_cards))
        category_score = 0.22 + hand_value[0] * 0.12
        kicker_score = sum(hand_value[1][:2]) / 28 if hand_value[1] else 0
        return min(0.98, category_score + kicker_score * 0.15 + len(context.community_cards) * 0.02)

    def estimate_preflop_strength(self, context: DecisionContext) -> float:
        cards = sorted(context.hole_cards, key=lambda card: card.value(), reverse=True)
        high = cards[0].value()
        low = cards[1].value()
        pair = high == low
        suited = cards[0].suit == cards[1].suit
        connected = abs(high - low) == 1

        score = (high + low) / 28 * 0.45
        if pair:
            score += 0.35 + high / 28 * 0.20
        if suited:
            score += 0.05
        if connected:
            score += 0.04
        if high >= 13:
            score += 0.05
        if low >= 10:
            score += 0.04
        return max(0.05, min(0.95, score))
