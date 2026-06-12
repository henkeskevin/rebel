"""Public-belief-state encoding used by ReBeL value and policy networks."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from .cards import (
    HOLE_COMBOS,
    NUM_CARDS,
    NUM_HOLE_COMBOS,
    combo_cards,
    valid_combo_mask,
)
from .game import BettingState, Street

VALUE_PUBLIC_FEATURE_SIZE = 2 + 2 + 4 + NUM_CARDS + 3
POLICY_PUBLIC_FEATURE_SIZE = VALUE_PUBLIC_FEATURE_SIZE + 3
VALUE_INPUT_SIZE = VALUE_PUBLIC_FEATURE_SIZE + 2 * NUM_HOLE_COMBOS
POLICY_INPUT_SIZE = POLICY_PUBLIC_FEATURE_SIZE + 2 * NUM_HOLE_COMBOS


def normalize_range(
    weights: Sequence[float], public_cards: Sequence[int]
) -> Tuple[float, ...]:
    if len(weights) != NUM_HOLE_COMBOS:
        raise ValueError(f"Expected {NUM_HOLE_COMBOS} range weights")
    mask = valid_combo_mask(public_cards)
    filtered = [
        max(0.0, float(weight)) if valid else 0.0
        for weight, valid in zip(weights, mask)
    ]
    total = sum(filtered)
    if total == 0:
        valid_count = sum(mask)
        return tuple((1.0 / valid_count) if valid else 0.0 for valid in mask)
    return tuple(weight / total for weight in filtered)


def bayes_update(
    prior: Sequence[float],
    action_probability: Sequence[float],
    public_cards: Sequence[int],
) -> Tuple[float, ...]:
    if len(prior) != NUM_HOLE_COMBOS:
        raise ValueError(f"Expected {NUM_HOLE_COMBOS} prior weights")
    if len(action_probability) != NUM_HOLE_COMBOS:
        raise ValueError(f"Expected {NUM_HOLE_COMBOS} action probabilities")
    posterior = [
        max(0.0, float(weight)) * max(0.0, float(probability))
        for weight, probability in zip(prior, action_probability)
    ]
    return normalize_range(posterior, public_cards)


def compatible_opponent_range(
    hero_combo: int,
    opponent_range: Sequence[float],
    public_cards: Sequence[int],
) -> Tuple[float, ...]:
    """Condition an opponent range on the hero's exact two cards."""
    if len(opponent_range) != NUM_HOLE_COMBOS:
        raise ValueError(f"Expected {NUM_HOLE_COMBOS} opponent weights")
    hero_cards = set(combo_cards(hero_combo))
    board = set(public_cards)
    filtered = []
    for weight, combo in zip(opponent_range, HOLE_COMBOS):
        valid = not (hero_cards & set(combo)) and not (board & set(combo))
        filtered.append(max(0.0, float(weight)) if valid else 0.0)
    total = sum(filtered)
    if total == 0:
        raise ValueError("Opponent range has no combo compatible with hero and board")
    return tuple(weight / total for weight in filtered)


@dataclass(frozen=True)
class PublicBeliefState:
    state: BettingState
    ranges: Tuple[Tuple[float, ...], Tuple[float, ...]]

    @classmethod
    def uniform(cls, state: BettingState) -> "PublicBeliefState":
        uniform = normalize_range([1.0] * NUM_HOLE_COMBOS, state.board)
        return cls(state=state, ranges=(uniform, uniform))

    def __post_init__(self) -> None:
        normalized = (
            normalize_range(self.ranges[0], self.state.board),
            normalize_range(self.ranges[1], self.state.board),
        )
        object.__setattr__(self, "ranges", normalized)

    def value_vector(self, traverser: int, stack_size: int) -> Tuple[float, ...]:
        if traverser not in (0, 1):
            raise ValueError(f"Invalid traverser: {traverser}")
        state = self.state
        vector = []
        vector.extend(_one_hot(state.player, 2))
        vector.extend(_one_hot(traverser, 2))
        vector.extend(_one_hot(int(state.street), len(Street)))
        vector.extend(1.0 if card in state.board else 0.0 for card in range(NUM_CARDS))
        vector.extend(
            (
                state.pot / stack_size,
                state.to_call / stack_size,
                min(state.stacks) / stack_size,
            )
        )
        vector.extend(self.ranges[0])
        vector.extend(self.ranges[1])
        if len(vector) != VALUE_INPUT_SIZE:
            raise AssertionError((len(vector), VALUE_INPUT_SIZE))
        return tuple(vector)

    def policy_vector(self, traverser: int, stack_size: int) -> Tuple[float, ...]:
        vector = list(self.value_vector(traverser, stack_size))
        insert_at = VALUE_PUBLIC_FEATURE_SIZE
        policy_extras = (
            self.state.total_commit[0] / stack_size,
            self.state.total_commit[1] / stack_size,
            1.0 if self.state.last_aggressor is not None else 0.0,
        )
        vector[insert_at:insert_at] = policy_extras
        if len(vector) != POLICY_INPUT_SIZE:
            raise AssertionError((len(vector), POLICY_INPUT_SIZE))
        return tuple(vector)


def _one_hot(index: int, size: int) -> Tuple[float, ...]:
    if not 0 <= index < size:
        raise ValueError((index, size))
    return tuple(1.0 if position == index else 0.0 for position in range(size))
