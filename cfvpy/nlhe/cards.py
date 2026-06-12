"""Card, range, and showdown utilities for heads-up hold'em."""

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence, Tuple

NUM_CARDS = 52
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def make_card(rank: int, suit: int) -> int:
    if not 0 <= rank < len(RANKS):
        raise ValueError(f"Invalid rank: {rank}")
    if not 0 <= suit < len(SUITS):
        raise ValueError(f"Invalid suit: {suit}")
    return suit * len(RANKS) + rank


def rank_of(card: int) -> int:
    _validate_card(card)
    return card % len(RANKS)


def suit_of(card: int) -> int:
    _validate_card(card)
    return card // len(RANKS)


def parse_card(value: str) -> int:
    value = value.strip()
    if len(value) != 2:
        raise ValueError(f"Expected a two-character card, got {value!r}")
    rank = value[0].upper()
    suit = value[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"Invalid card: {value!r}")
    return make_card(RANKS.index(rank), SUITS.index(suit))


def card_to_string(card: int) -> str:
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]


def _validate_card(card: int) -> None:
    if not isinstance(card, int) or not 0 <= card < NUM_CARDS:
        raise ValueError(f"Invalid card id: {card!r}")


HOLE_COMBOS: Tuple[Tuple[int, int], ...] = tuple(combinations(range(NUM_CARDS), 2))
NUM_HOLE_COMBOS = len(HOLE_COMBOS)
_COMBO_TO_INDEX = {combo: index for index, combo in enumerate(HOLE_COMBOS)}


def combo_index(first: int, second: int) -> int:
    _validate_card(first)
    _validate_card(second)
    if first == second:
        raise ValueError("A hole-card combo needs two distinct cards")
    return _COMBO_TO_INDEX[tuple(sorted((first, second)))]


def combo_cards(index: int) -> Tuple[int, int]:
    if not 0 <= index < NUM_HOLE_COMBOS:
        raise ValueError(f"Invalid combo index: {index}")
    return HOLE_COMBOS[index]


def validate_distinct_cards(cards: Iterable[int]) -> Tuple[int, ...]:
    cards = tuple(cards)
    for card in cards:
        _validate_card(card)
    if len(set(cards)) != len(cards):
        raise ValueError("Cards must be distinct")
    return cards


def valid_combo_mask(public_cards: Sequence[int]) -> Tuple[bool, ...]:
    blocked = set(validate_distinct_cards(public_cards))
    return tuple(
        first not in blocked and second not in blocked
        for first, second in HOLE_COMBOS
    )


def combos_compatible(first_index: int, second_index: int) -> bool:
    first = combo_cards(first_index)
    second = combo_cards(second_index)
    return not (set(first) & set(second))


def evaluate_five(cards: Sequence[int]) -> Tuple[int, ...]:
    """Return a lexicographically comparable five-card hand rank."""
    cards = validate_distinct_cards(cards)
    if len(cards) != 5:
        raise ValueError("evaluate_five expects exactly five cards")

    ranks = [rank_of(card) for card in cards]
    counts = Counter(ranks)
    ordered_ranks = sorted(ranks, reverse=True)
    groups = sorted(
        ((count, rank) for rank, count in counts.items()), reverse=True
    )
    is_flush = len({suit_of(card) for card in cards}) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[-1] == 4:
            straight_high = unique_ranks[0]
        elif unique_ranks == [12, 3, 2, 1, 0]:
            straight_high = 3  # Five-high wheel.

    if is_flush and straight_high is not None:
        return (8, straight_high)
    if groups[0][0] == 4:
        quad_rank = groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad_rank)
        return (7, quad_rank, kicker)
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if is_flush:
        return (5, *ordered_ranks)
    if straight_high is not None:
        return (4, straight_high)
    if groups[0][0] == 3:
        trip_rank = groups[0][1]
        kickers = sorted(
            (rank for rank in ranks if rank != trip_rank), reverse=True
        )
        return (3, trip_rank, *kickers)
    pairs = sorted(
        (rank for rank, count in counts.items() if count == 2), reverse=True
    )
    if len(pairs) == 2:
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (2, pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
        return (1, pair, *kickers)
    return (0, *ordered_ranks)


def evaluate_seven(cards: Sequence[int]) -> Tuple[int, ...]:
    cards = validate_distinct_cards(cards)
    if len(cards) != 7:
        raise ValueError("evaluate_seven expects exactly seven cards")
    return max(evaluate_five(candidate) for candidate in combinations(cards, 5))


def compare_hands(
    first_hole: Sequence[int],
    second_hole: Sequence[int],
    board: Sequence[int],
) -> int:
    first_hole = validate_distinct_cards(first_hole)
    second_hole = validate_distinct_cards(second_hole)
    board = validate_distinct_cards(board)
    if len(first_hole) != 2 or len(second_hole) != 2 or len(board) != 5:
        raise ValueError(
            "Hold'em showdown requires two hole cards and five board cards"
        )
    validate_distinct_cards((*first_hole, *second_hole, *board))
    first_rank = evaluate_seven((*first_hole, *board))
    second_rank = evaluate_seven((*second_hole, *board))
    return (first_rank > second_rank) - (first_rank < second_rank)
