"""Heads-up no-limit betting state for street-by-street ReBeL search."""

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
import math
from typing import Optional, Sequence, Tuple

from .cards import compare_hands, validate_distinct_cards


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class ActionType(Enum):
    FOLD = "fold"
    CHECK_CALL = "check_call"
    BET_RAISE = "bet_raise"
    ALL_IN = "all_in"


class TerminalType(Enum):
    FOLD = "fold"
    SHOWDOWN = "showdown"


@dataclass(frozen=True)
class Action:
    kind: ActionType
    amount: int
    slot: int

    def __str__(self) -> str:
        if self.kind == ActionType.FOLD:
            return "fold"
        if self.kind == ActionType.CHECK_CALL:
            return "check" if self.amount == 0 else f"call_to({self.amount})"
        if self.kind == ActionType.ALL_IN:
            return f"all_in_to({self.amount})"
        return f"raise_to({self.amount})"


@dataclass(frozen=True)
class ActionRecord:
    player: int
    street: Street
    action: Action


@dataclass(frozen=True)
class BettingConfig:
    # ACPC-style 200 big-blind stacks, represented in integer chips.
    stack_size: int = 20_000
    small_blind: int = 50
    big_blind: int = 100
    pot_fractions: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    max_actions: int = 9

    def __post_init__(self) -> None:
        if not 0 < self.small_blind < self.big_blind < self.stack_size:
            raise ValueError("Blinds and stack size are inconsistent")
        if len(self.pot_fractions) + 4 > self.max_actions:
            raise ValueError("Action abstraction exceeds max_actions")
        if any(fraction <= 0 for fraction in self.pot_fractions):
            raise ValueError("Pot fractions must be positive")


@dataclass(frozen=True)
class BettingState:
    street: Street
    board: Tuple[int, ...]
    player: int
    stacks: Tuple[int, int]
    street_commit: Tuple[int, int]
    total_commit: Tuple[int, int]
    current_bet: int
    last_full_raise_size: int
    last_aggressor: Optional[int] = None
    consecutive_checks: int = 0
    round_complete: bool = False
    all_in: bool = False
    terminal: bool = False
    terminal_type: Optional[TerminalType] = None
    winner: Optional[int] = None
    history: Tuple[ActionRecord, ...] = ()

    @property
    def pot(self) -> int:
        return self.total_commit[0] + self.total_commit[1]

    @property
    def to_call(self) -> int:
        return max(0, self.current_bet - self.street_commit[self.player])

    @property
    def effective_stack(self) -> int:
        opponent = 1 - self.player
        return min(
            self.stacks[self.player],
            self.stacks[opponent] + self.street_commit[opponent]
            - self.street_commit[self.player],
        )


class HeadsUpNoLimitHoldem:
    """Public betting mechanics for two-player no-limit hold'em.

    Search stops at the end of each street. Chance then reveals the next board
    card(s), and a new depth-limited subgame starts from ``advance_street``.
    """

    FOLD_SLOT = 0
    CHECK_CALL_SLOT = 1
    MIN_RAISE_SLOT = 2

    def __init__(self, config: BettingConfig = BettingConfig()):
        self.config = config
        self.ALL_IN_SLOT = config.max_actions - 1

    def initial_state(self) -> BettingState:
        cfg = self.config
        return BettingState(
            street=Street.PREFLOP,
            board=(),
            player=0,  # The button/small blind acts first preflop.
            stacks=(
                cfg.stack_size - cfg.small_blind,
                cfg.stack_size - cfg.big_blind,
            ),
            street_commit=(cfg.small_blind, cfg.big_blind),
            total_commit=(cfg.small_blind, cfg.big_blind),
            current_bet=cfg.big_blind,
            last_full_raise_size=cfg.big_blind,
        )

    def legal_actions(self, state: BettingState) -> Tuple[Action, ...]:
        if state.terminal or state.round_complete:
            return ()

        player = state.player
        opponent = 1 - player
        to_call = state.to_call
        actions = []

        if to_call > 0:
            actions.append(Action(ActionType.FOLD, 0, self.FOLD_SLOT))

        call_delta = min(to_call, state.stacks[player])
        call_target = state.street_commit[player] + call_delta
        actions.append(
            Action(ActionType.CHECK_CALL, call_target, self.CHECK_CALL_SLOT)
        )

        actor_max = state.street_commit[player] + state.stacks[player]
        opponent_max = state.street_commit[opponent] + state.stacks[opponent]
        max_raise_to = min(actor_max, opponent_max)
        if (
            state.stacks[player] <= to_call
            or state.stacks[opponent] == 0
            or max_raise_to <= state.current_bet
        ):
            return tuple(actions)

        min_raise_to = state.current_bet + state.last_full_raise_size
        normal_targets = []
        if min_raise_to < max_raise_to:
            normal_targets.append((min_raise_to, self.MIN_RAISE_SLOT))

        pot_after_call = state.pot + to_call
        base = state.street_commit[player] + to_call
        for offset, fraction in enumerate(self.config.pot_fractions):
            target = base + _round_chips(fraction * pot_after_call)
            if min_raise_to <= target < max_raise_to:
                normal_targets.append((target, self.MIN_RAISE_SLOT + 1 + offset))

        seen = set()
        for target, slot in sorted(normal_targets):
            if target in seen:
                continue
            seen.add(target)
            actions.append(Action(ActionType.BET_RAISE, target, slot))

        actions.append(Action(ActionType.ALL_IN, max_raise_to, self.ALL_IN_SLOT))
        return tuple(actions)

    def act(self, state: BettingState, action: Action) -> BettingState:
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action} for state {state}")

        player = state.player
        opponent = 1 - player
        record = ActionRecord(player, state.street, action)
        history = (*state.history, record)

        if action.kind == ActionType.FOLD:
            return replace(
                state,
                terminal=True,
                round_complete=True,
                terminal_type=TerminalType.FOLD,
                winner=opponent,
                history=history,
            )

        if action.kind == ActionType.CHECK_CALL:
            delta = action.amount - state.street_commit[player]
            stacks, street_commit, total_commit = _commit_chips(state, player, delta)
            was_call = state.to_call > 0
            if was_call:
                # Calling the forced big blind preserves the big blind's option.
                round_complete = (
                    state.last_aggressor is not None
                    or stacks[player] == 0
                    or stacks[opponent] == 0
                )
                checks = 0 if round_complete else 1
            else:
                checks = state.consecutive_checks + 1
                round_complete = checks == 2
            all_in = round_complete and (stacks[0] == 0 or stacks[1] == 0)
            terminal = round_complete and state.street == Street.RIVER
            return replace(
                state,
                player=opponent,
                stacks=stacks,
                street_commit=street_commit,
                total_commit=total_commit,
                consecutive_checks=checks,
                round_complete=round_complete,
                all_in=all_in,
                terminal=terminal,
                terminal_type=TerminalType.SHOWDOWN if terminal else None,
                history=history,
            )

        delta = action.amount - state.street_commit[player]
        stacks, street_commit, total_commit = _commit_chips(state, player, delta)
        raise_size = action.amount - state.current_bet
        last_full_raise_size = state.last_full_raise_size
        if raise_size >= state.last_full_raise_size:
            last_full_raise_size = raise_size
        return replace(
            state,
            player=opponent,
            stacks=stacks,
            street_commit=street_commit,
            total_commit=total_commit,
            current_bet=action.amount,
            last_full_raise_size=last_full_raise_size,
            last_aggressor=player,
            consecutive_checks=0,
            history=history,
        )

    def advance_street(
        self, state: BettingState, public_cards: Sequence[int]
    ) -> BettingState:
        if state.terminal or not state.round_complete:
            raise ValueError("A new street requires a completed non-terminal round")
        if state.street == Street.RIVER:
            raise ValueError("There is no street after the river")

        next_street = Street(state.street + 1)
        board = validate_distinct_cards(public_cards)
        expected_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
        if len(board) != expected_cards[next_street]:
            raise ValueError(
                f"{next_street.name} requires {expected_cards[next_street]} board cards"
            )
        if tuple(board[: len(state.board)]) != state.board:
            raise ValueError("The new board must extend the existing board")

        terminal = state.all_in and next_street == Street.RIVER
        return replace(
            state,
            street=next_street,
            board=board,
            player=1,  # The big blind acts first postflop.
            street_commit=(0, 0),
            current_bet=0,
            last_full_raise_size=self.config.big_blind,
            last_aggressor=None,
            consecutive_checks=0,
            round_complete=state.all_in,
            terminal=terminal,
            terminal_type=TerminalType.SHOWDOWN if terminal else None,
        )

    def terminal_payoff(
        self,
        state: BettingState,
        player: int,
        hole_cards: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    ) -> float:
        if not state.terminal or state.terminal_type is None:
            raise ValueError("Payoff is only defined for terminal states")
        if player not in (0, 1):
            raise ValueError(f"Invalid player: {player}")

        if state.terminal_type == TerminalType.FOLD:
            result = 1 if state.winner == player else -1
        else:
            if hole_cards is None:
                raise ValueError("Showdown payoff requires both players' hole cards")
            result = compare_hands(hole_cards[0], hole_cards[1], state.board)
            if player == 1:
                result *= -1

        if result > 0:
            return float(state.total_commit[1 - player])
        if result < 0:
            return float(-state.total_commit[player])
        return 0.0


def _round_chips(value: float) -> int:
    return int(math.floor(value + 0.5))


def _commit_chips(
    state: BettingState, player: int, amount: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    if amount < 0 or amount > state.stacks[player]:
        raise ValueError(f"Cannot commit {amount} chips")
    stacks = list(state.stacks)
    street_commit = list(state.street_commit)
    total_commit = list(state.total_commit)
    stacks[player] -= amount
    street_commit[player] += amount
    total_commit[player] += amount
    return tuple(stacks), tuple(street_commit), tuple(total_commit)
