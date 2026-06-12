"""Depth-limited CFR for a single heads-up no-limit public belief state."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .cards import (
    HOLE_COMBOS,
    NUM_HOLE_COMBOS,
    evaluate_seven,
    valid_combo_mask,
)
from .game import Action, BettingState, HeadsUpNoLimitHoldem, TerminalType
from .pbs import PublicBeliefState

LeafValueFn = Callable[
    [BettingState, int, np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class PublicTreeNode:
    state: BettingState
    depth: int
    actions: Tuple[Action, ...]
    children: Tuple[int, ...]


@dataclass(frozen=True)
class CFRResult:
    tree: Tuple[PublicTreeNode, ...]
    strategy: np.ndarray
    root_values: Tuple[np.ndarray, np.ndarray]
    root_counterfactual_values: Tuple[np.ndarray, np.ndarray]

    @property
    def root_strategy(self) -> np.ndarray:
        return self.strategy[0]


class CFRSolver:
    """Full-width private-hand CFR over a depth-limited public action tree.

    The leaf callback receives normalized ranges with shape ``[2, 1326]`` and
    returns conditional values for all 1,326 hands of ``traverser``. The solver
    converts them to counterfactual values using blocker-compatible opponent
    reach mass.
    """

    def __init__(
        self,
        game: HeadsUpNoLimitHoldem,
        root: PublicBeliefState,
        *,
        max_depth: int = 2,
        leaf_value_fn: Optional[LeafValueFn] = None,
        root_strategy_prior: Optional[np.ndarray] = None,
        regret_matching_plus: bool = True,
        linear_averaging: bool = True,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self.game = game
        self.root = root
        self.max_depth = max_depth
        self.regret_matching_plus = regret_matching_plus
        self.linear_averaging = linear_averaging
        self.tree = _build_public_tree(game, root.state, max_depth)
        has_pseudo_leaf = any(
            not node.children and not node.state.terminal for node in self.tree
        )
        if has_pseudo_leaf and leaf_value_fn is None:
            raise ValueError(
                "A leaf_value_fn is required when search stops before a terminal"
            )
        self.leaf_value_fn = leaf_value_fn or zero_leaf_values

        shape = (len(self.tree), NUM_HOLE_COMBOS, game.config.max_actions)
        self.regrets = np.zeros(shape, dtype=np.float32)
        self.strategy_sum = np.zeros(shape, dtype=np.float32)
        self.strategy = np.zeros(shape, dtype=np.float32)
        self.legal_mask = np.zeros(
            (len(self.tree), game.config.max_actions), dtype=bool
        )
        self._initialize_uniform_strategy()
        if root_strategy_prior is not None:
            self._set_root_strategy_prior(root_strategy_prior)

        self.initial_ranges = np.asarray(root.ranges, dtype=np.float32)
        self.root_value_sum = [
            np.zeros(NUM_HOLE_COMBOS, dtype=np.float64),
            np.zeros(NUM_HOLE_COMBOS, dtype=np.float64),
        ]
        self.traversals = [0, 0]

    def solve(self, num_iterations: int) -> CFRResult:
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        for iteration in range(num_iterations):
            self.step(iteration % 2)
        counterfactual_values = (
            (self.root_value_sum[0] / max(1, self.traversals[0])).astype(
                np.float32
            ),
            (self.root_value_sum[1] / max(1, self.traversals[1])).astype(
                np.float32
            ),
        )
        conditional_values = []
        compatibility = _compatibility_matrix()
        for traverser in (0, 1):
            opponent_mass = compatibility.dot(
                self.initial_ranges[1 - traverser]
            )
            values = np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)
            np.divide(
                counterfactual_values[traverser],
                opponent_mass,
                out=values,
                where=opponent_mass > 0,
            )
            conditional_values.append(values)
        return CFRResult(
            tree=self.tree,
            strategy=self.average_strategy(),
            root_values=(conditional_values[0], conditional_values[1]),
            root_counterfactual_values=counterfactual_values,
        )

    def step(self, traverser: int) -> np.ndarray:
        if traverser not in (0, 1):
            raise ValueError(f"Invalid traverser: {traverser}")

        reaches = self._compute_reaches()
        values = self._leaf_counterfactual_values(traverser, reaches)

        for node_id in range(len(self.tree) - 1, -1, -1):
            node = self.tree[node_id]
            if not node.children:
                continue

            if node.state.player == traverser:
                node_value = np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)
                for action, child_id in zip(node.actions, node.children):
                    node_value += (
                        self.strategy[node_id, :, action.slot] * values[child_id]
                    )
                values[node_id] = node_value
                for action, child_id in zip(node.actions, node.children):
                    self.regrets[node_id, :, action.slot] += (
                        values[child_id] - node_value
                    )
            else:
                node_value = np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)
                for child_id in node.children:
                    node_value += values[child_id]
                values[node_id] = node_value

        self.root_value_sum[traverser] += values[0]
        self.traversals[traverser] += 1
        if self.regret_matching_plus:
            for node_id, node in enumerate(self.tree):
                if not node.actions or node.state.player != traverser:
                    continue
                slots = [action.slot for action in node.actions]
                self.regrets[node_id][:, slots] = np.maximum(
                    self.regrets[node_id][:, slots], 0.0
                )
        self._regret_match(traverser)

        updated_reaches = self._compute_reaches()
        average_weight = (
            float(sum(self.traversals)) if self.linear_averaging else 1.0
        )
        for node_id, node in enumerate(self.tree):
            if not node.actions or node.state.player != traverser:
                continue
            weight = updated_reaches[traverser, node_id, :, None]
            self.strategy_sum[node_id] += (
                average_weight * weight * self.strategy[node_id]
            )
        return values[0].copy()

    def average_strategy(self) -> np.ndarray:
        average = np.zeros_like(self.strategy_sum)
        for node_id, node in enumerate(self.tree):
            if not node.actions:
                continue
            legal_slots = [action.slot for action in node.actions]
            node_sum = self.strategy_sum[node_id][:, legal_slots]
            totals = node_sum.sum(axis=1, keepdims=True)
            positive = totals[:, 0] > 0
            normalized = np.empty_like(node_sum)
            if np.any(positive):
                normalized[positive] = node_sum[positive] / totals[positive]
            if np.any(~positive):
                normalized[~positive] = 1.0 / len(legal_slots)
            average[node_id][:, legal_slots] = normalized
        return average

    def _initialize_uniform_strategy(self) -> None:
        for node_id, node in enumerate(self.tree):
            if not node.actions:
                continue
            probability = 1.0 / len(node.actions)
            for action in node.actions:
                self.legal_mask[node_id, action.slot] = True
                self.strategy[node_id, :, action.slot] = probability

    def _set_root_strategy_prior(self, prior: np.ndarray) -> None:
        expected = (NUM_HOLE_COMBOS, self.game.config.max_actions)
        prior = np.asarray(prior, dtype=np.float32)
        if prior.shape != expected:
            raise ValueError(
                f"root_strategy_prior must have shape {expected}, "
                f"got {prior.shape}"
            )
        slots = [action.slot for action in self.tree[0].actions]
        clipped = np.maximum(prior[:, slots], 0.0)
        totals = clipped.sum(axis=1, keepdims=True)
        normalized = np.empty_like(clipped)
        positive = totals[:, 0] > 0
        normalized[positive] = clipped[positive] / totals[positive]
        normalized[~positive] = 1.0 / len(slots)
        self.strategy[0] = 0.0
        self.strategy[0][:, slots] = normalized

    def _regret_match(self, traverser: int) -> None:
        for node_id, node in enumerate(self.tree):
            if not node.actions or node.state.player != traverser:
                continue
            slots = [action.slot for action in node.actions]
            positive = np.maximum(self.regrets[node_id][:, slots], 0.0)
            totals = positive.sum(axis=1, keepdims=True)
            matched = np.empty_like(positive)
            nonzero = totals[:, 0] > 0
            matched[nonzero] = positive[nonzero] / totals[nonzero]
            matched[~nonzero] = 1.0 / len(slots)
            self.strategy[node_id] = 0.0
            self.strategy[node_id][:, slots] = matched

    def _compute_reaches(self) -> np.ndarray:
        reaches = np.zeros(
            (2, len(self.tree), NUM_HOLE_COMBOS), dtype=np.float32
        )
        reaches[:, 0, :] = self.initial_ranges
        for node_id, node in enumerate(self.tree):
            for action, child_id in zip(node.actions, node.children):
                reaches[:, child_id, :] = reaches[:, node_id, :]
                actor = node.state.player
                reaches[actor, child_id, :] *= self.strategy[
                    node_id, :, action.slot
                ]
        return reaches

    def _leaf_counterfactual_values(
        self, traverser: int, reaches: np.ndarray
    ) -> np.ndarray:
        values = np.zeros(
            (len(self.tree), NUM_HOLE_COMBOS), dtype=np.float32
        )
        pseudo_node_ids = []
        pseudo_states = []
        pseudo_ranges = []
        pseudo_masses = []
        for node_id, node in enumerate(self.tree):
            if node.children:
                continue
            state = node.state
            opponent_reach = reaches[1 - traverser, node_id]
            if state.terminal:
                values[node_id] = _terminal_values(
                    self.game, state, traverser, opponent_reach
                )
                continue

            normalized_ranges = np.stack(
                (
                    _normalize_reach(reaches[0, node_id], state.board),
                    _normalize_reach(reaches[1, node_id], state.board),
                )
            )
            pseudo_node_ids.append(node_id)
            pseudo_states.append(state)
            pseudo_ranges.append(normalized_ranges)
            pseudo_masses.append(
                _compatibility_matrix().dot(opponent_reach)
            )

        if not pseudo_node_ids:
            return values
        if hasattr(self.leaf_value_fn, "batch"):
            conditional_batch = self.leaf_value_fn.batch(
                pseudo_states,
                traverser,
                np.stack(pseudo_ranges),
            )
        else:
            conditional_batch = np.stack(
                [
                    self.leaf_value_fn(state, traverser, ranges)
                    for state, ranges in zip(pseudo_states, pseudo_ranges)
                ]
            )
        conditional_batch = np.asarray(conditional_batch, dtype=np.float32)
        expected_shape = (len(pseudo_node_ids), NUM_HOLE_COMBOS)
        if conditional_batch.shape != expected_shape:
            raise ValueError(
                f"Leaf evaluator must return shape {expected_shape}, "
                f"got {conditional_batch.shape}"
            )
        for row, node_id in enumerate(pseudo_node_ids):
            conditional_values = conditional_batch[row]
            if conditional_values.shape != (NUM_HOLE_COMBOS,):
                raise ValueError(
                    "leaf_value_fn must return shape "
                    f"({NUM_HOLE_COMBOS},), got {conditional_values.shape}"
                )
            compatible_mass = pseudo_masses[row]
            valid = np.asarray(valid_combo_mask(pseudo_states[row].board))
            values[node_id, valid] = (
                conditional_values[valid] * compatible_mass[valid]
            )
        return values


def _build_public_tree(
    game: HeadsUpNoLimitHoldem, root: BettingState, max_depth: int
) -> Tuple[PublicTreeNode, ...]:
    mutable_nodes: List[Tuple[BettingState, int, Tuple[Action, ...], List[int]]] = [
        (root, 0, (), [])
    ]
    node_id = 0
    while node_id < len(mutable_nodes):
        state, depth, _, children = mutable_nodes[node_id]
        if (
            depth >= max_depth
            or state.terminal
            or state.round_complete
        ):
            node_id += 1
            continue
        actions = game.legal_actions(state)
        mutable_nodes[node_id] = (state, depth, actions, children)
        for action in actions:
            child_id = len(mutable_nodes)
            children.append(child_id)
            mutable_nodes.append((game.act(state, action), depth + 1, (), []))
        node_id += 1
    return tuple(
        PublicTreeNode(state, depth, actions, tuple(children))
        for state, depth, actions, children in mutable_nodes
    )


def _normalize_reach(reach: np.ndarray, board: Sequence[int]) -> np.ndarray:
    valid = np.asarray(valid_combo_mask(board))
    normalized = np.where(valid, np.maximum(reach, 0.0), 0.0).astype(np.float32)
    total = float(normalized.sum())
    if total > 0:
        normalized /= total
    else:
        normalized[valid] = 1.0 / int(valid.sum())
    return normalized


def _terminal_values(
    game: HeadsUpNoLimitHoldem,
    state: BettingState,
    traverser: int,
    opponent_reach: np.ndarray,
) -> np.ndarray:
    compatible = _compatibility_matrix()
    compatible_mass = compatible.dot(opponent_reach)
    valid = np.asarray(valid_combo_mask(state.board))

    if state.terminal_type == TerminalType.FOLD:
        payoff = game.terminal_payoff(state, traverser)
        values = compatible_mass * payoff
        values[~valid] = 0.0
        return values.astype(np.float32)

    if state.terminal_type != TerminalType.SHOWDOWN:
        raise ValueError(f"Unsupported terminal type: {state.terminal_type}")

    outcomes = _showdown_outcomes(tuple(state.board))
    if traverser == 0:
        perspective = outcomes
    else:
        perspective = -outcomes.T
    win_mass = (perspective > 0).dot(opponent_reach)
    loss_mass = (perspective < 0).dot(opponent_reach)
    values = (
        win_mass * state.total_commit[1 - traverser]
        - loss_mass * state.total_commit[traverser]
    )
    values[~valid] = 0.0
    return values.astype(np.float32)


@lru_cache(maxsize=1)
def _compatibility_matrix() -> np.ndarray:
    first = np.asarray([combo[0] for combo in HOLE_COMBOS], dtype=np.int16)
    second = np.asarray([combo[1] for combo in HOLE_COMBOS], dtype=np.int16)
    return (
        (first[:, None] != first[None, :])
        & (first[:, None] != second[None, :])
        & (second[:, None] != first[None, :])
        & (second[:, None] != second[None, :])
    )


@lru_cache(maxsize=16)
def _showdown_outcomes(board: Tuple[int, ...]) -> np.ndarray:
    if len(board) != 5:
        raise ValueError("Showdown requires a five-card board")
    valid = valid_combo_mask(board)
    ranks = [
        evaluate_seven((*combo, *board)) if is_valid else None
        for combo, is_valid in zip(HOLE_COMBOS, valid)
    ]
    ordered = {
        rank: index
        for index, rank in enumerate(
            sorted({rank for rank in ranks if rank is not None})
        )
    }
    strength = np.asarray(
        [ordered[rank] if rank is not None else -1 for rank in ranks],
        dtype=np.int16,
    )
    outcomes = np.sign(strength[:, None] - strength[None, :]).astype(np.int8)
    outcomes[~_compatibility_matrix()] = 0
    invalid = ~np.asarray(valid)
    outcomes[invalid, :] = 0
    outcomes[:, invalid] = 0
    return outcomes


def zero_leaf_values(
    state: BettingState, traverser: int, ranges: np.ndarray
) -> np.ndarray:
    """Explicit smoke-test evaluator; it does not produce a poker strategy."""
    del state, traverser, ranges
    return np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)


def showdown_leaf_values(
    state: BettingState, traverser: int, ranges: np.ndarray
) -> np.ndarray:
    """Exact check-down values on a complete five-card board."""
    numerator, compatible_mass = showdown_leaf_value_components(
        state, traverser, ranges
    )
    values = np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)
    np.divide(
        numerator,
        compatible_mass,
        out=values,
        where=compatible_mass > 0,
    )
    return values


def showdown_leaf_value_components(
    state: BettingState, traverser: int, ranges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return payoff-weighted mass and compatible mass for check-down values."""
    if len(state.board) != 5:
        raise ValueError("showdown_leaf_values requires a five-card board")
    if ranges.shape != (2, NUM_HOLE_COMBOS):
        raise ValueError(
            f"Expected ranges with shape (2, {NUM_HOLE_COMBOS})"
        )
    outcomes = _showdown_outcomes(tuple(state.board))
    perspective = outcomes if traverser == 0 else -outcomes.T
    opponent_range = ranges[1 - traverser]
    valid = np.asarray(valid_combo_mask(state.board))
    pair_valid = (
        _compatibility_matrix()
        & valid[:, None]
        & valid[None, :]
    )
    compatible_mass = pair_valid.dot(opponent_range)
    win_mass = (perspective > 0).dot(opponent_range)
    loss_mass = (perspective < 0).dot(opponent_range)
    numerator = (
        win_mass * state.total_commit[1 - traverser]
        - loss_mass * state.total_commit[traverser]
    )
    numerator[~valid] = 0.0
    compatible_mass[~valid] = 0.0
    return numerator.astype(np.float32), compatible_mass.astype(np.float32)
