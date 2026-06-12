"""CFR training-data generation and shard storage for HUNL."""

from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .cards import NUM_CARDS, NUM_HOLE_COMBOS, valid_combo_mask
from .game import (
    ActionType,
    BettingConfig,
    BettingState,
    HeadsUpNoLimitHoldem,
    Street,
)
from .pbs import PublicBeliefState, normalize_range
from .solver import (
    CFRSolver,
    LeafValueFn,
    showdown_leaf_value_components,
    showdown_leaf_values,
)


@dataclass(frozen=True)
class GenerationConfig:
    examples: int = 128
    shard_size: int = 32
    cfr_iterations: int = 32
    search_depth: int = 4
    street: Street = Street.RIVER
    seed: int = 1
    range_temperature: float = 1.25
    rollout_boards: int = 4
    stack_size: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    pot_fractions: tuple = (0.5, 1.0)


class MonteCarloCheckdownEvaluator:
    """Blocker-correct check-down baseline for incomplete public boards."""

    def __init__(self, samples: int, seed: int):
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, state: BettingState, traverser: int, ranges: np.ndarray
    ) -> np.ndarray:
        if len(state.board) == 5:
            return showdown_leaf_values(state, traverser, ranges)
        missing = 5 - len(state.board)
        deck = np.asarray(
            [card for card in range(NUM_CARDS) if card not in state.board],
            dtype=np.int16,
        )
        numerator = np.zeros(NUM_HOLE_COMBOS, dtype=np.float64)
        denominator = np.zeros(NUM_HOLE_COMBOS, dtype=np.float64)
        for _ in range(self.samples):
            runout = tuple(
                int(card)
                for card in self.rng.choice(deck, size=missing, replace=False)
            )
            full_board = (*state.board, *runout)
            river_state = replace(
                state,
                street=Street.RIVER,
                board=full_board,
            )
            value_mass, compatible_mass = showdown_leaf_value_components(
                river_state, traverser, ranges
            )
            numerator += value_mass
            denominator += compatible_mass
        result = np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)
        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator > 0,
        )
        return result


def sample_range(
    board: Sequence[int],
    rng: np.random.Generator,
    temperature: float,
) -> tuple:
    logits = rng.normal(0.0, temperature, size=NUM_HOLE_COMBOS)
    weights = np.exp(logits - logits.max())
    return normalize_range(weights, board)


def sample_public_belief_state(
    game: HeadsUpNoLimitHoldem,
    street: Street,
    rng: np.random.Generator,
    range_temperature: float,
) -> PublicBeliefState:
    for _ in range(100):
        state = game.initial_state()
        deck = list(range(NUM_CARDS))
        rng.shuffle(deck)
        board_by_street = {
            Street.FLOP: tuple(deck[:3]),
            Street.TURN: tuple(deck[:4]),
            Street.RIVER: tuple(deck[:5]),
        }

        failed = False
        while state.street < street:
            state = _play_conservative_round(game, state, rng)
            if state.terminal or state.all_in:
                failed = True
                break
            state = game.advance_street(
                state, board_by_street[Street(state.street + 1)]
            )
        if failed:
            continue

        prefix_actions = int(rng.integers(0, 3))
        for _ in range(prefix_actions):
            if state.round_complete or state.terminal:
                break
            action = _sample_conservative_action(game, state, rng)
            state = game.act(state, action)
        if state.round_complete or state.terminal:
            continue

        ranges = (
            sample_range(state.board, rng, range_temperature),
            sample_range(state.board, rng, range_temperature),
        )
        return PublicBeliefState(state=state, ranges=ranges)
    raise RuntimeError("Could not sample a non-terminal public belief state")


def generate_shards(
    output_dir: Path,
    config: GenerationConfig,
    leaf_value_fn: Optional[LeafValueFn] = None,
    prefix: str = "shard",
) -> Iterable[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    game = HeadsUpNoLimitHoldem(
        BettingConfig(
            stack_size=config.stack_size,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            pot_fractions=config.pot_fractions,
        )
    )
    rng = np.random.default_rng(config.seed)
    if leaf_value_fn is None:
        if config.street == Street.RIVER:
            leaf_value_fn = showdown_leaf_values
        else:
            leaf_value_fn = MonteCarloCheckdownEvaluator(
                config.rollout_boards, config.seed + 10_000
            )

    shard = []
    shard_id = 0
    for example_id in range(config.examples):
        pbs = sample_public_belief_state(
            game, config.street, rng, config.range_temperature
        )
        result = CFRSolver(
            game,
            pbs,
            max_depth=config.search_depth,
            leaf_value_fn=leaf_value_fn,
        ).solve(config.cfr_iterations)
        shard.append(_pack_example(game, pbs, result))
        if len(shard) >= config.shard_size or example_id + 1 == config.examples:
            path = output_dir / f"{prefix}_{shard_id:05d}.npz"
            _save_shard(path, shard, config)
            yield path
            shard = []
            shard_id += 1


def generate_shards_parallel(
    output_dir: Path,
    config: GenerationConfig,
    workers: int,
) -> Iterable[Path]:
    if workers <= 1:
        yield from generate_shards(output_dir, config)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = [
        config.examples // workers + (worker < config.examples % workers)
        for worker in range(workers)
    ]
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for worker, count in enumerate(counts):
            if count == 0:
                continue
            worker_config = replace(
                config,
                examples=count,
                seed=config.seed + worker * 100_003,
            )
            jobs.append(
                executor.submit(
                    _generate_worker,
                    output_dir,
                    worker_config,
                    f"worker{worker:03d}",
                )
            )
        for future in as_completed(jobs):
            for path in future.result():
                yield Path(path)


def load_shard(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _pack_example(game, pbs, result):
    stack_size = game.config.stack_size
    valid = np.asarray(valid_combo_mask(pbs.state.board), dtype=np.uint8)
    value_inputs = np.asarray(
        [
            pbs.value_vector(0, stack_size),
            pbs.value_vector(1, stack_size),
        ],
        dtype=np.float16,
    )
    value_targets = (
        np.asarray(result.root_values, dtype=np.float32) / stack_size
    ).astype(np.float16)
    policy_input = np.asarray(
        pbs.policy_vector(pbs.state.player, stack_size), dtype=np.float16
    )
    policy_target = np.clip(
        np.rint(result.root_strategy * 255.0), 0, 255
    ).astype(np.uint8)
    actor_range = np.asarray(pbs.ranges[pbs.state.player], dtype=np.float32)
    policy_weight = actor_range + valid * (0.05 / max(1, valid.sum()))
    policy_weight /= policy_weight.sum()
    legal_mask = np.zeros(game.config.max_actions, dtype=np.uint8)
    for action in result.tree[0].actions:
        legal_mask[action.slot] = 1
    return {
        "value_inputs": value_inputs,
        "value_targets": value_targets,
        "value_masks": np.stack((valid, valid)),
        "policy_input": policy_input,
        "policy_target": policy_target,
        "policy_weight": policy_weight.astype(np.float16),
        "legal_mask": legal_mask,
    }


def _save_shard(path: Path, examples, config: GenerationConfig) -> None:
    arrays = {
        "value_inputs": np.concatenate(
            [example["value_inputs"] for example in examples], axis=0
        ),
        "value_targets": np.concatenate(
            [example["value_targets"] for example in examples], axis=0
        ),
        "value_masks": np.concatenate(
            [example["value_masks"] for example in examples], axis=0
        ),
        "policy_inputs": np.stack(
            [example["policy_input"] for example in examples]
        ),
        "policy_targets": np.stack(
            [example["policy_target"] for example in examples]
        ),
        "policy_weights": np.stack(
            [example["policy_weight"] for example in examples]
        ),
        "legal_masks": np.stack(
            [example["legal_mask"] for example in examples]
        ),
        "street": np.asarray([int(config.street)], dtype=np.int8),
        "stack_size": np.asarray([config.stack_size], dtype=np.int32),
    }
    np.savez(path, **arrays)


def _generate_worker(output_dir, config, prefix):
    return [
        str(path)
        for path in generate_shards(output_dir, config, prefix=prefix)
    ]


def _play_conservative_round(game, state, rng):
    for _ in range(12):
        if state.round_complete or state.terminal:
            return state
        state = game.act(state, _sample_conservative_action(game, state, rng))
    raise RuntimeError("Betting round did not complete")


def _sample_conservative_action(game, state, rng):
    actions = game.legal_actions(state)
    check_calls = [
        action for action in actions if action.kind == ActionType.CHECK_CALL
    ]
    raises = [
        action
        for action in actions
        if action.kind == ActionType.BET_RAISE
    ]
    if raises and rng.random() < 0.25:
        return raises[int(rng.integers(0, min(2, len(raises))))]
    return check_calls[0]
