"""Neural leaf evaluation and policy inference for trained checkpoints."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .cards import NUM_HOLE_COMBOS
from .game import BettingState, HeadsUpNoLimitHoldem
from .pbs import PublicBeliefState
from .training import load_model


class TorchLeafEvaluator:
    def __init__(
        self,
        checkpoint: Path,
        game: HeadsUpNoLimitHoldem,
        device: str = "cuda",
        compile_model: bool = False,
    ):
        self.game = game
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model = load_model(checkpoint, str(self.device))
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def __call__(
        self, state: BettingState, traverser: int, ranges: np.ndarray
    ) -> np.ndarray:
        return self.batch([state], traverser, ranges[None, ...])[0]

    def batch(self, states, traverser: int, ranges: np.ndarray) -> np.ndarray:
        packed_rows = []
        for state, state_ranges in zip(states, ranges):
            pbs = PublicBeliefState(
                state,
                (
                    tuple(float(value) for value in state_ranges[0]),
                    tuple(float(value) for value in state_ranges[1]),
                ),
            )
            packed_rows.append(
                pbs.value_vector(traverser, self.game.config.stack_size)
            )
        packed = torch.tensor(
            packed_rows,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            values = self.model(packed, head="value")
        return (
            values.float().cpu().numpy() * self.game.config.stack_size
        ).astype(np.float32)


def policy_for_state(
    checkpoint: Path,
    game: HeadsUpNoLimitHoldem,
    pbs: PublicBeliefState,
    device: str = "cuda",
    model: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    target = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = model or load_model(checkpoint, str(target))
    packed = torch.tensor(
        pbs.policy_vector(pbs.state.player, game.config.stack_size),
        dtype=torch.float32,
        device=target,
    )[None, :]
    legal = np.zeros(game.config.max_actions, dtype=bool)
    for action in game.legal_actions(pbs.state):
        legal[action.slot] = True
    with torch.inference_mode(), torch.autocast(
        device_type=target.type,
        dtype=torch.bfloat16,
        enabled=target.type == "cuda",
    ):
        logits = model(packed, head="policy")[0].float()
        mask = torch.tensor(legal, device=target)
        logits = logits.masked_fill(~mask[None, :], -1e9)
        policy = torch.softmax(logits, dim=-1)
    if policy.shape != (NUM_HOLE_COMBOS, game.config.max_actions):
        raise AssertionError(policy.shape)
    return policy.cpu().numpy()
