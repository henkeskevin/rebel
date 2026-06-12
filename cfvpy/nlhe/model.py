"""Structured value and policy network for heads-up no-limit hold'em."""

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn

from .cards import HOLE_COMBOS, NUM_HOLE_COMBOS
from .pbs import (
    POLICY_PUBLIC_FEATURE_SIZE,
    VALUE_PUBLIC_FEATURE_SIZE,
)


@dataclass(frozen=True)
class PokerModelConfig:
    d_model: int = 1024
    n_blocks: int = 8
    range_dim: int = 384
    card_dim: int = 96
    hand_dim: int = 128
    dropout: float = 0.05
    max_actions: int = 9

    @classmethod
    def profile(cls, name: str) -> "PokerModelConfig":
        profiles = {
            "smoke": cls(
                d_model=128,
                n_blocks=2,
                range_dim=64,
                card_dim=32,
                hand_dim=32,
                dropout=0.0,
            ),
            "base": cls(
                d_model=512,
                n_blocks=6,
                range_dim=256,
                card_dim=64,
                hand_dim=96,
            ),
            "h100": cls(),
        }
        if name not in profiles:
            raise ValueError(f"Unknown model profile: {name}")
        return profiles[name]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 4, bias=False)
        self.fc2 = nn.Linear(width * 4, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = nn.functional.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return residual + x


class ComboHead(nn.Module):
    def __init__(self, d_model: int, hand_dim: int, output_dim: int):
        super().__init__()
        self.context = nn.Linear(d_model, hand_dim)
        self.norm = nn.LayerNorm(hand_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hand_dim, hand_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hand_dim * 2, output_dim),
        )

    def forward(
        self, context: torch.Tensor, combo_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.context(context)[:, None, :] + combo_embedding[None, :, :]
        return self.mlp(self.norm(hidden))


class PokerReBeLNet(nn.Module):
    """Shared structured backbone with per-combo value and policy heads.

    Compared with a flat 1,326-output MLP, this model explicitly shares card
    representations between related hands and uses separate encoders for the
    two public ranges, board cards, and scalar betting features.
    """

    def __init__(self, config: PokerModelConfig = PokerModelConfig()):
        super().__init__()
        self.config = config

        self.card_embedding = nn.Embedding(52, config.card_dim)
        self.rank_embedding = nn.Embedding(13, config.card_dim)
        self.suit_embedding = nn.Embedding(4, config.card_dim)
        self.board_projection = nn.Linear(config.card_dim, config.card_dim)

        public_without_board = VALUE_PUBLIC_FEATURE_SIZE - 52
        policy_without_board = POLICY_PUBLIC_FEATURE_SIZE - 52
        self.value_public = nn.Linear(public_without_board, config.card_dim)
        self.policy_public = nn.Linear(policy_without_board, config.card_dim)

        self.range_encoder = nn.Sequential(
            nn.Linear(NUM_HOLE_COMBOS, config.range_dim, bias=False),
            nn.LayerNorm(config.range_dim),
            nn.GELU(),
            nn.Linear(config.range_dim, config.range_dim, bias=False),
        )
        context_input = config.card_dim * 2 + config.range_dim * 4
        self.context_projection = nn.Linear(context_input, config.d_model)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.d_model, config.dropout)
            for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        combo_cards = torch.tensor(HOLE_COMBOS, dtype=torch.long)
        self.register_buffer("combo_cards", combo_cards, persistent=False)
        self.combo_projection = nn.Sequential(
            nn.Linear(config.card_dim * 3 + 4, config.hand_dim),
            nn.LayerNorm(config.hand_dim),
            nn.GELU(),
        )
        self.value_head = ComboHead(config.d_model, config.hand_dim, 1)
        self.policy_head = ComboHead(
            config.d_model, config.hand_dim, config.max_actions
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.card_embedding.weight, std=0.02)
        nn.init.normal_(self.rank_embedding.weight, std=0.02)
        nn.init.normal_(self.suit_embedding.weight, std=0.02)

    def _combo_embeddings(self) -> torch.Tensor:
        cards = self.combo_cards
        first = cards[:, 0]
        second = cards[:, 1]
        first_embedding = (
            self.card_embedding(first)
            + self.rank_embedding(first % 13)
            + self.suit_embedding(first // 13)
        )
        second_embedding = (
            self.card_embedding(second)
            + self.rank_embedding(second % 13)
            + self.suit_embedding(second // 13)
        )
        rank_gap = (first % 13 - second % 13).abs().float() / 12.0
        features = torch.stack(
            (
                ((first % 13) == (second % 13)).float(),
                ((first // 13) == (second // 13)).float(),
                rank_gap,
                (rank_gap == 1.0 / 12.0).float(),
            ),
            dim=1,
        )
        combined = torch.cat(
            (
                first_embedding + second_embedding,
                (first_embedding - second_embedding).abs(),
                first_embedding * second_embedding,
                features,
            ),
            dim=1,
        )
        return self.combo_projection(combined)

    def _encode_ranges(self, ranges: torch.Tensor) -> torch.Tensor:
        scaled = ranges * NUM_HOLE_COMBOS
        log_ranges = torch.log1p(scaled) / torch.log(
            ranges.new_tensor(float(NUM_HOLE_COMBOS + 1))
        )
        return torch.cat(
            (
                self.range_encoder(ranges[:, 0]),
                self.range_encoder(ranges[:, 1]),
                self.range_encoder(log_ranges[:, 0]),
                self.range_encoder(log_ranges[:, 1]),
            ),
            dim=1,
        )

    def _encode(
        self, packed_input: torch.Tensor, *, policy: bool
    ) -> torch.Tensor:
        public_size = (
            POLICY_PUBLIC_FEATURE_SIZE if policy else VALUE_PUBLIC_FEATURE_SIZE
        )
        public = packed_input[:, :public_size]
        ranges = packed_input[:, public_size:].reshape(
            -1, 2, NUM_HOLE_COMBOS
        )

        board = public[:, 8:60]
        card_ids = torch.arange(52, device=public.device)
        card_features = (
            self.card_embedding(card_ids)
            + self.rank_embedding(card_ids % 13)
            + self.suit_embedding(card_ids // 13)
        )
        board_count = board.sum(dim=1, keepdim=True).clamp_min(1.0)
        board_context = board @ card_features / board_count
        board_context = self.board_projection(board_context)

        public_without_board = torch.cat((public[:, :8], public[:, 60:]), dim=1)
        public_context = (
            self.policy_public(public_without_board)
            if policy
            else self.value_public(public_without_board)
        )
        context = torch.cat(
            (
                board_context,
                public_context,
                self._encode_ranges(ranges),
            ),
            dim=1,
        )
        context = self.context_projection(context)
        for block in self.blocks:
            context = block(context)
        return self.final_norm(context)

    def forward_value(self, packed_input: torch.Tensor) -> torch.Tensor:
        context = self._encode(packed_input, policy=False)
        return self.value_head(context, self._combo_embeddings()).squeeze(-1)

    def forward_policy(self, packed_input: torch.Tensor) -> torch.Tensor:
        context = self._encode(packed_input, policy=True)
        return self.policy_head(context, self._combo_embeddings())

    def forward(
        self, packed_input: torch.Tensor, head: str = "value"
    ) -> torch.Tensor:
        if head == "value":
            return self.forward_value(packed_input)
        if head == "policy":
            return self.forward_policy(packed_input)
        raise ValueError(f"Unknown head: {head}")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
