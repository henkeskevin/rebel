# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import torch
from torch import nn

from cfvpy.nlhe.cards import NUM_HOLE_COMBOS
from cfvpy.nlhe.pbs import POLICY_INPUT_SIZE, VALUE_INPUT_SIZE


def build_mlp(
    *,
    n_in,
    n_hidden,
    n_layers,
    out_size=None,
    act=None,
    use_layer_norm=False,
    dropout=0,
):
    if act is None:
        act = GELU()
    build_norm_layer = (
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    vals_net = []
    for _ in range(n_layers):
        vals_net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        vals_net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*vals_net)


def input_size(num_faces, num_dice):
    return 1 + 1 + (2 * num_faces * num_dice + 1) + 2 * output_size(num_faces, num_dice)


def output_size(num_faces, num_dice):
    return num_faces ** num_dice


class Net2(nn.Module):
    def __init__(
        self,
        *,
        num_faces,
        num_dice,
        n_hidden=256,
        use_layer_norm=False,
        dropout=0,
        n_layers=3,
    ):
        super().__init__()

        n_in = input_size(num_faces, num_dice)
        self.body = build_mlp(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.output = nn.Linear(
            n_hidden if n_layers > 0 else n_in, output_size(num_faces, num_dice)
        )
        # Make initial predictions closer to 0.
        with torch.no_grad():
            self.output.weight.data *= 0.01
            self.output.bias *= 0.01

    def forward(self, packed_input: torch.Tensor):
        return self.output(self.body(packed_input))


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


class PokerValueNet(nn.Module):
    """Counterfactual value network from the ReBeL HUNL architecture."""

    def __init__(
        self,
        *,
        n_hidden=1536,
        n_layers=6,
        use_layer_norm=True,
        dropout=0,
    ):
        super().__init__()
        self.body = build_mlp(
            n_in=VALUE_INPUT_SIZE,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        body_output_size = n_hidden if n_layers > 0 else VALUE_INPUT_SIZE
        self.output = nn.Linear(body_output_size, NUM_HOLE_COMBOS)
        with torch.no_grad():
            self.output.weight.data *= 0.01
            self.output.bias.data *= 0.01

    def forward(self, packed_input: torch.Tensor):
        return self.output(self.body(packed_input))


class PokerPolicyNet(nn.Module):
    """Policy logits for every private hand and abstract action slot."""

    def __init__(
        self,
        *,
        max_actions=9,
        n_hidden=1536,
        n_layers=6,
        use_layer_norm=True,
        dropout=0,
    ):
        super().__init__()
        self.max_actions = max_actions
        self.body = build_mlp(
            n_in=POLICY_INPUT_SIZE,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        body_output_size = n_hidden if n_layers > 0 else POLICY_INPUT_SIZE
        self.output = nn.Linear(body_output_size, NUM_HOLE_COMBOS * max_actions)

    def forward(self, packed_input: torch.Tensor):
        logits = self.output(self.body(packed_input))
        return logits.view(-1, NUM_HOLE_COMBOS, self.max_actions)
