"""Evaluate a HUNL checkpoint on generated shards."""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cfvpy.nlhe.data import load_shard
from cfvpy.nlhe.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = (
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = load_model(args.checkpoint, device)
    value_errors = []
    policy_cross_entropies = []
    with torch.inference_mode():
        for path in sorted(args.data.rglob("*.npz")):
            shard = load_shard(path)
            value_input = torch.from_numpy(
                shard["value_inputs"].astype(np.float32)
            ).to(device)
            value_target = torch.from_numpy(
                shard["value_targets"].astype(np.float32)
            ).to(device)
            value_mask = torch.from_numpy(
                shard["value_masks"].astype(np.float32)
            ).to(device)
            prediction = model(value_input, head="value")
            error = ((prediction - value_target).abs() * value_mask).sum()
            value_errors.append(float(error / value_mask.sum().clamp_min(1)))

            policy_input = torch.from_numpy(
                shard["policy_inputs"].astype(np.float32)
            ).to(device)
            target = torch.from_numpy(
                shard["policy_targets"].astype(np.float32) / 255.0
            ).to(device)
            target /= target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            legal = torch.from_numpy(
                shard["legal_masks"].astype(np.bool_)
            ).to(device)
            logits = model(policy_input, head="policy")
            logits = logits.masked_fill(~legal[:, None, :], -1e9)
            ce = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            weights = torch.from_numpy(
                shard["policy_weights"].astype(np.float32)
            ).to(device)
            policy_cross_entropies.append(
                float((ce * weights).sum() / weights.sum().clamp_min(1e-6))
            )
    print(
        json.dumps(
            {
                "value_mae": float(np.mean(value_errors)),
                "policy_cross_entropy": float(
                    np.mean(policy_cross_entropies)
                ),
                "shards": len(value_errors),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
