"""Modern PyTorch training loop for the structured HUNL ReBeL network."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from .data import load_shard
from .model import PokerModelConfig, PokerReBeLNet


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    seed: int = 1
    profile: str = "h100"
    compile_model: bool = True
    device: str = "cuda"
    checkpoint_every: int = 1

    @classmethod
    def profile_defaults(cls, profile: str) -> "TrainConfig":
        if profile == "smoke":
            return cls(
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                warmup_steps=2,
                profile="smoke",
                compile_model=False,
                device="cpu",
            )
        if profile == "base":
            return cls(batch_size=128, profile="base")
        if profile == "h100":
            return cls(batch_size=384, profile="h100")
        raise ValueError(f"Unknown profile: {profile}")


def train(
    data_dir: Path,
    output_dir: Path,
    config: TrainConfig,
    resume: Optional[Path] = None,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model_config = PokerModelConfig(**checkpoint["model_config"])
    else:
        model_config = PokerModelConfig.profile(config.profile)
    model = PokerReBeLNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    train_model = model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(model, mode="max-autotune")

    shards = sorted(data_dir.rglob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No .npz shards found in {data_dir}")
    total_batches = sum(
        math.ceil(len(load_shard(path)["policy_inputs"]) / config.batch_size)
        for path in shards
    )
    total_steps = max(1, total_batches * config.epochs)

    metrics = {}
    for epoch in range(start_epoch, config.epochs):
        random.Random(config.seed + epoch).shuffle(shards)
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_batches = 0
        started = time.time()
        model.train()
        for shard_path in shards:
            shard = load_shard(shard_path)
            for batch in _iter_batches(shard, config.batch_size, config.seed + epoch):
                lr_scale = _learning_rate_scale(
                    global_step, config.warmup_steps, total_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * lr_scale

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    value_prediction = train_model(
                        batch["value_inputs"].to(device), head="value"
                    )
                    value_target = batch["value_targets"].to(device)
                    value_mask = batch["value_masks"].to(device)
                    value_loss_raw = torch.nn.functional.smooth_l1_loss(
                        value_prediction,
                        value_target,
                        reduction="none",
                        beta=0.02,
                    )
                    value_loss = (
                        value_loss_raw * value_mask
                    ).sum() / value_mask.sum().clamp_min(1.0)

                    policy_logits = train_model(
                        batch["policy_inputs"].to(device), head="policy"
                    )
                    legal_mask = batch["legal_masks"].to(device).bool()
                    policy_logits = policy_logits.masked_fill(
                        ~legal_mask[:, None, :], -1e9
                    )
                    target = batch["policy_targets"].to(device)
                    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
                        1e-6
                    )
                    cross_entropy = -(
                        target
                        * torch.nn.functional.log_softmax(policy_logits, dim=-1)
                    ).sum(dim=-1)
                    policy_weight = batch["policy_weights"].to(device)
                    policy_loss = (
                        cross_entropy * policy_weight
                    ).sum() / policy_weight.sum().clamp_min(1e-6)
                    loss = value_loss + config.policy_loss_weight * policy_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                optimizer.step()
                global_step += 1
                epoch_batches += 1
                epoch_value_loss += float(value_loss.detach())
                epoch_policy_loss += float(policy_loss.detach())

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "value_loss": epoch_value_loss / max(1, epoch_batches),
            "policy_loss": epoch_policy_loss / max(1, epoch_batches),
            "seconds": time.time() - started,
            "parameters": model.parameter_count(),
        }
        print(json.dumps(metrics, sort_keys=True))
        if (epoch + 1) % config.checkpoint_every == 0:
            _save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
                metrics,
            )
            _save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
                metrics,
            )
    return metrics


def load_model(
    checkpoint_path: Path, device: str = "cuda"
) -> PokerReBeLNet:
    target = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=target, weights_only=False
    )
    model_config = PokerModelConfig(**checkpoint["model_config"])
    model = PokerReBeLNet(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(target).eval()
    return model


def _iter_batches(
    shard: Dict[str, np.ndarray], batch_size: int, seed: int
) -> Iterable[Dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    value_order = rng.permutation(len(shard["value_inputs"]))
    policy_order = rng.permutation(len(shard["policy_inputs"]))
    steps = max(
        math.ceil(len(value_order) / batch_size),
        math.ceil(len(policy_order) / batch_size),
    )
    for step in range(steps):
        value_indices = _cyclic_indices(value_order, step, batch_size)
        policy_indices = _cyclic_indices(policy_order, step, batch_size)
        yield {
            "value_inputs": torch.from_numpy(
                shard["value_inputs"][value_indices].astype(np.float32)
            ),
            "value_targets": torch.from_numpy(
                shard["value_targets"][value_indices].astype(np.float32)
            ),
            "value_masks": torch.from_numpy(
                shard["value_masks"][value_indices].astype(np.float32)
            ),
            "policy_inputs": torch.from_numpy(
                shard["policy_inputs"][policy_indices].astype(np.float32)
            ),
            "policy_targets": torch.from_numpy(
                shard["policy_targets"][policy_indices].astype(np.float32)
                / 255.0
            ),
            "policy_weights": torch.from_numpy(
                shard["policy_weights"][policy_indices].astype(np.float32)
            ),
            "legal_masks": torch.from_numpy(
                shard["legal_masks"][policy_indices].astype(np.bool_)
            ),
        }


def _cyclic_indices(order, step, batch_size):
    start = step * batch_size
    indices = np.arange(start, start + batch_size) % len(order)
    return order[indices]


def _learning_rate_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    config,
    metrics,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_config": asdict(config),
            "model_config": model.config.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
