# Training HUNL ReBeL on Google Colab H100

This pipeline is an approximation of ReBeL, not a reproduction of Meta's
private poker system or checkpoint. It provides a complete loop:

1. sample public belief states;
2. solve depth-limited public trees with CFR;
3. store conditional values and per-hand policies;
4. train a shared value/policy network;
5. reuse the checkpoint as a leaf evaluator for another data-generation pass.

The `h100` model has about 70 million parameters and trains in BF16 with
`torch.compile`. Data generation remains CPU-heavy because the reference CFR
solver is NumPy-based.

## Colab setup

Select **Runtime > Change runtime type > H100 GPU**, then:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/henkeskevin/rebel.git /content/rebel
%cd /content/rebel
!pip install -q -r requirements-nlhe.txt
```

Confirm the accelerator:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

## End-to-end smoke test

Run this before a long job:

```bash
!python scripts/generate_nlhe_data.py \
  --output /content/nlhe_smoke/data/river \
  --street river --examples 16 --shard-size 8 \
  --cfr-iterations 4 --search-depth 2 --workers 2

!python scripts/train_nlhe.py \
  --data /content/nlhe_smoke/data \
  --output /content/nlhe_smoke/run \
  --profile smoke --device cuda --epochs 1 \
  --batch-size 8 --no-compile

!python scripts/eval_nlhe.py \
  --checkpoint /content/nlhe_smoke/run/latest.pt \
  --data /content/nlhe_smoke/data --device cuda
```

## Recommended bootstrap

Persist outputs in Drive:

```python
ROOT = "/content/drive/MyDrive/rebel_hunl"
```

Start with river because its pseudo-leaves have exact showdown values:

```bash
!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data/bootstrap/river" \
  --street river --examples 20000 --shard-size 64 \
  --cfr-iterations 128 --search-depth 6 --workers 8

!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data_validation/river" \
  --street river --examples 1000 --shard-size 64 \
  --cfr-iterations 128 --search-depth 6 --workers 8 --seed 9001
```

Add earlier streets with Monte Carlo check-down values. These are bootstrap
targets, not final equilibrium labels:

```bash
!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data/bootstrap/turn" \
  --street turn --examples 10000 --shard-size 64 \
  --cfr-iterations 64 --search-depth 4 \
  --rollout-boards 4 --workers 8

!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data/bootstrap/flop" \
  --street flop --examples 5000 --shard-size 64 \
  --cfr-iterations 64 --search-depth 4 \
  --rollout-boards 4 --workers 8

!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data/bootstrap/preflop" \
  --street preflop --examples 2500 --shard-size 64 \
  --cfr-iterations 32 --search-depth 3 \
  --rollout-boards 2 --workers 8
```

Train the mixed-street model:

```bash
!python scripts/train_nlhe.py \
  --data "$ROOT/data/bootstrap" \
  --output "$ROOT/runs/bootstrap_h100" \
  --profile h100 --device cuda --epochs 20 \
  --batch-size 384
```

If memory is tight, reduce `--batch-size` to `256` or `128`. If
`torch.compile` causes a driver-specific failure, add `--no-compile`.

## ReBeL-style improvement pass

Generate fresh targets using the trained value network at search leaves:

```bash
!python scripts/generate_nlhe_data.py \
  --output "$ROOT/data/iteration_01/river" \
  --street river --examples 10000 --shard-size 64 \
  --cfr-iterations 256 --search-depth 6 \
  --checkpoint "$ROOT/runs/bootstrap_h100/latest.pt" \
  --device cuda --workers 1
```

Then continue training on both old and new data:

```bash
!python scripts/train_nlhe.py \
  --data "$ROOT/data" \
  --output "$ROOT/runs/iteration_01" \
  --profile h100 --device cuda --epochs 30 \
  --batch-size 384 \
  --resume "$ROOT/runs/bootstrap_h100/latest.pt"
```

Repeat data generation and training. Increase CFR iterations before increasing
network size; stronger search targets usually matter more than another layer.

## Evaluate and inspect

```bash
!python scripts/eval_nlhe.py \
  --checkpoint "$ROOT/runs/iteration_01/latest.pt" \
  --data "$ROOT/data_validation/river" --device cuda

!python scripts/solve_nlhe.py \
  --checkpoint "$ROOT/runs/iteration_01/latest.pt" \
  --device cuda --iterations 256 --depth 4 --hand AsKd
```

Track at least:

- held-out value MAE in stack fractions;
- policy cross-entropy;
- head-to-head duplicate matches against older checkpoints;
- local best-response or exploitability estimates in the action abstraction;
- throughput of generated PBS examples per second.

Do not judge poker strength from training loss alone.
