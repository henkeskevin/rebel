"""Train the structured HUNL value and policy network."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cfvpy.nlhe.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "base", "h100"), default="h100"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = TrainConfig.profile_defaults(args.profile)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.device is not None:
        config.device = args.device
    if args.no_compile:
        config.compile_model = False
    train(args.data, args.output, config, resume=args.resume)


if __name__ == "__main__":
    main()
