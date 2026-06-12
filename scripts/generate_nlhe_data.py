"""Generate HUNL CFR training shards."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cfvpy.nlhe import Street
from cfvpy.nlhe.data import (
    GenerationConfig,
    generate_shards,
    generate_shards_parallel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--street",
        choices=("preflop", "flop", "turn", "river"),
        default="river",
    )
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--cfr-iterations", type=int, default=32)
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--rollout-boards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = GenerationConfig(
        examples=args.examples,
        shard_size=args.shard_size,
        cfr_iterations=args.cfr_iterations,
        search_depth=args.search_depth,
        street=Street[args.street.upper()],
        rollout_boards=args.rollout_boards,
        seed=args.seed,
    )
    if args.checkpoint is not None:
        if args.workers != 1:
            raise ValueError("--checkpoint currently requires --workers 1")
        from cfvpy.nlhe.game import BettingConfig, HeadsUpNoLimitHoldem
        from cfvpy.nlhe.inference import TorchLeafEvaluator

        game = HeadsUpNoLimitHoldem(
            BettingConfig(
                stack_size=config.stack_size,
                small_blind=config.small_blind,
                big_blind=config.big_blind,
                pot_fractions=config.pot_fractions,
            )
        )
        evaluator = TorchLeafEvaluator(
            args.checkpoint, game, device=args.device
        )
        paths = generate_shards(args.output, config, evaluator)
    else:
        paths = generate_shards_parallel(args.output, config, args.workers)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
