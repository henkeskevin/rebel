"""Run the reference depth-limited HUNL CFR solver from the command line."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cfvpy.nlhe import (
    CFRSolver,
    HeadsUpNoLimitHoldem,
    PublicBeliefState,
    card_to_string,
    combo_index,
    parse_card,
    zero_leaf_values,
)


def parse_hand(value: str):
    value = value.strip()
    if len(value) != 4:
        raise argparse.ArgumentTypeError("Use four characters, for example AsKd")
    try:
        return parse_card(value[:2]), parse_card(value[2:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the initial HUNL public state with depth-limited CFR"
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hand", type=parse_hand, default=parse_hand("AsKd"))
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    game = HeadsUpNoLimitHoldem()
    root = PublicBeliefState.uniform(game.initial_state())
    leaf_value_fn = zero_leaf_values
    if args.checkpoint is not None:
        from cfvpy.nlhe.inference import TorchLeafEvaluator

        leaf_value_fn = TorchLeafEvaluator(
            args.checkpoint, game, device=args.device
        )
    result = CFRSolver(
        game,
        root,
        max_depth=args.depth,
        leaf_value_fn=leaf_value_fn,
    ).solve(args.iterations)
    hand_index = combo_index(*args.hand)
    hand_name = "".join(card_to_string(card) for card in args.hand)

    print(f"public nodes: {len(result.tree)}")
    print(f"hand: {hand_name} (combo {hand_index})")
    if args.checkpoint is None:
        print("warning: zero-valued leaves; this is a solver smoke test, not a bot")
    else:
        value = result.root_values[root.state.player][hand_index]
        print(f"estimated value: {value / game.config.big_blind:.3f} bb")
    for action in result.tree[0].actions:
        probability = result.root_strategy[hand_index, action.slot]
        print(f"{str(action):>20}: {probability:7.3%}")


if __name__ == "__main__":
    main()
