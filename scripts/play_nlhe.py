"""Play heads-up no-limit hold'em against a trained policy checkpoint."""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cfvpy.nlhe import (
    CFRSolver,
    HeadsUpNoLimitHoldem,
    PublicBeliefState,
    Street,
    bayes_update,
    card_to_string,
    combo_index,
    normalize_range,
)
from cfvpy.nlhe.cards import NUM_CARDS
from cfvpy.nlhe.inference import TorchLeafEvaluator
from cfvpy.nlhe.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--human-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--search-iterations", type=int, default=128)
    parser.add_argument("--search-depth", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = (
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    game = HeadsUpNoLimitHoldem()
    model = load_model(args.checkpoint, device)
    leaf_evaluator = TorchLeafEvaluator(
        args.checkpoint, game, device=device, compile_model=False
    )
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    hole_cards = ((deck[0], deck[1]), (deck[2], deck[3]))
    runout = tuple(deck[4:9])
    state = game.initial_state()
    pbs = PublicBeliefState.uniform(state)

    human = args.human_seat
    bot = 1 - human
    print(f"Vous etes siege {human}; bot siege {bot}.")
    print(f"Votre main: {_cards_to_string(hole_cards[human])}")
    print("Les montants sont en jetons; 100 jetons = 1 grosse blinde.")

    while not state.terminal:
        if state.round_complete:
            state = _advance_completed_round(game, state, runout)
            pbs = PublicBeliefState(
                state,
                (
                    normalize_range(pbs.ranges[0], state.board),
                    normalize_range(pbs.ranges[1], state.board),
                ),
            )
            if state.terminal:
                break
            print(f"\n{state.street.name}: {_cards_to_string(state.board)}")
            print(f"Pot: {state.pot}")
            continue

        legal = game.legal_actions(state)
        policy_prior = _policy_matrix(model, game, pbs, device)
        search = CFRSolver(
            game,
            pbs,
            max_depth=args.search_depth,
            leaf_value_fn=leaf_evaluator,
            root_strategy_prior=policy_prior,
        ).solve(args.search_iterations)
        if state.player == human:
            action = _ask_human_action(legal)
        else:
            action = _sample_search_action(
                search.root_strategy,
                legal,
                combo_index(*hole_cards[bot]),
                rng,
            )
            print(f"Bot: {action}")

        actor = state.player
        action_likelihood = search.root_strategy[:, action.slot]
        updated_ranges = list(pbs.ranges)
        updated_ranges[actor] = bayes_update(
            updated_ranges[actor], action_likelihood, state.board
        )
        state = game.act(state, action)
        pbs = PublicBeliefState(state, tuple(updated_ranges))
        print(f"Pot: {state.pot}")

    print("\nMain terminee.")
    print(f"Board: {_cards_to_string(state.board)}")
    print(f"Votre main: {_cards_to_string(hole_cards[human])}")
    if state.terminal_type.value == "showdown":
        print(f"Main bot: {_cards_to_string(hole_cards[bot])}")
    payoff = game.terminal_payoff(state, human, hole_cards)
    print(f"Resultat: {payoff / game.config.big_blind:+.2f} bb")


def _policy_matrix(model, game, pbs, device):
    packed = torch.tensor(
        pbs.policy_vector(pbs.state.player, game.config.stack_size),
        dtype=torch.float32,
        device=device,
    )[None, :]
    legal_mask = torch.zeros(
        game.config.max_actions, dtype=torch.bool, device=device
    )
    for action in game.legal_actions(pbs.state):
        legal_mask[action.slot] = True
    with torch.inference_mode(), torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        logits = model(packed, head="policy")[0].float()
        logits = logits.masked_fill(~legal_mask[None, :], -1e9)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _sample_search_action(policy_matrix, legal, combo, rng):
    policy = policy_matrix[combo]
    probabilities = np.asarray(
        [max(0.0, float(policy[action.slot])) for action in legal]
    )
    probabilities /= probabilities.sum()
    choice = rng.choices(range(len(legal)), weights=probabilities, k=1)[0]
    return legal[choice]


def _ask_human_action(legal):
    print("\nActions:")
    for index, action in enumerate(legal):
        print(f"  {index}: {action}")
    while True:
        try:
            choice = int(input("Votre action: "))
            return legal[choice]
        except (ValueError, IndexError):
            print("Choisissez un numero valide.")


def _advance_completed_round(game, state, runout):
    if state.all_in:
        while not state.terminal:
            next_street = Street(state.street + 1)
            state = game.advance_street(
                state, runout[: _board_size(next_street)]
            )
        return state
    next_street = Street(state.street + 1)
    return game.advance_street(state, runout[: _board_size(next_street)])


def _board_size(street):
    return {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]


def _cards_to_string(cards):
    return " ".join(card_to_string(card) for card in cards) or "-"


if __name__ == "__main__":
    main()
