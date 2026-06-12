import math
import unittest

import numpy as np

from cfvpy.nlhe import (
    ActionType,
    BettingConfig,
    CFRSolver,
    HeadsUpNoLimitHoldem,
    NUM_HOLE_COMBOS,
    PublicBeliefState,
    Street,
    card_to_string,
    combo_cards,
    combo_index,
    compare_hands,
    parse_card,
    valid_combo_mask,
    zero_leaf_values,
)
from cfvpy.nlhe.pbs import POLICY_INPUT_SIZE, VALUE_INPUT_SIZE


def cards(values):
    return tuple(parse_card(value) for value in values.split())


class CardTest(unittest.TestCase):
    def test_card_and_combo_round_trip(self):
        ace_spades = parse_card("As")
        king_hearts = parse_card("Kh")
        self.assertEqual(card_to_string(ace_spades), "As")
        index = combo_index(ace_spades, king_hearts)
        self.assertEqual(set(combo_cards(index)), {ace_spades, king_hearts})
        self.assertEqual(NUM_HOLE_COMBOS, 1326)

    def test_board_blockers(self):
        mask = valid_combo_mask(cards("As Kd 2c"))
        self.assertEqual(sum(mask), math.comb(49, 2))

    def test_showdown_comparison(self):
        board = cards("Ah Kh Qh 2c 2d")
        straight_flush = cards("Jh Th")
        full_house = cards("Ac Ad")
        self.assertEqual(compare_hands(straight_flush, full_house, board), 1)

    def test_wheel_straight(self):
        board = cards("2c 3d 4h 9s Kc")
        wheel = cards("As 5d")
        pair = cards("Kh Qd")
        self.assertEqual(compare_hands(wheel, pair, board), 1)


class BettingTest(unittest.TestCase):
    def setUp(self):
        self.game = HeadsUpNoLimitHoldem()

    def action(self, state, kind, amount=None):
        matches = [
            action
            for action in self.game.legal_actions(state)
            if action.kind == kind and (amount is None or action.amount == amount)
        ]
        self.assertEqual(len(matches), 1, (state, self.game.legal_actions(state)))
        return matches[0]

    def test_preflop_call_and_check_complete_round(self):
        state = self.game.initial_state()
        self.assertEqual(state.to_call, 50)
        self.assertLessEqual(len(self.game.legal_actions(state)), 9)

        state = self.game.act(
            state, self.action(state, ActionType.CHECK_CALL, amount=100)
        )
        self.assertFalse(state.round_complete)
        self.assertEqual(state.player, 1)
        self.assertEqual(state.to_call, 0)

        state = self.game.act(
            state, self.action(state, ActionType.CHECK_CALL, amount=100)
        )
        self.assertTrue(state.round_complete)
        self.assertEqual(state.pot, 200)

    def test_raise_call_and_flop_checks(self):
        state = self.game.initial_state()
        state = self.game.act(
            state, self.action(state, ActionType.BET_RAISE, amount=200)
        )
        state = self.game.act(
            state, self.action(state, ActionType.CHECK_CALL, amount=200)
        )
        self.assertTrue(state.round_complete)
        self.assertEqual(state.total_commit, (200, 200))

        state = self.game.advance_street(state, cards("As Kd 2c"))
        self.assertEqual(state.street, Street.FLOP)
        self.assertEqual(state.player, 1)
        self.assertFalse(state.round_complete)

        state = self.game.act(
            state, self.action(state, ActionType.CHECK_CALL, amount=0)
        )
        state = self.game.act(
            state, self.action(state, ActionType.CHECK_CALL, amount=0)
        )
        self.assertTrue(state.round_complete)

    def test_fold_payoff_is_zero_sum(self):
        state = self.game.initial_state()
        state = self.game.act(state, self.action(state, ActionType.FOLD))
        self.assertEqual(self.game.terminal_payoff(state, 0), -50.0)
        self.assertEqual(self.game.terminal_payoff(state, 1), 50.0)

    def test_all_in_runout_reaches_showdown(self):
        game = HeadsUpNoLimitHoldem(
            BettingConfig(stack_size=400, small_blind=1, big_blind=2)
        )
        state = game.initial_state()
        all_in = next(
            action
            for action in game.legal_actions(state)
            if action.kind == ActionType.ALL_IN
        )
        state = game.act(state, all_in)
        call = next(
            action
            for action in game.legal_actions(state)
            if action.kind == ActionType.CHECK_CALL
        )
        state = game.act(state, call)
        self.assertTrue(state.all_in)
        state = game.advance_street(state, cards("2c 3d 4h"))
        state = game.advance_street(state, cards("2c 3d 4h 5s"))
        state = game.advance_street(state, cards("2c 3d 4h 5s 9c"))
        self.assertTrue(state.terminal)
        payoff = game.terminal_payoff(
            state, 0, (cards("6s Kd"), cards("Ah Qd"))
        )
        self.assertEqual(payoff, 400.0)


class PublicBeliefStateTest(unittest.TestCase):
    def test_uniform_ranges_and_network_vectors(self):
        game = HeadsUpNoLimitHoldem()
        state = game.initial_state()
        pbs = PublicBeliefState.uniform(state)
        self.assertAlmostEqual(sum(pbs.ranges[0]), 1.0)
        self.assertEqual(
            len(pbs.value_vector(0, game.config.stack_size)), VALUE_INPUT_SIZE
        )
        self.assertEqual(
            len(pbs.policy_vector(0, game.config.stack_size)), POLICY_INPUT_SIZE
        )

    def test_public_cards_zero_blocked_combos(self):
        game = HeadsUpNoLimitHoldem()
        state = game.initial_state()
        state = game.act(
            state,
            next(
                action
                for action in game.legal_actions(state)
                if action.kind == ActionType.CHECK_CALL
            ),
        )
        state = game.act(
            state,
            next(
                action
                for action in game.legal_actions(state)
                if action.kind == ActionType.CHECK_CALL
            ),
        )
        state = game.advance_street(state, cards("As Kd 2c"))
        pbs = PublicBeliefState.uniform(state)
        mask = valid_combo_mask(state.board)
        self.assertTrue(
            all(
                weight == 0.0
                for weight, valid in zip(pbs.ranges[0], mask)
                if not valid
            )
        )


class CFRSolverTest(unittest.TestCase):
    def test_depth_limited_solver_returns_legal_normalized_strategy(self):
        game = HeadsUpNoLimitHoldem()
        root = PublicBeliefState.uniform(game.initial_state())
        result = CFRSolver(
            game, root, max_depth=2, leaf_value_fn=zero_leaf_values
        ).solve(4)

        self.assertGreater(len(result.tree), 1)
        self.assertEqual(
            result.strategy.shape,
            (len(result.tree), NUM_HOLE_COMBOS, game.config.max_actions),
        )
        root_slots = [action.slot for action in result.tree[0].actions]
        root_policy = result.root_strategy[:, root_slots]
        self.assertTrue((root_policy >= 0).all())
        self.assertTrue(
            np.allclose(root_policy.sum(axis=1), 1.0, atol=1e-6)
        )
        illegal_slots = sorted(set(range(game.config.max_actions)) - set(root_slots))
        self.assertTrue((result.root_strategy[:, illegal_slots] == 0).all())

    def test_leaf_callback_receives_normalized_ranges(self):
        game = HeadsUpNoLimitHoldem()
        root = PublicBeliefState.uniform(game.initial_state())
        calls = []

        def leaf_value(state, traverser, ranges):
            calls.append((state, traverser, ranges.sum(axis=1)))
            return np.zeros(NUM_HOLE_COMBOS, dtype=np.float32)

        CFRSolver(
            game, root, max_depth=1, leaf_value_fn=leaf_value
        ).solve(2)
        self.assertTrue(calls)
        for _, _, sums in calls:
            self.assertTrue(np.allclose(sums, (1.0, 1.0), atol=1e-6))

    def test_showdown_counterfactual_values_are_zero_sum(self):
        game = HeadsUpNoLimitHoldem(
            BettingConfig(stack_size=40, small_blind=1, big_blind=2)
        )
        state = game.initial_state()
        state = game.act(
            state,
            next(
                action
                for action in game.legal_actions(state)
                if action.kind == ActionType.ALL_IN
            ),
        )
        state = game.act(
            state,
            next(
                action
                for action in game.legal_actions(state)
                if action.kind == ActionType.CHECK_CALL
            ),
        )
        state = game.advance_street(state, cards("2c 7d Jh"))
        state = game.advance_street(state, cards("2c 7d Jh Qs"))
        state = game.advance_street(state, cards("2c 7d Jh Qs Ac"))
        root = PublicBeliefState.uniform(state)
        result = CFRSolver(game, root, max_depth=0).solve(2)
        ev0 = float(
            np.dot(
                np.asarray(root.ranges[0]),
                result.root_counterfactual_values[0],
            )
        )
        ev1 = float(
            np.dot(
                np.asarray(root.ranges[1]),
                result.root_counterfactual_values[1],
            )
        )
        self.assertAlmostEqual(ev0 + ev1, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
