# Heads-up no-limit hold'em adaptation

The upstream ReBeL release contains only the Liar's Dice domain. The poker
agent described in the paper was not open-sourced, so this repository cannot
be converted correctly by renaming the existing `Game` class.

This first implementation milestone adds the domain pieces required by a
ReBeL-style HUNL agent:

- all 1,326 private two-card combinations and public-card blockers;
- a complete heads-up no-limit betting state with blinds, min-raises,
  short all-ins, folds, calls, and street transitions;
- a stable action abstraction of at most nine slots: fold, check/call,
  min-raise, five pot-fraction raises, and all-in;
- exact seven-card showdown evaluation;
- public belief states containing both players' normalized ranges;
- value and policy network definitions matching the paper's six-layer,
  1,536-unit MLP design;
- a NumPy depth-limited CFR solver with per-hand regrets, reach propagation,
  blocker-aware fold/showdown values, and a batched value callback at
  pseudo-leaves;
- a structured shared value/policy network with card, board, range, and combo
  encoders;
- BF16 H100 training, quantized policy targets, checkpoint resume, shard
  evaluation, and model-backed solving.

## Why search is street-by-street

Brown et al. report that their poker agent always searches to the end of the
current betting round. At the next street, chance reveals public cards and a
new public belief state starts another depth-limited subgame. The
`HeadsUpNoLimitHoldem.advance_street` API follows that design.

The paper states that its action abstraction used at most nine legal actions
and hand-picked typical bet sizes, but it does not publish every exact size.
This implementation uses min-raise plus 0.5, 0.75, 1.0, 1.5, and 2.0 times the
pot, followed by all-in. These fractions are configurable in `BettingConfig`.

All-in runouts are also kept as chance transitions rather than replaced by a
precomputed equity table. This mirrors the paper's choice to learn all-in
values.

## What remains

The original C++ CFR implementation is tightly coupled to Liar's Dice. The new
Python solver establishes the correct poker semantics and batches neural leaf
evaluation on GPU, but public-tree traversal and regret updates remain NumPy
CPU work. A paper-scale implementation still needs to:

1. port the slot-based solver and blocker matrices to C++ or CUDA;
2. move chance transitions and range filtering into the accelerated pipeline;
3. overlap self-play generation, replay storage, and GPU training;
4. add exploitability/local-best-response evaluation for the abstract game.

The current code is therefore a tested HUNL domain and reference solver, not a
claim to reproduce the paper's superhuman trained checkpoint. The NumPy solver
is intended for correctness tests and small experiments; paper-scale self-play
requires the planned C++/GPU data-generation path.

`CFRSolver` requires an explicit leaf value callback whenever the search ends
before a terminal. `zero_leaf_values` exists only for smoke tests. A useful
agent must connect `PokerReBeLNet` (or another trained evaluator) at those
pseudo-leaves.

For the complete Colab H100 workflow, see
[`training_colab.md`](training_colab.md).

## Primary references

- [ReBeL: Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games](https://arxiv.org/abs/2007.13544)
- [Safe and Nested Subgame Solving for Imperfect-Information
  Games](https://arxiv.org/abs/1705.02955)
- [Depth-Limited Solving for Imperfect-Information
  Games](https://arxiv.org/abs/1805.08195)
