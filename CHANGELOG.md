# Changelog

## v1.2.0

### Fixed
- Evaluation now pairs agents by identical episode seeds: every agent plays
  the same explicit seed list for the ELO and head-to-head tables, instead of
  MCTS running on seed0=7000 while others used 5000 with truncated arrays
  mislabeled as paired.
- Larger independent per-agent evaluations are retained but reported
  separately (`scores_independent`), never mixed into the paired stats.
- `scripts/evaluate.py` no longer imports torch at module load; the DQN
  policy imports it lazily, so the paired-seed logic is testable without
  torch.

### Tests
- 2 new regression tests in `tests/test_evaluate.py`; 23/23 runnable tests
  green.

## v1.1.0

### Fixed
- A completely filled board is now an explicit win: eating the last food ends
  the episode with reason "win" and a well-defined terminal observation,
  instead of raising TypeError when the state encoder unpacked the missing
  food position.
- Stepping after a win raises a clear error (call reset() first).

### Tests
- 2 new regression tests in `tests/test_snake.py`; 21/21 runnable tests green
  (DQN tests require torch, unavailable in this environment).
