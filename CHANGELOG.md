# Changelog

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
