"""Optional native backend adapter (CPP-04).

This module is the ONLY supported bridge between the Python package and
the C++ core exposed by the `narcade_native` pybind11 extension. Nothing
in the package imports it; callers opt in explicitly:

    from narcade import optional_backend
    env = optional_backend.SnakeEnv(width=12, height=12, seed=7)

Backend selection is per call and driven by one environment variable:

- ``NARCADE_NATIVE`` unset or not "1": the factories below return the
  pure-Python classes from :mod:`narcade.snake` / :mod:`narcade.mcts`.
  Python is the default and the reference implementation.
- ``NARCADE_NATIVE=1``: the factories return thin wrappers over
  `narcade_native`, with the same constructor signatures and the same
  method surface the training/evaluation code uses (reset/step returning
  (state, reward, done, info), body/food/heading/score/steps/done,
  encode_state, search/act). If the extension is not importable in this
  mode, ImportError is raised rather than silently falling back: asking
  for the native backend and not getting it is a configuration error.

Two deliberate deviations, both forced by the native API:

- The C++ classes require an explicit integer seed (CPython's seed=None
  OS-entropy seeding has no native counterpart). Passing seed=None with
  the native backend on draws a 63-bit seed from system entropy here, so
  the call still means "unseeded", just not reproducibly.
- The wrapper's step() rebuilds Python's info dict ({"reason": ...} or
  {}) from the native step result's reason string.
"""

from __future__ import annotations

import os
import random as _random

from .mcts import MCTSPlayer as _PythonMCTSPlayer
from .snake import SnakeEnv as _PythonSnakeEnv

__all__ = ["SnakeEnv", "MCTSPlayer", "native_enabled", "native_available"]

_ENTROPY = _random.Random()  # system-entropy seeded; only for seed=None


def native_enabled():
    """True when the caller has requested the native backend."""
    return os.environ.get("NARCADE_NATIVE") == "1"


def _load_native():
    try:
        import narcade_native
    except ImportError:
        return None
    return narcade_native


def native_available():
    """True when the narcade_native extension is importable here."""
    return _load_native() is not None


def _require_native():
    module = _load_native()
    if module is None:
        raise ImportError(
            "NARCADE_NATIVE=1 but the narcade_native extension is not "
            "importable; build it from native/ (target narcade_native) "
            "and put the build directory on sys.path"
        )
    return module


def _native_seed(seed):
    return _ENTROPY.randrange(2**63) if seed is None else seed


class _NativeSnakeEnv:
    """SnakeEnv-shaped wrapper over narcade_native.SnakeEnv."""

    FOOD_REWARD = 10.0
    DEATH_REWARD = -10.0
    CLOSER_REWARD = 0.5
    FARTHER_REWARD = -0.5

    def __init__(self, width=12, height=12, seed=None, max_steps=None):
        native = _require_native()
        seed = _native_seed(seed)
        if max_steps is None:
            self._env = native.SnakeEnv(width, height, seed)
        else:
            self._env = native.SnakeEnv(width, height, seed, max_steps)

    # The wrapped native env, for callers that want the raw module API.
    @property
    def native_env(self):
        return self._env

    def reset(self):
        return self._env.reset()

    def step(self, action):
        state, reward, done, reason = self._env.step(action)
        info = {"reason": reason} if reason is not None else {}
        return state, reward, done, info

    def encode_state(self):
        return self._env.encode_state()

    @property
    def width(self):
        return self._env.width

    @property
    def height(self):
        return self._env.height

    @property
    def body(self):
        # The Python body is a deque; consumers here only read/iterate it.
        from collections import deque

        return deque(self._env.body)

    @property
    def food(self):
        return self._env.food

    @property
    def heading(self):
        return self._env.heading

    @property
    def score(self):
        return self._env.score

    @property
    def steps(self):
        return self._env.steps

    @property
    def done(self):
        return self._env.done

    @property
    def max_steps(self):
        return self._env.max_steps

    @property
    def occupied(self):
        return set(self._env.body)


class _NativeMCTSPlayer:
    """MCTSPlayer-shaped wrapper over narcade_native.MCTSPlayer."""

    def __init__(self, simulations=300, rollout_steps=100, c=1.0, seed=None):
        native = _require_native()
        self.simulations = simulations
        self.rollout_steps = rollout_steps
        self.c = c
        self._player = native.MCTSPlayer(
            simulations, rollout_steps, c, _native_seed(seed)
        )

    @property
    def native_player(self):
        return self._player

    def search(self, env, walls=frozenset()):
        # The binding snapshots whatever env it is given (Python env,
        # native env, or _NativeSnakeEnv) via its public attributes;
        # unwrap our own wrapper to the raw native env for clarity.
        target = env.native_env if isinstance(env, _NativeSnakeEnv) else env
        return self._player.search(target, walls)

    def act(self, env, walls=frozenset()):
        return self.search(env, walls=walls)


def SnakeEnv(width=12, height=12, seed=None, max_steps=None):
    """Factory with narcade.snake.SnakeEnv's signature.

    Returns the Python env by default, the native wrapper when
    NARCADE_NATIVE=1.
    """
    if native_enabled():
        return _NativeSnakeEnv(width, height, seed, max_steps)
    return _PythonSnakeEnv(width, height, seed, max_steps)


def MCTSPlayer(simulations=300, rollout_steps=100, c=1.0, seed=None):
    """Factory with narcade.mcts.MCTSPlayer's signature.

    Returns the Python player by default, the native wrapper when
    NARCADE_NATIVE=1.
    """
    if native_enabled():
        return _NativeMCTSPlayer(simulations, rollout_steps, c, seed)
    return _PythonMCTSPlayer(simulations, rollout_steps, c, seed)
