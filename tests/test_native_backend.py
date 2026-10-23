"""CPP-04 tests: pybind11 bindings + optional native backend.

Skip-safety contract (load-bearing): pyproject.toml collects everything
under tests/ and CI runs the suite on machines WITHOUT the extension.
Every native-dependent test below is guarded by `requires_native`, which
skips unless `narcade_native` is importable (directly, or from a local
native/build* directory). Skipping is fast and unconditional-import-free:
nothing at module level touches the extension beyond one import attempt.
The one unguarded test covers the adapter's Python-by-default behavior,
which must hold with or without the extension.
"""

import glob
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_native():
    try:
        import narcade_native

        return narcade_native
    except ImportError:
        pass
    for build_dir in sorted(glob.glob(os.path.join(ROOT, "native", "build*"))):
        if os.path.isdir(build_dir) and build_dir not in sys.path:
            sys.path.insert(0, build_dir)
            try:
                import narcade_native

                return narcade_native
            except ImportError:
                sys.path.remove(build_dir)
    return None


native = _load_native()
requires_native = pytest.mark.skipif(
    native is None, reason="narcade_native extension not built"
)

from narcade import optional_backend  # noqa: E402
from narcade.mcts import MCTSPlayer as PythonMCTSPlayer  # noqa: E402
from narcade.policies import GreedyPolicy  # noqa: E402
from narcade.qlearn import QLearningAgent as PythonQLearningAgent  # noqa: E402
from narcade.snake import SnakeEnv as PythonSnakeEnv  # noqa: E402


def test_adapter_defaults_to_python(monkeypatch):
    """Without NARCADE_NATIVE the adapter returns the Python classes,
    whether or not the extension exists."""
    monkeypatch.delenv("NARCADE_NATIVE", raising=False)
    env = optional_backend.SnakeEnv(width=12, height=12, seed=1)
    assert isinstance(env, PythonSnakeEnv)
    player = optional_backend.MCTSPlayer(seed=1)
    assert isinstance(player, PythonMCTSPlayer)


@requires_native
def test_adapter_selects_native_only_with_flag(monkeypatch):
    monkeypatch.setenv("NARCADE_NATIVE", "1")
    env = optional_backend.SnakeEnv(width=12, height=12, seed=1)
    assert not isinstance(env, PythonSnakeEnv)
    assert env.reset() == PythonSnakeEnv(12, 12, seed=1).reset()
    player = optional_backend.MCTSPlayer(simulations=5, seed=1)
    assert not isinstance(player, PythonMCTSPlayer)


@requires_native
def test_python_random_matches_cpython():
    for seed in (0, 7, 999, 31337):
        ref = random.Random(seed)
        nat = native.PythonRandom(seed)
        for _ in range(500):
            assert nat.random() == ref.random()
        for k in (0, 1, 2, 8, 16, 31, 32, 33, 53, 64):
            for _ in range(20):
                assert nat.getrandbits(k) == ref.getrandbits(k)
        for _ in range(200):
            assert nat.randint(-5, 5) == ref.randint(-5, 5)
        for _ in range(200):
            assert nat.randrange(1000) == ref.randrange(1000)
        seq = list(range(7))
        for _ in range(200):
            assert nat.choice_index(7) == ref.choice(seq)


@requires_native
def test_qlearning_agent_matches_python():
    ref = PythonQLearningAgent(epsilon=0.25, seed=2024)
    nat = native.QLearningAgent(32, 3, 0.15, 0.9, 0.25, 2024)
    driver = random.Random(99)
    rewards = (10.0, -10.0, 0.5, -0.5, 0.0)
    for _ in range(1000):
        s = driver.randrange(32)
        assert nat.act(s) == ref.act(s)
        a = driver.randrange(3)
        r = rewards[driver.randrange(len(rewards))]
        s_next = driver.randrange(32)
        done = driver.randrange(16) == 0
        nat.update(s, a, r, s_next, done)
        ref.update(s, a, r, s_next, done)
    assert nat.q == ref.q


def _run_episode_trio(envs):
    """Step a Python env, the raw native env, and the adapter env (native
    on) through one identical GreedyPolicy action stream after a reset;
    compare every (state, reward, done, score). The envs are reused across
    calls so consecutive episodes continue each RNG stream, exactly like
    the Python evaluation loop. Returns the number of steps compared."""
    py_env, nat_env, ad_env = envs
    greedy = GreedyPolicy()
    assert nat_env.reset() == py_env.reset() == ad_env.reset()
    steps = 0
    while not py_env.done:
        action = greedy.act(py_env)
        # The native policy free function agrees on the same snapshot.
        assert native.greedy_policy_act(nat_env) == action
        p_state, p_reward, p_done, _ = py_env.step(action)
        n_state, n_reward, n_done, _reason = nat_env.step(action)
        a_state, a_reward, a_done, a_info = ad_env.step(action)
        assert (n_state, n_reward, n_done) == (p_state, p_reward, p_done)
        assert (a_state, a_reward, a_done) == (p_state, p_reward, p_done)
        assert nat_env.score == ad_env.score == py_env.score
        assert a_info == ({"reason": _reason} if _reason is not None else {})
        steps += 1
    return steps


@requires_native
def test_native_episodes_match_python(monkeypatch):
    monkeypatch.setenv("NARCADE_NATIVE", "1")
    episodes = 0
    total_steps = 0
    for seed in (11, 222, 3333, 424242):
        envs = (
            PythonSnakeEnv(12, 12, seed=seed, max_steps=400),
            native.SnakeEnv(12, 12, seed, 400),
            optional_backend.SnakeEnv(
                width=12, height=12, seed=seed, max_steps=400
            ),
        )
        for _ in range(6):
            total_steps += _run_episode_trio(envs)
            episodes += 1
    assert episodes >= 20
    assert total_steps > episodes  # the comparison actually stepped


@requires_native
def test_native_mcts_actions_match_python(monkeypatch):
    monkeypatch.setenv("NARCADE_NATIVE", "1")
    params = dict(simulations=50, rollout_steps=100, c=1.0, seed=31337)
    py_player = PythonMCTSPlayer(**params)
    nat_player = native.MCTSPlayer(
        params["simulations"], params["rollout_steps"], params["c"],
        params["seed"],
    )
    ad_player = optional_backend.MCTSPlayer(**params)
    greedy = GreedyPolicy()
    compared = 0
    for seed in (5000, 5001, 5002):
        env = PythonSnakeEnv(12, 12, seed=seed)
        while not env.done and compared < 60:
            expected = py_player.search(env)
            assert nat_player.search(env) == expected
            assert ad_player.search(env) == expected
            compared += 1
            env.step(greedy.act(env))
    assert compared >= 50
