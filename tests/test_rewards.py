"""Stage 3 deeparcade R-03: reward-shaping science, not removal.

Acceptance (handoffs section 012):
- exact reward decomposition in logs (reward = base + shaping on every step);
- telescope test: discounted shaping sums telescope over a non-terminal
  stretch;
- terminal test: death/win shaping equals -Phi(s) with no sign leak;
- no sparse-reward regression: a sparse (no-shaping) arm runs at the same
  fixed budget as the dense and potential arms;
- fixed-budget multiseed evaluation with score, loop (timeout) and death
  trade-offs; if current shaping outperforms, keep it.

Design (see issue #85):
- The default stays the current dense baseline: closer +0.5 / farther -0.5
  on Manhattan distance to food, food +10, death -10.
- Potential shaping is F(s,a,s') = gamma*Phi(s') - Phi(s) with
  Phi(s) = -scale * Manhattan(head, food) / (width + height - 2), so Phi
  lives in [-scale, 0]. Defaults: gamma 0.95 (the DQN discount),
  scale 1.0.
- Food respawn: on eating, Phi(s') is evaluated against the newly placed
  food.
- Terminal states: Phi(terminal) = 0, so the shaping term on a death or
  win step is -Phi(s). It is bounded by the shaping scale, so a death step
  stays negative and a win step stays positive: no reward leak.
- Timeout is a truncation, not a terminal state: Phi is evaluated
  normally on the timeout step.
- With gamma 1.0 and scale (width + height - 2) / 2, the potential form
  reduces exactly to the dense baseline; that identity is pinned below.
- Ng et al. 1999 establishes conditions for policy invariance of
  correctly designed potential shaping in a Markov reward setting, not a
  guarantee of strong finite-sample Snake performance; the comparison in
  runs/shaping_study.json is empirical.
"""

import os
import random
import sys
from collections import deque

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from narcade.snake import SnakeEnv  # noqa: E402
from scripts.train_dqn import train_dqn  # noqa: E402

# Known-answer trace recorded on main (HEAD 733220c) before this change:
# SnakeEnv(seed=7), actions drawn from random.Random(4). One food eaten
# (step 16), death on step 37, final score 1.
TRACE_ACTIONS = [0, 1, 0, 2, 1, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 2, 2, 1, 1,
                 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 1, 1, 2, 2, 1, 0, 2, 1]
TRACE_REWARDS = [-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5, 10.0, 0.5, -0.5, -0.5, -0.5,
                 -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
                 -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -10.0]
TRACE_SCORES = [0] * 15 + [1] * 22


def _phi(env):
    """Independent recomputation of Phi(s) from public env state."""
    if env.food is None:
        return 0.0
    hr, hc = env.body[-1]
    d = abs(hr - env.food[0]) + abs(hc - env.food[1])
    return -env.shaping_scale * d / (env.width + env.height - 2)


def _play(env, actions):
    """Play actions; return (rewards, infos, dones)."""
    rewards, infos, dones = [], [], []
    for a in actions:
        _, r, done, info = env.step(a)
        rewards.append(r)
        infos.append(info)
        dones.append(done)
        if done:
            break
    return rewards, infos, dones


def test_stage3_deeparcade_03_default_dense_unchanged():
    """The default env reproduces the exact pre-change reward trace."""
    env = SnakeEnv(seed=7)
    rewards, infos, dones = _play(env, TRACE_ACTIONS)
    assert rewards == TRACE_REWARDS
    assert dones[-1] is True
    assert infos[-1]["reason"] == "death"
    assert env.score == 1
    for info, expected in zip(infos, TRACE_REWARDS):
        assert info["base_reward"] + info["shaping_reward"] == expected
        assert info["shaping"] == "dense"
    # Dense mode: the shaping term is the whole non-food/death reward.
    assert infos[0]["base_reward"] == 0.0
    assert infos[0]["shaping_reward"] == -0.5
    assert infos[15]["base_reward"] == 10.0
    assert infos[15]["shaping_reward"] == 0.0
    assert infos[-1]["base_reward"] == -10.0
    assert infos[-1]["shaping_reward"] == 0.0


def test_stage3_deeparcade_03_potential_exact_value():
    """One hand-computed potential step: F = gamma*Phi(s') - Phi(s)."""
    env = SnakeEnv(width=5, height=5, seed=1, shaping="potential",
                   shaping_gamma=0.95, shaping_scale=1.0)
    env.body = deque([(2, 0), (2, 1), (2, 2)])
    env.heading = 1  # right
    env.food = (2, 4)
    # Head (2,2), d=2, dmax=8: Phi = -2/8 = -0.25. After moving straight
    # to (2,3), d=1: Phi' = -1/8 = -0.125.
    # F = 0.95 * (-0.125) - (-0.25) = 0.13125; base reward 0.
    _, reward, done, info = env.step(0)
    assert done is False
    assert info["base_reward"] == 0.0
    assert info["shaping_reward"] == pytest.approx(0.13125)
    assert reward == pytest.approx(0.13125)


def test_stage3_deeparcade_03_potential_reduces_to_dense():
    """gamma 1.0, scale dmax/2: potential mode equals the dense baseline
    on every ordinary step. The eating step and the death step differ by
    design: the eating step carries Phi(new food) - Phi(old food), and
    the death step carries -Phi(s) because terminal Phi is 0, so both
    are pinned separately here."""
    env = SnakeEnv(seed=7, shaping="potential", shaping_gamma=1.0,
                   shaping_scale=11.0)
    rewards = []
    eat_shaping = None
    death_shaping = None
    for i, a in enumerate(TRACE_ACTIONS):
        phi_before = _phi(env)
        _, r, done, info = env.step(a)
        rewards.append(r)
        if i == 15:  # the eating step
            eat_shaping = info["shaping_reward"]
            assert eat_shaping == pytest.approx(_phi(env) - phi_before)
            assert r == pytest.approx(10.0 + eat_shaping)
        if i == 36:  # the death step
            death_shaping = info["shaping_reward"]
            assert death_shaping == pytest.approx(-phi_before)
            assert r == pytest.approx(-10.0 + death_shaping)
        if done:
            break
    assert rewards[:15] == TRACE_REWARDS[:15]
    assert rewards[16:36] == TRACE_REWARDS[16:36]
    assert eat_shaping is not None
    assert death_shaping is not None


def test_stage3_deeparcade_03_telescope():
    """Discounted shaping sums telescope over a non-terminal stretch."""
    gamma = 0.95
    env = SnakeEnv(seed=7, shaping="potential", shaping_gamma=gamma,
                   shaping_scale=1.0)
    phi0 = _phi(env)
    actions = TRACE_ACTIONS[:10]  # no eat, no death in this prefix
    rewards, infos, dones = _play(env, actions)
    assert not any(dones)
    assert all(info["base_reward"] == 0.0 for info in infos)
    discounted = sum((gamma ** t) * info["shaping_reward"]
                     for t, info in enumerate(infos))
    phi_t = _phi(env)
    assert discounted == pytest.approx((gamma ** len(actions)) * phi_t - phi0,
                                       abs=1e-9)


def test_stage3_deeparcade_03_terminal_death_no_leak():
    """Death step shaping is -Phi(s); the total stays negative."""
    env = SnakeEnv(width=5, height=5, seed=1, shaping="potential",
                   shaping_gamma=0.95, shaping_scale=1.0)
    env.body = deque([(2, 2), (2, 3), (2, 4)])
    env.heading = 1  # right, into the east wall
    env.food = (0, 0)
    # Head (2,4), d=6, dmax=8: Phi = -6/8 = -0.75. Terminal Phi is 0,
    # so F = 0 - (-0.75) = 0.75; total = -10 + 0.75 = -9.25.
    phi_before = _phi(env)
    assert phi_before == pytest.approx(-0.75)
    _, reward, done, info = env.step(0)
    assert done is True
    assert info["reason"] == "death"
    assert info["base_reward"] == -10.0
    assert info["shaping_reward"] == pytest.approx(-phi_before)
    assert reward == pytest.approx(-9.25)
    assert reward < 0.0


def test_stage3_deeparcade_03_terminal_win_no_leak():
    """Win step shaping is -Phi(s); the total stays positive."""
    env = SnakeEnv(width=3, height=3, seed=1, shaping="potential",
                   shaping_gamma=0.95, shaping_scale=1.0)
    env.body = deque([(0, 0), (0, 1), (0, 2), (1, 2),
                      (1, 1), (1, 0), (2, 0), (2, 1)])
    env.heading = 1  # right, onto the last free cell
    env.food = (2, 2)
    # Head (2,1), d=1, dmax=4: Phi = -1/4 = -0.25. Terminal Phi is 0,
    # so F = 0.25; total = 10 + 0.25 = 10.25.
    phi_before = _phi(env)
    assert phi_before == pytest.approx(-0.25)
    _, reward, done, info = env.step(0)
    assert done is True
    assert info["reason"] == "win"
    assert env.food is None
    assert env.potential() == 0.0
    assert info["base_reward"] == 10.0
    assert info["shaping_reward"] == pytest.approx(-phi_before)
    assert reward == pytest.approx(10.25)
    assert reward > 0.0


def test_stage3_deeparcade_03_respawn_uses_new_food():
    """The eating-step shaping term uses the newly placed food."""
    env = SnakeEnv(width=5, height=5, seed=1, shaping="potential",
                   shaping_gamma=0.95, shaping_scale=1.0)
    env.body = deque([(2, 0), (2, 1), (2, 2)])
    env.heading = 1  # right
    env.food = (2, 3)
    phi_before = _phi(env)
    assert phi_before == pytest.approx(-1 / 8)
    _, reward, done, info = env.step(0)
    assert done is False
    assert env.score == 1
    assert env.food != (2, 3)  # food respawned
    phi_after = _phi(env)
    assert info["base_reward"] == 10.0
    assert info["shaping_reward"] == pytest.approx(
        0.95 * phi_after - phi_before)
    assert reward == pytest.approx(10.0 + 0.95 * phi_after - phi_before)
    # Shaping stays bounded by the scale on the eating step too.
    assert abs(info["shaping_reward"]) <= 1.0


def test_stage3_deeparcade_03_sparse_mode():
    """Sparse mode pays only food and death; nothing for moving."""
    env = SnakeEnv(seed=7, shaping="none")
    rewards, infos, dones = _play(env, TRACE_ACTIONS)
    expected = [0.0 if r in (-0.5, 0.5) else r for r in TRACE_REWARDS]
    assert rewards == expected
    assert dones[-1] is True
    for r, info in zip(rewards, infos):
        assert info["shaping_reward"] == 0.0
        assert info["base_reward"] == r
        assert info["shaping"] == "none"


def test_stage3_deeparcade_03_decomposition_exact_potential():
    """reward == base + shaping exactly, on every step, in potential mode."""
    env = SnakeEnv(seed=7, shaping="potential")
    rewards, infos, dones = _play(env, TRACE_ACTIONS)
    assert dones[-1] is True
    for r, info in zip(rewards, infos):
        assert r == info["base_reward"] + info["shaping_reward"]
        assert info["shaping"] == "potential"
        assert abs(info["shaping_reward"]) <= 1.0


def test_stage3_deeparcade_03_invalid_shaping_rejected():
    with pytest.raises(ValueError):
        SnakeEnv(seed=7, shaping="bogus")


def test_stage3_deeparcade_03_train_dqn_wiring():
    """train_dqn accepts a shaping mode and reports the decomposition."""
    _, _, _, meta = train_dqn(steps=2100, seed=7, verbose=False,
                              shaping="potential")
    assert meta["shaping"] == "potential"
    assert meta["shaping_gamma"] == 0.95
    assert meta["shaping_scale"] == 1.0
    assert "final_mean100_base_reward" in meta
    assert "final_mean100_shaping_reward" in meta
    assert "outcomes" in meta
    assert sum(meta["outcomes"].values()) == meta["episodes"]
    _, _, _, meta_dense = train_dqn(steps=2100, seed=7, verbose=False)
    assert meta_dense["shaping"] == "dense"
    assert meta_dense["episodes"] > 0
    assert sum(meta_dense["outcomes"].values()) == meta_dense["episodes"]
