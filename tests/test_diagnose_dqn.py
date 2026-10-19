"""DA4 regression tests: DQN diagnosis instrumentation.

Fast, tiny-scale checks that the diagnosis variants run, log the expected
diagnostics, and behave differently from each other where they should.
"""

import numpy as np
import torch

from scripts.diagnose_dqn import (
    evaluate_policy,
    sparse_reward,
    train_imitate,
    train_rl,
)
from scripts.train_dqn import DQN
from narcade.snake import SnakeEnv


def test_sparse_reward_strips_shaping():
    env = SnakeEnv(seed=3)
    env.reset()
    # Ordinary step reward (+/-0.5 shaping or 0) becomes 0.
    r = sparse_reward(env, 0.5, False, {}, env.score)
    assert r == 0.0
    r = sparse_reward(env, -0.5, False, {}, env.score)
    assert r == 0.0
    # Death keeps -10.
    r = sparse_reward(env, -10.0, True, {"reason": "death"}, env.score)
    assert r == -10.0
    # Timeout (no food, no death) becomes 0.
    r = sparse_reward(env, 0.5, True, {"reason": "timeout"}, env.score)
    assert r == 0.0


def test_rl_variants_run_tiny_and_log():
    for variant in ("dqn", "ddqn", "noshape"):
        summary, net = train_rl(variant, steps=2600, seed=11, log_every_eps=50)
        assert summary["variant"] == variant
        assert summary["episodes"] > 0
        assert summary["updates"] > 0  # replay warmed past 2000
        assert summary["curve_score"], "score curve must be logged"
        assert summary["curve_loss"], "loss curve must be logged"
        assert summary["curve_target"], "target magnitudes must be logged"
        assert summary["curve_maxq"], "max |Q| must be logged"
        assert 0.0 <= summary["death_frac"] <= 1.0
        assert isinstance(net, DQN)


def test_noshape_differs_from_shaped():
    # With shaping stripped, stored rewards are only ever +10 / -10 / 0,
    # so Bellman targets stay on a coarser grid than the shaped variant.
    s_shaped, _ = train_rl("dqn", steps=2600, seed=11, log_every_eps=50)
    s_sparse, _ = train_rl("noshape", steps=2600, seed=11, log_every_eps=50)
    assert s_sparse["curve_target"] != s_shaped["curve_target"]


def test_imitate_learns_greedy_actions_tiny():
    summary, net = train_imitate(steps=1200, seed=5, collect_n=3000, eval_eps=5)
    assert summary["variant"] == "imitate"
    assert summary["updates"] == 150
    # Training loss must decrease substantially over the run (machinery works;
    # full convergence is measured in the diagnosis run, not here).
    losses = summary["imitate_train_loss"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0] - 0.2, f"loss {losses[0]} -> {losses[-1]}"
    # Eval runs and records scores.
    assert len(summary["imitate_eval_scores"]) == 5
    assert all(s >= 0 for s in summary["imitate_eval_scores"])


def test_evaluate_policy_scores():
    net = DQN()
    scores = evaluate_policy(net, seeds=[1, 2, 3])
    assert len(scores) == 3
    assert all(s >= 0 for s in scores)
