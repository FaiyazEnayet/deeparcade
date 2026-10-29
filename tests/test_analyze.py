"""Regression tests for scripts/analyze.py's DQN policy loading.

analyze.py used to carry a private DQN loader that read runs/dqn.pt.
That file is gitignored and absent on a fresh checkout, so the script
died with FileNotFoundError before analyzing anything. The DQN policy
now comes from scripts.evaluate.DQNPolicy over the committed
runs/dqn_weights.json, through the strict shared checkpoint loader.
These tests pin that path: the policy analyze.py runs must build from
the committed weights and produce a normal move-level report.
"""

import os

import scripts.analyze as analyze_mod
from scripts.evaluate import DQNPolicy


def _pro_policy():
    # Built exactly the way scripts.analyze.main builds it.
    return DQNPolicy(os.path.join(analyze_mod.RUNS, "dqn_weights.json"))


def test_analyze_pro_policy_loads_from_committed_json_weights():
    policy = _pro_policy()
    assert policy.checkpoint_info is not None
    assert policy.net is not None


def test_analyze_pro_policy_produces_move_report():
    report = analyze_mod.analyze(_pro_policy().act, 2, seed0=31000)
    assert report["episodes"] == 2
    assert len(report["actions"]) == 3
    assert abs(sum(report["actions"]) - 1.0) < 0.01
    assert report["food_per_100_steps"] >= 0
    assert report["deaths"]
