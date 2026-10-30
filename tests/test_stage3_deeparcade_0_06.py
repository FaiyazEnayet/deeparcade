"""Stage 3 deeparcade P0-06: hard split registry / RNG lifecycle.

Acceptance (handoffs section 007):
- no intersections of RNG keys across named disjoint namespaces;
- same (episode key, policy id, model id) reproduces the same outcome
  regardless of evaluation order;
- explicit test demonstrating old seeds overlap;
- differential behavior of resets documented;
- independent and paired samples traceable.

Old results stay labeled "historical, training-seed overlap / MCTS
mismatch" in runs/splits_manifest.json; no published number is rewritten
here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from narcade.experiment_seeds import (  # noqa: E402
    EpisodeKey,
    build_splits_manifest,
    check_split_use,
    derive_key,
    episode_keys,
    legacy_eval_seed_range,
    legacy_training_seed_range,
    load_splits_manifest,
)
from narcade.mcts import MCTSPlayer  # noqa: E402
from narcade.policies import RandomPolicy  # noqa: E402
from narcade.snake import SnakeEnv  # noqa: E402
from scripts.evaluate import run_episode_rows, run_keyed_episodes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stage3_deeparcade_0_06_namespaces_disjoint():
    """Env keys and policy keys never intersect across namespaces."""
    seen_env = {}
    seen_policy = {}
    for ns in ("train", "validation", "test"):
        for key in episode_keys(ns, 0, 200):
            assert key.env_key not in seen_env, (
                f"env key collision: {ns} index {key.index} "
                f"reuses {seen_env[key.env_key]}"
            )
            assert key.policy_key not in seen_policy, (
                f"policy key collision: {ns} index {key.index} "
                f"reuses {seen_policy[key.policy_key]}"
            )
            assert key.env_key != key.policy_key
            seen_env[key.env_key] = (ns, key.index)
            seen_policy[key.policy_key] = (ns, key.index)
    # Env and policy streams are separate domains, not just separate ranges.
    assert not (set(seen_env) & set(seen_policy))


def test_stage3_deeparcade_0_06_derive_key_deterministic():
    assert derive_key("train", 3, "env") == derive_key("train", 3, "env")
    assert derive_key("train", 3, "env") != derive_key("train", 4, "env")
    assert derive_key("train", 3, "env") != derive_key("train", 3, "policy")
    assert derive_key("train", 3, "env") != derive_key("validation", 3, "env")


def test_stage3_deeparcade_0_06_same_key_reproduces_regardless_of_order():
    """Keyed RandomPolicy episodes are identical run forward or reversed."""
    keys = episode_keys("validation", 0, 6)

    def make_policy(policy_key):
        return RandomPolicy(seed=policy_key)

    forward = run_keyed_episodes(make_policy, keys)
    backward = run_keyed_episodes(make_policy, list(reversed(keys)))
    by_key_forward = {row["episode_key"]: row["score"] for row in forward}
    by_key_backward = {row["episode_key"]: row["score"] for row in backward}
    assert by_key_forward == by_key_backward
    assert len(by_key_forward) == 6


def test_stage3_deeparcade_0_06_keyed_mcts_order_independent():
    """Keyed MCTS episodes are identical run forward or reversed."""
    keys = episode_keys("validation", 0, 2)

    def make_policy(policy_key):
        return MCTSPlayer(simulations=5, seed=policy_key)

    forward = run_keyed_episodes(make_policy, keys, width=8, height=8)
    backward = run_keyed_episodes(make_policy, list(reversed(keys)),
                                  width=8, height=8)
    by_key_forward = {row["episode_key"]: row["score"] for row in forward}
    by_key_backward = {row["episode_key"]: row["score"] for row in backward}
    assert by_key_forward == by_key_backward


def test_stage3_deeparcade_0_06_old_shared_policy_order_dependent():
    """Old defect, pinned: a shared policy's stream makes an episode's
    outcome depend on what ran before it. RandomPolicy(seed=1): episode
    6000 scores 1 when played first, 0 when played after episode 5001."""
    first = RandomPolicy(seed=1)
    played_first = run_episode_rows(first.act, [6000])[0]["score"]
    second = RandomPolicy(seed=1)
    played_second = run_episode_rows(second.act, [5001, 6000])[1]["score"]
    assert played_first == 1
    assert played_second == 0


def test_stage3_deeparcade_0_06_old_seeds_overlap():
    """Legacy evaluation seeds sit inside the legacy training seed range:
    train.py defaults (seed 7, 30000 episodes) use env seeds 8..30007,
    while evaluate.py uses independent seeds 5000..5499. All 500 legacy
    evaluation seeds collide with training episode seeds."""
    train_lo, train_hi = legacy_training_seed_range(seed=7, episodes=30000)
    eval_lo, eval_hi = legacy_eval_seed_range(n=500)
    assert (train_lo, train_hi) == (8, 30007)
    assert (eval_lo, eval_hi) == (5000, 5499)
    overlap = set(range(train_lo, train_hi + 1)) & set(range(eval_lo, eval_hi + 1))
    assert overlap == set(range(5000, 5500))
    assert len(overlap) == 500


def test_stage3_deeparcade_0_06_reset_seed_semantics():
    """Constructor opening is one counted draw; a bare reset continues the
    stream; reset(seed=...) reproduces the constructor opening exactly."""
    env = SnakeEnv(seed=123)
    opening = env.food
    env.reset()
    assert env.food != opening  # second draw, not the counted opening
    reseeded = SnakeEnv(seed=999)
    reseeded.reset(seed=123)
    assert reseeded.food == opening
    assert reseeded.food == SnakeEnv(seed=123).food


def test_stage3_deeparcade_0_06_keyed_rows_carry_immutable_identity():
    """Every keyed row traces to one namespace, index, env key and policy
    key, and rerunning the same keys reproduces the same rows exactly."""
    keys = episode_keys("test", 0, 3)

    def make_policy(policy_key):
        return RandomPolicy(seed=policy_key)

    rows_a = run_keyed_episodes(make_policy, keys)
    rows_b = run_keyed_episodes(make_policy, keys)
    assert rows_a == rows_b
    for row, key in zip(rows_a, keys):
        assert isinstance(key, EpisodeKey)
        assert row["namespace"] == "test"
        assert row["index"] == key.index
        assert row["env_key"] == key.env_key
        assert row["policy_key"] == key.policy_key
        assert row["episode_key"] == key.label
    # Paired subset is a traceable prefix of the independent key list.
    assert [r["episode_key"] for r in rows_a[:2]] == [
        k.label for k in keys[:2]
    ]


def test_stage3_deeparcade_0_06_split_use_protocol():
    """Development tuning reads validation only; the final test split is
    write-protected as a protocol: tuning on test raises, final on test
    passes, tuning on validation passes."""
    check_split_use("validation", "tuning")
    check_split_use("test", "final")
    for namespace, purpose in (("test", "tuning"), ("train", "tuning"),
                               ("validation", "final"), ("train", "final")):
        try:
            check_split_use(namespace, purpose)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"check_split_use({namespace!r}, {purpose!r}) must raise"
            )


def test_stage3_deeparcade_0_06_manifest_loads_and_recomputes():
    """runs/splits_manifest.json loads, its test split is write-protected,
    its sample keys recompute from the derivation, and old results carry
    the historical label instead of being rewritten."""
    path = os.path.join(ROOT, "runs", "splits_manifest.json")
    manifest = load_splits_manifest(path)
    assert manifest["version"] == 1
    assert manifest["namespaces"]["test"]["write_protected"] is True
    assert manifest["namespaces"]["test"]["purpose"] == "final evaluation only"
    assert manifest["namespaces"]["validation"]["purpose"] == "tuning only"
    assert "training-seed overlap" in manifest["historical_results"]["label"]
    assert "MCTS mismatch" in manifest["historical_results"]["label"]
    for sample in manifest["sample_keys"]:
        key = derive_key(sample["namespace"], sample["index"], sample["domain"])
        assert key == sample["key"]
    rebuilt = build_splits_manifest()
    assert rebuilt["namespaces"] == manifest["namespaces"]
    assert rebuilt["sample_keys"] == manifest["sample_keys"]
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == manifest
