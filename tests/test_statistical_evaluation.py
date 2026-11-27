"""Stage 3 P1-05: statistical research benchmark.

The evaluation pipeline must report uncertainty next to every ranking,
reproduce its intervals from the stored per-episode arrays, keep the
train and test splits disjoint, and never select a model on the final
test split. The imitation arm must report BOTH offline test-state
action accuracy and on-policy rollouts, with claims conditional on the
state distribution.

Failing-first fixtures: the shared benchmark module
(scripts/statistical_benchmark.py) did not exist, the imitation eval
seeds (10000+) overlapped the training env-key range, and neither
scripts/evaluate.py nor scripts/analyze.py stored per-episode rows
with failure causes.
"""

import copy
import json

import pytest

from narcade.experiment_seeds import episode_keys
from narcade.policies import GreedyPolicy
from scripts.evaluate import (
    build_eval_document,
    head_to_head_with_counts,
    summarize_with_uncertainty,
    validate_eval_document,
)
from scripts.statistical_benchmark import (
    BOOTSTRAP_SEED,
    assert_disjoint_episode_keys,
    build_benchmark_document,
    imitation_offline_accuracy,
    mean_bootstrap_ci,
    paired_bootstrap_ci,
    rank_with_uncertainty,
    select_best_on_validation,
    validate_benchmark_document,
    win_draw_loss,
)


def _rows(policy_scores):
    """Per-episode rows for one policy on test indices 0..n-1."""
    keys = episode_keys("test", 0, len(policy_scores))
    return [
        {
            "episode_key": k.label,
            "namespace": k.namespace,
            "index": k.index,
            "env_key": k.env_key,
            "score": s,
            "outcome": "death" if s < 5 else "timeout",
        }
        for k, s in zip(keys, policy_scores)
    ]


def _doc():
    rows = {
        "greedy": _rows([18, 20, 22, 19, 21, 20, 23, 18, 21, 20]),
        "rookie": _rows([19, 21, 20, 22, 20, 21, 22, 19, 20, 21]),
    }
    return build_benchmark_document(
        rows,
        search_cost={
            "greedy": {"steps": 0, "updates": 0, "seconds": 1.5, "episodes": 10},
            "rookie": {"steps": 30000, "updates": 0, "seconds": 12.0, "episodes": 10},
        },
        curves={
            "rookie": {"training_steps": [1000, 2000], "mean_score": [8.5, 11.2]},
        },
        multiseed={
            "rookie": {
                "per_seed": {"7": 20.1, "8": 20.4},
                "across_seed_mean": 20.25,
                "across_seed_std": 0.15,
                "n_seeds": 2,
            },
        },
        selection={
            "selected_on": "validation",
            "selected_policy": "rookie",
            "rule": "highest validation mean",
        },
    )


def test_stage3_deeparcade_1_05_bootstrap_ci_reproducible_from_stored_arrays():
    """The paired interval is a deterministic function of the stored
    arrays: same arrays and seed reproduce it exactly, including after
    a JSON round trip, as a stored artifact would be read."""
    a = [18, 20, 22, 19, 21, 20, 23, 18, 21, 20]
    b = [15, 17, 16, 18, 15, 17, 16, 15, 18, 16]
    first = paired_bootstrap_ci(a, b)
    second = paired_bootstrap_ci(a, b)
    assert first == second
    assert first["n"] == 10
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]
    assert first["seed"] == BOOTSTRAP_SEED
    saved = json.loads(json.dumps(first))
    assert paired_bootstrap_ci(a, b) == saved
    mean_ci = mean_bootstrap_ci(a)
    assert mean_ci["n"] == 10
    assert mean_ci["ci_low"] <= mean_ci["mean"] <= mean_ci["ci_high"]


def test_stage3_deeparcade_1_05_bootstrap_ci_moves_when_array_changes():
    """Tampering with one stored score must move the interval, so a
    corrupted artifact cannot keep its old interval."""
    a = [18, 20, 22, 19, 21, 20, 23, 18, 21, 20]
    b = [15, 17, 16, 18, 15, 17, 16, 15, 18, 16]
    base = paired_bootstrap_ci(a, b)
    tampered = paired_bootstrap_ci([18, 20, 22, 19, 21, 20, 23, 18, 21, 99], b)
    assert tampered["mean_diff"] != base["mean_diff"]
    assert tampered["ci_low"] != base["ci_low"] or tampered["ci_high"] != base["ci_high"]


def test_stage3_deeparcade_1_05_win_draw_loss_counts_sum_to_n():
    """Win/draw/loss counts partition the paired episodes exactly."""
    a = [18, 20, 22, 19, 21]
    b = [18, 17, 22, 18, 21]
    counts = win_draw_loss(a, b)
    assert counts == {"wins": 2, "draws": 3, "losses": 0, "n": 5}
    all_draw = win_draw_loss(a, list(a))
    assert all_draw == {"wins": 0, "draws": 5, "losses": 0, "n": 5}


def test_stage3_deeparcade_1_05_ranking_carries_n_and_ci_adjacent():
    """Every ranking entry carries its N and its interval next to the
    mean, and the ranking is sorted by mean with a deterministic
    tie-break."""
    rows = {
        "greedy": _rows([18, 20, 22, 19, 21, 20, 23, 18, 21, 20]),
        "rookie": _rows([19, 21, 20, 22, 20, 21, 22, 19, 20, 21]),
        "random": _rows([2, 0, 3, 1, 2, 0, 1, 3, 2, 1]),
    }
    ranking = rank_with_uncertainty(rows)
    assert [r["policy"] for r in ranking] == ["rookie", "greedy", "random"]
    for entry in ranking:
        assert entry["n"] == 10
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]
        assert entry["ci_low"] < entry["ci_high"]
        for other, pair in entry["pairwise"].items():
            assert pair["n"] == 10
            assert pair["wins"] + pair["draws"] + pair["losses"] == 10
            assert pair["ci_low"] <= pair["mean_diff"] <= pair["ci_high"]


def test_stage3_deeparcade_1_05_ranking_rejects_misaligned_episodes():
    """A ranking over different episode sets fails visibly instead of
    comparing different episodes."""
    rows = {
        "greedy": _rows([18, 20, 22]),
        "rookie": _rows([19, 21, 20])[:-1],
    }
    with pytest.raises(ValueError, match="episode"):
        rank_with_uncertainty(rows)


def test_stage3_deeparcade_1_05_no_train_test_key_overlap():
    """Train and test episode keys are disjoint, and the check fails
    visibly on any shared key."""
    train_keys = [k.env_key for k in episode_keys("train", 0, 50)]
    test_keys = [k.env_key for k in episode_keys("test", 0, 50)]
    assert_disjoint_episode_keys(train_keys, test_keys, "train", "test")
    overlapping = test_keys[:1] + train_keys[1:]
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_episode_keys(overlapping, test_keys, "train", "test")


def test_stage3_deeparcade_1_05_test_split_refuses_model_selection():
    """Model selection reads validation only: asking it to select on
    the final test split raises, so no favorable model can be picked
    from the test data."""
    scores = {"greedy": [18, 20, 22], "rookie": [20, 21, 22]}
    picked = select_best_on_validation(scores, "validation")
    assert picked["selected_policy"] == "rookie"
    assert picked["selected_on"] == "validation"
    assert picked["mean"] == 21.0
    with pytest.raises(ValueError, match="test"):
        select_best_on_validation(scores, "test")
    with pytest.raises(ValueError, match="train"):
        select_best_on_validation(scores, "train")


def test_stage3_deeparcade_1_05_benchmark_document_round_trip_and_tamper():
    """The benchmark document validates after a JSON round trip, and
    tampering with a stored score, a ranking interval, or the
    selection split fails visibly."""
    doc = _doc()
    validate_benchmark_document(json.loads(json.dumps(doc)))
    assert doc["ranking"][0]["policy"] == "rookie"
    assert doc["selection"]["selected_on"] == "validation"
    assert doc["search_cost"]["rookie"]["steps"] == 30000
    assert doc["curves"]["rookie"]["training_steps"] == [1000, 2000]
    assert doc["multiseed"]["rookie"]["n_seeds"] == 2

    tampered_score = copy.deepcopy(doc)
    tampered_score["policies"]["greedy"]["rows"][0]["score"] += 5
    with pytest.raises(ValueError):
        validate_benchmark_document(tampered_score)

    tampered_rank = copy.deepcopy(doc)
    tampered_rank["ranking"][0]["ci_low"] += 1.0
    with pytest.raises(ValueError):
        validate_benchmark_document(tampered_rank)

    stripped_rank = copy.deepcopy(doc)
    del stripped_rank["ranking"][0]["ci_low"]
    with pytest.raises(ValueError, match="ci_low"):
        validate_benchmark_document(stripped_rank)

    test_selection = copy.deepcopy(doc)
    test_selection["selection"]["selected_on"] = "test"
    with pytest.raises(ValueError, match="test"):
        validate_benchmark_document(test_selection)


def test_stage3_deeparcade_1_05_imitate_offline_accuracy_and_keyed_rollouts():
    """The imitation arm reports offline test-state action accuracy AND
    on-policy rollouts, both on test-namespace episodes that never
    overlap the training env keys used to collect demonstrations."""
    from scripts.diagnose_dqn import train_imitate

    summary, net = train_imitate(steps=1200, seed=5, collect_n=3000, eval_eps=5)
    assert summary["variant"] == "imitate"
    assert summary["eval_split"] == "test"
    assert len(summary["imitate_eval_scores"]) == 5
    assert all(s >= 0 for s in summary["imitate_eval_scores"])
    assert len(summary["eval_episode_keys"]) == 5
    assert all(k.startswith("test:") for k in summary["eval_episode_keys"])
    offline = summary["imitate_offline_accuracy"]
    assert offline["split"] == "test"
    assert offline["n_states"] > 0
    assert 0.0 <= offline["accuracy"] <= 1.0
    assert "test-state distribution" in offline["scope"]
    # The offline states come from test episodes, so their env keys
    # cannot intersect the training env keys used in collection.
    assert_disjoint_episode_keys(
        summary["train_env_keys"], summary["offline_env_keys"],
        "train", "test")


def test_stage3_deeparcade_1_05_offline_accuracy_known_answer():
    """Offline accuracy counts exact greedy-action matches: a net that
    always picks the labeled action scores 1.0, a net that never does
    scores 0.0."""
    import numpy as np
    import torch

    from scripts.train_dqn import DQN

    torch.manual_seed(0)
    net = DQN()
    states = np.random.RandomState(0).randn(8, 436).astype(np.float32)
    with torch.no_grad():
        actions = net(torch.from_numpy(states)).argmax(1).numpy()
    result = imitation_offline_accuracy(net, states, actions)
    assert result == {"accuracy": 1.0, "n_states": 8, "correct": 8}
    wrong = (actions + 1) % 3
    result = imitation_offline_accuracy(net, states, wrong)
    assert result == {"accuracy": 0.0, "n_states": 8, "correct": 0}


def test_stage3_deeparcade_1_05_evaluate_statistics_section_validates():
    """build_eval_document now carries a statistics section: per-policy
    mean intervals and pairwise paired bootstrap intervals with
    win/draw/loss counts and N, all recomputed from the stored paired
    scores by the validator."""
    from scripts.evaluate import run_episode_rows

    seeds = [5000, 5001, 5002, 5003, 5004]
    greedy = run_episode_rows(GreedyPolicy().act, seeds)
    stub = [{"seed": s, "score": g["score"] + 1} for s, g in zip(seeds, greedy)]
    rows = {"greedy": greedy, "stub": stub}
    paired = {n: r[:3] for n, r in rows.items()}
    paired["grandmaster"] = [{"seed": s, "score": v}
                             for s, v in zip(seeds[:3], [40, 44, 48])]
    meta = {n: {"policy_id": f"test-{n}", "artifact": None}
            for n in ("greedy", "stub", "grandmaster")}
    doc = build_eval_document(
        paired, rows, {"n": 5, "mcts_n": 3, "mcts_sims": 80}, meta)
    validate_eval_document(json.loads(json.dumps(doc)))
    stats = doc["statistics"]
    assert stats["per_policy"]["greedy"]["n"] == 3
    assert stats["per_policy"]["greedy"]["ci_low"] <= \
        stats["per_policy"]["greedy"]["mean"] <= \
        stats["per_policy"]["greedy"]["ci_high"]
    pair = stats["pairwise"]["stub"]["greedy"]
    assert pair["n"] == 3
    assert pair["wins"] + pair["draws"] + pair["losses"] == 3
    assert pair["mean_diff"] == 1.0
    tampered = copy.deepcopy(doc)
    tampered["statistics"]["pairwise"]["stub"]["greedy"]["ci_low"] += 1.0
    with pytest.raises(ValueError, match="statistics"):
        validate_eval_document(tampered)


def test_stage3_deeparcade_1_05_head_to_head_counts_sum_to_n():
    """Head-to-head with counts reports wins/draws/losses that sum to
    N next to the rates."""
    table = head_to_head_with_counts({"a": [3, 2, 1], "b": [3, 1, 1]})
    cell = table["a"]["b"]
    assert cell["wins"] == 1 and cell["draws"] == 2 and cell["losses"] == 0
    assert cell["n"] == 3
    assert cell["win_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert table["a"]["a"] is None


def test_stage3_deeparcade_1_05_summarize_with_uncertainty():
    """summarize_with_uncertainty keeps the legacy summary fields and
    adds the interval, N, and the bootstrap settings next to it."""
    summary = summarize_with_uncertainty([18, 20, 22, 19, 21])
    assert summary["n"] == 5
    assert summary["mean"] == 20.0
    assert summary["ci_low"] <= 20.0 <= summary["ci_high"]
    assert summary["n_boot"] == 10000
    assert summary["seed"] == BOOTSTRAP_SEED


def test_stage3_deeparcade_1_05_analyze_rows_store_score_and_failure_cause():
    """analyze_rows stores every episode's score and failure cause, and
    the aggregate analyze() built from those rows matches exactly."""
    from scripts.analyze import analyze, analyze_rows

    rows = analyze_rows(GreedyPolicy().act, 4, seed0=31000)
    assert len(rows) == 4
    for row in rows:
        assert row["score"] >= 0
        assert row["outcome"] in ("death", "win", "timeout")
        assert row["steps"] > 0
    report = analyze(GreedyPolicy().act, 4, seed0=31000)
    assert report["episodes"] == 4
    assert report["deaths"] == {
        r["outcome"]: sum(1 for x in rows if x["outcome"] == r["outcome"])
        for r in rows
    }


def test_stage3_deeparcade_1_05_keyed_policy_eval_reproducible():
    """Keyed policy evaluation reproduces the same score for the same
    episode key regardless of order, and each row carries its key."""
    from scripts.diagnose_dqn import evaluate_policy_keyed
    from scripts.train_dqn import DQN

    torch = pytest.importorskip("torch")
    del torch
    keys = episode_keys("test", 0, 3)
    # One fixed net for both evaluations: the check is that episode
    # order cannot change an episode's outcome, not that two
    # differently initialized nets agree (they need not).
    net = DQN()
    rows = evaluate_policy_keyed(net, keys)
    assert [r["episode_key"] for r in rows] == ["test:0", "test:1", "test:2"]
    assert all(r["namespace"] == "test" for r in rows)
    again = evaluate_policy_keyed(net, list(reversed(keys)))
    by_key = {r["episode_key"]: r["score"] for r in again}
    for r in rows:
        assert by_key[r["episode_key"]] == r["score"]
