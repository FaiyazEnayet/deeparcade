"""DA2: evaluation pairs agents by identical episode seeds."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import run_episodes, elo_ratings, head_to_head
from narcade.policies import GreedyPolicy


def test_same_seeds_give_paired_environments():
    """Two agents on the same seed list face identical environments.

    A deterministic policy must produce identical scores on the same
    seeds, proving the seed list (not the agent) determines the episode.
    """
    seeds = [5000, 5001, 5002]
    a = run_episodes(GreedyPolicy().act, seeds)
    b = run_episodes(GreedyPolicy().act, seeds)
    assert a == b
    assert len(a) == len(seeds)


def test_paired_stats_compare_same_index_same_seed():
    """ELO and head-to-head treat index i as the same environment.

    With identical seed lists, a stronger deterministic policy beats a
    weaker one on the shared episodes; the tables reflect that.
    """
    seeds = [6000, 6001, 6002, 6003]
    greedy_scores = run_episodes(GreedyPolicy().act, seeds)

    def random_act(env):
        import random
        return random.Random(999).choice([0, 1, 2])

    # head-to-head on truly paired data: greedy vs itself draws everything
    paired = {"greedy": greedy_scores, "greedy2": list(greedy_scores)}
    h2h = head_to_head(paired)
    assert h2h["greedy"]["greedy2"]["win_rate"] == 0.0
    assert h2h["greedy"]["greedy2"]["draw_rate"] == 1.0
    # ELO of identical agents stays tied
    elo = elo_ratings(paired)
    assert elo["greedy"] == elo["greedy2"]


# Stage 3 P0-01 (2026-10-28): evaluation provenance chain.
import copy
import json

import pytest

from scripts.evaluate import (
    DQNPolicy,
    build_eval_document,
    elo_ratings_paired,
    head_to_head_paired,
    run_episode_rows,
    summarize,
    summarize_full,
    validate_eval_document,
    validate_paired_rows,
)

_SEEDS = [5000, 5001, 5002, 5003, 5004]


def _rows(scores, seeds):
    return [{"seed": s, "score": v} for s, v in zip(seeds, scores)]


def _fixture_rows():
    greedy = run_episode_rows(GreedyPolicy().act, _SEEDS)
    stub = _rows([g["score"] + 1 for g in greedy], _SEEDS)
    return {"greedy": greedy, "stub": stub}


def _fixture_doc():
    rows = _fixture_rows()
    paired = {n: r[:3] for n, r in rows.items()}
    paired["grandmaster"] = _rows([40, 44, 48], _SEEDS[:3])
    meta = {n: {"policy_id": f"test-{n}", "artifact": None}
            for n in ("greedy", "stub", "grandmaster")}
    return build_eval_document(
        paired, rows, {"n": 5, "mcts_n": 3, "mcts_sims": 80}, meta)


def test_stage3_deeparcade_0_01_paired_seed_sets_identical():
    """Paired rows share exactly one episode-key set; the symmetric
    difference is empty and validation returns the common seeds."""
    rows = _fixture_rows()
    assert validate_paired_rows(rows) == _SEEDS


def test_stage3_deeparcade_0_01_mismatched_seed_sets_raise():
    """A missing or extra episode key in any agent fails visibly."""
    rows = _fixture_rows()
    missing = copy.deepcopy(rows)
    missing["stub"] = missing["stub"][:-1]
    with pytest.raises(ValueError, match="symmetric difference"):
        validate_paired_rows(missing)
    extra = copy.deepcopy(rows)
    extra["stub"] = extra["stub"] + [{"seed": 9999, "score": 0}]
    with pytest.raises(ValueError, match="symmetric difference"):
        validate_paired_rows(extra)


def test_stage3_deeparcade_0_01_row_order_does_not_change_results():
    """Alignment is by episode key, not list position: permuting one
    agent's rows leaves ELO and head-to-head exactly unchanged. The old
    index-based pairing would silently compare different episodes here."""
    rows = _fixture_rows()
    permuted = copy.deepcopy(rows)
    permuted["stub"] = list(reversed(permuted["stub"]))
    assert elo_ratings_paired(permuted) == elo_ratings_paired(rows)
    assert head_to_head_paired(permuted) == head_to_head_paired(rows)


def test_stage3_deeparcade_0_01_summaries_rebuild_within_1e10():
    """Summaries rebuilt from the saved rows (after a JSON round trip,
    as a stored artifact would be read) agree to <= 1e-10."""
    doc = _fixture_doc()
    saved = json.loads(json.dumps(doc))
    validate_eval_document(saved)
    for name, scores in saved["paired"]["scores"].items():
        rebuilt = summarize_full(scores)
        stored = saved["paired"]["summary"][name]
        for metric in ("mean", "std", "median"):
            assert abs(rebuilt[metric] - stored[metric]) <= 1e-10


def test_stage3_deeparcade_0_01_corrupted_artifact_fails_visibly():
    """Wrong version, missing section, truncated scores, or tampered
    ELO/summary must raise, never silently yield numbers."""
    base = _fixture_doc()

    wrong_version = copy.deepcopy(base)
    wrong_version["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        validate_eval_document(wrong_version)

    missing = copy.deepcopy(base)
    del missing["paired"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_eval_document(missing)

    truncated = copy.deepcopy(base)
    truncated["paired"]["scores"]["greedy"] = truncated["paired"]["scores"]["greedy"][:-1]
    with pytest.raises(ValueError, match="length"):
        validate_eval_document(truncated)

    tampered_elo = copy.deepcopy(base)
    tampered_elo["elo"]["greedy"] += 1
    with pytest.raises(ValueError, match="elo does not recompute"):
        validate_eval_document(tampered_elo)

    tampered_summary = copy.deepcopy(base)
    tampered_summary["paired"]["summary"]["greedy"]["mean"] += 1.0
    with pytest.raises(ValueError, match="does not rebuild"):
        validate_eval_document(tampered_summary)


def test_stage3_deeparcade_0_01_dqn_weights_json_matches_architecture():
    """The committed DQN weights hold exactly the architecture
    scripts/train_dqn.py declares (436 -> 128 -> 64 -> 3). No torch
    needed: this guards the loader's input contract everywhere."""
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "dqn_weights.json")
    with open(path, encoding="utf-8") as f:
        w = json.load(f)
    assert w["input_dim"] == 436
    assert len(w["layers"]) == 3
    expected = [(128, 436, 128), (64, 128, 64), (3, 64, 3)]
    for layer, (rows_n, cols_n, bias_n) in zip(w["layers"], expected):
        assert len(layer["w"]) == rows_n
        assert all(len(r) == cols_n for r in layer["w"])
        assert len(layer["b"]) == bias_n


def test_stage3_deeparcade_0_01_dqn_policy_loads_from_json_weights():
    """DQNPolicy loads from the committed JSON weights (not the
    gitignored .pt) and acts deterministically."""
    torch = pytest.importorskip("torch")
    del torch
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "dqn_weights.json")
    pol = DQNPolicy(path)
    from narcade.snake import SnakeEnv
    env = SnakeEnv(width=12, height=12, seed=5000)
    env.reset()
    first = pol.act(env)
    assert first in (0, 1, 2)
    env2 = SnakeEnv(width=12, height=12, seed=5000)
    env2.reset()
    assert pol.act(env2) == first


def test_stage3_deeparcade_0_01_legacy_aliases_documented():
    """Top-level scores/summary keep their documented contents:
    independent pool for non-MCTS policies, paired pool for
    grandmaster, with the mixture stated in provenance."""
    doc = _fixture_doc()
    rows = _fixture_rows()
    assert doc["scores"]["greedy"] == [r["score"] for r in rows["greedy"]]
    assert doc["scores"]["grandmaster"] == [40, 44, 48]
    assert doc["summary"]["grandmaster"] == summarize([40, 44, 48])
    assert "paired pool for grandmaster" in doc["provenance"]["legacy_keys_note"]
