"""Stage 3 deeparcade DA6: evaluate generalization and uncertainty.

Every headline number in the repo comes from one board size (12x12),
one training run per learned agent, and unequal evaluation counts
(500 episodes for most policies, 50 for MCTS). These tests pin the
generalization sweep that answers that: held-out board sizes and the
browser game's maze obstacle layout, per-seed raw outcomes behind
every summary, paired bootstrap score gaps with 95% intervals,
survival and food efficiency, consistent full-board-win and timeout
handling, and disclosed training/evaluation seed sets.

The sweep lives in scripts/generalization.py and writes
runs/generalization.json. It uses keyed episodes from the P0-06 test
namespace at indices the DA8 comparison did not touch, so its scores
are a separate labeled namespace from runs/eval.json (legacy) and
from the DA8 keyed comparison.
"""

import hashlib
import json
import os

import pytest

from narcade.snake import SnakeEnv
from scripts import generalization as gen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")


# ---------------------------------------------------------------- maze layout

def test_maze_walls_match_browser_layout_12x12():
    """The 12x12 maze is the browser game's mazeWalls set exactly:
    four symmetric 3-cell corner blocks."""
    expected = {
        (3, 3), (3, 4), (4, 3),
        (3, 8), (3, 7), (4, 8),
        (8, 3), (8, 4), (7, 3),
        (8, 8), (8, 7), (7, 8),
    }
    assert gen.maze_walls(12, 12) == frozenset(expected)


def test_maze_walls_scale_with_board():
    walls = gen.maze_walls(16, 16)
    assert len(walls) == 12
    assert (3, 3) in walls and (12, 12) in walls and (11, 12) in walls
    # symmetric under 180-degree rotation
    assert {(15 - r, 15 - c) for r, c in walls} == set(walls)


def test_maze_layout_must_not_cover_the_opening():
    """The snake always opens on the center row. The browser layout
    overlaps that opening on 8x8 and 10x10 boards, so those sizes are
    excluded from the maze conditions and the guard refuses them."""
    for size in (12, 16, 20):
        gen.assert_layout_playable(size, size, gen.maze_walls(size, size))
    for size in (8, 10):
        with pytest.raises(ValueError):
            gen.assert_layout_playable(size, size, gen.maze_walls(size, size))
    maze_ids = [c["id"] for c in gen.CONDITIONS if c["kind"] == "maze"]
    assert maze_ids == ["maze-12x12", "maze-16x16"]


# ---------------------------------------------------------------- walled env

def test_wall_cell_is_lethal_and_food_never_spawns_on_walls():
    walls = frozenset({(6, 7)})
    env = gen.WalledSnakeEnv(width=12, height=12, seed=5, walls=walls)
    # opening head is (6, 6) heading right; the wall is one step ahead
    _, _, done, info = env.step(0)  # STRAIGHT
    assert done and info["reason"] == "death"
    maze = gen.maze_walls(12, 12)
    for seed in range(50):
        env = gen.WalledSnakeEnv(width=12, height=12, seed=seed, walls=maze)
        assert env.food not in maze


# ---------------------------------------------------------------- episode rows

def _tiny_keys(n=5):
    from narcade.experiment_seeds import episode_keys
    return episode_keys(gen.SWEEP_NAMESPACE, gen.INDEX_START, n)


def test_rows_carry_outcome_steps_and_reproduce():
    cond = {"id": "classic-8x8", "kind": "classic", "width": 8, "height": 8}
    rows1 = gen.run_condition_episodes(cond, "greedy", _tiny_keys())
    rows2 = gen.run_condition_episodes(cond, "greedy", _tiny_keys())
    assert rows1 == rows2  # same key, same outcome, regardless of order
    for row in rows1:
        assert row["outcome"] in ("death", "win", "timeout")
        assert row["steps"] >= 1
        assert row["score"] >= 0
        assert row["episode_key"].startswith("test:")


def test_timeout_outcome_is_recorded_consistently():
    cond = {"id": "classic-12x12", "kind": "classic", "width": 12, "height": 12}
    rows = gen.run_condition_episodes(
        cond, "greedy", _tiny_keys(3),
        env_factory=lambda key, c: SnakeEnv(width=12, height=12,
                                            seed=key.env_key, max_steps=2))
    assert all(r["outcome"] == "timeout" and r["steps"] == 2 for r in rows)


def test_full_board_win_outcome_is_recorded():
    from collections import deque

    def factory(key, cond):
        env = SnakeEnv(width=4, height=4, seed=key.env_key)
        cells = [(r, c) for r in range(4) for c in range(4)
                 if (r, c) != (0, 0)]
        cells.remove((0, 1))
        cells.append((0, 1))  # head last, adjacent to the only free cell
        env.body = deque(cells)
        env.heading = 3  # left, toward (0, 0)
        env.food = (0, 0)
        env.score = 0
        env.steps = 0
        env.done = False
        return env

    class Straight:
        def act(self, env):
            return 0

    cond = {"id": "classic-4x4", "kind": "classic", "width": 4, "height": 4}
    rows = gen.run_condition_episodes(cond, "greedy", _tiny_keys(1),
                                      env_factory=factory,
                                      policy_override=Straight())
    assert rows[0]["outcome"] == "win"
    assert rows[0]["score"] == 1 and rows[0]["steps"] == 1


# ---------------------------------------------------------------- bootstrap

def _rows(scores, start=0):
    return [{"episode_key": f"test:{2000 + start + i}", "score": s,
             "steps": 10 * (i + 1), "outcome": "death"}
            for i, s in enumerate(scores)]


def test_paired_bootstrap_known_values():
    a = _rows([10, 10, 10, 10])
    b = _rows([8, 8, 8, 8])
    out = gen.paired_bootstrap(a, b, resamples=500, seed=1)
    assert out["n"] == 4
    assert out["mean_diff"] == 2.0
    assert out["ci_low"] == 2.0 and out["ci_high"] == 2.0
    assert out["frac_positive"] == 1.0
    same = gen.paired_bootstrap(a, a, resamples=500, seed=1)
    assert same["ci_low"] == 0.0 and same["ci_high"] == 0.0


def test_paired_bootstrap_is_deterministic_and_covers_zero():
    a = _rows([5, 9, 3, 12, 7, 6])
    b = _rows([6, 8, 4, 11, 7, 5])
    out1 = gen.paired_bootstrap(a, b, resamples=2000, seed=7)
    out2 = gen.paired_bootstrap(a, b, resamples=2000, seed=7)
    assert out1 == out2
    assert out1["ci_low"] < 0.0 < out1["ci_high"]


def test_paired_bootstrap_requires_shared_episode_keys():
    a = _rows([1, 2, 3])
    b = _rows([1, 2, 3], start=100)
    with pytest.raises(ValueError):
        gen.paired_bootstrap(a, b, resamples=100, seed=1)


# ---------------------------------------------------------------- summaries

def test_summarize_reports_distribution_outcomes_efficiency():
    rows = [
        {"episode_key": "test:2000", "score": 4, "steps": 100, "outcome": "death"},
        {"episode_key": "test:2001", "score": 8, "steps": 200, "outcome": "death"},
        {"episode_key": "test:2002", "score": 12, "steps": 300, "outcome": "win"},
        {"episode_key": "test:2003", "score": 12, "steps": 400, "outcome": "timeout"},
    ]
    s = gen.summarize_rows(rows)
    assert s["n"] == 4
    assert s["score"]["mean"] == 9.0 and s["score"]["median"] == 10.0
    assert s["score"]["max"] == 12 and s["score"]["min"] == 4
    assert s["outcomes"] == {"death": 2, "win": 1, "timeout": 1}
    rates = s["outcome_rates"]
    assert abs(sum(rates.values()) - 1.0) < 1e-12
    assert s["mean_steps"] == 250.0
    # food per 100 steps, averaged per episode: .04, .04, .04, .03 -> 3.75
    assert s["mean_food_per_100_steps"] == pytest.approx(3.75)


# ---------------------------------------------------------------- document

def _tiny_document():
    conds = [{"id": "classic-8x8", "kind": "classic", "width": 8, "height": 8}]
    results = {"classic-8x8": {
        "greedy": _rows([3, 5, 4]),
        "random": _rows([0, 1, 0]),
    }}
    return gen.build_generalization_document(results, conds, resamples=500)


def test_document_validates_and_records_seed_disclosure():
    doc = _tiny_document()
    assert gen.validate_generalization_document(doc) is doc
    disc = doc["seed_disclosure"]
    assert disc["evaluation"]["namespace"] == "test"
    assert disc["evaluation"]["purpose"] == "final"
    assert disc["evaluation"]["index_start"] == gen.INDEX_START
    # training seed sets are read from the committed training metadata
    assert disc["training"]["rookie"]["seed"] == 7
    assert disc["training"]["rookie"]["episodes"] == 30000
    assert disc["training"]["pro"]["seed"] == 11
    assert disc["training"]["pro"]["steps"] == 300000
    assert doc["namespace_note"]  # labeled as its own namespace


def test_paired_gaps_use_the_shared_subset_but_refuse_partial_overlap():
    """The MCTS subset convention: a policy evaluated on a subset of
    the shared keys pairs over exactly that subset. Partially
    overlapping pools (neither contains the other) get no gap."""
    conds = [{"id": "classic-12x12", "kind": "classic",
              "width": 12, "height": 12}]
    full = _rows([10, 12, 14, 16])
    subset = _rows([9, 11])  # first two keys only, grandmaster-style
    results = {"classic-12x12": {"greedy": full, "grandmaster": subset}}
    doc = gen.build_generalization_document(results, conds, resamples=200)
    gaps = doc["conditions"]["classic-12x12"]["paired_score_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["n"] == 2
    assert gaps[0]["a"] == "grandmaster" and gaps[0]["b"] == "greedy"
    assert gaps[0]["mean_diff"] == pytest.approx(-1.0)  # (9-10 + 11-12) / 2
    assert gen.validate_generalization_document(doc) is doc

    partial = _rows([9, 11, 13, 15], start=4)  # keys 2002-2005: overlap of 2, neither contains
    results = {"classic-12x12": {"greedy": full, "grandmaster": partial}}
    doc = gen.build_generalization_document(results, conds, resamples=200)
    assert doc["conditions"]["classic-12x12"]["paired_score_gaps"] == []


def test_document_validation_catches_tampering():
    doc = _tiny_document()
    doc["conditions"]["classic-8x8"]["policies"]["greedy"]["score"]["mean"] += 1
    with pytest.raises(ValueError):
        gen.validate_generalization_document(doc)
    doc = _tiny_document()
    pair = doc["conditions"]["classic-8x8"]["paired_score_gaps"][0]
    pair["ci_high"] += 1.0
    with pytest.raises(ValueError):
        gen.validate_generalization_document(doc)
    doc = _tiny_document()
    doc["conditions"]["classic-8x8"]["rows"]["greedy"][0]["score"] += 1
    with pytest.raises(ValueError):
        gen.validate_generalization_document(doc)


# ---------------------------------------------------------------- neural bound

def test_neural_encoders_are_bound_to_12x12():
    """The report's central scope fact, measured: the flat DQN encoder
    asserts the training board size outright."""
    from scripts.train_dqn import encode_dqn
    with pytest.raises(AssertionError):
        encode_dqn(SnakeEnv(width=10, height=10, seed=1))
    support = gen.neural_board_support()
    assert support["pro"]["evaluable_board_sizes"] == [12]
    assert support["cnn_residual_seed9"]["evaluable_board_sizes"] == [12]
    assert "12" in support["pro"]["reason"]


# ---------------------------------------------------------------- no side effects

def test_sweep_does_not_touch_frozen_eval_artifacts():
    def digest(name):
        with open(os.path.join(RUNS, name), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    before = (digest("eval.json"), digest("eval_legacy.json"))
    cond = {"id": "classic-8x8", "kind": "classic", "width": 8, "height": 8}
    gen.run_condition_episodes(cond, "random", _tiny_keys(3))
    assert (digest("eval.json"), digest("eval_legacy.json")) == before


def test_committed_generalization_document_validates():
    path = os.path.join(RUNS, "generalization.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    assert gen.validate_generalization_document(doc) is doc
    # the committed sweep covers every declared condition and policy row
    for cond_id, section in doc["conditions"].items():
        assert section["rows"], cond_id
