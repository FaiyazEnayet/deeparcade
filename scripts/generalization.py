"""Generalization and uncertainty evaluation (Stage 3 DA6).

Every headline number in this repo comes from one board size (12x12),
one training run per learned agent, and unequal evaluation counts
(500 episodes for most policies, 50 for MCTS). That cannot establish
a general agent hierarchy. This module evaluates the same policies on
held-out conditions and reports distributions and paired uncertainty
instead of a single ranking:

  - Conditions: classic boards at 8x8, 10x10, 12x12, 16x16 and 20x20,
    plus the browser game's maze obstacle layout at 12x12 and 16x16
    (the layout overlaps the fixed center opening on 8x8 and 10x10
    boards, so those sizes are excluded and the exclusion is enforced
    by assert_layout_playable, not just documented).
  - Per-seed raw outcomes: every episode is a keyed row carrying its
    score, its step count (survival) and its outcome (death, full-board
    win, or timeout), so score distributions are inspectable and the
    win/timeout handling is the environment's own, identical in every
    condition.
  - Paired bootstrap: for every pair of policies that share episode
    keys inside a condition, the per-episode score (and steps)
    differences are bootstrapped over episode keys; the report carries
    the mean gap, the 95% percentile interval and the share of episodes
    the first policy won.
  - Seed disclosure: the training seed sets are read from the
    committed training metadata (runs/train_meta.json,
    runs/dqn_meta.json, runs/dqn_manifest.json and the CNN metas), and
    the evaluation keys are P0-06 test-namespace keys at indices
    2000..2199, disjoint from the DA8 comparison's indices 0..499.

Namespace discipline: these scores are a separate labeled namespace.
They are NOT comparable to runs/eval.json (legacy unkeyed seeds with
training-seed overlap) and they are not the DA8 comparison (different
episode indices, same keyed protocol family). runs/eval.json and
runs/eval_legacy.json are never written by this module.

Neural scope, measured not asserted: both neural encoders assert the
12x12 training board (encode_dqn and encode_cnn_dqn), so the frozen
MLP and the CNN cannot play any other board size; and their spatial
encodings carry no wall channel, so they are not evaluated in the maze
(the browser game disables the learned tiers in maze mode for the same
reason). The tabular Q-table's 32-state encoding does read the walled
environment's danger bits, so rookie is evaluated in the maze. MCTS is
evaluated on a disclosed subset of conditions and episode keys (the
first MCTS_EPISODES keys of each chosen condition) because an
80-simulation search costs roughly two orders of magnitude more per
move than the other policies; the subset convention mirrors the
paired-subset convention of runs/eval.json.

Additive only: SnakeEnv is not modified. Walls are added by the
WalledSnakeEnv subclass below, which overrides exactly the two hooks
the environment already exposes (_cell_blocked, _place_food).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from narcade.experiment_seeds import (
    check_split_use,
    episode_keys,
    legacy_training_seed_range,
)
from narcade.mcts import MCTSPlayer
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.snake import SnakeEnv
from scripts.evaluate import _git_info, sha256_file

HERE = _HERE
RUNS = os.path.join(HERE, "..", "runs")

SCHEMA_VERSION = 1

# Evaluation keys: P0-06 test namespace ("final" purpose only), at an
# index range no earlier keyed run has used.
SWEEP_NAMESPACE = "test"
SWEEP_PURPOSE = "final"
INDEX_START = 2000
N_EPISODES = 200

# MCTS subset (see module docstring): 80 simulations per move, the
# repo's evaluation default, on the first MCTS_EPISODES shared keys.
MCTS_SIMS = 80
MCTS_EPISODES = 16
MCTS_CONDITION_IDS = (
    "classic-10x10", "classic-12x12", "classic-16x16",
    "maze-12x12", "maze-16x16",
)

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261107

NEURAL_POLICIES = ("pro", "cnn_residual_seed9")

POLICY_META = {
    "random": {"policy_id": "random-per-episode-policy-key", "kind": "baseline",
               "artifact": None},
    "greedy": {"policy_id": "greedy-heuristic", "kind": "heuristic",
               "artifact": None},
    "rookie": {"policy_id": "qtable", "kind": "learned",
               "artifact": "runs/qtable.json"},
    "pro": {"policy_id": "dqn-json-weights", "kind": "learned",
            "artifact": "runs/dqn_weights.json"},
    "cnn_residual_seed9": {"policy_id": "cnn-residual-seed9", "kind": "learned",
                           "artifact": "runs/cnn_dqn_residual_seed9_weights.json"},
    "grandmaster": {"policy_id": f"mcts-sims-{MCTS_SIMS}", "kind": "search",
                    "artifact": None},
}


def maze_walls(width, height):
    """The browser game's maze-mode wall layout (web/game.html
    mazeWalls, mirrored in tools/native_reference.py): four symmetric
    3-cell corner blocks. A pure function of the board size, exactly
    as in the game."""
    blocks = [[3, 3], [3, 4], [4, 3],
              [3, width - 4], [3, width - 5], [4, width - 4],
              [height - 4, 3], [height - 4, 4], [height - 5, 3],
              [height - 4, width - 4], [height - 4, width - 5],
              [height - 5, width - 4]]
    return frozenset((r, c) for r, c in blocks)


def opening_cells(width, height):
    """The fixed opening body: center row, three cells ending at the
    center, as placed by SnakeEnv.reset."""
    cy, cx = height // 2, width // 2
    return {(cy, cx - 2), (cy, cx - 1), (cy, cx)}


def assert_layout_playable(width, height, walls):
    """Refuse a wall layout that covers the fixed opening. On 8x8 and
    10x10 boards the browser maze layout does exactly that, which is
    why those sizes have no maze condition."""
    overlap = set(walls) & opening_cells(width, height)
    if overlap:
        raise ValueError(
            f"wall layout overlaps the opening body on a {width}x{height} "
            f"board at {sorted(overlap)}; the episode could not start"
        )


def _classic(size):
    return {"id": f"classic-{size}x{size}", "kind": "classic",
            "width": size, "height": size}


def _maze(size):
    return {"id": f"maze-{size}x{size}", "kind": "maze",
            "width": size, "height": size}


CONDITIONS = [
    _classic(8), _classic(10), _classic(12), _classic(16), _classic(20),
    _maze(12), _maze(16),
]


class WalledSnakeEnv(SnakeEnv):
    """SnakeEnv with a fixed wall set, for the maze conditions.

    Walls block exactly like body cells (movement, danger bits in
    encode_state, MCTS's view through _cell_blocked readers) and food
    never spawns on a wall. A full-board win then means the body
    covers every non-wall cell, which is the environment's own rule:
    _place_food finds no free cell. SnakeEnv itself is unchanged."""

    def __init__(self, *args, walls=frozenset(), **kwargs):
        self.walls = frozenset(walls)
        super().__init__(*args, **kwargs)

    def _cell_blocked(self, r, c, ignore_tail=True):
        if (r, c) in self.walls:
            return True
        return super()._cell_blocked(r, c, ignore_tail=ignore_tail)

    def _place_food(self):
        occupied = set(self.body) | self.walls
        free = [(r, c) for r in range(self.height) for c in range(self.width)
                if (r, c) not in occupied]
        self.food = self.rng.choice(free) if free else None


# ------------------------------------------------------------------ policies

class _PolicyContext:
    """Lazily loaded shared artifacts for one sweep process: the
    Q-table and the two neural policies (deterministic and stateless
    per call, so one instance serves every episode)."""

    def __init__(self, runs_dir):
        self.runs_dir = runs_dir
        self._qtable = None
        self._pro = None
        self._cnn = None

    @property
    def qtable(self):
        if self._qtable is None:
            with open(os.path.join(self.runs_dir, "qtable.json")) as f:
                self._qtable = json.load(f)
        return self._qtable

    @property
    def pro(self):
        if self._pro is None:
            from scripts.evaluate import DQNPolicy
            self._pro = DQNPolicy(os.path.join(self.runs_dir, "dqn_weights.json"))
        return self._pro

    @property
    def cnn(self):
        if self._cnn is None:
            from scripts.evaluate import CNNDQNPolicy
            self._cnn = CNNDQNPolicy(os.path.join(
                self.runs_dir, "cnn_dqn_residual_seed9_weights.json"))
        return self._cnn


def evaluable(condition, policy_name):
    """(True, None) when the sweep evaluates this policy in this
    condition, else (False, reason). The reasons are recorded in the
    report; nothing is dropped silently."""
    if policy_name in NEURAL_POLICIES:
        if condition["id"] == "classic-12x12":
            return True, None
        if condition["kind"] == "maze":
            return False, (
                "the neural encodings carry no wall channel, so the policy "
                "would play blind to the walls; the browser game disables "
                "the learned tiers in maze mode for the same reason"
            )
        return False, (
            "the neural encoders assert the 12x12 training board and "
            "cannot encode any other board size (see neural_board_support)"
        )
    if policy_name == "grandmaster" and condition["id"] not in MCTS_CONDITION_IDS:
        return False, (
            "outside the disclosed MCTS subset: an 80-simulation search "
            "is evaluated on the first "
            f"{MCTS_EPISODES} shared episode keys of "
            f"{', '.join(MCTS_CONDITION_IDS)} only"
        )
    return True, None


def _make_action_fn(policy_name, policy_key, condition, ctx):
    """A fresh policy for one episode, as the keyed protocol requires:
    stochastic policies are seeded from the episode's policy key, and
    MCTS receives the condition's wall set explicitly."""
    walls = maze_walls(condition["width"], condition["height"]) \
        if condition["kind"] == "maze" else frozenset()
    if policy_name == "random":
        pol = RandomPolicy(seed=policy_key)
        return pol.act
    if policy_name == "greedy":
        return GreedyPolicy().act
    if policy_name == "rookie":
        return QTablePolicy(ctx.qtable).act
    if policy_name == "pro":
        return ctx.pro.act
    if policy_name == "cnn_residual_seed9":
        return ctx.cnn.act
    if policy_name == "grandmaster":
        pol = MCTSPlayer(simulations=MCTS_SIMS, seed=policy_key)
        if walls:
            return lambda env: pol.act(env, walls=walls)
        return pol.act
    raise ValueError(f"unknown policy {policy_name!r}")


def _default_env_factory(key, condition):
    if condition["kind"] == "maze":
        walls = maze_walls(condition["width"], condition["height"])
        assert_layout_playable(condition["width"], condition["height"], walls)
        return WalledSnakeEnv(width=condition["width"],
                              height=condition["height"],
                              seed=key.env_key, walls=walls)
    return SnakeEnv(width=condition["width"], height=condition["height"],
                    seed=key.env_key)


def run_condition_episodes(condition, policy_name, keys, runs_dir=RUNS,
                           env_factory=None, policy_override=None, ctx=None):
    """One keyed episode per EpisodeKey in one condition. The
    environment is constructed with the episode's environment key and
    NOT reset again, so the constructor opening is the counted
    opening, exactly as in scripts.evaluate.run_keyed_episodes. Each
    row records the outcome the environment itself reported
    (death / win / timeout), so full-board wins and timeouts are
    handled identically in every condition."""
    if env_factory is None:
        env_factory = _default_env_factory
    if ctx is None and policy_override is None:
        ctx = _PolicyContext(runs_dir)
    rows = []
    for key in keys:
        env = env_factory(key, condition)
        if policy_override is not None:
            act = policy_override.act
        else:
            act = _make_action_fn(policy_name, key.policy_key, condition, ctx)
        info = {}
        done = False
        while not done:
            _, _, done, info = env.step(act(env))
        rows.append({
            "namespace": key.namespace,
            "index": key.index,
            "env_key": key.env_key,
            "policy_key": key.policy_key,
            "episode_key": key.label,
            "score": env.score,
            "steps": env.steps,
            "outcome": info.get("reason", "death"),
        })
    return rows


# ------------------------------------------------------------------ summaries

def _percentile(sorted_vals, p):
    if not sorted_vals:
        raise ValueError("percentile of an empty sample")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_rows(rows):
    """Distribution-first summary of one policy's rows in one
    condition: score quantiles (not only the mean), outcome counts
    and rates, survival (steps) and food efficiency."""
    import statistics
    if not rows:
        raise ValueError("cannot summarize zero episodes")
    scores = sorted(r["score"] for r in rows)
    steps = [r["steps"] for r in rows]
    outcomes = {"death": 0, "win": 0, "timeout": 0}
    for r in rows:
        if r["outcome"] not in outcomes:
            raise ValueError(f"unknown episode outcome {r['outcome']!r}")
        outcomes[r["outcome"]] += 1
    n = len(rows)
    return {
        "n": n,
        "score": {
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores) if n > 1 else 0.0,
            "median": statistics.median(scores),
            "q25": _percentile(scores, 25),
            "q75": _percentile(scores, 75),
            "min": scores[0],
            "max": scores[-1],
        },
        "outcomes": outcomes,
        "outcome_rates": {k: v / n for k, v in outcomes.items()},
        "mean_steps": statistics.mean(steps),
        "median_steps": statistics.median(steps),
        "mean_food_per_100_steps": statistics.mean(
            [100.0 * r["score"] / r["steps"] for r in rows]),
    }


def _aligned_diffs(rows_a, rows_b, field):
    """Per-episode differences a - b, aligned by episode key. Any key
    mismatch is a hard error: a paired gap over different episodes
    would be meaningless and must never be computed silently."""
    map_a = {}
    for r in rows_a:
        if r["episode_key"] in map_a:
            raise ValueError(f"duplicate episode key {r['episode_key']!r}")
        map_a[r["episode_key"]] = r[field]
    map_b = {}
    for r in rows_b:
        if r["episode_key"] in map_b:
            raise ValueError(f"duplicate episode key {r['episode_key']!r}")
        map_b[r["episode_key"]] = r[field]
    if set(map_a) != set(map_b):
        diff = sorted(set(map_a).symmetric_difference(set(map_b)))
        raise ValueError(
            f"paired comparison requires identical episode keys; "
            f"symmetric difference (sorted) = {diff}"
        )
    keys = sorted(map_a)
    return [map_a[k] - map_b[k] for k in keys]


def paired_bootstrap(rows_a, rows_b, resamples=BOOTSTRAP_RESAMPLES,
                     seed=BOOTSTRAP_SEED, field="score"):
    """Percentile bootstrap of the paired per-episode difference.

    Resamples episode keys with replacement under a fixed disclosed
    seed, so the interval is a deterministic function of the rows and
    validate_generalization_document can recompute it exactly."""
    import numpy as np
    diffs = _aligned_diffs(rows_a, rows_b, field)
    n = len(diffs)
    if n == 0:
        raise ValueError("paired bootstrap needs at least one shared episode")
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = arr[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return {
        "n": n,
        "mean_diff": float(arr.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "frac_positive": float(np.mean(arr > 0)),
        "frac_negative": float(np.mean(arr < 0)),
    }


# ------------------------------------------------------------------ neural scope

def neural_board_support():
    """Measure, per neural policy, where it can play: attempt the
    encoder on a 10x10 board and record the assertion it raises. The
    report cites this instead of assuming the limitation."""
    out = {}
    probes = {}
    try:
        from scripts.train_dqn import encode_dqn
        probes["pro"] = ("encode_dqn", encode_dqn)
    except Exception:
        pass
    try:
        from narcade.models import encode_cnn_dqn
        probes["cnn_residual_seed9"] = ("encode_cnn_dqn", encode_cnn_dqn)
    except Exception:
        pass
    for name, (encoder_name, encoder) in probes.items():
        try:
            encoder(SnakeEnv(width=10, height=10, seed=1))
            out[name] = {"evaluable_board_sizes": [10, 12],
                         "reason": f"{encoder_name} accepted a 10x10 board"}
        except AssertionError as exc:
            out[name] = {"evaluable_board_sizes": [12],
                         "reason": f"{encoder_name} asserts the training "
                                   f"board: {exc}"}
    return out


# ------------------------------------------------------------------ document

def _seed_disclosure(runs_dir=RUNS):
    with open(os.path.join(runs_dir, "train_meta.json")) as f:
        train_meta = json.load(f)
    with open(os.path.join(runs_dir, "dqn_meta.json")) as f:
        dqn_meta = json.load(f)
    with open(os.path.join(runs_dir, "dqn_manifest.json")) as f:
        dqn_manifest = json.load(f)
    with open(os.path.join(runs_dir, "cnn_dqn_residual_seed9_meta.json")) as f:
        cnn_meta = json.load(f)
    lo, hi = legacy_training_seed_range(train_meta["seed"],
                                        train_meta["episodes"])
    return {
        "training": {
            "rookie": {
                "artifact": "runs/qtable.json",
                "source": "runs/train_meta.json",
                "seed": train_meta["seed"],
                "episodes": train_meta["episodes"],
                "env_seed_range": [lo, hi],
            },
            "pro": {
                "artifact": "runs/dqn_weights.json",
                "source": "runs/dqn_meta.json + runs/dqn_manifest.json",
                "seed": dqn_meta["seed"],
                "steps": dqn_meta["steps"],
                "episodes": dqn_meta["episodes"],
                "episode_seed_range":
                    dqn_manifest["data_key_range"]["episode_seeds"],
            },
            "cnn_residual_seed9": {
                "artifact": "runs/cnn_dqn_residual_seed9_weights.json",
                "source": "runs/cnn_dqn_residual_seed9_meta.json",
                "seed": cnn_meta["seed"],
                "steps": cnn_meta["steps"],
                "episodes": cnn_meta["episodes"],
                "training_partition": (
                    "episode-key training partition of the P0-06 split "
                    "registry, disjoint from the test namespace"
                ),
            },
        },
        "evaluation": {
            "namespace": SWEEP_NAMESPACE,
            "purpose": SWEEP_PURPOSE,
            "index_start": INDEX_START,
            "episodes_per_condition": N_EPISODES,
            "mcts_episodes_per_condition": MCTS_EPISODES,
            "key_derivation": (
                "sha256('deeparcade-v1:<namespace>:<index>:<domain>') "
                "first 8 bytes, unsigned (src/narcade/experiment_seeds.py)"
            ),
        },
    }


def _paired_gaps(rows_by_policy, resamples, field):
    """Paired gaps over shared episode keys. Two policies pair on the
    intersection of their key sets when one set contains the other
    (identical pools, or the disclosed MCTS subset of a full pool);
    partially overlapping pools refuse to pair, because a gap over
    ragged overlap would mix different episode sets."""
    names = sorted(rows_by_policy)
    gaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys_a = {r["episode_key"] for r in rows_by_policy[a]}
            keys_b = {r["episode_key"] for r in rows_by_policy[b]}
            shared = keys_a & keys_b
            if not shared:
                continue
            if not (keys_a <= keys_b or keys_b <= keys_a):
                continue
            sub_a = [r for r in rows_by_policy[a]
                     if r["episode_key"] in shared]
            sub_b = [r for r in rows_by_policy[b]
                     if r["episode_key"] in shared]
            gap = paired_bootstrap(sub_a, sub_b,
                                   resamples=resamples, field=field)
            gaps.append({"a": a, "b": b, **gap})
    return gaps


def build_generalization_document(results, conditions,
                                  resamples=BOOTSTRAP_RESAMPLES,
                                  runs_dir=RUNS):
    """Assemble the generalization document from raw rows.

    results: {condition_id: {policy_name: rows}}.
    conditions: the condition dicts (id/kind/width/height)."""
    check_split_use(SWEEP_NAMESPACE, SWEEP_PURPOSE)
    doc_conditions = {}
    for cond in conditions:
        rows_by_policy = results.get(cond["id"], {})
        section = {
            "condition": cond,
            "rows": rows_by_policy,
            "policies": {name: summarize_rows(rows)
                         for name, rows in rows_by_policy.items()},
            "paired_score_gaps": _paired_gaps(rows_by_policy, resamples,
                                              "score"),
            "paired_steps_gaps": _paired_gaps(rows_by_policy, resamples,
                                              "steps"),
            "not_evaluated": {},
        }
        for name in POLICY_META:
            if name not in rows_by_policy:
                ok, reason = evaluable(cond, name)
                section["not_evaluated"][name] = (
                    reason if not ok else
                    "not run in this sweep invocation")
        doc_conditions[cond["id"]] = section
    commit, branch = _git_info()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/generalization.py",
        "namespace_note": (
            "Keyed episodes from the P0-06 test namespace, indices "
            f"{INDEX_START}..{INDEX_START + N_EPISODES - 1}, purpose "
            "'final'. A separate labeled namespace: not comparable to "
            "runs/eval.json (legacy unkeyed seeds 5000..5499 with "
            "training-seed overlap) and not the DA8 keyed comparison "
            "(test indices 0..499 on the 12x12 classic board only)."
        ),
        "seed_disclosure": _seed_disclosure(runs_dir),
        "bootstrap": {
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
            "method": (
                "percentile bootstrap over paired per-episode "
                "differences, resampling episode keys with replacement"
            ),
        },
        "mcts_subset": {
            "conditions": list(MCTS_CONDITION_IDS),
            "episodes_per_condition": MCTS_EPISODES,
            "simulations": MCTS_SIMS,
            "note": (
                "Grandmaster is evaluated on the first "
                f"{MCTS_EPISODES} shared episode keys of each listed "
                "condition; paired gaps involving grandmaster use "
                "exactly those shared keys."
            ),
        },
        "neural_board_support": neural_board_support(),
        "policies": {
            name: {**meta,
                   "artifact_sha256": sha256_file(os.path.join(
                       _ROOT, meta["artifact"])) if meta["artifact"] else None}
            for name, meta in POLICY_META.items()
        },
        "conditions": doc_conditions,
        "provenance": {
            "generated_from_commit": commit,
            "generated_on_branch": branch,
        },
    }


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def _check_summary(stored, rebuilt, where):
    for field in ("mean", "std", "median", "q25", "q75"):
        if not _close(stored["score"][field], rebuilt["score"][field]):
            raise ValueError(
                f"{where}: stored score {field} {stored['score'][field]} "
                f"does not rebuild from the saved rows "
                f"({rebuilt['score'][field]})"
            )
    for field in ("mean_steps", "median_steps", "mean_food_per_100_steps"):
        if not _close(stored[field], rebuilt[field]):
            raise ValueError(
                f"{where}: stored {field} {stored[field]} does not "
                f"rebuild from the saved rows ({rebuilt[field]})"
            )
    if stored["outcomes"] != rebuilt["outcomes"]:
        raise ValueError(f"{where}: stored outcome counts do not rebuild")
    if stored["n"] != rebuilt["n"]:
        raise ValueError(f"{where}: stored episode count does not rebuild")


def validate_generalization_document(doc):
    """Fail visibly on any inconsistency: every summary and every
    bootstrap interval must recompute from the saved per-seed rows.
    Returns the document unchanged on success."""
    if not isinstance(doc, dict):
        raise ValueError("generalization document must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported generalization schema_version "
            f"{doc.get('schema_version')!r}; this code reads "
            f"schema_version {SCHEMA_VERSION}"
        )
    for key in ("seed_disclosure", "bootstrap", "conditions"):
        if key not in doc:
            raise ValueError(
                f"generalization document is missing required key {key!r}")
    resamples = doc["bootstrap"]["resamples"]
    for cond_id, section in doc["conditions"].items():
        rows_by_policy = section["rows"]
        for name, rows in rows_by_policy.items():
            _check_summary(section["policies"][name], summarize_rows(rows),
                           f"{cond_id}/{name}")
        for field, key in (("score", "paired_score_gaps"),
                           ("steps", "paired_steps_gaps")):
            rebuilt = _paired_gaps(rows_by_policy, resamples, field)
            stored = section[key]
            if len(stored) != len(rebuilt):
                raise ValueError(
                    f"{cond_id}: stored {key} has {len(stored)} entries, "
                    f"rebuilt {len(rebuilt)}"
                )
            for s_gap, r_gap in zip(stored, rebuilt):
                if (s_gap["a"], s_gap["b"]) != (r_gap["a"], r_gap["b"]):
                    raise ValueError(f"{cond_id}: {key} pair order mismatch")
                for metric in ("mean_diff", "ci_low", "ci_high",
                               "frac_positive", "frac_negative"):
                    if not _close(s_gap[metric], r_gap[metric]):
                        raise ValueError(
                            f"{cond_id}: stored {key} for "
                            f"{s_gap['a']} vs {s_gap['b']} does not "
                            f"recompute: {metric} stored {s_gap[metric]} "
                            f"vs rebuilt {r_gap[metric]}"
                        )
    return doc


def load_generalization_document(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return validate_generalization_document(doc)


# ------------------------------------------------------------------ sweep

def _run_job(job):
    """One (condition, policy) job. Top-level so multiprocessing can
    pickle it. Keyed episodes are order-independent by construction,
    so parallel execution cannot change any row."""
    cond, policy_name, keys, runs_dir = job
    rows = run_condition_episodes(cond, policy_name, keys, runs_dir=runs_dir)
    return cond["id"], policy_name, rows


def run_sweep(runs_dir=RUNS, episodes=N_EPISODES, mcts_episodes=MCTS_EPISODES,
              workers=1):
    check_split_use(SWEEP_NAMESPACE, SWEEP_PURPOSE)
    keys = episode_keys(SWEEP_NAMESPACE, INDEX_START, episodes)
    mcts_keys = keys[:mcts_episodes]
    jobs = []
    for cond in CONDITIONS:
        for name in POLICY_META:
            ok, _ = evaluable(cond, name)
            if not ok:
                continue
            job_keys = mcts_keys if name == "grandmaster" else keys
            jobs.append((cond, name, job_keys, runs_dir))
    results = {cond["id"]: {} for cond in CONDITIONS}
    mcts_jobs = [j for j in jobs if j[1] == "grandmaster"]
    plain_jobs = [j for j in jobs if j[1] != "grandmaster"]
    for cond, name, job_keys, rd in plain_jobs:
        print(f"evaluating {name} on {cond['id']} "
              f"({len(job_keys)} episodes)...", flush=True)
        cid, pname, rows = _run_job((cond, name, job_keys, rd))
        results[cid][pname] = rows
    if workers > 1 and len(mcts_jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for cid, pname, rows in pool.imap_unordered(_run_job, mcts_jobs):
                print(f"evaluating grandmaster on {cid} "
                      f"({len(rows)} episodes) done", flush=True)
                results[cid][pname] = rows
    else:
        for job in mcts_jobs:
            print(f"evaluating grandmaster on {job[0]['id']} "
                  f"({len(job[2])} episodes)...", flush=True)
            cid, pname, rows = _run_job(job)
            results[cid][pname] = rows
    for section in results.values():
        for rows in section.values():
            rows.sort(key=lambda r: r["index"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--mcts-episodes", type=int, default=MCTS_EPISODES)
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    results = run_sweep(runs_dir=args.out, episodes=args.episodes,
                        mcts_episodes=args.mcts_episodes,
                        workers=args.workers)
    doc = build_generalization_document(results, CONDITIONS,
                                        resamples=args.resamples,
                                        runs_dir=args.out)
    validate_generalization_document(doc)
    out_path = os.path.join(args.out, "generalization.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {out_path}")
    for cond_id, section in doc["conditions"].items():
        for name, s in section["policies"].items():
            print(f"  {cond_id:16s} {name:20s} n={s['n']:<4d} "
                  f"mean={s['score']['mean']:<7.2f} "
                  f"median={s['score']['median']:<5} "
                  f"mean_steps={s['mean_steps']:.0f} "
                  f"outcomes={s['outcomes']}")
        for gap in section["paired_score_gaps"]:
            print(f"  {cond_id:16s} gap {gap['a']} - {gap['b']}: "
                  f"{gap['mean_diff']:+.2f} "
                  f"[{gap['ci_low']:+.2f}, {gap['ci_high']:+.2f}] "
                  f"(n={gap['n']})")


if __name__ == "__main__":
    main()
