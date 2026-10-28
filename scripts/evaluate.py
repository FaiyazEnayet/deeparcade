"""Evaluate all AI tiers + baselines. Saves runs/eval.json.

Policies: random, greedy (heuristic), rookie (tabular Q), pro (DQN), grandmaster (MCTS).
Metrics: score arrays, mean/std/median, head-to-head win rates, ELO ratings.
All numbers are measured, seeds fixed.

Evaluation schema v2 (Stage 3 P0-01):
  Every paired comparison is built from explicit per-episode rows
  ({"seed": ..., "score": ...}), never from bare score lists. The paired
  subset is the first ``mcts_n`` seeds of the independent seed range, and
  that convention is recorded in the artifact itself, along with policy
  IDs, artifact SHA-256 hashes, and the generating commit. ELO and
  head-to-head are computed ONLY on the paired subset. Legacy top-level
  keys (``scores``, ``summary``, ``elo``, ``head_to_head``, ``n_eval``,
  ``mcts_n``, ``mcts_sims``) are kept so existing readers (make_game,
  make_dashboard, renderer) keep working; their exact contents are
  documented in the artifact's ``provenance`` block:
    - ``scores`` / ``summary``: the independent pool for the four
      non-MCTS policies, plus the paired pool for grandmaster (MCTS
      plays no independent pool; it is too slow for 500 episodes).
    - ``elo`` / ``head_to_head``: paired-subset only.
  A document that fails validation (wrong schema version, missing keys,
  length mismatch, seed-set mismatch, or summaries/ELO that do not
  recompute from the saved rows) must fail visibly, never silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from narcade.mcts import MCTSPlayer
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.snake import SnakeEnv

HERE = _HERE
RUNS = os.path.join(HERE, "..", "runs")

SCHEMA_VERSION = 2


class DQNPolicy:
    """DQN loaded from the COMMITTED JSON weights export.

    The old loader read ``runs/dqn.pt``, which is gitignored and absent
    on a fresh checkout, so evaluation could not run from committed
    artifacts alone. ``runs/dqn_weights.json`` is written by
    ``scripts/train_dqn.py`` next to the .pt file and holds the full
    state dict (weights + biases for all three linear layers); loading
    it reproduces the .pt policy exactly.
    """

    def __init__(self, weights_json_path):
        import torch

        from scripts.train_dqn import DQN, INPUT_DIM

        with open(weights_json_path, encoding="utf-8") as f:
            w = json.load(f)
        if w.get("input_dim") != INPUT_DIM:
            raise ValueError(
                f"DQN weights input_dim {w.get('input_dim')} != expected {INPUT_DIM} "
                f"({weights_json_path}); refusing to load mismatched weights."
            )
        layers = w.get("layers", [])
        if len(layers) != 3:
            raise ValueError(
                f"DQN weights must hold exactly 3 layers, found {len(layers)} "
                f"({weights_json_path})."
            )
        keys = ["net.0", "net.2", "net.4"]
        sd = {}
        for key, layer in zip(keys, layers):
            sd[f"{key}.weight"] = torch.tensor(layer["w"], dtype=torch.float32)
            sd[f"{key}.bias"] = torch.tensor(layer["b"], dtype=torch.float32)
        self.net = DQN()
        self.net.load_state_dict(sd)
        self.net.eval()

    def act(self, env):
        import numpy as np
        import torch

        from scripts.train_dqn import encode_dqn

        with torch.no_grad():
            q = self.net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
        return int(q.argmax(1).item())


def run_episodes(policy_fn, seeds, width=12, height=12):
    """Run one episode per seed. Seeds are explicit so every agent can
    share the identical list for paired comparisons.

    Returns a bare score list, in seed order. Kept for existing callers
    (tools/native_reference.py, tools/native_eval_fuzz.py, older tests)
    that pass aligned seed lists. New code MUST use run_episode_rows
    so pairing is verified by episode ID, not by list position.
    """
    scores = []
    for seed in seeds:
        env = SnakeEnv(width=width, height=height, seed=seed)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(policy_fn(env))
        scores.append(env.score)
    return scores


def run_episode_rows(policy_fn, seeds, width=12, height=12):
    """Same as run_episodes, but each episode is a row carrying its own
    episode key: {"seed": seed, "score": score}. Pairing downstream is
    done by matching these keys, so row order can never silently
    misalign two agents' episodes."""
    return [{"seed": seed, "score": score}
            for seed, score in zip(seeds, run_episodes(policy_fn, seeds, width, height))]


def rows_to_seed_map(rows):
    """{seed: score} for a row list. Duplicate seeds are a hard error:
    one episode key must mean exactly one episode."""
    out = {}
    for row in rows:
        seed = row["seed"]
        if seed in out:
            raise ValueError(f"duplicate episode seed {seed} in evaluation rows")
        out[seed] = row["score"]
    return out


def validate_paired_rows(rows_by_agent):
    """Require every agent to share exactly the same episode-key set.

    Returns the common seeds in sorted order. Raises ValueError naming
    the symmetric difference when any agent's set differs, so a broken
    pairing fails visibly instead of silently comparing different
    episodes."""
    names = list(rows_by_agent.keys())
    if not names:
        raise ValueError("no agents supplied for paired validation")
    seed_sets = {n: set(rows_to_seed_map(rows_by_agent[n]).keys()) for n in names}
    base = seed_sets[names[0]]
    for n in names[1:]:
        diff = base.symmetric_difference(seed_sets[n])
        if diff:
            raise ValueError(
                f"paired episode-key mismatch between {names[0]} and {n}: "
                f"symmetric difference (sorted) = {sorted(diff)}; "
                f"paired comparisons require identical episode keys"
            )
    return sorted(base)


def aligned_paired_scores(rows_by_agent):
    """{name: [scores]} aligned by common episode key (sorted seeds).

    Row order in the input is irrelevant; alignment is by episode ID.
    This is the fix for the old index-based pairing, where permuting
    one agent's rows silently compared different episodes."""
    seeds = validate_paired_rows(rows_by_agent)
    maps = {n: rows_to_seed_map(rows) for n, rows in rows_by_agent.items()}
    return {n: [maps[n][s] for s in seeds] for n in rows_by_agent}, seeds


def summarize(scores):
    import statistics
    return {"n": len(scores), "mean": round(statistics.mean(scores), 2),
            "std": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
            "median": statistics.median(scores),
            "max": max(scores), "min": min(scores)}


def summarize_full(scores):
    """Unrounded summaries, for the v2 artifact sections. Rebuilding
    these from the saved rows must agree to <= 1e-10 (see
    validate_eval_document); the rounded summarize() above is kept for
    legacy readers and tools."""
    import statistics
    return {"n": len(scores), "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "median": statistics.median(scores),
            "max": max(scores), "min": min(scores)}


def _elo_from_aligned(scores_by_name, k=32):
    """ELO round-robin over aligned score lists. Arithmetic identical
    to the original elo_ratings; only the input alignment changed."""
    names = list(scores_by_name.keys())
    elo = {n: 1200.0 for n in names}
    n = len(next(iter(scores_by_name.values())))
    for i in range(n):
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                sa, sb = scores_by_name[names[a]][i], scores_by_name[names[b]][i]
                ea = 1 / (1 + 10 ** ((elo[names[b]] - elo[names[a]]) / 400))
                ra = 1.0 if sa > sb else (0.5 if sa == sb else 0.0)
                elo[names[a]] += k * (ra - ea)
                elo[names[b]] += k * ((1 - ra) - (1 - ea))
    return {n: round(e) for n, e in elo.items()}


def elo_ratings(results, k=32):
    """results: {name: [scores]}. Round-robin on paired seeds; ELO from scratch.

    Legacy list API: the caller guarantees the lists are aligned by
    episode (same seeds, same order). New code uses elo_ratings_paired,
    which verifies alignment by episode key instead of trusting it."""
    return _elo_from_aligned(results, k=k)


def elo_ratings_paired(rows_by_agent, k=32):
    """ELO from per-episode rows, aligned and validated by episode key."""
    scores, _ = aligned_paired_scores(rows_by_agent)
    return _elo_from_aligned(scores, k=k)


def _h2h_from_aligned(scores_by_name):
    names = list(scores_by_name.keys())
    n = len(next(iter(scores_by_name.values())))
    table = {}
    for a in names:
        table[a] = {}
        for b in names:
            if a == b:
                table[a][b] = None
                continue
            w = sum(1 for i in range(n) if scores_by_name[a][i] > scores_by_name[b][i])
            d = sum(1 for i in range(n) if scores_by_name[a][i] == scores_by_name[b][i])
            table[a][b] = {"win_rate": round(w / n, 3), "draw_rate": round(d / n, 3)}
    return table


def head_to_head(results):
    """Legacy list API: caller guarantees alignment (see elo_ratings)."""
    return _h2h_from_aligned(results)


def head_to_head_paired(rows_by_agent):
    """Head-to-head from per-episode rows, aligned by episode key."""
    scores, _ = aligned_paired_scores(rows_by_agent)
    return _h2h_from_aligned(scores)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info():
    """(commit, branch) of the checkout holding this script, or
    (None, None) when git is unavailable. Never fails the run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        branch = None
    return commit, branch


def build_eval_document(paired_rows, independent_rows, params, policies_meta):
    """Assemble a schema-v2 evaluation document.

    paired_rows: {name: rows} for the paired subset (all agents).
    independent_rows: {name: rows} for the independent pool (non-MCTS
      policies only; MCTS plays no independent pool).
    params: {"n": ..., "mcts_n": ..., "mcts_sims": ...}.
    policies_meta: {name: {"policy_id": str, "artifact": relpath-or-None}}.
      When artifact names a file under the runs dir, its SHA-256 is
      recorded; a missing artifact is a hard error, never a null hash.
    """
    paired_scores, paired_seeds = aligned_paired_scores(paired_rows)
    indep_seeds = sorted({r["seed"] for rows in independent_rows.values() for r in rows})
    indep_scores = {}
    for name, rows in independent_rows.items():
        m = rows_to_seed_map(rows)
        if sorted(m.keys()) != indep_seeds:
            raise ValueError(
                f"independent rows for {name} do not cover the full independent "
                f"seed set; refusing to build a document with ragged pools"
            )
        indep_scores[name] = [m[s] for s in indep_seeds]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/evaluate.py",
        "paired": {
            "seeds": paired_seeds,
            "scores": paired_scores,
            "summary": {k: summarize_full(v) for k, v in paired_scores.items()},
        },
        "independent": {
            "seeds": indep_seeds,
            "scores": indep_scores,
            "summary": {k: summarize_full(v) for k, v in indep_scores.items()},
        },
        "elo": elo_ratings_paired(paired_rows),
        "head_to_head": head_to_head_paired(paired_rows),
        "n_eval": params["n"],
        "mcts_n": params["mcts_n"],
        "mcts_sims": params["mcts_sims"],
    }
    # Legacy top-level keys, with exactly the contents readers already
    # expect: independent pool for the four non-MCTS policies, paired
    # pool for grandmaster; summaries rounded as before. Documented in
    # provenance so no reader mistakes them for a single pool.
    legacy_scores = dict(indep_scores)
    legacy_scores["grandmaster"] = paired_scores["grandmaster"]
    legacy_summary = {k: summarize(v) for k, v in indep_scores.items()}
    legacy_summary["grandmaster"] = summarize(paired_scores["grandmaster"])
    doc["scores"] = legacy_scores
    doc["summary"] = legacy_summary

    commit, branch = _git_info()
    artifacts = {}
    for name, meta in policies_meta.items():
        entry = {"policy_id": meta["policy_id"]}
        artifact = meta.get("artifact")
        if artifact is not None:
            path = os.path.join(_ROOT, artifact) if not os.path.isabs(artifact) \
                else artifact
            if not os.path.exists(path):
                raise ValueError(
                    f"policy artifact missing for {name}: {artifact}; refusing "
                    f"to record an evaluation against an artifact that is not present"
                )
            entry["artifact"] = artifact
            entry["artifact_sha256"] = sha256_file(path)
        artifacts[name] = entry
    doc["policies"] = artifacts
    doc["provenance"] = {
        "generated_from_commit": commit,
        "generated_on_branch": branch,
        "seed_convention": (
            "independent seeds are range(5000, 5000 + n_eval); paired seeds "
            "are the first mcts_n of that range, shared by every agent"
        ),
        "paired_derivation": (
            "elo and head_to_head are computed ONLY on paired.seeds; "
            "no mixed paired/independent rate exists in this document"
        ),
        "legacy_keys_note": (
            "top-level scores/summary hold the independent pool for "
            "random, greedy, rookie and pro, and the paired pool for "
            "grandmaster (MCTS plays no independent pool); use "
            "paired.* and independent.* for any cross-agent comparison"
        ),
        "supersedes": (
            "runs/eval_legacy.json preserves the pre-v2 artifact "
            "byte-identical; its elo/head_to_head were verified "
            "reproducible from its first-50 scores before replacement"
        ),
    }
    return doc


def validate_eval_document(doc):
    """Fail visibly on any inconsistency. Raises ValueError with a
    specific message; returns the document unchanged on success."""
    if not isinstance(doc, dict):
        raise ValueError("evaluation document must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evaluation schema_version {doc.get('schema_version')!r}; "
            f"this code reads schema_version {SCHEMA_VERSION}. Regenerate "
            f"runs/eval.json with scripts/evaluate.py"
        )
    for key in ("paired", "independent", "elo", "head_to_head", "policies"):
        if key not in doc:
            raise ValueError(f"evaluation document is missing required key {key!r}")
    paired = doc["paired"]
    for key in ("seeds", "scores", "summary"):
        if key not in paired:
            raise ValueError(f"evaluation document paired section is missing {key!r}")
    seeds = paired["seeds"]
    for name, scores in paired["scores"].items():
        if len(scores) != len(seeds):
            raise ValueError(
                f"paired scores for {name} have length {len(scores)} but "
                f"paired seeds have length {len(seeds)}; truncated artifacts "
                f"must fail, never silently compare ragged pools"
            )
    # Recompute paired summaries from the saved scores at full precision.
    for name, scores in paired["scores"].items():
        rebuilt = summarize_full(scores)
        stored = paired["summary"][name]
        for metric in ("mean", "std", "median"):
            if abs(rebuilt[metric] - stored[metric]) > 1e-10:
                raise ValueError(
                    f"paired summary for {name} does not rebuild from saved "
                    f"scores: {metric} stored {stored[metric]} vs rebuilt "
                    f"{rebuilt[metric]}"
                )
    # Recompute ELO and head-to-head from the paired scores.
    rows_by_agent = {
        n: [{"seed": s, "score": paired["scores"][n][i]} for i, s in enumerate(seeds)]
        for n in paired["scores"]
    }
    if elo_ratings_paired(rows_by_agent) != doc["elo"]:
        raise ValueError("stored elo does not recompute from paired scores")
    if head_to_head_paired(rows_by_agent) != doc["head_to_head"]:
        raise ValueError("stored head_to_head does not recompute from paired scores")
    return doc


def load_eval_document(path):
    """Load + validate an evaluation artifact. Corrupted or incomplete
    artifacts raise ValueError here, before any consumer reads a number."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return validate_eval_document(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mcts_n", type=int, default=50)
    ap.add_argument("--mcts_sims", type=int, default=80)
    args = ap.parse_args()

    with open(os.path.join(args.out, "qtable.json")) as f:
        qtable = json.load(f)
    policies = {
        "random": RandomPolicy(seed=999),
        "greedy": GreedyPolicy(),
        "rookie": QTablePolicy(qtable),
        "pro": DQNPolicy(os.path.join(args.out, "dqn_weights.json")),
    }
    # One run per policy over the full independent seed range. The paired
    # subset is the first mcts_n rows of that same run, so paired and
    # independent scores can never disagree about a shared episode.
    # (The old code ran paired and independent separately; for
    # deterministic policies the values are identical, and the single
    # run additionally guarantees it.)
    indep_seeds = list(range(5000, 5000 + args.n))
    paired_seeds = indep_seeds[:args.mcts_n]
    independent_rows = {}
    for name, pol in policies.items():
        print(f"evaluating {name} (independent, {args.n} episodes)...", flush=True)
        independent_rows[name] = run_episode_rows(pol.act, indep_seeds)
    print(f"evaluating grandmaster (MCTS, paired, {args.mcts_n} episodes)...", flush=True)
    mcts = MCTSPlayer(simulations=args.mcts_sims, seed=31337)
    paired_rows = {name: rows[:args.mcts_n] for name, rows in independent_rows.items()}
    paired_rows["grandmaster"] = run_episode_rows(mcts.act, paired_seeds)

    policies_meta = {
        "random": {"policy_id": "random-seed-999", "artifact": None},
        "greedy": {"policy_id": "greedy-heuristic", "artifact": None},
        "rookie": {"policy_id": "qtable", "artifact": "runs/qtable.json"},
        "pro": {"policy_id": "dqn-json-weights", "artifact": "runs/dqn_weights.json"},
        "grandmaster": {"policy_id": f"mcts-sims-{args.mcts_sims}-seed-31337",
                        "artifact": None},
    }
    doc = build_eval_document(
        paired_rows, independent_rows,
        {"n": args.n, "mcts_n": args.mcts_n, "mcts_sims": args.mcts_sims},
        policies_meta,
    )
    validate_eval_document(doc)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(doc, f)
    print("ELO (paired):", doc["elo"])
    for k, s in doc["paired"]["summary"].items():
        print(f"  {k:12s} paired mean={s['mean']:<6.2f} median={s['median']}")
    for k, s in doc["independent"]["summary"].items():
        print(f"  {k:12s} indep. mean={s['mean']:<6.2f} median={s['median']}")


if __name__ == "__main__":
    main()
