"""Compact model cards for the deeparcade policies (Stage 3 DA7).

One document, runs/model_cards.json, ties every evaluated agent to its
lineage in a verifiable place:

  - learned policies (rookie, pro): the training record (environment
    and encoding version, steps, final reward and score, seed), the
    checkpoint artifact and the browser export, each pinned by SHA-256,
    and the evaluation figures together with the artifact hash the
    evaluator itself recorded, so the card's checkpoint is provably
    the evaluated model. The pro card also names the DQN parity
    fixture and the weights hash recorded in it, tying the same
    artifact to the JS parity checks.
  - search (grandmaster), heuristic (greedy) and baseline (random)
    policies carry no checkpoint and no training record; the cards
    say so explicitly instead of implying one.
  - every card records a measured inference cost (mean wall time per
    decision, with the method), measured at build time on seeded
    mid-game positions.

Commands (from the repo root):
  python -m scripts.model_cards build    rewrite runs/model_cards.json
  python -m scripts.model_cards verify   recompute and compare; exit 1
                                         on any drift
  python -m scripts.model_cards smoke    one-command small smoke
                                         training: tiny Q-learning and
                                         DQN runs in a temp directory,
                                         exported, carded and verified
                                         end to end

Verification rebuilds the cards from the artifacts on disk and
compares field by field. The only fields exempt from equality are the
measured inference times (timings are build-machine facts, checked
for presence and plausibility instead) and the build commit. Any
other drift, a tampered hash, a stale evaluation figure, a missing
field, is reported as a problem.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, ".."))

try:
    import narcade  # noqa: F401
except ImportError:  # running uninstalled: use this repo's sources
    sys.path.insert(0, os.path.join(DEFAULT_ROOT, "src"))
    sys.path.insert(0, DEFAULT_ROOT)

MODEL_CARDS_KIND = "deeparcade-model-cards"
MODEL_CARDS_SCHEMA_VERSION = 1
CARD_ORDER = ["random", "greedy", "rookie", "pro", "grandmaster"]
POLICY_NAMES = {"random": "Random", "greedy": "Greedy",
                "rookie": "Rookie", "pro": "Pro",
                "grandmaster": "Grandmaster"}
POLICY_KINDS = {"random": "baseline", "greedy": "heuristic",
                "rookie": "learned", "pro": "learned",
                "grandmaster": "search"}
POLICY_ALGORITHMS = {
    "random": "uniform random action selection",
    "greedy": "heuristic: move toward the food, avoid instant death",
    "rookie": "tabular Q-learning",
    "pro": "deep Q-network (DQN) with experience replay and a target "
           "network",
    "grandmaster": "Monte Carlo tree search (UCT) with greedy rollouts",
}
TABULAR_ENCODING = {
    "name": "encode_state",
    "location": "src/narcade/snake.py",
    "states": 32,
    "actions": 3,
    "layout": ("danger straight/left/right bits, then food direction "
               "relative to heading (0 ahead, 1 right, 2 left, 3 behind)"),
}
EVAL_PROTOCOL = (
    "scripts/evaluate.py (schema v2): 500 independent episodes per "
    "agent on seeds 5000-5499; Elo and head-to-head computed on the "
    "paired subset, seeds 5000-5049; MCTS at 80 simulations per move")
PARITY_FIXTURE = "tests/fixtures/dqn_parity_observations.json"
MEASUREMENT_SIMS = 80  # the recorded evaluation's MCTS budget per move


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_commit(root):
    """HEAD commit of the checkout at root, or None when unavailable.
    Never fails the build."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------
# Measured inference cost
# ---------------------------------------------------------------------

def _snapshot_envs(max_snapshots=12):
    """Deterministic mid-game positions: greedy rollouts on SnakeEnv
    seeds 101..112, snapshotted at steps 6, 12 and 18. The same set is
    timed for every policy, so the recorded costs are comparable."""
    from narcade.policies import GreedyPolicy
    from narcade.snake import SnakeEnv

    greedy = GreedyPolicy()
    envs = []
    for seed in range(101, 113):
        env = SnakeEnv(width=12, height=12, seed=seed)
        for step in range(1, 41):
            _, _, done, _ = env.step(greedy.act(env))
            if step in (6, 12, 18) and not done:
                envs.append(copy.deepcopy(env))
            if done:
                break
        if len(envs) >= max_snapshots:
            break
    return envs[:max_snapshots]


def _time_act(act, envs, rounds):
    for env in envs:  # warmup, untimed
        act(env)
    t0 = time.perf_counter()
    count = 0
    for _ in range(rounds):
        for env in envs:
            act(env)
            count += 1
    return (time.perf_counter() - t0) * 1000.0 / count, count


def _measure_inference(runs):
    """Mean wall time per decision for each evaluated policy, timed on
    the shared snapshot positions through the same code paths the
    evaluation uses."""
    from narcade.mcts import MCTSPlayer
    from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
    from scripts.evaluate import DQNPolicy

    envs = _snapshot_envs()
    n = len(envs)
    base = f"over {n} seeded mid-game SnakeEnv snapshots"
    qtable = _load_json(os.path.join(runs, "qtable.json"))
    dqn = DQNPolicy(os.path.join(runs, "dqn_weights.json"))
    mcts_envs = envs[:4]
    timed = {
        "random": (RandomPolicy(seed=999).act, envs, 25,
                   f"mean wall time of narcade.policies.RandomPolicy.act "
                   f"{base}, 25 rounds"),
        "greedy": (GreedyPolicy().act, envs, 25,
                   f"mean wall time of narcade.policies.GreedyPolicy.act "
                   f"{base}, 25 rounds"),
        "rookie": (QTablePolicy(qtable).act, envs, 25,
                   f"mean wall time of narcade.policies.QTablePolicy.act "
                   f"(encode_state + argmax over the Q-table) {base}, "
                   f"25 rounds"),
        "pro": (dqn.act, envs, 25,
                f"mean wall time of scripts.evaluate.DQNPolicy.act "
                f"(encode_dqn + float32 forward + argmax) {base}, "
                f"25 rounds"),
        "grandmaster": (
            MCTSPlayer(simulations=MEASUREMENT_SIMS, seed=31337).act,
            mcts_envs, 1,
            f"mean wall time of narcade.mcts.MCTSPlayer.act at "
            f"{MEASUREMENT_SIMS} simulations per move over "
            f"{len(mcts_envs)} seeded mid-game SnakeEnv snapshots, "
            f"1 round"),
    }
    out = {}
    for card_id, (act, pool, rounds, method) in timed.items():
        ms, count = _time_act(act, pool, rounds)
        out[card_id] = {"ms_per_decision": round(ms, 4),
                        "decisions_timed": count, "method": method}
    return out


# ---------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------

def _last_or_none(values):
    """Last entry of a learning-curve series, or None when the run
    was too short to complete a single 100-episode window (the curve
    only records one point per 100 episodes)."""
    return values[-1] if values else None


def _evaluation_block(eval_doc, eval_sha, card_id):
    if eval_doc is None:
        return None
    pol = eval_doc["policies"][card_id]
    summ = eval_doc["summary"][card_id]
    return {
        "artifact": "runs/eval.json",
        "artifact_sha256": eval_sha,
        "schema_version": eval_doc["schema_version"],
        "policy_id": pol["policy_id"],
        "evaluated_artifact": pol.get("artifact"),
        "evaluated_artifact_sha256": pol.get("artifact_sha256"),
        "pool": "paired" if card_id == "grandmaster" else "independent",
        "n": summ["n"],
        "mean": summ["mean"],
        "median": summ["median"],
        "std": summ["std"],
        "elo": eval_doc["elo"][card_id],
    }


def build_model_cards(root):
    """Build the model cards document from the artifacts under root.

    Every value is read or recomputed from the runs/ artifacts (and
    the parity fixture) at build time; nothing is carried over from a
    previous cards document."""
    runs = os.path.join(root, "runs")
    train_meta = _load_json(os.path.join(runs, "train_meta.json"))
    dqn_meta = _load_json(os.path.join(runs, "dqn_meta.json"))
    dqn_manifest = _load_json(os.path.join(runs, "dqn_manifest.json"))
    curve = _load_json(os.path.join(runs, "curve.json"))
    dqn_curve = _load_json(os.path.join(runs, "dqn_curve.json"))
    eval_path = os.path.join(runs, "eval.json")
    eval_doc = _load_json(eval_path) if os.path.exists(eval_path) else None
    eval_sha = _sha256_file(eval_path) if eval_doc is not None else None
    inference = _measure_inference(runs)

    def card(card_id):
        return {"id": card_id, "name": POLICY_NAMES[card_id],
                "kind": POLICY_KINDS[card_id],
                "algorithm": POLICY_ALGORITHMS[card_id],
                "encoding": None, "training": None, "checkpoint": None,
                "export": None,
                "evaluation": _evaluation_block(eval_doc, eval_sha,
                                                card_id),
                "inference_cost": inference[card_id]}

    cards = {cid: card(cid) for cid in CARD_ORDER}

    rookie = cards["rookie"]
    rookie["encoding"] = dict(TABULAR_ENCODING)
    rookie["training"] = {
        "trainer": "scripts/train.py",
        "steps": train_meta["episodes"],
        "steps_unit": "training episodes",
        "episodes": train_meta["episodes"],
        "seed": train_meta["seed"],
        "eps_start": train_meta["eps_start"],
        "eps_end": train_meta["eps_end"],
        "seconds": train_meta["seconds"],
        "final_mean100_score": train_meta["final_mean100_score"],
        "final_mean_reward": _last_or_none(curve["mean_reward"]),
        "final_elo": train_meta["final_elo"],
    }
    rookie["checkpoint"] = {
        "artifact": "runs/qtable.json",
        "sha256": _sha256_file(os.path.join(runs, "qtable.json")),
        "format": "JSON array, 32 states x 3 actions of Q-values",
    }
    rookie["export"] = {
        "artifact": "runs/policy.json",
        "sha256": _sha256_file(os.path.join(runs, "policy.json")),
        "format": ("greedy argmax of the Q-table, first-max tie-break; "
                   "injected into the browser game by "
                   "scripts/make_game.py"),
    }

    pro = cards["pro"]
    pro["encoding"] = dict(dqn_manifest["encoder"])
    pro["training"] = {
        "trainer": "scripts/train_dqn.py",
        "steps": dqn_meta["steps"],
        "steps_unit": "training steps",
        "episodes": dqn_meta["episodes"],
        "seed": dqn_meta["seed"],
        "seconds": dqn_meta["seconds"],
        "final_mean100_score": dqn_meta["final_mean100"],
        "final_mean_reward": _last_or_none(dqn_curve["mean_reward"]),
        "data_key_range": dqn_manifest["data_key_range"],
    }
    pro["checkpoint"] = {
        "artifact": "runs/dqn_weights.json",
        "sha256": _sha256_file(os.path.join(runs, "dqn_weights.json")),
        "weights_sha256": dqn_manifest["weights_sha256"],
        "checkpoint_schema_version":
            dqn_manifest["checkpoint_schema_version"],
        "manifest": "runs/dqn_manifest.json",
        "float_regime": dqn_manifest["float_regime"],
        "parity_float_regime": dqn_manifest["parity_float_regime"],
        "format": ("JSON weights export (input_dim + layers); the "
                   "checkpoint file is itself the browser export"),
    }
    pro["export"] = {
        "artifact": "runs/dqn_weights.json",
        "sha256": pro["checkpoint"]["sha256"],
        "format": ("the checkpoint file, injected into the browser "
                   "game by scripts/make_game.py"),
    }
    fixture_path = os.path.join(root, PARITY_FIXTURE)
    if os.path.exists(fixture_path):
        fixture = _load_json(fixture_path)
        pro["parity_fixture"] = {
            "path": PARITY_FIXTURE,
            "sha256": _sha256_file(fixture_path),
            "weights_artifact_sha256":
                fixture["weights_artifact_sha256"],
            "observations": len(fixture["observations"]),
        }
    else:
        pro["parity_fixture"] = None

    if eval_doc is not None:
        cards["grandmaster"]["search"] = {
            "simulations_per_move": eval_doc["mcts_sims"],
            "episodes_evaluated": eval_doc["mcts_n"],
        }

    return {
        "schema_version": MODEL_CARDS_SCHEMA_VERSION,
        "kind": MODEL_CARDS_KIND,
        "generated_by": "scripts/model_cards.py",
        "environment": {
            "name": "SnakeEnv",
            "implementation": "src/narcade/snake.py",
            "width": train_meta["width"],
            "height": train_meta["height"],
            "code_commit": _git_commit(root),
        },
        "evaluation": None if eval_doc is None else {
            "artifact": "runs/eval.json",
            "sha256": eval_sha,
            "schema_version": eval_doc["schema_version"],
            "protocol": EVAL_PROTOCOL,
        },
        "cards": [cards[cid] for cid in CARD_ORDER],
    }


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def _diff(path, stored, current, problems):
    if path.endswith("environment.code_commit") or \
            path.endswith("inference_cost.ms_per_decision"):
        return  # build-time facts, checked separately for sanity
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in sorted(set(stored) | set(current)):
            if key not in stored:
                problems.append(f"{path}.{key}: missing from the "
                                f"stored model cards")
            elif key not in current:
                problems.append(f"{path}.{key}: not a current card "
                                f"field")
            else:
                _diff(f"{path}.{key}", stored[key], current[key],
                      problems)
    elif isinstance(stored, list) and isinstance(current, list):
        if len(stored) != len(current):
            problems.append(f"{path}: stored {len(stored)} entries, "
                            f"current {len(current)}")
        else:
            for i, (a, b) in enumerate(zip(stored, current)):
                _diff(f"{path}[{i}]", a, b, problems)
    elif stored != current:
        problems.append(f"{path}: stored {stored!r} != current "
                        f"{current!r}")


def verify_model_cards(root):
    """Recompute the lineage under root and compare it with the
    stored runs/model_cards.json. Returns a list of problems; an
    empty list means the cards verify."""
    cards_path = os.path.join(root, "runs", "model_cards.json")
    if not os.path.exists(cards_path):
        return ["runs/model_cards.json is missing; build it with "
                "python -m scripts.model_cards build"]
    try:
        stored = _load_json(cards_path)
    except (OSError, json.JSONDecodeError) as e:
        return [f"runs/model_cards.json is not readable JSON: {e}"]
    problems = []
    if stored.get("schema_version") != MODEL_CARDS_SCHEMA_VERSION:
        problems.append(
            f"model cards schema_version "
            f"{stored.get('schema_version')!r} != "
            f"{MODEL_CARDS_SCHEMA_VERSION}")
    if stored.get("kind") != MODEL_CARDS_KIND:
        problems.append(f"model cards kind {stored.get('kind')!r} != "
                        f"{MODEL_CARDS_KIND!r}")
    # The build commit is a build-time fact: it records the commit
    # the cards were built from, so it is checked for presence and
    # plausibility, never for equality with the current HEAD (the
    # document cannot name the commit that contains it). A null is
    # accepted: it is what a build outside a git checkout records.
    env = stored.get("environment")
    if not isinstance(env, dict) or "code_commit" not in env:
        problems.append("environment.code_commit is missing from "
                        "the stored model cards")
    else:
        commit = env["code_commit"]
        if commit is not None and not (
                isinstance(commit, str) and len(commit) == 40 and
                all(c in "0123456789abcdefABCDEF" for c in commit)):
            problems.append(
                f"environment.code_commit {commit!r} is not a "
                f"40-character hex commit hash")
    ids = [c.get("id") for c in stored.get("cards", [])]
    if ids != CARD_ORDER:
        problems.append(f"model cards must cover {CARD_ORDER} in "
                        f"order, found {ids}")
        return problems
    for card in stored["cards"]:
        cost = card.get("inference_cost") or {}
        ms = cost.get("ms_per_decision")
        if not isinstance(ms, (int, float)) or isinstance(ms, bool) \
                or not math.isfinite(ms) or ms <= 0:
            problems.append(
                f"card {card['id']}: inference_cost.ms_per_decision "
                f"must be a positive measured number, found {ms!r}")
        if not cost.get("method"):
            problems.append(f"card {card['id']}: inference_cost.method "
                            f"is missing")
        for block_name in ("checkpoint", "export"):
            block = card.get(block_name)
            if block is None:
                continue
            artifact = os.path.join(root, block["artifact"])
            if not os.path.exists(artifact):
                problems.append(f"card {card['id']}: {block_name} "
                                f"artifact {block['artifact']} is "
                                f"missing")
            elif _sha256_file(artifact) != block["sha256"]:
                problems.append(
                    f"card {card['id']}: {block_name} "
                    f"{block['artifact']} sha256 on disk does not "
                    f"match the card ({block['sha256']})")
    rebuilt = build_model_cards(root)
    _diff("cards", stored, rebuilt, problems)
    return problems


# ---------------------------------------------------------------------
# Smoke run
# ---------------------------------------------------------------------

def run_smoke(root, q_episodes=240, dqn_steps=2100, seed=3):
    """One-command small smoke training: train a tiny Q-learning
    agent and a tiny DQN in root/runs, export both, build the model
    cards over them and verify the whole chain. Returns a summary
    dict; summary["problems"] is empty when the chain verifies."""
    from narcade.policies import QTablePolicy
    from scripts.train import train as train_qtable
    from scripts.train_dqn import (build_checkpoint_manifest,
                                   export_weights, train_dqn)

    runs = os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)

    agent, scores, rewards, _elo_hist, meta = train_qtable(
        episodes=q_episodes, seed=seed, checkpoint_every=q_episodes + 1,
        elo_games=1, verbose=False)
    with open(os.path.join(runs, "qtable.json"), "w") as f:
        json.dump(agent.q, f)
    with open(os.path.join(runs, "policy.json"), "w") as f:
        json.dump(QTablePolicy(agent.q).to_jsonable(), f)
    with open(os.path.join(runs, "curve.json"), "w") as f:
        json.dump({"mean_score": scores, "mean_reward": rewards}, f)
    with open(os.path.join(runs, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    net, dqn_scores, dqn_rewards, dqn_meta = train_dqn(
        steps=dqn_steps, seed=seed, verbose=False)
    doc = export_weights(
        net,
        training_args={"steps": dqn_steps, "seed": seed},
        data_key_range={"episode_seeds":
                        [seed, seed + dqn_meta["episodes"] - 1]},
        code_commit=None)
    weights_path = os.path.join(runs, "dqn_weights.json")
    meta_path = os.path.join(runs, "dqn_meta.json")
    with open(weights_path, "w") as f:
        json.dump(doc, f)
    with open(os.path.join(runs, "dqn_curve.json"), "w") as f:
        json.dump({"mean_score": dqn_scores,
                   "mean_reward": dqn_rewards}, f)
    with open(meta_path, "w") as f:
        json.dump(dqn_meta, f, indent=2)
    manifest = build_checkpoint_manifest(weights_path, meta_path,
                                         weights_file_commit=None)
    with open(os.path.join(runs, "dqn_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    cards = build_model_cards(root)
    with open(os.path.join(runs, "model_cards.json"), "w") as f:
        json.dump(cards, f, indent=2)
    problems = verify_model_cards(root)
    return {"q_episodes": q_episodes, "dqn_steps": dqn_steps,
            "cards": len(cards["cards"]), "problems": problems}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build, verify or smoke-test the deeparcade "
                    "model cards.")
    ap.add_argument("command", choices=("build", "verify", "smoke"))
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="repo root holding runs/ (default: this repo)")
    ap.add_argument("--out", default=None,
                    help="smoke only: output root (default: a temp "
                         "directory)")
    args = ap.parse_args()

    if args.command == "build":
        cards = build_model_cards(args.root)
        out = os.path.join(args.root, "runs", "model_cards.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)
        print(f"model cards -> {out} ({len(cards['cards'])} cards)")
        return 0
    if args.command == "verify":
        problems = verify_model_cards(args.root)
        if problems:
            for p in problems:
                print(f"PROBLEM: {p}")
            return 1
        print("model cards verify clean")
        return 0
    out = args.out or tempfile.mkdtemp(prefix="deeparcade-smoke-")
    summary = run_smoke(out)
    print(f"smoke root: {out}")
    print(f"trained q-learning for {summary['q_episodes']} episodes "
          f"and a DQN for {summary['dqn_steps']} steps; built "
          f"{summary['cards']} model cards")
    if summary["problems"]:
        for p in summary["problems"]:
            print(f"PROBLEM: {p}")
        return 1
    print("smoke OK: cards verify end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
