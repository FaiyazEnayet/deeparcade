"""Artifact integrity manifest for the runs/ data artifacts.

runs/artifact_manifest.json pins the SHA-256 hash and the production
metadata (architecture, seed, steps, evaluation protocol) of the five
artifacts the shipped pages are built from: the Q-table, the exported
policy, the exported DQN weights, the evaluation results and the DQN
diagnosis results. Builders that inject these files into a page verify
the manifest first, so a stale, truncated or hand-edited artifact fails
the build instead of silently changing the published game or its numbers.

Regenerate after retraining or re-evaluating:

    python -m scripts.artifact_manifest
"""

from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MANIFEST_REL = os.path.join("runs", "artifact_manifest.json")

ARTIFACTS = {
    "qtable": "qtable.json",
    "policy": "policy.json",
    "dqn_weights": "dqn_weights.json",
    "eval": "eval.json",
    "diagnosis": "da4_diagnosis.json",
}

METADATA_FIELDS = ("architecture", "seed", "steps", "evaluation_protocol")


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _eval_protocol(ev):
    paired = ev["paired"]["seeds"]
    indep = ev["independent"]["seeds"]
    return (
        f"scripts/evaluate.py (schema v{ev['schema_version']}): "
        f"{len(indep)} independent episodes per agent on seeds "
        f"{indep[0]}-{indep[-1]}; Elo and head-to-head computed on the "
        f"paired subset, seeds {paired[0]}-{paired[-1]}; MCTS at "
        f"{ev['mcts_sims']} simulations per move"
    )


def build_manifest(root=ROOT):
    """Build the manifest dict from the artifacts and their meta files.

    Metadata is read from the recorded training/evaluation metadata
    (train_meta.json, dqn_meta.json, dqn_manifest.json, and the artifacts
    themselves) rather than restated by hand, so a regenerated manifest
    cannot drift from the runs that produced the artifacts.
    """
    runs = os.path.join(root, "runs")
    train_meta = _load_json(os.path.join(runs, "train_meta.json"))
    dqn_meta = _load_json(os.path.join(runs, "dqn_meta.json"))
    dqn_manifest = _load_json(os.path.join(runs, "dqn_manifest.json"))
    ev = _load_json(os.path.join(runs, "eval.json"))
    diag = _load_json(os.path.join(runs, "da4_diagnosis.json"))
    qtable = _load_json(os.path.join(runs, "qtable.json"))

    protocol = _eval_protocol(ev)
    layers = [pair[0] for pair in dqn_manifest["architecture"]]
    layers.append(dqn_manifest["architecture"][-1][1])
    encoder = dqn_manifest["encoder"]

    metadata = {
        "qtable": {
            "architecture": (
                f"tabular Q-table, {len(qtable)} states x "
                f"{len(qtable[0])} actions"
            ),
            "seed": train_meta["seed"],
            "steps": train_meta["episodes"],
            "steps_unit": "training episodes",
            "evaluation_protocol": protocol,
        },
        "policy": {
            "architecture": (
                "greedy policy: per-state argmax of runs/qtable.json, "
                "first-max tie-break"
            ),
            "seed": train_meta["seed"],
            "steps": train_meta["episodes"],
            "steps_unit": "training episodes",
            "evaluation_protocol": protocol,
        },
        "dqn_weights": {
            "architecture": (
                f"MLP {' -> '.join(str(n) for n in layers)}; state encoder "
                f"{encoder['name']} rev {encoder['revision']}, input "
                f"{encoder['input_dim']}"
            ),
            "seed": dqn_meta["seed"],
            "steps": dqn_meta["steps"],
            "steps_unit": "training steps",
            "evaluation_protocol": protocol,
        },
        "eval": {
            "architecture": (
                "evaluation record from scripts/evaluate.py: paired and "
                "independent episode scores, Elo and head-to-head for "
                "random, greedy, rookie, pro and grandmaster"
            ),
            "seed": ev["independent"]["seeds"][0],
            "steps": ev["n_eval"],
            "steps_unit": "independent episodes per agent",
            "evaluation_protocol": protocol,
        },
        "diagnosis": {
            "architecture": (
                "DQN ablation record from scripts/diagnose_dqn.py: "
                "variants " + ", ".join(diag["variants"])
            ),
            "seed": diag["seeds"],
            "steps": diag["steps_per_run"],
            "steps_unit": "training steps per run",
            "evaluation_protocol": (
                f"each variant trained for {diag['steps_per_run']} steps "
                f"per seed and evaluated on seeds "
                f"{diag['eval_seeds'][0]}-{diag['eval_seeds'][-1]} "
                f"({len(diag['eval_seeds'])} episodes)"
            ),
        },
    }

    artifacts = {}
    for key, filename in ARTIFACTS.items():
        artifacts[key] = {
            "path": f"runs/{filename}",
            "sha256": _sha256(os.path.join(runs, filename)),
            "metadata": metadata[key],
        }
    return {
        "schema_version": 1,
        "kind": "deeparcade-artifact-manifest",
        "artifacts": artifacts,
    }


def _metadata_problems(key, meta):
    if not isinstance(meta, dict):
        return [f"{key}: metadata is missing or not an object"]
    problems = []
    arch = meta.get("architecture")
    if not isinstance(arch, str) or not arch:
        problems.append(f"{key}: metadata architecture is missing or empty")
    seed = meta.get("seed")
    if isinstance(seed, list):
        if not seed or not all(isinstance(s, int) for s in seed):
            problems.append(f"{key}: metadata seed is missing or invalid")
    elif not isinstance(seed, int):
        problems.append(f"{key}: metadata seed is missing or invalid")
    steps = meta.get("steps")
    if not isinstance(steps, int) or steps <= 0:
        problems.append(f"{key}: metadata steps is missing or invalid")
    protocol = meta.get("evaluation_protocol")
    if not isinstance(protocol, str) or not protocol:
        problems.append(
            f"{key}: metadata evaluation_protocol is missing or empty")
    return problems


def verify_manifest(root=ROOT):
    """Return the list of integrity problems for the artifacts under root.

    An empty list means every artifact matches its pinned SHA-256 hash,
    carries complete metadata, and the policy is the exact argmax of the
    qtable. Nothing is repaired: drift is reported, never fixed silently.
    """
    problems = []
    manifest_path = os.path.join(root, MANIFEST_REL)
    if not os.path.isfile(manifest_path):
        return [f"manifest missing: {MANIFEST_REL}"]
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError) as exc:
        return [f"manifest is not valid JSON: {exc}"]
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return ["manifest has no artifacts mapping"]

    runs = os.path.join(root, "runs")
    for key, filename in ARTIFACTS.items():
        entry = artifacts.get(key)
        if entry is None:
            problems.append(f"{key}: missing from manifest")
            continue
        if entry.get("path") != f"runs/{filename}":
            problems.append(
                f"{key}: manifest path is {entry.get('path')!r}, "
                f"expected 'runs/{filename}'")
        fpath = os.path.join(runs, filename)
        if not os.path.isfile(fpath):
            problems.append(f"{key}: artifact file missing: runs/{filename}")
            continue
        if entry.get("sha256") != _sha256(fpath):
            problems.append(f"{key}: sha256 mismatch for runs/{filename}")
        problems.extend(_metadata_problems(key, entry.get("metadata")))

    try:
        qtable = _load_json(os.path.join(runs, "qtable.json"))
        policy = _load_json(os.path.join(runs, "policy.json"))
        expected = [row.index(max(row)) for row in qtable]
        if policy.get("best_action") != expected:
            problems.append(
                "policy: best_action is not the exact argmax of "
                "runs/qtable.json")
        if policy.get("states") != len(qtable) \
                or policy.get("actions") != len(qtable[0]):
            problems.append(
                "policy: states/actions do not match the qtable dimensions")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        problems.append(f"policy/qtable: could not verify argmax ({exc})")
    return problems


def main():
    manifest = build_manifest()
    out = os.path.join(ROOT, MANIFEST_REL)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"manifest -> {out}")
    problems = verify_manifest()
    if problems:
        raise SystemExit("verification failed: " + "; ".join(problems))
    print("verification: all artifacts match the manifest")


if __name__ == "__main__":
    main()
