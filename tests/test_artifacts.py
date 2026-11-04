"""P1-08: data artifact integrity audit for the runs/ artifacts.

The manifest in runs/artifact_manifest.json pins the SHA-256 hash and the
production metadata (architecture, seed, steps, evaluation protocol) of the
qtable, the policy, the exported DQN weights, the evaluation results and
the DQN diagnosis results. These tests check the committed artifacts
against that manifest and exercise the verifier's failure modes on copies:
a hash mismatch, a policy that is not the exact argmax of the qtable, and
missing metadata must all be reported.
"""

import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
MANIFEST_PATH = os.path.join(RUNS, "artifact_manifest.json")

ARTIFACT_FILES = {
    "qtable": "qtable.json",
    "policy": "policy.json",
    "dqn_weights": "dqn_weights.json",
    "eval": "eval.json",
    "diagnosis": "da4_diagnosis.json",
}


def _load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _metadata_ok(meta):
    if not isinstance(meta, dict):
        return False
    if not isinstance(meta.get("architecture"), str) or not meta["architecture"]:
        return False
    seed = meta.get("seed")
    if isinstance(seed, list):
        if not seed or not all(isinstance(s, int) for s in seed):
            return False
    elif not isinstance(seed, int):
        return False
    if not isinstance(meta.get("steps"), int) or meta["steps"] <= 0:
        return False
    if not isinstance(meta.get("evaluation_protocol"), str) \
            or not meta["evaluation_protocol"]:
        return False
    return True


def _copy_runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ARTIFACT_FILES.values():
        shutil.copy(os.path.join(RUNS, name), runs / name)
    shutil.copy(MANIFEST_PATH, runs / "artifact_manifest.json")
    return tmp_path


def test_stage3_deeparcade_1_08_manifest_covers_all_artifacts():
    assert os.path.isfile(MANIFEST_PATH), "runs/artifact_manifest.json is missing"
    manifest = _load_manifest()
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, dict), "manifest has no artifacts mapping"
    for key, filename in ARTIFACT_FILES.items():
        assert key in artifacts, f"manifest is missing the {key!r} artifact"
        entry = artifacts[key]
        assert entry.get("path") == f"runs/{filename}"
        digest = entry.get("sha256")
        assert isinstance(digest, str) and len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
        assert _metadata_ok(entry.get("metadata")), \
            f"{key} metadata is missing architecture/seed/steps/evaluation_protocol"


def test_stage3_deeparcade_1_08_manifest_hashes_match_files():
    manifest = _load_manifest()
    for key, filename in ARTIFACT_FILES.items():
        entry = manifest["artifacts"][key]
        actual = _sha256(os.path.join(RUNS, filename))
        assert entry["sha256"] == actual, \
            f"{key}: manifest sha256 does not match runs/{filename}"


def test_stage3_deeparcade_1_08_policy_is_exact_argmax_of_qtable():
    with open(os.path.join(RUNS, "qtable.json"), encoding="utf-8") as f:
        qtable = json.load(f)
    with open(os.path.join(RUNS, "policy.json"), encoding="utf-8") as f:
        policy = json.load(f)
    assert policy["states"] == len(qtable)
    assert policy["actions"] == len(qtable[0])
    # Exact argmax with the first-max tie-break used when the policy was
    # exported from the table: a later equal value never replaces the
    # first maximum.
    expected = [row.index(max(row)) for row in qtable]
    assert policy["best_action"] == expected


def test_stage3_deeparcade_1_08_verify_accepts_current_artifacts():
    from scripts.artifact_manifest import verify_manifest
    assert verify_manifest(ROOT) == []


def test_stage3_deeparcade_1_08_verify_fails_on_hash_mismatch(tmp_path):
    from scripts.artifact_manifest import verify_manifest
    root = _copy_runs(tmp_path)
    with open(root / "runs" / "qtable.json", "ab") as f:
        f.write(b" ")
    problems = verify_manifest(str(root))
    assert any("sha256" in p for p in problems), problems


def test_stage3_deeparcade_1_08_verify_fails_on_non_argmax_policy(tmp_path):
    from scripts.artifact_manifest import verify_manifest
    root = _copy_runs(tmp_path)
    policy_path = root / "runs" / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["best_action"][0] = (policy["best_action"][0] + 1) % policy["actions"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    # Refresh the manifest hash so only the argmax violation remains.
    manifest_path = root / "runs" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["policy"]["sha256"] = _sha256(str(policy_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = verify_manifest(str(root))
    assert any("argmax" in p for p in problems), problems


def test_stage3_deeparcade_1_08_verify_fails_on_missing_metadata(tmp_path):
    from scripts.artifact_manifest import verify_manifest
    root = _copy_runs(tmp_path)
    manifest_path = root / "runs" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["qtable"]["metadata"]["seed"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = verify_manifest(str(root))
    assert any("metadata" in p for p in problems), problems
