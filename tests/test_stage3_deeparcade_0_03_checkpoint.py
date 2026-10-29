"""Stage 3 P0-03: DQN checkpoint reproducibility.

Covers the versioned checkpoint schema for the DQN weights export, the
strict inverse loader back to PyTorch, the checkpoint manifest for the
committed artifact, and browser parity on frozen observations.

The parity fixture (tests/fixtures/dqn_parity_observations.json) was
generated with pre-existing code only: greedy rollouts of the committed
DQN policy on SnakeEnv(12, 12), seeds 101..112, capturing encode_dqn
vectors at steps 3 and 10 (nonterminal) and at episode end (terminal).
36 observations: 24 nonterminal, 12 terminal.

Declared float regimes (recorded in the checkpoint schema):
  - weights are stored as float32 values;
  - parity between the PyTorch policy and the browser export is
    evaluated in float64. The browser (web/game.html dqnAct) accumulates
    in float64 and stores hidden activations in Float32Arrays; the
    PyTorch policy evaluated in float32 differs from the browser export
    by up to 3.546e-06 on this fixture (measured 2026-10-29) purely from
    float32 accumulation, with greedy actions still agreeing on all 36
    observations. In the declared float64 regime the measured max abs
    logit difference is 6.948e-07, inside the 1e-6 acceptance bound.
"""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "runs", "dqn_weights.json")
META = os.path.join(ROOT, "runs", "dqn_meta.json")
MANIFEST = os.path.join(ROOT, "runs", "dqn_manifest.json")
EVAL = os.path.join(ROOT, "runs", "eval.json")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "dqn_parity_observations.json")

from scripts.train_dqn import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    browser_export_logits,
    build_checkpoint_manifest,
    export_weights,
    load_checkpoint,
    validate_checkpoint_document,
    weights_sha256,
    DQN,
)


def _fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _observations():
    fx = _fixture()
    return np.array([o["vector"] for o in fx["observations"]], dtype=np.float64)


def _torch_logits_float64(net, observations):
    net64 = copy.deepcopy(net).double().eval()
    with torch.no_grad():
        return net64(torch.tensor(observations, dtype=torch.float64)).numpy()


def test_stage3_deeparcade_0_03_committed_checkpoint_loads_and_matches_manifest():
    """The committed exported JSON loads through the strict loader, and
    its hashes agree with the manifest and with the hash P0-01 recorded
    in runs/eval.json."""
    import hashlib

    net, info = load_checkpoint(WEIGHTS)
    assert info["schema_version"] == 0  # legacy unversioned export
    with open(WEIGHTS, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["artifact_sha256"] == file_sha
    assert info["weights_sha256"] == manifest["weights_sha256"]
    with open(EVAL, encoding="utf-8") as f:
        ev = json.load(f)
    assert manifest["artifact_sha256"] == ev["policies"]["pro"]["artifact_sha256"]
    out = _torch_logits_float64(net, _observations()[:1])
    assert out.shape == (1, 3)
    assert np.isfinite(out).all()


def test_stage3_deeparcade_0_03_manifest_rebuilds_from_committed_artifacts():
    """Every derived field of the committed manifest recomputes exactly
    from runs/dqn_weights.json + runs/dqn_meta.json. The one recorded
    historical fact (the commit that introduced the weights file) is fed
    back from the manifest itself and separately pinned to its known
    value."""
    with open(MANIFEST, encoding="utf-8") as f:
        committed = json.load(f)
    assert committed["provenance"]["weights_file_introduced_in_commit"] == \
        "8680dbf7cde47091a3ce1bafa811c6ef4df5d03b"
    rebuilt = build_checkpoint_manifest(
        WEIGHTS, META,
        weights_file_commit=committed["provenance"]["weights_file_introduced_in_commit"],
    )
    assert rebuilt == committed


def test_stage3_deeparcade_0_03_versioned_export_schema_and_roundtrip():
    """A fresh export carries the versioned schema with metadata and a
    verifiable weights hash, keeps the legacy top-level keys the game
    build reads, and round-trips to an identical PyTorch network."""
    torch.manual_seed(20261029)
    net = DQN()
    doc = export_weights(
        net,
        training_args={"steps": 0, "seed": 1},
        data_key_range={"episode_seeds": [1, 1]},
        code_commit="test-commit-sentinel",
    )
    assert doc["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 1
    assert doc["kind"] == "deeparcade-dqn-checkpoint"
    assert doc["float_regime"] == "float32"
    assert doc["parity_float_regime"] == "float64"
    assert doc["encoder"]["name"] == "encode_dqn"
    assert doc["input_dim"] == 436 and len(doc["layers"]) == 3
    assert doc["weights_sha256"] == weights_sha256(doc["layers"])
    assert doc["metadata"]["training_args"] == {"steps": 0, "seed": 1}
    assert doc["metadata"]["data_key_range"] == {"episode_seeds": [1, 1]}
    assert doc["metadata"]["code_commit"] == "test-commit-sentinel"
    validate_checkpoint_document(doc)
    net2, info = load_checkpoint(doc)
    assert info["schema_version"] == 1
    assert info["weights_sha256"] == doc["weights_sha256"]
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), net2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    # No training metadata supplied: fields are present and null, never absent.
    bare = export_weights(DQN())
    assert bare["metadata"] == {"training_args": None, "data_key_range": None,
                                "code_commit": None}
    validate_checkpoint_document(bare)


def test_stage3_deeparcade_0_03_clean_process_save_and_reload():
    """Train a small learned checkpoint, save the versioned export, and
    reload it in a separate clean process. SHA and metadata must match,
    and the reloaded logits must reproduce the in-process ones."""
    from scripts.train_dqn import train_dqn

    net, _, _, meta = train_dqn(steps=1500, seed=123, verbose=False)
    doc = export_weights(
        net,
        training_args={"steps": 1500, "seed": 123},
        data_key_range={"episode_seeds": [123, 123 + meta["episodes"] - 1]},
        code_commit="clean-process-test",
    )

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        snippet = (
            "import json, sys, torch\n"
            f"sys.path.insert(0, {ROOT!r}); sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
            "from scripts.train_dqn import load_checkpoint, browser_export_logits\n"
            "net, info = load_checkpoint(sys.argv[1])\n"
            "fx = json.load(open(sys.argv[2]))\n"
            "obs = [o['vector'] for o in fx['observations']]\n"
            "net64 = net.double().eval()\n"
            "with torch.no_grad():\n"
            "    tl = net64(torch.tensor(obs, dtype=torch.float64)).tolist()\n"
            "doc = json.load(open(sys.argv[1]))\n"
            "bl = browser_export_logits(doc, obs).tolist()\n"
            "print(json.dumps({'info': info, 'torch_logits': tl, 'browser_logits': bl}))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + os.path.join(ROOT, "src")
        proc = subprocess.run(
            [sys.executable, "-c", snippet, path, FIXTURE],
            capture_output=True, text=True, env=env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["info"]["weights_sha256"] == doc["weights_sha256"]
    assert out["info"]["metadata"] == doc["metadata"]
    net2, _ = load_checkpoint(doc)
    mine = _torch_logits_float64(net2, _observations())
    theirs = np.array(out["torch_logits"])
    assert np.abs(mine - theirs).max() <= 1e-12
    browser = np.array(out["browser_logits"])
    assert np.abs(browser - theirs).max() <= 1e-6


def test_stage3_deeparcade_0_03_browser_parity_frozen_observations():
    """PyTorch policy logits vs browser-export policy logits on the
    frozen fixture, in the declared float64 parity regime."""
    fx = _fixture()
    rows = fx["observations"]
    assert len(rows) >= 20
    assert sum(1 for o in rows if o["terminal"]) >= 5
    assert sum(1 for o in rows if not o["terminal"]) >= 5
    import hashlib
    with open(WEIGHTS, "rb") as f:
        assert fx["weights_artifact_sha256"] == hashlib.sha256(f.read()).hexdigest()

    with open(WEIGHTS, encoding="utf-8") as f:
        doc = json.load(f)
    net, _ = load_checkpoint(WEIGHTS)
    obs = _observations()
    torch_logits = _torch_logits_float64(net, obs)
    browser_logits = browser_export_logits(doc, obs)
    diff = np.abs(torch_logits - browser_logits).max()
    assert diff <= 1e-6, f"max abs logit diff {diff:.3e} exceeds 1e-6 (float64 regime)"

    # Exact greedy action equality on every non-tied case (top-2 gap in
    # the browser logits larger than the 1e-6 tolerance).
    order = np.sort(browser_logits, axis=1)
    gap = order[:, 2] - order[:, 1]
    nontied = gap > 1e-6
    assert nontied.sum() >= 20
    assert (torch_logits.argmax(1)[nontied] == browser_logits.argmax(1)[nontied]).all()

    # The evaluator's policy loads through the same strict loader and
    # holds exactly the same weights.
    from scripts.evaluate import DQNPolicy
    pol = DQNPolicy(WEIGHTS)
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), pol.net.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_stage3_deeparcade_0_03_corrupt_checkpoint_fails_closed():
    """Every corruption of a legacy or versioned checkpoint raises
    CheckpointError before any tensor is built."""
    with open(WEIGHTS, encoding="utf-8") as f:
        legacy = json.load(f)
    versioned = export_weights(DQN())

    def variants(base):
        out = {}
        d = copy.deepcopy(base); d["layers"][0]["w"][0][0] = float("nan"); out["nan weight"] = d
        d = copy.deepcopy(base); d["layers"][1]["b"][0] = float("inf"); out["inf bias"] = d
        d = copy.deepcopy(base); d["layers"][0]["w"][0] = d["layers"][0]["w"][0][:-1]; out["truncated row"] = d
        d = copy.deepcopy(base); d["layers"] = d["layers"][:2]; out["missing layer"] = d
        d = copy.deepcopy(base); d["layers"][2]["b"] = [0.0, 0.0]; out["short bias"] = d
        d = copy.deepcopy(base); d["layers"][0]["w"][0][0] = "1.0"; out["string weight"] = d
        d = copy.deepcopy(base); d["layers"][0]["w"][0][0] = True; out["bool weight"] = d
        d = copy.deepcopy(base); d["input_dim"] = 435; out["input_dim mismatch"] = d
        d = copy.deepcopy(base); del d["layers"]; out["no layers key"] = d
        return out

    for base_name, base in [("legacy", legacy), ("versioned", versioned)]:
        for name, bad in variants(base).items():
            with pytest.raises(CheckpointError):
                validate_checkpoint_document(bad)
            with pytest.raises(CheckpointError):
                load_checkpoint(bad)

    v = copy.deepcopy(versioned); v["weights_sha256"] = "0" * 64
    with pytest.raises(CheckpointError):
        validate_checkpoint_document(v)
    v = copy.deepcopy(versioned); v["schema_version"] = 2
    with pytest.raises(CheckpointError):
        validate_checkpoint_document(v)
    v = copy.deepcopy(versioned); v["encoder"]["revision"] = 999
    with pytest.raises(CheckpointError):
        validate_checkpoint_document(v)
    v = copy.deepcopy(versioned); v["kind"] = "something-else"
    with pytest.raises(CheckpointError):
        validate_checkpoint_document(v)

    # A truncated JSON file on disk also fails closed, as CheckpointError.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bad.json")
        with open(path, "w") as f:
            f.write('{"input_dim": 436, "layers": [{"w": [[0.1], ')
        with pytest.raises(CheckpointError):
            load_checkpoint(path)
        # A NaN literal inside a real file is caught by the finite check:
        # swap the first weight value after the "layers" key for NaN.
        import re
        with open(path, "w") as f:
            json.dump(legacy, f)
        text = open(path).read()
        head, sep, tail = text.partition('"layers"')
        assert sep
        tail = re.sub(r"-?\d+\.?\d*(?:e-?\d+)?", "NaN", tail, count=1)
        with open(path, "w") as f:
            f.write(head + sep + tail)
        with pytest.raises(CheckpointError):
            load_checkpoint(path)
