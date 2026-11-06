"""Stage 3 DA7: compact model cards and reproducible checkpoints.

The defect: the lineage of the browser weights, the trainer checkpoints
and the evaluation record is split across files that each cover only
part of the chain. runs/dqn_manifest.json describes the DQN export only;
runs/artifact_manifest.json pins artifact hashes but carries production
metadata as prose; the training facts live in runs/train_meta.json and
runs/dqn_meta.json; the evaluation figures and the artifact hashes the
evaluator actually used live in runs/eval.json. No single document ties
each trained policy to its checkpoint by hash together with its
training record (environment and encoding version, steps, reward,
seed), its evaluation figures and a measured inference cost, and
nothing verifies that whole chain in one place.

The fix under test: runs/model_cards.json, built and verified by
scripts/model_cards.py. One compact card per evaluated agent; learned
cards are pinned to their checkpoint and browser export by SHA-256 and
cross-checked against the hashes recorded in runs/eval.json and in the
DQN parity fixture (tests/fixtures/dqn_parity_observations.json), so
the export hash on the card is provably the evaluated model and the
model the JS fixture was generated from. Cards also distinguish the
policy kinds: learned (rookie, pro), search (grandmaster), heuristic
(greedy) and baseline (random). A one-command smoke run trains tiny
Q-learning and DQN models and rebuilds verifiable cards end to end.
"""

import hashlib
import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
CARDS = os.path.join(RUNS, "model_cards.json")
FIXTURE = os.path.join(ROOT, "tests", "fixtures",
                       "dqn_parity_observations.json")

from scripts.model_cards import run_smoke, verify_model_cards  # noqa: E402

# Known answers, frozen from the committed artifacts (the same values
# runs/artifact_manifest.json and runs/dqn_manifest.json record).
PINNED_SHA256 = {
    "qtable": "f46fb1a226996381ed478eaec199c8691b17bf63ee42063b1222368dc49346d6",
    "policy": "1bc7979eb7705b2a2a155c059ff9557b28f990f40106a6828c96bc0b51764141",
    "dqn_weights": "b171d8a20e99e2ad4c472f03a940c83399570eb04f253d8da4aa299df57c7861",
    "dqn_weights_payload": "704e56c763985c31463b90939fd46394a5f79bfed70b37e7de336053efd28c61",
    "eval": "4373ee84712479f512d35764ecb5c6d06aa07f1b5289b2b7dd4bdd535e90274a",
}
PINNED_ELO = {"random": 573, "greedy": 1317, "rookie": 1366,
              "pro": 925, "grandmaster": 1819}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cards():
    doc = _load(CARDS)
    return doc, {c["id"]: c for c in doc["cards"]}


def test_model_cards_document_verifies_clean():
    doc, by_id = _cards()
    assert doc["schema_version"] == 1
    assert doc["kind"] == "deeparcade-model-cards"
    assert [c["id"] for c in doc["cards"]] == [
        "random", "greedy", "rookie", "pro", "grandmaster"]
    assert verify_model_cards(ROOT) == []


def test_learned_cards_are_pinned_to_checkpoints_by_hash():
    _, by_id = _cards()
    ev = _load(os.path.join(RUNS, "eval.json"))

    rookie = by_id["rookie"]
    assert rookie["checkpoint"]["artifact"] == "runs/qtable.json"
    assert rookie["checkpoint"]["sha256"] == PINNED_SHA256["qtable"]
    assert rookie["checkpoint"]["sha256"] == _sha256(
        os.path.join(RUNS, "qtable.json"))
    assert rookie["export"]["artifact"] == "runs/policy.json"
    assert rookie["export"]["sha256"] == PINNED_SHA256["policy"]
    assert rookie["export"]["sha256"] == _sha256(
        os.path.join(RUNS, "policy.json"))
    # The card's checkpoint is the artifact the evaluator recorded.
    assert rookie["evaluation"]["evaluated_artifact_sha256"] == \
        ev["policies"]["rookie"]["artifact_sha256"] == \
        rookie["checkpoint"]["sha256"]

    pro = by_id["pro"]
    assert pro["checkpoint"]["artifact"] == "runs/dqn_weights.json"
    assert pro["checkpoint"]["sha256"] == PINNED_SHA256["dqn_weights"]
    assert pro["checkpoint"]["sha256"] == _sha256(
        os.path.join(RUNS, "dqn_weights.json"))
    assert pro["checkpoint"]["weights_sha256"] == \
        PINNED_SHA256["dqn_weights_payload"]
    assert pro["evaluation"]["evaluated_artifact_sha256"] == \
        ev["policies"]["pro"]["artifact_sha256"] == \
        pro["checkpoint"]["sha256"]
    # ... and the artifact the JS parity fixture was generated from.
    fixture = _load(FIXTURE)
    assert pro["parity_fixture"]["weights_artifact_sha256"] == \
        fixture["weights_artifact_sha256"] == pro["checkpoint"]["sha256"]
    assert pro["parity_fixture"]["sha256"] == _sha256(FIXTURE)


def test_cards_distinguish_policy_kinds():
    _, by_id = _cards()
    kinds = {cid: c["kind"] for cid, c in by_id.items()}
    assert kinds == {"random": "baseline", "greedy": "heuristic",
                     "rookie": "learned", "pro": "learned",
                     "grandmaster": "search"}
    # Only learned policies carry a checkpoint and a training record.
    for cid in ("random", "greedy", "grandmaster"):
        assert by_id[cid]["checkpoint"] is None
        assert by_id[cid]["training"] is None
    for cid in ("rookie", "pro"):
        assert by_id[cid]["checkpoint"] is not None
        assert by_id[cid]["training"] is not None


def test_cards_record_training_facts_and_rewards():
    _, by_id = _cards()
    train_meta = _load(os.path.join(RUNS, "train_meta.json"))
    dqn_meta = _load(os.path.join(RUNS, "dqn_meta.json"))
    curve = _load(os.path.join(RUNS, "curve.json"))
    dqn_curve = _load(os.path.join(RUNS, "dqn_curve.json"))

    rookie = by_id["rookie"]["training"]
    assert rookie["steps"] == 30000
    assert rookie["steps_unit"] == "training episodes"
    assert rookie["episodes"] == train_meta["episodes"] == 30000
    assert rookie["seed"] == train_meta["seed"] == 7
    assert rookie["final_mean100_score"] == \
        train_meta["final_mean100_score"] == 11.54
    assert rookie["final_mean_reward"] == curve["mean_reward"][-1]

    pro = by_id["pro"]["training"]
    assert pro["steps"] == 300000
    assert pro["steps_unit"] == "training steps"
    assert pro["episodes"] == dqn_meta["episodes"] == 6984
    assert pro["seed"] == dqn_meta["seed"] == 11
    assert pro["final_mean100_score"] == dqn_meta["final_mean100"] == 4.54
    assert pro["final_mean_reward"] == dqn_curve["mean_reward"][-1]


def test_cards_record_environment_and_encodings():
    doc, by_id = _cards()
    env = doc["environment"]
    assert env["name"] == "SnakeEnv"
    assert (env["width"], env["height"]) == (12, 12)

    pro_enc = by_id["pro"]["encoding"]
    assert pro_enc["name"] == "encode_dqn"
    assert pro_enc["revision"] == 1
    assert pro_enc["input_dim"] == 436

    rookie_enc = by_id["rookie"]["encoding"]
    assert rookie_enc["name"] == "encode_state"
    assert rookie_enc["states"] == 32
    assert rookie_enc["actions"] == 3


def test_cards_record_evaluation_figures():
    doc, by_id = _cards()
    ev = _load(os.path.join(RUNS, "eval.json"))
    assert doc["evaluation"]["artifact"] == "runs/eval.json"
    assert doc["evaluation"]["sha256"] == PINNED_SHA256["eval"]
    assert doc["evaluation"]["sha256"] == _sha256(
        os.path.join(RUNS, "eval.json"))
    for cid, card in by_id.items():
        block = card["evaluation"]
        assert block["elo"] == PINNED_ELO[cid] == ev["elo"][cid]
        assert block["mean"] == ev["summary"][cid]["mean"]
        assert block["median"] == ev["summary"][cid]["median"]
        assert block["n"] == ev["summary"][cid]["n"]
    # Grandmaster's summary pool is the paired pool; the rest are the
    # independent pool (runs/eval.json provenance, legacy_keys_note).
    assert by_id["grandmaster"]["evaluation"]["pool"] == "paired"
    assert by_id["rookie"]["evaluation"]["pool"] == "independent"


def test_cards_record_measured_inference_cost():
    _, by_id = _cards()
    for cid, card in by_id.items():
        cost = card["inference_cost"]
        assert isinstance(cost["ms_per_decision"], (int, float))
        assert cost["ms_per_decision"] > 0
        assert cost["decisions_timed"] >= 1
        assert cost["method"]
    # An 80-simulation tree search per move costs orders of magnitude
    # more than one forward pass of the exported network.
    assert by_id["grandmaster"]["inference_cost"]["ms_per_decision"] > \
        by_id["pro"]["inference_cost"]["ms_per_decision"]


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("da7-smoke"))
    summary = run_smoke(root)
    return root, summary


def test_smoke_training_builds_verifiable_cards(smoke_root):
    root, summary = smoke_root
    assert summary["problems"] == []
    # The smoke run is small: hundreds of episodes, a few thousand
    # steps, not the committed 30000-episode / 300000-step protocols.
    assert summary["q_episodes"] <= 1000
    assert summary["dqn_steps"] <= 5000
    doc = _load(os.path.join(root, "runs", "model_cards.json"))
    by_id = {c["id"]: c for c in doc["cards"]}
    assert by_id["rookie"]["training"]["episodes"] == summary["q_episodes"]
    assert by_id["pro"]["training"]["steps"] == summary["dqn_steps"]
    # The smoke DQN export is a versioned (schema 1) checkpoint and
    # loads through the strict checkpoint path.
    from scripts.train_dqn import load_checkpoint
    _, info = load_checkpoint(os.path.join(root, "runs", "dqn_weights.json"))
    assert info["schema_version"] == 1
    assert info["weights_sha256"] == \
        by_id["pro"]["checkpoint"]["weights_sha256"]


def test_verify_detects_a_tampered_checkpoint_hash(smoke_root, tmp_path):
    root, _ = smoke_root
    tampered = str(tmp_path / "tampered")
    shutil.copytree(root, tampered)
    cards_path = os.path.join(tampered, "runs", "model_cards.json")
    doc = _load(cards_path)
    for card in doc["cards"]:
        if card["id"] == "rookie":
            card["checkpoint"]["sha256"] = "0" * 64
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    problems = verify_model_cards(tampered)
    assert problems
    assert any("qtable" in p for p in problems)
