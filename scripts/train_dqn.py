"""Deep Q-Network for Snake: the "Pro" tier.

Real DQN: experience replay, target network, epsilon decay. Trains on the
full 12x12 grid as 3 binary channels (head, body, food) + heading one-hot.
Weights are exported to JSON so the exact trained network runs in the browser.

Checkpoint schema (Stage 3 P0-03):
  export_weights writes a versioned checkpoint document (schema_version 1)
  carrying the encoder name and revision, the float regimes, a SHA-256
  over the canonical weights payload, and metadata (training args, data
  key range, code commit). load_checkpoint is the strict inverse: it
  validates the document (supported schema, exact layer shapes, every
  value a finite number, hash match on versioned documents) and only then
  builds the PyTorch network, so a corrupt checkpoint fails closed with
  CheckpointError. Documents without a schema_version are the legacy
  pre-schema exports (schema 0); the committed runs/dqn_weights.json is
  one, and it is described by runs/dqn_manifest.json. The legacy
  top-level keys (input_dim, layers) are kept in versioned documents so
  the game build keeps reading the same fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from narcade.experiment_seeds import training_env_key
from narcade.snake import SnakeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

INPUT_DIM = 12 * 12 * 3 + 4
N_ACTIONS = 3

# Checkpoint schema constants. ENCODER_REVISION identifies the encode_dqn
# layout below; bump it if the layout ever changes, so a checkpoint can
# never be silently paired with a different encoder.
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "deeparcade-dqn-checkpoint"
MANIFEST_KIND = "deeparcade-dqn-checkpoint-manifest"
ENCODER_NAME = "encode_dqn"
ENCODER_REVISION = 1
ENCODER_LAYOUT = ("head 12x12, body 12x12, food 12x12 (cell index r*12 + c), "
                  "heading one-hot 4")
WEIGHT_FLOAT_REGIME = "float32"
# Parity between the PyTorch policy and the browser export is declared in
# float64: the browser accumulates in float64 (web/game.html dqnAct) over
# the stored float32 values. A float32 PyTorch evaluation differs from the
# browser export by up to 3.546e-06 on the frozen parity fixture purely
# from float32 accumulation; greedy actions still agree everywhere.
PARITY_FLOAT_REGIME = "float64"
LAYER_SHAPES = [(128, INPUT_DIM), (64, 128), (N_ACTIONS, 64)]  # (out, in)
STATE_DICT_KEYS = ["net.0", "net.2", "net.4"]


class CheckpointError(ValueError):
    """A checkpoint document is malformed, unsupported, or corrupt.

    Subclasses ValueError so existing callers that guard loads with
    ValueError keep working, while checkpoint failures stay
    distinguishable from unrelated value errors."""


def encode_dqn(env):
    """Flattened state vector. MUST match the browser's encodeDQN exactly.

    Layout: 144 head cells, 144 body cells, 144 food cells, 4 heading one-hot.
    Index within each 144-block is r*12 + c.
    """
    w, h = env.width, env.height
    assert w == 12 and h == 12, "DQN was trained on 12x12"
    v = np.zeros(INPUT_DIM, dtype=np.float32)
    hr, hc = env.body[-1]
    v[hr * 12 + hc] = 1.0
    for r, c in list(env.body)[:-1]:
        v[144 + r * 12 + c] = 1.0
    fr, fc = env.food
    v[288 + fr * 12 + fc] = 1.0
    v[432 + env.heading] = 1.0
    return v


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def train_dqn(steps=200000, seed=7, verbose=True):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cpu")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    replay = []  # plain list: random.sample on a deque is O(n) per draw
    replay_max = 50000
    gamma, batch = 0.95, 64
    eps, eps_end = 1.0, 0.05
    eps_decay_steps = 120000
    target_sync = 2000
    update_every = 8

    env = SnakeEnv(seed=seed)
    s = encode_dqn(env)
    ep_reward, ep_score = 0.0, 0
    scores100, rewards100 = [], []
    b_scores, b_rewards = [], []
    n_ep = 0

    for step in range(1, steps + 1):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * step / eps_decay_steps)
        if rng.random() < eps:
            a = rng.choice([0, 1, 2])
        else:
            with torch.no_grad():
                a = int(policy_net(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())
        s2_list, r, done, _ = env_step(env, a)
        replay.append((s, a, r, s2_list, done))
        if len(replay) > replay_max:
            del replay[:replay_max // 10]
        s = s2_list
        ep_reward += r
        if done:
            n_ep += 1
            b_scores.append(env.score)
            b_rewards.append(ep_reward)
            if n_ep % 100 == 0:
                scores100.append(sum(b_scores[-100:]) / 100)
                rewards100.append(sum(b_rewards[-100:]) / 100)
                if verbose and n_ep % 1000 == 0:
                    print(f"ep {n_ep:6d} step {step:7d} mean100={scores100[-1]:5.2f} "
                          f"eps={eps:.3f}", flush=True)
            env = SnakeEnv(seed=training_env_key(seed, n_ep))
            s = encode_dqn(env)
            ep_reward, ep_score = 0.0, 0

        if len(replay) >= 2000 and step % update_every == 0:
            idx = rng.sample(range(len(replay)), batch)
            bs = torch.from_numpy(np.stack([replay[i][0] for i in idx]))
            ba = torch.tensor([replay[i][1] for i in idx]).unsqueeze(1)
            br = torch.tensor([replay[i][2] for i in idx])
            bs2 = torch.from_numpy(np.stack([replay[i][3] for i in idx]))
            bd = torch.tensor([replay[i][4] for i in idx])
            q = policy_net(bs).gather(1, ba).squeeze(1)
            with torch.no_grad():
                q_next = target_net(bs2).max(1)[0]
                target = br + gamma * q_next * (~bd)
            loss = loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % target_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

    meta = {"steps": steps, "episodes": n_ep, "seed": seed,
            "seconds": round(time.time() - t0, 1),
            "final_mean100": scores100[-1] if scores100 else 0.0}
    return policy_net, scores100, rewards100, meta


def env_step(env, a):
    s2, r, done, _ = env.step(a)
    return encode_dqn(env), r, done, _


def _layers_from_state_dict(sd):
    return [
        {"w": sd["net.0.weight"].tolist(), "b": sd["net.0.bias"].tolist()},
        {"w": sd["net.2.weight"].tolist(), "b": sd["net.2.bias"].tolist()},
        {"w": sd["net.4.weight"].tolist(), "b": sd["net.4.bias"].tolist()},
    ]


def weights_sha256(layers):
    """SHA-256 over the canonical JSON encoding of the layers payload.

    Canonical = json.dumps with sorted keys and tight separators over
    the exact float values, so the hash is stable across processes and
    platforms for identical weights."""
    canonical = json.dumps({"layers": layers}, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def validate_checkpoint_document(doc):
    """Validate a checkpoint document; return its normalized info dict.

    Info: {"schema_version": int, "weights_sha256": str,
    "metadata": dict-or-None, "encoder_revision": int}.
    Raises CheckpointError naming the first defect found. Nothing is
    constructed and no tensor is built here; this is the fail-closed
    gate every load passes through."""
    if not isinstance(doc, dict):
        raise CheckpointError("checkpoint document must be a JSON object")
    schema = doc.get("schema_version", 0)
    if schema not in (0, CHECKPOINT_SCHEMA_VERSION):
        raise CheckpointError(
            f"unsupported checkpoint schema_version {schema!r}; this code "
            f"reads legacy schema 0 and schema_version "
            f"{CHECKPOINT_SCHEMA_VERSION}. Re-export with scripts/train_dqn.py"
        )
    if doc.get("input_dim") != INPUT_DIM:
        raise CheckpointError(
            f"checkpoint input_dim {doc.get('input_dim')!r} != expected "
            f"{INPUT_DIM}; refusing to load mismatched weights"
        )
    layers = doc.get("layers")
    if not isinstance(layers, list) or len(layers) != len(LAYER_SHAPES):
        raise CheckpointError(
            f"checkpoint must hold exactly {len(LAYER_SHAPES)} layers, "
            f"found {len(layers) if isinstance(layers, list) else type(layers).__name__}"
        )
    for li, (layer, (rows, cols)) in enumerate(zip(layers, LAYER_SHAPES)):
        if not isinstance(layer, dict) or set(layer.keys()) != {"w", "b"}:
            raise CheckpointError(
                f"layer {li} must be an object with exactly keys 'w' and 'b'"
            )
        w, b = layer["w"], layer["b"]
        if not isinstance(w, list) or len(w) != rows:
            raise CheckpointError(
                f"layer {li} weight must have {rows} rows, "
                f"found {len(w) if isinstance(w, list) else type(w).__name__}"
            )
        for ri, row in enumerate(w):
            if not isinstance(row, list) or len(row) != cols:
                raise CheckpointError(
                    f"layer {li} weight row {ri} must have {cols} values, "
                    f"found {len(row) if isinstance(row, list) else type(row).__name__}"
                )
            for v in row:
                if not _is_finite_number(v):
                    raise CheckpointError(
                        f"layer {li} weight row {ri} holds a non-numeric or "
                        f"non-finite value {v!r}"
                    )
        if not isinstance(b, list) or len(b) != rows:
            raise CheckpointError(
                f"layer {li} bias must have {rows} values, "
                f"found {len(b) if isinstance(b, list) else type(b).__name__}"
            )
        for v in b:
            if not _is_finite_number(v):
                raise CheckpointError(
                    f"layer {li} bias holds a non-numeric or non-finite "
                    f"value {v!r}"
                )
    digest = weights_sha256(layers)
    metadata = None
    if schema == CHECKPOINT_SCHEMA_VERSION:
        if doc.get("kind") != CHECKPOINT_KIND:
            raise CheckpointError(
                f"checkpoint kind {doc.get('kind')!r} != expected "
                f"{CHECKPOINT_KIND!r}"
            )
        encoder = doc.get("encoder")
        if not isinstance(encoder, dict) \
                or encoder.get("name") != ENCODER_NAME \
                or encoder.get("revision") != ENCODER_REVISION:
            raise CheckpointError(
                f"checkpoint encoder {encoder!r} does not match the declared "
                f"encoder {ENCODER_NAME} revision {ENCODER_REVISION}"
            )
        if doc.get("float_regime") != WEIGHT_FLOAT_REGIME:
            raise CheckpointError(
                f"checkpoint float_regime {doc.get('float_regime')!r} != "
                f"declared {WEIGHT_FLOAT_REGIME!r}"
            )
        if doc.get("weights_sha256") != digest:
            raise CheckpointError(
                "checkpoint weights_sha256 does not match the weights "
                "payload; the document is corrupt or tampered"
            )
        metadata = doc.get("metadata")
        if not isinstance(metadata, dict) or set(metadata.keys()) != {
                "training_args", "data_key_range", "code_commit"}:
            raise CheckpointError(
                "versioned checkpoint metadata must be an object with "
                "exactly keys 'training_args', 'data_key_range', "
                "'code_commit' (values may be null when unknown)"
            )
    return {"schema_version": schema, "weights_sha256": digest,
            "metadata": metadata, "encoder_revision": ENCODER_REVISION}


def export_weights(net, training_args=None, data_key_range=None, code_commit=None):
    """Export a DQN as a versioned checkpoint document (schema 1).

    The legacy top-level keys (input_dim, layers) are unchanged, so the
    game build keeps reading the same fields. Metadata values that are
    unknown are recorded as null, never omitted."""
    layers = _layers_from_state_dict(net.state_dict())
    doc = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "float_regime": WEIGHT_FLOAT_REGIME,
        "parity_float_regime": PARITY_FLOAT_REGIME,
        "encoder": {"name": ENCODER_NAME, "revision": ENCODER_REVISION,
                    "input_dim": INPUT_DIM, "layout": ENCODER_LAYOUT},
        "input_dim": INPUT_DIM,
        "layers": layers,
        "weights_sha256": weights_sha256(layers),
        "metadata": {"training_args": training_args,
                     "data_key_range": data_key_range,
                     "code_commit": code_commit},
    }
    validate_checkpoint_document(doc)
    return doc


def load_checkpoint(source):
    """Strict inverse of the JSON weights export: document -> (DQN, info).

    source is a filesystem path or an already-parsed document dict.
    Validation (validate_checkpoint_document) runs before any tensor is
    built; a corrupt checkpoint raises CheckpointError here, never a
    silent partial load. The returned network is in eval mode."""
    if isinstance(source, (str, os.PathLike)):
        try:
            with open(source, encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError as e:
            raise CheckpointError(
                f"checkpoint file is not valid JSON ({source}): {e}"
            ) from e
    else:
        doc = source
    info = validate_checkpoint_document(doc)
    sd = {}
    for key, layer in zip(STATE_DICT_KEYS, doc["layers"]):
        sd[f"{key}.weight"] = torch.tensor(layer["w"], dtype=torch.float32)
        sd[f"{key}.bias"] = torch.tensor(layer["b"], dtype=torch.float32)
    net = DQN()
    net.load_state_dict(sd)
    net.eval()
    return net, info


def browser_export_logits(doc, observations):
    """Logits of the browser export policy over the given observations.

    A faithful mirror of the forward pass in web/game.html (dqnAct):
    each unit starts from its bias and accumulates w[i] * x[i] in index
    order in float64, ReLU between layers, and hidden activations are
    rounded to float32, matching the game's Float32Array storage. The
    document is validated first, so a corrupt export fails closed here
    too. observations is an (n, INPUT_DIM) array-like of encoded states.
    Returns an (n, 3) float64 numpy array."""
    validate_checkpoint_document(doc)
    layers = doc["layers"]
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise CheckpointError(
            f"observations must have shape (n, {INPUT_DIM}), got {x.shape}"
        )
    x = x.astype(np.float32).astype(np.float64)  # browser inputs are Float32Array
    acts = x
    for li, layer in enumerate(layers):
        w = np.asarray(layer["w"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        out = np.empty((acts.shape[0], w.shape[0]), dtype=np.float64)
        for j in range(w.shape[0]):
            s = np.full(acts.shape[0], b[j], dtype=np.float64)
            row = w[j]
            for i in range(w.shape[1]):
                s += row[i] * acts[:, i]
            out[:, j] = s
        if li < len(layers) - 1:
            out = np.maximum(out, 0.0).astype(np.float32).astype(np.float64)
        acts = out
    return acts


def build_checkpoint_manifest(weights_path, meta_path, weights_file_commit):
    """Build the checkpoint manifest for a committed weights artifact.

    Every field is recomputed from the weights file and the training
    meta file, except weights_file_commit, a recorded historical fact
    (the commit that introduced the weights file) supplied by the
    caller. The data key range is derived from the training meta: the
    trainer reseeds each new episode environment with seed + episode
    index, so the episode seeds used are seed .. seed + episodes - 1."""
    with open(weights_path, "rb") as f:
        raw = f.read()
    doc = json.loads(raw.decode("utf-8"))
    info = validate_checkpoint_document(doc)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    seed, episodes = meta["seed"], meta["episodes"]
    return {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "artifact": "runs/dqn_weights.json",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "weights_sha256": info["weights_sha256"],
        "checkpoint_schema_version": info["schema_version"],
        "encoder": {"name": ENCODER_NAME, "revision": ENCODER_REVISION,
                    "input_dim": INPUT_DIM, "layout": ENCODER_LAYOUT},
        "architecture": [[cols, rows] for rows, cols in LAYER_SHAPES],
        "float_regime": WEIGHT_FLOAT_REGIME,
        "parity_float_regime": PARITY_FLOAT_REGIME,
        "training": meta,
        "data_key_range": {
            "episode_seeds": [seed, seed + episodes - 1],
            "derivation": ("train_dqn reseeds each new episode environment "
                           "with seed + episode index; range derived from "
                           "runs/dqn_meta.json"),
        },
        "provenance": {
            "weights_file_introduced_in_commit": weights_file_commit,
            "training_code_commit": None,
            "note": ("The weights file entered git in the recorded commit "
                     "(web upload). The original training run did not "
                     "record a code commit; versioned exports record "
                     "code_commit in the checkpoint metadata instead."),
        },
        "parity_fixture": "tests/fixtures/dqn_parity_observations.json",
    }


def _git_commit():
    """HEAD commit of the checkout holding this script, or None when git
    is unavailable. Never fails the run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(HERE), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=RUNS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    net, scores, rewards, meta = train_dqn(steps=args.steps, seed=args.seed)
    torch.save(net.state_dict(), os.path.join(args.out, "dqn.pt"))
    commit = _git_commit()
    doc = export_weights(
        net,
        training_args={"steps": args.steps, "seed": args.seed},
        data_key_range={"episode_seeds": [args.seed,
                                          args.seed + meta["episodes"] - 1]},
        code_commit=commit,
    )
    weights_path = os.path.join(args.out, "dqn_weights.json")
    meta_path = os.path.join(args.out, "dqn_meta.json")
    with open(weights_path, "w") as f:
        json.dump(doc, f)
    with open(os.path.join(args.out, "dqn_curve.json"), "w") as f:
        json.dump({"mean_score": scores, "mean_reward": rewards}, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    manifest = build_checkpoint_manifest(weights_path, meta_path,
                                         weights_file_commit=commit)
    with open(os.path.join(args.out, "dqn_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done in {meta['seconds']}s, {meta['episodes']} episodes, "
          f"final mean100={meta['final_mean100']:.2f}")


if __name__ == "__main__":
    main()
