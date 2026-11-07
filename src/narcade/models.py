"""Convolutional DQN for Snake (Stage 3 deeparcade DA8).

The flat MLP Pro scores 5.39 on the committed 500-episode evaluation
while Greedy scores 19.89 and Rookie scores 20.78, and DA4 showed the
MLP memorizes flat pixels instead of learning spatial abstractions.
This module tests whether spatial structure helps under matched
conditions. It does not assume the CNN will win.

The CNN reads the same 12x12 board as three spatial channels (head,
body, food) plus heading as conditioning, and outputs three Q-values.
The encoder is small and configurable, with residual and non-residual
candidates; no layer layout is claimed best in advance. Normalized
coordinate channels and head-relative cropping are separate variants
reserved for later items, not folded into this arm.

The MLP in scripts/train_dqn.py is untouched: this module is additive,
and the frozen MLP baseline keeps its own checkpoint format. CNN
checkpoints use their own versioned schema (kind
``deeparcade-cnn-dqn-checkpoint``, schema_version 1) with the same
fail-closed discipline: the document is validated (supported schema,
exact layer shapes for its recorded config, every value a finite
number, weights hash match) before any tensor is built.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 3
BOARD = 12

CNN_CHECKPOINT_SCHEMA_VERSION = 1
CNN_CHECKPOINT_KIND = "deeparcade-cnn-dqn-checkpoint"
CNN_ENCODER_NAME = "encode_cnn_dqn"
CNN_ENCODER_REVISION = 1
CNN_ENCODER_LAYOUT = (
    "head 12x12, body 12x12, food 12x12 (cell index r*12 + c), "
    "heading one-hot 4 as conditioning after pooling"
)
CNN_WEIGHT_FLOAT_REGIME = "float32"
# Parity between the PyTorch policy and a NumPy/browser mirror is
# declared in float64: the mirror accumulates in float64 over the stored
# float32 values, with hidden activations rounded to float32 after each
# ReLU, matching the browser Float32Array convention used by the MLP
# mirror in scripts/train_dqn.py.
CNN_PARITY_FLOAT_REGIME = "float64"


class CNNCheckpointError(ValueError):
    """A CNN checkpoint document is malformed, unsupported, or corrupt.

    Subclasses ValueError so callers that guard loads with ValueError
    keep working, while checkpoint failures stay distinguishable.
    """


def encode_cnn_dqn(env):
    """Spatial state encoding. MUST match the flat encode_dqn bit layout.

    Returns (spatial, heading): spatial is a float32 (3, 12, 12) array
    with planes head, body, food (cell index r*12 + c within a plane),
    and heading is a float32 (4,) one-hot. The food plane stays empty
    on a terminal win, exactly as in the flat encoder.
    """
    w, h = env.width, env.height
    assert w == 12 and h == 12, "CNN DQN was trained on 12x12"
    spatial = np.zeros((3, 12, 12), dtype=np.float32)
    hr, hc = env.body[-1]
    spatial[0, hr, hc] = 1.0
    for r, c in list(env.body)[:-1]:
        spatial[1, r, c] = 1.0
    if env.food is not None:
        fr, fc = env.food
        spatial[2, fr, fc] = 1.0
    heading = np.zeros(4, dtype=np.float32)
    heading[env.heading] = 1.0
    return spatial, heading


@dataclass(frozen=True)
class CNNConfig:
    """One CNN architecture candidate.

    channels: output channels of each 3x3 convolution in order.
    residual: add one residual block (two 3x3 convolutions with a skip)
        at the last channel width after the plain convolutions.
    pool_size: adaptive average pooling output side (pool_size x
        pool_size) before the head; flattening without pooling is not
        used here because it would grow the head past the MLP size.
    hidden: width of the fully connected head after pooling.
    """

    channels: tuple = (16, 32)
    kernel: int = 3
    residual: bool = False
    pool_size: int = 2
    hidden: int = 64

    def __post_init__(self):
        channels = tuple(int(c) for c in self.channels)
        if not channels or any(c <= 0 for c in channels):
            raise ValueError(f"channels must be positive ints, got {self.channels!r}")
        if self.kernel <= 0 or self.kernel % 2 == 0:
            raise ValueError(f"kernel must be a positive odd int, got {self.kernel!r}")
        if not 1 <= self.pool_size <= BOARD:
            raise ValueError(f"pool_size must be in 1..12, got {self.pool_size!r}")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden!r}")
        object.__setattr__(self, "channels", channels)

    def to_dict(self):
        d = asdict(self)
        d["channels"] = list(d["channels"])
        return d

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise CNNCheckpointError("CNN config must be an object")
        try:
            return cls(
                channels=tuple(d["channels"]),
                kernel=d.get("kernel", 3),
                residual=d.get("residual", False),
                pool_size=d.get("pool_size", 2),
                hidden=d.get("hidden", 64),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CNNCheckpointError(f"invalid CNN config: {e}") from e


class CNNDQN(nn.Module):
    """Small convolutional Q-network for 12x12 Snake.

    Conv encoder over the three spatial planes, adaptive average
    pooling, heading one-hot concatenated as conditioning, then a
    two-layer head to three Q-values.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else CNNConfig()
        cfg = self.config
        pad = cfg.kernel // 2
        in_ch = 3
        self.convs = nn.ModuleList()
        for out_ch in cfg.channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, cfg.kernel, padding=pad))
            in_ch = out_ch
        last = cfg.channels[-1]
        if cfg.residual:
            self.res1 = nn.Conv2d(last, last, 3, padding=1)
            self.res2 = nn.Conv2d(last, last, 3, padding=1)
        else:
            self.res1 = None
            self.res2 = None
        features = last * cfg.pool_size * cfg.pool_size
        self.head1 = nn.Linear(features + 4, cfg.hidden)
        self.head2 = nn.Linear(cfg.hidden, N_ACTIONS)

    def forward(self, spatial, heading):
        x = spatial
        for conv in self.convs:
            x = F.relu(conv(x))
        if self.res1 is not None:
            r = F.relu(self.res1(x))
            r = self.res2(r)
            x = F.relu(x + r)
        x = F.adaptive_avg_pool2d(x, (self.config.pool_size, self.config.pool_size))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, heading], dim=1)
        x = F.relu(self.head1(x))
        return self.head2(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def describe(self):
        return {
            "input_channels": 3,
            "board": BOARD,
            "channels": list(self.config.channels),
            "kernel": self.config.kernel,
            "residual": self.config.residual,
            "pool_size": self.config.pool_size,
            "hidden": self.config.hidden,
            "n_actions": N_ACTIONS,
            "params": self.num_params(),
        }


def _layers_from_state_dict(sd):
    layers = []
    for name in sd:
        tensor = sd[name]
        layers.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "values": tensor.detach().cpu().tolist(),
            }
        )
    return layers


def cnn_weights_sha256(layers):
    """SHA-256 over the canonical JSON encoding of the layers payload.

    Canonical = json.dumps with sorted keys and tight separators over
    the exact float values, so the hash is stable across processes and
    platforms for identical weights.
    """
    canonical = json.dumps({"layers": layers}, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _check_finite_nested(values, where):
    if isinstance(values, list):
        for v in values:
            _check_finite_nested(v, where)
    elif not _is_finite_number(values):
        raise CNNCheckpointError(
            f"layer {where} holds a non-numeric or non-finite value {values!r}"
        )


def _nested_shape(values):
    shape = []
    cur = values
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape


def validate_cnn_checkpoint_document(doc):
    """Validate a CNN checkpoint document; return its normalized info.

    Info: {"schema_version": int, "weights_sha256": str,
    "metadata": dict, "encoder": dict, "config": CNNConfig}.
    Raises CNNCheckpointError naming the first defect found. Nothing
    is constructed and no tensor is built here.
    """
    if not isinstance(doc, dict):
        raise CNNCheckpointError("checkpoint document must be a JSON object")
    schema = doc.get("schema_version")
    if schema != CNN_CHECKPOINT_SCHEMA_VERSION:
        raise CNNCheckpointError(
            f"unsupported CNN checkpoint schema_version {schema!r}; this code "
            f"reads schema_version {CNN_CHECKPOINT_SCHEMA_VERSION} only. "
            "Re-export with scripts/train_cnn_dqn.py"
        )
    if doc.get("kind") != CNN_CHECKPOINT_KIND:
        raise CNNCheckpointError(
            f"checkpoint kind {doc.get('kind')!r} != expected "
            f"{CNN_CHECKPOINT_KIND!r}"
        )
    encoder = doc.get("encoder")
    if not isinstance(encoder, dict) \
            or encoder.get("name") != CNN_ENCODER_NAME \
            or encoder.get("revision") != CNN_ENCODER_REVISION:
        raise CNNCheckpointError(
            f"checkpoint encoder {encoder!r} does not match the declared "
            f"encoder {CNN_ENCODER_NAME} revision {CNN_ENCODER_REVISION}"
        )
    if doc.get("float_regime") != CNN_WEIGHT_FLOAT_REGIME:
        raise CNNCheckpointError(
            f"checkpoint float_regime {doc.get('float_regime')!r} != "
            f"declared {CNN_WEIGHT_FLOAT_REGIME!r}"
        )
    config = CNNConfig.from_dict(doc.get("config"))
    expected = CNNDQN(config)
    expected_keys = [name for name, _ in expected.named_parameters()] \
        + [name for name, _ in expected.named_buffers()]
    layers = doc.get("layers")
    if not isinstance(layers, list):
        raise CNNCheckpointError("checkpoint layers must be a list")
    got_names = [l.get("name") if isinstance(l, dict) else None for l in layers]
    if got_names != list(expected.state_dict().keys()):
        raise CNNCheckpointError(
            "checkpoint layer names do not match the recorded config "
            f"(expected {list(expected.state_dict().keys())}, "
            f"found {got_names})"
        )
    sd_shapes = {k: list(v.shape) for k, v in expected.state_dict().items()}
    for layer in layers:
        if set(layer.keys()) != {"name", "shape", "values"}:
            raise CNNCheckpointError(
                f"layer {layer.get('name')!r} must hold exactly keys "
                "'name', 'shape', 'values'"
            )
        want = sd_shapes[layer["name"]]
        if layer["shape"] != want or _nested_shape(layer["values"]) != want:
            raise CNNCheckpointError(
                f"layer {layer['name']!r} shape {layer['shape']!r} != "
                f"expected {want!r} for the recorded config"
            )
        _check_finite_nested(layer["values"], layer["name"])
    digest = cnn_weights_sha256(layers)
    if doc.get("weights_sha256") != digest:
        raise CNNCheckpointError(
            "checkpoint weights_sha256 does not match the weights payload; "
            "the document is corrupt or tampered"
        )
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict) or set(metadata.keys()) != {
            "training_args", "data_key_range", "code_commit"}:
        raise CNNCheckpointError(
            "CNN checkpoint metadata must be an object with exactly keys "
            "'training_args', 'data_key_range', 'code_commit' (values may "
            "be null when unknown)"
        )
    return {
        "schema_version": schema,
        "weights_sha256": digest,
        "metadata": metadata,
        "encoder": {
            "name": CNN_ENCODER_NAME,
            "revision": CNN_ENCODER_REVISION,
            "input_channels": 3,
            "layout": CNN_ENCODER_LAYOUT,
        },
        "config": config,
    }


def export_cnn_weights(net, config=None, training_args=None,
                       data_key_range=None, code_commit=None):
    """Export a CNNDQN as a versioned checkpoint document (schema 1).

    Metadata values that are unknown are recorded as null, never
    omitted. The config is stored at the top level so a loader can
    rebuild the exact architecture before touching the weights.
    """
    cfg = config if config is not None else net.config
    layers = _layers_from_state_dict(net.state_dict())
    doc = {
        "schema_version": CNN_CHECKPOINT_SCHEMA_VERSION,
        "kind": CNN_CHECKPOINT_KIND,
        "float_regime": CNN_WEIGHT_FLOAT_REGIME,
        "parity_float_regime": CNN_PARITY_FLOAT_REGIME,
        "encoder": {"name": CNN_ENCODER_NAME, "revision": CNN_ENCODER_REVISION,
                    "input_channels": 3, "layout": CNN_ENCODER_LAYOUT},
        "config": cfg.to_dict(),
        "layers": layers,
        "weights_sha256": cnn_weights_sha256(layers),
        "metadata": {"training_args": training_args,
                     "data_key_range": data_key_range,
                     "code_commit": code_commit},
    }
    validate_cnn_checkpoint_document(doc)
    return doc


def _tensor_from_nested(values, shape):
    arr = np.asarray(values, dtype=np.float32)
    if list(arr.shape) != shape:
        raise CNNCheckpointError(
            f"layer values shape {list(arr.shape)} != declared {shape}"
        )
    return torch.from_numpy(arr)


def load_cnn_checkpoint(source):
    """Strict inverse of the CNN weights export: document -> (net, info).

    source is a filesystem path or an already-parsed document dict.
    Validation runs before any tensor is built; a corrupt checkpoint
    raises CNNCheckpointError here, never a silent partial load. The
    returned network is in eval mode.
    """
    if isinstance(source, (str, os.PathLike)):
        try:
            with open(source, encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError as e:
            raise CNNCheckpointError(
                f"checkpoint file is not valid JSON ({source}): {e}"
            ) from e
    else:
        doc = source
    info = validate_cnn_checkpoint_document(doc)
    sd = {}
    for layer in doc["layers"]:
        sd[layer["name"]] = _tensor_from_nested(layer["values"], layer["shape"])
    net = CNNDQN(info["config"])
    net.load_state_dict(sd)
    net.eval()
    return net, info


def _conv2d_numpy(x, w, b):
    """2D convolution with zero padding, float64 accumulation.

    x: (B, C_in, H, W); w: (C_out, C_in, K, K); b: (C_out,).
    Padding is (K-1)//2 on every side. Returns (B, C_out, H, W).
    """
    batch, _, h, wdt = x.shape
    out_ch, _, k, _ = w.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((batch, out_ch, h, wdt), dtype=np.float64)
    for oc in range(out_ch):
        acc = np.full((batch, h, wdt), float(b[oc]), dtype=np.float64)
        for ic in range(w.shape[1]):
            for kh in range(k):
                for kw in range(k):
                    acc += w[oc, ic, kh, kw] * xp[:, ic, kh:kh + h, kw:kw + wdt]
        out[:, oc] = acc
    return out


def _adaptive_avg_pool2d_numpy(x, out_size):
    """Adaptive average pooling, matching torch semantics."""
    batch, ch, h, wdt = x.shape
    oh, ow = out_size
    out = np.zeros((batch, ch, oh, ow), dtype=np.float64)
    for i in range(oh):
        r0 = int(math.floor(i * h / oh))
        r1 = int(math.ceil((i + 1) * h / oh))
        for j in range(ow):
            c0 = int(math.floor(j * wdt / ow))
            c1 = int(math.ceil((j + 1) * wdt / ow))
            out[:, :, i, j] = x[:, :, r0:r1, c0:c1].mean(axis=(2, 3))
    return out


def cnn_logits_numpy(doc, spatial, heading):
    """Logits of the CNN export policy over the given observations.

    A faithful mirror of the CNNDQN forward pass: each convolution
    accumulates in float64 over the stored float32 values, ReLU between
    layers, hidden activations rounded to float32 after each ReLU
    (matching Float32Array storage in a browser mirror), adaptive
    average pooling, heading concatenated as conditioning, then the
    two-layer head. The document is validated first, so a corrupt
    export fails closed here too. spatial is (n, 3, 12, 12), heading
    is (n, 4). Returns an (n, 3) float64 numpy array.
    """
    info = validate_cnn_checkpoint_document(doc)
    config = info["config"]
    sd = {}
    for layer in doc["layers"]:
        sd[layer["name"]] = np.asarray(layer["values"], dtype=np.float64)
    x = np.asarray(spatial, dtype=np.float32).astype(np.float64)
    h = np.asarray(heading, dtype=np.float32).astype(np.float64)
    if x.ndim != 4 or x.shape[1:] != (3, BOARD, BOARD):
        raise CNNCheckpointError(
            f"spatial must have shape (n, 3, 12, 12), got {x.shape}"
        )
    if h.ndim != 2 or h.shape[1] != 4 or h.shape[0] != x.shape[0]:
        raise CNNCheckpointError(
            f"heading must have shape (n, 4) matching spatial, got {h.shape}"
        )
    for i in range(len(config.channels)):
        w = sd[f"convs.{i}.weight"]
        b = sd[f"convs.{i}.bias"]
        x = np.maximum(_conv2d_numpy(x, w, b), 0.0)
        x = x.astype(np.float32).astype(np.float64)
    if config.residual:
        last = config.channels[-1]
        r = np.maximum(
            _conv2d_numpy(x, sd["res1.weight"], sd["res1.bias"]), 0.0
        )
        r = r.astype(np.float32).astype(np.float64)
        r = _conv2d_numpy(r, sd["res2.weight"], sd["res2.bias"])
        x = np.maximum(x + r, 0.0)
        x = x.astype(np.float32).astype(np.float64)
    x = _adaptive_avg_pool2d_numpy(x, (config.pool_size, config.pool_size))
    x = x.reshape(x.shape[0], -1)
    x = np.concatenate([x, h], axis=1)
    x = np.maximum(x @ sd["head1.weight"].T + sd["head1.bias"], 0.0)
    x = x.astype(np.float32).astype(np.float64)
    return x @ sd["head2.weight"].T + sd["head2.bias"]
