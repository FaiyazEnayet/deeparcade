"""Hard split registry / RNG lifecycle (Stage 3 deeparcade P0-06).

Every episode gets an immutable episode identity and an environment key
sampled from a named disjoint namespace. Keys are derived, never drawn
from a shared stream, so no two namespaces can intersect and no episode
can inherit unknown RNG state from an earlier episode.

Namespaces and their protocol (enforced by :func:`check_split_use`):
- ``train``: training only.
- ``validation``: development tuning reads validation only.
- ``test``: final evaluation only. The test split is write-protected as
  a protocol: it is sealed from optimization, not from the user. Nothing
  here hides test data; the seal is the rule that tuning must never read
  it, enforced by raising instead of silently allowing it.

Derivation: ``derive_key`` hashes ``deeparcade-v1:<namespace>:<index>:
<domain>`` with SHA-256 and takes the first 8 bytes as an unsigned int.
The namespace and the domain (``env`` vs ``policy``) are part of the
hashed input, so keys from different namespaces or domains are disjoint
by construction, not by hoping two ranges do not overlap.

Legacy conventions, kept for the historical record (see
``runs/splits_manifest.json``):
- ``train.py`` numbers training episodes from 1, so with its defaults
  (seed 7, 30000 episodes) it uses environment seeds 8..30007.
- ``scripts/evaluate.py`` uses independent seeds 5000..5499 by default.
  Those 500 evaluation seeds lie inside the legacy training range: the
  old results carry training-seed overlap and must not be compared with
  keyed episodes.
- Pre-v2 MCTS was evaluated on unpaired seeds (7000+) while the other
  policies used 5000+; that mismatch is why the old MCTS scores cannot
  be reproduced from current code.

The legacy runners are unchanged: :func:`scripts.evaluate.run_episodes`
still plays the constructor opening followed by a second reset draw, and
policies created once still share one RNG stream across episodes. Keyed
episodes are the new path: :func:`scripts.evaluate.run_keyed_episodes`
builds one fresh policy per episode from the episode's policy key and
plays exactly the constructor opening, so the same (episode key, policy
id, model id) reproduces the same outcome regardless of evaluation
order.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

NAMESPACES = ("train", "validation", "test")
DOMAINS = ("env", "policy")

# Purpose -> the one namespace allowed to serve it. Anything else raises.
_PURPOSE_NAMESPACE = {
    "training": "train",
    "tuning": "validation",
    "final": "test",
}

HISTORICAL_LABEL = "historical, training-seed overlap / MCTS mismatch"


def derive_key(namespace, index, domain):
    """Deterministic RNG key for one (namespace, index, domain) triple.

    Domain separation is in the hashed input: ``env`` keys and ``policy``
    keys for the same episode can never coincide, and neither can keys
    from two namespaces.
    """
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown split namespace {namespace!r}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown key domain {domain!r}")
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"episode index must be a non-negative int, got {index!r}")
    payload = f"deeparcade-v1:{namespace}:{index}:{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class EpisodeKey:
    """Immutable identity of one episode: its namespace, index, and the
    two independent RNG keys (environment stream, policy stream)."""

    namespace: str
    index: int
    env_key: int
    policy_key: int

    @property
    def label(self):
        return f"{self.namespace}:{self.index}"


def episode_keys(namespace, start, count):
    """``count`` episode keys starting at ``start`` in ``namespace``."""
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown split namespace {namespace!r}")
    if count < 0:
        raise ValueError(f"episode count must be non-negative, got {count!r}")
    return [
        EpisodeKey(
            namespace=namespace,
            index=i,
            env_key=derive_key(namespace, i, "env"),
            policy_key=derive_key(namespace, i, "policy"),
        )
        for i in range(start, start + count)
    ]


def training_env_key(base_seed, episode_index):
    """Legacy training environment seed, centralized without changing it.

    ``scripts/train.py`` builds one environment per episode as
    ``SnakeEnv(seed=base_seed + episode_index)`` with episodes numbered
    from 1; ``scripts/train_dqn.py`` and ``scripts/diagnose_dqn.py``
    advance episodes the same way. This helper names that derivation so
    the split registry can reason about it; it does not change any value
    a past run produced.
    """
    return base_seed + episode_index


def legacy_training_seed_range(seed=7, episodes=30000):
    """Environment seed range of ``train.py`` at the given defaults:
    episodes numbered from 1, so ``seed + 1`` through ``seed + episodes``."""
    return (seed + 1, seed + episodes)


def legacy_eval_seed_range(n=500):
    """Independent evaluation seed range of ``scripts/evaluate.py``:
    5000 through ``5000 + n - 1``."""
    return (5000, 5000 + n - 1)


def check_split_use(namespace, purpose):
    """Protocol gate: tuning reads validation only, final reads test
    only, training reads train only. Any other pairing raises
    ValueError instead of silently letting tuning touch the test split."""
    if purpose not in _PURPOSE_NAMESPACE:
        raise ValueError(f"unknown split purpose {purpose!r}")
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown split namespace {namespace!r}")
    allowed = _PURPOSE_NAMESPACE[purpose]
    if namespace != allowed:
        raise ValueError(
            f"split protocol violation: purpose {purpose!r} may only read "
            f"the {allowed!r} split, not {namespace!r}"
        )


def build_splits_manifest():
    """The committed ``runs/splits_manifest.json`` document, built from
    this module so the file and the code can never disagree. Sample keys
    are the first three indices of each namespace in each domain, so any
    reader can recompute them with :func:`derive_key`."""
    sample_keys = []
    for namespace in NAMESPACES:
        for index in range(3):
            for domain in DOMAINS:
                sample_keys.append(
                    {
                        "namespace": namespace,
                        "index": index,
                        "domain": domain,
                        "key": derive_key(namespace, index, domain),
                    }
                )
    train_lo, train_hi = legacy_training_seed_range()
    eval_lo, eval_hi = legacy_eval_seed_range()
    return {
        "version": 1,
        "derivation": (
            "sha256('deeparcade-v1:<namespace>:<index>:<domain>') "
            "first 8 bytes, unsigned"
        ),
        "namespaces": {
            "train": {
                "purpose": "training only",
                "write_protected": False,
            },
            "validation": {
                "purpose": "tuning only",
                "write_protected": False,
            },
            "test": {
                "purpose": "final evaluation only",
                "write_protected": True,
            },
        },
        "protocol": (
            "Development tuning reads validation only. The test split is "
            "write-protected as a protocol: it is sealed from optimization, "
            "not from the user. check_split_use in "
            "src/narcade/experiment_seeds.py enforces this."
        ),
        "historical_results": {
            "label": HISTORICAL_LABEL,
            "note": (
                f"Legacy evaluation seeds {eval_lo}..{eval_hi} lie inside "
                f"the legacy training seed range {train_lo}..{train_hi} "
                "(train.py defaults seed 7, 30000 episodes), so all 500 "
                "legacy evaluation episodes reuse training episode seeds. "
                "Pre-v2 MCTS was evaluated on unpaired seeds (7000+) while "
                "the other policies used 5000+. Those results are preserved "
                "as historical and are not comparable to keyed episodes."
            ),
        },
        "sample_keys": sample_keys,
    }


def load_splits_manifest(path):
    """Load and validate a splits manifest. Raises ValueError on any
    inconsistency: wrong version, missing namespace, an unprotected test
    split, or a sample key that does not recompute."""
    if not os.path.exists(path):
        raise ValueError(f"splits manifest missing: {path}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != 1:
        raise ValueError(
            f"unsupported splits manifest version {manifest.get('version')!r}"
        )
    namespaces = manifest.get("namespaces", {})
    if set(namespaces) != set(NAMESPACES):
        raise ValueError(
            f"splits manifest namespaces must be {sorted(NAMESPACES)}, "
            f"got {sorted(namespaces)}"
        )
    if namespaces["test"].get("write_protected") is not True:
        raise ValueError("splits manifest test split must be write-protected")
    for sample in manifest.get("sample_keys", []):
        recomputed = derive_key(
            sample["namespace"], sample["index"], sample["domain"]
        )
        if recomputed != sample["key"]:
            raise ValueError(
                f"splits manifest sample key mismatch for "
                f"{sample['namespace']}:{sample['index']}:{sample['domain']}"
            )
    return manifest
