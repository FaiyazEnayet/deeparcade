"""Statistical research benchmark (Stage 3 deeparcade P1-05).

Shared statistics for the evaluation pipeline, used by
scripts/evaluate.py, scripts/diagnose_dqn.py and scripts/analyze.py
so the three scripts cannot drift apart silently.

What this module guarantees, and what it refuses:

- Every interval is a percentile bootstrap over the stored
  per-episode differences, resampled under one fixed disclosed seed,
  so any reader recomputes the exact interval from the stored arrays.
- Every ranking carries its N and its interval next to the mean, and
  every pairwise cell carries win/draw/loss counts with N.
- Train, validation and test splits come from the P0-06 split
  registry (src/narcade/experiment_seeds.py); the benchmark checks
  the stored train and test episode keys are disjoint.
- Model selection reads validation only. Selecting on the final test
  split raises instead of quietly picking the most favorable model
  from the test data.
- For imitation, offline test-state action accuracy and on-policy
  rollouts are reported side by side, and any claim is conditional
  on the test-state distribution the offline states were drawn from.

The bootstrap follows scripts/generalization.py: 10,000 resamples of
the per-episode differences with replacement under a fixed seed.
"""

from __future__ import annotations

import statistics

import numpy as np

from narcade.experiment_seeds import check_split_use

SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20270121
CONFIDENCE = 0.95


def _check_scores(scores, name="scores"):
    if len(scores) == 0:
        raise ValueError(f"cannot summarize zero {name}")
    return [float(s) for s in scores]


def paired_bootstrap_ci(scores_a, scores_b, n_boot=BOOTSTRAP_RESAMPLES,
                        seed=BOOTSTRAP_SEED, confidence=CONFIDENCE):
    """Percentile bootstrap CI for the paired mean difference a - b.

    The two lists must already be aligned by episode (same episodes,
    same order); callers align by episode key first. Resampling is
    over episode indices with replacement under the fixed seed, so
    the interval is a deterministic function of the two stored
    arrays. Returns n, the mean difference, the interval bounds, and
    the bootstrap settings that reproduce it."""
    a = _check_scores(scores_a, "scores_a")
    b = _check_scores(scores_b, "scores_b")
    if len(a) != len(b):
        raise ValueError(
            f"paired lists must share a length, got {len(a)} and {len(b)}")
    n = len(a)
    diffs = np.asarray([x - y for x, y in zip(a, b)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    ci_low, ci_high = np.percentile(means, [tail, 100.0 - tail])
    return {
        "n": n,
        "mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": n_boot,
        "seed": seed,
        "confidence": confidence,
    }


def mean_bootstrap_ci(scores, n_boot=BOOTSTRAP_RESAMPLES,
                      seed=BOOTSTRAP_SEED, confidence=CONFIDENCE):
    """Percentile bootstrap CI for one policy's mean score, from its
    stored per-episode array under the fixed seed."""
    values = _check_scores(scores)
    n = len(values)
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    ci_low, ci_high = np.percentile(means, [tail, 100.0 - tail])
    return {
        "n": n,
        "mean": float(arr.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": n_boot,
        "seed": seed,
        "confidence": confidence,
    }


def win_draw_loss(scores_a, scores_b):
    """Win/draw/loss counts for a (candidate) against b, aligned by
    episode. The counts always sum to n."""
    a = _check_scores(scores_a, "scores_a")
    b = _check_scores(scores_b, "scores_b")
    if len(a) != len(b):
        raise ValueError(
            f"paired lists must share a length, got {len(a)} and {len(b)}")
    diffs = [x - y for x, y in zip(a, b)]
    return {
        "wins": sum(1 for d in diffs if d > 0),
        "draws": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
        "n": len(diffs),
    }


def _aligned_scores(rows_by_policy):
    """{name: [scores]} aligned by episode key, sorted by key.

    Every policy must share exactly the same episode-key set; any
    mismatch raises, so a ranking can never silently compare
    different episodes."""
    names = list(rows_by_policy.keys())
    if not names:
        raise ValueError("no policies supplied for ranking")
    maps = {}
    for name in names:
        rows = rows_by_policy[name]
        if not rows:
            raise ValueError(f"no episodes stored for policy {name!r}")
        seen = {}
        for row in rows:
            key = row["episode_key"]
            if key in seen:
                raise ValueError(f"duplicate episode key {key!r} for {name!r}")
            seen[key] = row["score"]
        maps[name] = seen
    base = set(maps[names[0]])
    for name in names[1:]:
        diff = base.symmetric_difference(set(maps[name]))
        if diff:
            raise ValueError(
                f"episode-key mismatch between {names[0]!r} and {name!r}: "
                f"symmetric difference (sorted) = {sorted(diff)}; paired "
                f"comparisons require identical episode keys"
            )
    keys = sorted(base)
    return {n: [maps[n][k] for k in keys] for n in names}, keys


def rank_with_uncertainty(rows_by_policy, n_boot=BOOTSTRAP_RESAMPLES,
                          seed=BOOTSTRAP_SEED, confidence=CONFIDENCE):
    """Rank policies by paired mean score, with uncertainty and N
    adjacent to every entry.

    Each entry carries policy, n, mean, its bootstrap interval, and a
    pairwise cell against every other policy (mean difference with
    its paired bootstrap interval plus win/draw/loss counts and n).
    Sorted by mean descending, ties broken by policy name so the
    order is deterministic."""
    scores, _ = _aligned_scores(rows_by_policy)
    names = sorted(scores, key=lambda n: (-statistics.mean(scores[n]), n))
    ranking = []
    for name in names:
        entry = {
            "policy": name,
            "n": len(scores[name]),
            "mean": statistics.mean(scores[name]),
        }
        ci = mean_bootstrap_ci(scores[name], n_boot=n_boot, seed=seed,
                               confidence=confidence)
        entry["ci_low"] = ci["ci_low"]
        entry["ci_high"] = ci["ci_high"]
        entry["pairwise"] = {}
        for other in scores:
            if other == name:
                continue
            pair_ci = paired_bootstrap_ci(scores[name], scores[other],
                                          n_boot=n_boot, seed=seed,
                                          confidence=confidence)
            counts = win_draw_loss(scores[name], scores[other])
            entry["pairwise"][other] = {
                "mean_diff": pair_ci["mean_diff"],
                "ci_low": pair_ci["ci_low"],
                "ci_high": pair_ci["ci_high"],
                "wins": counts["wins"],
                "draws": counts["draws"],
                "losses": counts["losses"],
                "n": counts["n"],
            }
        ranking.append(entry)
    return ranking


def select_best_on_validation(scores_by_policy, namespace):
    """Pick the best policy by mean score on the given split.

    The split must be the validation split: check_split_use raises
    for "test" (and for "train"), so no model is ever selected on
    the final test data. Ties break by policy name, deterministically.
    Returns the pick with its mean, n, the split it was selected on,
    and the rule used."""
    check_split_use(namespace, "tuning")
    if not scores_by_policy:
        raise ValueError("no policies supplied for model selection")
    best = None
    for name in sorted(scores_by_policy):
        scores = _check_scores(scores_by_policy[name], f"scores for {name!r}")
        mean = statistics.mean(scores)
        if best is None or mean > best["mean"]:
            best = {"selected_policy": name, "mean": mean, "n": len(scores)}
    best["selected_on"] = namespace
    best["rule"] = "highest validation mean"
    return best


def assert_disjoint_episode_keys(keys_a, keys_b, name_a="train",
                                 name_b="test"):
    """Fail visibly on any shared episode key between two splits."""
    set_a = set(keys_a)
    set_b = set(keys_b)
    overlap = sorted(set_a & set_b)
    if overlap:
        raise ValueError(
            f"episode-key overlap between {name_a!r} and {name_b!r} splits: "
            f"{len(overlap)} shared keys, first few (sorted) = {overlap[:5]}; "
            f"train and test splits must be disjoint"
        )


def _row_env_key(row):
    """The environment key a stored row was played on: the explicit
    env_key when present, else the legacy seed."""
    if "env_key" in row:
        return row["env_key"]
    if "seed" in row:
        return row["seed"]
    raise ValueError(f"stored row has no env_key or seed: {row!r}")


def build_benchmark_document(policies_rows, search_cost=None, curves=None,
                             multiseed=None, selection=None,
                             n_boot=BOOTSTRAP_RESAMPLES,
                             seed=BOOTSTRAP_SEED,
                             confidence=CONFIDENCE):
    """Assemble a schema-v1 statistical benchmark document.

    policies_rows: {policy name: [per-episode rows]}. Each row carries
    episode_key, namespace, index, score and outcome (the failure
    cause: "death", "win" or "timeout"). Rows may also carry env_key;
    legacy rows carry seed instead.
    search_cost: {policy: {steps, updates, seconds, episodes}}.
    curves: {variant: {training_steps: [...], mean_score: [...]}},
      the training-step-vs-performance curve.
    multiseed: {variant: {per_seed: {seed: mean}, across_seed_mean,
      across_seed_std, n_seeds}}.
    selection: {selected_on, selected_policy, rule}; selected_on must
      not be "test" (enforced here and by the validator).
    The ranking is computed from the stored rows, so the document
    carries the rows and the ranking side by side."""
    if selection is not None and selection.get("selected_on") == "test":
        raise ValueError(
            "benchmark selection must not be made on the test split; "
            "select on validation only"
        )
    ranking = rank_with_uncertainty(policies_rows, n_boot=n_boot, seed=seed,
                                    confidence=confidence)
    split = {}
    for name, rows in policies_rows.items():
        for row in rows:
            namespace = row["namespace"]
            split.setdefault(namespace, {"policies": [], "indices": []})
            if name not in split[namespace]["policies"]:
                split[namespace]["policies"].append(name)
            split[namespace]["indices"].append(row["index"])
    for namespace in split:
        split[namespace]["policies"] = sorted(split[namespace]["policies"])
        indices = sorted(set(split[namespace]["indices"]))
        split[namespace]["index_range"] = [indices[0], indices[-1]] if indices else []
        split[namespace]["n_episodes"] = len(split[namespace]["indices"])
        del split[namespace]["indices"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/statistical_benchmark.py",
        "bootstrap": {
            "n_boot": n_boot,
            "seed": seed,
            "confidence": confidence,
        },
        "split": split,
        "policies": {
            name: {"rows": rows} for name, rows in policies_rows.items()
        },
        "ranking": ranking,
        "search_cost": search_cost or {},
        "curves": curves or {},
        "multiseed": multiseed or {},
        "selection": selection,
    }


def validate_benchmark_document(doc):
    """Fail visibly on any inconsistency. Raises ValueError with a
    specific message; returns the document unchanged on success.

    Recomputes the ranking from the stored rows with the stored
    bootstrap settings and requires an exact match, so a tampered
    score or a tampered interval fails. Requires every ranking entry
    to carry n and both interval bounds, requires the stored train
    and test environment keys to be disjoint, and refuses a
    selection recorded against the test split."""
    if not isinstance(doc, dict):
        raise ValueError("benchmark document must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark schema_version {doc.get('schema_version')!r}; "
            f"this code reads schema_version {SCHEMA_VERSION}"
        )
    for key in ("policies", "ranking", "bootstrap"):
        if key not in doc:
            raise ValueError(f"benchmark document is missing required key {key!r}")
    bootstrap = doc["bootstrap"]
    rows_by_policy = {}
    for name, section in doc["policies"].items():
        rows = section.get("rows")
        if not rows:
            raise ValueError(f"benchmark document has no stored rows for {name!r}")
        for row in rows:
            for field in ("episode_key", "namespace", "index", "score", "outcome"):
                if field not in row:
                    raise ValueError(
                        f"stored row for {name!r} is missing {field!r}: {row!r}")
        rows_by_policy[name] = rows
    # Train and test environment keys must be disjoint.
    train_keys = [_row_env_key(r) for rows in rows_by_policy.values()
                  for r in rows if r["namespace"] == "train"]
    test_keys = [_row_env_key(r) for rows in rows_by_policy.values()
                 for r in rows if r["namespace"] == "test"]
    if train_keys and test_keys:
        assert_disjoint_episode_keys(train_keys, test_keys, "train", "test")
    # Every ranking entry carries n and its interval next to the mean.
    for entry in doc["ranking"]:
        for field in ("policy", "n", "mean", "ci_low", "ci_high", "pairwise"):
            if field not in entry:
                raise ValueError(
                    f"ranking entry for {entry.get('policy', '?')!r} is missing "
                    f"{field!r}; uncertainty and N must sit next to every ranking"
                )
        if entry["n"] != len(rows_by_policy[entry["policy"]]):
            raise ValueError(
                f"ranking n for {entry['policy']!r} is {entry['n']} but "
                f"{len(rows_by_policy[entry['policy']])} rows are stored"
            )
    rebuilt = rank_with_uncertainty(
        rows_by_policy, n_boot=bootstrap["n_boot"], seed=bootstrap["seed"],
        confidence=bootstrap["confidence"])
    if rebuilt != doc["ranking"]:
        raise ValueError(
            "stored ranking does not recompute from the stored rows; "
            "the interval or a score was changed after the document "
            "was built"
        )
    selection = doc.get("selection")
    if selection is not None and selection.get("selected_on") == "test":
        raise ValueError(
            "benchmark selection recorded on the test split; model "
            "selection reads validation only"
        )
    return doc


def imitation_offline_accuracy(net, states, actions):
    """Offline action accuracy of a cloned net on held-out states.

    states: (n, 436) float32 array of encoded states; actions: (n,)
    array of the reference policy's actions on those states. Returns
    the fraction of exact matches with n and the correct count. The
    states must come from the test split (see train_imitate); the
    accuracy is conditional on that test-state distribution, not a
    claim about states the reference policy never visits."""
    import torch

    states = np.asarray(states)
    actions = np.asarray(actions)
    if len(states) == 0:
        raise ValueError("offline accuracy needs at least one state")
    if len(states) != len(actions):
        raise ValueError(
            f"states and actions must share a length, got {len(states)} "
            f"and {len(actions)}")
    with torch.no_grad():
        predicted = net(torch.from_numpy(
            np.ascontiguousarray(states, dtype=np.float32))).argmax(1).numpy()
    correct = int((predicted == actions).sum())
    return {
        "accuracy": correct / len(actions),
        "n_states": len(actions),
        "correct": correct,
    }
