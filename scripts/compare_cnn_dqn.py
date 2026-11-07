"""Paired held-out comparison for DA8: CNN DQN vs frozen MLP, Rookie, Greedy.

Every policy plays the same keyed test episodes (namespace "test",
P0-06 split registry), so pairing is by episode key and the test split
never overlaps training: training env seeds are base seed + episode
index, test env keys are SHA-256-derived and disjoint by construction.
The frozen MLP is the committed runs/dqn_weights.json (300,000 steps,
seed 11); Rookie is the committed runs/qtable.json; Greedy is the
heuristic. CNN checkpoints are the versioned exports written by
scripts/train_cnn_dqn.py.

Output is runs/da8_comparison.json: per-episode rows for every policy,
paired differences (CNN minus each comparator) with a normal-approx
95% confidence interval, win/draw/loss counts, and a measured
inference rate (decisions per second) per policy on fixed mid-game
states. Nothing here tunes on the test split: the comparison reads it
once, at the end, and reports whatever it finds, including a null.

Usage:
    python scripts/compare_cnn_dqn.py \\
        --cnn runs/cnn_dqn_plain_seed7_weights.json \\
        --cnn runs/cnn_dqn_residual_seed7_weights.json \\
        --mlp runs/mlp_control_seed7_weights.json \\
        --n-test 200 --out runs/da8_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from narcade.experiment_seeds import check_split_use, episode_keys
from narcade.policies import GreedyPolicy, QTablePolicy
from narcade.snake import SnakeEnv
from scripts.evaluate import CNNDQNPolicy, DQNPolicy, run_keyed_episodes

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def paired_difference(a_scores, b_scores):
    """Paired stats for equal-length score lists a (candidate) vs b.

    Returns n, per-policy means, mean difference, sample std of the
    differences, standard error, a normal-approx 95% confidence
    interval (valid at the comparison size used here, n >= 100), and
    win/draw/loss counts for the candidate.
    """
    if len(a_scores) != len(b_scores):
        raise ValueError(
            f"paired lists must share a length, got {len(a_scores)} "
            f"and {len(b_scores)}"
        )
    n = len(a_scores)
    if n < 2:
        raise ValueError(f"paired comparison needs at least 2 episodes, got {n}")
    diffs = [a - b for a, b in zip(a_scores, b_scores)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var)
    se = std_diff / math.sqrt(n)
    half = 1.96 * se
    return {
        "n": n,
        "mean_a": sum(a_scores) / n,
        "mean_b": sum(b_scores) / n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": se,
        "ci_low": mean_diff - half,
        "ci_high": mean_diff + half,
        "ci_width": 2 * half,
        "wins": sum(1 for d in diffs if d > 0),
        "draws": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
    }


def benchmark_policy(act_fn, n_states=50, reps=20):
    """Decisions per second for one policy on fixed mid-game states.

    States are reached by playing the greedy heuristic 25 steps from
    fixed seeds, so every policy faces the same states. The policy is
    timed only; the environment is never stepped during timing.
    """
    states = []
    greedy = GreedyPolicy()
    for i in range(n_states):
        env = SnakeEnv(seed=1000 + i)
        for _ in range(25):
            if env.score >= 0:
                _, _, done, _ = env.step(greedy.act(env))
                if done:
                    break
        states.append(env)
    t0 = time.time()
    total = 0
    for _ in range(reps):
        for env in states:
            act_fn(env)
            total += 1
    seconds = time.time() - t0
    return {"decisions": total, "seconds": round(seconds, 3),
            "decisions_per_second": round(total / seconds, 1)
            if seconds > 0 else None}


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", action="append", default=[],
                    help="CNN weights JSON path (repeatable)")
    ap.add_argument("--mlp", action="append", default=[],
                    help="MLP control weights JSON path (repeatable: "
                    "retained MLP arms trained at the same step budget)")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(RUNS, "da8_comparison.json"))
    args = ap.parse_args()
    if not args.cnn:
        raise SystemExit("give at least one --cnn weights file")
    check_split_use("test", "final")
    keys = episode_keys("test", 0, args.n_test)

    policies = {}
    labels = {}
    for path in args.cnn:
        name = os.path.basename(path).replace("_weights.json", "")
        policies[name] = CNNDQNPolicy(path)
        labels[name] = {"kind": "cnn", "artifact": os.path.basename(path)}
    policies["mlp_frozen"] = DQNPolicy(os.path.join(RUNS, "dqn_weights.json"))
    labels["mlp_frozen"] = {"kind": "mlp", "artifact": "dqn_weights.json"}
    for path in args.mlp:
        name = os.path.basename(path).replace("_weights.json", "")
        policies[name] = DQNPolicy(path)
        labels[name] = {"kind": "mlp_control", "artifact": os.path.basename(path)}
    with open(os.path.join(RUNS, "qtable.json")) as f:
        qtable = json.load(f)
    policies["rookie"] = QTablePolicy(qtable)
    labels["rookie"] = {"kind": "qtable", "artifact": "qtable.json"}
    policies["greedy"] = GreedyPolicy()
    labels["greedy"] = {"kind": "heuristic", "artifact": None}

    rows = {}
    for name, pol in policies.items():
        print(f"evaluating {name} ({args.n_test} keyed test episodes)...",
              flush=True)
        rows[name] = run_keyed_episodes(lambda _key, p=pol: p, keys)

    benchmarks = {}
    for name, pol in policies.items():
        benchmarks[name] = benchmark_policy(pol.act)

    comparisons = {}
    cnn_names = [n for n in policies if labels[n]["kind"] == "cnn"]
    mlp_controls = [n for n in policies if labels[n]["kind"] == "mlp_control"]
    for cnn in cnn_names:
        comparisons[cnn] = {}
        a = [r["score"] for r in rows[cnn]]
        for other in ["mlp_frozen", "rookie", "greedy"] + mlp_controls:
            b = [r["score"] for r in rows[other]]
            comparisons[cnn][f"vs_{other}"] = paired_difference(a, b)

    doc = {
        "protocol": {
            "namespace": "test",
            "n_test": args.n_test,
            "episode_indices": [0, args.n_test - 1],
            "paired_by": "episode_key",
            "note": ("Keyed test episodes (P0-06). Training env seeds are "
                     "base seed + episode index; test env keys are "
                     "SHA-256-derived and disjoint by construction. The "
                     "legacy 5000..5499 eval seeds overlap training seeds "
                     "and are not used here."),
        },
        "policies": labels,
        "rows": rows,
        "comparisons": comparisons,
        "inference": benchmarks,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    for cnn in cnn_names:
        for other, stats in comparisons[cnn].items():
            print(f"{cnn} {other}: mean diff {stats['mean_diff']:+.2f} "
                  f"[{stats['ci_low']:+.2f}, {stats['ci_high']:+.2f}] "
                  f"W{stats['wins']}/D{stats['draws']}/L{stats['losses']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
