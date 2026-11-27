"""Move-level analysis: action mix, death causes, food efficiency per policy.

Saves runs/moves.json. Fast policies: 200 episodes; MCTS: fewer (it's slow).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from narcade.mcts import MCTSPlayer
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.snake import SnakeEnv
from scripts.evaluate import DQNPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def analyze_rows(policy_fn, n, seed0=0, keys=None):
    """Per-episode rows for one policy (Stage 3 P1-05).

    Each row stores the episode identity, the score, the step count
    and the failure cause (outcome: "death", "win" or "timeout"), so
    every score in the aggregate report traces back to the episode
    that produced it. With keys (P0-06 EpisodeKeys) the episodes are
    keyed: the environment is constructed with the episode key and
    played as-is. Without keys the legacy seed0+i episodes are
    played, exactly as analyze() always did."""
    rows = []
    if keys is not None:
        episodes = [(key.label, key.env_key, key) for key in keys]
    else:
        episodes = [(f"legacy:{seed0 + i}", seed0 + i, None)
                    for i in range(n)]
    for label, seed, key in episodes:
        env = SnakeEnv(seed=seed)
        if key is None:
            env.reset()
        done = False
        info = {}
        acts = Counter()
        while not done:
            a = policy_fn(env)
            acts[a] += 1
            _, _, done, info = env.step(a)
        row = {
            "episode": label,
            "score": env.score,
            "steps": env.steps,
            "outcome": info.get("reason", "?"),
            "actions": [acts[0], acts[1], acts[2]],
        }
        if key is not None:
            row["episode_key"] = key.label
            row["namespace"] = key.namespace
            row["index"] = key.index
            row["env_key"] = key.env_key
        else:
            row["seed"] = seed
        rows.append(row)
    return rows


def summarize_rows(rows):
    """Aggregate report from per-episode rows: action mix, death
    causes, food efficiency. Identical to what analyze() always
    returned; the rows are the single source now."""
    acts = [0, 0, 0]
    deaths = Counter()
    foods, steps = 0, 0
    for row in rows:
        for a in range(3):
            acts[a] += row["actions"][a]
        deaths[row["outcome"]] += 1
        foods += row["score"]
        steps += row["steps"]
    total_a = sum(acts)
    return {
        "actions": [round(acts[0] / total_a, 3), round(acts[1] / total_a, 3),
                    round(acts[2] / total_a, 3)],
        "deaths": dict(deaths),
        "food_per_100_steps": round(100 * foods / max(1, steps), 2),
        "episodes": len(rows),
    }


def analyze(policy_fn, n, seed0, mcts_sims=None):
    return summarize_rows(analyze_rows(policy_fn, n, seed0=seed0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mcts_n", type=int, default=30)
    ap.add_argument("--mcts_sims", type=int, default=80)
    args = ap.parse_args()

    with open(os.path.join(args.out, "qtable.json")) as f:
        qtable = json.load(f)
    policies = {
        "random": (RandomPolicy(seed=77).act, args.n),
        "greedy": (GreedyPolicy().act, args.n),
        "rookie": (QTablePolicy(qtable).act, args.n),
        # DQN loads from the committed JSON weights export through
        # scripts.evaluate.DQNPolicy (strict shared loader). The old
        # private loader read runs/dqn.pt, which is gitignored and
        # absent on a fresh checkout, so this script could not run.
        "pro": (DQNPolicy(os.path.join(args.out, "dqn_weights.json")).act,
                args.n),
    }
    out = {}
    detail = {}
    for name, (fn, n) in policies.items():
        print(f"analyzing {name}...", flush=True)
        detail[name] = analyze_rows(fn, n, seed0=31000)
        out[name] = summarize_rows(detail[name])
    print("analyzing grandmaster...", flush=True)
    mcts = MCTSPlayer(simulations=args.mcts_sims, seed=4242)
    detail["grandmaster"] = analyze_rows(mcts.act, args.mcts_n, seed0=33000)
    out["grandmaster"] = summarize_rows(detail["grandmaster"])
    with open(os.path.join(args.out, "moves.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, "moves_detail.json"), "w") as f:
        json.dump(detail, f, indent=2)
    for k, v in out.items():
        print(f"  {k:12s} actions={v['actions']} deaths={v['deaths']} "
              f"food/100steps={v['food_per_100_steps']}")


if __name__ == "__main__":
    main()
