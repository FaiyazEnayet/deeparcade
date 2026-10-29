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


def analyze(policy_fn, n, seed0, mcts_sims=None):
    acts = Counter()
    deaths = Counter()
    foods, steps = 0, 0
    for i in range(n):
        env = SnakeEnv(seed=seed0 + i)
        env.reset()
        done = False
        while not done:
            a = policy_fn(env)
            acts[a] += 1
            _, _, done, info = env.step(a)
            steps += 1
        deaths[info.get("reason", "?")] += 1
        foods += env.score
    total_a = sum(acts.values())
    return {
        "actions": [round(acts[0] / total_a, 3), round(acts[1] / total_a, 3),
                    round(acts[2] / total_a, 3)],
        "deaths": dict(deaths),
        "food_per_100_steps": round(100 * foods / max(1, steps), 2),
        "episodes": n,
    }


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
    for name, (fn, n) in policies.items():
        print(f"analyzing {name}...", flush=True)
        out[name] = analyze(fn, n, seed0=31000)
    print("analyzing grandmaster...", flush=True)
    mcts = MCTSPlayer(simulations=args.mcts_sims, seed=4242)
    out["grandmaster"] = analyze(mcts.act, args.mcts_n, seed0=33000)
    with open(os.path.join(args.out, "moves.json"), "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"  {k:12s} actions={v['actions']} deaths={v['deaths']} "
              f"food/100steps={v['food_per_100_steps']}")


if __name__ == "__main__":
    main()
