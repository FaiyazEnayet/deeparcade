"""Evaluate all AI tiers + baselines. Saves runs/eval.json.

Policies: random, greedy (heuristic), rookie (tabular Q), pro (DQN), grandmaster (MCTS).
Metrics: score arrays, mean/std/median, head-to-head win rates, ELO ratings.
All numbers are measured, seeds fixed.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from narcade.mcts import MCTSPlayer
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.snake import SnakeEnv
from scripts.train_dqn import DQN, encode_dqn

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


class DQNPolicy:
    def __init__(self, pt_path):
        self.net = DQN()
        self.net.load_state_dict(torch.load(pt_path, map_location="cpu"))
        self.net.eval()

    def act(self, env):
        import numpy as np
        with torch.no_grad():
            q = self.net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
        return int(q.argmax(1).item())


def run_episodes(policy_fn, n, seed0, width=12, height=12):
    scores = []
    for i in range(n):
        env = SnakeEnv(width=width, height=height, seed=seed0 + i)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(policy_fn(env))
        scores.append(env.score)
    return scores


def summarize(scores):
    import statistics
    return {"n": len(scores), "mean": round(statistics.mean(scores), 2),
            "std": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
            "median": statistics.median(scores),
            "max": max(scores), "min": min(scores)}


def elo_ratings(results, k=32):
    """results: {name: [scores]}. Round-robin on paired seeds; ELO from scratch."""
    names = list(results.keys())
    elo = {n: 1200.0 for n in names}
    n = len(next(iter(results.values())))
    for i in range(n):
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                sa, sb = results[names[a]][i], results[names[b]][i]
                ea = 1 / (1 + 10 ** ((elo[names[b]] - elo[names[a]]) / 400))
                ra = 1.0 if sa > sb else (0.5 if sa == sb else 0.0)
                elo[names[a]] += k * (ra - ea)
                elo[names[b]] += k * ((1 - ra) - (1 - ea))
    return {n: round(e) for n, e in elo.items()}


def head_to_head(results):
    names = list(results.keys())
    n = len(next(iter(results.values())))
    table = {}
    for a in names:
        table[a] = {}
        for b in names:
            if a == b:
                table[a][b] = None
                continue
            w = sum(1 for i in range(n) if results[a][i] > results[b][i])
            d = sum(1 for i in range(n) if results[a][i] == results[b][i])
            table[a][b] = {"win_rate": round(w / n, 3), "draw_rate": round(d / n, 3)}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--mcts_n", type=int, default=50)
    ap.add_argument("--mcts_sims", type=int, default=80)
    args = ap.parse_args()

    with open(os.path.join(args.out, "qtable.json")) as f:
        qtable = json.load(f)
    policies = {
        "random": RandomPolicy(seed=999),
        "greedy": GreedyPolicy(),
        "rookie": QTablePolicy(qtable),
        "pro": DQNPolicy(os.path.join(args.out, "dqn.pt")),
    }
    results = {}
    for name, pol in policies.items():
        print(f"evaluating {name}...", flush=True)
        results[name] = run_episodes(pol.act, args.n, seed0=5000)
    print("evaluating grandmaster (MCTS)...", flush=True)
    mcts = MCTSPlayer(simulations=args.mcts_sims, seed=31337)
    results["grandmaster"] = run_episodes(mcts.act, args.mcts_n, seed0=7000)

    # align lengths for paired comparisons (use min n)
    m = min(len(v) for v in results.values())
    aligned = {k: v[:m] for k, v in results.items()}

    out = {
        "scores": {k: v for k, v in results.items()},
        "summary": {k: summarize(v) for k, v in results.items()},
        "elo": elo_ratings(aligned),
        "head_to_head": head_to_head(aligned),
        "n_eval": args.n, "mcts_n": args.mcts_n, "mcts_sims": args.mcts_sims,
    }
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(out, f)
    print("ELO:", out["elo"])
    for k, s in out["summary"].items():
        print(f"  {k:12s} mean={s['mean']:<6} std={s['std']:<6} median={s['median']}")


if __name__ == "__main__":
    main()
