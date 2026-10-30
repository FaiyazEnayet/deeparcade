"""Train the Snake Q-learning agent and save everything the site needs.

Outputs (runs/):
  qtable.json   - the trained Q-table (32 x 3)
  policy.json   - {best_action: [...]} for the browser game
  curve.json    - per-100-episode mean score and mean reward (learning curve)
  elo.json      - AI ELO vs fixed baselines at training checkpoints
  train_meta.json - hyperparameters, seeds, timing, final stats
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

from narcade.experiment_seeds import training_env_key
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.qlearn import QLearningAgent
from narcade.snake import SnakeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def play_episode(env, policy):
    env.reset()
    total = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(policy.act(env))
        total += r
    return env.score


def train(episodes=30000, seed=7, eps_start=1.0, eps_end=0.05,
          checkpoint_every=2000, elo_games=50, width=12, height=12,
          verbose=True):
    t0 = time.time()
    agent = QLearningAgent(epsilon=eps_start, seed=seed)
    eps_decay = (eps_start - eps_end) / episodes

    curve_scores, curve_rewards = [], []
    ep_scores, ep_rewards = [], []
    elo_hist = []
    elo = 1200.0
    K = 32.0

    baselines = {"random": RandomPolicy(seed=1234), "greedy": GreedyPolicy()}

    for ep in range(1, episodes + 1):
        env = SnakeEnv(width=width, height=height,
                       seed=training_env_key(seed, ep))
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            ep_r += r
        ep_scores.append(env.score)
        ep_rewards.append(ep_r)
        agent.epsilon = max(eps_end, agent.epsilon - eps_decay)

        if ep % 100 == 0:
            curve_scores.append(sum(ep_scores[-100:]) / 100.0)
            curve_rewards.append(sum(ep_rewards[-100:]) / 100.0)

        if ep % checkpoint_every == 0:
            policy = QTablePolicy(agent.q)
            for name, base in baselines.items():
                for g in range(elo_games):
                    e1 = SnakeEnv(width=width, height=height,
                                  seed=900000 + ep + g)
                    e2 = SnakeEnv(width=width, height=height,
                                  seed=900000 + ep + g)
                    s_ai = play_episode(e1, policy)
                    s_base = play_episode(e2, base)
                    actual = 1.0 if s_ai > s_base else (0.5 if s_ai == s_base else 0.0)
                    elo += K * (actual - expected_score(elo, 1200.0))
            elo_hist.append({"episode": ep, "elo": elo})
            if verbose:
                print(f"ep {ep:6d}  mean100={curve_scores[-1]:5.2f}  "
                      f"eps={agent.epsilon:.3f}  elo={elo:.0f}", flush=True)

    meta = {
        "episodes": episodes, "seed": seed, "width": width, "height": height,
        "eps_start": eps_start, "eps_end": eps_end,
        "checkpoint_every": checkpoint_every, "elo_games": elo_games,
        "seconds": round(time.time() - t0, 1),
        "final_mean100_score": curve_scores[-1],
        "final_elo": elo,
    }
    return agent, curve_scores, curve_rewards, elo_hist, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=RUNS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    agent, scores, rewards, elo_hist, meta = train(episodes=args.episodes,
                                                  seed=args.seed)
    policy = QTablePolicy(agent.q)
    with open(os.path.join(args.out, "qtable.json"), "w") as f:
        json.dump(agent.q, f)
    with open(os.path.join(args.out, "policy.json"), "w") as f:
        json.dump(policy.to_jsonable(), f)
    with open(os.path.join(args.out, "curve.json"), "w") as f:
        json.dump({"mean_score": scores, "mean_reward": rewards}, f)
    with open(os.path.join(args.out, "elo.json"), "w") as f:
        json.dump(elo_hist, f)
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['seconds']}s -> {args.out}")
    print(f"final mean100 score={meta['final_mean100_score']:.2f}  "
          f"elo={meta['final_elo']:.0f}")


if __name__ == "__main__":
    main()
