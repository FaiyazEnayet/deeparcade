"""Deep Q-Network for Snake: the "Pro" tier.

Real DQN: experience replay, target network, epsilon decay. Trains on the
full 12x12 grid as 3 binary channels (head, body, food) + heading one-hot.
Weights are exported to JSON so the exact trained network runs in the browser.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from narcade.snake import SnakeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

INPUT_DIM = 12 * 12 * 3 + 4
N_ACTIONS = 3


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
            env = SnakeEnv(seed=seed + n_ep)
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


def export_weights(net):
    sd = net.state_dict()
    return {
        "input_dim": INPUT_DIM,
        "layers": [
            {"w": sd["net.0.weight"].tolist(), "b": sd["net.0.bias"].tolist()},
            {"w": sd["net.2.weight"].tolist(), "b": sd["net.2.bias"].tolist()},
            {"w": sd["net.4.weight"].tolist(), "b": sd["net.4.bias"].tolist()},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=RUNS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    net, scores, rewards, meta = train_dqn(steps=args.steps, seed=args.seed)
    torch.save(net.state_dict(), os.path.join(args.out, "dqn.pt"))
    with open(os.path.join(args.out, "dqn_weights.json"), "w") as f:
        json.dump(export_weights(net), f)
    with open(os.path.join(args.out, "dqn_curve.json"), "w") as f:
        json.dump({"mean_score": scores, "mean_reward": rewards}, f)
    with open(os.path.join(args.out, "dqn_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['seconds']}s, {meta['episodes']} episodes, "
          f"final mean100={meta['final_mean100']:.2f}")


if __name__ == "__main__":
    main()
