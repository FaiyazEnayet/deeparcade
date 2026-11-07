"""Train the convolutional DQN for Snake (Stage 3 deeparcade DA8).

Same training protocol as scripts/train_dqn.py train_dqn, held
constant across the MLP and CNN arms: same reward shaping (the
environment step, shared), same optimizer family (Adam), same gamma,
batch, learning rate, replay capacity and update cadence, same target
sync, same epsilon schedule, same warmup, same episode-key training
partition (training_env_key), same environment steps, same hardware
(CPU). The only difference is the network and its encoder: CNNDQN over
the three spatial planes plus heading conditioning, from
src/narcade/models.py.

Usage:
    python scripts/train_cnn_dqn.py --steps 300000 --seed 7 \\
        --config-name plain --out runs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from narcade.experiment_seeds import training_env_key
from narcade.models import (
    CNNDQN,
    CNNConfig,
    encode_cnn_dqn,
    export_cnn_weights,
)
from narcade.snake import SnakeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

# Training protocol constants. These literals are the ones in
# scripts/train_dqn.py train_dqn; the stage3_deeparcade_8 test pins
# them so a drift between the two arms fails loudly instead of
# silently changing the comparison.
GAMMA = 0.95
BATCH = 64
LR = 1e-3
REPLAY_MAX = 50000
UPDATE_EVERY = 8
TARGET_SYNC = 2000
EPS_END = 0.05
EPS_DECAY_STEPS = 120000
WARMUP_STEPS = 2000

CONFIG_PRESETS = {
    "plain": CNNConfig(channels=(16, 32), residual=False),
    "residual": CNNConfig(channels=(16, 32), residual=True),
}


def train_cnn_dqn(steps=300000, seed=7, config=None, verbose=True):
    """Q-learning run with the CNN, mirroring train_dqn step for step.

    Returns (policy_net, scores100, rewards100, meta). meta carries
    steps, episodes, seed, seconds, final_mean100, the config, the
    parameter count, and decisions-per-second (environment steps over
    training wall time).
    """
    if config is None:
        config = CNNConfig()
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cpu")

    policy_net = CNNDQN(config).to(device)
    target_net = CNNDQN(config).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    opt = torch.optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    replay = []  # plain list: random.sample on a deque is O(n) per draw
    replay_max = REPLAY_MAX
    gamma, batch = GAMMA, BATCH
    eps, eps_end = 1.0, EPS_END
    eps_decay_steps = min(EPS_DECAY_STEPS, steps)
    target_sync = TARGET_SYNC
    update_every = UPDATE_EVERY

    env = SnakeEnv(seed=seed)
    s_spatial, s_heading = encode_cnn_dqn(env)
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
                q = policy_net(
                    torch.from_numpy(s_spatial).unsqueeze(0),
                    torch.from_numpy(s_heading).unsqueeze(0),
                )
                a = int(q.argmax(1).item())
        s2, r, done, _ = env.step(a)
        s2_spatial, s2_heading = encode_cnn_dqn(env)
        replay.append((s_spatial, s_heading, a, r, s2_spatial, s2_heading, done))
        if len(replay) > replay_max:
            del replay[:replay_max // 10]
        s_spatial, s_heading = s2_spatial, s2_heading
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
            s_spatial, s_heading = encode_cnn_dqn(env)
            ep_reward, ep_score = 0.0, 0

        if len(replay) >= WARMUP_STEPS and step % update_every == 0:
            idx = rng.sample(range(len(replay)), batch)
            bs = torch.from_numpy(
                np.stack([replay[i][0] for i in idx]))
            bh = torch.from_numpy(
                np.stack([replay[i][1] for i in idx]))
            ba = torch.tensor([replay[i][2] for i in idx]).unsqueeze(1)
            br = torch.tensor([replay[i][3] for i in idx])
            bs2 = torch.from_numpy(
                np.stack([replay[i][4] for i in idx]))
            bh2 = torch.from_numpy(
                np.stack([replay[i][5] for i in idx]))
            bd = torch.tensor([replay[i][6] for i in idx])
            q = policy_net(bs, bh).gather(1, ba).squeeze(1)
            with torch.no_grad():
                q_next = target_net(bs2, bh2).max(1)[0]
                target = br + gamma * q_next * (~bd)
            loss = loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % target_sync == 0:
            target_net.load_state_dict(policy_net.state_dict())

    seconds = time.time() - t0
    meta = {"steps": steps, "episodes": n_ep, "seed": seed,
            "seconds": round(seconds, 1),
            "final_mean100": scores100[-1] if scores100 else 0.0,
            "config": config.to_dict(),
            "params": policy_net.num_params(),
            "decisions_per_second": round(steps / seconds, 1)
            if seconds > 0 else None,
            "hardware": "cpu"}
    return policy_net, scores100, rewards100, meta


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
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config-name", default="plain",
                    choices=sorted(CONFIG_PRESETS))
    ap.add_argument("--out", default=RUNS)
    args = ap.parse_args()
    config = CONFIG_PRESETS[args.config_name]
    os.makedirs(args.out, exist_ok=True)
    net, scores, rewards, meta = train_cnn_dqn(
        steps=args.steps, seed=args.seed, config=config)
    stem = f"cnn_dqn_{args.config_name}_seed{args.seed}"
    torch.save(net.state_dict(), os.path.join(args.out, f"{stem}.pt"))
    commit = _git_commit()
    doc = export_cnn_weights(
        net,
        config=config,
        training_args={"steps": args.steps, "seed": args.seed,
                       "config": args.config_name},
        data_key_range={"episode_seeds": [args.seed,
                                          args.seed + meta["episodes"] - 1]},
        code_commit=commit,
    )
    with open(os.path.join(args.out, f"{stem}_weights.json"), "w") as f:
        json.dump(doc, f)
    with open(os.path.join(args.out, f"{stem}_curve.json"), "w") as f:
        json.dump({"mean_score": scores, "mean_reward": rewards}, f)
    with open(os.path.join(args.out, f"{stem}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['seconds']}s, {meta['episodes']} episodes, "
          f"final mean100={meta['final_mean100']:.2f}, "
          f"{meta['decisions_per_second']} decisions/s")


if __name__ == "__main__":
    main()
