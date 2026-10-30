"""DA4: Diagnose the DQN failure before scaling it.

The committed DQN (Pro tier) scores ~5 vs greedy ~20 and tabular Q ~21 on
paired games. This script instruments training and runs ablations with a
FIXED compute/data budget so the failure can be diagnosed honestly:

  variants:
    dqn      - baseline, current training code
    ddqn     - double-DQN target (policy net picks argmax, target net evaluates)
    noshape  - sparse rewards only (+10 food, -10 death, 0 otherwise)
    imitate  - supervised clone of the greedy policy (architecture sanity check:
               if imitation scores well, the net can represent a good policy
               and the failure is RL-specific)

Instrumentation per run: episode scores, TD loss, Bellman target magnitude,
max |Q|, epsilon schedule, death fraction, food per episode, all across the
seeds given. Output is JSON written to runs/da4_diagnosis.json.

Usage:
    python scripts/diagnose_dqn.py --steps 60000 --seeds 7 8 9
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from narcade.experiment_seeds import training_env_key
from narcade.snake import SnakeEnv
from narcade.policies import GreedyPolicy
from scripts.train_dqn import DQN, INPUT_DIM, encode_dqn

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

GAMMA = 0.95
BATCH = 64
LR = 1e-3
REPLAY_MAX = 50000
UPDATE_EVERY = 8
TARGET_SYNC = 2000
EPS_END = 0.05
EPS_DECAY_STEPS = 120000


def sparse_reward(env, reward, done, info, score_before):
    """Strip the dense shaping: +10 on food, -10 on death, 0 otherwise."""
    if env.score > score_before:
        return 10.0
    if done and info.get("reason") == "death":
        return -10.0
    return 0.0


def make_env(seed):
    return SnakeEnv(seed=seed)


def make_keyed_env(episode_key):
    """Environment for one keyed episode (Stage 3 P0-06): constructed
    with the episode's environment key and NOT reset again, so the
    constructor opening is the counted episode opening."""
    return SnakeEnv(seed=episode_key.env_key)


def train_rl(variant, steps, seed, log_every_eps=100, log_every_steps=2000):
    """Instrumented Q-learning run. variant in {'dqn', 'ddqn', 'noshape'}."""
    assert variant in ("dqn", "ddqn", "noshape")
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    opt = torch.optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    replay = []
    eps_decay = min(EPS_DECAY_STEPS, steps)

    env = make_env(seed)
    s = encode_dqn(env)
    score_before = env.score

    n_ep = 0
    ep_scores, ep_deaths, ep_foods = [], [], []
    b_scores, b_deaths, b_foods = [], [], []
    curve_score, curve_loss, curve_target, curve_maxq, curve_eps = [], [], [], [], []
    loss_acc, target_acc, maxq_acc, loss_n = 0.0, 0.0, 0.0, 0
    updates = 0

    for step in range(1, steps + 1):
        eps = max(EPS_END, 1.0 - (1.0 - EPS_END) * step / eps_decay)
        if rng.random() < eps:
            a = rng.choice([0, 1, 2])
        else:
            with torch.no_grad():
                a = int(policy_net(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())
        s2, r, done, info = env.step(a)
        s2v = encode_dqn(env)
        if variant == "noshape":
            r = sparse_reward(env, r, done, info, score_before)
        score_before = env.score
        replay.append((s, a, r, s2v, done))
        if len(replay) > REPLAY_MAX:
            del replay[:REPLAY_MAX // 10]
        s = s2v

        if done:
            n_ep += 1
            b_scores.append(env.score)
            b_deaths.append(1 if info.get("reason") == "death" else 0)
            b_foods.append(env.score)
            if n_ep % log_every_eps == 0:
                curve_score.append(sum(b_scores[-log_every_eps:]) / log_every_eps)
                ep_deaths.append(sum(b_deaths[-log_every_eps:]) / log_every_eps)
                ep_foods.append(sum(b_foods[-log_every_eps:]) / log_every_eps)
            env = make_env(training_env_key(seed, n_ep))
            s = encode_dqn(env)
            score_before = env.score

        if len(replay) >= 2000 and step % UPDATE_EVERY == 0:
            idx = rng.sample(range(len(replay)), BATCH)
            bs = torch.from_numpy(np.stack([replay[i][0] for i in idx]))
            ba = torch.tensor([replay[i][1] for i in idx]).unsqueeze(1)
            br = torch.tensor([replay[i][2] for i in idx], dtype=torch.float32)
            bs2 = torch.from_numpy(np.stack([replay[i][3] for i in idx]))
            bd = torch.tensor([replay[i][4] for i in idx])
            q = policy_net(bs).gather(1, ba).squeeze(1)
            with torch.no_grad():
                if variant == "ddqn":
                    best_a = policy_net(bs2).argmax(1, keepdim=True)
                    q_next = target_net(bs2).gather(1, best_a).squeeze(1)
                else:
                    q_next = target_net(bs2).max(1)[0]
                target = br + GAMMA * q_next * (~bd)
            loss = loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            loss_acc += loss.item()
            target_acc += target.abs().mean().item()
            maxq_acc += q.abs().max().item()
            loss_n += 1

        if step % TARGET_SYNC == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % log_every_steps == 0:
            curve_eps.append(round(eps, 4))
            if loss_n:
                curve_loss.append(loss_acc / loss_n)
                curve_target.append(target_acc / loss_n)
                curve_maxq.append(maxq_acc / loss_n)
            else:
                curve_loss.append(0.0)
                curve_target.append(0.0)
                curve_maxq.append(0.0)
            loss_acc = target_acc = maxq_acc = 0.0
            loss_n = 0

    return {
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "episodes": n_ep,
        "updates": updates,
        "seconds": round(time.time() - t0, 1),
        "final_mean100": round(curve_score[-1], 3) if curve_score else 0.0,
        "final_mean_loss": round(curve_loss[-1], 4) if curve_loss else 0.0,
        "final_mean_abs_target": round(curve_target[-1], 3) if curve_target else 0.0,
        "final_mean_maxq": round(curve_maxq[-1], 3) if curve_maxq else 0.0,
        "death_frac": round(ep_deaths[-1], 3) if ep_deaths else 0.0,
        "curve_score": [round(x, 3) for x in curve_score],
        "curve_loss": [round(x, 4) for x in curve_loss],
        "curve_target": [round(x, 3) for x in curve_target],
        "curve_maxq": [round(x, 3) for x in curve_maxq],
        "curve_eps": [round(x, 4) for x in curve_eps],
    }, policy_net


def train_imitate(steps, seed, collect_n=20000, eval_eps=100):
    """Supervised clone of the greedy policy. Compute budget matched to the
    RL variants: same number of gradient updates they perform in `steps`."""
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    greedy = GreedyPolicy()

    n_updates = max(1, steps // UPDATE_EVERY)
    # Collect demonstrations with the greedy policy.
    states, actions = [], []
    env = make_env(seed)
    s = encode_dqn(env)
    while len(states) < collect_n:
        a = greedy.act(env)
        states.append(s)
        actions.append(a)
        _, _, done, _ = env.step(a)
        if done:
            env = make_env(training_env_key(seed, len(states)))
        s = encode_dqn(env)
    states = np.stack(states)
    actions = np.array(actions)

    net = DQN()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for u in range(n_updates):
        idx = rng.sample(range(len(states)), BATCH)
        bs = torch.from_numpy(states[idx])
        ba = torch.tensor(actions[idx])
        logits = net(bs)
        loss = loss_fn(logits, ba)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if u % 20 == 0:
            losses.append(round(loss.item(), 4))

    # Evaluate the cloned policy on fresh episodes.
    scores = []
    with torch.no_grad():
        for e in range(eval_eps):
            env = make_env(10_000 + e)
            env.reset()
            done = False
            while not done:
                q = net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
                _, _, done, _ = env.step(int(q.argmax(1).item()))
            scores.append(env.score)
    return {
        "variant": "imitate",
        "seed": seed,
        "steps": steps,
        "updates": n_updates,
        "seconds": round(time.time() - t0, 1),
        "final_mean100": round(sum(scores) / len(scores), 3),
        "imitate_train_loss": losses,
        "imitate_eval_scores": scores,
    }, net


def evaluate_policy(net, seeds):
    """Greedy-Argmax episode scores of a trained net on fixed seeds."""
    scores = []
    net.eval()
    with torch.no_grad():
        for sd in seeds:
            env = make_env(sd)
            env.reset()
            done = False
            while not done:
                q = net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
                _, _, done, _ = env.step(int(q.argmax(1).item()))
            scores.append(env.score)
    net.train()
    return scores


def run_diagnosis(steps, seeds, variants, eval_seeds):
    results = []
    for variant in variants:
        for sd in seeds:
            if variant == "imitate":
                summary, net = train_imitate(steps, sd)
            else:
                summary, net = train_rl(variant, steps, sd)
            summary["eval_scores"] = evaluate_policy(net, eval_seeds)
            summary["eval_mean"] = round(
                sum(summary["eval_scores"]) / len(summary["eval_scores"]), 3)
            results.append(summary)
            print(f"done {variant} seed={sd} eval_mean={summary['eval_mean']} "
                  f"final100={summary['final_mean100']}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    ap.add_argument("--variants", nargs="+",
                    default=["dqn", "ddqn", "noshape", "imitate"])
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--out", default=RUNS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eval_seeds = list(range(5000, 5000 + args.eval_n))
    results = run_diagnosis(args.steps, args.seeds, args.variants, eval_seeds)
    payload = {
        "steps_per_run": args.steps,
        "seeds": args.seeds,
        "variants": args.variants,
        "eval_seeds": eval_seeds,
        "gamma": GAMMA,
        "batch": BATCH,
        "lr": LR,
        "results": results,
    }
    path = os.path.join(args.out, "da4_diagnosis.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
