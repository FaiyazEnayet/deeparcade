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

from narcade.experiment_seeds import episode_keys, training_env_key
from narcade.snake import SnakeEnv
from narcade.policies import GreedyPolicy
from scripts.statistical_benchmark import (
    assert_disjoint_episode_keys,
    imitation_offline_accuracy,
)
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
    curve_step_points = []
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
                curve_step_points.append(step)
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
        "curve_step_points": curve_step_points,
        "curve_loss": [round(x, 4) for x in curve_loss],
        "curve_target": [round(x, 3) for x in curve_target],
        "curve_maxq": [round(x, 3) for x in curve_maxq],
        "curve_eps": [round(x, 4) for x in curve_eps],
    }, policy_net


def collect_offline_states(policy, start_index, n_states, namespace="test"):
    """Held-out (state, reference action) pairs from keyed episodes.

    Plays the reference policy on ``namespace`` episodes starting at
    ``start_index`` and records the encoded state and the reference
    action at each step until ``n_states`` pairs are collected. The
    episodes are never used for training and never overlap the
    on-policy eval episodes (which use indices below ``start_index``),
    so the offline accuracy is measured on states the clone never
    trained on. Returns (states, actions, env_keys)."""
    states, actions, env_keys = [], [], []
    index = start_index
    while len(states) < n_states:
        key = episode_keys(namespace, index, 1)[0]
        env = make_keyed_env(key)
        env_keys.append(key.env_key)
        done = False
        while not done and len(states) < n_states:
            s = encode_dqn(env)
            a = policy.act(env)
            states.append(s)
            actions.append(a)
            _, _, done, _ = env.step(a)
        index += 1
    return np.stack(states), np.array(actions), env_keys


def train_imitate(steps, seed, collect_n=20000, eval_eps=100, offline_n=200):
    """Supervised clone of the greedy policy. Compute budget matched to the
    RL variants: same number of gradient updates they perform in `steps`.

    Two evaluations, both on the P0-06 test split, which is disjoint
    from training by construction (training env keys are base seed +
    episode index; test env keys are SHA-256-derived):

    - on-policy rollouts: the cloned net plays test-namespace
      episodes (indices 0..eval_eps-1) greedily; every score is
      stored in imitate_eval_scores with its episode key.
    - offline test-state action accuracy: the net's action agreement
      with the greedy policy on held-out states visited by greedy on
      test-namespace episodes (indices eval_eps onward). The accuracy
      is conditional on that test-state distribution: it says how
      well the clone matches greedy where greedy actually goes, not
      on states greedy never visits.

    The old code evaluated on legacy seeds 10000+; those lie inside
    the training env-key range (seed + episode index, up to
    seed + collect_n), so the old eval overlapped training. That
    overlap is gone: the recorded train and offline env keys are
    checked disjoint here, and the check raises instead of silently
    reporting an overlapped score."""
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    greedy = GreedyPolicy()

    n_updates = max(1, steps // UPDATE_EVERY)
    # Collect demonstrations with the greedy policy (training split only).
    states, actions = [], []
    train_env_keys = [seed]
    env = make_env(seed)
    s = encode_dqn(env)
    while len(states) < collect_n:
        a = greedy.act(env)
        states.append(s)
        actions.append(a)
        _, _, done, _ = env.step(a)
        if done:
            next_key = training_env_key(seed, len(states))
            train_env_keys.append(next_key)
            env = make_env(next_key)
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

    # On-policy rollouts on fresh test-namespace episodes.
    eval_keys = episode_keys("test", 0, eval_eps)
    scores = []
    with torch.no_grad():
        for key in eval_keys:
            env = make_keyed_env(key)
            done = False
            while not done:
                q = net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
                _, _, done, _ = env.step(int(q.argmax(1).item()))
            scores.append(env.score)

    # Offline test-state action accuracy on later test episodes.
    off_states, off_actions, offline_env_keys = collect_offline_states(
        greedy, eval_eps, offline_n)
    offline = imitation_offline_accuracy(net, off_states, off_actions)
    offline["split"] = "test"
    offline["scope"] = (
        "conditional on the test-state distribution: states visited "
        "by the greedy policy on test-namespace episodes"
    )
    assert_disjoint_episode_keys(
        train_env_keys, offline_env_keys, "train", "test")
    return {
        "variant": "imitate",
        "seed": seed,
        "steps": steps,
        "updates": n_updates,
        "seconds": round(time.time() - t0, 1),
        "final_mean100": round(sum(scores) / len(scores), 3),
        "imitate_train_loss": losses,
        "imitate_eval_scores": scores,
        "eval_split": "test",
        "eval_episode_keys": [k.label for k in eval_keys],
        "imitate_offline_accuracy": offline,
        "train_env_keys": train_env_keys,
        "offline_env_keys": offline_env_keys,
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


def evaluate_policy_keyed(net, keys):
    """Greedy-argmax episode rows of a trained net on keyed episodes
    (Stage 3 P1-05).

    Each row carries the episode key, namespace, index, environment
    key, score and the failure cause (outcome), so every score stays
    traceable to the episode that produced it, and the same key
    reproduces the same score regardless of evaluation order."""
    rows = []
    net.eval()
    with torch.no_grad():
        for key in keys:
            env = make_keyed_env(key)
            done = False
            info = {}
            while not done:
                q = net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
                _, _, done, info = env.step(int(q.argmax(1).item()))
            rows.append({
                "episode_key": key.label,
                "namespace": key.namespace,
                "index": key.index,
                "env_key": key.env_key,
                "score": env.score,
                "outcome": info.get("reason", "death"),
            })
    net.train()
    return rows


def run_diagnosis(steps, seeds, variants, eval_seeds=None, eval_keys=None):
    """Run every variant on every seed and evaluate on the eval split.

    Pass eval_keys (P0-06 test-namespace EpisodeKeys) to evaluate on
    the test split with keyed rows; the legacy eval_seeds path is kept
    for callers that still pass plain seeds, and the recorded
    eval_split says which one was used. Each result records its search
    cost (steps, updates, seconds, episodes) next to its scores, and
    the training-step-vs-performance curve points travel in the
    per-run curve_score / curve_step_points pair."""
    if eval_keys is None and eval_seeds is None:
        eval_seeds = list(range(5000, 5000 + 100))
    results = []
    for variant in variants:
        for sd in seeds:
            if variant == "imitate":
                summary, net = train_imitate(steps, sd)
            else:
                summary, net = train_rl(variant, steps, sd)
            if eval_keys is not None:
                rows = evaluate_policy_keyed(net, eval_keys)
                summary["eval_scores"] = [r["score"] for r in rows]
                summary["eval_episode_keys"] = [r["episode_key"] for r in rows]
                summary["eval_rows"] = rows
                summary["eval_split"] = "test"
            else:
                summary["eval_scores"] = evaluate_policy(net, eval_seeds)
                summary["eval_episode_keys"] = [f"legacy:{s}" for s in eval_seeds]
                summary["eval_split"] = "legacy"
            summary["eval_mean"] = round(
                sum(summary["eval_scores"]) / len(summary["eval_scores"]), 3)
            summary["search_cost"] = {
                "steps": steps,
                "updates": summary["updates"],
                "seconds": summary["seconds"],
                "episodes": summary.get("episodes"),
            }
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
    # Final evaluation reads the test split only, on keyed episodes
    # disjoint from training by construction (P0-06 registry). The
    # legacy 5000+ seed range overlaps the training env-key range and
    # must not be used for a final claim.
    eval_keys = episode_keys("test", 0, args.eval_n)
    results = run_diagnosis(args.steps, args.seeds, args.variants,
                            eval_keys=eval_keys)
    import statistics as _statistics
    multiseed = {}
    curves = {}
    search_cost_total = {"steps": 0, "updates": 0, "seconds": 0.0,
                         "episodes": 0}
    for variant in args.variants:
        variant_results = [r for r in results if r["variant"] == variant]
        means = {str(r["seed"]): r["eval_mean"] for r in variant_results}
        multiseed[variant] = {
            "per_seed": means,
            "across_seed_mean": round(_statistics.mean(means.values()), 3),
            "across_seed_std": round(
                _statistics.pstdev(means.values()), 3) if len(means) > 1 else 0.0,
            "n_seeds": len(means),
        }
        for r in variant_results:
            curves[f"{variant}/seed{r['seed']}"] = {
                "training_steps": r.get("curve_step_points", []),
                "mean_score": r.get("curve_score", []),
            }
        for r in variant_results:
            cost = r["search_cost"]
            search_cost_total["steps"] += cost["steps"]
            search_cost_total["updates"] += cost["updates"]
            search_cost_total["seconds"] += cost["seconds"]
            search_cost_total["episodes"] += cost["episodes"] or 0
    payload = {
        "steps_per_run": args.steps,
        "seeds": args.seeds,
        "variants": args.variants,
        "eval_split": "test",
        "eval_episode_keys": [k.label for k in eval_keys],
        "gamma": GAMMA,
        "batch": BATCH,
        "lr": LR,
        "results": results,
        "multiseed": multiseed,
        "curves": curves,
        "search_cost_total": search_cost_total,
    }
    path = os.path.join(args.out, "da4_diagnosis.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
