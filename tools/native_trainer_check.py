#!/usr/bin/env python3
"""CPP-06a parity driver: scripts/train.py vs the native trainer.

Runs the genuine Python train() (scripts/train.py, unmodified) and the
native train_main on identical configurations and compares EVERY
protocol output:

  * per-episode scores (exact) and rewards (exact) — captured from the
    Python run by a passive RecordingEnv subclass swapped into
    scripts.train's namespace; training envs are identified by their
    seeds (seed+ep), Elo envs (>= 900000) are filtered out
  * final Q-table: max abs diff (gate <= 1e-12, aiming for exact)
  * learning curves: same checkpoints, values within 1e-12
  * checkpoint Elo history: same order, values within 1e-12
  * policy.json: exact best_action for all 32 states
  * train_meta.json: every deterministic field equal ("seconds" is
    wall time and is the one field allowed to differ)
  * the agent's final RNG state (CPython getstate words + index)
    versus the native trainer's final PythonRandom state — direct
    proof the RNG consumption sequence is identical

Configs: two small pinned ones (different seeds/checkpoints), then the
full default protocol (train.py defaults: 30000 episodes, seed 7,
checkpoint_every 2000, elo_games 50).

Usage: native_trainer_check.py --trainer /path/to/train_main
Exit code is non-zero on the first failed gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.train as train_mod          # noqa: E402
from narcade.policies import QTablePolicy  # noqa: E402
from narcade.snake import SnakeEnv         # noqa: E402

RECORDS: list = []


class RecordingEnv(SnakeEnv):
    """Passive observer: records (seed, score, reward_sum) per finished
    episode without changing any behavior or RNG consumption."""

    def __init__(self, width=12, height=12, seed=None, max_steps=None):
        self._train_seed = seed
        self._rsum = 0.0
        super().__init__(width=width, height=height, seed=seed,
                         max_steps=max_steps)

    def reset(self):
        self._rsum = 0.0
        return super().reset()

    def step(self, action):
        s, r, done, info = super().step(action)
        self._rsum += r
        if done:
            RECORDS.append((self._train_seed, self.score, self._rsum))
        return s, r, done, info


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def run_python(cfg):
    RECORDS.clear()
    train_mod.SnakeEnv = RecordingEnv
    try:
        agent, cs, cr, elo_hist, meta = train_mod.train(
            episodes=cfg["episodes"], seed=cfg["seed"],
            checkpoint_every=cfg["checkpoint_every"],
            elo_games=cfg["elo_games"], verbose=False)
    finally:
        train_mod.SnakeEnv = SnakeEnv
    stream = [(sc, rw) for (sd, sc, rw) in RECORDS
              if sd is not None and sd < 900000]
    if len(stream) != cfg["episodes"]:
        fail(f"python stream length {len(stream)} != {cfg['episodes']}")
    return agent, cs, cr, elo_hist, meta, stream


def run_native(trainer, cfg, workdir):
    out = Path(workdir) / "out"
    dump = Path(workdir) / "episodes.json"
    cmd = [str(trainer),
           "--episodes", str(cfg["episodes"]),
           "--seed", str(cfg["seed"]),
           "--out", str(out),
           "--checkpoint-every", str(cfg["checkpoint_every"]),
           "--elo-games", str(cfg["elo_games"]),
           "--quiet",
           "--dump-episodes", str(dump)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        fail(f"native trainer exited {res.returncode}: {res.stderr}")
    load = lambda name: json.loads((out / name).read_text())
    return (load("qtable.json"), load("policy.json"), load("curve.json"),
            load("elo.json"), load("train_meta.json"),
            json.loads(dump.read_text()))


def check_config(trainer, name, cfg):
    agent, cs, cr, elo_hist, meta, stream = run_python(cfg)
    with tempfile.TemporaryDirectory() as td:
        q, policy, curve, elo, nmeta, dump = run_native(trainer, cfg, td)

    # Per-episode stream: scores and rewards EXACT.
    p_scores = [s for s, _ in stream]
    p_rewards = [r for _, r in stream]
    if dump["scores"] != p_scores:
        fail(f"[{name}] episode scores differ")
    worst_r = max((abs(a - b) for a, b in zip(dump["rewards"], p_rewards)),
                  default=0.0)
    if dump["rewards"] != p_rewards:
        fail(f"[{name}] episode rewards not exact (worst {worst_r:.3e})")

    # Final Q-table.
    worst_q = max(abs(agent.q[s][a] - q[s][a])
                  for s in range(32) for a in range(3))
    if worst_q > 1e-12:
        fail(f"[{name}] Q-table worst diff {worst_q:.3e} > 1e-12")

    # Learning curves.
    if len(curve["mean_score"]) != len(cs) or len(curve["mean_reward"]) != len(cr):
        fail(f"[{name}] curve lengths differ")
    worst_c = max([abs(a - b) for a, b in zip(curve["mean_score"], cs)] +
                  [abs(a - b) for a, b in zip(curve["mean_reward"], cr)],
                  default=0.0)
    if worst_c > 1e-12:
        fail(f"[{name}] curve worst diff {worst_c:.3e} > 1e-12")

    # Checkpoint Elo history: order and values.
    if len(elo) != len(elo_hist):
        fail(f"[{name}] elo history length differs")
    worst_e = 0.0
    for got, want in zip(elo, elo_hist):
        if got["episode"] != want["episode"]:
            fail(f"[{name}] elo checkpoint order differs")
        worst_e = max(worst_e, abs(got["elo"] - want["elo"]))
    if worst_e > 1e-12:
        fail(f"[{name}] elo worst diff {worst_e:.3e} > 1e-12")

    # policy.json exact.
    if policy != QTablePolicy(agent.q).to_jsonable():
        fail(f"[{name}] policy.json differs")

    # train_meta.json: deterministic fields ("seconds" exempt).
    for key, want in meta.items():
        if key == "seconds":
            continue
        got = nmeta[key]
        if isinstance(want, float):
            if abs(got - want) > 1e-12:
                fail(f"[{name}] meta[{key}] {got} != {want}")
        elif got != want:
            fail(f"[{name}] meta[{key}] {got} != {want}")

    # Agent RNG final state: CPython getstate vs native PythonRandom.
    state = agent.rng.getstate()
    words, index = state[1][:624], state[1][624]
    if dump["final_rng_state"]["mt"] != list(words):
        fail(f"[{name}] final RNG words differ")
    if dump["final_rng_state"]["index"] != index:
        fail(f"[{name}] final RNG index differs")
    if dump["final_epsilon"] != agent.epsilon:
        fail(f"[{name}] final epsilon differs")

    print(f"[{name}] PASS  episodes={cfg['episodes']} "
          f"worst|dQ|={worst_q:.3e} worst|dcurve|={worst_c:.3e} "
          f"worst|delo|={worst_e:.3e} worst|dreward|={worst_r:.3e} "
          f"rng_state=exact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--small-only", action="store_true")
    args = ap.parse_args()

    check_config(args.trainer, "small-a",
                 {"episodes": 1000, "seed": 7,
                  "checkpoint_every": 250, "elo_games": 10})
    check_config(args.trainer, "small-b",
                 {"episodes": 300, "seed": 99,
                  "checkpoint_every": 100, "elo_games": 5})
    if not args.small_only:
        check_config(args.trainer, "full-default",
                     {"episodes": 30000, "seed": 7,
                      "checkpoint_every": 2000, "elo_games": 50})
    print("result: PASS")


if __name__ == "__main__":
    main()
