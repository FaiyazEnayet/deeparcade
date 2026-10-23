#!/usr/bin/env python3
"""CPP-00 golden fixture generator for the DeepArcade native conversion.

Freezes the CURRENT Python engine's behavior into versioned fixtures under
native/golden/. The C++ implementation must reproduce these fixtures exactly
(CPython-compatible RNG mode) before any parity claim is made.

Deterministic and stdlib-only. Regeneration from the same source SHA and
Python version must be byte-identical; the caller verifies by sha256.

Fixtures:
  rng_traces.txt   line-based RNG operation traces (see format in file header)
  transitions.json SnakeEnv episodes under the deterministic GreedyPolicy
  mcts_oracle.json MCTS action oracles, incl. the shared-player RNG lifecycle
  q_updates.json   tabular Q-learning update stream and final table
  meta.json        provenance: source SHA, Python version, fixture version
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from narcade.mcts import MCTSPlayer          # noqa: E402
from narcade.policies import GreedyPolicy    # noqa: E402
from narcade.qlearn import QLearningAgent    # noqa: E402
from narcade.snake import SnakeEnv           # noqa: E402

OUT = ROOT / "native" / "golden"
FIXTURE_VERSION = "golden-v1"


def dbl_hex(x: float) -> str:
    return struct.pack(">d", x).hex()


# ---------------------------------------------------------------- RNG traces
# Line format (one op per line, replayed in order by the C++ trace test):
#   SEED <n>            fresh random.Random(n)
#   R <hex64>           random() -> IEEE754 bit pattern, big-endian hex
#   B <k> <hex>         getrandbits(k), hex zero-padded to ceil(k/4) digits
#   G <n> <v>           randrange(n)
#   C <len> <idx>       choice(list(range(len))) == idx
#   I <v>               randint(0, 999999)
#   SNAPSEED <n>        fresh Random(n); next 100 R lines precede a snapshot
#   SNAPSHOT            state snapshot point; the following ops must be
#                       reproducible from the restored snapshot state
OPS_PATTERN = (
    [("R",)] * 3
    + [("B", 1), ("B", 5), ("B", 32), ("G", 3), ("C", 3)]
    + [("R",)] * 2
    + [("B", 8), ("B", 16), ("G", 10), ("C", 10), ("I",)]
    + [("B", 2), ("B", 3), ("B", 13), ("G", 144), ("C", 144)]
    + [("R",), ("B", 21), ("B", 26), ("B", 31), ("G", 7), ("C", 50)]
    + [("B", 33), ("B", 48), ("B", 53), ("B", 64), ("G", 1000), ("C", 1)]
    + [("R",), ("G", 1), ("G", 2), ("C", 2), ("C", 4), ("C", 9), ("C", 27)]
    + [("B", 31), ("R",), ("I",), ("G", 36), ("C", 141)]
)


def emit_ops(rng: random.Random, count: int, lines: list[str]) -> None:
    i = 0
    while i < count:
        op = OPS_PATTERN[i % len(OPS_PATTERN)]
        kind = op[0]
        if kind == "R":
            lines.append(f"R {dbl_hex(rng.random())}")
        elif kind == "B":
            k = op[1]
            v = rng.getrandbits(k)
            lines.append(f"B {k} {v:0{(k + 3) // 4}x}")
        elif kind == "G":
            n = op[1]
            lines.append(f"G {n} {rng.randrange(n)}")
        elif kind == "C":
            ln = op[1]
            lines.append(f"C {ln} {rng.choice(list(range(ln)))}")
        elif kind == "I":
            lines.append(f"I {rng.randint(0, 999999)}")
        i += 1


def gen_rng_traces() -> str:
    lines = [
        "# DeepArcade native golden RNG traces v1",
        "# Format: see tools/native_reference.py. Doubles are IEEE754 hex.",
    ]
    for seed in (0, 7, 999, 31337, 5000, 5001):
        lines.append(f"SEED {seed}")
        emit_ops(random.Random(seed), 600, lines)
    # Snapshot/restore segment: 100 draws, snapshot, then 1000 mixed ops.
    lines.append("SNAPSEED 424242")
    rng = random.Random(424242)
    for _ in range(100):
        lines.append(f"R {dbl_hex(rng.random())}")
    lines.append("SNAPSHOT")
    emit_ops(rng, 1000, lines)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- transitions
def gen_transitions() -> dict:
    greedy = GreedyPolicy()
    episodes = []
    for seed in (11, 222, 3333):
        env = SnakeEnv(12, 12, seed=seed)
        steps = []
        while not env.done and len(steps) < 2000:
            state_before = env.encode_state()
            body_before = [list(c) for c in env.body]
            food_before = list(env.food) if env.food is not None else None
            heading_before = env.heading
            action = greedy.act(env)
            state_after, reward, done, info = env.step(action)
            steps.append({
                "state_before": state_before,
                "body_before": body_before,
                "food_before": food_before,
                "heading_before": heading_before,
                "action": action,
                "state_after": state_after,
                "reward": reward,
                "done": done,
                "reason": info.get("reason"),
                "body_after": [list(c) for c in env.body],
                "food_after": list(env.food) if env.food is not None else None,
                "score": env.score,
                "steps": env.steps,
            })
        episodes.append({"seed": seed, "width": 12, "height": 12,
                         "steps": steps})
    return {"episodes": episodes}


# ------------------------------------------------------------- MCTS oracle
def gen_mcts_oracle() -> dict:
    cases = []
    # Fresh player per env seed (mirrors nothing in evaluate.py; isolates RNG).
    for env_seed in (5000, 5001):
        env = SnakeEnv(12, 12, seed=env_seed)
        player = MCTSPlayer(simulations=80, seed=31337)
        cases.append({"kind": "fresh_player", "env_seed": env_seed,
                      "simulations": 80, "player_seed": 31337,
                      "action": player.search(env)})
    # Shared player across consecutive decisions on one env (evaluate.py
    # lifecycle: RNG state carries across calls and episodes).
    env = SnakeEnv(12, 12, seed=5000)
    player = MCTSPlayer(simulations=80, seed=31337)
    actions = []
    while not env.done and len(actions) < 30:
        a = player.search(env)
        actions.append(a)
        env.step(a)
    cases.append({"kind": "shared_player_episode", "env_seed": 5000,
                  "simulations": 80, "player_seed": 31337,
                  "actions": actions, "final_score": env.score,
                  "final_steps": env.steps})
    # Tiny budgets and a small board.
    for sims in (1, 2):
        for pseed in (1, 2, 3):
            env = SnakeEnv(12, 12, seed=99)
            p = MCTSPlayer(simulations=sims, seed=pseed)
            cases.append({"kind": "tiny_budget", "env_seed": 99,
                          "simulations": sims, "player_seed": pseed,
                          "action": p.search(env)})
    env = SnakeEnv(6, 6, seed=7)
    p = MCTSPlayer(simulations=300, seed=5)
    cases.append({"kind": "small_board", "env_seed": 7, "width": 6,
                  "height": 6, "simulations": 300, "player_seed": 5,
                  "action": p.search(env)})
    return {"cases": cases}


# ------------------------------------------------------------- Q updates
def gen_q_updates() -> dict:
    stream_rng = random.Random(555)
    stream = []
    for i in range(10000):
        stream.append({
            "s": stream_rng.randrange(32),
            "a": stream_rng.randrange(3),
            "r": stream_rng.choice([10.0, -10.0, 0.5, -0.5, 0.0]),
            "s_next": stream_rng.randrange(32),
            "done": (i % 37 == 36),
        })
    agent = QLearningAgent(seed=2024, epsilon=0.25)
    for u in stream:
        agent.update(u["s"], u["a"], u["r"], u["s_next"], u["done"])
    # Action-selection stream with exploration RNG live.
    act_states = [stream_rng.randrange(32) for _ in range(1000)]
    act_agent = QLearningAgent(seed=2024, epsilon=0.25)
    act_agent.q = [row[:] for row in agent.q]
    actions = [act_agent.act(s) for s in act_states]
    return {"updates": stream, "final_q": agent.q,
            "act_states": act_states, "act_actions": actions,
            "agent": {"alpha": 0.15, "gamma": 0.9, "epsilon": 0.25,
                      "seed": 2024}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    files = {
        "rng_traces.txt": gen_rng_traces(),
        "transitions.json": json.dumps(gen_transitions(), sort_keys=True),
        "mcts_oracle.json": json.dumps(gen_mcts_oracle(), sort_keys=True),
        "q_updates.json": json.dumps(gen_q_updates(), sort_keys=True),
    }
    hashes = {}
    for name, text in files.items():
        data = text.encode()
        (OUT / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    meta = {"fixture_version": FIXTURE_VERSION, "source_sha": sha,
            "python_version": sys.version.split()[0], "file_sha256": hashes,
            "generator": "tools/native_reference.py"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
