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
  snake_corpus.txt frozen SnakeEnv runner sessions (>= 110,000 transitions)
  policies_corpus.txt frozen policies/Q-learning runner sessions
                   (>= 34,532 compared states, plus the q_updates fixture
                   replay over the QUPDATES/QACT protocol)
  mcts_cases.txt   frozen MCTS runner sessions (>= 1,000 searches with
                   root children, plus the oracle session)
  mcts_wall_cases.txt frozen MCTS maze-mode (wall) runner sessions:
                   searches run with the browser game's maze wall
                   layouts (web/game.html mazeWalls) in the WALLS
                   protocol overlay
  meta.json        provenance: source SHA, Python version, fixture version

The corpora freeze full runner sessions: lines starting "> " are inputs
to a C++ runner, lines starting "< " are the expected replies (the same
line formats the runners speak interactively), "#" lines are comments.
The runners replay a corpus in batch mode and compare every field, so
no Python runs at test time; regeneration re-runs this live reference
and must reproduce every file byte-identically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import narcade.mcts as mcts_mod                       # noqa: E402
from narcade.mcts import MCTSPlayer                   # noqa: E402
from narcade.policies import (GreedyPolicy,           # noqa: E402
                              QTablePolicy, RandomPolicy)
from narcade.qlearn import QLearningAgent             # noqa: E402
from narcade.snake import DIRS, SnakeEnv              # noqa: E402

OUT = ROOT / "native" / "golden"
FIXTURE_VERSION = "golden-v3"


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


# ------------------------------------------------------------- corpora
# Shared helpers for the three frozen runner-session corpora. The
# episode/state generation schemes mirror the retired live drivers
# (same seeds, same action generators), so coverage is a superset of
# what the live differential runs compared.
REASON_CODE = {None: 0, "death": 1, "win": 2, "timeout": 3}
GREEDY = GreedyPolicy()


def food_tok(food) -> str:
    return "-1,-1" if food is None else f"{food[0]},{food[1]}"


def body_tok(body) -> str:
    return ";".join(f"{r},{c}" for r, c in body)


class Corpus:
    """A frozen runner session: inputs interleaved with expected replies."""

    def __init__(self, title: str):
        self.lines = [
            f"# {title}",
            "# Session format: \"> \" lines are runner inputs, \"< \" "
            "lines are the expected replies, \"#\" lines are comments.",
            "# Generated by tools/native_reference.py from the live "
            "Python reference; regeneration must reproduce this file "
            "byte-identically.",
        ]

    def note(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def inp(self, line: str) -> None:
        self.lines.append("> " + line)

    def out(self, line: str) -> None:
        self.lines.append("< " + line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def live_step_line(env: SnakeEnv, action: int) -> str:
    """Step a live env; return the runner's S/X line for the step."""
    state_before = env.encode_state()
    try:
        state_after, reward, done, info = env.step(action)
    except RuntimeError:
        return "X runtime_error"
    except ValueError:
        return "X invalid_argument"
    return (f"S {state_before} {state_after} {dbl_hex(reward)} "
            f"{1 if done else 0} {REASON_CODE[info.get('reason')]} "
            f"{food_tok(env.food)} {env.score} {env.steps} "
            f"{body_tok(env.body)}")


def python_episode(width, height, seed, max_steps, actions):
    """Run the Python env over `actions`; return (init, records, meta).

    Records mirror the snake runner's output fields; meta[i] is True
    when step i moved into the vacating tail cell. (Derived from the
    retired tools/native_fuzz.py driver, removed when the differential
    checks were frozen as corpora.)
    """
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    init = (env.encode_state(),
            tuple(env.food) if env.food is not None else None,
            [tuple(c) for c in env.body])
    records = []
    meta = []
    for a in actions:
        tail_vac = False
        if not env.done and a in (0, 1, 2):
            nh = env.heading
            if a == 1:
                nh = (nh + 3) % 4
            elif a == 2:
                nh = (nh + 1) % 4
            hr, hc = env.body[-1]
            target = (hr + DIRS[nh][0], hc + DIRS[nh][1])
            ate = env.food is not None and target == tuple(env.food)
            tail_vac = target == tuple(env.body[0]) and not ate
        state_before = env.encode_state()
        try:
            state_after, reward, done, info = env.step(a)
        except RuntimeError:
            records.append(("X", "runtime_error"))
            meta.append(False)
            continue
        except ValueError:
            records.append(("X", "invalid_argument"))
            meta.append(False)
            continue
        records.append(("S", state_before, state_after, dbl_hex(reward),
                        1 if done else 0, REASON_CODE[info.get("reason")],
                        tuple(env.food) if env.food is not None else None,
                        env.score, env.steps,
                        [tuple(c) for c in env.body]))
        meta.append(tail_vac)
    return init, records, meta


def greedy_actions(width, height, seed, max_steps, cap=6000):
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    actions = []
    while not env.done and len(actions) < cap:
        a = GREEDY.act(env)
        actions.append(a)
        env.step(a)
    return actions


def random_actions(width, height, seed, max_steps, action_seed):
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    rng = random.Random(action_seed)
    actions = []
    while not env.done and len(actions) < 2000:
        a = rng.randrange(3)
        actions.append(a)
        env.step(a)
    actions += [0, 9]  # post-terminal probes: legal and illegal actions
    return actions


def plan_actions(width, height, seed, max_steps, plan, action_seed):
    """Tokens: 'g' greedy, 'r' random, or a literal int action."""
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    rng = random.Random(action_seed)
    actions = []
    for tok in plan:
        if tok == "g":
            a = 0 if env.done else GREEDY.act(env)
        elif tok == "r":
            a = rng.randrange(3)
        else:
            a = tok
        actions.append(a)
        try:
            env.step(a)
        except (RuntimeError, ValueError):
            pass  # state unchanged; the replay records the exception
    return actions


def gen_snake_corpus() -> tuple[str, dict]:
    corpus = Corpus("DeepArcade native golden corpus: SnakeEnv episodes "
                    "(golden-v2)")
    stats = {"episodes": 0, "transitions": 0, "exceptions": 0, "eats": 0,
             "deaths": 0, "wins": 0, "timeouts": 0, "tail_vacations": 0}
    counter = [0]

    def next_seed(base):
        counter[0] += 1
        return base + 1009 * counter[0]

    def record(width, height, seed, max_steps, actions):
        ms = -1 if max_steps is None else max_steps
        corpus.inp(f"EPISODE {width} {height} {seed} {ms} "
                   f"{len(actions)}")
        for a in actions:
            corpus.inp(str(a))
        init, records, meta = python_episode(width, height, seed,
                                             max_steps, actions)
        corpus.out(f"INIT {init[0]} {food_tok(init[1])} "
                   f"{body_tok(init[2])}")
        for rec, tail_vac in zip(records, meta):
            if rec[0] == "X":
                corpus.out(f"X {rec[1]}")
                stats["exceptions"] += 1
                continue
            corpus.out(f"S {rec[1]} {rec[2]} {rec[3]} {rec[4]} {rec[5]} "
                       f"{food_tok(rec[6])} {rec[7]} {rec[8]} "
                       f"{body_tok(rec[9])}")
            stats["transitions"] += 1
            if rec[3] == dbl_hex(10.0):
                stats["eats"] += 1
            if rec[5] == 1:
                stats["deaths"] += 1
            elif rec[5] == 2:
                stats["wins"] += 1
            elif rec[5] == 3:
                stats["timeouts"] += 1
            if tail_vac:
                stats["tail_vacations"] += 1
        corpus.out("DONE")
        stats["episodes"] += 1

    corpus.note("transitions.json episodes (greedy, seeds 11/222/3333)")
    for seed in (11, 222, 3333):
        record(12, 12, seed, None, greedy_actions(12, 12, seed, None))
    corpus.note("greedy volume across board sizes (small boards win)")
    for width, height, count, base in ((12, 12, 400, 700000),
                                       (6, 6, 150, 710000),
                                       (5, 5, 100, 720000),
                                       (4, 4, 200, 730000)):
        for _ in range(count):
            seed = next_seed(base)
            record(width, height, seed, None,
                   greedy_actions(width, height, seed, None))
    corpus.note("random action streams + post-terminal probes")
    for width, height, count, base in ((12, 12, 150, 740000),
                                       (6, 6, 50, 750000)):
        for _ in range(count):
            seed = next_seed(base)
            record(width, height, seed, None,
                   random_actions(width, height, seed, None, seed + 17))
    corpus.note("illegal actions mid-episode, then play continues")
    for _ in range(20):
        seed = next_seed(760000)
        plan = ["g"] * 5 + [7, -1, 3] + ["g"] * 25
        record(12, 12, seed, None,
               plan_actions(12, 12, seed, None, plan, seed + 23))
    corpus.note("steps after the episode is done (legal and illegal)")
    for _ in range(20):
        seed = next_seed(770000)
        record(12, 12, seed, None,
               greedy_actions(12, 12, seed, None) + [0, 1, 9])
    corpus.note("forced timeouts via small max_steps")
    for i in range(30):
        seed = next_seed(780000)
        ms = (1, 3, 7)[i % 3]
        record(12, 12, seed, ms, greedy_actions(12, 12, seed, ms))
    corpus.note("top-up greedy episodes to the volume threshold")
    while stats["transitions"] < 111000:
        seed = next_seed(790000)
        record(12, 12, seed, None, greedy_actions(12, 12, seed, None))
    assert stats["wins"] > 0 and stats["timeouts"] > 0
    assert stats["tail_vacations"] > 0 and stats["exceptions"] > 0
    return corpus.text(), stats


# ------------------------------------------------------- policies corpus
def gen_greedy(width, height, seed, max_steps, cap=6000):
    return greedy_actions(width, height, seed, max_steps, cap)


def gen_qtable(width, height, seed, max_steps, table, cap=6000):
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    policy = QTablePolicy(table)
    actions = []
    while not env.done and len(actions) < cap:
        a = policy.act(env)
        actions.append(a)
        env.step(a)
    return actions


def gen_random(width, height, seed, max_steps, action_seed):
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    rng = random.Random(action_seed)
    actions = []
    while not env.done and len(actions) < 2000:
        a = rng.randrange(3)
        actions.append(a)
        env.step(a)
    actions += [0, 1]  # post-terminal probes (P -1 -1 / X runtime_error)
    return actions


def table_inp(corpus: Corpus, table) -> None:
    for row in table:
        corpus.inp(" ".join(dbl_hex(v) for v in row))


def gen_policies_corpus(q_data: dict) -> tuple[str, dict]:
    corpus = Corpus("DeepArcade native golden corpus: policies and "
                    "tabular Q-learning (golden-v2)")
    stats = {"episodes": 0, "states": 0, "exceptions": 0, "updates": 0,
             "act_compared": 0, "rand_compared": 0}
    agent = q_data["agent"]
    final_q = q_data["final_q"]

    # Fixture layer: the q_updates.json replay over the runner protocol.
    corpus.note("fixture: QUPDATES replay of q_updates.json")
    corpus.inp(f"QUPDATES {dbl_hex(agent['alpha'])} "
               f"{dbl_hex(agent['gamma'])} {dbl_hex(agent['epsilon'])} "
               f"{agent['seed']} {len(q_data['updates'])}")
    for u in q_data["updates"]:
        corpus.inp(f"U {u['s']} {u['a']} {dbl_hex(u['r'])} "
                   f"{u['s_next']} {1 if u['done'] else 0}")
    stats["updates"] = len(q_data["updates"])
    corpus.out("QRESULT")
    for row in final_q:
        corpus.out(" ".join(dbl_hex(v) for v in row))
    corpus.note("fixture: QACT 1,000-state action stream")
    corpus.inp(f"QACT {agent['seed']} {dbl_hex(agent['epsilon'])}")
    corpus.inp("QTABLE")
    table_inp(corpus, final_q)
    corpus.inp(f"NSTATES {len(q_data['act_states'])}")
    for s in q_data["act_states"]:
        corpus.inp(str(s))
    corpus.out("ACTIONS")
    corpus.out(" ".join(str(a) for a in q_data["act_actions"]))
    stats["act_compared"] = len(q_data["act_states"])
    corpus.note("fixture: QTablePolicy best_action (to_jsonable)")
    corpus.inp("SETTABLE")
    table_inp(corpus, final_q)
    corpus.inp("BESTACTION")
    corpus.out("BEST")
    best = QTablePolicy(final_q).to_jsonable()["best_action"]
    corpus.out(" ".join(str(a) for a in best))
    corpus.note("fixture: RandomPolicy streams over matched seeds")
    for seed in (0, 1, 42, 2024, 999999):
        policy = RandomPolicy(seed=seed)
        draws = [policy.act(None) for _ in range(2000)]
        corpus.inp(f"RANDSEED {seed} 2000")
        corpus.out("RAND")
        corpus.out(" ".join(str(a) for a in draws))
        stats["rand_compared"] += 2000

    # Episode layer: per-step P lines (greedy + qtable) and S lines.
    zeros = [[0.0] * 3 for _ in range(32)]
    pat_rng = random.Random(7)
    pattern = [[pat_rng.uniform(-5.0, 5.0) for _ in range(3)]
               for _ in range(32)]
    tables = [final_q, zeros, pattern]
    ti = [0]

    def next_table():
        table = tables[ti[0] % len(tables)]
        ti[0] += 1
        return table

    counter = [0]

    def next_seed(base):
        counter[0] += 1
        return base + 1009 * counter[0]

    def record(width, height, seed, max_steps, actions, table):
        corpus.inp("SETTABLE")
        table_inp(corpus, table)
        ms = -1 if max_steps is None else max_steps
        corpus.inp(f"EPISODE {width} {height} {seed} {ms} "
                   f"{len(actions)}")
        for a in actions:
            corpus.inp(str(a))
        env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
        qpolicy = QTablePolicy(table)
        corpus.out(f"INIT {env.encode_state()} {food_tok(env.food)} "
                   f"{body_tok(env.body)}")
        for a in actions:
            if env.done:
                corpus.out("P -1 -1")
            else:
                corpus.out(f"P {GREEDY.act(env)} {qpolicy.act(env)}")
                stats["states"] += 1
            line = live_step_line(env, a)
            corpus.out(line)
            if line.startswith("X"):
                stats["exceptions"] += 1
        corpus.out("DONE")
        stats["episodes"] += 1

    corpus.note("episodes: greedy-driven streams, rotating tables")
    for w, h, count, base in ((12, 12, 18, 800000), (6, 6, 12, 810000),
                              (5, 5, 8, 820000), (4, 4, 8, 830000)):
        for _ in range(count):
            seed = next_seed(base)
            record(w, h, seed, None, gen_greedy(w, h, seed, None),
                   next_table())
    corpus.note("episodes: qtable-driven streams (final_q, pattern)")
    for table in (final_q, pattern):
        for w, h, count, base in ((12, 12, 8, 840000), (6, 6, 6, 850000)):
            for _ in range(count):
                seed = next_seed(base)
                record(w, h, seed, None,
                       gen_qtable(w, h, seed, None, table), table)
    corpus.note("episodes: qtable-driven, all-zeros table (ties)")
    for _ in range(4):
        seed = next_seed(860000)
        record(12, 12, seed, None,
               gen_qtable(12, 12, seed, None, zeros), zeros)
    corpus.note("episodes: random streams + post-terminal probes")
    for w, h, count, base in ((12, 12, 15, 870000), (6, 6, 8, 880000)):
        for _ in range(count):
            seed = next_seed(base)
            record(w, h, seed, None,
                   gen_random(w, h, seed, None, seed + 17), next_table())
    corpus.note("episodes: forced timeouts (pre-timeout states live)")
    for i in range(6):
        seed = next_seed(890000)
        ms = (5, 25)[i % 2]
        record(12, 12, seed, ms, gen_greedy(12, 12, seed, ms),
               next_table())
    corpus.note("top-up greedy episodes to the state threshold")
    while stats["states"] < 34532 + 1500:
        seed = next_seed(900000)
        record(12, 12, seed, None, gen_greedy(12, 12, seed, None),
               next_table())
    return corpus.text(), stats


# ----------------------------------------------------------- MCTS corpus
def python_search_with_root(player, env):
    """Run the real MCTSPlayer.search and capture its root node."""
    captured = {}
    orig_init = mcts_mod._Node.__init__

    def patched(self, parent, action):
        orig_init(self, parent, action)
        if parent is None:
            captured["root"] = self

    mcts_mod._Node.__init__ = patched
    try:
        action = player.search(env)
    finally:
        mcts_mod._Node.__init__ = orig_init
    root = captured["root"]
    children = [(a, ch.visits, ch.value)
                for a, ch in root.children.items()]
    return action, children


def env_view(width, height, body, food, heading):
    env = SnakeEnv(width, height, seed=1)
    env.body = deque(body)
    env.food = food
    env.heading = heading
    return env


def harvest_states(min_states, boards=((12, 12), (6, 6), (5, 5), (4, 4)),
                   seed_base=700000):
    states = []
    seed = seed_base
    while len(states) < min_states:
        for w, h in boards:
            seed += 1013
            env = SnakeEnv(w, h, seed=seed)
            rng = random.Random(seed + 7)
            steps = 0
            while not env.done and steps < 400:
                if steps % 3 == 0:
                    states.append((w, h, [tuple(c) for c in env.body],
                                   tuple(env.food), env.heading))
                a = GREEDY.act(env) if rng.random() < 0.7 \
                    else rng.randrange(3)
                env.step(a)
                steps += 1
    return states


def gen_mcts_cases(oracle: dict) -> tuple[str, dict]:
    corpus = Corpus("DeepArcade native golden corpus: MCTS searches "
                    "(golden-v2)")
    stats = {"searches": 0, "root_children": 0}
    c_hex = dbl_hex(1.0)

    def emit_player(sims, pseed):
        corpus.inp(f"PLAYER {sims} 100 {c_hex} {pseed}")
        corpus.out("OK")

    def emit_env(env, seed):
        corpus.inp(f"ENV {env.width} {env.height} {seed}")
        corpus.out(f"INIT {env.encode_state()} {food_tok(env.food)} "
                   f"{body_tok(env.body)}")

    def search_reply(action, children):
        corpus.out(f"ACTION {action}")
        corpus.out(f"ROOT {len(children)}")
        for a, visits, value in children:
            corpus.out(f"CHILD {a} {visits} {dbl_hex(value)}")
        stats["searches"] += 1
        stats["root_children"] += len(children)

    corpus.note("oracle session (mirrors the mcts_oracle.json cases)")
    for case in oracle["cases"]:
        kind = case["kind"]
        sims = case["simulations"]
        pseed = case["player_seed"]
        if kind in ("fresh_player", "tiny_budget", "small_board"):
            w = case.get("width", 12)
            h = case.get("height", 12)
            env = SnakeEnv(w, h, seed=case["env_seed"])
            emit_env(env, case["env_seed"])
            emit_player(sims, pseed)
            py = MCTSPlayer(simulations=sims, seed=pseed)
            action, children = python_search_with_root(py, env)
            assert action == case["action"]
            corpus.inp("SEARCH")
            search_reply(action, children)
        elif kind == "shared_player_episode":
            env = SnakeEnv(12, 12, seed=case["env_seed"])
            emit_env(env, case["env_seed"])
            emit_player(sims, pseed)
            py = MCTSPlayer(simulations=sims, seed=pseed)
            for want_a in case["actions"]:
                action, children = python_search_with_root(py, env)
                assert action == want_a
                corpus.inp("SEARCH")
                search_reply(action, children)
                corpus.inp(f"STEP {want_a}")
                corpus.out(live_step_line(env, want_a))
            assert (env.score, env.steps) == (case["final_score"],
                                              case["final_steps"])
        else:
            raise ValueError(f"unknown oracle case kind {kind}")

    corpus.note("fuzz session: harvested mid-game states, fresh player "
                "per search")
    states = harvest_states(1000)
    sims_cycle = (80, 10, 2, 1, 10, 2)
    pseeds = (31337, 101, 202, 303, 404, 7)
    for i, (w, h, body, food, heading) in enumerate(states[:1000]):
        sims = sims_cycle[i % len(sims_cycle)]
        pseed = pseeds[i % len(pseeds)]
        env = env_view(w, h, body, food, heading)
        py = MCTSPlayer(simulations=sims, seed=pseed)
        action, children = python_search_with_root(py, env)
        emit_player(sims, pseed)
        corpus.inp(f"SEARCHSTATE {w} {h} {food[0]} {food[1]} {heading} "
                   f"{len(body)}")
        for r, c in body:
            corpus.inp(f"{r} {c}")
        search_reply(action, children)
    return corpus.text(), stats


# ---------------------------------------------------- MCTS wall corpus
def maze_walls(width, height):
    """The browser game's maze-mode wall layout (web/game.html
    mazeWalls): four symmetric 3-cell corner blocks. Returns a
    frozenset of (r, c) cells, the representation MCTSPlayer.search
    takes. The layout is a pure function of the board size, exactly
    as in the game."""
    blocks = [[3, 3], [3, 4], [4, 3],
              [3, width - 4], [3, width - 5], [4, width - 4],
              [height - 4, 3], [height - 4, 4], [height - 5, 3],
              [height - 4, width - 4], [height - 4, width - 5],
              [height - 5, width - 4]]
    return frozenset((r, c) for r, c in blocks)


def python_search_with_walls(player, env, walls):
    """python_search_with_root, with the wall set passed to search."""
    captured = {}
    orig_init = mcts_mod._Node.__init__

    def patched(self, parent, action):
        orig_init(self, parent, action)
        if parent is None:
            captured["root"] = self

    mcts_mod._Node.__init__ = patched
    try:
        action = player.search(env, walls=walls)
    finally:
        mcts_mod._Node.__init__ = orig_init
    root = captured["root"]
    children = [(a, ch.visits, ch.value)
                for a, ch in root.children.items()]
    return action, children


def gen_mcts_wall_cases() -> tuple[str, dict]:
    corpus = Corpus("DeepArcade native golden corpus: MCTS maze-mode "
                    "(wall) searches (golden-v3)")
    stats = {"searches": 0, "root_children": 0}
    c_hex = dbl_hex(1.0)
    current_walls = [None]

    def emit_player(sims, pseed):
        corpus.inp(f"PLAYER {sims} 100 {c_hex} {pseed}")
        corpus.out("OK")

    def emit_walls(walls):
        if current_walls[0] == walls:
            return
        current_walls[0] = walls
        corpus.inp(f"WALLS {len(walls)}")
        for r, c in sorted(walls):
            corpus.inp(f"{r} {c}")
        corpus.out("OK")

    def search_reply(action, children):
        corpus.out(f"ACTION {action}")
        corpus.out(f"ROOT {len(children)}")
        for a, visits, value in children:
            corpus.out(f"CHILD {a} {visits} {dbl_hex(value)}")
        stats["searches"] += 1
        stats["root_children"] += len(children)

    def emit_searchstate(w, h, body, food, heading):
        corpus.inp(f"SEARCHSTATE {w} {h} {food[0]} {food[1]} {heading} "
                   f"{len(body)}")
        for r, c in body:
            corpus.inp(f"{r} {c}")

    walls12 = maze_walls(12, 12)

    corpus.note("the mcts_oracle.json opening state (seed 5000) "
                "searched with the 12x12 maze wall set")
    env = SnakeEnv(12, 12, seed=5000)
    emit_walls(walls12)
    emit_player(80, 31337)
    py = MCTSPlayer(simulations=80, seed=31337)
    action, children = python_search_with_walls(
        py, env_view(12, 12, [tuple(c) for c in env.body],
                     tuple(env.food), env.heading), walls12)
    emit_searchstate(12, 12, [tuple(c) for c in env.body],
                     tuple(env.food), env.heading)
    search_reply(action, children)
    for sims in (1, 2):
        emit_player(sims, 1)
        py = MCTSPlayer(simulations=sims, seed=1)
        action, children = python_search_with_walls(
            py, env_view(12, 12, [tuple(c) for c in env.body],
                         tuple(env.food), env.heading), walls12)
        emit_searchstate(12, 12, [tuple(c) for c in env.body],
                         tuple(env.food), env.heading)
        search_reply(action, children)

    corpus.note("crafted cases: maze geometry decides the outcome; "
                "the forced actions are asserted at generation time")
    # Head (4,4) heading north: straight (3,4) and left (4,3) are
    # wall cells; right steps onto the food at (4,5). Forced RIGHT.
    emit_player(80, 101)
    py = MCTSPlayer(simulations=80, seed=101)
    action, children = python_search_with_walls(
        py, env_view(12, 12, [(6, 4), (5, 4), (4, 4)], (4, 5), 0),
        walls12)
    assert action == 2, f"crafted forced-right case: got {action}"
    emit_searchstate(12, 12, [(6, 4), (5, 4), (4, 4)], (4, 5), 0)
    search_reply(action, children)
    # Head (4,4) heading west: straight (4,3) and right (3,4) are
    # wall cells; left steps onto the food at (5,4). Forced LEFT.
    emit_player(80, 101)
    py = MCTSPlayer(simulations=80, seed=101)
    action, children = python_search_with_walls(
        py, env_view(12, 12, [(6, 5), (5, 5), (4, 5), (4, 4)],
                     (5, 4), 3), walls12)
    assert action == 1, f"crafted forced-left case: got {action}"
    emit_searchstate(12, 12, [(6, 5), (5, 5), (4, 5), (4, 4)],
                     (5, 4), 3)
    search_reply(action, children)
    # Boxed head (4,4) heading west: straight and right are walls,
    # left is the snake's own body. Every root child dies at once.
    emit_player(80, 7)
    py = MCTSPlayer(simulations=80, seed=7)
    action, children = python_search_with_walls(
        py, env_view(12, 12, [(5, 3), (5, 4), (5, 5), (4, 5), (4, 4)],
                     (10, 10), 3), walls12)
    emit_searchstate(12, 12, [(5, 3), (5, 4), (5, 5), (4, 5), (4, 4)],
                     (10, 10), 3)
    search_reply(action, children)

    corpus.note("fuzz session: harvested mid-game states that are "
                "legal on the maze board (no body cell or food on a "
                "wall), fresh player per search")
    sims_cycle = (80, 10, 2, 1, 10, 2)
    pseeds = (31337, 101, 202, 303, 404, 7)
    idx = [0]

    def fuzz_layout(w, h, want):
        walls = maze_walls(w, h)
        legal = []
        round_no = 0
        while len(legal) < want:
            pool = harvest_states(
                400, boards=((w, h),),
                seed_base=950000 + 1000 * w + 7919 * round_no)
            legal.extend(
                s for s in pool
                if s[3] is not None and s[3] not in walls
                and not any(cell in walls for cell in s[2]))
            round_no += 1
        emit_walls(walls)
        for (bw, bh, body, food, heading) in legal[:want]:
            i = idx[0]
            idx[0] += 1
            sims = sims_cycle[i % len(sims_cycle)]
            pseed = pseeds[i % len(pseeds)]
            py = MCTSPlayer(simulations=sims, seed=pseed)
            action, children = python_search_with_walls(
                py, env_view(bw, bh, body, food, heading), walls)
            emit_player(sims, pseed)
            emit_searchstate(bw, bh, body, food, heading)
            search_reply(action, children)

    fuzz_layout(12, 12, 170)
    fuzz_layout(8, 8, 90)
    fuzz_layout(6, 6, 90)

    corpus.note("shared player across consecutive wall searches "
                "(the RNG lifecycle under a wall set)")
    emit_walls(walls12)
    emit_player(40, 2024)
    py = MCTSPlayer(simulations=40, seed=2024)
    pool = harvest_states(400, boards=((12, 12),), seed_base=990000)
    legal = [(bw, bh, body, food, heading)
             for (bw, bh, body, food, heading) in pool
             if food is not None and food not in walls12
             and not any(cell in walls12 for cell in body)]
    assert len(legal) >= 48
    for (bw, bh, body, food, heading) in legal[:48]:
        action, children = python_search_with_walls(
            py, env_view(bw, bh, body, food, heading), walls12)
        emit_searchstate(bw, bh, body, food, heading)
        search_reply(action, children)
    return corpus.text(), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="golden output directory (default: native/golden)")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    q_data = gen_q_updates()
    oracle = gen_mcts_oracle()
    transitions = gen_transitions()
    snake_text, snake_stats = gen_snake_corpus()
    policies_text, policies_stats = gen_policies_corpus(q_data)
    mcts_text, mcts_stats = gen_mcts_cases(oracle)
    wall_text, wall_stats = gen_mcts_wall_cases()
    files = {
        "rng_traces.txt": gen_rng_traces(),
        "transitions.json": json.dumps(transitions, sort_keys=True),
        "mcts_oracle.json": json.dumps(oracle, sort_keys=True),
        "q_updates.json": json.dumps(q_data, sort_keys=True),
        "snake_corpus.txt": snake_text,
        "policies_corpus.txt": policies_text,
        "mcts_cases.txt": mcts_text,
        "mcts_wall_cases.txt": wall_text,
    }
    hashes = {}
    for name, text in files.items():
        data = text.encode()
        (out_dir / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    counts = {
        "q_updates": len(q_data["updates"]),
        "q_act_stream": len(q_data["act_states"]),
        "transitions_episodes": len(transitions["episodes"]),
        "transitions_steps": sum(len(e["steps"])
                                  for e in transitions["episodes"]),
        "mcts_oracle_cases": len(oracle["cases"]),
        "snake_episodes": snake_stats["episodes"],
        "snake_transitions": snake_stats["transitions"],
        "snake_exceptions": snake_stats["exceptions"],
        "snake_eats": snake_stats["eats"],
        "snake_deaths": snake_stats["deaths"],
        "snake_wins": snake_stats["wins"],
        "snake_timeouts": snake_stats["timeouts"],
        "snake_tail_vacations": snake_stats["tail_vacations"],
        "policies_episodes": policies_stats["episodes"],
        "policies_states": policies_stats["states"],
        "policies_exceptions": policies_stats["exceptions"],
        "policies_rand_draws": policies_stats["rand_compared"],
        "mcts_searches": mcts_stats["searches"],
        "mcts_root_children": mcts_stats["root_children"],
        "mcts_wall_searches": wall_stats["searches"],
        "mcts_wall_root_children": wall_stats["root_children"],
    }
    meta = {"fixture_version": FIXTURE_VERSION, "source_sha": sha,
            "python_version": sys.version.split()[0], "file_sha256": hashes,
            "counts": counts, "generator": "tools/native_reference.py"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2,
                                                  sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
