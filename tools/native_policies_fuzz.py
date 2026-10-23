#!/usr/bin/env python3
"""CPP-02 differential driver: Python policies/Q-learning vs the C++ port.

Layers:
  fixture  native/golden/q_updates.json is replayed through the C++
           runner: all 10,000 Q updates (final table compared against
           final_q, worst abs error reported, gate <= 1e-12), the
           1,000-state action stream (exact, exercises the agent RNG),
           QTablePolicy.to_jsonable's best_action, and RandomPolicy
           streams over matched seeds (10,000 draws, exact).
  fuzz     episodes on varied boards/seeds are replayed through the
           C++ runner; at every live state the runner's GreedyPolicy
           and QTablePolicy actions are compared exactly against the
           Python policies evaluated live on the same state. Three
           Q-table variants (fixture final_q, all-zeros, patterned)
           exercise the first-max tie rule. >= 10,000 states required.

Protocol with the runner is line-based text; see
native/tests/policies_diff_runner.cpp for the format. Doubles travel
as IEEE754 hex bit patterns.

Exit code is non-zero on the first mismatch or if a gate is not met.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy  # noqa: E402
from narcade.snake import SnakeEnv                                   # noqa: E402

REASON_CODE = {None: 0, "death": 1, "win": 2, "timeout": 3}
GREEDY = GreedyPolicy()


def dbl_hex(x: float) -> str:
    return struct.pack(">d", x).hex()


def hex_dbl(tok: str) -> float:
    return struct.unpack(">d", bytes.fromhex(tok))[0]


class Mismatch(Exception):
    pass


# ------------------------------------------------------------ runner link
class Runner:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    def _send(self, text: str):
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def _line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise Mismatch("runner closed the stream unexpectedly")
        return line.strip()

    def _table_lines(self, table) -> str:
        return "".join(
            " ".join(dbl_hex(v) for v in row) + "\n" for row in table)

    def q_replay(self, alpha, gamma, epsilon, seed, updates):
        self._send(f"QUPDATES {dbl_hex(alpha)} {dbl_hex(gamma)} "
                   f"{dbl_hex(epsilon)} {seed} {len(updates)}\n")
        for u in updates:
            self._send(f"U {u['s']} {u['a']} {dbl_hex(u['r'])} "
                       f"{u['s_next']} {1 if u['done'] else 0}\n")
        if self._line() != "QRESULT":
            raise Mismatch("missing QRESULT")
        return [[hex_dbl(tok) for tok in self._line().split()]
                for _ in range(32)]

    def q_act(self, seed, epsilon, table, states):
        self._send(f"QACT {seed} {dbl_hex(epsilon)}\n")
        self._send("QTABLE\n" + self._table_lines(table))
        self._send(f"NSTATES {len(states)}\n")
        for s in states:
            self._send(f"{s}\n")
        if self._line() != "ACTIONS":
            raise Mismatch("missing ACTIONS")
        return [int(tok) for tok in self._line().split()]

    def set_table(self, table):
        self._send("SETTABLE\n" + self._table_lines(table))

    def best_action(self):
        self._send("BESTACTION\n")
        if self._line() != "BEST":
            raise Mismatch("missing BEST")
        return [int(tok) for tok in self._line().split()]

    def rand_actions(self, seed, n):
        self._send(f"RANDSEED {seed} {n}\n")
        if self._line() != "RAND":
            raise Mismatch("missing RAND")
        return [int(tok) for tok in self._line().split()]

    def run_episode(self, width, height, seed, max_steps, actions):
        ms = -1 if max_steps is None else max_steps
        self._send(f"EPISODE {width} {height} {seed} {ms} "
                   f"{len(actions)}\n")
        for a in actions:
            self._send(f"{a}\n")
        init = self._line().split()
        if not init or init[0] != "INIT":
            raise Mismatch(f"bad INIT line: {init}")
        steps = []
        for _ in actions:
            p = self._line().split()
            s = self._line().split()
            if not p or p[0] != "P" or not s or s[0] not in ("S", "X"):
                raise Mismatch(f"bad step lines: {p} / {s}")
            steps.append((p, s))
        if self._line() != "DONE":
            raise Mismatch("bad episode terminator")
        return init, steps

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def parse_food(tok: str):
    r, c = tok.split(",")
    r, c = int(r), int(c)
    return None if (r, c) == (-1, -1) else (r, c)


def parse_body(tok: str):
    if not tok:
        return []
    return [tuple(int(x) for x in pair.split(",")) for pair in tok.split(";")]


# ------------------------------------------------------------ fixture layer
class FixtureStats:
    def __init__(self):
        self.updates = 0
        self.worst_q_err = 0.0
        self.q_exact = 0
        self.act_compared = 0
        self.rand_compared = 0


def fixture_checks(runner) -> FixtureStats:
    data = json.loads((ROOT / "native" / "golden" /
                       "q_updates.json").read_text())
    agent = data["agent"]
    stats = FixtureStats()

    # 1) Replay all 10,000 updates; compare the table with final_q.
    got = runner.q_replay(agent["alpha"], agent["gamma"],
                          agent["epsilon"], agent["seed"],
                          data["updates"])
    stats.updates = len(data["updates"])
    for s in range(32):
        for a in range(3):
            err = abs(got[s][a] - data["final_q"][s][a])
            stats.worst_q_err = max(stats.worst_q_err, err)
            if err == 0.0:
                stats.q_exact += 1
    if stats.worst_q_err > 1e-12:
        raise Mismatch(f"Q replay worst error {stats.worst_q_err:.3e} "
                       f"exceeds 1e-12")

    # 2) Action stream: fresh agent (same seed) + fixture table.
    actions = runner.q_act(agent["seed"], agent["epsilon"],
                           data["final_q"], data["act_states"])
    stats.act_compared = len(data["act_states"])
    if actions != data["act_actions"]:
        for i, (g, e) in enumerate(zip(actions, data["act_actions"])):
            if g != e:
                raise Mismatch(f"act stream step {i}: runner={g} "
                               f"fixture={e}")
        raise Mismatch("act stream length differs from fixture")

    # 3) QTablePolicy best_action (to_jsonable equivalence).
    runner.set_table(data["final_q"])
    want_best = QTablePolicy(data["final_q"]).to_jsonable()["best_action"]
    if runner.best_action() != want_best:
        raise Mismatch("best_action differs from Python to_jsonable")

    # 4) RandomPolicy streams over matched seeds (act ignores the env).
    for seed in (0, 1, 42, 2024, 999999):
        policy = RandomPolicy(seed=seed)
        want = [policy.act(None) for _ in range(2000)]
        got_rand = runner.rand_actions(seed, 2000)
        stats.rand_compared += 2000
        if got_rand != want:
            for i, (g, e) in enumerate(zip(got_rand, want)):
                if g != e:
                    raise Mismatch(f"RandomPolicy seed={seed} draw {i}: "
                                   f"runner={g} python={e}")
            raise Mismatch("RandomPolicy stream length differs")
    return stats


# ------------------------------------------------------------- state fuzz
class FuzzStats:
    def __init__(self):
        self.episodes = 0
        self.states = 0
        self.greedy_compared = 0
        self.qtable_compared = 0
        self.exceptions = 0


def compare_episode(runner, label, width, height, seed, max_steps,
                    actions, table, stats):
    qpolicy = QTablePolicy(table)
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    runner.set_table(table)
    init_r, steps_r = runner.run_episode(width, height, seed, max_steps,
                                         actions)
    got_init = (int(init_r[1]), parse_food(init_r[2]),
                parse_body(init_r[3]))
    want_init = (env.encode_state(),
                 tuple(env.food) if env.food is not None else None,
                 [tuple(c) for c in env.body])
    if got_init != want_init:
        raise Mismatch(f"{label}: INIT differs: runner={got_init} "
                       f"python={want_init}")
    for i, a in enumerate(actions):
        p_tok, s_tok = steps_r[i]
        if env.done:
            if p_tok != ["P", "-1", "-1"]:
                raise Mismatch(f"{label} step {i}: expected P -1 -1 on "
                               f"done env, runner sent {p_tok}")
        else:
            g = GREEDY.act(env)
            q = qpolicy.act(env)
            if p_tok != ["P", str(g), str(q)]:
                raise Mismatch(
                    f"{label} step {i} (state={env.encode_state()}, "
                    f"heading={env.heading}, food={tuple(env.food)}, "
                    f"head={tuple(env.body[-1])}): runner P={p_tok[1:]} "
                    f"python greedy={g} qtable={q}")
            stats.states += 1
            stats.greedy_compared += 1
            stats.qtable_compared += 1
        state_before = env.encode_state()
        try:
            state_after, reward, done, info = env.step(a)
        except RuntimeError:
            if s_tok != ["X", "runtime_error"]:
                raise Mismatch(f"{label} step {i}: expected "
                               f"X runtime_error, runner sent {s_tok}")
            stats.exceptions += 1
            continue
        except ValueError:
            if s_tok != ["X", "invalid_argument"]:
                raise Mismatch(f"{label} step {i}: expected "
                               f"X invalid_argument, runner sent {s_tok}")
            stats.exceptions += 1
            continue
        want = ["S", str(state_before), str(state_after), dbl_hex(reward),
                "1" if done else "0", str(REASON_CODE[info.get("reason")]),
                (f"{env.food[0]},{env.food[1]}" if env.food is not None
                 else "-1,-1"),
                str(env.score), str(env.steps),
                ";".join(f"{r},{c}" for r, c in env.body)]
        if s_tok != want:
            raise Mismatch(f"{label} step {i}:\n  runner: {s_tok}\n"
                           f"  python: {want}")
    stats.episodes += 1


def gen_greedy(width, height, seed, max_steps, cap=6000):
    env = SnakeEnv(width, height, seed=seed, max_steps=max_steps)
    actions = []
    while not env.done and len(actions) < cap:
        a = GREEDY.act(env)
        actions.append(a)
        env.step(a)
    return actions


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


def state_fuzz(runner, min_states) -> FuzzStats:
    data = json.loads((ROOT / "native" / "golden" /
                       "q_updates.json").read_text())
    final_q = data["final_q"]
    zeros = [[0.0] * 3 for _ in range(32)]
    pat_rng = random.Random(7)
    pattern = [[pat_rng.uniform(-5.0, 5.0) for _ in range(3)]
               for _ in range(32)]
    tables = [final_q, zeros, pattern]
    stats = FuzzStats()
    counter = 0

    def next_seed(base):
        nonlocal counter
        counter += 1
        return base + 1009 * counter

    def run(label, w, h, seed, ms, actions, table):
        compare_episode(runner, label, w, h, seed, ms, actions, table,
                        stats)

    ti = 0

    def next_table():
        nonlocal ti
        table = tables[ti % len(tables)]
        ti += 1
        return table

    for w, h, count, base in ((12, 12, 18, 800000), (6, 6, 12, 810000),
                              (5, 5, 8, 820000), (4, 4, 8, 830000)):
        for _ in range(count):
            seed = next_seed(base)
            run(f"greedy {w}x{h} seed={seed}", w, h, seed, None,
                gen_greedy(w, h, seed, None), next_table())
    for table, tname in ((final_q, "final_q"), (pattern, "pattern")):
        for w, h, count, base in ((12, 12, 8, 840000), (6, 6, 6, 850000)):
            for _ in range(count):
                seed = next_seed(base)
                run(f"qtable-{tname} {w}x{h} seed={seed}", w, h, seed,
                    None, gen_qtable(w, h, seed, None, table), table)
    for _ in range(4):  # zeros table: every state ties, action always 0
        seed = next_seed(860000)
        run(f"qtable-zeros seed={seed}", 12, 12, seed, None,
            gen_qtable(12, 12, seed, None, zeros), zeros)
    for w, h, count, base in ((12, 12, 15, 870000), (6, 6, 8, 880000)):
        for _ in range(count):
            seed = next_seed(base)
            run(f"random {w}x{h} seed={seed}", w, h, seed, None,
                gen_random(w, h, seed, None, seed + 17), next_table())
    for i in range(6):  # forced timeouts; pre-timeout states still live
        seed = next_seed(890000)
        ms = (5, 25)[i % 2]
        run(f"timeout ms={ms} seed={seed}", 12, 12, seed, ms,
            gen_greedy(12, 12, seed, ms), next_table())
    while stats.states < min_states + 2000:
        seed = next_seed(900000)
        run(f"topup seed={seed}", 12, 12, seed, None,
            gen_greedy(12, 12, seed, None), next_table())
    return stats


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--fixture-only", action="store_true")
    ap.add_argument("--min-states", type=int, default=10000)
    args = ap.parse_args()

    runner = Runner(args.runner)
    try:
        fs = fixture_checks(runner)
        print(f"fixture: updates replayed={fs.updates} "
              f"worst Q error={fs.worst_q_err:.3e} "
              f"(exact entries {fs.q_exact}/96)")
        print(f"fixture: act stream compared={fs.act_compared} exact, "
              f"best_action OK, RandomPolicy draws compared="
              f"{fs.rand_compared} exact")
        if args.fixture_only:
            print("result: PASS")
            return 0
        stats = state_fuzz(runner, args.min_states)
    except Mismatch as exc:
        print(f"MISMATCH: {exc}")
        return 1
    finally:
        runner.close()

    print(f"fuzz: episodes={stats.episodes} states compared="
          f"{stats.states} (greedy {stats.greedy_compared}, qtable "
          f"{stats.qtable_compared}), exceptions={stats.exceptions}")
    ok = (stats.states >= args.min_states
          and stats.greedy_compared == stats.states
          and stats.qtable_compared == stats.states)
    print("result:", "PASS" if ok else "FAIL (state threshold unmet)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
