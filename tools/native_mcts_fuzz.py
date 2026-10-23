#!/usr/bin/env python3
"""CPP-03 differential driver: Python MCTSPlayer vs the C++ port.

Layers:
  oracle  native/golden/mcts_oracle.json is reproduced exactly by
          the C++ runner (fresh-player actions, tiny budgets, small
          board, and the shared-player 30-action episode including
          final score/steps). Each case is also recomputed live in
          Python first, so a fixture/Python drift fails loudly here
          rather than masquerading as a C++ mismatch.
  fuzz    >= 1,000 mid-game states harvested from live SnakeEnv
          episodes across boards 12x12/6x6/5x5/4x4 and varied seeds.
          For each state a fresh Python MCTSPlayer(seed=S,
          simulations in {1, 2, 10, 80}) and a fresh C++ player with
          identical parameters search the same serialized state;
          actions must match EXACTLY. Root children (action, visits,
          value) are compared for every search: visits exact, value
          worst deviation reported (gate <= 1e-12).

The Python root is captured by monkeypatching _Node.__init__ during
search() so the genuine Python search code path produces the
reference children; nothing about the reference is reimplemented.

Protocol with the runner is line-based text; see
native/tests/mcts_diff_runner.cpp for the format. Doubles travel as
IEEE754 hex bit patterns.

Exit code is non-zero on the first mismatch or if a gate is not met.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import narcade.mcts as mcts_mod                                # noqa: E402
from narcade.mcts import MCTSPlayer                            # noqa: E402
from narcade.policies import GreedyPolicy                      # noqa: E402
from narcade.snake import SnakeEnv                             # noqa: E402

REASON_CODE = {None: 0, "death": 1, "win": 2, "timeout": 3}
GREEDY = GreedyPolicy()
ROLLOUT_STEPS = 100
C_PARAM = 1.0


def dbl_hex(x: float) -> str:
    return struct.pack(">d", x).hex()


def hex_dbl(tok: str) -> float:
    return struct.unpack(">d", bytes.fromhex(tok))[0]


class Mismatch(Exception):
    pass


# ------------------------------------------------------- Python reference
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

    def player(self, simulations, seed):
        self._send(f"PLAYER {simulations} {ROLLOUT_STEPS} "
                   f"{dbl_hex(C_PARAM)} {seed}\n")
        if self._line() != "OK":
            raise Mismatch("missing OK after PLAYER")

    def env(self, width, height, seed):
        self._send(f"ENV {width} {height} {seed}\n")
        init = self._line().split()
        if not init or init[0] != "INIT":
            raise Mismatch(f"bad INIT line: {init}")
        return init

    def _search_reply(self):
        a_tok = self._line().split()
        if not a_tok or a_tok[0] != "ACTION":
            raise Mismatch(f"bad ACTION line: {a_tok}")
        r_tok = self._line().split()
        if not r_tok or r_tok[0] != "ROOT":
            raise Mismatch(f"bad ROOT line: {r_tok}")
        children = []
        for _ in range(int(r_tok[1])):
            c_tok = self._line().split()
            if not c_tok or c_tok[0] != "CHILD":
                raise Mismatch(f"bad CHILD line: {c_tok}")
            children.append((int(c_tok[1]), int(c_tok[2]),
                             hex_dbl(c_tok[3])))
        return int(a_tok[1]), children

    def search(self):
        self._send("SEARCH\n")
        return self._search_reply()

    def step(self, action):
        self._send(f"STEP {action}\n")
        return self._line().split()

    def search_state(self, width, height, body, food, heading):
        self._send(f"SEARCHSTATE {width} {height} {food[0]} {food[1]} "
                   f"{heading} {len(body)}\n")
        for r, c in body:
            self._send(f"{r} {c}\n")
        return self._search_reply()

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


def check_init(init_tok, env, label):
    got = (int(init_tok[1]), parse_food(init_tok[2]),
           parse_body(init_tok[3]))
    want = (env.encode_state(),
            tuple(env.food) if env.food is not None else None,
            [tuple(c) for c in env.body])
    if got != want:
        raise Mismatch(f"{label}: INIT differs: runner={got} "
                       f"python={want}")


def compare_children(got, want, label, stats):
    if [c[0] for c in got] != [c[0] for c in want]:
        raise Mismatch(f"{label}: root child order differs: "
                       f"runner={[c[0] for c in got]} "
                       f"python={[c[0] for c in want]}")
    for (ga, gv, gval), (_, wv, wval) in zip(got, want):
        if gv != wv:
            raise Mismatch(f"{label}: root child {ga} visits differ: "
                           f"runner={gv} python={wv}")
        dev = abs(gval - wval)
        stats["root_children"] += 1
        stats["worst_dev"] = max(stats["worst_dev"], dev)
        if dev == 0.0:
            stats["bitexact"] += 1


# ------------------------------------------------------------ oracle layer
def oracle_checks(runner) -> dict:
    data = json.loads((ROOT / "native" / "golden" /
                       "mcts_oracle.json").read_text())
    stats = {"cases": 0, "actions": 0, "root_children": 0,
             "worst_dev": 0.0, "bitexact": 0}
    for case in data["cases"]:
        kind = case["kind"]
        sims = case["simulations"]
        pseed = case["player_seed"]
        if kind in ("fresh_player", "tiny_budget", "small_board"):
            w = case.get("width", 12)
            h = case.get("height", 12)
            env = SnakeEnv(w, h, seed=case["env_seed"])
            py = MCTSPlayer(simulations=sims, seed=pseed)
            want_action, want_children = python_search_with_root(py, env)
            if want_action != case["action"]:
                raise Mismatch(
                    f"oracle {kind} env_seed={case['env_seed']}: "
                    f"Python drift: live={want_action} "
                    f"fixture={case['action']}")
            init = runner.env(w, h, case["env_seed"])
            check_init(init, env, f"oracle {kind}")
            runner.player(sims, pseed)
            got_action, got_children = runner.search()
            if got_action != case["action"]:
                raise Mismatch(
                    f"oracle {kind} env_seed={case['env_seed']} "
                    f"sims={sims} pseed={pseed}: runner action="
                    f"{got_action} fixture={case['action']}")
            compare_children(got_children, want_children,
                             f"oracle {kind}", stats)
            stats["actions"] += 1
        elif kind == "shared_player_episode":
            env = SnakeEnv(12, 12, seed=case["env_seed"])
            init = runner.env(12, 12, case["env_seed"])
            check_init(init, env, "oracle shared_player_episode")
            runner.player(sims, pseed)
            py = MCTSPlayer(simulations=sims, seed=pseed)
            for i, want_a in enumerate(case["actions"]):
                py_a, _ = python_search_with_root(py, env)
                if py_a != want_a:
                    raise Mismatch(
                        f"oracle shared step {i}: Python drift: "
                        f"live={py_a} fixture={want_a}")
                got_a, _ = runner.search()
                if got_a != want_a:
                    raise Mismatch(
                        f"oracle shared step {i}: runner action="
                        f"{got_a} fixture={want_a}")
                stats["actions"] += 1
                env.step(want_a)
                s_tok = runner.step(want_a)
                if not s_tok or s_tok[0] != "S":
                    raise Mismatch(
                        f"oracle shared step {i}: bad STEP reply "
                        f"{s_tok}")
                last_score, last_steps = int(s_tok[7]), int(s_tok[8])
            if (last_score, last_steps) != (case["final_score"],
                                             case["final_steps"]):
                raise Mismatch(
                    f"oracle shared: runner final score/steps="
                    f"({last_score}, {last_steps}) fixture="
                    f"({case['final_score']}, {case['final_steps']})")
            if (env.score, env.steps) != (case["final_score"],
                                           case["final_steps"]):
                raise Mismatch("oracle shared: Python env drift on "
                               "final score/steps")
        else:
            raise Mismatch(f"unknown oracle case kind {kind}")
        stats["cases"] += 1
    if stats["worst_dev"] > 1e-12:
        raise Mismatch(f"oracle root value worst deviation "
                       f"{stats['worst_dev']:.3e} exceeds 1e-12")
    return stats


# --------------------------------------------------------------- fuzz layer
def harvest_states(min_states):
    states = []
    boards = [(12, 12), (6, 6), (5, 5), (4, 4)]
    seed = 700000
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


def fuzz_searches(runner, min_searches) -> dict:
    stats = {"searches": 0, "root_children": 0, "root_searches": 0,
             "worst_dev": 0.0, "bitexact": 0}
    states = harvest_states(min_searches)
    sims_cycle = (80, 10, 2, 1, 10, 2)
    pseeds = (31337, 101, 202, 303, 404, 7)
    for i, (w, h, body, food, heading) in enumerate(states):
        sims = sims_cycle[i % len(sims_cycle)]
        pseed = pseeds[i % len(pseeds)]
        env = env_view(w, h, body, food, heading)
        py = MCTSPlayer(simulations=sims, seed=pseed)
        want_action, want_children = python_search_with_root(py, env)
        runner.player(sims, pseed)
        got_action, got_children = runner.search_state(w, h, body, food,
                                                       heading)
        label = (f"fuzz #{i} {w}x{h} sims={sims} pseed={pseed} "
                 f"body={len(body)} head={body[-1]} food={food} "
                 f"heading={heading}")
        if got_action != want_action:
            raise Mismatch(f"{label}: runner action={got_action} "
                           f"python={want_action}")
        compare_children(got_children, want_children, label, stats)
        stats["searches"] += 1
        stats["root_searches"] += 1
        if stats["searches"] >= min_searches and i >= min_searches - 1:
            break
    if stats["worst_dev"] > 1e-12:
        raise Mismatch(f"fuzz root value worst deviation "
                       f"{stats['worst_dev']:.3e} exceeds 1e-12")
    return stats


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--min-searches", type=int, default=1000)
    args = ap.parse_args()

    runner = Runner(args.runner)
    try:
        os_ = oracle_checks(runner)
        print(f"oracle: cases={os_['cases']} actions compared="
              f"{os_['actions']} exact, root children compared="
              f"{os_['root_children']} (bitexact {os_['bitexact']}, "
              f"worst dev {os_['worst_dev']:.3e})")
        if args.oracle_only:
            print("result: PASS")
            return 0
        fs = fuzz_searches(runner, args.min_searches)
    except Mismatch as exc:
        print(f"MISMATCH: {exc}")
        return 1
    finally:
        runner.close()

    print(f"fuzz: searches compared={fs['searches']} exact actions, "
          f"root children compared={fs['root_children']} "
          f"(bitexact {fs['bitexact']}, worst dev "
          f"{fs['worst_dev']:.3e})")
    ok = (fs["searches"] >= args.min_searches
          and fs["root_searches"] >= 100)
    print("result:", "PASS" if ok else "FAIL (search threshold unmet)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
