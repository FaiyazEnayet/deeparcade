#!/usr/bin/env python3
"""CPP-01b differential fuzz driver: Python SnakeEnv vs the C++ simulator.

Two layers:
  replay  native/golden/transitions.json episodes are fed to the C++
          snake_diff_runner and every recorded field is compared against
          the fixture (and against a live Python re-run, as a drift check).
  fuzz    freshly generated episodes (greedy / random / targeted) are run
          through both implementations; every field of every step must
          agree. Default volume: >= 1,000 episodes and >= 100,000
          transitions, including illegal-action and post-terminal probes,
          forced timeouts (small max_steps), small boards (wins), and
          moving-into-the-vacating-tail steps (counted, must be > 0).

Protocol with the runner is line-based text; see
native/tests/snake_diff_runner.cpp for the format.

Exit code is non-zero on the first mismatch or if the volume/coverage
thresholds are not met.
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

from narcade.policies import GreedyPolicy          # noqa: E402
from narcade.snake import DIRS, SnakeEnv           # noqa: E402

REASON_CODE = {None: 0, "death": 1, "win": 2, "timeout": 3}
GREEDY = GreedyPolicy()


def dbl_hex(x: float) -> str:
    return struct.pack(">d", x).hex()


class Mismatch(Exception):
    pass


# ------------------------------------------------------------ runner link
class Runner:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    def run_episode(self, width, height, seed, max_steps, actions):
        p = self.proc
        ms = -1 if max_steps is None else max_steps
        p.stdin.write(f"EPISODE {width} {height} {seed} {ms} "
                      f"{len(actions)}\n")
        for a in actions:
            p.stdin.write(f"{a}\n")
        p.stdin.flush()
        init = p.stdout.readline().split()
        if not init or init[0] != "INIT":
            raise Mismatch(f"bad INIT line: {init}")
        records = []
        for _ in actions:
            parts = p.stdout.readline().split()
            if not parts:
                raise Mismatch("runner closed the stream mid-episode")
            records.append(parts)
        tail = p.stdout.readline().strip()
        if tail != "DONE":
            raise Mismatch(f"bad episode terminator: {tail!r}")
        return init, records

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


# ------------------------------------------------------- Python reference
def python_episode(width, height, seed, max_steps, actions):
    """Run the Python env over `actions`; return (init, records, meta).

    records mirror the runner's output fields, one per action:
      ("X", "runtime_error"|"invalid_argument") or
      ("S", state_before, state_after, reward_hex, done, reason_code,
            food, score, steps, body)
    meta[i] is True when step i moved into the vacating tail cell.
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


# ------------------------------------------------------------ comparison
class Stats:
    def __init__(self):
        self.episodes = 0
        self.transitions = 0
        self.exceptions = 0
        self.eats = 0
        self.deaths = 0
        self.wins = 0
        self.timeouts = 0
        self.tail_vacations = 0


def compare_episode(label, init_r, recs_r, init_e, recs_e, meta, stats):
    if (int(init_r[1]), parse_food(init_r[2]), parse_body(init_r[3])) != \
            (init_e[0], init_e[1], init_e[2]):
        raise Mismatch(f"{label}: INIT differs: runner={init_r} "
                       f"python={init_e}")
    for i, (got, exp) in enumerate(zip(recs_r, recs_e)):
        if exp[0] == "X":
            if got != ["X", exp[1]]:
                raise Mismatch(f"{label} step {i}: expected {exp}, "
                               f"runner sent {got}")
            stats.exceptions += 1
            continue
        # Expected token list in the runner's field order.
        want = ["S", str(exp[1]), str(exp[2]), exp[3], str(exp[4]),
                str(exp[5]),
                f"{exp[6][0]},{exp[6][1]}" if exp[6] is not None
                else "-1,-1",
                str(exp[7]), str(exp[8]),
                ";".join(f"{r},{c}" for r, c in exp[9])]
        if got != want:
            raise Mismatch(f"{label} step {i}:\n  runner: {got}\n"
                           f"  python: {want}")
        stats.transitions += 1
        if exp[3] == dbl_hex(10.0):
            stats.eats += 1
        if exp[5] == 1:
            stats.deaths += 1
        elif exp[5] == 2:
            stats.wins += 1
        elif exp[5] == 3:
            stats.timeouts += 1
        if meta[i]:
            stats.tail_vacations += 1
    stats.episodes += 1


def run_and_compare(runner, label, width, height, seed, max_steps, actions,
                    stats):
    init_e, recs_e, meta = python_episode(width, height, seed, max_steps,
                                          actions)
    init_r, recs_r = runner.run_episode(width, height, seed, max_steps,
                                        actions)
    compare_episode(label, init_r, recs_r, init_e, recs_e, meta, stats)


# ------------------------------------------------------- action generation
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


# ------------------------------------------------------------------ replay
def replay(runner, stats):
    data = json.loads((ROOT / "native" / "golden" /
                       "transitions.json").read_text())
    for ep in data["episodes"]:
        steps = ep["steps"]
        actions = [s["action"] for s in steps]
        label = f"replay seed={ep['seed']}"
        init_r, recs_r = runner.run_episode(ep["width"], ep["height"],
                                            ep["seed"], None, actions)
        # Runner vs the recorded fixture, field by field.
        first = steps[0]
        want_init = (first["state_before"],
                     tuple(first["food_before"]),
                     [tuple(c) for c in first["body_before"]])
        got_init = (int(init_r[1]), parse_food(init_r[2]),
                    parse_body(init_r[3]))
        if got_init != want_init:
            raise Mismatch(f"{label}: INIT differs from fixture: "
                           f"runner={got_init} fixture={want_init}")
        for i, (got, s) in enumerate(zip(recs_r, steps)):
            want = ["S", str(s["state_before"]), str(s["state_after"]),
                    dbl_hex(s["reward"]), "1" if s["done"] else "0",
                    str(REASON_CODE[s["reason"]]),
                    (f"{s['food_after'][0]},{s['food_after'][1]}"
                     if s["food_after"] is not None else "-1,-1"),
                    str(s["score"]), str(s["steps"]),
                    ";".join(f"{r},{c}" for r, c in s["body_after"])]
            if got != want:
                raise Mismatch(f"{label} step {i}:\n  runner:  {got}\n"
                               f"  fixture: {want}")
            stats.transitions += 1
            if s["reason"] == "death":
                stats.deaths += 1
            elif s["reason"] == "win":
                stats.wins += 1
            elif s["reason"] == "timeout":
                stats.timeouts += 1
            if s["reward"] == 10.0:
                stats.eats += 1
        stats.episodes += 1
        # Drift check: live Python must still reproduce the fixture.
        run_and_compare_drift(ep, actions)


def run_and_compare_drift(ep, actions):
    init_e, recs_e, _ = python_episode(ep["width"], ep["height"], ep["seed"],
                                       None, actions)
    first = ep["steps"][0]
    if init_e != (first["state_before"], tuple(first["food_before"]),
                  [tuple(c) for c in first["body_before"]]):
        raise Mismatch(f"replay seed={ep['seed']}: Python INIT drifted "
                       f"from fixture")
    for i, (rec, s) in enumerate(zip(recs_e, ep["steps"])):
        want = ("S", s["state_before"], s["state_after"],
                dbl_hex(s["reward"]), 1 if s["done"] else 0,
                REASON_CODE[s["reason"]],
                tuple(s["food_after"]) if s["food_after"] is not None
                else None,
                s["score"], s["steps"],
                [tuple(c) for c in s["body_after"]])
        if rec != want:
            raise Mismatch(f"replay seed={ep['seed']} step {i}: Python "
                           f"drifted from fixture:\n  live:    {rec}\n"
                           f"  fixture: {want}")


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--min-transitions", type=int, default=100000)
    ap.add_argument("--min-episodes", type=int, default=1000)
    args = ap.parse_args()

    runner = Runner(args.runner)
    stats = Stats()
    try:
        replay(runner, stats)
        print(f"replay: episodes={stats.episodes} "
              f"transitions={stats.transitions} OK")
        if args.replay_only:
            return 0

        counter = 0

        def next_seed(base):
            nonlocal counter
            counter += 1
            return base + 1009 * counter

        # Greedy volume across board sizes (small boards force wins).
        for width, height, count, base in ((12, 12, 400, 700000),
                                           (6, 6, 150, 710000),
                                           (5, 5, 100, 720000),
                                           (4, 4, 200, 730000)):
            for _ in range(count):
                seed = next_seed(base)
                acts = greedy_actions(width, height, seed, None)
                run_and_compare(runner, f"greedy {width}x{height} "
                                f"seed={seed}", width, height, seed, None,
                                acts, stats)
        # Uniformly random action streams (+ post-terminal probes).
        for width, height, count, base in ((12, 12, 150, 740000),
                                           (6, 6, 50, 750000)):
            for _ in range(count):
                seed = next_seed(base)
                acts = random_actions(width, height, seed, None, seed + 17)
                run_and_compare(runner, f"random {width}x{height} "
                                f"seed={seed}", width, height, seed, None,
                                acts, stats)
        # Targeted: illegal actions mid-episode, then play continues.
        for i in range(20):
            seed = next_seed(760000)
            plan = ["g"] * 5 + [7, -1, 3] + ["g"] * 25
            acts = plan_actions(12, 12, seed, None, plan, seed + 23)
            run_and_compare(runner, f"illegal seed={seed}", 12, 12, seed,
                            None, acts, stats)
        # Targeted: steps after the episode is done (legal and illegal;
        # Python checks done first, so all must raise RuntimeError).
        for i in range(20):
            seed = next_seed(770000)
            acts = greedy_actions(12, 12, seed, None) + [0, 1, 9]
            run_and_compare(runner, f"after-done seed={seed}", 12, 12,
                            seed, None, acts, stats)
        # Targeted: forced timeouts via small max_steps.
        for i in range(30):
            seed = next_seed(780000)
            ms = (1, 3, 7)[i % 3]
            acts = greedy_actions(12, 12, seed, ms)
            run_and_compare(runner, f"timeout ms={ms} seed={seed}", 12, 12,
                            seed, ms, acts, stats)
        # Top-up: guarantee the volume thresholds deterministically.
        while stats.transitions < args.min_transitions + 10000 or \
                stats.episodes < args.min_episodes:
            seed = next_seed(790000)
            acts = greedy_actions(12, 12, seed, None)
            run_and_compare(runner, f"topup seed={seed}", 12, 12, seed,
                            None, acts, stats)
    except Mismatch as exc:
        print(f"MISMATCH: {exc}")
        return 1
    finally:
        runner.close()

    print(f"episodes compared:   {stats.episodes}")
    print(f"transitions compared:{stats.transitions}")
    print(f"exceptions compared: {stats.exceptions}")
    print(f"eats: {stats.eats}  deaths: {stats.deaths}  wins: {stats.wins}  "
          f"timeouts: {stats.timeouts}")
    print(f"tail-vacation steps: {stats.tail_vacations}")
    ok = (stats.transitions >= args.min_transitions
          and stats.episodes >= args.min_episodes
          and stats.wins > 0 and stats.timeouts > 0
          and stats.tail_vacations > 0 and stats.exceptions > 0)
    print("result:", "PASS" if ok else "FAIL (coverage thresholds unmet)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
