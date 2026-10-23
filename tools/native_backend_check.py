#!/usr/bin/env python3
"""CPP-04 check: build-or-locate the narcade_native extension and verify
the pybind11 bindings + optional backend against the Python reference.

Usage:
    python3 tools/native_backend_check.py [--build-dir DIR]

If the extension is not importable (directly or from an existing
native/build* directory), it is built into native/build with CMake
first. The comparisons mirror tests/test_native_backend.py: CPython RNG
draws, Q-learning acts/updates, full Snake episodes (raw module and the
optional_backend adapter) and MCTS actions, all of which must match the
Python implementation exactly. Prints per-section counts and a final
PASS/FAIL; exit code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def locate_or_build(build_dir):
    try:
        import narcade_native

        return narcade_native
    except ImportError:
        pass
    candidates = [build_dir] if build_dir else []
    candidates += sorted(glob.glob(os.path.join(ROOT, "native", "build*")))
    for cand in candidates:
        if cand and os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import narcade_native

                return narcade_native
            except ImportError:
                sys.path.remove(cand)
    target = build_dir or os.path.join(ROOT, "native", "build")
    print(f"extension not found; building into {target}")
    subprocess.run(
        ["cmake", "-S", os.path.join(ROOT, "native"), "-B", target,
         "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(["cmake", "--build", target, "-j4"], check=True)
    if target not in sys.path:
        sys.path.insert(0, target)
    import narcade_native

    return narcade_native


class Section:
    def __init__(self, name):
        self.name = name
        self.checked = 0
        self.mismatches = 0

    def expect(self, ok):
        self.checked += 1
        if not ok:
            self.mismatches += 1

    def report(self):
        status = "PASS" if self.mismatches == 0 else "FAIL"
        print(f"[{status}] {self.name}: {self.checked} compared, "
              f"{self.mismatches} mismatches")
        return self.mismatches == 0


def check_rng(native):
    from narcade.qlearn import QLearningAgent as PyAgent

    sec = Section("PythonRandom vs CPython random.Random")
    for seed in (0, 7, 999, 31337):
        ref = random.Random(seed)
        nat = native.PythonRandom(seed)
        for _ in range(500):
            sec.expect(nat.random() == ref.random())
        for k in (0, 1, 2, 8, 16, 31, 32, 33, 53, 64):
            for _ in range(20):
                sec.expect(nat.getrandbits(k) == ref.getrandbits(k))
        for _ in range(200):
            sec.expect(nat.randint(-5, 5) == ref.randint(-5, 5))
        for _ in range(200):
            sec.expect(nat.randrange(1000) == ref.randrange(1000))
        for _ in range(200):
            sec.expect(nat.choice_index(7) == ref.choice(list(range(7))))

    sec_q = Section("QLearningAgent acts/updates vs Python")
    ref = PyAgent(epsilon=0.25, seed=2024)
    nat = native.QLearningAgent(32, 3, 0.15, 0.9, 0.25, 2024)
    driver = random.Random(99)
    rewards = (10.0, -10.0, 0.5, -0.5, 0.0)
    for _ in range(1000):
        s = driver.randrange(32)
        sec_q.expect(nat.act(s) == ref.act(s))
        a = driver.randrange(3)
        r = rewards[driver.randrange(len(rewards))]
        s_next = driver.randrange(32)
        done = driver.randrange(16) == 0
        nat.update(s, a, r, s_next, done)
        ref.update(s, a, r, s_next, done)
    sec_q.expect(nat.q == ref.q)
    return [sec, sec_q]


def check_episodes(native):
    from narcade import optional_backend
    from narcade.policies import GreedyPolicy
    from narcade.snake import SnakeEnv as PyEnv

    os.environ["NARCADE_NATIVE"] = "1"
    sec = Section("Snake episodes (raw module + adapter) vs Python")
    greedy = GreedyPolicy()
    episodes = 0
    for seed in (11, 222, 3333, 424242):
        py_env = PyEnv(12, 12, seed=seed, max_steps=400)
        nat_env = native.SnakeEnv(12, 12, seed, 400)
        ad_env = optional_backend.SnakeEnv(
            width=12, height=12, seed=seed, max_steps=400
        )
        for _ in range(6):
            sec.expect(
                nat_env.reset() == py_env.reset() == ad_env.reset()
            )
            while not py_env.done:
                action = greedy.act(py_env)
                sec.expect(native.greedy_policy_act(nat_env) == action)
                p = py_env.step(action)
                n = nat_env.step(action)
                a = ad_env.step(action)
                sec.expect((n[0], n[1], n[2]) == (p[0], p[1], p[2]))
                sec.expect((a[0], a[1], a[2]) == (p[0], p[1], p[2]))
                sec.expect(nat_env.score == py_env.score == ad_env.score)
                sec.expect(
                    a[3] == ({"reason": n[3]} if n[3] is not None else {})
                )
            episodes += 1
    print(f"       episodes compared: {episodes}")
    return [sec]


def check_mcts(native):
    from narcade import optional_backend
    from narcade.mcts import MCTSPlayer as PyPlayer
    from narcade.policies import GreedyPolicy
    from narcade.snake import SnakeEnv as PyEnv

    os.environ["NARCADE_NATIVE"] = "1"
    sec = Section("MCTSPlayer actions vs Python")
    params = dict(simulations=50, rollout_steps=100, c=1.0, seed=31337)
    py_player = PyPlayer(**params)
    nat_player = native.MCTSPlayer(50, 100, 1.0, 31337)
    ad_player = optional_backend.MCTSPlayer(**params)
    greedy = GreedyPolicy()
    compared = 0
    for seed in (5000, 5001, 5002):
        env = PyEnv(12, 12, seed=seed)
        while not env.done and compared < 60:
            expected = py_player.search(env)
            sec.expect(nat_player.search(env) == expected)
            sec.expect(ad_player.search(env) == expected)
            compared += 1
            env.step(greedy.act(env))
    print(f"       states compared: {compared}")
    return [sec]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=None,
                        help="existing build dir to look in / build into")
    args = parser.parse_args()

    native = locate_or_build(args.build_dir)
    print(f"narcade_native loaded from {native.__file__}")

    sections = []
    sections += check_rng(native)
    sections += check_episodes(native)
    sections += check_mcts(native)
    ok = all(sec.report() for sec in sections)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
