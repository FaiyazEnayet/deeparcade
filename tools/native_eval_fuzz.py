#!/usr/bin/env python3
"""CPP-05 differential driver: Python evaluate.py vs the native
evaluation engine (narcade_evaluate).

The reference is scripts/evaluate.py's OWN code — run_episodes,
summarize, elo_ratings, head_to_head are imported and called, never
reimplemented. Policy construction mirrors evaluate.py main()
(lines 104-110, 122): RandomPolicy(seed=999), GreedyPolicy(),
QTablePolicy(runs/qtable.json), and ONE shared
MCTSPlayer(simulations=80, seed=31337). Paired protocol: n=50,
seeds 5000..5049, for random / greedy / rookie / grandmaster.

evaluate.py's DQN player ("pro", a PyTorch .pt) is deliberately out
of scope: the native evaluator is a non-neural evaluator and declares
so in its report; this driver asserts that declaration is present.

Step counts: run_episodes returns only scores, so steps are captured
by an observing subclass of the genuine SnakeEnv swapped into
evaluate.py's namespace during its own run_episodes calls — the same
single pass produces scores and steps (the subclass only records
(score, steps) when a step returns done; all semantics are the
genuine env's).

Pre-rounding references: elo_ratings / summarize / head_to_head apply
the builtin round() at their return statements. This driver calls
the genuine functions twice — normally (rounded outputs) and with
builtins.round temporarily replaced by the identity (raw outputs) —
so all reference values come from evaluate.py's own code paths.

Gates: per-episode (score, steps) EXACT on all 200 episodes; report
scores/summary/elo/head_to_head equal to Python's (ints exact,
rounded floats exactly equal as parsed numbers); raw elo and raw
summary mean/std within 1e-12 (bit-exact expected); head_to_head
win/draw counts exact.

Exit code is non-zero on the first mismatch or unmet gate.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from narcade.mcts import MCTSPlayer          # noqa: E402
from narcade.policies import (                # noqa: E402
    GreedyPolicy,
    QTablePolicy,
    RandomPolicy,
)
from narcade.snake import SnakeEnv           # noqa: E402

spec = importlib.util.spec_from_file_location(
    "evaluate_ref", ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

EPISODES = 50
SEED0 = 5000
SIMS = 80
TOL = 1e-12


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def hex_to_float(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def make_policies():
    """evaluate.py main() lines 104-110 and 122, DQN excluded."""
    with open(ROOT / "runs" / "qtable.json") as f:
        qtable = json.load(f)
    return {
        "random": RandomPolicy(seed=999),
        "greedy": GreedyPolicy(),
        "rookie": QTablePolicy(qtable),
        "grandmaster": MCTSPlayer(simulations=SIMS, seed=31337),
    }


class RecordingEnv(SnakeEnv):
    """SnakeEnv that records (score, steps) at each terminal step."""

    records: list = []

    def step(self, action):
        out = super().step(action)
        if out[2]:
            RecordingEnv.records.append((self.score, self.steps))
        return out


def python_reference():
    seeds = list(range(SEED0, SEED0 + EPISODES))
    policies = make_policies()
    scores = {}
    steps = {}
    orig_env = evaluate.SnakeEnv
    evaluate.SnakeEnv = RecordingEnv
    try:
        for name, pol in policies.items():
            RecordingEnv.records = []
            scores[name] = evaluate.run_episodes(pol.act, seeds)
            rec = list(RecordingEnv.records)
            if [s for s, _ in rec] != scores[name]:
                fail(f"reference self-check: {name} recorded scores "
                     f"disagree with run_episodes")
            steps[name] = [st for _, st in rec]
    finally:
        evaluate.SnakeEnv = orig_env

    summary = {k: evaluate.summarize(v) for k, v in scores.items()}
    elo = evaluate.elo_ratings(scores)
    h2h = evaluate.head_to_head(scores)

    # Raw (pre-rounding) outputs from the same functions: identity
    # round() makes their return statements pass values through.
    identity = lambda x, *a: x  # noqa: E731
    with mock.patch.object(builtins, "round", identity):
        raw_elo = evaluate.elo_ratings(scores)
        raw_summary = {k: evaluate.summarize(v) for k, v in scores.items()}
    return seeds, scores, steps, summary, elo, h2h, raw_elo, raw_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True,
                    help="path to the narcade_evaluate executable")
    ap.add_argument("--ref-cache",
                    help="JSON file caching the Python reference "
                         "(written on first run, reused if present); "
                         "the reference itself always comes from "
                         "evaluate.py's own functions")
    args = ap.parse_args()

    if args.ref_cache and Path(args.ref_cache).exists():
        with open(args.ref_cache) as f:
            c = json.load(f)
        seeds, scores, steps = c["seeds"], c["scores"], c["steps"]
        summary, elo, h2h = c["summary"], c["elo"], c["h2h"]
        raw_elo, raw_summary = c["raw_elo"], c["raw_summary"]
        print(f"loaded Python reference from cache {args.ref_cache}")
    else:
        (seeds, scores, steps, summary, elo, h2h,
         raw_elo, raw_summary) = python_reference()
        if args.ref_cache:
            with open(args.ref_cache, "w") as f:
                json.dump({"seeds": seeds, "scores": scores,
                           "steps": steps, "summary": summary, "elo": elo,
                           "h2h": h2h, "raw_elo": raw_elo,
                           "raw_summary": raw_summary}, f)
            print(f"cached Python reference to {args.ref_cache}")

    proc = subprocess.run(
        [args.runner, "--episodes", str(EPISODES), "--seed0", str(SEED0),
         "--sims", str(SIMS), "--qtable", str(ROOT / "runs" / "qtable.json"),
         "--player", "random", "--player", "greedy", "--player", "qtable",
         "--player", "mcts"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"native evaluator exited {proc.returncode}: {proc.stderr}")

    native_eps: dict[str, list[tuple[int, int, int]]] = {}
    raw_elo_n: dict[str, float] = {}
    raw_sum_n: dict[str, tuple[float, float]] = {}
    raw_h2h_n: dict[tuple[str, str], tuple[int, int]] = {}
    report = None
    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "EPISODE":
            _, name, seed, score, st = parts
            native_eps.setdefault(name, []).append(
                (int(seed), int(score), int(st)))
        elif parts[0] == "RAW_SUMMARY":
            raw_sum_n[parts[1]] = (hex_to_float(parts[2]),
                                   hex_to_float(parts[3]))
        elif parts[0] == "RAW_ELO":
            raw_elo_n[parts[1]] = hex_to_float(parts[2])
        elif parts[0] == "RAW_H2H":
            raw_h2h_n[(parts[1], parts[2])] = (int(parts[3]), int(parts[4]))
        elif parts[0] == "REPORT":
            report = json.loads(lines[i + 1])

    if report is None:
        fail("native evaluator printed no REPORT")

    # Scope declaration: the DQN boundary must be explicit.
    scope = report.get("scope", "")
    if "non-neural" not in scope or "pro" not in scope \
            or "NOT covered" not in scope:
        fail(f"report scope declaration missing/incorrect: {scope!r}")
    if "pro" in report["scores_paired"]:
        fail("native report claims DQN ('pro') results")

    if report["paired_seeds"] != seeds:
        fail("paired_seeds mismatch")

    total = 0
    per_player = {}
    for name in ("random", "greedy", "rookie", "grandmaster"):
        eps = native_eps.get(name, [])
        if len(eps) != EPISODES:
            fail(f"{name}: native ran {len(eps)} episodes, want {EPISODES}")
        for i, (seed, score, st) in enumerate(eps):
            if seed != seeds[i]:
                fail(f"{name} ep {i}: seed {seed} != {seeds[i]}")
            if score != scores[name][i]:
                fail(f"{name} seed {seed}: native score {score} != "
                     f"python {scores[name][i]}")
            if st != steps[name][i]:
                fail(f"{name} seed {seed}: native steps {st} != "
                     f"python {steps[name][i]}")
            total += 1
        per_player[name] = len(eps)
        if report["scores_paired"][name] != scores[name]:
            fail(f"{name}: report scores_paired mismatch")
        if report["summary_paired"][name] != summary[name]:
            fail(f"{name}: summary mismatch: "
                 f"native {report['summary_paired'][name]} != "
                 f"python {summary[name]}")
        d_mean = abs(raw_sum_n[name][0] - raw_summary[name]["mean"])
        d_std = abs(raw_sum_n[name][1] - raw_summary[name]["std"])
        if d_mean > TOL or d_std > TOL:
            fail(f"{name}: raw summary deviation mean {d_mean:.3e} "
                 f"std {d_std:.3e} exceeds {TOL}")
        print(f"{name}: {len(eps)} episodes exact; raw summary worst "
              f"deviation {max(d_mean, d_std):.3e}")

    if report["elo"] != elo:
        fail(f"elo mismatch: native {report['elo']} != python {elo}")
    worst_elo = 0.0
    for name, v in raw_elo.items():
        d = abs(raw_elo_n[name] - v)
        worst_elo = max(worst_elo, d)
        if d > TOL:
            fail(f"raw elo {name}: deviation {d:.3e} exceeds {TOL}")
    print(f"elo: ints exact; raw worst deviation {worst_elo:.3e}")

    if report["head_to_head"] != h2h:
        fail(f"head_to_head mismatch:\nnative {report['head_to_head']}\n"
             f"python {h2h}")
    n = EPISODES
    for a in scores:
        for b in scores:
            if a == b:
                continue
            w = sum(1 for i in range(n) if scores[a][i] > scores[b][i])
            d = sum(1 for i in range(n) if scores[a][i] == scores[b][i])
            if raw_h2h_n[(a, b)] != (w, d):
                fail(f"h2h counts {a} vs {b}: native {raw_h2h_n[(a, b)]} "
                     f"!= python {(w, d)}")
    print("head_to_head: rates and counts exact")

    if total != 200:
        fail(f"compared {total} episodes, gate is 200")
    print(f"episodes compared={total} per-player={per_player}")
    print("result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
