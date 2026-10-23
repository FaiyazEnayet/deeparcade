#!/usr/bin/env python3
"""CPP-06b parity driver: scripts/analyze.py vs native instrumentation.

Runs scripts/analyze.py's own analyze() (unmodified) on the Python
side and the native analyze_main on identical policies and seeds:

  random      RandomPolicy(seed=77)   (one persistent instance)
  greedy      GreedyPolicy()
  rookie      QTablePolicy(runs/qtable.json)
  grandmaster MCTSPlayer(simulations=80, seed=4242)

at analyze.py's seed bases (31000 fast policies, 33000 grandmaster).
Aggregated outputs are rounded decimals and integer counters, so the
comparison is EXACT equality per policy: action fractions, death
Counter, food_per_100_steps, episode count.

"pro" (DQN) is excluded on BOTH sides of this check with this note:
native DQN inference is a separate module (CPP-06d); analyze.py's
DQNPolicy needs the PyTorch checkpoint, which is not part of the
native instrumentation's scope.

scripts/analyze.py imports torch, so when torch is unavailable this
driver re-execs itself once under the flowforge venv python
(~/workspace/flowforge/.venv/bin/python), which has torch.

Usage: native_analyze_check.py --analyzer /path/to/analyze_main
       [--n 200] [--mcts-n 30] [--mcts-sims 80] [--fast]
Exit code is non-zero on the first mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch  # noqa: F401  (scripts/analyze.py imports it)
except ImportError:
    venv_py = Path.home() / "workspace" / "flowforge" / ".venv" / "bin" / "python"
    if not os.environ.get("NARCADE_ANALYZE_REEXEC") and venv_py.exists():
        os.environ["NARCADE_ANALYZE_REEXEC"] = "1"
        os.execv(str(venv_py),
                 [str(venv_py), str(Path(__file__).resolve()),
                  *sys.argv[1:]])
    raise

import scripts.analyze as az                    # noqa: E402
from narcade.mcts import MCTSPlayer             # noqa: E402
from narcade.policies import (GreedyPolicy,     # noqa: E402
                              QTablePolicy, RandomPolicy)


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyzer", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mcts-n", type=int, default=30)
    ap.add_argument("--mcts-sims", type=int, default=80)
    ap.add_argument("--fast", action="store_true",
                    help="smaller episode counts (n=40, mcts_n=8)")
    args = ap.parse_args()
    n = 40 if args.fast else args.n
    mcts_n = 8 if args.fast else args.mcts_n

    with open(ROOT / "runs" / "qtable.json") as f:
        qtable = json.load(f)

    # Python reference: policy set built exactly like analyze.py:62-77.
    reference = {
        "random": az.analyze(RandomPolicy(seed=77).act, n, seed0=31000),
        "greedy": az.analyze(GreedyPolicy().act, n, seed0=31000),
        "rookie": az.analyze(QTablePolicy(qtable).act, n, seed0=31000),
        "grandmaster": az.analyze(
            MCTSPlayer(simulations=args.mcts_sims, seed=4242).act,
            mcts_n, seed0=33000),
    }

    # Native run: needs qtable.json in its --out dir; writes moves.json.
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(ROOT / "runs" / "qtable.json",
                    Path(td) / "qtable.json")
        res = subprocess.run(
            [str(args.analyzer), "--out", td,
             "--n", str(n), "--mcts-n", str(mcts_n),
             "--mcts-sims", str(args.mcts_sims)],
            capture_output=True, text=True)
        if res.returncode != 0:
            fail(f"native analyzer exited {res.returncode}: {res.stderr}")
        native = json.loads((Path(td) / "moves.json").read_text())

    if set(native) != set(reference):
        fail(f"policy sets differ: native {sorted(native)} vs "
             f"python {sorted(reference)}")
    for name, want in reference.items():
        got = native[name]
        for key in ("actions", "deaths", "food_per_100_steps", "episodes"):
            if got[key] != want[key]:
                fail(f"[{name}] {key}: native {got[key]} != "
                     f"python {want[key]}")
        print(f"[{name}] PASS  actions={got['actions']} "
              f"deaths={got['deaths']} "
              f"food/100steps={got['food_per_100_steps']} "
              f"episodes={got['episodes']}")
    print("result: PASS (pro/DQN excluded on both sides; see module "
          "note)")


if __name__ == "__main__":
    main()
