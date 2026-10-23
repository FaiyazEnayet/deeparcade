#!/usr/bin/env python3
"""CPP-06d differential driver: C++ DQN inference vs the PyTorch reference.

Must run under a Python with torch installed (the flowforge venv:
~/workspace/flowforge/.venv/bin/python). The reference is the genuine
scripts/train_dqn.py DQN architecture with the exported
runs/dqn_weights.json loaded into it; nothing about the reference is
reimplemented. Reference encodings come from train_dqn.encode_dqn
itself, applied to namespace views of the harvested states.

Harvest: >= --states states (default 12,000) recorded before every
step of seeded 12x12 SnakeEnv episodes, rotating greedy / random /
qtable policies by episode, so early, mid and late game positions and
the near-death states that end random episodes are all covered. Seeds
are fixed; the harvest is deterministic.

Comparisons (all reported, nothing hidden):
  encodings   bit-identical on every state (gate: 0 mismatches)
  logits      max abs difference over all states and actions
  actions     C++ argmax vs torch argmax. Every disagreement is
              printed with the reference Q-margin (top1 - top2) and
              the state's max logit difference. A disagreement with
              margin > 1e-5 (the predeclared float32 investigation
              band) fails the gate; near-tie disagreements inside the
              band are reported as warnings.
  activations layer-1 / layer-2 post-ReLU max abs differences on the
              first --act-subset states (default 500).

Protocol with the runner is line-based text; see
native/tests/dqn_diff_runner.cpp. Floats travel as IEEE754 binary32
hex bit patterns.

Exit code is non-zero if a gate fails.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy  # noqa: E402
from narcade.snake import SnakeEnv  # noqa: E402
from scripts.train_dqn import DQN, encode_dqn  # noqa: E402

MARGIN_BAND = 1e-5


def f32_from_hex(tok: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(tok, 16)), dtype=np.float32)[0]


def harvest(target: int):
    """Deterministic state harvest; returns [(body, food, heading)]."""
    with open(ROOT / "runs" / "qtable.json") as f:
        qtable = json.load(f)
    policies = [GreedyPolicy(), RandomPolicy(seed=777), QTablePolicy(qtable)]
    states = []
    ep = 0
    while len(states) < target and ep < 100000:
        env = SnakeEnv(width=12, height=12, seed=5000 + ep)
        env.reset()
        pol = policies[ep % 3]
        if isinstance(pol, RandomPolicy):
            pol = RandomPolicy(seed=777 + ep)
        done = False
        while not done:
            states.append((list(env.body), env.food, env.heading))
            _, _, done, _ = env.step(pol.act(env))
        ep += 1
    return states


def load_reference(weights_path: Path):
    with open(weights_path) as f:
        exported = json.load(f)
    layers = exported["layers"]
    sd = {}
    for i, layer in enumerate(layers):
        sd[f"net.{2 * i}.weight"] = torch.tensor(layer["w"], dtype=torch.float32)
        sd[f"net.{2 * i}.bias"] = torch.tensor(layer["b"], dtype=torch.float32)
    net = DQN()
    net.load_state_dict(sd)
    net.eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--weights", default=str(ROOT / "runs" / "dqn_weights.json"))
    ap.add_argument("--states", type=int, default=12000)
    ap.add_argument("--act-subset", type=int, default=500)
    args = ap.parse_args()

    states = harvest(args.states)
    n = len(states)
    print(f"harvested {n} states from seeded episodes")

    # Reference encodings / logits / activations from the genuine code.
    encs = np.stack([
        encode_dqn(SimpleNamespace(width=12, height=12, body=list(body),
                                   food=food, heading=heading))
        for body, food, heading in states
    ])
    net = load_reference(Path(args.weights))
    with torch.no_grad():
        x = torch.from_numpy(encs)
        h1 = torch.relu(net.net[0](x)).numpy()
        h2 = torch.relu(net.net[2](torch.from_numpy(h1))).numpy()
        logits = net.net[4](torch.from_numpy(h2)).numpy()
    ref_actions = logits.argmax(axis=1)
    top2 = np.partition(logits, -2, axis=1)
    margins = top2[:, -1] - top2[:, -2]

    proc = subprocess.Popen([args.runner], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdin is not None and proc.stdout is not None

    def send(line: str) -> None:
        proc.stdin.write(line + "\n")

    def recv() -> str:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("runner closed stdout unexpectedly")
        return line.strip()

    send(f"LOAD {args.weights}")
    proc.stdin.flush()
    reply = recv()
    if reply != "OK":
        print(f"FAIL: runner LOAD replied: {reply}")
        proc.kill()
        return 1

    enc_mismatch = 0
    action_disagree = []
    max_logit_diff = 0.0
    max_h1_diff = 0.0
    max_h2_diff = 0.0
    enc_bits_ref = encs.view(np.uint32)

    for i, (body, food, heading) in enumerate(states):
        mode = 1 if i < args.act_subset else 0
        send(f"STATE {mode} {food[0]} {food[1]} {heading} {len(body)}")
        for r, c in body:
            send(f"{r} {c}")
        proc.stdin.flush()

        enc_line = recv().split()
        if enc_line[0] != "ENC" or len(enc_line) != 1 + 436:
            print(f"FAIL: state {i}: bad ENC reply ({enc_line[0]}, "
                  f"{len(enc_line)} tokens)")
            proc.kill()
            return 1
        cpp_bits = np.array([int(t, 16) for t in enc_line[1:]],
                            dtype=np.uint32)
        if not np.array_equal(cpp_bits, enc_bits_ref[i]):
            enc_mismatch += 1
            if enc_mismatch <= 3:
                idx = int(np.nonzero(cpp_bits != enc_bits_ref[i])[0][0])
                print(f"  encoding mismatch state {i} at {idx}: "
                      f"cpp={enc_line[1 + idx]} ref={enc_bits_ref[i][idx]:08x}")

        out_line = recv().split()
        if out_line[0] != "OUT" or len(out_line) != 5:
            print(f"FAIL: state {i}: bad OUT reply")
            proc.kill()
            return 1
        cpp_action = int(out_line[1])
        cpp_logits = np.array([f32_from_hex(t) for t in out_line[2:5]],
                              dtype=np.float32)
        diff = float(np.max(np.abs(cpp_logits.astype(np.float64)
                                   - logits[i].astype(np.float64))))
        max_logit_diff = max(max_logit_diff, diff)
        if cpp_action != int(ref_actions[i]):
            action_disagree.append((i, cpp_action, int(ref_actions[i]),
                                    float(margins[i]), diff))
        if mode == 1:
            h1_line = recv().split()
            h2_line = recv().split()
            if h1_line[0] != "H1" or h2_line[0] != "H2":
                print(f"FAIL: state {i}: bad H1/H2 reply")
                proc.kill()
                return 1
            cpp_h1 = np.array([f32_from_hex(t) for t in h1_line[1:]],
                              dtype=np.float32)
            cpp_h2 = np.array([f32_from_hex(t) for t in h2_line[1:]],
                              dtype=np.float32)
            max_h1_diff = max(max_h1_diff, float(np.max(np.abs(
                cpp_h1.astype(np.float64) - h1[i].astype(np.float64)))))
            max_h2_diff = max(max_h2_diff, float(np.max(np.abs(
                cpp_h2.astype(np.float64) - h2[i].astype(np.float64)))))

    proc.stdin.close()
    proc.wait(timeout=60)

    lens = [len(b) for b, _, _ in states]
    print(f"body length coverage: min={min(lens)} max={max(lens)}")
    print(f"encoding mismatches: {enc_mismatch} / {n}")
    print(f"max abs logit diff: {max_logit_diff:.3e}")
    print(f"action agreement: {n - len(action_disagree)} / {n}")
    for i, ca, ra, margin, diff in action_disagree:
        print(f"  DISAGREE state {i}: cpp={ca} ref={ra} "
              f"ref_margin={margin:.3e} max_logit_diff={diff:.3e}")
    print(f"activations (first {args.act_subset} states): "
          f"max h1 diff={max_h1_diff:.3e} max h2 diff={max_h2_diff:.3e}")

    hard_fail = enc_mismatch > 0 or any(
        margin > MARGIN_BAND for _, _, _, margin, _ in action_disagree)
    if hard_fail:
        print("result: FAIL")
        return 1
    if action_disagree:
        print("result: PASS (with near-tie disagreements inside the "
              "predeclared band, listed above)")
    else:
        print("result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
