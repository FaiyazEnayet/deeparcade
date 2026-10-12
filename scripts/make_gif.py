"""Render docs/demo.gif: the trained Pro (DQN) playing Snake.

Matplotlib animation from a real greedy rollout of the trained network
(seed fixed, so the footage is reproducible). Dark arcade styling with a
motion trail.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from narcade.snake import SnakeEnv
from scripts.train_dqn import DQN, encode_dqn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUNS = os.path.join(ROOT, "runs")
DOCS = os.path.join(ROOT, "docs")


def main(seed=2026, max_frames=240):
    net = DQN()
    net.load_state_dict(torch.load(os.path.join(RUNS, "dqn.pt"),
                                  map_location="cpu"))
    net.eval()

    env = SnakeEnv(seed=seed)
    env.reset()
    frames = []  # (body, food, score)
    with torch.no_grad():
        for _ in range(max_frames):
            q = net(torch.from_numpy(encode_dqn(env)).unsqueeze(0))
            a = int(q.argmax(1).item())
            _, _, done, _ = env.step(a)
            frames.append((list(env.body), env.food, env.score))
            if done:
                break

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.patch.set_facecolor("#0b0e1a")
    ax.set_facecolor("#0d1226")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 11.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#243056")

    n_trail = 6
    trail = [Rectangle((0, 0), 0.92, 0.92, fc="#166534", ec="none", alpha=0.0)
             for _ in range(n_trail)]
    for r in trail:
        ax.add_patch(r)
    head = Rectangle((0, 0), 0.92, 0.92, fc="#4ade80", ec="#bbf7d0", lw=1.5)
    ax.add_patch(head)
    food_dot, = ax.plot([], [], "o", ms=14, mfc="#4ade80", mec="#bbf7d0", mew=1.5)
    title = ax.set_title("", color="#e2e8f0", fontsize=13, pad=10)

    def draw(i):
        body, food, score = frames[i]
        # trail = last segments with fading alpha
        segs = body[-1 - n_trail:-1]
        for j, rect in enumerate(trail):
            if j < len(segs):
                r, c = segs[j]
                rect.set_xy((c - 0.46, r - 0.46))
                rect.set_alpha(0.12 + 0.5 * (j + 1) / max(1, len(segs)))
            else:
                rect.set_alpha(0.0)
        hr, hc = body[-1]
        head.set_xy((hc - 0.46, hr - 0.46))
        food_dot.set_data([food[1]], [food[0]])
        title.set_text(f"deeparcade — Pro (DQN) playing  ·  score {score}")
        return trail + [head, food_dot, title]

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False, interval=90)
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "demo.gif")
    anim.save(out, writer="pillow", fps=11)
    plt.close(fig)
    print(f"gif -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames, "
          f"final score {frames[-1][2]})")


if __name__ == "__main__":
    main(seed=2036)
