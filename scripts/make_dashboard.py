"""Build docs/dashboard.html: self-contained Plotly dashboard (no server needed).

Panels:
  1. Learning curves — DQN + tabular Q mean score per 100 episodes
  2. ELO history — Q-learner ELO over training checkpoints
  3. Final ELO ratings — all policies
  4. Score distributions — all policies (eval episodes)
  5. Head-to-head win rates — heatmap
  6. Move-level analysis — action mix, death causes, food efficiency
"""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUNS = os.path.join(ROOT, "runs")

COLORS = {"random": "#64748b", "greedy": "#38bdf8", "rookie": "#4ade80",
          "pro": "#a78bfa", "grandmaster": "#fbbf24"}
NAMES = {"random": "Random", "greedy": "Greedy heuristic", "rookie": "Rookie (tabular Q)",
         "pro": "Pro (DQN)", "grandmaster": "Grandmaster (MCTS)"}
DARK = dict(paper_bgcolor="#0b0e1a", plot_bgcolor="#0d1226",
            font=dict(color="#e2e8f0"))


def load(name):
    with open(os.path.join(RUNS, name)) as f:
        return json.load(f)


def main():
    curve = load("curve.json")          # tabular Q
    dqn_curve = load("dqn_curve.json")  # DQN
    elo_hist = load("elo.json")         # Q-learner ELO checkpoints
    ev = load("eval.json")
    moves = load("moves.json")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Learning curves: mean score per 100 episodes",
                        "ELO rating over training (Rookie vs baselines)",
                        "Final ELO ratings (all policies)",
                        "Score distributions (evaluation episodes)",
                        "Head-to-head win rates",
                        "Move-level analysis"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.10, horizontal_spacing=0.10)

    # 1. learning curves
    x_q = [(i + 1) * 100 for i in range(len(curve["mean_score"]))]
    fig.add_trace(go.Scatter(x=x_q, y=curve["mean_score"], name="Rookie (tabular Q)",
                             line=dict(color=COLORS["rookie"], width=2)), row=1, col=1)
    x_d = [(i + 1) * 100 for i in range(len(dqn_curve["mean_score"]))]
    fig.add_trace(go.Scatter(x=x_d, y=dqn_curve["mean_score"], name="Pro (DQN)",
                             line=dict(color=COLORS["pro"], width=2)), row=1, col=1)
    fig.update_xaxes(title_text="episodes", row=1, col=1)
    fig.update_yaxes(title_text="mean score", row=1, col=1)

    # 2. ELO history
    fig.add_trace(go.Scatter(x=[p["episode"] for p in elo_hist],
                             y=[p["elo"] for p in elo_hist],
                             name="Rookie ELO", line=dict(color=COLORS["rookie"], width=2),
                             fill="tozeroy", fillcolor="rgba(74,222,128,0.12)"),
                  row=1, col=2)
    fig.add_hline(y=1200, line_dash="dash", line_color="#64748b",
                  annotation_text="baseline 1200", row=1, col=2)
    fig.update_xaxes(title_text="episodes", row=1, col=2)
    fig.update_yaxes(title_text="ELO", row=1, col=2)

    # 3. final ELO bars
    order = ["random", "greedy", "rookie", "pro", "grandmaster"]
    fig.add_trace(go.Bar(x=[NAMES[k] for k in order],
                         y=[ev["elo"][k] for k in order],
                         marker_color=[COLORS[k] for k in order],
                         name="ELO", showlegend=False,
                         text=[ev["elo"][k] for k in order], textposition="outside"),
                  row=2, col=1)
    fig.update_yaxes(title_text="ELO", row=2, col=1)

    # 4. score distributions
    for k in order:
        fig.add_trace(go.Histogram(x=ev["scores"][k], name=NAMES[k],
                                   marker_color=COLORS[k], opacity=0.65,
                                   nbinsx=20), row=2, col=2)
    fig.update_xaxes(title_text="score", row=2, col=2)
    fig.update_yaxes(title_text="episodes", row=2, col=2)

    # 5. head-to-head heatmap
    z = [[(ev["head_to_head"][a][b]["win_rate"] if ev["head_to_head"][a][b] else 0.5)
          for b in order] for a in order]
    fig.add_trace(go.Heatmap(z=z, x=[NAMES[k] for k in order],
                             y=[NAMES[k] for k in order],
                             colorscale="Greens", zmin=0, zmax=1,
                             text=[[f"{v:.0%}" for v in row] for row in z],
                             texttemplate="%{text}", showscale=False),
                  row=3, col=1)

    # 6. move-level: action mix (stacked) + efficiency text
    pols = [k for k in order if k in moves]
    fig.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                         y=[moves[k]["actions"][0] for k in pols],
                         name="straight", marker_color="#4ade80"), row=3, col=2)
    fig.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                         y=[moves[k]["actions"][1] for k in pols],
                         name="left", marker_color="#38bdf8"), row=3, col=2)
    fig.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                         y=[moves[k]["actions"][2] for k in pols],
                         name="right", marker_color="#fbbf24"), row=3, col=2)
    fig.update_yaxes(title_text="action share", row=3, col=2)

    fig.update_layout(height=1150, title="deeparcade — training the machine",
                      barmode="stack", **DARK,
                      legend=dict(orientation="h", y=1.02))
    # readable subplot titles on dark bg
    for ann in fig.layout.annotations:
        ann.font.color = "#e2e8f0"

    # summary strip with measured numbers
    s = ev["summary"]
    note = (f"Measured: Pro (DQN) {s['pro']['mean']}±{s['pro']['std']} vs "
            f"Greedy {s['greedy']['mean']}±{s['greedy']['std']} vs "
            f"Random {s['random']['mean']}±{s['random']['std']} "
            f"over {ev['n_eval']} seeded episodes each. "
            f"Grandmaster (MCTS, {ev['mcts_sims']} sims) {s['grandmaster']['mean']} "
            f"over {ev['mcts_n']} episodes.")
    fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=-0.04,
                       showarrow=False, font=dict(size=11, color="#94a3b8"),
                       align="left")

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "dashboard.html")
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"dashboard -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
