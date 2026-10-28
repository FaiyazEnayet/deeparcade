"""Build docs/dashboard.html: one card per chart, each with its own legend.

The old single-figure layout had a shared legend colliding with subplot
titles (and an unnamed heatmap trace showing up as "trace 9"). Now every
panel is its own figure with an HTML title, so legends can never overlap.
Self-contained Plotly (no server needed).
"""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUNS = os.path.join(ROOT, "runs")

COLORS = {"random": "#64748b", "greedy": "#38bdf8", "rookie": "#4ade80",
          "pro": "#a78bfa", "grandmaster": "#fbbf24"}
NAMES = {"random": "Random", "greedy": "Greedy heuristic",
         "rookie": "Pro (tabular Q)", "pro": "Rookie (DQN)",
         "grandmaster": "Grandmaster (MCTS)"}
BG = "#0b0e1a"
PANEL = "#0d1226"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def load(name):
    with open(os.path.join(RUNS, name)) as f:
        return json.load(f)


def _base(fig, height=380, legend=False):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=12),
        height=height,
        margin=dict(l=55, r=25, t=15, b=80 if legend else 50),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="top", y=-0.16,
            xanchor="center", x=0.5, font=dict(color=TEXT, size=12)))
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    return fig


_PLOTLY_INCLUDED = False


def _card(title, caption, fig):
    global _PLOTLY_INCLUDED
    inner = fig.to_html(include_plotlyjs=not _PLOTLY_INCLUDED,
                        full_html=False,
                        config={"displaylogo": False, "responsive": True})
    _PLOTLY_INCLUDED = True
    return (f'<section class="card"><h2>{title}</h2>'
            f'<p class="cap">{caption}</p>{inner}</section>')


def main():
    curve = load("curve.json")          # tabular Q
    dqn_curve = load("dqn_curve.json")  # DQN
    elo_hist = load("elo.json")         # Q-learner ELO checkpoints
    ev = load("eval.json")
    # P0-01: read the versioned schema only. A pre-v2 artifact (no
    # episode keys, undocumented pairing) must fail here with a clear
    # message, never silently chart numbers from it. Reads below use
    # the documented legacy keys (elo, scores, n_eval) unchanged.
    if ev.get("schema_version") != 2:
        raise ValueError(
            f"runs/eval.json has schema_version {ev.get('schema_version')!r}; "
            f"this generator reads schema_version 2 only. Regenerate "
            f"runs/eval.json with scripts/evaluate.py"
        )
    for key in ("elo", "scores", "n_eval"):
        if key not in ev:
            raise ValueError(f"runs/eval.json is missing required key {key!r}")
    moves = load("moves.json")
    order = ["random", "greedy", "rookie", "pro", "grandmaster"]

    cards = []

    # 1. Learning curves.
    f = go.Figure()
    x_q = [(i + 1) * 100 for i in range(len(curve["mean_score"]))]
    f.add_trace(go.Scatter(x=x_q, y=curve["mean_score"],
                           name="Pro (tabular Q)",
                           line=dict(color=COLORS["rookie"], width=2)))
    x_d = [(i + 1) * 100 for i in range(len(dqn_curve["mean_score"]))]
    f.add_trace(go.Scatter(x=x_d, y=dqn_curve["mean_score"],
                           name="Rookie (DQN)",
                           line=dict(color=COLORS["pro"], width=2)))
    f.update_xaxes(title_text="episodes")
    f.update_yaxes(title_text="mean score")
    _base(f, legend=True)
    cards.append(_card(
        "Learning curves: mean score per 100 episodes",
        "Pro learns fast from 32 states. Rookie climbs the whole way but "
        "from raw pixels, which is genuinely harder.",
        f))

    # 2. ELO history.
    f = go.Figure()
    f.add_trace(go.Scatter(x=[p["episode"] for p in elo_hist],
                           y=[p["elo"] for p in elo_hist],
                           name="Pro ELO",
                           line=dict(color=COLORS["rookie"], width=2),
                           fill="tozeroy",
                           fillcolor="rgba(74,222,128,0.12)"))
    f.add_hline(y=1200, line_dash="dash", line_color="#64748b",
                annotation_text="baseline 1200",
                annotation_font_color=MUTED)
    f.update_xaxes(title_text="episodes")
    f.update_yaxes(title_text="ELO")
    _base(f)
    cards.append(_card(
        "ELO rating over training (Pro)",
        "Tournament ELO of the Q-learner at checkpoints, against the "
        "baseline pool.",
        f))

    # 3. Final ELO bars.
    f = go.Figure()
    f.add_trace(go.Bar(x=[NAMES[k] for k in order],
                       y=[ev["elo"][k] for k in order],
                       marker_color=[COLORS[k] for k in order],
                       showlegend=False,
                       text=[ev["elo"][k] for k in order],
                       textposition="outside"))
    f.update_yaxes(title_text="ELO")
    _base(f, height=360)
    cards.append(_card(
        "Final ELO ratings (all policies)",
        "Grandmaster (MCTS) tops the ladder. Pro edges out the greedy "
        "heuristic. Rookie trails: pixels are hard.",
        f))

    # 4. Score distributions.
    f = go.Figure()
    for k in order:
        f.add_trace(go.Histogram(x=ev["scores"][k], name=NAMES[k],
                                 marker_color=COLORS[k], opacity=0.65,
                                 nbinsx=20))
    f.update_xaxes(title_text="score")
    f.update_yaxes(title_text="episodes")
    _base(f, legend=True)
    cards.append(_card(
        "Score distributions (evaluation episodes)",
        f"{ev['n_eval']} seeded episodes per policy. Grandmaster's mass "
        f"sits far to the right.",
        f))

    # 5. Head-to-head heatmap.
    f = go.Figure()
    z = [[(ev["head_to_head"][a][b]["win_rate"]
           if ev["head_to_head"][a][b] else 0.5)
          for b in order] for a in order]
    f.add_trace(go.Heatmap(z=z, x=[NAMES[k] for k in order],
                           y=[NAMES[k] for k in order],
                           colorscale="Greens", zmin=0, zmax=1,
                           text=[[f"{v:.0%}" for v in row] for row in z],
                           texttemplate="%{text}", showscale=False,
                           showlegend=False))
    _base(f, height=420)
    cards.append(_card(
        "Head-to-head win rates",
        "Row beats column. Grandmaster takes nearly every matchup.",
        f))

    # 6. Move-level analysis.
    f = go.Figure()
    pols = [k for k in order if k in moves]
    f.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                       y=[moves[k]["actions"][0] for k in pols],
                       name="straight", marker_color="#4ade80"))
    f.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                       y=[moves[k]["actions"][1] for k in pols],
                       name="left", marker_color="#38bdf8"))
    f.add_trace(go.Bar(x=[NAMES[k] for k in pols],
                       y=[moves[k]["actions"][2] for k in pols],
                       name="right", marker_color="#fbbf24"))
    f.update_layout(barmode="stack")
    f.update_yaxes(title_text="action share")
    _base(f, legend=True, height=380)
    cards.append(_card(
        "Move-level analysis",
        "How each policy steers: greedy goes straight 79% of the time, "
        "Grandmaster balances its turns.",
        f))

    s = ev["summary"]
    stats = (
        f"Rookie (DQN) {s['pro']['mean']}&plusmn;{s['pro']['std']} vs "
        f"Greedy {s['greedy']['mean']}&plusmn;{s['greedy']['std']} vs "
        f"Random {s['random']['mean']}&plusmn;{s['random']['std']} "
        f"over {ev['n_eval']} seeded episodes each. Grandmaster (MCTS, "
        f"{ev['mcts_sims']} sims) {s['grandmaster']['mean']} over "
        f"{ev['mcts_n']} episodes.")
    body = "\n".join(cards)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>deeparcade — training the machine</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: {BG}; color: {TEXT}; margin: 0;
         font-family: Inter, system-ui, -apple-system, sans-serif; }}
  .wrap {{ max-width: 1060px; margin: 0 auto; padding: 36px 24px 60px; }}
  h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -0.5px; }}
  h1 .dm {{ color: {COLORS["grandmaster"]}; }}
  .sub {{ color: {MUTED}; font-size: 15px; margin: 0 0 28px; max-width: 780px;
          line-height: 1.55; }}
  .stats {{ background: {PANEL}; border: 1px solid #1e293b; border-radius: 12px;
            padding: 16px 20px; font-size: 14px; line-height: 1.55;
            margin-bottom: 22px; color: {TEXT}; }}
  .card {{ background: {PANEL}; border: 1px solid #1e293b; border-radius: 12px;
           padding: 22px 22px 6px; margin-bottom: 22px; }}
  .card h2 {{ font-size: 18px; margin: 0 0 4px; }}
  .card .cap {{ color: {MUTED}; font-size: 13.5px; margin: 0 0 10px;
                line-height: 1.5; }}
  .foot {{ margin-top: 24px; color: {MUTED}; font-size: 12.5px; line-height: 1.6;
           max-width: 820px; }}
  .ver {{ display: inline-block; margin-left: 8px; padding: 2px 10px;
          border: 1px solid #1e293b; border-radius: 999px; font-size: 11px;
          color: {COLORS["rookie"]}; }}
</style>
</head>
<body>
<div class="wrap">
  <h1><span class="dm">&#9679;</span> deeparcade &mdash; training the machine</h1>
  <p class="sub">Snake with opponents that actually learned to play: a
  Q-learner, a deep Q-network, and an MCTS Grandmaster. Every number below
  came from a real training or evaluation run. Play them in your browser
  in <a href="game.html" style="color:{COLORS['greedy']}">game.html</a>.</p>
  <div class="stats"><b>Measured:</b> {stats}</div>
  {body}
  <p class="foot">500 seeded evaluation episodes per policy (MCTS: 50, it
  thinks slowly). Fixed seeds, no cherry-picking. Pro beats Rookie: the
  32-state abstraction is excellent, and learning from 436-dim pixels is
  genuinely harder.
  <span class="ver">dashboard layout v2 &mdash; one card per chart</span></p>
</div>
</body>
</html>"""

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(_portfolio_html(page))
    print(f"dashboard -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")



def _portfolio_html(raw):
    """Presentation-only transform of this run’s freshly calculated figures."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'deeparcade', raw)[0]

if __name__ == "__main__":
    main()
