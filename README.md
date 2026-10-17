<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="DeepArcade: A playable comparison of Snake agents" width="1200"></p>

[![Tests](https://github.com/FaiyazEnayet/deeparcade/actions/workflows/ci.yml/badge.svg)](https://github.com/FaiyazEnayet/deeparcade/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-007EC6)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB) ![Built with PyTorch + browser inference](https://img.shields.io/badge/built%20with-PyTorch%20%2B%20browser%20inference-555555) [![Interactive demo](https://img.shields.io/badge/demo-interactive-4C9D39)](https://faiyazenayet.github.io/deeparcade/docs/dashboard.html)

<p align="center"><a href="https://faiyazenayet.github.io/deeparcade/docs/dashboard.html"><b>Open interactive demo</b></a> · <a href="#method">Method</a> · <a href="#run-locally">Run locally</a></p>

# DeepArcade

DeepArcade compares random, heuristic, tabular Q-learning, DQN and MCTS Snake agents. The dashboard pairs evaluation results with the actual browser game, so the behavior can be inspected as well as summarized.


[![DeepArcade demo](docs/demo.gif?v=showcase-20261017b)](https://faiyazenayet.github.io/deeparcade/docs/dashboard.html)


Playing against the trained policies makes their evaluation easier to interpret. A strong average score can coexist with brittle turns, long loops or inconsistent behavior across seeds, which a single showcase run cannot establish. The agent comparison therefore keeps score distributions, training curves and evaluation sample counts alongside the playable experience.

## Explore

- Compare agent score distributions, learning behavior and head-to-head outcomes.
- Load the genuine game inside the dashboard or open the standalone HTML game.
- Use keyboard controls, then restart directly over the board after a run ends.

## Method

Agents are evaluated on seeded classic-board games. Learning curves, score distributions, pairwise outcomes and strategy statistics retain their original evaluation conditions. The browser game uses exported policies and a JavaScript forward pass.

## Evidence at a glance

| Scope | Recorded result or contract |
| :--- | :--- |
| Evaluation games | 500 per policy, except MCTS with 50 |
| Mean food score | Random 0.16; greedy 18.89; tabular Q 20.78 |
| Same evaluation | DQN 6.39; MCTS 44.1 |
| Interpretation | MCTS has a smaller sample; the DQN does not outperform the simple baselines |

## Updating this project

Dashboard source language: **Python and browser JavaScript**. Generator: `scripts/make_dashboard.py and scripts/make_game.py`. To keep the interactive page, figures, data animation, and README consistent, follow [the coding-agent update guide](AGENT-DASHBOARD-GUIDE.md). The local interactive output is `docs/dashboard.html`.

## Run locally


```sh
python -m pip install -e ".[dev]"
python -m pip install -r portfolio_ui/requirements.txt
python -m pytest tests -q
python scripts/make_game.py
python scripts/make_dashboard.py
```

Open [the standalone game](https://faiyazenayet.github.io/deeparcade/docs/game.html) to play without the dashboard.

## Technical design

The agent roster deliberately mixes simple baselines, learned policies and search. Tabular Q-learning and DQN learn policies; MCTS searches from the current state. Exported policy data lets the browser execute the learned behavior without a model server. The same game remains available within the dashboard and as a standalone file.

```mermaid
flowchart LR
  N0["Seeded Snake environment"] --> N1["Q learning"]
  N0["Seeded Snake environment"] --> N2["DQN training"]
  N0["Seeded Snake environment"] --> N3["MCTS search"]
  N1["Q learning"] --> N4["Agent evaluation"]
  N2["DQN training"] --> N4["Agent evaluation"]
  N3["MCTS search"] --> N4["Agent evaluation"]
  N4["Agent evaluation"] --> N5["Browser policies and dashboard"]
```

## Evaluation protocol

Recorded evaluation uses seeded classic-board games, with 500 games for most policies and 50 for MCTS. Scores, distributions, learning curves and pairwise outcomes answer different questions. Browser maze/blitz modes and adaptive MCTS budgets are not equivalent to the fixed recorded evaluation.

![Agent evaluation, with unequal sample sizes](docs/results.png)

The DQN underperforms the greedy and tabular baselines in the committed run. The MCTS mean is high but comes from a smaller sample. A convincing comparison requires the stated game conditions, seeds and sample counts rather than agent names or apparent sophistication.

## Code and data

`src/narcade/` contains the Snake environment and agents. `runs/eval.json` stores the evaluation. `web/game.html` is the browser template; `scripts/make_game.py` injects policy data.


## Limitations

- Recorded evaluation uses the classic board; other browser modes are distribution shifts.
- Browser MCTS adapts its search budget, so browser scores are not the fixed evaluation protocol.
- Tournament state handling needs a correction before its ELO changes are treated as reliable. Replay UI and direction buttons were intentionally removed from this edition.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
