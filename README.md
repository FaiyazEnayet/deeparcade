<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="DeepArcade: A playable comparison of Snake agents" width="1200"></p>

<h1 align="center">DeepArcade</h1>

<p align="center">
  <a href="https://github.com/FaiyazEnayet/deeparcade/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/FaiyazEnayet/deeparcade/ci.yml?branch=main&amp;label=Python%20CI&amp;style=flat-square&amp;labelColor=24292f" alt="Python CI status"></a>
  <a href="https://github.com/FaiyazEnayet/deeparcade/actions/workflows/native.yml"><img src="https://img.shields.io/github/actions/workflow/status/FaiyazEnayet/deeparcade/native.yml?branch=main&amp;label=Native%20CI&amp;style=flat-square&amp;labelColor=24292f" alt="Native CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-d4c277?style=flat-square&amp;labelColor=24292f" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;labelColor=24292f" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&amp;labelColor=24292f" alt="C++20">
  <img src="https://img.shields.io/badge/built%20with-PyTorch%20%2B%20browser%20inference-301934?style=flat-square&amp;labelColor=24292f" alt="Built with PyTorch and browser inference">
</p>

<p align="center">
  <a href="https://faiyazenayet.github.io/deeparcade/docs/game.html"><img src="https://img.shields.io/badge/Play-Snake-f4edda?style=for-the-badge&amp;labelColor=0b1410" alt="Play the Snake game"></a>
  <a href="https://faiyazenayet.github.io/deeparcade/docs/dashboard.html"><img src="https://img.shields.io/badge/Explore-Dashboard-9ad7ac?style=for-the-badge&amp;labelColor=0b1410" alt="Explore the interactive dashboard"></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Run-Locally-13251b?style=for-the-badge&amp;labelColor=0b1410" alt="Run it locally"></a>
</p>

DeepArcade is a Snake arcade that pits five very different players against the same seeded games: a random mover, a greedy heuristic, tabular Q-learning trained over 30,000 episodes, deep Q-networks trained for 300,000 steps (a flat MLP, plus plain and residual CNNs that read the board as spatial channels), and MCTS searching from the current position. The learned agents are genuinely trained rather than scripted, and every score in this README comes from recorded evaluation episodes, not showcase runs.

<p align="center"><a href="https://faiyazenayet.github.io/deeparcade/docs/dashboard.html"><img src="docs/demo.gif?v=showcase-20261017b" alt="DeepArcade demo"></a></p>

Pro is a residual convolutional DQN. It reads the board as spatial channels and takes the heading in after pooling, and the browser plays its seed 9 export. Rookie is tabular Q-learning, Grandmaster is MCTS, and the frozen flat MLP stays selectable as Previous Pro.

The CNNs were trained under the MLP's protocol, unchanged, so the network is the only difference: 300,000 environment steps, seeds 7, 8 and 9, the same reward shaping, Adam at 1e-3, gamma 0.95, batch 64, replay 50,000, updates every 8 steps, target sync every 2,000 steps, epsilon from 1.0 down to 0.05 over 120,000 steps, warmup 2,000, the same episode-key training partition, and CPU execution.

## Explore

- Compare every seed's held-out score and paired confidence intervals against the frozen MLP, Rookie and Greedy.
- Inspect the recorded learning curves and the inference-speed tradeoff.
- Play or watch Pro's residual CNN, the frozen MLP, tabular Q-learning or MCTS in the browser.

## Held-out comparison

`runs/da8_comparison.json` records 500 keyed test episodes per policy, paired by episode key, on test keys that never overlap the training partition. They form a separate evaluation namespace from the legacy table below.

| Policy | Seed 7 | Seed 8 | Seed 9 |
| :--- | ---: | ---: | ---: |
| Retrained MLP control | 5.61 | 5.16 | 5.60 |
| Plain CNN | 12.16 | 13.19 | 12.97 |
| Residual CNN | 16.82 | 20.42 | 21.86 |

The frozen MLP scores 5.28, Rookie scores 20.78, and Greedy scores 20.61 in this keyed namespace. The plain encoder has 13,795 parameters and the residual encoder has 32,291.

| Paired difference against frozen MLP | Seed 7 | Seed 8 | Seed 9 |
| :--- | ---: | ---: | ---: |
| Plain CNN | +6.88 | +7.92 | +7.69 |
| Residual CNN | +11.54 | +15.14 | +16.58 |

Every one of those differences has a 95% confidence interval far from zero. Against Rookie, the residual CNN trails by -3.96 (CI -4.83 to -3.09) for seed 7, ties at -0.36 (CI -1.34 to +0.62) for seed 8, and edges ahead at +1.08 (CI +0.07 to +2.09) for seed 9. Against Greedy, seed 9 is +1.25 (CI +0.22 to +2.27). The plain CNN trails Rookie and Greedy by roughly 7.4 to 8.6 points on the same pairing.

Seeing the board as a spatial input, rather than a flat vector, is worth a lot. The residual CNN reaches the tabular and heuristic baselines but does not clearly beat them, and its seeds vary widely, from 16.82 to 21.86, so the dashboard shows all three trained seeds next to the seed 9 policy the browser plays.

## Training and inference

| Final training mean-100 score | Seed 7 | Seed 8 | Seed 9 |
| :--- | ---: | ---: | ---: |
| MLP control | 4.24 | 4.90 | 4.36 |
| Plain CNN | 9.85 | 10.29 | 9.89 |
| Residual CNN | 11.59 | 11.61 | 13.64 |

Inference runs at about 21,600 decisions per second for the MLP, 2,000 for the plain CNN and 1,000 for the residual CNN, so the residual model costs roughly 20 times the decision time of the MLP. The measurements come from `runs/da8_comparison.json`; training curves and final scores for each seed are committed under `runs/`.

## Legacy evaluation

The archived legacy results below come from `runs/eval_legacy.json`. The non-MCTS policies have 500 episodes, while MCTS has 50. This namespace is separate from the keyed test comparison.

| Agent | Mean food score |
| :--- | ---: |
| Random | 0.16 |
| Greedy | 19.89 |
| Rookie, tabular Q-learning | 20.78 |
| Previous Pro, frozen MLP | 5.39 |
| Grandmaster, MCTS | 44.1 |

The current paired-pool record in `runs/eval.json` reports a Grandmaster mean of 46.14 and a median of 47.0 over 50 episodes. That figure and the archived 44.1 come from different protocols, so the two are not interchangeable. The ELO ratings and head-to-head results in that file belong to the evaluated policies and give the CNN no rating, so tournament rounds against the CNN compare scores without updating the player's ELO.

## Dashboard and game

The dashboard shows the matched comparison with its confidence intervals, the recorded training curves, and the legacy records under their own labels. The game embeds the residual CNN seed 9 weights and runs the forward pass locally in the browser, with the frozen MLP still available in the model selector. The animation at the top of this README is a real residual CNN rollout.

## Run locally


```sh
python -m pip install -e ".[dev]"
python -m pip install -r portfolio_ui/requirements.txt
python -m pytest tests -q
python scripts/make_game.py
python scripts/make_dashboard.py
```

The native C++ engine builds and tests separately:

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j
ctest --test-dir native/build --output-on-failure
```

With the pybind11 extension built, `NARCADE_NATIVE=1` switches the Python package to the native backend.

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
  N0["Seeded Snake environment"] --> N6["Native C++ engine (native/)"]
```

## Native C++ engine

DeepArcade includes a native C++20 engine under `native/` as an optional backend. It implements the Snake environment, a CPython-compatible RNG, the tabular policies and Q-learning agent, MCTS, the evaluation protocol, the training protocol, move-level trace instrumentation, DQN and CNN inference over the exported JSON weights, and a tournament runner that plays repeated round-robin rounds of paired episodes between all five players and computes Elo standings. Python remains the reference and the default; the native backend is enabled with `NARCADE_NATIVE=1` once the pybind11 extension is built.

Parity is enforced, not assumed. Golden fixtures generated from the Python reference (`native/golden/`, regenerable with `tools/native_reference.py`) pin the behavior, and the C++ runners replay frozen corpora against them: 111,032 Snake transitions, 36,129 policy states, 1,039 MCTS searches with bit-exact root values, 404 further MCTS searches on maze-mode wall layouts (the browser game's walled arena, 977 root children, also bit-exact), 200 evaluation episodes, the full 30,000-episode training protocol, and 13,824 DQN states with complete action agreement. Two boundaries are explicit: DQN training stays in Python (the native side is inference only), and the native evaluator covers the non-neural players, which its report states in a scope field. The `regen_check` CTest regenerates the fixtures from the Python reference and byte-compares them, and `meta_check` (`native/tests/meta_check.cmake`) verifies each golden file's SHA-256 against `native/golden/meta.json`.

A separate Native workflow in GitHub Actions builds the engine and runs the CTest suite on every push and pull request; `eval_fuzz` is excluded there for runtime and remains a local gate: changes to the native evaluation code require a documented local `eval_fuzz` run. A frozen subset of the evaluation protocol (seeds 5000 to 5002 for all four players, `native/golden/eval_regression.json`, regenerated with `tools/native_reference.py --eval-regression`) is replayed by the `eval_regression` CTest, so hosted CI still guards evaluation parity on every run.

Build and test:

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j
ctest --test-dir native/build --output-on-failure
```

The tournament runner plays round-robin rounds between all five players (the DQN player included, through the native inference module) with Elo carried across rounds, and writes a JSON standings report:

```sh
native/build/narcade_tournament --out runs/tournament.json
```

## Code and data

`src/narcade/` contains the Snake environment and agents. `native/` contains the C++ engine: environment, RNG, policies, MCTS, evaluator, trainer and DQN and CNN inference, with golden corpora under `native/golden/`. `runs/eval.json` stores the evaluation. `runs/artifact_manifest.json` pins the SHA-256 hash and production metadata (architecture, seed, steps, evaluation protocol) of the qtable, policy, DQN weights, evaluation and diagnosis artifacts, and `scripts/make_game.py` verifies the artifacts against it before injecting policy data. `runs/model_cards.json` collects one compact model card per evaluated agent: its kind (baseline, heuristic, learned or search), the training record, the checkpoint and export hashes cross-checked against the hashes the evaluator recorded, the evaluation figures and a measured inference cost per decision; `scripts/model_cards.py` builds and verifies the cards, and its smoke command trains tiny Q-learning and DQN models and rebuilds verifiable cards end to end. `web/game.html` is the browser template.


## Limitations

- Recorded evaluation uses the classic board; other browser modes are distribution shifts.
- Browser MCTS adapts its search budget, so browser scores are not the fixed evaluation protocol.
- Human ELO is a browser-local single-score estimate; it is separate from the recorded paired-policy evaluation. Tournament standings retain the Rookie → Pro → Grandmaster round sequence, with restart, free play and main-menu choices at completion.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
