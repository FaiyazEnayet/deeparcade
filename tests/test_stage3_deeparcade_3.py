"""Stage 3 deeparcade DA3: unify Python and browser game contracts.

Acceptance (handoffs section 223):
- boundary, tail-vacating, food, timeout and terminal cases agree exactly
  between the Python environment (src/narcade/snake.py) and the browser
  core (web/game.html);
- DQN logits from the real browser forward pass agree with the PyTorch
  policy within the declared float64 tolerance of 1e-6 on the frozen
  parity observations (tests/fixtures/dqn_parity_observations.json);
- the fixture check detects a deliberate action-order mutation.

The transition fixtures (tests/fixtures/stage3_deeparcade_3_fixtures.json)
are canonical single-step cases: each one states the full pre-step state
(body, heading, food, score, steps), one action, the explicit post-eat
food (food_after, null when the board fills), and the expected post-step
record. No case consumes an RNG draw at replay time, so the fixtures are
seed-independent: both replay drivers inject the start state and place
exactly the stated food, and must reproduce every expected record.

This goes beyond P0-07, whose fixtures are multi-step play-throughs with
a scripted food queue and which never covered the tail-vacating move
(stepping into the cell the tail just left, legal when not eating and
lethal when eating into it). The DQN leg also goes beyond P0-03, which
compared PyTorch logits only against browser_export_logits, a Python
mirror of the browser forward pass; here the logits come from the real
browser code (web/game.html dqnLogits) run in a headless Node engine.

The browser side is exercised by tests/stage3_deeparcade_3_harness.js,
a nonvisual Node harness in the P0-07 pattern. It loads the real game
script from web/game.html, injects the committed DQN weights exactly as
scripts/make_game.py does at build time, replays the canonical cases,
reconstructs the frozen DQN observation states from their vectors and
runs the real dqnLogits forward pass on them. With
--mutate-action-order it swaps the LEFT and RIGHT turns inside the
loaded game script, which the fixtures must catch.
"""

import json
import os
import subprocess
import sys
from collections import deque

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "web", "game.html")
HARNESS = os.path.join(ROOT, "tests", "stage3_deeparcade_3_harness.js")
FIXTURES = os.path.join(
    ROOT, "tests", "fixtures", "stage3_deeparcade_3_fixtures.json")
DQN_FIXTURE = os.path.join(
    ROOT, "tests", "fixtures", "dqn_parity_observations.json")
WEIGHTS = os.path.join(ROOT, "runs", "dqn_weights.json")

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from narcade.snake import SnakeEnv  # noqa: E402

# Declared float64 parity tolerance for DQN logits (handoffs section 223,
# checkpoint schema in scripts/train_dqn.py).
LOGIT_TOLERANCE = 1e-6

CASE_NAMES = [
    "boundary_north",
    "boundary_east",
    "boundary_south",
    "boundary_west",
    "boundary_corner_turn",
    "tail_vacating_straight",
    "tail_vacating_turn",
    "tail_vacating_eat_is_death",
    "food_eat_straight",
    "food_eat_turn",
    "food_closer",
    "food_farther",
    "timeout_4x4",
    "terminal_win_3x3",
]


def _fixtures():
    with open(FIXTURES, encoding="utf-8") as f:
        return json.load(f)


def _make_env(case):
    """SnakeEnv with the fixture start state injected and food placed
    exactly as stated: the initial food first, then food_after on an eat.
    No RNG draw is consumed, so the replay is seed-independent."""
    env = SnakeEnv(case["width"], case["height"], seed=0,
                   max_steps=case["max_steps"])
    queue = [tuple(case["food"]), case["food_after"]]
    queue = [tuple(c) if c is not None else None for c in queue]

    def placer():
        cell = queue.pop(0) if queue else None
        if cell is not None:
            # Food may sit on the tail cell only: that is exactly the
            # tail_vacating_eat_is_death case, which pins that eating
            # into the tail is lethal because the tail does not vacate.
            assert cell not in set(list(env.body)[1:]), \
                "stated food on the body"
        env.food = cell

    env._place_food = placer
    env.body = deque(tuple(c) for c in case["start"]["body"])
    env.heading = case["start"]["heading"]
    env.score = case["start"]["score"]
    env.steps = case["start"]["steps"]
    env.done = False
    placer()  # initial stated food
    return env


def _snap(env, reward, reason, state):
    return {
        "body": [list(c) for c in env.body],
        "heading": env.heading,
        "food": list(env.food) if env.food is not None else None,
        "score": env.score,
        "steps": env.steps,
        "reward": reward,
        "done": env.done,
        "reason": reason,
        "state": state,
    }


def _replay_python(case):
    env = _make_env(case)
    records = [_snap(env, None, None, env.encode_state())]
    state, reward, done, info = env.step(case["action"])
    records.append(_snap(env, reward, info.get("reason"), state))
    return env, records


def test_stage3_deeparcade_3_fixture_schema():
    doc = _fixtures()
    assert doc["schema"] == 1
    assert [c["name"] for c in doc["cases"]] == CASE_NAMES
    # Seed-independence: no case carries a seed (the about text may
    # describe the fixtures as seed-independent), and every case states
    # its post-eat food explicitly (null only when the board fills).
    for case in doc["cases"]:
        assert "seed" not in case, case["name"]
    for case in doc["cases"]:
        assert "food_after" in case, case["name"]
        assert set(case) == {
            "name", "category", "width", "height", "start", "food",
            "food_after", "max_steps", "action", "expected",
        }, case["name"]
        assert set(case["start"]) == {
            "body", "heading", "score", "steps",
        }, case["name"]
        assert len(case["expected"]) == 2
    categories = [c["category"] for c in doc["cases"]]
    assert categories.count("boundary") == 5
    assert categories.count("tail-vacating") == 3
    assert categories.count("food") == 4
    assert categories.count("timeout") == 1
    assert categories.count("terminal") == 1


def test_stage3_deeparcade_3_python_replay_matches_fixtures():
    doc = _fixtures()
    for case in doc["cases"]:
        _, records = _replay_python(case)
        assert records == case["expected"], case["name"]


def test_stage3_deeparcade_3_python_terminal_outcomes():
    doc = _fixtures()
    by_name = {c["name"]: c for c in doc["cases"]}
    expected = {
        "boundary_north": "death",
        "boundary_east": "death",
        "boundary_south": "death",
        "boundary_west": "death",
        "boundary_corner_turn": "death",
        "tail_vacating_eat_is_death": "death",
        "timeout_4x4": "timeout",
        "terminal_win_3x3": "win",
    }
    for name, reason in expected.items():
        env, records = _replay_python(by_name[name])
        assert records[-1]["reason"] == reason, name
        assert env.done
        assert env.encode_state() == records[-1]["state"]
        with pytest.raises(RuntimeError):
            env.step(0)


def test_stage3_deeparcade_3_tail_vacating_asymmetry():
    """The tail cell is free exactly when the snake does not eat: the
    same geometry is a legal move with food elsewhere and a death with
    the food on the tail cell, because the tail does not vacate on an
    eat. This is the case P0-07 never pinned."""
    doc = _fixtures()
    by_name = {c["name"]: c for c in doc["cases"]}
    legal = by_name["tail_vacating_straight"]
    _, records = _replay_python(legal)
    assert records[-1]["done"] is False
    assert records[-1]["body"] == [[2, 2], [1, 2], [1, 1], [2, 1]]
    lethal = by_name["tail_vacating_eat_is_death"]
    _, records = _replay_python(lethal)
    assert records[-1]["done"] is True
    assert records[-1]["reason"] == "death"
    assert records[-1]["reward"] == -10.0


def test_stage3_deeparcade_3_harness_passes():
    proc = subprocess.run(
        ["node", HARNESS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "harness: all checks passed" in proc.stdout


def test_stage3_deeparcade_3_dqn_logits_js_match_pytorch():
    """Real browser forward pass (web/game.html dqnLogits in Node) vs
    the PyTorch policy in float64 on the 36 frozen observations."""
    torch = pytest.importorskip("torch")
    import hashlib

    import numpy as np

    from scripts.train_dqn import browser_export_logits, load_checkpoint

    with open(DQN_FIXTURE, encoding="utf-8") as f:
        fx = json.load(f)
    rows = fx["observations"]
    assert len(rows) == 36
    assert sum(1 for o in rows if o["terminal"]) == 12
    assert sum(1 for o in rows if not o["terminal"]) == 24
    with open(WEIGHTS, "rb") as f:
        assert fx["weights_artifact_sha256"] == hashlib.sha256(f.read()).hexdigest()

    proc = subprocess.run(
        ["node", HARNESS, "--print-dqn-logits"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = [ln for ln in proc.stdout.splitlines()
            if ln.startswith("DA3_DQN_LOGITS ")]
    assert len(line) == 1
    js_logits = np.array(json.loads(line[0][len("DA3_DQN_LOGITS "):]),
                         dtype=np.float64)
    assert js_logits.shape == (36, 3)
    assert np.isfinite(js_logits).all()

    net, _ = load_checkpoint(WEIGHTS)
    obs = np.array([o["vector"] for o in rows], dtype=np.float64)
    net64 = net.double().eval()
    with torch.no_grad():
        torch_logits = net64(torch.tensor(obs, dtype=torch.float64)).numpy()
    diff = np.abs(torch_logits - js_logits).max()
    assert diff <= LOGIT_TOLERANCE, (
        f"max abs logit diff {diff:.3e} exceeds {LOGIT_TOLERANCE:.0e} "
        "(float64 regime)"
    )

    # Exact greedy action equality on every non-tied case (top-2 gap in
    # the browser logits larger than the tolerance).
    order = np.sort(js_logits, axis=1)
    gap = order[:, 2] - order[:, 1]
    nontied = gap > LOGIT_TOLERANCE
    assert nontied.sum() >= 20
    assert (torch_logits.argmax(1)[nontied]
            == js_logits.argmax(1)[nontied]).all()

    # The Python mirror of the browser forward pass agrees with the real
    # browser pass to the same tolerance, so the mirror stays a faithful
    # stand-in where Node is unavailable.
    with open(WEIGHTS, encoding="utf-8") as f:
        doc = json.load(f)
    mirror = browser_export_logits(doc, obs)
    assert np.abs(mirror - js_logits).max() <= LOGIT_TOLERANCE


def test_stage3_deeparcade_3_mutation_action_order_detected():
    """A deliberate LEFT/RIGHT swap inside the loaded game script must
    fail the fixture check: the turning cases catch it, so a drifted
    action convention cannot ship silently."""
    proc = subprocess.run(
        ["node", HARNESS, "--mutate-action-order"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "check(s) failed" in proc.stdout
    failed = [ln for ln in proc.stdout.splitlines()
              if ln.startswith("FAIL - case ")]
    assert len(failed) >= 1
    assert any("tail_vacating_turn" in ln or "food_eat_turn" in ln
               or "boundary_corner_turn" in ln for ln in failed)
    # Straight-only cases still pass under the swap, so the failure is
    # the fixtures catching the mutation, not a harness crash.
    assert "case boundary_north matches the Python transitions" in proc.stdout


if __name__ == "__main__":
    sys.exit(0)
