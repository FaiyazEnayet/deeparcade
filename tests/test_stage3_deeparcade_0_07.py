"""Stage 3 deeparcade P0-07: browser core parity and game preservation.

Acceptance (handoffs section 008):
- a full-board win never crashes the browser game and ends it as a win;
- Python and browser transitions agree step for step under identical
  actions and controlled food placement (body, head, heading, food,
  score, terminal flags);
- action routing works, classic human play remains responsive, and
  maze/blitz behavior is separately smoke-tested;
- no source/export overwrites outside the owner's intended workflow.

The browser side is exercised by tests/stage3_deeparcade_0_07_harness.js,
a nonvisual Node harness in the P0-02 pattern. The Python side is
replayed here from the same fixtures
(tests/fixtures/stage3_deeparcade_0_07_fixtures.json), whose expected
records were generated from src/narcade/snake.py with food placement
scripted by a queue (injectable food choices): the initial placement is
the first queue entry, each eat consumes one more, and an exhausted queue
on a filled board means food is None.
"""

import json
import os
import re
import subprocess
import sys
from collections import deque

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "web", "game.html")
HARNESS = os.path.join(ROOT, "tests", "stage3_deeparcade_0_07_harness.js")
FIXTURES = os.path.join(
    ROOT, "tests", "fixtures", "stage3_deeparcade_0_07_fixtures.json")

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from narcade.snake import SnakeEnv  # noqa: E402
from scripts.train_dqn import encode_dqn  # noqa: E402


def _fixtures():
    with open(FIXTURES, encoding="utf-8") as f:
        return json.load(f)


def _source():
    with open(GAME, encoding="utf-8") as f:
        return f.read()


def _make_env(scn):
    """SnakeEnv with the fixture start state injected and food scripted."""
    env = SnakeEnv(scn["width"], scn["height"], seed=0)
    queue = [tuple(c) for c in scn["food_script"]]

    def placer():
        if queue:
            cell = queue.pop(0)
            assert cell not in set(env.body), "scripted food on the body"
            env.food = cell
        else:
            env.food = None

    env._place_food = placer
    env.body = deque(tuple(c) for c in scn["start"]["body"])
    env.heading = scn["start"]["heading"]
    env.score = scn["start"]["score"]
    env.steps = scn["start"]["steps"]
    env.done = False
    placer()  # initial scripted food
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


def _replay_python(scn):
    env = _make_env(scn)
    records = [_snap(env, None, None, env.encode_state())]
    for a in scn["actions"]:
        state, reward, done, info = env.step(a)
        records.append(_snap(env, reward, info.get("reason"), state))
        if done:
            break
    return env, records


def test_stage3_deeparcade_0_07_python_replay_matches_fixtures():
    doc = _fixtures()
    names = [s["name"] for s in doc["scenarios"]]
    assert names == [
        "win_playthrough_4x4", "wall_death_5x5", "self_collision_5x5",
        "timeout_4x4", "win_terminal_2x3", "dqn_state_12x12",
        "win_terminal_12x12",
    ]
    for scn in doc["scenarios"]:
        _, records = _replay_python(scn)
        assert records == scn["records"], scn["name"]


def test_stage3_deeparcade_0_07_python_terminal_outcomes():
    doc = _fixtures()
    by_name = {s["name"]: s for s in doc["scenarios"]}
    expected = {
        "win_playthrough_4x4": "win",
        "wall_death_5x5": "death",
        "self_collision_5x5": "death",
        "timeout_4x4": "timeout",
        "win_terminal_2x3": "win",
        "win_terminal_12x12": "win",
    }
    for name, reason in expected.items():
        env, records = _replay_python(by_name[name])
        assert records[-1]["reason"] == reason, name
        assert env.done
        # Terminal observation is well-defined and repeatable, and the
        # finished episode refuses further steps.
        assert env.encode_state() == records[-1]["state"]
        with pytest.raises(RuntimeError):
            env.step(0)


def test_stage3_deeparcade_0_07_dqn_encoding_parity():
    doc = _fixtures()
    by_name = {s["name"]: s for s in doc["scenarios"]}
    for name in ("dqn_state_12x12", "win_terminal_12x12"):
        scn = by_name[name]
        env = _make_env(scn)
        assert [float(x) for x in encode_dqn(env)] == scn["dqn_before"], name
    # The post-win vector follows the documented layout with an empty
    # food plane (the Python encoder is only defined while food exists;
    # the browser must still return this consistent terminal observation).
    scn = by_name["win_terminal_12x12"]
    final = scn["records"][-1]
    v = [0.0] * 436
    body = [tuple(c) for c in final["body"]]
    hr, hc = body[-1]
    v[hr * 12 + hc] = 1.0
    for r, c in body[:-1]:
        v[144 + r * 12 + c] = 1.0
    v[432 + final["heading"]] = 1.0
    assert scn["dqn_after"] == v
    assert all(x == 0.0 for x in scn["dqn_after"][288:432])


def test_stage3_deeparcade_0_07_harness_passes():
    proc = subprocess.run(
        ["node", HARNESS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "harness: all checks passed" in proc.stdout


def test_stage3_deeparcade_0_07_browser_terminal_win_source():
    src = _source()
    # The winning step ends the game explicitly instead of leaving a
    # live game with null food.
    assert re.search(r"if\(!this\.food\)\{\s*this\.done=true;\s*this\.won=true;\s*\}", src), (
        "step must mark done and won when food placement finds no free cell"
    )
    assert re.search(r"this\.done=false;\s*this\.won=false;", src), (
        "reset must clear the win flag alongside done"
    )
    # Terminal observations must not dereference null food.
    assert re.search(r"encodeState\(\)\{[\s\S]*?if\(this\.food\)\{", src), (
        "encodeState must guard the food direction on a terminal win"
    )
    assert re.search(r"if\(this\.food\)\s*v\[288\+this\.food\[0\]\*12\+this\.food\[1\]\]=1;", src), (
        "encodeDQN must guard the food plane on a terminal win"
    )


if __name__ == "__main__":
    sys.exit(0)
