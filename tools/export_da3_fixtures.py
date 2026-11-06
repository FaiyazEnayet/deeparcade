"""Export the canonical DA3 transition fixtures.

Generates tests/fixtures/stage3_deeparcade_3_fixtures.json from the
Python environment (src/narcade/snake.py), the specification the browser
core mirrors. Each case states the full pre-step state, one action, the
explicit post-eat food, and the expected post-step record. Food placement
is injected, never drawn from the RNG, so the fixtures are
seed-independent and replay identically in Python and in the headless
JS harness (tests/stage3_deeparcade_3_harness.js).

Usage: python3 tools/export_da3_fixtures.py
"""

import json
import os
import sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from narcade.snake import SnakeEnv  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures",
                   "stage3_deeparcade_3_fixtures.json")

# Headings: 0=up, 1=right, 2=down, 3=left. Actions: 0=straight, 1=left,
# 2=right. Body lists are tail-first, head last.
CASES = [
    # Boundary: one death per wall, plus a turn into the wall at a corner.
    {"name": "boundary_north", "category": "boundary", "width": 5,
     "height": 5, "body": [[1, 1], [0, 1]], "heading": 0, "food": [4, 4],
     "food_after": None, "score": 0, "steps": 0, "max_steps": 500,
     "action": 0},
    {"name": "boundary_east", "category": "boundary", "width": 5,
     "height": 5, "body": [[2, 3], [2, 4]], "heading": 1, "food": [0, 0],
     "food_after": None, "score": 0, "steps": 0, "max_steps": 500,
     "action": 0},
    {"name": "boundary_south", "category": "boundary", "width": 5,
     "height": 5, "body": [[3, 2], [4, 2]], "heading": 2, "food": [0, 0],
     "food_after": None, "score": 0, "steps": 0, "max_steps": 500,
     "action": 0},
    {"name": "boundary_west", "category": "boundary", "width": 5,
     "height": 5, "body": [[2, 1], [2, 0]], "heading": 3, "food": [4, 4],
     "food_after": None, "score": 0, "steps": 0, "max_steps": 500,
     "action": 0},
    {"name": "boundary_corner_turn", "category": "boundary", "width": 5,
     "height": 5, "body": [[1, 4], [0, 4]], "heading": 0, "food": [4, 0],
     "food_after": None, "score": 0, "steps": 0, "max_steps": 500,
     "action": 2},
    # Tail-vacating: the tail cell is free when not eating, lethal when
    # eating into it, because the tail does not vacate on an eat.
    {"name": "tail_vacating_straight", "category": "tail-vacating",
     "width": 5, "height": 5,
     "body": [[2, 1], [2, 2], [1, 2], [1, 1]], "heading": 2,
     "food": [0, 4], "food_after": None, "score": 0, "steps": 0,
     "max_steps": 500, "action": 0},
    {"name": "tail_vacating_turn", "category": "tail-vacating",
     "width": 5, "height": 5,
     "body": [[2, 1], [2, 2], [1, 2], [1, 1]], "heading": 1,
     "food": [4, 4], "food_after": None, "score": 0, "steps": 0,
     "max_steps": 500, "action": 2},
    {"name": "tail_vacating_eat_is_death", "category": "tail-vacating",
     "width": 5, "height": 5,
     "body": [[2, 1], [2, 2], [1, 2], [1, 1]], "heading": 2,
     "food": [2, 1], "food_after": None, "score": 0, "steps": 0,
     "max_steps": 500, "action": 0},
    # Food: eating straight and on a turn, with the next food stated.
    {"name": "food_eat_straight", "category": "food", "width": 5,
     "height": 5, "body": [[2, 0], [2, 1], [2, 2]], "heading": 1,
     "food": [2, 3], "food_after": [0, 0], "score": 0, "steps": 0,
     "max_steps": 500, "action": 0},
    {"name": "food_eat_turn", "category": "food", "width": 5,
     "height": 5, "body": [[2, 0], [2, 1], [2, 2]], "heading": 1,
     "food": [1, 2], "food_after": [4, 4], "score": 0, "steps": 0,
     "max_steps": 500, "action": 1},
    {"name": "food_closer", "category": "food", "width": 5,
     "height": 5, "body": [[2, 0], [2, 1], [2, 2]], "heading": 1,
     "food": [2, 4], "food_after": None, "score": 0, "steps": 0,
     "max_steps": 500, "action": 0},
    {"name": "food_farther", "category": "food", "width": 5,
     "height": 5, "body": [[2, 0], [2, 1], [2, 2]], "heading": 1,
     "food": [0, 0], "food_after": None, "score": 0, "steps": 0,
     "max_steps": 500, "action": 0},
    # Timeout: the step that reaches 20 * width * height ends the game.
    {"name": "timeout_4x4", "category": "timeout", "width": 4,
     "height": 4, "body": [[2, 0], [2, 1], [2, 2]], "heading": 1,
     "food": [0, 0], "food_after": None, "score": 0, "steps": 319,
     "max_steps": 320, "action": 0},
    # Terminal win: the last free cell is eaten on a 3x3 board.
    {"name": "terminal_win_3x3", "category": "terminal", "width": 3,
     "height": 3,
     "body": [[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0], [2, 0],
              [2, 1]],
     "heading": 1, "food": [2, 2], "food_after": None, "score": 7,
     "steps": 7, "max_steps": 180, "action": 0},
]


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


def build_case(spec):
    env = SnakeEnv(spec["width"], spec["height"], seed=0,
                   max_steps=spec["max_steps"])
    queue = [tuple(spec["food"]), spec["food_after"]]
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
    env.body = deque(tuple(c) for c in spec["body"])
    env.heading = spec["heading"]
    env.score = spec["score"]
    env.steps = spec["steps"]
    env.done = False
    placer()
    records = [_snap(env, None, None, env.encode_state())]
    state, reward, done, info = env.step(spec["action"])
    records.append(_snap(env, reward, info.get("reason"), state))
    return {
        "name": spec["name"],
        "category": spec["category"],
        "width": spec["width"],
        "height": spec["height"],
        "start": {
            "body": [list(c) for c in spec["body"]],
            "heading": spec["heading"],
            "score": spec["score"],
            "steps": spec["steps"],
        },
        "food": list(spec["food"]),
        "food_after": (list(spec["food_after"])
                       if spec["food_after"] is not None else None),
        "max_steps": spec["max_steps"],
        "action": spec["action"],
        "expected": records,
    }


def main():
    doc = {
        "schema": 1,
        "about": (
            "Canonical seed-independent single-step transition fixtures "
            "for deeparcade DA3. Expected records were generated from "
            "the Python environment (src/narcade/snake.py), the "
            "specification the browser core mirrors, with food placement "
            "injected: the initial food is stated, and an eat places "
            "exactly food_after (null when the board fills), so no case "
            "consumes an RNG draw. Both replay drivers inject the start "
            "state and the stated food, apply the same single action, "
            "and must reproduce every expected record exactly. "
            "Regenerate with: python3 tools/export_da3_fixtures.py"
        ),
        "cases": [build_case(spec) for spec in CASES],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT} ({len(doc['cases'])} cases)")


if __name__ == "__main__":
    main()
