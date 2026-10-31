"""Stage 3 deeparcade P0-02: repair tournament action-state identity.

Acceptance (handoffs section 003):
- mutating global G cannot change a fixed aiGame action;
- AI terminal tests and wins/draws;
- repeatable paired fixtures;
- ordinary human keyboard controls, restart, and standalone game unaffected.

The browser game logic lives in web/game.html. The Node harness
(tests/stage3_deeparcade_0_02_harness.js) loads that real script with
stubbed browser globals and drives it without a DOM. The static checks
below pin the source shape of the fix: an explicit kind parameter, no
bare aiAction() call falling back to the live game, and a tournament AI
game built from the human episode's recorded seed and mode instead of a
fresh random seed.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "web", "game.html")
HARNESS = os.path.join(ROOT, "tests", "stage3_deeparcade_0_02_harness.js")


def _source():
    with open(GAME, encoding="utf-8") as f:
        return f.read()


def test_stage3_deeparcade_0_02_harness_passes():
    proc = subprocess.run(
        ["node", HARNESS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "harness: all checks passed" in proc.stdout


def test_stage3_deeparcade_0_02_ai_action_has_explicit_kind():
    src = _source()
    assert re.search(r"function aiAction\(game,\s*kind\)", src), (
        "aiAction must take (game, kind) with no default to the live game"
    )
    assert "aiAction()" not in src, (
        "no bare aiAction() call may fall back to the global live game"
    )
    assert re.search(r"action\s*=\s*aiAction\(G,\s*aiKind\)", src), (
        "the live AI turn must pass the live game and its kind explicitly"
    )


def test_stage3_deeparcade_0_02_tournament_uses_paired_episode():
    src = _source()
    assert "saveKind" not in src, (
        "the tournament must not swap the global aiKind selector"
    )
    assert re.search(
        r"new Snake\(12,\s*12,\s*recording\.seed,\s*recording\.mode\)", src
    ), "tournament AI game must reuse the human episode seed and mode"
    assert re.search(r"aiAction\(aiGame,\s*opp\)", src), (
        "tournament AI actions must carry the fixed opponent kind"
    )


def test_stage3_deeparcade_0_02_human_controls_untouched():
    src = _source()
    assert "addEventListener('keydown'" in src
    assert "KEYDIR" in src and "inputQueue" in src
    assert re.search(r"function newGame\(m,\s*kind,\s*ai\)", src)
    assert "playKind==='human'" in src


if __name__ == "__main__":
    sys.exit(0)
