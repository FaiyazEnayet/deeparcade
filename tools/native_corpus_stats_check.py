#!/usr/bin/env python3
"""Independent recount of the frozen golden corpora vs meta.json.

tools/native_reference.py records 19 corpus counts in
native/golden/meta.json at generation time, and the meta_check CTest
verifies the recorded values are positive whole numbers under exactly
the expected key set, plus the golden files' SHA-256 hashes. That
leaves one gap: a wrong-but-positive count (a generator tally bug, a
hand edit) passes. This checker closes it by recounting the records
from the corpus files themselves, using the same record definitions
the generator tallied and the C++ batch runners parse (see the
protocol headers in native/tests/*_diff_runner.cpp), and comparing
every derived count against meta.json.

Record definitions per file:

  transitions.json   episodes = len(episodes);
                     steps = sum of len(episode.steps).
  q_updates.json     q_updates = len(updates);
                     q_act_stream = len(act_states).
  mcts_oracle.json   mcts_oracle_cases = len(cases).
  snake_corpus.txt   session file ("> " inputs, "< " replies):
                     episodes = EPISODE records; transitions = "S"
                     step lines; exceptions = "X" step lines. On an
                     "S" line (S <state_before> <state_after>
                     <reward_hex> <done> <reason> <food> <score>
                     <steps> <body>): eats = reward_hex equals the
                     IEEE754 hex of 10.0; deaths / wins / timeouts =
                     reason code 1 / 2 / 3. tail_vacations are not a
                     stored field; they are reconstructed exactly as
                     the generator defined them: a step whose new head
                     cell (last cell of the post-step body, bodies are
                     tail-first) equals the pre-step tail cell
                     (body[0] before the step, tracked across the
                     episode from INIT and each post-step body) on a
                     step that was not already done and did not eat
                     (new head != pre-step food).
  policies_corpus.txt  episodes = EPISODE records; states = "P"
                     replies other than the post-terminal "P -1 -1"
                     placeholder; exceptions = "X" step lines;
                     rand_draws = integers on the data line following
                     each "RAND" reply.
  mcts_cases.txt     searches = "ACTION" replies; root_children =
                     "CHILD" reply lines.

Usage: native_corpus_stats_check.py [--golden-dir native/golden]
Exit code is non-zero on any mismatch. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# dbl_hex(10.0) in tools/native_reference.py: the reward a step gets
# for eating, as the generator's eats tally compares it.
EAT_REWARD_HEX = struct.pack(">d", 10.0).hex()


def parse_body(tok: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in cell.split(","))
            for cell in tok.split(";")]


def parse_food(tok: str) -> tuple[int, int] | None:
    r, c = (int(x) for x in tok.split(","))
    return None if (r, c) == (-1, -1) else (r, c)


def count_transitions(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    episodes = data["episodes"]
    return {
        "transitions_episodes": len(episodes),
        "transitions_steps": sum(len(e["steps"]) for e in episodes),
    }


def count_q_updates(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {
        "q_updates": len(data["updates"]),
        "q_act_stream": len(data["act_states"]),
    }


def count_mcts_oracle(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {"mcts_oracle_cases": len(data["cases"])}


def count_snake_corpus(path: Path) -> dict[str, int]:
    counts = {
        "snake_episodes": 0,
        "snake_transitions": 0,
        "snake_exceptions": 0,
        "snake_eats": 0,
        "snake_deaths": 0,
        "snake_wins": 0,
        "snake_timeouts": 0,
        "snake_tail_vacations": 0,
    }
    pre_body: list[tuple[int, int]] | None = None
    pre_food: tuple[int, int] | None = None
    pre_done = False
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("> EPISODE "):
                counts["snake_episodes"] += 1
                pre_body = None
                pre_food = None
                pre_done = False
            elif line.startswith("< INIT "):
                fields = line[len("< INIT "):].split(" ")
                pre_food = parse_food(fields[1])
                pre_body = parse_body(fields[2])
            elif line.startswith("< X "):
                # step() raised; env state (and the done flag) unchanged.
                counts["snake_exceptions"] += 1
            elif line.startswith("< S "):
                fields = line[len("< S "):].split(" ")
                reward_hex = fields[2]
                done = int(fields[3])
                reason = int(fields[4])
                post_food = parse_food(fields[5])
                post_body = parse_body(fields[8])
                counts["snake_transitions"] += 1
                if reward_hex == EAT_REWARD_HEX:
                    counts["snake_eats"] += 1
                if reason == 1:
                    counts["snake_deaths"] += 1
                elif reason == 2:
                    counts["snake_wins"] += 1
                elif reason == 3:
                    counts["snake_timeouts"] += 1
                if not pre_done and pre_body:
                    target = post_body[-1]
                    ate = pre_food is not None and target == pre_food
                    if target == pre_body[0] and not ate:
                        counts["snake_tail_vacations"] += 1
                pre_body = post_body
                pre_food = post_food
                pre_done = done == 1
    return counts


def count_policies_corpus(path: Path) -> dict[str, int]:
    counts = {
        "policies_episodes": 0,
        "policies_states": 0,
        "policies_exceptions": 0,
        "policies_rand_draws": 0,
    }
    expect_rand_data = False
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("> EPISODE "):
                counts["policies_episodes"] += 1
            elif line.startswith("< P "):
                if line != "< P -1 -1":
                    counts["policies_states"] += 1
            elif line.startswith("< X "):
                counts["policies_exceptions"] += 1
            elif line == "< RAND":
                expect_rand_data = True
            elif expect_rand_data:
                counts["policies_rand_draws"] += len(line[2:].split())
                expect_rand_data = False
    return counts


def count_mcts_cases(path: Path) -> dict[str, int]:
    counts = {"mcts_searches": 0, "mcts_root_children": 0}
    with path.open() as fh:
        for line in fh:
            if line.startswith("< ACTION "):
                counts["mcts_searches"] += 1
            elif line.startswith("< CHILD "):
                counts["mcts_root_children"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", default=str(ROOT / "native" / "golden"),
                    help="golden corpus directory (default: native/golden)")
    args = ap.parse_args()
    golden = Path(args.golden_dir)

    derived: dict[str, int] = {}
    derived.update(count_transitions(golden / "transitions.json"))
    derived.update(count_q_updates(golden / "q_updates.json"))
    derived.update(count_mcts_oracle(golden / "mcts_oracle.json"))
    derived.update(count_snake_corpus(golden / "snake_corpus.txt"))
    derived.update(count_policies_corpus(golden / "policies_corpus.txt"))
    derived.update(count_mcts_cases(golden / "mcts_cases.txt"))

    meta = json.loads((golden / "meta.json").read_text())
    recorded = meta["counts"]

    failures = []
    for key in sorted(set(derived) | set(recorded)):
        got = derived.get(key)
        want = recorded.get(key)
        if got is None:
            failures.append(f"{key}: recorded {want} but not derivable "
                            "from the corpus files")
            status = "NOT DERIVED"
        elif want is None:
            failures.append(f"{key}: derived {got} but absent from "
                            "meta.json counts")
            status = "NOT RECORDED"
        elif got != want:
            failures.append(f"{key}: derived {got} != recorded {want}")
            status = "MISMATCH"
        else:
            status = "ok"
        print(f"{key:26s} derived={got!s:>8} recorded={want!s:>8} "
              f"{status}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("result: FAIL")
        return 1
    print(f"result: PASS ({len(derived)} counts recounted from the "
          "corpus files, all match meta.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
