"""Build docs/game.html: inject trained policies into web/game.html.

Replacements:
  /*__QPOLICY__*/[]      -> best-action array from runs/policy.json
  /*__DQNW__*/null       -> dqn_weights.json
  /*__AI_ELO__*/{...}    -> measured ELO ratings from runs/eval.json
  /*__AI_REF__*/{...}    -> median eval scores per AI (human-ELO reference)
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def main():
    with open(os.path.join(ROOT, "web", "game.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(ROOT, "runs", "policy.json"), encoding="utf-8") as f:
        qpolicy = json.load(f)["best_action"]
    with open(os.path.join(ROOT, "runs", "dqn_weights.json"), encoding="utf-8") as f:
        dqnw = json.load(f)
    with open(os.path.join(ROOT, "runs", "eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    # P0-01: read the versioned schema only. A pre-v2 artifact (no
    # episode keys, undocumented pairing) must fail here with a clear
    # message, never silently inject numbers from it.
    if ev.get("schema_version") != 2:
        raise ValueError(
            f"runs/eval.json has schema_version {ev.get('schema_version')!r}; "
            f"this generator reads schema_version 2 only. Regenerate "
            f"runs/eval.json with scripts/evaluate.py"
        )
    for key in ("elo", "summary"):
        if key not in ev:
            raise ValueError(f"runs/eval.json is missing required key {key!r}")

    elo = {k: ev["elo"][k] for k in ("rookie", "pro", "grandmaster")}
    ref = {k: ev["summary"][k]["median"] for k in ("rookie", "pro", "grandmaster")}

    html = html.replace("/*__QPOLICY__*/[]", json.dumps(qpolicy))
    html = html.replace("/*__DQNW__*/null", json.dumps(dqnw))
    html = html.replace('/*__AI_ELO__*/{"rookie":1240,"pro":1500,"grandmaster":1750}',
                        json.dumps(elo))
    html = html.replace('/*__AI_REF__*/{"rookie":6,"pro":14,"grandmaster":28}',
                        json.dumps(ref))

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "game.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"game -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
