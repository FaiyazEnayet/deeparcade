"""Baseline and learned policies for the Snake environment."""

from __future__ import annotations

import random

from .snake import DIRS, LEFT, RIGHT, STRAIGHT


class RandomPolicy:
    """Uniform random actions. The dumb baseline."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, env):
        return self.rng.choice([STRAIGHT, LEFT, RIGHT])


class GreedyPolicy:
    """Heuristic baseline: move toward the food, avoid instant death.

    Picks the action that minimizes Manhattan distance to the food after the
    move, discarding moves that kill immediately when a safe move exists.
    """

    def act(self, env):
        hr, hc = env.body[-1]
        best, best_d = None, None
        safe_moves = []
        for a in (STRAIGHT, LEFT, RIGHT):
            h = env.heading
            if a == LEFT:
                h = (h + 3) % 4
            elif a == RIGHT:
                h = (h + 1) % 4
            dr, dc = DIRS[h]
            nr, nc = hr + dr, hc + dc
            if env._cell_blocked(nr, nc, ignore_tail=True):
                continue
            safe_moves.append(a)
            d = abs(nr - env.food[0]) + abs(nc - env.food[1])
            if best_d is None or d < best_d:
                best_d, best = d, a
        if safe_moves:
            return best
        return STRAIGHT  # doomed; pick anything


class QTablePolicy:
    """Greedy policy from a trained Q-table: dict/list {state: [q0, q1, q2]}."""

    def __init__(self, qtable):
        # qtable: list of 32 rows, each a list of 3 Q-values
        self.qtable = qtable

    def act(self, env):
        s = env.encode_state()
        qs = self.qtable[s]
        best = 0
        for a in (1, 2):
            if qs[a] > qs[best]:
                best = a
        return best

    def to_jsonable(self):
        return {"states": 32, "actions": 3,
                "best_action": [max(range(3), key=lambda a: row[a])
                                for row in self.qtable]}
