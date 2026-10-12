"""MCTS player for Snake: real search with UCT and greedy rollouts.

This is the "Grandmaster" tier: no training, just lookahead. It works in any
mode (classic, maze) because it searches the actual game dynamics.
"""

from __future__ import annotations

import math
import random

from .snake import DIRS, LEFT, RIGHT, STRAIGHT

ACTIONS = (STRAIGHT, LEFT, RIGHT)


def _rel_heading(heading, action):
    if action == LEFT:
        return (heading + 3) % 4
    if action == RIGHT:
        return (heading + 1) % 4
    return heading


class _Node:
    __slots__ = ("parent", "action", "children", "untried", "visits", "value")

    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = {}
        self.untried = [STRAIGHT, LEFT, RIGHT]
        self.visits = 0
        self.value = 0.0


class MCTSPlayer:
    """UCT search. `simulations` = search budget per move (real search depth)."""

    def __init__(self, simulations=300, rollout_steps=100, c=1.0, seed=None):
        self.simulations = simulations
        self.rollout_steps = rollout_steps
        self.c = c
        self.rng = random.Random(seed)

    # ---- state helpers (lightweight tuples, no env copies) ----
    @staticmethod
    def _blocked(body_set, walls, w, h, r, c):
        if not (0 <= r < h and 0 <= c < w):
            return True
        return (r, c) in body_set or (r, c) in walls

    def _greedy_rollout_action(self, body, food, heading, walls, w, h):
        hr, hc = body[-1]
        body_set = set(body)
        best, best_d = None, None
        safe = []
        for a in ACTIONS:
            hh = _rel_heading(heading, a)
            dr, dc = DIRS[hh]
            nr, nc = hr + dr, hc + dc
            tail = body[0]
            occ = body_set - {tail}
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in occ \
                    or (nr, nc) in walls:
                continue
            safe.append((a, hh, nr, nc))
            d = abs(nr - food[0]) + abs(nc - food[1])
            if best_d is None or d < best_d:
                best_d, best = d, (a, hh, nr, nc)
        if best is not None:
            return best
        if safe:
            return self.rng.choice(safe)
        return None

    def _step_state(self, body, food, heading, walls, w, h, action, rng):
        """Apply action to a lightweight state. Returns (body, food, heading, ate, dead)."""
        heading = _rel_heading(heading, action)
        hr, hc = body[-1]
        dr, dc = DIRS[heading]
        nr, nc = hr + dr, hc + dc
        ate = (nr, nc) == food
        occ = set(body)
        if not ate:
            occ.discard(body[0])
        if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in occ or (nr, nc) in walls:
            return body, food, heading, False, True
        body = body + [(nr, nc)]
        if ate:
            occ = set(body)
            free = [(r, c) for r in range(h) for c in range(w)
                    if (r, c) not in occ and (r, c) not in walls]
            food = rng.choice(free) if free else (-1, -1)
        else:
            body = body[1:]
        return body, food, heading, ate, False

    def _rollout(self, body, food, heading, walls, w, h, rng, disc0=1.0,
                 gamma=0.95):
        """Greedy rollout with discounted return.

        value = sum_t gamma^t * [food eaten at step t]. Discounting makes the
        search sensitive to the immediate action: eating soon beats eating late.
        Without it, a competent greedy rollout washes out all value differences.
        disc0 lets the caller align the rollout's discount with rewards already
        accumulated along the tree path.
        """
        disc, value = disc0, 0.0
        for _ in range(self.rollout_steps):
            mv = self._greedy_rollout_action(body, food, heading, walls, w, h)
            if mv is None:
                break
            a, hh, nr, nc = mv
            body, food, heading, ate, dead = self._step_state(
                body, food, heading, walls, w, h, a, rng)
            if dead:
                break
            if ate:
                value += disc
            disc *= gamma
        return value

    def _uct(self, node):
        best, best_v = None, -1e18
        for a, ch in node.children.items():
            if ch.visits == 0:
                v = 1e17
            else:
                v = (ch.value / ch.visits
                     + self.c * math.sqrt(math.log(node.visits) / ch.visits))
            if v > best_v:
                best_v, best = v, a
        return best

    def search(self, env, walls=frozenset()):
        """Return the best relative action from the env's current state."""
        w, h = env.width, env.height
        root_body = tuple(env.body)
        root_food = env.food
        root_heading = env.heading
        rng = self.rng

        root = _Node(None, None)
        gamma = 0.95
        for _ in range(self.simulations):
            # --- selection ---
            node = root
            body = list(root_body)
            food = root_food
            heading = root_heading
            dead = False
            # Accumulate discounted immediate rewards along the tree path so
            # food eaten inside the tree is credited (not just rollout food).
            path_value, disc = 0.0, 1.0
            while node.children and not node.untried and not dead:
                a = self._uct(node)
                node = node.children[a]
                body, food, heading, ate, dead = self._step_state(
                    body, food, heading, walls, w, h, a, rng)
                if ate:
                    path_value += disc
                disc *= gamma
            # --- expansion ---
            if not dead and node.untried:
                a = node.untried.pop(rng.randrange(len(node.untried)))
                child = _Node(node, a)
                node.children[a] = child
                node = child
                body, food, heading, ate, dead = self._step_state(
                    body, food, heading, walls, w, h, a, rng)
                if ate:
                    path_value += disc
                disc *= gamma
            # --- simulation ---
            if dead:
                value = path_value
            else:
                value = path_value + self._rollout(body, food, heading, walls,
                                                  w, h, rng, disc0=disc,
                                                  gamma=gamma)
            # --- backprop ---
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # most-visited child = best action
        best_a, best_v = STRAIGHT, -1
        for a, ch in root.children.items():
            if ch.visits > best_v:
                best_v, best_a = ch.visits, a
        return best_a

    def act(self, env, walls=frozenset()):
        return self.search(env, walls=walls)
