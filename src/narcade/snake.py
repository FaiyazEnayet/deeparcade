"""Snake environment: pure logic, no rendering.

Shared by the Python trainer and mirrored 1:1 in the browser game (web/game.html).
The state encoding documented here MUST stay identical in both implementations.
"""

from __future__ import annotations

import random
from collections import deque

# Headings: 0=up, 1=right, 2=down, 3=left. Row 0 is the top row.
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Relative actions.
STRAIGHT, LEFT, RIGHT = 0, 1, 2


class SnakeEnv:
    """Minimal Snake. step() returns (state, reward, done, info).

    Reward shaping (Stage 3 R-03): the default shaping is "dense", the
    long-standing baseline: +0.5 for moving closer to the food in
    Manhattan distance, -0.5 for moving farther, 0 for no change, food
    +10, death -10. Two further modes share the same base rewards:
    "potential" adds F(s,a,s') = gamma*Phi(s') - Phi(s) with
    Phi(s) = -scale * Manhattan(head, food) / (width + height - 2), and
    "none" pays only the base rewards. Every step's info carries the
    exact decomposition: reward == base_reward + shaping_reward.
    """

    FOOD_REWARD = 10.0
    DEATH_REWARD = -10.0
    CLOSER_REWARD = 0.5
    FARTHER_REWARD = -0.5

    SHAPING_MODES = ("dense", "potential", "none")

    def __init__(self, width=12, height=12, seed=None, max_steps=None,
                 shaping="dense", shaping_gamma=0.95, shaping_scale=1.0):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.episode_seed = seed
        self.max_steps = max_steps if max_steps is not None else 20 * width * height
        if shaping not in self.SHAPING_MODES:
            raise ValueError(
                f"unknown shaping {shaping!r}; expected one of "
                f"{self.SHAPING_MODES}")
        self.shaping = shaping
        self.shaping_gamma = float(shaping_gamma)
        self.shaping_scale = float(shaping_scale)
        # The constructor performs exactly one opening reset: the first
        # food placement consumes the first draw of the seeded stream.
        # That opening is the episode opening for keyed episodes (see
        # scripts/evaluate.py run_keyed_episodes, which does not reset
        # again). A caller that calls bare reset() afterwards plays a
        # second draw, not the constructor opening.
        self.reset()

    def reset(self, seed=None):
        """Start a new episode on this environment.

        With no seed, the episode continues the current RNG stream: the
        food placement is the next draw, so two bare resets in a row
        give different openings and an episode replayed on a reused
        environment is not the same episode. With an explicit seed, the
        stream restarts first, so reset(seed=X) reproduces the opening
        of SnakeEnv(seed=X) exactly. Keyed episodes pass the episode's
        environment key here (or, equivalently, construct a fresh env
        with it) so every episode opening is counted and reproducible.
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.episode_seed = seed
        cy, cx = self.height // 2, self.width // 2
        self.body = deque([(cy, cx - 2), (cy, cx - 1), (cy, cx)])
        self.heading = 1  # right
        self.score = 0
        self.steps = 0
        self.done = False
        self._place_food()
        return self.encode_state()

    def _place_food(self):
        occupied = set(self.body)
        free = [(r, c) for r in range(self.height) for c in range(self.width)
                if (r, c) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def _cell_blocked(self, r, c, ignore_tail=True):
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        body = set(self.body)
        if ignore_tail and len(self.body) > 0:
            body.discard(self.body[0])  # tail vacates this turn (unless eating)
        return (r, c) in body

    def encode_state(self):
        """32-state encoding. MUST match web/game.html exactly.

        Bits: danger straight / left / right (cell ahead is wall or body),
        then food direction relative to heading: 0=ahead, 1=right, 2=left, 3=behind.
        state = ((ds*2 + dl)*2 + dr) * 4 + food_dir  -> 0..31

        On a filled board (food is None, terminal win) there is no food
        direction; the encoding uses food_dir 0 and the danger bits as
        computed (all set, since every neighbor is body or wall).
        """
        hr, hc = self.body[-1]
        h = self.heading
        ahead = DIRS[h]
        left = DIRS[(h + 3) % 4]
        right = DIRS[(h + 1) % 4]
        ds = 1 if self._cell_blocked(hr + ahead[0], hc + ahead[1]) else 0
        dl = 1 if self._cell_blocked(hr + left[0], hc + left[1]) else 0
        dr = 1 if self._cell_blocked(hr + right[0], hc + right[1]) else 0

        if self.food is None:
            food_dir = 0  # terminal win: no food direction
        else:
            fr, fc = self.food
            dy, dx = fr - hr, fc - hc
            # Rotate (dx, dy) into the heading frame: fa = forward comp, ra = right comp.
            if h == 0:      # up
                fa, ra = -dy, dx
            elif h == 1:    # right
                fa, ra = dx, dy
            elif h == 2:    # down
                fa, ra = dy, -dx
            else:           # left
                fa, ra = -dx, -dy
            if abs(fa) >= abs(ra):
                food_dir = 0 if fa > 0 else 3
            else:
                food_dir = 1 if ra > 0 else 2
        return ((ds * 2 + dl) * 2 + dr) * 4 + food_dir

    def potential(self):
        """Phi(s) for the current state.

        Negative normalized Manhattan distance from the head to the
        food, scaled: -scale * d / (width + height - 2), so Phi lives
        in [-scale, 0]. Terminal states have no food (a filled board is
        a win), and Phi is 0 there by definition: the shaping term on a
        terminal step is gamma * 0 - Phi(s) = -Phi(s), bounded by the
        scale, so a death step stays negative and a win step stays
        positive. Timeout is a truncation, not a terminal state, so Phi
        is evaluated normally on the timeout step.
        """
        if self.food is None:
            return 0.0
        hr, hc = self.body[-1]
        d = abs(hr - self.food[0]) + abs(hc - self.food[1])
        return -self.shaping_scale * d / (self.width + self.height - 2)

    def step(self, action):
        """Take a relative action. Returns (state, reward, done, info).

        info carries the exact reward decomposition: reward is always
        base_reward + shaping_reward, with shaping_reward 0 in "none"
        mode, the closer/farther term in "dense" mode, and the
        potential term gamma*Phi(s') - Phi(s) in "potential" mode.
        On an eating step Phi(s') is evaluated against the newly placed
        food, so a respawn cannot leak reward from the old food.
        """
        if self.done:
            raise RuntimeError("step() called on finished episode; call reset() first")
        if action == LEFT:
            self.heading = (self.heading + 3) % 4
        elif action == RIGHT:
            self.heading = (self.heading + 1) % 4
        elif action != STRAIGHT:
            raise ValueError(f"bad action {action}")

        hr, hc = self.body[-1]
        dr, dc = DIRS[self.heading]
        nr, nc = hr + dr, hc + dc
        self.steps += 1

        phi_before = self.potential() if self.shaping == "potential" else 0.0

        ate = (nr, nc) == self.food
        # The tail vacates unless we eat; check collision against the post-move body.
        if self._cell_blocked(nr, nc, ignore_tail=not ate):
            self.done = True
            base = self.DEATH_REWARD
            shaping = -phi_before if self.shaping == "potential" else 0.0
            return self.encode_state(), base + shaping, True, {
                "reason": "death", "base_reward": base,
                "shaping_reward": shaping, "shaping": self.shaping}

        dist_before = abs(hr - self.food[0]) + abs(hc - self.food[1])
        self.body.append((nr, nc))
        if ate:
            self.score += 1
            self._place_food()
            base = self.FOOD_REWARD
            if self.shaping == "potential":
                # Phi(s') against the newly placed food (0 on a win,
                # where the board is full and the food is None).
                shaping = self.shaping_gamma * self.potential() - phi_before
            else:
                shaping = 0.0
            if self.food is None:
                # Board completely filled: win.
                self.done = True
                return self.encode_state(), base + shaping, True, {
                    "reason": "win", "base_reward": base,
                    "shaping_reward": shaping, "shaping": self.shaping}
        else:
            self.body.popleft()
            dist_after = abs(nr - self.food[0]) + abs(nc - self.food[1])
            base = 0.0
            if self.shaping == "dense":
                if dist_after < dist_before:
                    shaping = self.CLOSER_REWARD
                elif dist_after > dist_before:
                    shaping = self.FARTHER_REWARD
                else:
                    shaping = 0.0
            elif self.shaping == "potential":
                shaping = self.shaping_gamma * self.potential() - phi_before
            else:
                shaping = 0.0

        if self.steps >= self.max_steps:
            self.done = True
            return self.encode_state(), base + shaping, True, {
                "reason": "timeout", "base_reward": base,
                "shaping_reward": shaping, "shaping": self.shaping}
        return self.encode_state(), base + shaping, False, {
            "base_reward": base, "shaping_reward": shaping,
            "shaping": self.shaping}

    @property
    def occupied(self):
        return set(self.body)

    def render_ascii(self):
        rows = []
        occ = set(self.body)
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) == self.food:
                    row.append("*")
                elif (r, c) == self.body[-1]:
                    row.append("H")
                elif (r, c) in occ:
                    row.append("o")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)
