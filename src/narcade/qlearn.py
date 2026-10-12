"""Tabular Q-learning for Snake."""

from __future__ import annotations

import random

from .snake import LEFT, RIGHT, STRAIGHT


class QLearningAgent:
    def __init__(self, n_states=32, n_actions=3, alpha=0.15, gamma=0.9,
                 epsilon=1.0, seed=None):
        self.q = [[0.0] * n_actions for _ in range(n_states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice([STRAIGHT, LEFT, RIGHT])
        return self.greedy(state)

    def greedy(self, state):
        qs = self.q[state]
        best = 0
        for a in (1, 2):
            if qs[a] > qs[best]:
                best = a
        return best

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * max(self.q[s_next])
        self.q[s][a] += self.alpha * (target - self.q[s][a])
