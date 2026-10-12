"""deeparcade: a Snake arcade game with a genuinely trained AI opponent."""

from .snake import SnakeEnv
from .policies import RandomPolicy, GreedyPolicy, QTablePolicy
from .qlearn import QLearningAgent

__all__ = ["SnakeEnv", "RandomPolicy", "GreedyPolicy", "QTablePolicy",
           "QLearningAgent"]
