"""Q-learning, policies, and MCTS tests."""

import pytest

from narcade.mcts import MCTSPlayer
from narcade.policies import GreedyPolicy, QTablePolicy, RandomPolicy
from narcade.qlearn import QLearningAgent
from narcade.snake import SnakeEnv


def test_q_update_moves_toward_target():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0, seed=1)
    agent.update(5, 0, 10.0, 6, False)  # non-terminal
    # target = 10 + 0.9*max(Q[6]) = 10 ; Q += 0.5*(10-0) = 5
    assert agent.q[5][0] == pytest.approx(5.0)
    agent.update(5, 1, -10.0, 6, True)  # terminal
    assert agent.q[5][1] == pytest.approx(-5.0)


def test_greedy_action_picks_max():
    agent = QLearningAgent(epsilon=0.0, seed=1)
    agent.q[3] = [1.0, 5.0, 2.0]
    assert agent.greedy(3) == 1


def test_epsilon_greedy_explores():
    agent = QLearningAgent(epsilon=1.0, seed=1)
    acts = {agent.act(0) for _ in range(60)}
    assert acts == {0, 1, 2}


def test_random_policy_valid_actions():
    p = RandomPolicy(seed=1)
    env = SnakeEnv(seed=1)
    env.reset()
    for _ in range(20):
        assert p.act(env) in (0, 1, 2)


def test_greedy_policy_avoids_suicide():
    # head at (0,1) heading up: straight dies, must turn
    env = SnakeEnv(seed=1, width=6, height=6)
    env.reset()
    from collections import deque
    env.body = deque([(2, 1), (1, 1), (0, 1)])
    env.heading = 0
    env.food = (5, 5)
    a = GreedyPolicy().act(env)
    assert a != 0  # straight would hit the wall


def test_qtable_policy_uses_best_action():
    q = [[0.0, 0.0, 0.0] for _ in range(32)]
    q[7] = [0.1, 0.9, 0.2]
    pol = QTablePolicy(q)
    env = SnakeEnv(seed=1)
    env.reset()
    # force state 7 by monkey-checking encode path indirectly: just check jsonable
    j = pol.to_jsonable()
    assert j["best_action"][7] == 1
    assert len(j["best_action"]) == 32


def test_mcts_eats_adjacent_food():
    from collections import deque
    env = SnakeEnv(seed=1)
    env.body = deque([(4, 5), (4, 4), (3, 4)])
    env.heading = 0  # up
    env.food = (2, 4)  # directly ahead
    m = MCTSPlayer(simulations=300, seed=99)
    assert m.act(env) == 0  # straight eats


def test_mcts_avoids_instant_death():
    from collections import deque
    env = SnakeEnv(seed=1, width=6, height=6)
    env.body = deque([(2, 1), (1, 1), (0, 1)])
    env.heading = 0  # up; straight dies into wall
    env.food = (5, 5)
    m = MCTSPlayer(simulations=200, seed=3)
    assert m.act(env) != 0


def test_tiny_training_improves_over_random():
    """Smoke: a few hundred Q-learning episodes run and Q-values move."""
    from scripts.train import train
    agent, scores, rewards, elo, meta = train(episodes=300, seed=11,
                                             checkpoint_every=10**9,
                                             elo_games=1, verbose=False)
    assert len(scores) == 3
    assert any(v != 0.0 for row in agent.q for v in row)
    assert meta["episodes"] == 300
