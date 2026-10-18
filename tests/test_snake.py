"""Game-logic tests: movement, food, collisions, state encoding."""

import pytest
from collections import deque

from narcade.snake import SnakeEnv, STRAIGHT, LEFT, RIGHT


def test_reset_gives_valid_state():
    env = SnakeEnv(seed=1)
    s = env.reset()
    assert 0 <= s < 32
    assert len(env.body) == 3
    assert env.score == 0
    assert not env.done


def test_straight_moves_forward():
    env = SnakeEnv(seed=1)
    env.reset()
    head_before = env.body[-1]
    env.step(STRAIGHT)
    hr, hc = head_before
    assert env.body[-1] == (hr, hc + 1)  # heading starts right


def test_relative_turns():
    env = SnakeEnv(seed=1)
    env.reset()
    assert env.heading == 1
    env.step(LEFT)
    assert env.heading == 0  # right -> up
    env2 = SnakeEnv(seed=1)
    env2.reset()
    env2.step(RIGHT)
    assert env2.heading == 2  # right -> down


def test_wall_death():
    env = SnakeEnv(seed=1, width=5, height=5)
    env.reset()
    # drive into the right wall
    done = False
    for _ in range(10):
        _, _, done, info = env.step(STRAIGHT)
        if done:
            break
    assert done
    assert info["reason"] == "death"


def test_food_eating_grows_snake():
    env = SnakeEnv(seed=1)
    env.reset()
    # place food directly ahead
    hr, hc = env.body[-1]
    env.food = (hr, hc + 1)
    n_before = len(env.body)
    _, reward, done, _ = env.step(STRAIGHT)
    assert not done
    assert len(env.body) == n_before + 1
    assert env.score == 1
    assert reward == pytest.approx(10.0)


def test_food_respawns_on_free_cell():
    env = SnakeEnv(seed=2)
    env.reset()
    hr, hc = env.body[-1]
    env.food = (hr, hc + 1)
    env.step(STRAIGHT)
    assert env.food not in set(env.body)


def test_state_encoding_bounds_and_determinism():
    env = SnakeEnv(seed=3)
    env.reset()
    for _ in range(30):
        s1 = env.encode_state()
        assert 0 <= s1 < 32
        _, _, done, _ = env.step(STRAIGHT)
        if done:
            break
    # determinism: same seed -> same trajectory
    a = SnakeEnv(seed=9)
    b = SnakeEnv(seed=9)
    a.reset(); b.reset()
    for _ in range(20):
        assert a.encode_state() == b.encode_state()
        _, _, da, _ = a.step(STRAIGHT)
        _, _, db, _ = b.step(STRAIGHT)
        assert da == db
        if da:
            break


def test_shaping_reward_sign():
    env = SnakeEnv(seed=4)
    env.reset()
    hr, hc = env.body[-1]
    # food far ahead-right; moving straight (right) should approach or keep distance
    env.food = (hr, hc + 5)
    _, r, done, _ = env.step(STRAIGHT)
    assert not done
    assert r == pytest.approx(0.5)  # closer


def test_step_on_done_raises():
    env = SnakeEnv(seed=1, width=4, height=4)
    env.reset()
    for _ in range(20):
        _, _, done, _ = env.step(STRAIGHT)
        if done:
            break
    assert done
    with pytest.raises(RuntimeError):
        env.step(STRAIGHT)


def test_bad_action_raises():
    env = SnakeEnv(seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_filled_board_is_win_not_crash():
    """Eating the last food on a full board ends the episode as a win."""
    env = SnakeEnv(width=2, height=3, seed=1)
    env.reset()
    # 5-segment snake, one free cell (2,1) holding the food.
    env.body = deque([(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])
    env.heading = 1  # right; head (2,0) -> (2,1)
    env.food = (2, 1)
    env.score = 4
    env.done = False
    state, reward, done, info = env.step(STRAIGHT)
    assert done is True
    assert info["reason"] == "win"
    assert env.score == 5
    assert len(env.body) == 6  # board completely filled
    assert 0 <= state < 32  # terminal observation is well-defined


def test_step_after_win_raises():
    env = SnakeEnv(width=2, height=3, seed=1)
    env.reset()
    env.body = deque([(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)])
    env.heading = 1
    env.food = (2, 1)
    env.done = False
    _, _, done, info = env.step(STRAIGHT)
    assert done and info["reason"] == "win"
    with pytest.raises(RuntimeError):
        env.step(STRAIGHT)
