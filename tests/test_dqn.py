"""DQN smoke test at tiny scale."""

import torch

from scripts.train_dqn import DQN, INPUT_DIM, encode_dqn, export_weights, train_dqn
from narcade.snake import SnakeEnv


def test_dqn_forward_shape():
    net = DQN()
    x = torch.randn(4, INPUT_DIM)
    out = net(x)
    assert out.shape == (4, 3)


def test_encode_dqn_layout():
    env = SnakeEnv(seed=1)
    env.reset()
    v = encode_dqn(env)
    assert len(v) == INPUT_DIM
    hr, hc = env.body[-1]
    assert v[hr * 12 + hc] == 1.0  # head channel
    fr, fc = env.food
    assert v[288 + fr * 12 + fc] == 1.0  # food channel
    assert v[432 + env.heading] == 1.0  # heading one-hot
    assert v.sum() == len(env.body) + 2  # head+body+food+heading bits


def test_export_weights_structure():
    w = export_weights(DQN())
    assert w["input_dim"] == INPUT_DIM
    assert len(w["layers"]) == 3
    assert len(w["layers"][0]["w"]) == 128
    assert len(w["layers"][0]["w"][0]) == INPUT_DIM


def test_tiny_dqn_trains():
    net, scores, rewards, meta = train_dqn(steps=1500, seed=123, verbose=False)
    assert meta["episodes"] > 0
    assert all(s >= 0 for s in scores)
    # weights actually moved
    assert any(p.abs().sum().item() > 0 for p in net.parameters())
