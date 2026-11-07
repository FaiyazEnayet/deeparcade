"""Stage 3 deeparcade DA8: rebuild the DQN with a CNN architecture.

The flat MLP Pro scores 5.39 on the committed 500-episode evaluation
while Greedy scores 19.89 and Rookie scores 20.78. DA4 showed the MLP
memorizes flat pixels instead of learning spatial abstractions. These
tests pin the CNN replacement: spatial encoder layout, forward shapes,
deterministic checkpoint load and JSON/export parity, a NumPy mirror of
the forward pass, the held-constant training protocol, tiny end-to-end
training for both encoder candidates, split disjointness, and a legal
deterministic CNN policy action.
"""

import json

import numpy as np
import pytest
import torch

from narcade.experiment_seeds import (
    check_split_use,
    derive_key,
    episode_keys,
    training_env_key,
)
from narcade.models import (
    CNNCheckpointError,
    CNNDQN,
    CNNConfig,
    cnn_logits_numpy,
    encode_cnn_dqn,
    export_cnn_weights,
    load_cnn_checkpoint,
)
from narcade.snake import SnakeEnv
from scripts.compare_cnn_dqn import paired_difference
from scripts.evaluate import CNNDQNPolicy
from scripts.train_cnn_dqn import (
    BATCH,
    EPS_DECAY_STEPS,
    EPS_END,
    GAMMA,
    LR,
    REPLAY_MAX,
    TARGET_SYNC,
    UPDATE_EVERY,
    WARMUP_STEPS,
    train_cnn_dqn,
)
from scripts.train_dqn import DQN, encode_dqn


def _fixed_env(seed=3):
    env = SnakeEnv(seed=seed)
    env.reset()
    return env


def test_stage3_deeparcade_8_encoder_layout_matches_flat_encoder():
    env = _fixed_env()
    spatial, heading = encode_cnn_dqn(env)
    assert spatial.shape == (3, 12, 12)
    assert spatial.dtype == np.float32
    assert heading.shape == (4,)
    flat = encode_dqn(env)
    assert np.array_equal(spatial[0].reshape(-1), flat[0:144])
    assert np.array_equal(spatial[1].reshape(-1), flat[144:288])
    assert np.array_equal(spatial[2].reshape(-1), flat[288:432])
    assert np.array_equal(heading, flat[432:436])
    assert spatial.sum() == len(env.body) + 1  # head + body + food bits
    assert heading.sum() == 1.0
    assert heading[env.heading] == 1.0


def test_stage3_deeparcade_8_encoder_is_deterministic():
    env = _fixed_env()
    s1, h1 = encode_cnn_dqn(env)
    s2, h2 = encode_cnn_dqn(env)
    assert np.array_equal(s1, s2)
    assert np.array_equal(h1, h2)


def test_stage3_deeparcade_8_forward_shapes_both_candidates():
    torch.manual_seed(0)
    spatial = torch.randn(4, 3, 12, 12)
    heading = torch.zeros(4, 4)
    heading[:, 1] = 1.0
    for residual in (False, True):
        net = CNNDQN(CNNConfig(channels=(16, 32), residual=residual))
        out = net(spatial, heading)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()
        assert net.num_params() > 0
        desc = net.describe()
        assert desc["residual"] == residual
        assert desc["channels"] == [16, 32]
        assert desc["input_channels"] == 3
        assert desc["n_actions"] == 3


def test_stage3_deeparcade_8_checkpoint_round_trip_parity():
    torch.manual_seed(5)
    net = CNNDQN(CNNConfig(channels=(16, 32), residual=False))
    net.eval()
    spatial = torch.randn(6, 3, 12, 12)
    heading = torch.zeros(6, 4)
    heading[:, 2] = 1.0
    with torch.no_grad():
        before = net(spatial, heading).numpy()
    doc = export_cnn_weights(
        net,
        training_args={"steps": 0, "seed": 5},
        data_key_range=None,
        code_commit=None,
    )
    doc = json.loads(json.dumps(doc))  # JSON round trip, as stored on disk
    loaded, info = load_cnn_checkpoint(doc)
    assert info["schema_version"] == 1
    assert info["encoder"]["name"] == "encode_cnn_dqn"
    with torch.no_grad():
        after = loaded(spatial, heading).numpy()
    assert np.max(np.abs(before - after)) == 0.0
    # Deterministic load: a second load gives identical weights.
    loaded2, _ = load_cnn_checkpoint(json.loads(json.dumps(doc)))
    assert torch.equal(
        torch.cat([p.flatten() for p in loaded.parameters()]),
        torch.cat([p.flatten() for p in loaded2.parameters()]),
    )


def test_stage3_deeparcade_8_checkpoint_rejects_tampering():
    torch.manual_seed(5)
    net = CNNDQN(CNNConfig(channels=(16, 32), residual=False))
    doc = export_cnn_weights(net)
    bad = json.loads(json.dumps(doc))
    bad["layers"][0]["values"][0][0][0][0] += 1.0
    with pytest.raises(CNNCheckpointError):
        load_cnn_checkpoint(bad)
    wrong_kind = json.loads(json.dumps(doc))
    wrong_kind["kind"] = "deeparcade-dqn-checkpoint"
    with pytest.raises(CNNCheckpointError):
        load_cnn_checkpoint(wrong_kind)
    wrong_encoder = json.loads(json.dumps(doc))
    wrong_encoder["encoder"]["name"] = "encode_dqn"
    with pytest.raises(CNNCheckpointError):
        load_cnn_checkpoint(wrong_encoder)


def test_stage3_deeparcade_8_numpy_mirror_matches_torch():
    torch.manual_seed(11)
    net = CNNDQN(CNNConfig(channels=(16, 32), residual=True))
    net.eval()
    doc = export_cnn_weights(net)
    rng = np.random.default_rng(4)
    spatial = rng.integers(0, 2, size=(5, 3, 12, 12)).astype(np.float32)
    heading = np.zeros((5, 4), dtype=np.float32)
    heading[:, 0] = 1.0
    with torch.no_grad():
        torch_logits = net(
            torch.from_numpy(spatial), torch.from_numpy(heading)
        ).numpy()
    mirror = cnn_logits_numpy(doc, spatial, heading)
    assert mirror.shape == (5, 3)
    assert np.max(np.abs(torch_logits - mirror)) < 1e-5
    assert np.array_equal(
        np.argmax(torch_logits, axis=1), np.argmax(mirror, axis=1)
    )


def test_stage3_deeparcade_8_training_constants_match_mlp():
    # Held constant across the MLP and CNN arms: same reward shaping
    # (SnakeEnv.step, shared), same optimizer family (Adam), same gamma,
    # batch, learning rate, replay capacity, update cadence, target sync,
    # epsilon schedule, and warmup. The values below are the literals in
    # scripts/train_dqn.py train_dqn; a drift here fails this test.
    assert GAMMA == 0.95
    assert BATCH == 64
    assert LR == 1e-3
    assert REPLAY_MAX == 50000
    assert UPDATE_EVERY == 8
    assert TARGET_SYNC == 2000
    assert EPS_END == 0.05
    assert EPS_DECAY_STEPS == 120000
    assert WARMUP_STEPS == 2000


def test_stage3_deeparcade_8_tiny_trains_both_candidates():
    for residual in (False, True):
        net, scores, rewards, meta = train_cnn_dqn(
            steps=1500,
            seed=123,
            config=CNNConfig(channels=(16, 32), residual=residual),
            verbose=False,
        )
        assert meta["episodes"] > 0
        assert meta["steps"] == 1500
        assert meta["seed"] == 123
        assert meta["seconds"] > 0
        assert meta["params"] == net.num_params()
        assert all(s >= 0 for s in scores)
        assert all(np.isfinite(r) for r in rewards)
        assert any(p.abs().sum().item() > 0 for p in net.parameters())


def test_stage3_deeparcade_8_split_disjointness():
    # Training env seeds are base seed + episode index (legacy
    # derivation, P0-06). Test episodes use SHA-256-derived keys from
    # the "test" namespace. The two sets must never intersect.
    train_seeds = {training_env_key(7, i) for i in range(1, 7001)}
    train_seeds |= {training_env_key(8, i) for i in range(1, 7001)}
    train_seeds |= {training_env_key(9, i) for i in range(1, 7001)}
    test_keys = episode_keys("test", 0, 200)
    test_env_keys = {k.env_key for k in test_keys}
    assert train_seeds.isdisjoint(test_env_keys)
    assert derive_key("test", 0, "env") != derive_key("train", 0, "env")
    check_split_use("test", "final")
    with pytest.raises(ValueError):
        check_split_use("train", "final")


def test_stage3_deeparcade_8_paired_difference_known_answer():
    stats = paired_difference([1, 2, 3], [1, 1, 1])
    assert stats["n"] == 3
    assert stats["mean_diff"] == pytest.approx(1.0)
    assert stats["wins"] == 2
    assert stats["draws"] == 1
    assert stats["losses"] == 0
    assert stats["ci_low"] < stats["mean_diff"] < stats["ci_high"]
    assert stats["ci_width"] > 0


def test_stage3_deeparcade_8_cnn_policy_act_is_legal_and_deterministic():
    net, _, _, _ = train_cnn_dqn(
        steps=1500,
        seed=123,
        config=CNNConfig(channels=(16, 32), residual=False),
        verbose=False,
    )
    doc = export_cnn_weights(net)
    pol1 = CNNDQNPolicy(doc)
    pol2 = CNNDQNPolicy(json.loads(json.dumps(doc)))
    env = _fixed_env()
    actions1 = [pol1.act(env) for _ in range(5)]
    env2 = _fixed_env()
    actions2 = [pol2.act(env2) for _ in range(5)]
    assert actions1 == actions2
    assert all(a in (0, 1, 2) for a in actions1)
