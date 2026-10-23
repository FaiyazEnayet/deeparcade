"""Unit tests for the pass/fail gates of tools/native_dqn_check.py.

The checker is a driver script, not a package module, and it imports
torch at module level for its reference model. assess() is pure, so
these tests stub the heavy imports (torch, and the modules the
script imports names from) before loading the script by file path;
the gates are then exercised on synthetic inputs. This keeps the
gate logic testable on any interpreter, with or without torch.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


def _load_checker():
    _stub("torch")
    _stub("scripts")
    _stub("scripts.train_dqn", DQN=object, encode_dqn=lambda env: None)
    _stub("narcade")
    _stub("narcade.policies", GreedyPolicy=object, QTablePolicy=object,
          RandomPolicy=object)
    _stub("narcade.snake", SnakeEnv=object)
    spec = importlib.util.spec_from_file_location(
        "native_dqn_check", ROOT / "tools" / "native_dqn_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


assess = _load_checker().assess


def _disagree(margin, diff=0.0):
    # (state index, cpp action, ref action, ref margin, state logit diff)
    return (0, 1, 2, margin, diff)


def test_clean_inputs_pass():
    hard_fail, reasons = assess(0, [], 9.5e-6)
    assert hard_fail is False
    assert reasons == []


def test_logit_diff_above_band_fails_without_disagreements():
    # Regression test: the max logit difference used to be reported
    # but never enforced as a gate.
    hard_fail, reasons = assess(0, [], 2e-5)
    assert hard_fail is True
    assert any("logit band" in r for r in reasons)


def test_nonfinite_logit_diff_fails():
    for bad in (float("nan"), float("inf")):
        hard_fail, reasons = assess(0, [], bad)
        assert hard_fail is True
        assert reasons


def test_disagreement_above_margin_band_fails():
    hard_fail, reasons = assess(0, [_disagree(2e-5)], 1e-6)
    assert hard_fail is True
    assert reasons


def test_disagreement_within_margin_band_passes():
    hard_fail, reasons = assess(0, [_disagree(5e-6, diff=1e-6)], 1e-6)
    assert hard_fail is False
    assert reasons == []


def test_encoding_mismatch_fails():
    hard_fail, reasons = assess(1, [], 0.0)
    assert hard_fail is True
    assert reasons
