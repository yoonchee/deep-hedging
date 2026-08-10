"""Tests for shared training-device selection (src/common/device.py).

Mocks torch.cuda.is_available/torch.backends.mps.is_available so the
priority logic (cuda > mps > cpu) is verified deterministically, independent
of what hardware actually runs the test suite.
"""

from common.device import select_device


def test_explicit_override_bypasses_auto_detection(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert select_device("cpu").type == "cpu"


def test_cuda_preferred_over_mps_when_both_available(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert select_device().type == "cuda"


def test_mps_preferred_over_cpu_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert select_device().type == "mps"


def test_falls_back_to_cpu_when_nothing_else_available(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert select_device().type == "cpu"
