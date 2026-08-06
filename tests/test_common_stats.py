"""Tests for shared distribution statistics (src/common/stats.py)."""

import torch

from common.stats import (
    excess_kurtosis,
    excess_kurtosis_tensor,
    skewness,
    skewness_tensor,
    terminal_log_return,
)


def test_skewness_tensor_matches_float_version() -> None:
    x = torch.randn(500)
    assert abs(skewness_tensor(x).item() - skewness(x)) < 1e-6


def test_excess_kurtosis_tensor_matches_float_version() -> None:
    x = torch.randn(500)
    assert abs(excess_kurtosis_tensor(x).item() - excess_kurtosis(x)) < 1e-6


def test_skewness_tensor_is_differentiable() -> None:
    x = torch.randn(200, requires_grad=True)
    loss = skewness_tensor(x) ** 2
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_excess_kurtosis_tensor_is_differentiable() -> None:
    x = torch.randn(200, requires_grad=True)
    loss = excess_kurtosis_tensor(x) ** 2
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_terminal_log_return_matches_manual_calc() -> None:
    prices = torch.tensor([[[100.0], [105.0], [110.0]], [[50.0], [45.0], [40.0]]])
    returns = terminal_log_return(prices)
    expected = torch.log(torch.tensor([110.0 / 100.0, 40.0 / 50.0]))
    assert torch.allclose(returns, expected)
