"""Tests for shared distribution statistics (src/common/stats.py)."""

import pytest
import torch

from common.stats import (
    excess_kurtosis,
    excess_kurtosis_tensor,
    lag1_autocorrelation,
    skewness,
    skewness_tensor,
    step_log_returns,
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


def test_step_log_returns_matches_manual_calc() -> None:
    prices = torch.tensor([[[100.0], [105.0], [110.0]], [[50.0], [45.0], [40.0]]])
    returns = step_log_returns(prices)
    expected = torch.log(torch.tensor([[105.0 / 100.0, 110.0 / 105.0], [45.0 / 50.0, 40.0 / 45.0]]))
    assert torch.allclose(returns, expected)


def test_step_log_returns_shape_is_time_minus_one() -> None:
    prices = torch.rand(8, 31, 1) + 0.5
    assert step_log_returns(prices).shape == (8, 30)


def test_lag1_autocorrelation_is_one_for_a_perfectly_linear_trend() -> None:
    # x[:, t] = t exactly -> x[:, :-1] and x[:, 1:] are perfectly linearly
    # related (a shifted copy of the same arithmetic sequence), so Pearson
    # correlation must be exactly 1.0 for every path.
    x = torch.arange(10, dtype=torch.float32).unsqueeze(0).repeat(5, 1)
    assert lag1_autocorrelation(x) == pytest.approx(1.0, abs=1e-5)


def test_lag1_autocorrelation_is_near_zero_for_iid_noise() -> None:
    torch.manual_seed(0)
    x = torch.randn(2000, 30)  # large batch/length -> tight estimate
    assert abs(lag1_autocorrelation(x)) < 0.05


def test_lag1_autocorrelation_detects_alternating_sign_as_negative() -> None:
    # x alternates +1,-1,+1,-1,... -- each step is the exact negation of the
    # previous one, so lag-1 correlation must be exactly -1.0.
    x = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]).repeat(4, 1)
    assert lag1_autocorrelation(x) == pytest.approx(-1.0, abs=1e-5)


def test_lag1_autocorrelation_is_nan_for_a_fully_degenerate_batch() -> None:
    x = torch.ones(5, 10)  # every path constant -> zero variance everywhere
    assert lag1_autocorrelation(x) != lag1_autocorrelation(x)  # NaN != NaN
