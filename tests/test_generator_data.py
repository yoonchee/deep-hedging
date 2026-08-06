"""Tests for the historical price-window sampler (src/generator/data.py).

These test the pure windowing/normalization logic only -- no network calls,
so the suite stays fast, deterministic, and runnable offline.
"""

import pytest
import torch

from generator.data import sample_price_windows


def test_sample_price_windows_output_shape() -> None:
    prices = torch.linspace(10.0, 20.0, steps=200)
    batch_size, seq_len = 32, 30

    windows = sample_price_windows(prices, batch_size, seq_len)

    assert windows.shape == (batch_size, seq_len, 1)


def test_sample_price_windows_start_at_initial_price() -> None:
    prices = torch.linspace(10.0, 20.0, steps=200)
    initial_price = 2.5

    windows = sample_price_windows(prices, batch_size=16, seq_len=10, initial_price=initial_price)

    # [Batch, Time_Steps, 1] -> [Batch] (first observation of each window)
    first_values = windows[:, 0, 0]
    assert torch.allclose(first_values, torch.full_like(first_values, initial_price), atol=1e-4)


def test_sample_price_windows_preserves_relative_shape() -> None:
    # A strictly increasing series: every sampled window must itself be
    # strictly increasing after normalization (rescaling doesn't reorder).
    prices = torch.arange(1.0, 201.0)

    windows = sample_price_windows(prices, batch_size=8, seq_len=15)

    diffs = windows[:, 1:, 0] - windows[:, :-1, 0]
    assert torch.all(diffs > 0)


def test_sample_price_windows_rejects_seq_len_longer_than_history() -> None:
    prices = torch.linspace(10.0, 20.0, steps=10)

    with pytest.raises(ValueError):
        sample_price_windows(prices, batch_size=4, seq_len=20)


def test_sample_price_windows_reproducible_with_generator() -> None:
    prices = torch.linspace(10.0, 20.0, steps=500)

    windows_a = sample_price_windows(
        prices, batch_size=8, seq_len=10, generator=torch.Generator().manual_seed(0)
    )
    windows_b = sample_price_windows(
        prices, batch_size=8, seq_len=10, generator=torch.Generator().manual_seed(0)
    )

    assert torch.equal(windows_a, windows_b)
