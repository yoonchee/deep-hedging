"""Tests for the historical price-window sampler (src/generator/data.py).

These test the pure windowing/normalization logic only -- no network calls,
so the suite stays fast, deterministic, and runnable offline.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from generator.data import HistoricalPriceLoader, sample_multivariate_price_windows, sample_price_windows


@pytest.mark.skipif(
    not (Path("data") / "GSPC.csv").exists(),
    reason="requires the cached ^GSPC CSV (data/ is gitignored, not present on a fresh clone/CI)",
)
def test_historical_price_loader_honors_date_range_on_cache_hit() -> None:
    # download_or_load_ohlcv only applies [start, end] on a *fresh* download;
    # a cache hit used to return the whole cached file regardless of the
    # requested range. This is what a genuine temporal train/test split
    # (two loaders, non-overlapping ranges, same cache file) depends on.
    early = HistoricalPriceLoader(start="1950-01-03", end="1960-01-01")
    late = HistoricalPriceLoader(start="2015-01-01", end="2021-01-25")

    assert early.df.index.max() < pd.Timestamp("1960-01-01")
    assert late.df.index.min() >= pd.Timestamp("2015-01-01")
    assert len(early.prices) < len(late.prices) or early.df.index.max() < late.df.index.min()
    assert early.df.index.max() < late.df.index.min()


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


def _synthetic_ohlcv(n: int) -> torch.Tensor:
    # [N] -> [N, 4] (Open, High, Low, Close all track a shared trend) + [N, 1] Volume, unrelated scale
    close = torch.linspace(10.0, 20.0, steps=n)
    open_ = close - 0.1
    high = close + 0.5
    low = close - 0.5
    volume = torch.linspace(1_000_000.0, 2_000_000.0, steps=n)
    return torch.stack([open_, high, low, close, volume], dim=-1)


def test_sample_multivariate_price_windows_output_shape() -> None:
    features = _synthetic_ohlcv(200)
    batch_size, seq_len = 16, 30

    windows = sample_multivariate_price_windows(
        features, batch_size, seq_len, price_index=3, rescale_indices=[0, 1, 2, 3]
    )

    assert windows.shape == (batch_size, seq_len, 5)


def test_sample_multivariate_price_windows_price_column_starts_at_initial_price() -> None:
    features = _synthetic_ohlcv(200)
    initial_price = 2.5

    windows = sample_multivariate_price_windows(
        features, batch_size=16, seq_len=10, price_index=3, rescale_indices=[0, 1, 2, 3], initial_price=initial_price
    )

    # [Batch, Time_Steps, F] -> [Batch] (Close column, first observation)
    first_close = windows[:, 0, 3]
    assert torch.allclose(first_close, torch.full_like(first_close, initial_price), atol=1e-4)


def test_sample_multivariate_price_windows_preserves_intra_window_relationships() -> None:
    # High >= Close >= Low must survive rescaling, since all three share the same divisor.
    features = _synthetic_ohlcv(200)

    windows = sample_multivariate_price_windows(
        features, batch_size=16, seq_len=10, price_index=3, rescale_indices=[0, 1, 2, 3]
    )

    high, low, close = windows[..., 1], windows[..., 2], windows[..., 3]
    assert torch.all(high >= close) and torch.all(close >= low)


def test_sample_multivariate_price_windows_leaves_unrescaled_columns_on_raw_scale() -> None:
    features = _synthetic_ohlcv(200)

    windows = sample_multivariate_price_windows(
        features, batch_size=16, seq_len=10, price_index=3, rescale_indices=[0, 1, 2, 3], initial_price=1.0
    )

    # Volume (index 4) isn't in rescale_indices -- values should match the
    # original raw scale (millions), not be pulled down near 1.0 like Close.
    assert windows[..., 4].min() > 1000.0


def test_sample_multivariate_price_windows_rejects_seq_len_longer_than_history() -> None:
    features = _synthetic_ohlcv(10)

    with pytest.raises(ValueError):
        sample_multivariate_price_windows(features, batch_size=4, seq_len=20, price_index=3, rescale_indices=[0, 1, 2, 3])
