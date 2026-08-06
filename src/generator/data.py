"""Real historical market data for training the WGAN-GP market generator.

Downloads OHLCV data via yfinance -- Open, High, Low, Close, Adj Close,
Volume, the same six-feature schema described in Kim (2021) "Deep Hedging,
Generative Adversarial Networks, and Beyond" -- and exposes it as
normalized price-path windows, replacing the synthetic GBM placeholder
previously used in train_gan.py's `sample_real_prices()`.
"""

import math
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import torch


def download_or_load_ohlcv(
    ticker: Annotated[str, "Yahoo Finance ticker, e.g. '^GSPC' for the S&P 500 index"],
    start: Annotated[str, "start date, 'YYYY-MM-DD'"],
    end: Annotated[str, "end date, 'YYYY-MM-DD'"],
    cache_path: Annotated[Path, "local CSV cache path"],
) -> pd.DataFrame:
    """Loads OHLCV data from a local CSV cache, downloading via yfinance if absent."""
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Recent yfinance versions return (Price, Ticker) MultiIndex columns
        # even for a single ticker; flatten to the paper's plain six columns.
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"yfinance returned no data for ticker={ticker!r} in [{start}, {end}]")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def sample_price_windows(
    prices: Annotated[torch.Tensor, "[N] 1D historical price series"],
    batch_size: Annotated[int, "number of price paths to sample"],
    seq_len: Annotated[int, "number of price observations per path"],
    initial_price: Annotated[float, "S_0 each sampled window is rescaled to"] = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] normalized real price paths"]:
    """Samples random contiguous windows from a historical price series.

    Each window is rescaled to start at `initial_price`: the market
    Generator (market_gan.py) always starts synthetic paths at a fixed S_0,
    so "real" and "fake" batches must share that convention for the WGAN-GP
    critic to learn anything meaningful. Rescaling also removes the raw
    index's huge level drift (e.g. ~17 in 1950 vs. ~3800 in 2021 for the
    S&P 500) so the network sees comparable-scale return paths throughout.
    """
    n = prices.shape[0]
    if seq_len > n:
        raise ValueError(f"seq_len={seq_len} exceeds available history length {n}")

    max_start = n - seq_len
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)

    # [Batch, Time_Steps]
    windows = torch.stack([prices[s : s + seq_len] for s in starts.tolist()])

    # [Batch, Time_Steps] -> [Batch, Time_Steps] (each window starts at initial_price)
    normalized = windows / windows[:, :1] * initial_price

    # [Batch, Time_Steps] -> [Batch, Time_Steps, 1]
    return normalized.unsqueeze(-1)


def sample_real_prices(
    batch_size: Annotated[int, "number of price paths"],
    seq_len: Annotated[int, "number of price observations per path"],
    s0: Annotated[float, "initial asset price S_0"] = 1.0,
    vol: Annotated[float, "single-regime GBM volatility"] = 0.2,
    dt: Annotated[float, "time increment per step"] = 1.0,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] strictly positive 'real' price paths"]:
    """Single-regime GBM 'real' market data.

    Offline, deterministic fallback data source (train_gan.py's
    --data-source synthetic) for quick smoke tests when no network is
    available. For actual historical data, see `HistoricalPriceLoader`
    (--data-source yfinance).
    """
    # [Batch, Time_Steps - 1] i.i.d. standard normal shocks
    z = torch.randn(batch_size, seq_len - 1)

    # [Batch, Time_Steps - 1] -> [Batch, Time_Steps] (cumulative log-return, S_0 fixed)
    log_returns = -0.5 * vol**2 * dt + vol * math.sqrt(dt) * z
    log_prices = torch.cat(
        [torch.zeros(batch_size, 1), torch.cumsum(log_returns, dim=1)], dim=1
    )

    # [Batch, Time_Steps] -> [Batch, Time_Steps, 1]
    prices = s0 * torch.exp(log_prices)
    return prices.unsqueeze(-1)


class HistoricalPriceLoader:
    """Downloads (or loads a cached copy of) a real price series and samples
    normalized training windows from it for the WGAN-GP market generator.
    """

    def __init__(
        self,
        ticker: Annotated[str, "Yahoo Finance ticker"] = "^GSPC",
        start: Annotated[str, "start date, 'YYYY-MM-DD'"] = "1950-01-03",
        end: Annotated[str, "end date, 'YYYY-MM-DD'"] = "2021-01-25",
        price_column: Annotated[str, "which OHLCV column to use as the price series"] = "Adj Close",
        cache_dir: Annotated[Path, "directory for the local CSV cache"] = Path("data"),
    ) -> None:
        safe_ticker = ticker.replace("^", "").replace("/", "_")
        cache_path = Path(cache_dir) / f"{safe_ticker}.csv"
        df = download_or_load_ohlcv(ticker, start, end, cache_path)

        if price_column not in df.columns:
            raise KeyError(f"{price_column!r} not in downloaded columns: {list(df.columns)}")

        prices = df[price_column].dropna().to_numpy(dtype="float64")
        if len(prices) < 2:
            raise ValueError(f"Not enough data points for ticker={ticker!r}: {len(prices)}")

        self.ticker = ticker
        self.cache_path = cache_path
        self.prices = torch.tensor(prices, dtype=torch.float32)

    def sample(
        self,
        batch_size: Annotated[int, "number of price paths to sample"],
        seq_len: Annotated[int, "number of price observations per path"],
        initial_price: Annotated[float, "S_0 each sampled window is rescaled to"] = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] normalized real price paths"]:
        return sample_price_windows(
            self.prices, batch_size, seq_len, initial_price=initial_price, generator=generator
        )
