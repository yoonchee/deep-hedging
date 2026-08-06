"""Real historical market data for training the WGAN-GP market generator.

Downloads OHLCV data via yfinance -- Open, High, Low, Close, Adj Close,
Volume, the same six-feature schema described in Kim (2021) "Deep Hedging,
Generative Adversarial Networks, and Beyond" -- and exposes it as
normalized price-path windows, replacing the synthetic GBM placeholder
previously used in train_gan.py's `sample_real_prices()`.
"""

import math
from pathlib import Path
from typing import Annotated, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn


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


def sample_multivariate_price_windows(
    features: Annotated[torch.Tensor, "[N, F] historical multi-feature series (e.g. O,H,L,C,V)"],
    batch_size: Annotated[int, "number of paths to sample"],
    seq_len: Annotated[int, "number of time observations per path"],
    price_index: Annotated[int, "feature column whose initial window value anchors the rescaling"],
    rescale_indices: Annotated[
        Sequence[int],
        "feature columns rescaled by the same divisor as price_index (e.g. Open/High/Low/Close); "
        "columns not listed (e.g. Volume, a different unit entirely) are left on their own scale",
    ],
    initial_price: Annotated[float, "value the price_index column starts each window at"] = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, F] windows; rescale_indices columns normalized"]:
    """Multi-feature generalization of `sample_price_windows`, for TimeGAN.

    Every rescaled column in a window is divided by the SAME divisor (the
    window's own initial price_index value, e.g. Close_0) rather than each
    column's own start value -- this preserves intra-window relationships
    like High >= Close that per-column normalization would destroy (every
    column would independently start at exactly `initial_price`).
    """
    n, feature_dim = features.shape
    if seq_len > n:
        raise ValueError(f"seq_len={seq_len} exceeds available history length {n}")

    max_start = n - seq_len
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)

    # [Batch, Time_Steps, F]
    windows = torch.stack([features[s : s + seq_len] for s in starts.tolist()])

    # [Batch, 1, 1] -> [Batch, Time_Steps, F] (each window's own initial price_index value)
    divisor = windows[:, :1, price_index : price_index + 1] / initial_price

    rescale_mask = torch.zeros(feature_dim, dtype=torch.bool)
    rescale_mask[list(rescale_indices)] = True

    # [F] -> [Batch, Time_Steps, F] (only rescale_indices columns divided by the shared divisor)
    return torch.where(rescale_mask, windows / divisor, windows)


class MinMaxScaler(nn.Module):
    """Per-feature min-max scaling to [0, 1] -- what TimeGAN's Embedder/Recovery
    sigmoid output assumes of its input (see generator/timegan.py).

    A plain nn.Module (buffers, not parameters) so the fitted min/max travel
    with .to(device) and the checkpoint's state_dict automatically.
    """

    def __init__(self, feature_dim: Annotated[int, "number of features F"]) -> None:
        super().__init__()
        self.register_buffer("min_vals", torch.zeros(feature_dim))
        self.register_buffer("max_vals", torch.ones(feature_dim))

    def fit(self, x: Annotated[torch.Tensor, "[Batch, Time_Steps, F] sample to fit min/max on"]) -> None:
        # [Batch, Time_Steps, F] -> [Batch * Time_Steps, F]
        flat = x.reshape(-1, x.shape[-1])
        self.min_vals = flat.min(dim=0).values
        self.max_vals = flat.max(dim=0).values

    def transform(
        self, x: Annotated[torch.Tensor, "[..., F] raw-scale values"]
    ) -> Annotated[torch.Tensor, "[..., F] scaled to [0, 1]"]:
        span = (self.max_vals - self.min_vals).clamp(min=1e-8)
        return (x - self.min_vals) / span

    def inverse_transform(
        self, x: Annotated[torch.Tensor, "[..., F] values in [0, 1]"]
    ) -> Annotated[torch.Tensor, "[..., F] raw-scale values"]:
        span = (self.max_vals - self.min_vals).clamp(min=1e-8)
        return x * span + self.min_vals


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
        self.df = df
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

    def sample_multivariate(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time observations per path"],
        feature_columns: Annotated[
            Sequence[str], "OHLCV columns to use, in output-channel order"
        ] = ("Open", "High", "Low", "Close", "Volume"),
        price_column: Annotated[str, "which feature column is 'the price'"] = "Close",
        rescale_columns: Annotated[
            Sequence[str], "columns rescaled by the shared price_column divisor (Volume is excluded)"
        ] = ("Open", "High", "Low", "Close"),
        initial_price: Annotated[float, "value the price column starts each window at"] = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, F] normalized real multi-feature paths"]:
        """Six-column OHLCV is reduced to 5 here (no Adjusted Close): for this
        loader's tickers (pure indices like ^GSPC), Adj Close == Close always
        (documented in RESULTS.md's known limitations), so a 6th column would
        just be a perfectly-correlated duplicate of Close.
        """
        feature_columns = list(feature_columns)
        subset = self.df[feature_columns].dropna()
        if len(subset) < seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds available history length {len(subset)}")

        features = torch.tensor(subset.to_numpy(dtype="float64"), dtype=torch.float32)
        price_index = feature_columns.index(price_column)
        rescale_indices = [feature_columns.index(c) for c in rescale_columns]

        return sample_multivariate_price_windows(
            features,
            batch_size,
            seq_len,
            price_index=price_index,
            rescale_indices=rescale_indices,
            initial_price=initial_price,
            generator=generator,
        )
