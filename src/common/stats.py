"""Shared distribution statistics (src/common/stats.py).

Used across the backtester, the GAN fidelity check, and the Part I
replication study -- anywhere a PnL or return distribution's shape needs
summarizing.
"""

from typing import Annotated

import torch


def skewness_tensor(x: Annotated[torch.Tensor, "[Batch] samples"]) -> Annotated[torch.Tensor, "scalar, differentiable"]:
    centered = x - x.mean()
    return centered.pow(3).mean() / centered.pow(2).mean().pow(1.5)


def excess_kurtosis_tensor(
    x: Annotated[torch.Tensor, "[Batch] samples"]
) -> Annotated[torch.Tensor, "scalar, differentiable"]:
    centered = x - x.mean()
    return centered.pow(4).mean() / centered.pow(2).mean().pow(2) - 3.0


def skewness(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    return skewness_tensor(x).item()


def excess_kurtosis(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    return excess_kurtosis_tensor(x).item()


def terminal_log_return(
    prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price paths"]
) -> Annotated[torch.Tensor, "[Batch] log(S_T / S_0), differentiable"]:
    s0 = prices[:, 0, 0]
    sT = prices[:, -1, 0]
    return torch.log(sT / s0)


def step_log_returns(
    prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price paths"]
) -> Annotated[torch.Tensor, "[Batch, Time_Steps - 1] log(S_t / S_{t-1}) per step, differentiable"]:
    return torch.log(prices[:, 1:, 0] / prices[:, :-1, 0])


def lag1_autocorrelation(
    x: Annotated[torch.Tensor, "[Batch, Time] per-path series, e.g. step_log_returns(...)"]
) -> Annotated[
    float,
    "per-path Pearson correlation between x[:, :-1] and x[:, 1:], averaged "
    "across the batch; NaN if every path is degenerate (zero variance)",
]:
    """Vectorized lag-1 autocorrelation -- no per-path Python loop, so this
    stays fast at the batch sizes validate_generator_fidelity runs at
    (thousands of paths). Used both on raw (signed) returns, where a
    nonzero value flags momentum/mean-reversion structure a terminal-only
    check can't see, and on |returns|, where it flags volatility clustering
    (ARCH effects) -- see RESULTS.md's "Investigating why the best-fidelity
    generator produced the worst policies" writeup, which found a
    terminal-distribution-"OK" TimeGAN generator with per-step dynamics
    (2x real volatility, much stronger momentum and clustering) invisible
    to every check that only inspects the terminal/cumulative distribution.
    """
    a, b = x[:, :-1], x[:, 1:]
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    den = (a_c.pow(2).sum(dim=1)).sqrt() * (b_c.pow(2).sum(dim=1)).sqrt()
    valid = den > 1e-8
    if valid.sum() == 0:
        return float("nan")
    return (num[valid] / den[valid]).mean().item()


def lag1_autocorrelation_tensor(
    x: Annotated[torch.Tensor, "[Batch, Time] per-path series, e.g. step_log_returns(...)"]
) -> Annotated[
    torch.Tensor,
    "scalar, differentiable version of lag1_autocorrelation -- for use directly "
    "inside a training loss (see train_timegan.py's path-dynamics-matching loss). "
    "Not a thin wrapper around lag1_autocorrelation: falls back to 0.0 (not NaN) "
    "for a fully degenerate batch, since NaN would poison every downstream "
    "gradient rather than just leave this one term's contribution undefined -- "
    "this divergence is intentional, not an inconsistency.",
]:
    a, b = x[:, :-1], x[:, 1:]
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    den = (a_c.pow(2).sum(dim=1)).sqrt() * (b_c.pow(2).sum(dim=1)).sqrt()
    valid = den > 1e-8
    if valid.sum() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return (num[valid] / den[valid]).mean()
