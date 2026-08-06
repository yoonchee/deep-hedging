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
