"""Shared distribution statistics (src/common/stats.py).

Used across the backtester, the GAN fidelity check, and the Part I
replication study -- anywhere a PnL or return distribution's shape needs
summarizing.
"""

from typing import Annotated

import torch


def skewness(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    centered = x - x.mean()
    return (centered.pow(3).mean() / centered.pow(2).mean().pow(1.5)).item()


def excess_kurtosis(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    centered = x - x.mean()
    return (centered.pow(4).mean() / centered.pow(2).mean().pow(2) - 3.0).item()
