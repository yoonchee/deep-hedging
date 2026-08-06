"""Convex Risk Measure loss: Expected Shortfall / CVaR (math_spec.md, section 3).

Implements the Rockafellar-Uryasev differentiable formulation:

    CVaR_alpha(X) = inf_h { h + (1 / (1 - alpha)) * E[max(-X - h, 0)] }

where X is the portfolio P&L (or terminal wealth). The auxiliary variable h
(the VaR threshold) is a learnable parameter that is optimized jointly with
the rest of the model during backpropagation, so at convergence h equals the
alpha-VaR of X and the module's output equals CVaR_alpha(X).
"""

from typing import Annotated

import torch
import torch.nn as nn


class CVaRLoss(nn.Module):
    """Differentiable CVaR (Expected Shortfall) loss with a learnable VaR threshold."""

    def __init__(
        self,
        alpha: Annotated[float, "confidence level in (0, 1), e.g. 0.95 or 0.99"] = 0.95,
        h_init: Annotated[float, "initial value of the learnable VaR threshold h"] = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        self.alpha = alpha
        self.h = nn.Parameter(torch.tensor(float(h_init)))

    def forward(
        self, x: Annotated[torch.Tensor, "[Batch] or [Batch, 1] portfolio P&L samples"]
    ) -> Annotated[torch.Tensor, "scalar CVaR_alpha(X) estimate"]:
        # [Batch, ...] -> [Batch] (flatten any trailing singleton dims)
        pnl = x.reshape(-1)

        # [Batch] -> [Batch] (per-sample shortfall beyond the VaR threshold h)
        shortfall = torch.relu(-pnl - self.h)

        # [Batch] -> scalar
        loss = self.h + shortfall.mean() / (1.0 - self.alpha)
        return loss
