"""Analytic Black-Scholes call delta (src/common/black_scholes.py).

Extracted from backtester/evaluate.py into common/ so it can be imported
from both the backtester and policy/train_policy.py (a Black-Scholes wealth
baseline is used there as a variance-reduction control variate for CVaR
training -- see PolicyTrainer's `use_bs_baseline`) without a circular
import (evaluate.py itself imports PolicyTrainer).
"""

import math
from typing import Annotated

import torch


def _standard_normal_cdf(
    x: Annotated[torch.Tensor, "any shape"]
) -> Annotated[torch.Tensor, "same shape, N(x)"]:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(
    S0: Annotated[float, "initial asset price"],
    K: Annotated[float, "option strike price"],
    tau: Annotated[float, "time to maturity T (> 0)"],
    sigma: Annotated[float, "volatility"],
    r: Annotated[float, "risk-free rate"] = 0.0,
) -> Annotated[float, "analytic Black-Scholes call price C_0, the fair option premium"]:
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S0 * N(d1) - K * math.exp(-r * tau) * N(d2)


def black_scholes_call_delta(
    S: Annotated[torch.Tensor, "[Batch, 1] current asset price"],
    K: Annotated[float, "option strike price"],
    tau: Annotated[torch.Tensor, "[Batch, 1] time to maturity T - t (> 0)"],
    sigma: Annotated[torch.Tensor, "[Batch, 1] implied volatility"],
    r: Annotated[float, "risk-free rate"] = 0.0,
) -> Annotated[torch.Tensor, "[Batch, 1] analytic call delta N(d1) in [0, 1]"]:
    # [Batch, 1] -> [Batch, 1]
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    return _standard_normal_cdf(d1)


class BlackScholesDeltaPolicy:
    """Analytic delta-hedging strategy exposing the same interface as HedgingAgent.

    Satisfies `environment.market_env.HedgingPolicy`, so it can be evaluated
    through `MarketEnvironment` exactly like a trained `HedgingAgent`.
    """

    def __init__(
        self,
        strike: Annotated[float, "option strike price K"],
        r: Annotated[float, "risk-free rate"] = 0.0,
    ) -> None:
        self.strike = strike
        self.r = r

    def __call__(
        self,
        state: Annotated[torch.Tensor, "[Batch, 4] = (S_t, delta_{t-1}, T-t, implied_vol)"],
    ) -> Annotated[torch.Tensor, "[Batch, 1] delta_t = N(d1)"]:
        S_t = state[:, 0:1]
        tau = state[:, 2:3]
        sigma = state[:, 3:4]
        return black_scholes_call_delta(S_t, self.strike, tau, sigma, self.r)
