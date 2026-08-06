"""Market environment simulator (src/environment/market_env.py).

Computes the terminal hedged-portfolio wealth for a short European call
position, following math_spec.md sections 1 and 2:

    Payoff(S_T) = max(S_T - K, 0)
    Cost_t = kappa * |delta_t - delta_{t-1}| * S_t
    Wealth_T = -Payoff(S_T) + sum_{t=0}^{T-1} [delta_t * (S_{t+1} - S_t) - Cost_t]
"""

from typing import Annotated, Protocol, Tuple, Union

import torch


class HedgingPolicy(Protocol):
    """Structural type for anything usable as a hedging policy in `MarketEnvironment`.

    Satisfied by `HedgingAgent` (via `nn.Module.__call__`) as well as
    non-learned strategies such as an analytic Black-Scholes delta policy.
    """

    def __call__(
        self, state: Annotated[torch.Tensor, "[Batch, 4] = (S_t, delta_{t-1}, T-t, implied_vol)"]
    ) -> Annotated[torch.Tensor, "[Batch, 1] hedge ratio delta_t"]: ...


def transaction_cost(
    delta_t: Annotated[torch.Tensor, "[Batch, 1] current hedge ratio"],
    delta_prev: Annotated[torch.Tensor, "[Batch, 1] previous hedge ratio"],
    S_t: Annotated[torch.Tensor, "[Batch, 1] asset price at t"],
    proportional_fee: Annotated[float, "kappa, proportional transaction fee rate"],
) -> Annotated[torch.Tensor, "[Batch, 1] rebalancing cost at t"]:
    # [Batch, 1] -> [Batch, 1]
    return proportional_fee * torch.abs(delta_t - delta_prev) * S_t


def european_call_payoff(
    S_T: Annotated[torch.Tensor, "[Batch, 1] terminal asset price"],
    strike: Annotated[float, "option strike price K"],
) -> Annotated[torch.Tensor, "[Batch, 1] call option payoff max(S_T - K, 0)"]:
    # [Batch, 1] -> [Batch, 1]
    return torch.clamp(S_T - strike, min=0.0)


class MarketEnvironment:
    """Simulates the hedged wealth path of a short European call position."""

    def __init__(
        self,
        strike: Annotated[float, "option strike price K"],
        proportional_fee: Annotated[float, "kappa, proportional transaction fee rate"] = 0.0,
        dt: Annotated[float, "time increment per step, used to compute T - t"] = 1.0,
    ) -> None:
        self.strike = strike
        self.proportional_fee = proportional_fee
        self.dt = dt

    def simulate(
        self,
        policy: HedgingPolicy,
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
    ) -> Annotated[torch.Tensor, "[Batch] terminal portfolio wealth Wealth_T"]:
        wealth, _ = self._rollout(policy, prices, implied_vol)
        return wealth

    def simulate_with_costs(
        self,
        policy: HedgingPolicy,
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
    ) -> Annotated[
        Tuple[torch.Tensor, torch.Tensor],
        "([Batch] terminal wealth Wealth_T, [Batch] total transaction cost paid)",
    ]:
        return self._rollout(policy, prices, implied_vol)

    def _rollout(
        self,
        policy: HedgingPolicy,
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_prices, _ = prices.shape
        n_steps = num_prices - 1  # number of hedge rebalances, t = 0..n_steps-1
        device = prices.device

        if isinstance(implied_vol, torch.Tensor):
            iv = implied_vol.to(device).reshape(batch_size, 1)
        else:
            iv = torch.full((batch_size, 1), float(implied_vol), device=device)

        delta_prev = torch.zeros(batch_size, 1, device=device)
        wealth = torch.zeros(batch_size, device=device)
        total_cost = torch.zeros(batch_size, device=device)

        for t in range(n_steps):
            S_t = prices[:, t, :]  # [Batch, 1]

            # [Batch, 1] (T - t, broadcast as a constant feature per path)
            time_to_maturity = torch.full(
                (batch_size, 1), self.dt * (n_steps - t), device=device
            )

            # [Batch, 1] x4 -> [Batch, 4]
            state = torch.cat([S_t, delta_prev, time_to_maturity, iv], dim=-1)

            # [Batch, 4] -> [Batch, 1]
            delta_t = policy(state)

            S_next = prices[:, t + 1, :]  # [Batch, 1]
            hedge_pnl = delta_t * (S_next - S_t)  # [Batch, 1]
            cost = transaction_cost(delta_t, delta_prev, S_t, self.proportional_fee)  # [Batch, 1]

            # [Batch] + [Batch, 1] -> [Batch]
            wealth = wealth + (hedge_pnl - cost).squeeze(-1)
            total_cost = total_cost + cost.squeeze(-1)
            delta_prev = delta_t

        S_T = prices[:, -1, :]  # [Batch, 1]
        payoff = european_call_payoff(S_T, self.strike).squeeze(-1)  # [Batch]

        wealth = wealth - payoff
        return wealth, total_cost
