"""Market environment simulator (src/environment/market_env.py).

Computes the terminal hedged-portfolio wealth for a short European call
position, following math_spec.md sections 1, 1.1, and 2:

    Payoff(S_T) = max(S_T - K, 0)
    Cost_t = kappa * |delta_t - delta_{t-1}| * S_t
    Wealth_T = P_0 - Payoff(S_T) + sum_{t=0}^{T-1} [delta_t * (S_{t+1} - S_t) - Cost_t]

`premium` (P_0) defaults to 0.0 -- a caller that hasn't computed the
option's fair value gets the same behavior as before this term existed.
See math_spec.md section 1.1 for why omitting it made every mean wealth in
this repo negative, and RESULTS.md for where P_0 is (and isn't) wired in.
"""

from typing import Annotated, Callable, Optional, Protocol, Tuple, Union

import torch


class HedgingPolicy(Protocol):
    """Structural type for a stepwise hedging policy (the default `sequence_policy=False` mode).

    Called once per time step with a hand-crafted state vector. Satisfied by
    `HedgingAgent` (via `nn.Module.__call__`) as well as non-learned
    strategies such as an analytic Black-Scholes delta policy.
    """

    def __call__(
        self, state: Annotated[torch.Tensor, "[Batch, 4] = (S_t, delta_{t-1}, T-t, implied_vol)"]
    ) -> Annotated[torch.Tensor, "[Batch, 1] hedge ratio delta_t"]: ...


class SequenceHedgingPolicy(Protocol):
    """Structural type for a whole-path hedging policy (`sequence_policy=True` mode).

    Called once per rollout with the full price path, returning every
    delta_t in one pass. Satisfied by `RecurrentHedgingAgent`, whose own
    recurrent hidden state carries history and elapsed time implicitly
    instead of receiving them as explicit state features.
    """

    def __call__(
        self, prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] full price path S_0..S_N"]
    ) -> Annotated[
        torch.Tensor, "[Batch, Time_Steps - 1, 1] hedge ratios delta_0..delta_{N-1}"
    ]: ...


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


def estimate_premium_monte_carlo(
    sample_prices: Annotated[
        Callable[[int], torch.Tensor],
        "callable(batch_size) -> [Batch, Time_Steps, 1] price paths, from whatever "
        "simulator/generator is in use (regime-switching, WGAN-GP, TimeGAN, ...)",
    ],
    strike: Annotated[float, "option strike price K"],
    num_paths: Annotated[
        int,
        "500,000 chosen empirically: at this process's own regime-switching "
        "params, cross-seed std of the estimate is ~1% relative "
        "(50k/100k are 7-10% relative, too noisy to hold fixed for a whole "
        "training run) -- see RESULTS.md's P0 section for the convergence check",
    ] = 500_000,
    chunk_size: Annotated[
        int,
        "paths per sample_prices() call. A single 500k-path call is near-instant "
        "for the closed-form/analytic regime-switching simulator, but an RNN-based "
        "GAN generator forward pass at that batch size was observed to blow past "
        "any reasonable memory/time budget on CPU (100k took ~2s, 500k did not "
        "finish in 180s) -- chunking keeps peak memory bounded regardless of "
        "num_paths and works uniformly for both cases",
    ] = 50_000,
) -> Annotated[float, "Monte Carlo E[Payoff(S_T)], r=0 so this is also the fair option value"]:
    with torch.no_grad():
        total = 0.0
        remaining = num_paths
        while remaining > 0:
            batch = min(chunk_size, remaining)
            prices = sample_prices(batch)
            S_T = prices[:, -1, :]
            payoff = european_call_payoff(S_T, strike)
            total += payoff.sum().item()
            remaining -= batch
        return total / num_paths


class MarketEnvironment:
    """Simulates the hedged wealth path of a short European call position."""

    def __init__(
        self,
        strike: Annotated[float, "option strike price K"],
        proportional_fee: Annotated[float, "kappa, proportional transaction fee rate"] = 0.0,
        dt: Annotated[float, "time increment per step, used to compute T - t"] = 1.0,
        premium: Annotated[
            float, "P_0, the option premium collected for writing the option"
        ] = 0.0,
    ) -> None:
        self.strike = strike
        self.proportional_fee = proportional_fee
        self.dt = dt
        self.premium = premium

    def simulate(
        self,
        policy: Union[HedgingPolicy, SequenceHedgingPolicy],
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
        sequence_policy: Annotated[
            bool, "True for a whole-path RecurrentHedgingAgent-style policy"
        ] = False,
        chunk_size: Annotated[
            Optional[int],
            "process in batches of this size and concatenate, instead of one pass over "
            "the full batch. RNN/LSTM/GRU policies hit a severe (not just linear) CPU "
            "slowdown at very large batch sizes -- e.g. a 500,000-path single-pass "
            "evaluation through an LSTM policy didn't finish in 120s where chunks of "
            "50,000 complete in seconds each. None (default) preserves prior behavior.",
        ] = None,
    ) -> Annotated[torch.Tensor, "[Batch] terminal portfolio wealth Wealth_T"]:
        wealth, _ = self._rollout(policy, prices, implied_vol, sequence_policy, chunk_size)
        return wealth

    def simulate_with_costs(
        self,
        policy: Union[HedgingPolicy, SequenceHedgingPolicy],
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
        sequence_policy: Annotated[
            bool, "True for a whole-path RecurrentHedgingAgent-style policy"
        ] = False,
        chunk_size: Annotated[
            Optional[int], "see simulate()'s chunk_size docstring"
        ] = None,
    ) -> Annotated[
        Tuple[torch.Tensor, torch.Tensor],
        "([Batch] terminal wealth Wealth_T, [Batch] total transaction cost paid)",
    ]:
        return self._rollout(policy, prices, implied_vol, sequence_policy, chunk_size)

    def _rollout(
        self,
        policy: Union[HedgingPolicy, SequenceHedgingPolicy],
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
        implied_vol: Annotated[
            Union[float, torch.Tensor], "scalar or [Batch, 1] implied volatility"
        ],
        sequence_policy: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = prices.shape[0]
        if chunk_size is not None and chunk_size < batch_size:
            wealth_chunks = []
            cost_chunks = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                iv_chunk = (
                    implied_vol[start:end] if isinstance(implied_vol, torch.Tensor) else implied_vol
                )
                w, c = self._rollout(policy, prices[start:end], iv_chunk, sequence_policy)
                wealth_chunks.append(w)
                cost_chunks.append(c)
            return torch.cat(wealth_chunks), torch.cat(cost_chunks)

        if sequence_policy:
            wealth, total_cost = self._rollout_sequence(policy, prices)
        else:
            wealth, total_cost = self._rollout_stepwise(policy, prices, implied_vol)

        S_T = prices[:, -1, :]  # [Batch, 1]
        payoff = european_call_payoff(S_T, self.strike).squeeze(-1)  # [Batch]

        wealth = wealth + self.premium - payoff
        return wealth, total_cost

    def _rollout_stepwise(
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

        return wealth, total_cost

    def _rollout_sequence(
        self,
        policy: SequenceHedgingPolicy,
        prices: Annotated[
            torch.Tensor, "[Batch, Time_Steps, 1] simulated asset price path S_0..S_{N}"
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = prices.shape
        device = prices.device

        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps - 1, 1] (one delta per hedge step)
        delta_path = policy(prices)

        S_t = prices[:, :-1, :]  # [Batch, Time_Steps - 1, 1]
        S_next = prices[:, 1:, :]  # [Batch, Time_Steps - 1, 1]

        # [Batch, 1, 1] + [Batch, Time_Steps - 2, 1] -> [Batch, Time_Steps - 1, 1] (delta_{-1} = 0)
        zeros = torch.zeros(batch_size, 1, 1, device=device)
        delta_prev = torch.cat([zeros, delta_path[:, :-1, :]], dim=1)

        hedge_pnl = delta_path * (S_next - S_t)  # [Batch, Time_Steps - 1, 1]
        cost = transaction_cost(delta_path, delta_prev, S_t, self.proportional_fee)

        # [Batch, Time_Steps - 1, 1] -> [Batch]
        wealth = (hedge_pnl - cost).sum(dim=1).squeeze(-1)
        total_cost = cost.sum(dim=1).squeeze(-1)

        return wealth, total_cost
