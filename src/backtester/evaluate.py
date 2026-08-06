"""Backtesting suite: Deep Hedging vs. analytic Black-Scholes delta hedging.

Stress-tests a trained Deep Hedging policy (src/policy/hedging_agent.py)
against the closed-form Black-Scholes delta strategy on synthetic price
paths with regime-switching volatility and high transaction friction (10-50
bps), then reports PnL distributions, CVaR 95%/99% risk metrics (per
math_spec.md section 3), and total transaction costs incurred (per
math_spec.md section 2).
"""

import json
import math
import sys
from pathlib import Path
from typing import Annotated, Dict, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Allow `python src/backtester/evaluate.py` to resolve sibling packages the
# same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from environment.market_env import MarketEnvironment  # noqa: E402
from generator.market_gan import Generator  # noqa: E402
from loss.cvar import CVaRLoss  # noqa: E402
from policy.hedging_agent import HedgingAgent  # noqa: E402
from policy.train_policy import PolicyTrainer  # noqa: E402

# Categorical colors (fixed order, project dataviz palette slots 1 & 2):
# Deep Hedging is always blue, Black-Scholes Delta is always orange.
COLOR_DEEP_HEDGING = "#2a78d6"
COLOR_BLACK_SCHOLES = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"


# -----------------------------------------------------------------------------
# 1. Analytic Black-Scholes delta for a European call
# -----------------------------------------------------------------------------


def _standard_normal_cdf(
    x: Annotated[torch.Tensor, "any shape"]
) -> Annotated[torch.Tensor, "same shape, N(x)"]:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


# -----------------------------------------------------------------------------
# 2. Stress-test price paths: regime-switching volatility GBM
# -----------------------------------------------------------------------------


def simulate_regime_switching_paths(
    batch_size: Annotated[int, "number of stress paths"],
    seq_len: Annotated[int, "number of price observations per path"],
    s0: Annotated[float, "initial asset price S_0"] = 1.0,
    dt: Annotated[float, "time increment per step"] = 1.0,
    low_vol: Annotated[float, "annualized volatility in the calm regime"] = 0.15,
    high_vol: Annotated[float, "annualized volatility in the stressed regime"] = 0.60,
    switch_prob: Annotated[float, "per-step probability of a regime flip"] = 0.10,
    drift: Annotated[float, "log-return drift"] = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] strictly positive stress price paths"]:
    """GBM price paths with a Markov-switching volatility regime.

    Log-returns are always exponentiated into the price path
    (S_t = S_{t-1} * exp(log_return)), guaranteeing strict positivity
    regardless of the sampled regime or noise draw.
    """
    regime = torch.zeros(batch_size, dtype=torch.long)  # 0 = low vol, 1 = high vol
    low_vol_t = torch.tensor(low_vol)
    high_vol_t = torch.tensor(high_vol)

    # [Batch, Time_Steps] accumulator for log-prices
    log_prices = torch.empty(batch_size, seq_len)
    log_prices[:, 0] = math.log(s0)

    for t in range(1, seq_len):
        flip = torch.rand(batch_size, generator=generator) < switch_prob
        regime = torch.where(flip, 1 - regime, regime)
        sigma = torch.where(regime == 1, high_vol_t, low_vol_t)  # [Batch]

        z = torch.randn(batch_size, generator=generator)
        log_return = (drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        log_prices[:, t] = log_prices[:, t - 1] + log_return

    # [Batch, Time_Steps] -> [Batch, Time_Steps, 1]
    prices = torch.exp(log_prices).unsqueeze(-1)
    return prices


# -----------------------------------------------------------------------------
# 3. Risk metrics
# -----------------------------------------------------------------------------


def empirical_cvar(
    wealth: Annotated[torch.Tensor, "[Batch] terminal portfolio wealth samples"],
    alpha: Annotated[float, "confidence level in (0, 1), e.g. 0.95 or 0.99"],
) -> Annotated[float, "empirical CVaR_alpha of the wealth distribution's losses"]:
    """Empirical Rockafellar-Uryasev CVaR (math_spec.md section 3), evaluated
    at h = the empirical (1-alpha)-quantile of losses -- the exact minimizer
    of `inf_h {h + E[max(-X-h,0)] / (1-alpha)}` for a finite empirical sample.
    Equivalent to the value `loss.cvar.CVaRLoss` converges to during training,
    computed here directly for fast, deterministic backtest reporting.
    """
    losses = -wealth
    var = torch.quantile(losses, alpha)
    tail = losses[losses >= var]
    if tail.numel() == 0:
        return var.item()
    return tail.mean().item()


def _summarize_strategy(
    wealth: Annotated[torch.Tensor, "[Batch] terminal wealth"],
    total_cost: Annotated[torch.Tensor, "[Batch] total transaction cost per path"],
) -> Dict[str, float]:
    return {
        "mean_wealth": wealth.mean().item(),
        "std_wealth": wealth.std().item(),
        "cvar_95": empirical_cvar(wealth, 0.95),
        "cvar_99": empirical_cvar(wealth, 0.99),
        "mean_transaction_cost": total_cost.mean().item(),
        "total_transaction_cost": total_cost.sum().item(),
    }


# -----------------------------------------------------------------------------
# Plotting (project dataviz style: fixed categorical colors, thin marks,
# recessive gridlines, legend for multi-series charts)
# -----------------------------------------------------------------------------


def _style_axes(ax: "plt.Axes") -> None:
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    ax.xaxis.label.set_color(COLOR_TEXT_PRIMARY)
    ax.yaxis.label.set_color(COLOR_TEXT_PRIMARY)
    ax.title.set_color(COLOR_TEXT_PRIMARY)


def _plot_pnl_distribution(
    wealth_dh: torch.Tensor, wealth_bs: torch.Tensor, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    bins = 40
    ax.hist(
        wealth_dh.detach().numpy(),
        bins=bins,
        alpha=0.6,
        color=COLOR_DEEP_HEDGING,
        label="Deep Hedging",
        zorder=2,
    )
    ax.hist(
        wealth_bs.detach().numpy(),
        bins=bins,
        alpha=0.6,
        color=COLOR_BLACK_SCHOLES,
        label="Black-Scholes Delta",
        zorder=2,
    )
    ax.axvline(0.0, color=COLOR_BASELINE, linewidth=1.0, zorder=1)
    ax.set_xlabel("Terminal Wealth")
    ax.set_ylabel("Frequency")
    ax.set_title("Final PnL Distribution: Deep Hedging vs. Black-Scholes Delta")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def _plot_cvar_comparison(summary: Dict, path: Path) -> None:
    labels = ["CVaR 95%", "CVaR 99%"]
    dh_values = [summary["deep_hedging"]["cvar_95"], summary["deep_hedging"]["cvar_99"]]
    bs_values = [summary["black_scholes"]["cvar_95"], summary["black_scholes"]["cvar_99"]]

    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COLOR_SURFACE)
    ax.bar(
        [i - width / 2 for i in x],
        dh_values,
        width,
        color=COLOR_DEEP_HEDGING,
        label="Deep Hedging",
        zorder=2,
    )
    ax.bar(
        [i + width / 2 for i in x],
        bs_values,
        width,
        color=COLOR_BLACK_SCHOLES,
        label="Black-Scholes Delta",
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CVaR (Expected Shortfall of Loss)")
    ax.set_title("Tail Risk Comparison")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def _plot_transaction_costs(
    cost_dh: torch.Tensor, cost_bs: torch.Tensor, path: Path
) -> None:
    labels = ["Deep Hedging", "Black-Scholes Delta"]
    totals = [cost_dh.sum().item(), cost_bs.sum().item()]
    colors = [COLOR_DEEP_HEDGING, COLOR_BLACK_SCHOLES]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COLOR_SURFACE)
    ax.bar(labels, totals, color=colors, width=0.5, zorder=2)
    ax.set_ylabel("Total Transaction Cost (aggregate over all paths)")
    ax.set_title("Transaction Costs Incurred")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Backtest orchestration
# -----------------------------------------------------------------------------


def run_backtest(
    deep_hedging_policy: HedgingAgent,
    strike: Annotated[float, "option strike price K"],
    batch_size: Annotated[int, "number of stress test paths"] = 2000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    implied_vol: Annotated[float, "implied vol fed into both policies' state"] = 0.30,
    low_vol: Annotated[float, "stress regime: calm-period volatility"] = 0.15,
    high_vol: Annotated[float, "stress regime: turbulent-period volatility"] = 0.60,
    switch_prob: Annotated[float, "stress regime: per-step switch probability"] = 0.10,
    dt: Annotated[float, "time increment per step"] = 1.0,
    output_dir: Annotated[Union[str, Path], "directory for plots and JSON summary"] = "results",
    seed: Optional[int] = 0,
) -> Annotated[Dict, "benchmark summary, also written to <output_dir>/benchmark_summary.json"]:
    path_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    prices = simulate_regime_switching_paths(
        batch_size,
        seq_len,
        s0=strike,
        dt=dt,
        low_vol=low_vol,
        high_vol=high_vol,
        switch_prob=switch_prob,
        generator=path_generator,
    )

    environment = MarketEnvironment(strike=strike, proportional_fee=proportional_fee, dt=dt)
    bs_policy = BlackScholesDeltaPolicy(strike=strike)

    deep_hedging_policy.eval()
    with torch.no_grad():
        wealth_dh, cost_dh = environment.simulate_with_costs(
            deep_hedging_policy, prices, implied_vol
        )
        wealth_bs, cost_bs = environment.simulate_with_costs(bs_policy, prices, implied_vol)

    summary = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "strike": strike,
            "proportional_fee_bps": proportional_fee * 1e4,
            "implied_vol": implied_vol,
            "stress_regime": {
                "low_vol": low_vol,
                "high_vol": high_vol,
                "switch_prob": switch_prob,
            },
        },
        "deep_hedging": _summarize_strategy(wealth_dh, cost_dh),
        "black_scholes": _summarize_strategy(wealth_bs, cost_bs),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_pnl_distribution(wealth_dh, wealth_bs, output_dir / "pnl_distribution.png")
    _plot_cvar_comparison(summary, output_dir / "cvar_comparison.png")
    _plot_transaction_costs(cost_dh, cost_bs, output_dir / "transaction_costs.png")

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# Demo entrypoint: train a policy on market_gan.py paths, then backtest it
# under stress conditions
# -----------------------------------------------------------------------------


def _train_demo_policy(
    strike: float, implied_vol: float, device: torch.device
) -> HedgingAgent:
    """Trains a small HedgingAgent on in-distribution market_gan.py paths.

    Fallback for when no checkpoint exists at checkpoints/hedging_agent.pt:
    a short training loop gives `run_backtest` a non-trivial policy to
    stress-test. Prefer `python src/policy/train_policy.py` for a properly
    converged policy.
    """
    policy = HedgingAgent(hidden_dim=32, num_hidden_layers=2)
    generator = Generator(noise_dim=8, hidden_dim=32, num_layers=1, initial_price=strike)
    environment = MarketEnvironment(strike=strike, proportional_fee=0.003, dt=1.0)
    cvar_loss = CVaRLoss(alpha=0.95)

    trainer = PolicyTrainer(
        policy,
        environment,
        generator,
        cvar_loss,
        implied_vol=implied_vol,
        lr=1e-2,
        device=device,
    )
    for _ in range(200):
        trainer.train_step(batch_size=64, seq_len=30)

    return policy


def _load_or_train_policy(
    checkpoint_path: Annotated[Path, "path to a checkpoint saved by policy/train_policy.py"],
    device: torch.device,
) -> Annotated[Tuple[HedgingAgent, float, float], "(policy, strike, implied_vol)"]:
    """Loads a pretrained HedgingAgent, or trains a short demo policy as fallback."""
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy_args = checkpoint["args"]
        policy = HedgingAgent(
            hidden_dim=policy_args["hidden_dim"],
            num_hidden_layers=policy_args["num_hidden_layers"],
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded pretrained policy from {checkpoint_path}")
        return policy, policy_args["strike"], policy_args["implied_vol"]

    strike, implied_vol = 1.0, 0.30
    print(
        f"No policy checkpoint found at {checkpoint_path}; training a short demo policy. "
        "Run `python src/policy/train_policy.py` first for a properly converged policy."
    )
    policy = _train_demo_policy(strike, implied_vol, device)
    return policy, strike, implied_vol


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    demo_policy, strike, implied_vol = _load_or_train_policy(
        Path("checkpoints/hedging_agent.pt"), device
    )

    result = run_backtest(
        deep_hedging_policy=demo_policy,
        strike=strike,
        batch_size=2000,
        seq_len=30,
        proportional_fee=0.003,  # 30 bps, within the 10-50 bps stress range
        implied_vol=implied_vol,
        low_vol=0.15,
        high_vol=0.60,
        switch_prob=0.10,
        output_dir="results",
        seed=42,
    )

    print(json.dumps(result, indent=2))
