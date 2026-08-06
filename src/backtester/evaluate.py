"""Backtesting suite: Deep Hedging (MLP / Basic RNN / LSTM / GRU) vs. analytic
Black-Scholes delta hedging.

Stress-tests every available trained policy (src/policy/hedging_agent.py)
against the closed-form Black-Scholes delta strategy on synthetic price
paths with regime-switching volatility and high transaction friction (10-50
bps), then reports PnL distributions, CVaR 95%/99% + skewness/excess
kurtosis (per math_spec.md section 3), and total transaction costs incurred
(per math_spec.md section 2) -- mirroring the result tables and comparison
charts in Kim (2021), "Deep Hedging, Generative Adversarial Networks, and
Beyond", which compares exactly these architectures against Black-Scholes.
"""

import json
import math
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

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
from policy.hedging_agent import HedgingAgent, RecurrentHedgingAgent  # noqa: E402
from policy.train_policy import PolicyTrainer  # noqa: E402

# Fixed per-strategy colors (project dataviz palette, categorical slots 1-5).
# Color follows the entity, never its rank: a strategy keeps its color
# whether it's plotted alongside all four others or alone.
STRATEGY_COLORS: Dict[str, str] = {
    "Black-Scholes Delta": "#2a78d6",  # slot 1 blue -- fixed baseline reference
    "MLP": "#eb6834",  # slot 2 orange
    "Basic RNN": "#1baf7a",  # slot 3 aqua
    "LSTM": "#eda100",  # slot 4 yellow
    "GRU": "#e87ba4",  # slot 5 magenta
}
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"

# Maps train_policy.py's --architecture choice to a display name.
ARCHITECTURE_DISPLAY_NAMES: Dict[str, str] = {
    "mlp": "MLP",
    "rnn": "Basic RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


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


def _skewness(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    centered = x - x.mean()
    return (centered.pow(3).mean() / centered.pow(2).mean().pow(1.5)).item()


def _excess_kurtosis(x: Annotated[torch.Tensor, "[Batch] samples"]) -> float:
    centered = x - x.mean()
    return (centered.pow(4).mean() / centered.pow(2).mean().pow(2) - 3.0).item()


def _summarize_strategy(
    wealth: Annotated[torch.Tensor, "[Batch] terminal wealth"],
    total_cost: Annotated[torch.Tensor, "[Batch] total transaction cost per path"],
) -> Dict[str, float]:
    return {
        "mean_wealth": wealth.mean().item(),
        "std_wealth": wealth.std().item(),
        "cvar_95": empirical_cvar(wealth, 0.95),
        "cvar_99": empirical_cvar(wealth, 0.99),
        "skewness": _skewness(wealth),
        "excess_kurtosis": _excess_kurtosis(wealth),
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


def _ordered_strategy_names(names: Annotated[Any, "iterable of strategy names present"]) -> List[str]:
    """Orders strategy names by their fixed slot in STRATEGY_COLORS."""
    present = set(names)
    return [name for name in STRATEGY_COLORS if name in present]


def _plot_pnl_distribution(wealth_by_strategy: Dict[str, torch.Tensor], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_SURFACE)
    bins = 40
    for name in _ordered_strategy_names(wealth_by_strategy):
        ax.hist(
            wealth_by_strategy[name].detach().numpy(),
            bins=bins,
            alpha=0.55,
            color=STRATEGY_COLORS[name],
            label=name,
            zorder=2,
        )
    ax.axvline(0.0, color=COLOR_BASELINE, linewidth=1.0, zorder=1)
    ax.set_xlabel("Terminal Wealth")
    ax.set_ylabel("Frequency")
    ax.set_title("Final PnL Distribution")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def _plot_cvar_comparison(summary: Dict[str, Dict[str, float]], path: Path) -> None:
    names = _ordered_strategy_names(summary.keys())
    labels = ["CVaR 95%", "CVaR 99%"]
    x = list(range(len(labels)))
    n = len(names)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=COLOR_SURFACE)
    for i, name in enumerate(names):
        offset = (i - (n - 1) / 2) * width
        values = [summary[name]["cvar_95"], summary[name]["cvar_99"]]
        ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            color=STRATEGY_COLORS[name],
            label=name,
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


def _plot_transaction_costs(cost_by_strategy: Dict[str, torch.Tensor], path: Path) -> None:
    names = _ordered_strategy_names(cost_by_strategy)
    totals = [cost_by_strategy[name].sum().item() for name in names]
    colors = [STRATEGY_COLORS[name] for name in names]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=COLOR_SURFACE)
    ax.bar(names, totals, color=colors, width=0.5, zorder=2)
    ax.set_ylabel("Total Transaction Cost (aggregate over all paths)")
    ax.set_title("Transaction Costs Incurred")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Delta-convexity diagnostic (paper Figures 5/8/11): confirms the learned
# policy recovers a Black-Scholes-like convex Delta-vs-Spot curve.
# -----------------------------------------------------------------------------


def _delta_grid_stepwise(
    policy_obj: Annotated[Any, "a stepwise HedgingPolicy callable"],
    spot_grid: Annotated[torch.Tensor, "[Grid] spot prices to sweep"],
    t: Annotated[int, "hedge step index"],
    n_steps: Annotated[int, "total number of hedge steps"],
    strike: float,
    implied_vol: float,
    dt: float,
) -> Annotated[torch.Tensor, "[Grid] delta_t as a function of spot"]:
    grid_size = spot_grid.shape[0]
    S = spot_grid.reshape(grid_size, 1)
    delta_prev = torch.zeros(grid_size, 1)
    tau = torch.full((grid_size, 1), dt * (n_steps - t))
    iv = torch.full((grid_size, 1), implied_vol)
    state = torch.cat([S, delta_prev, tau, iv], dim=-1)  # [Grid, 4]
    with torch.no_grad():
        return policy_obj(state).squeeze(-1)  # [Grid]


def _delta_grid_sequence(
    policy_obj: Annotated[Any, "a SequenceHedgingPolicy callable"],
    spot_grid: Annotated[torch.Tensor, "[Grid] spot prices to sweep"],
    t: Annotated[int, "hedge step index"],
    strike: float,
) -> Annotated[torch.Tensor, "[Grid] delta_t as a function of spot"]:
    # Synthetic price path: flat at strike for steps 0..t-1, then the swept
    # spot value at step t, with one trailing duplicate (the policy's
    # forward pass drops the final price -- no hedge decision is needed at
    # the path's last observation, so a spare value is appended to recover
    # delta_t from history S_0..S_t).
    grid_size = spot_grid.shape[0]
    flat_part = torch.full((grid_size, t), strike)
    spot_col = spot_grid.reshape(grid_size, 1)
    prices = torch.cat([flat_part, spot_col, spot_col], dim=1).unsqueeze(-1)  # [Grid, t+2, 1]
    with torch.no_grad():
        delta_path = policy_obj(prices)  # [Grid, t+1, 1]
    return delta_path[:, -1, 0]  # [Grid]


def _plot_delta_convexity(
    all_strategies: Dict[str, Tuple[Any, bool]],
    strike: float,
    implied_vol: float,
    n_steps: Annotated[int, "total number of hedge steps (seq_len - 1)"],
    dt: float,
    path: Path,
    time_steps: Optional[List[int]] = None,
) -> None:
    if time_steps is None:
        candidates = [0, 1, 5, 10, 15, n_steps - 1]
        time_steps = sorted({t for t in candidates if 0 <= t < n_steps})

    spot_grid = torch.linspace(0.8 * strike, 1.2 * strike, 60)

    n_plots = len(time_steps)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), facecolor=COLOR_SURFACE
    )
    axes = [axes] if n_plots == 1 else axes.flatten()

    for ax, t in zip(axes, time_steps):
        for name in _ordered_strategy_names(all_strategies.keys()):
            policy_obj, sequence_policy = all_strategies[name]
            if hasattr(policy_obj, "eval"):
                policy_obj.eval()
            if sequence_policy:
                deltas = _delta_grid_sequence(policy_obj, spot_grid, t, strike)
            else:
                deltas = _delta_grid_stepwise(
                    policy_obj, spot_grid, t, n_steps, strike, implied_vol, dt
                )
            ax.scatter(
                spot_grid.numpy(),
                deltas.numpy(),
                s=10,
                alpha=0.8,
                color=STRATEGY_COLORS[name],
                label=name,
                zorder=2,
            )
        ax.set_title(f"Delta at Time {t}")
        ax.set_xlabel("Spot")
        ax.set_ylabel("Delta")
        ax.set_ylim(-0.05, 1.05)
        _style_axes(ax)

    for ax in axes[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), frameon=False,
        labelcolor=COLOR_TEXT_PRIMARY, bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle("Delta Convexity Across Time Steps", color=COLOR_TEXT_PRIMARY, y=1.10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Backtest orchestration
# -----------------------------------------------------------------------------


def run_backtest(
    policies: Annotated[
        Dict[str, Tuple[Any, bool]],
        "{display_name: (policy_object, sequence_policy_bool)}; "
        "'Black-Scholes Delta' is added automatically and should not be included",
    ],
    strike: Annotated[float, "option strike price K"],
    batch_size: Annotated[int, "number of stress test paths"] = 2000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    implied_vol: Annotated[float, "implied vol fed into every policy's state"] = 0.30,
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

    all_strategies: Dict[str, Tuple[Any, bool]] = {
        "Black-Scholes Delta": (BlackScholesDeltaPolicy(strike=strike), False)
    }
    all_strategies.update(policies)

    wealth_by_strategy: Dict[str, torch.Tensor] = {}
    cost_by_strategy: Dict[str, torch.Tensor] = {}
    strategy_summary: Dict[str, Dict[str, float]] = {}

    for name, (policy_obj, sequence_policy) in all_strategies.items():
        if hasattr(policy_obj, "eval"):
            policy_obj.eval()
        with torch.no_grad():
            wealth, cost = environment.simulate_with_costs(
                policy_obj, prices, implied_vol, sequence_policy=sequence_policy
            )
        wealth_by_strategy[name] = wealth
        cost_by_strategy[name] = cost
        strategy_summary[name] = _summarize_strategy(wealth, cost)

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
        "strategies": strategy_summary,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_pnl_distribution(wealth_by_strategy, output_dir / "pnl_distribution.png")
    _plot_cvar_comparison(strategy_summary, output_dir / "cvar_comparison.png")
    _plot_transaction_costs(cost_by_strategy, output_dir / "transaction_costs.png")
    _plot_delta_convexity(
        all_strategies,
        strike,
        implied_vol,
        n_steps=seq_len - 1,
        dt=dt,
        path=output_dir / "delta_convexity.png",
    )

    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# 5. Multi-alpha sweep (paper Figures 20-21): one architecture trained at
# several CVaR risk-aversion levels, visualizing the risk-return trade-off.
# -----------------------------------------------------------------------------

# Sequential blue ramp, light -> dark (ordinal steps >= 250 per the project's
# dataviz guide), one hue for the ordered "risk aversion" magnitude.
_SEQUENTIAL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _sequential_color(index: int, n: int) -> str:
    if n <= 1:
        return _SEQUENTIAL_RAMP[len(_SEQUENTIAL_RAMP) // 2]
    step = (len(_SEQUENTIAL_RAMP) - 1) / (n - 1)
    return _SEQUENTIAL_RAMP[round(index * step)]


def load_alpha_sweep_checkpoints(
    architecture: Annotated[str, "'mlp', 'rnn', 'lstm', or 'gru'"],
    alphas: List[float],
    checkpoint_dir: Path,
) -> Dict[float, Tuple[Any, bool]]:
    """Loads per-alpha checkpoints saved by `train_policy.py --alpha-sweep ...`."""
    policies: Dict[float, Tuple[Any, bool]] = {}
    for alpha in alphas:
        alpha_str = f"{alpha:.4g}".replace(".", "_")
        path = checkpoint_dir / f"hedging_agent_{architecture}_alpha{alpha_str}.pt"
        loaded = _load_policy_checkpoint(path)
        if loaded is None:
            print(f"No checkpoint at {path} -- skipping alpha={alpha}.")
            continue
        policy, sequence_policy, _ = loaded
        policies[alpha] = (policy, sequence_policy)
    return policies


def _plot_alpha_sweep_density(wealth_by_alpha: Dict[float, torch.Tensor], path: Path) -> None:
    alphas = sorted(wealth_by_alpha)
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_SURFACE)
    bins = 40
    for i, alpha in enumerate(alphas):
        ax.hist(
            wealth_by_alpha[alpha].detach().numpy(),
            bins=bins,
            alpha=0.55,
            color=_sequential_color(i, len(alphas)),
            label=f"alpha = {alpha}",
            zorder=2,
        )
    ax.axvline(0.0, color=COLOR_BASELINE, linewidth=1.0, zorder=1)
    ax.set_xlabel("Terminal Wealth")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution Across Risk Aversion Parameters")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def _plot_alpha_sweep_boxplot(wealth_by_alpha: Dict[float, torch.Tensor], path: Path) -> None:
    alphas = sorted(wealth_by_alpha)
    data = [wealth_by_alpha[a].detach().numpy() for a in alphas]
    colors = [_sequential_color(i, len(alphas)) for i in range(len(alphas))]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_SURFACE)
    bplot = ax.boxplot(data, patch_artist=True, tick_labels=[str(a) for a in alphas], widths=0.5)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bplot["medians"]:
        median.set_color(COLOR_TEXT_PRIMARY)
    ax.set_xlabel("Risk Aversion Parameter (alpha)")
    ax.set_ylabel("Terminal Wealth")
    ax.set_title("Risk-Return Trade-off Across Risk Aversion Parameters")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def run_alpha_sweep_backtest(
    architecture: Annotated[str, "'mlp', 'rnn', 'lstm', or 'gru'"],
    alphas: Annotated[List[float], "CVaR alphas to compare, e.g. [0.5, 0.75, 0.9, 0.95, 0.99]"],
    strike: Annotated[float, "option strike price K"],
    batch_size: Annotated[int, "number of stress test paths"] = 2000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    implied_vol: Annotated[float, "implied vol fed into every policy's state"] = 0.30,
    low_vol: Annotated[float, "stress regime: calm-period volatility"] = 0.15,
    high_vol: Annotated[float, "stress regime: turbulent-period volatility"] = 0.60,
    switch_prob: Annotated[float, "stress regime: per-step switch probability"] = 0.10,
    dt: Annotated[float, "time increment per step"] = 1.0,
    checkpoint_dir: Annotated[Union[str, Path], "directory with hedging_agent_*_alpha*.pt files"] = "checkpoints",
    output_dir: Annotated[Union[str, Path], "directory for plots and JSON summary"] = "results",
    seed: Optional[int] = 0,
) -> Annotated[Dict, "per-alpha summary, also written to <output_dir>/alpha_sweep_<architecture>_summary.json"]:
    policies = load_alpha_sweep_checkpoints(architecture, alphas, Path(checkpoint_dir))
    if not policies:
        raise FileNotFoundError(
            f"No alpha-sweep checkpoints found for architecture={architecture!r} in {checkpoint_dir}. "
            f"Run `python src/policy/train_policy.py --architecture {architecture} "
            f"--alpha-sweep {','.join(str(a) for a in alphas)}` first."
        )

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

    wealth_by_alpha: Dict[float, torch.Tensor] = {}
    summary_by_alpha: Dict[str, Dict[str, float]] = {}
    for alpha, (policy_obj, sequence_policy) in policies.items():
        if hasattr(policy_obj, "eval"):
            policy_obj.eval()
        with torch.no_grad():
            wealth, cost = environment.simulate_with_costs(
                policy_obj, prices, implied_vol, sequence_policy=sequence_policy
            )
        wealth_by_alpha[alpha] = wealth
        summary_by_alpha[str(alpha)] = _summarize_strategy(wealth, cost)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_alpha_sweep_density(
        wealth_by_alpha, output_dir / f"alpha_sweep_{architecture}_density.png"
    )
    _plot_alpha_sweep_boxplot(
        wealth_by_alpha, output_dir / f"alpha_sweep_{architecture}_boxplot.png"
    )

    summary = {
        "config": {
            "architecture": architecture,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "strike": strike,
            "proportional_fee_bps": proportional_fee * 1e4,
            "implied_vol": implied_vol,
        },
        "alphas": summary_by_alpha,
    }
    with open(output_dir / f"alpha_sweep_{architecture}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# Demo entrypoint: load every available trained policy checkpoint (mlp, rnn,
# lstm, gru), falling back to a short demo MLP training run if none exist,
# then backtest all of them together under stress conditions
# -----------------------------------------------------------------------------


def _train_demo_policy(
    strike: float, implied_vol: float, device: torch.device
) -> HedgingAgent:
    """Trains a small HedgingAgent (MLP) on in-distribution market_gan.py paths.

    Fallback for when no checkpoint exists at checkpoints/hedging_agent.pt:
    a short training loop gives `run_backtest` a non-trivial policy to
    stress-test. Prefer `python src/policy/train_policy.py` for a properly
    converged policy, and for the RNN/LSTM/GRU variants.
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


def _load_policy_checkpoint(
    checkpoint_path: Annotated[Path, "path to a checkpoint saved by policy/train_policy.py"],
) -> Annotated[
    Optional[Tuple[Any, bool, Dict[str, Any]]],
    "(policy, sequence_policy, saved_args) or None if the checkpoint doesn't exist",
]:
    """Loads a policy checkpoint, reconstructing the right class from its saved
    `architecture` field ('mlp' -> HedgingAgent, 'rnn'/'lstm'/'gru' -> RecurrentHedgingAgent).
    """
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_args = checkpoint["args"]
    architecture = policy_args.get("architecture", "mlp")

    if architecture == "mlp":
        policy = HedgingAgent(
            hidden_dim=policy_args["hidden_dim"],
            num_hidden_layers=policy_args["num_hidden_layers"],
        )
        sequence_policy = False
    else:
        policy = RecurrentHedgingAgent(
            cell_type=architecture,
            hidden_dim=policy_args["rnn_hidden_dim"],
            num_layers=policy_args["rnn_num_layers"],
        )
        sequence_policy = True

    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy, sequence_policy, policy_args


def _load_all_policies(
    checkpoint_dir: Annotated[Path, "directory containing hedging_agent*.pt checkpoints"],
    device: torch.device,
) -> Annotated[
    Tuple[Dict[str, Tuple[Any, bool]], float, float],
    "(policies, strike, implied_vol) -- policies maps display name -> (policy, sequence_policy)",
]:
    """Loads every available architecture's checkpoint, falling back to a
    short demo MLP training run if none exist at all.
    """
    checkpoint_paths = {
        "mlp": checkpoint_dir / "hedging_agent.pt",
        "rnn": checkpoint_dir / "hedging_agent_rnn.pt",
        "lstm": checkpoint_dir / "hedging_agent_lstm.pt",
        "gru": checkpoint_dir / "hedging_agent_gru.pt",
    }

    policies: Dict[str, Tuple[Any, bool]] = {}
    strike: Optional[float] = None
    implied_vol: Optional[float] = None

    for architecture, path in checkpoint_paths.items():
        loaded = _load_policy_checkpoint(path)
        if loaded is None:
            print(f"No checkpoint at {path} -- skipping {ARCHITECTURE_DISPLAY_NAMES[architecture]}.")
            continue
        policy, sequence_policy, policy_args = loaded
        display_name = ARCHITECTURE_DISPLAY_NAMES[architecture]
        policies[display_name] = (policy, sequence_policy)
        strike, implied_vol = policy_args["strike"], policy_args["implied_vol"]
        print(f"Loaded {display_name} from {path}")

    if policies:
        return policies, strike, implied_vol

    strike, implied_vol = 1.0, 0.30
    print(
        "No policy checkpoints found; training a short demo MLP policy. "
        "Run `python src/policy/train_policy.py --architecture {mlp,rnn,lstm,gru}` "
        "first for properly converged policies across all architectures."
    )
    demo_policy = _train_demo_policy(strike, implied_vol, device)
    return {"MLP": (demo_policy, False)}, strike, implied_vol


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    policies, strike, implied_vol = _load_all_policies(Path("checkpoints"), device)

    result = run_backtest(
        policies=policies,
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

    # Multi-alpha sweep demonstration (paper Figures 20-21): requires
    # checkpoints from `train_policy.py --architecture mlp --alpha-sweep ...`.
    sweep_alphas = [0.5, 0.75, 0.9, 0.95, 0.99]
    try:
        sweep_result = run_alpha_sweep_backtest(
            architecture="mlp",
            alphas=sweep_alphas,
            strike=strike,
            batch_size=2000,
            seq_len=30,
            proportional_fee=0.003,
            implied_vol=implied_vol,
            low_vol=0.15,
            high_vol=0.60,
            switch_prob=0.10,
            output_dir="results",
            seed=42,
        )
        print(json.dumps(sweep_result, indent=2))
    except FileNotFoundError as e:
        print(e)
