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

from backtester.plotting import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_SURFACE,
    COLOR_TEXT_PRIMARY,
    plot_boxplot_comparison,
    plot_cvar_comparison,
    plot_delta_convexity,
    plot_pnl_distribution,
    plot_transaction_costs,
    style_axes,
)
from common.black_scholes import BlackScholesDeltaPolicy  # noqa: E402
from common.checkpoints import checkpoint_filename  # noqa: E402
from common.stats import excess_kurtosis, skewness  # noqa: E402
from environment.market_env import MarketEnvironment, estimate_premium_monte_carlo  # noqa: E402
from generator.market_gan import Generator  # noqa: E402
from loss.cvar import CVaRLoss  # noqa: E402
from policy.hedging_agent import HedgingAgent, RecurrentHedgingAgent  # noqa: E402
from policy.train_policy import PolicyTrainer  # noqa: E402

# Maps train_policy.py's --architecture choice to a display name.
# Doubles as the architecture enumeration `_load_all_policies` iterates to
# build its checkpoint_paths dict -- adding a key here also changes which
# checkpoints get loaded, not just how they're displayed.
ARCHITECTURE_DISPLAY_NAMES: Dict[str, str] = {
    "mlp": "MLP",
    "rnn": "Basic RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


# -----------------------------------------------------------------------------
# 1. Analytic Black-Scholes delta for a European call
# -----------------------------------------------------------------------------


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
        "skewness": skewness(wealth),
        "excess_kurtosis": excess_kurtosis(wealth),
        "mean_transaction_cost": total_cost.mean().item(),
        "total_transaction_cost": total_cost.sum().item(),
    }


def tail_risk_summary(
    wealth: Annotated[torch.Tensor, "[Batch] terminal wealth"],
    thresholds: Annotated[
        Tuple[float, ...],
        "wealth levels to count paths below; RESULTS.md's checkpoint scan "
        "('Catastrophic tail risk, invisible below ~500,000 test paths') "
        "used -50 (a loss > 25x the option premium) and -10",
    ] = (-50.0, -10.0),
) -> Annotated[
    Dict[str, float],
    "count/fraction of paths below each threshold, plus the single worst loss -- "
    "the mean/CVaR/skew/kurtosis columns in _summarize_strategy all looked ordinary "
    "for the affected checkpoints; only these per-path tail counts caught the failure",
]:
    result: Dict[str, float] = {"worst_loss": wealth.min().item()}
    for threshold in thresholds:
        below = wealth < threshold
        key = f"below_{threshold:g}"
        result[f"{key}_count"] = int(below.sum().item())
        result[f"{key}_fraction"] = below.float().mean().item()
    return result


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
    batch_size: Annotated[
        int, "number of stress test paths; 500,000 matches the scale of Part I's paper-specified test set"
    ] = 500_000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    implied_vol: Annotated[float, "implied vol fed into every policy's state"] = 0.30,
    low_vol: Annotated[float, "stress regime: calm-period volatility"] = 0.15,
    high_vol: Annotated[float, "stress regime: turbulent-period volatility"] = 0.60,
    switch_prob: Annotated[float, "stress regime: per-step switch probability"] = 0.10,
    dt: Annotated[float, "time increment per step"] = 1.0,
    output_dir: Annotated[Union[str, Path], "directory for plots and JSON summary"] = "results",
    seed: Optional[int] = 0,
    include_premium: Annotated[
        bool, "include P0 (Monte Carlo E[Payoff(S_T)] under this process) in wealth -- see math_spec.md section 1.1"
    ] = True,
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

    premium = 0.0
    if include_premium:
        premium_generator = torch.Generator().manual_seed(seed + 1) if seed is not None else None
        premium = estimate_premium_monte_carlo(
            lambda n: simulate_regime_switching_paths(
                n, seq_len, s0=strike, dt=dt, low_vol=low_vol, high_vol=high_vol,
                switch_prob=switch_prob, generator=premium_generator,
            ),
            strike=strike,
        )

    environment = MarketEnvironment(strike=strike, proportional_fee=proportional_fee, dt=dt, premium=premium)

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
                policy_obj, prices, implied_vol, sequence_policy=sequence_policy, chunk_size=50_000
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
            "premium": premium,
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

    plot_pnl_distribution(wealth_by_strategy, output_dir / "pnl_distribution.png")
    plot_boxplot_comparison(wealth_by_strategy, output_dir / "pnl_boxplot.png")
    plot_cvar_comparison(strategy_summary, output_dir / "cvar_comparison.png")
    plot_transaction_costs(cost_by_strategy, output_dir / "transaction_costs.png")
    plot_delta_convexity(
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
    suffix: Annotated[
        str, "'_timegan' for TimeGAN-trained alpha-sweep checkpoints, '' (default) for WGAN-GP"
    ] = "",
) -> Dict[float, Tuple[Any, bool]]:
    """Loads per-alpha checkpoints saved by `train_policy.py --alpha-sweep ...`."""
    policies: Dict[float, Tuple[Any, bool]] = {}
    for alpha in alphas:
        path = checkpoint_dir / checkpoint_filename(architecture, alpha=alpha, suffix=suffix)
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
    style_axes(ax)
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
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def run_alpha_sweep_backtest(
    architecture: Annotated[str, "'mlp', 'rnn', 'lstm', or 'gru'"],
    alphas: Annotated[List[float], "CVaR alphas to compare, e.g. [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.997]"],
    strike: Annotated[float, "option strike price K"],
    batch_size: Annotated[
        int, "number of stress test paths; 500,000 matches the scale of Part I's paper-specified test set"
    ] = 500_000,
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
    include_premium: Annotated[
        bool, "include P0 (Monte Carlo E[Payoff(S_T)] under this process) in wealth -- see math_spec.md section 1.1"
    ] = True,
    suffix: Annotated[
        str, "'_timegan' for TimeGAN-trained alpha-sweep checkpoints, '' (default) for WGAN-GP"
    ] = "",
) -> Annotated[Dict, "per-alpha summary, also written to <output_dir>/alpha_sweep_<architecture>_summary.json"]:
    policies = load_alpha_sweep_checkpoints(architecture, alphas, Path(checkpoint_dir), suffix=suffix)
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

    premium = 0.0
    if include_premium:
        premium_generator = torch.Generator().manual_seed(seed + 1) if seed is not None else None
        premium = estimate_premium_monte_carlo(
            lambda n: simulate_regime_switching_paths(
                n, seq_len, s0=strike, dt=dt, low_vol=low_vol, high_vol=high_vol,
                switch_prob=switch_prob, generator=premium_generator,
            ),
            strike=strike,
        )

    environment = MarketEnvironment(strike=strike, proportional_fee=proportional_fee, dt=dt, premium=premium)

    wealth_by_alpha: Dict[float, torch.Tensor] = {}
    summary_by_alpha: Dict[str, Dict[str, float]] = {}
    for alpha, (policy_obj, sequence_policy) in policies.items():
        if hasattr(policy_obj, "eval"):
            policy_obj.eval()
        with torch.no_grad():
            wealth, cost = environment.simulate_with_costs(
                policy_obj, prices, implied_vol, sequence_policy=sequence_policy, chunk_size=50_000
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
            "premium": premium,
        },
        "alphas": summary_by_alpha,
    }
    with open(output_dir / f"alpha_sweep_{architecture}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# 6. Checkpoint tail-risk scan (RESULTS.md's "Catastrophic tail risk, invisible
# below ~500,000 test paths"): a committed, reproducible version of what was
# previously an ad hoc shell diagnostic. Mean/CVaR/skew/kurtosis in
# _summarize_strategy all looked ordinary for the affected checkpoints; only
# per-path counts against a fixed loss threshold caught it.
# -----------------------------------------------------------------------------


def scan_checkpoint_tail_risk(
    checkpoint_dir: Annotated[Union[str, Path], "directory with hedging_agent*.pt checkpoints"] = "checkpoints",
    batch_size: Annotated[int, "500,000 matches the scale that first surfaced this failure mode"] = 500_000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    low_vol: Annotated[float, "stress regime: calm-period volatility"] = 0.15,
    high_vol: Annotated[float, "stress regime: turbulent-period volatility"] = 0.60,
    switch_prob: Annotated[float, "stress regime: per-step switch probability"] = 0.10,
    dt: Annotated[float, "time increment per step"] = 1.0,
    seed: Annotated[int, "matches the seed used for RESULTS.md's own scan and main stress-test table"] = 42,
    thresholds: Annotated[Tuple[float, ...], "wealth levels to count paths below"] = (-50.0, -10.0),
    include_premium: bool = True,
    alpha_sweep_alphas: Tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.997),
) -> Annotated[
    Dict[str, Dict[str, float]],
    "{display_name: tail_risk_summary(...) | {'mean_transaction_cost': ...}} for every checkpoint found",
]:
    """Loads every available checkpoint (WGAN-GP, TimeGAN, and the alpha-sweep
    MLP set) and reports per-path tail-loss counts on the same regime-switching
    stress scenario used throughout this file, at the scale that first made
    this failure mode visible.
    """
    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device("cpu")

    policy_groups: List[Tuple[str, Dict[str, Tuple[Any, bool]], float, float]] = []
    for suffix, label in [("", ""), ("_timegan", " (TimeGAN)")]:
        policies, strike, implied_vol = _load_all_policies(checkpoint_dir, device, suffix=suffix)
        if policies and strike is not None:
            policy_groups.append((label, policies, strike, implied_vol))

    if policy_groups:
        _, _, base_strike, base_implied_vol = policy_groups[0]
        alpha_policies = load_alpha_sweep_checkpoints("mlp", list(alpha_sweep_alphas), checkpoint_dir)
        if alpha_policies:
            policy_groups.append((
                "",
                {f"MLP (alpha={alpha:g})": policy_and_flag for alpha, policy_and_flag in alpha_policies.items()},
                base_strike,
                base_implied_vol,
            ))

    results: Dict[str, Dict[str, float]] = {}
    for label, policies, group_strike, implied_vol in policy_groups:
        path_generator = torch.Generator().manual_seed(seed)
        prices = simulate_regime_switching_paths(
            batch_size, seq_len, s0=group_strike, dt=dt, low_vol=low_vol, high_vol=high_vol,
            switch_prob=switch_prob, generator=path_generator,
        )

        premium = 0.0
        if include_premium:
            premium_generator = torch.Generator().manual_seed(seed + 1)
            premium = estimate_premium_monte_carlo(
                lambda n, s=group_strike: simulate_regime_switching_paths(
                    n, seq_len, s0=s, dt=dt, low_vol=low_vol, high_vol=high_vol,
                    switch_prob=switch_prob, generator=premium_generator,
                ),
                strike=group_strike,
            )
        environment = MarketEnvironment(strike=group_strike, proportional_fee=proportional_fee, dt=dt, premium=premium)

        for name, (policy_obj, sequence_policy) in policies.items():
            if hasattr(policy_obj, "eval"):
                policy_obj.eval()
            with torch.no_grad():
                wealth, cost = environment.simulate_with_costs(
                    policy_obj, prices, implied_vol, sequence_policy=sequence_policy, chunk_size=50_000
                )
            summary = tail_risk_summary(wealth, thresholds)
            summary["mean_transaction_cost"] = cost.mean().item()
            results[f"{name}{label}"] = summary

    return results


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
    policy = HedgingAgent(hidden_dim=32, num_hidden_layers=2, strike=strike)
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
            strike=policy_args["strike"],
        )
        sequence_policy = False
    else:
        policy = RecurrentHedgingAgent(
            cell_type=architecture,
            hidden_dim=policy_args["rnn_hidden_dim"],
            num_layers=policy_args["rnn_num_layers"],
            strike=policy_args["strike"],
            implied_vol=policy_args["implied_vol"],
            time_to_maturity=policy_args["dt"] * (policy_args["seq_len"] - 1),
        )
        sequence_policy = True

    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy, sequence_policy, policy_args


def _load_all_policies(
    checkpoint_dir: Annotated[Path, "directory containing hedging_agent*.pt checkpoints"],
    device: torch.device,
    suffix: Annotated[
        str, "checkpoint filename suffix, e.g. '_timegan' for policies trained against TimeGAN"
    ] = "",
) -> Annotated[
    Tuple[Dict[str, Tuple[Any, bool]], float, float],
    "(policies, strike, implied_vol) -- policies maps display name -> (policy, sequence_policy)",
]:
    """Loads every available architecture's checkpoint.

    With the default (empty) suffix, falls back to a short demo MLP training
    run if none exist at all -- this is the primary WGAN-GP-trained
    comparison the CLI always runs. A non-empty suffix (e.g. '_timegan')
    looks for a distinct, parallel set of checkpoints instead and returns an
    empty dict with no fallback training if none are found, since that's an
    optional second comparison, not the primary one.
    """
    checkpoint_paths = {
        arch: checkpoint_dir / checkpoint_filename(arch, suffix=suffix)
        for arch in ARCHITECTURE_DISPLAY_NAMES
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

    if policies or suffix:
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
        batch_size=500_000,
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
    # Includes 0.995/0.997 to match the paper's own Part II alpha grid
    # ({0.5, 0.75, 0.99, 0.995, 0.997}), alongside this repo's own 0.9/0.95.
    sweep_alphas = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.997]
    try:
        sweep_result = run_alpha_sweep_backtest(
            architecture="mlp",
            alphas=sweep_alphas,
            strike=strike,
            batch_size=500_000,
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

    # Optional second comparison: policies trained against TimeGAN instead of
    # the single-feature WGAN-GP (checkpoints/*_timegan.pt, from
    # `train_policy.py --generator-type timegan`). Same stress-test scenario
    # (regime-switching GBM, independent of either generator) for a direct,
    # apples-to-apples comparison against the run above.
    timegan_policies, timegan_strike, timegan_implied_vol = _load_all_policies(
        Path("checkpoints"), device, suffix="_timegan"
    )
    if timegan_policies:
        timegan_result = run_backtest(
            policies=timegan_policies,
            strike=timegan_strike,
            batch_size=500_000,
            seq_len=30,
            proportional_fee=0.003,
            implied_vol=timegan_implied_vol,
            low_vol=0.15,
            high_vol=0.60,
            switch_prob=0.10,
            output_dir="results/timegan",
            seed=42,
        )
        print(json.dumps(timegan_result, indent=2))

    # Checkpoint tail-risk scan (RESULTS.md's "Catastrophic tail risk,
    # invisible below ~500,000 test paths") is deliberately *not* run here:
    # it re-simulates paths and re-estimates the premium per checkpoint
    # group, on top of what run_backtest/run_alpha_sweep_backtest already
    # computed above, tripling the 500,000-path cost this script already
    # pays. Call `scan_checkpoint_tail_risk` directly (see its docstring,
    # and tests/test_tail_risk.py for a fast 50,000-path regression version)
    # when the per-path tail-loss breakdown specifically is needed.
