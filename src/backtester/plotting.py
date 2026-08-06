"""Shared chart library for hedging-strategy comparisons (src/backtester/plotting.py).

Used by both the stress-test backtester (evaluate.py) and the Part I
frictionless replication study (replicate_part1.py), so both report on the
same fixed per-strategy colors and axis styling.
"""

import math
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

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


def style_axes(ax: "plt.Axes") -> None:
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


def ordered_strategy_names(names: Annotated[Any, "iterable of strategy names present"]) -> List[str]:
    """Orders strategy names by their fixed slot in STRATEGY_COLORS."""
    present = set(names)
    return [name for name in STRATEGY_COLORS if name in present]


def plot_pnl_distribution(
    wealth_by_strategy: Dict[str, torch.Tensor],
    path: Path,
    title: Annotated[str, "chart title"] = "Final PnL Distribution",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_SURFACE)
    bins = 40
    for name in ordered_strategy_names(wealth_by_strategy):
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
    ax.set_title(title)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_boxplot_comparison(
    wealth_by_strategy: Dict[str, torch.Tensor],
    path: Path,
    title: Annotated[str, "chart title"] = "PnL Comparison Across Strategies",
) -> None:
    """Boxplot of terminal wealth per strategy (paper Figures 4/7/10 style)."""
    names = ordered_strategy_names(wealth_by_strategy)
    data = [wealth_by_strategy[name].detach().numpy() for name in names]
    colors = [STRATEGY_COLORS[name] for name in names]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=COLOR_SURFACE)
    bplot = ax.boxplot(data, patch_artist=True, tick_labels=names, widths=0.5)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bplot["medians"]:
        median.set_color(COLOR_TEXT_PRIMARY)
    ax.set_ylabel("Terminal Wealth (PnL)")
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_cvar_comparison(summary: Dict[str, Dict[str, float]], path: Path) -> None:
    names = ordered_strategy_names(summary.keys())
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
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_transaction_costs(cost_by_strategy: Dict[str, torch.Tensor], path: Path) -> None:
    names = ordered_strategy_names(cost_by_strategy)
    totals = [cost_by_strategy[name].sum().item() for name in names]
    colors = [STRATEGY_COLORS[name] for name in names]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=COLOR_SURFACE)
    ax.bar(names, totals, color=colors, width=0.5, zorder=2)
    ax.set_ylabel("Total Transaction Cost (aggregate over all paths)")
    ax.set_title("Transaction Costs Incurred")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Delta-convexity diagnostic (paper Figures 5/8/11): confirms the learned
# policy recovers a Black-Scholes-like convex Delta-vs-Spot curve.
# -----------------------------------------------------------------------------


def delta_grid_stepwise(
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


def delta_grid_sequence(
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


def plot_delta_convexity(
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
        for name in ordered_strategy_names(all_strategies.keys()):
            policy_obj, sequence_policy = all_strategies[name]
            if hasattr(policy_obj, "eval"):
                policy_obj.eval()
            if sequence_policy:
                deltas = delta_grid_sequence(policy_obj, spot_grid, t, strike)
            else:
                deltas = delta_grid_stepwise(
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
        style_axes(ax)

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
