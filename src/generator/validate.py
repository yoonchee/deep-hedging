"""Fidelity check for the WGAN-GP market generator (src/generator/validate.py).

Compares synthetic (GAN-generated) price paths against real training data to
catch generative failures -- most importantly mode collapse, where the
generator produces near-identical paths regardless of input noise -- before
a degenerate generator cascades into every policy trained against it (see
train_policy.py, which treats the generator as a fixed market simulator).

Run directly as a CLI, against an already-trained checkpoint:

    python src/generator/validate.py --generator-checkpoint checkpoints/market_gan.pt
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Allow `python src/generator/validate.py` to resolve sibling packages the
# same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.stats import (  # noqa: E402
    excess_kurtosis,
    lag1_autocorrelation,
    skewness,
    step_log_returns,
    terminal_log_return,
)
from generator.data import HistoricalPriceLoader, sample_real_prices  # noqa: E402
from generator.market_gan import Generator  # noqa: E402

COLOR_REAL = "#2a78d6"  # slot 1 blue -- fixed baseline reference
COLOR_SYNTHETIC = "#eb6834"  # slot 2 orange
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"

# Heuristic, not a hard rule: below this fraction of real diversity, the
# generator is very likely mode-collapsed rather than just imperfectly fit.
DIVERSITY_WARNING_THRESHOLD = 0.3

# Heuristic upper counterpart: this repo's own TimeGAN history hit this
# failure mode directly -- a tanh-latent fix for mode collapse overshot to
# 214-224% of real diversity, and this checker printed "OK" the whole time
# because only the low-diversity side was ever checked (see RESULTS.md's
# TimeGAN section). 1.7x sits between that failure (214-224%, should warn)
# and a later, meaningfully-improved-but-still-imperfect run (130.2%,
# shouldn't) -- over-dispersion is just as capable of producing a
# degenerate downstream policy as mode collapse, only in the opposite
# direction (training on tail risk the test distribution doesn't have,
# instead of not training on tail risk it does).
DIVERSITY_OVERSHOOT_WARNING_THRESHOLD = 1.7

# Heuristic: if the synthetic mean sits more than this many real standard
# deviations away from the real mean, the generator has learned the wrong
# location for the whole distribution (e.g. a persistent decline that isn't
# in the data) -- a distinct failure mode from low diversity, and just as
# capable of producing a degenerate downstream policy.
MEAN_BIAS_WARNING_THRESHOLD_STD = 2.0

# Heuristics for distribution SHAPE, independent of mean/spread: a generator
# can have the right center and the right diversity while still missing real
# markets' tail asymmetry (crash risk) and fat tails entirely -- this is what
# let checkpoints/market_gan.pt pass the mean/diversity checks while still
# producing policies that failed badly on tail risk in the stress-test
# backtest (see results/benchmark_summary.json before this fix).
SKEW_WARNING_THRESHOLD = 0.5
KURTOSIS_WARNING_THRESHOLD = 2.0

# Heuristics for PATH DYNAMICS, independent of every check above -- all four
# checks above only ever inspect the terminal/cumulative return distribution
# (the price at path end), which turns out to be structurally blind to a
# real failure mode: a generator can have exactly the right terminal mean,
# spread, skew, and kurtosis while still taking a wildly unrealistic *route*
# to get there (much larger or more clustered per-step moves that compound
# back toward a realistic endpoint anyway). This is exactly what let a
# terminal-distribution-"OK" TimeGAN checkpoint (diversity 110.6%, skew diff
# -0.36, kurtosis diff -1.44, all within threshold) produce policies with
# catastrophic tail risk: its per-step volatility was 2x real markets', with
# 7.7x more frequent large single-step moves and much stronger momentum and
# volatility clustering -- none of which showed up above. See RESULTS.md's
# "Investigating why the best-fidelity generator produced the worst
# policies" writeup for the full measurement this threshold is based on.
STEP_VOL_RATIO_LOW_THRESHOLD = 0.67
STEP_VOL_RATIO_HIGH_THRESHOLD = 1.5

# Lag-1 autocorrelation of raw (signed) per-step returns: momentum or
# mean-reversion structure invisible to every terminal check above. Real
# ^GSPC daily data measures close to zero (near-random-walk) over 31-day
# windows; the investigation above found a "fidelity-OK" TimeGAN checkpoint
# at a 0.40 *difference* from real (real 0.07, synthetic 0.47).
SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD = 0.25

# Lag-1 autocorrelation of |return| -- volatility clustering (ARCH effects),
# a distinct path-dynamics property from momentum in the signed returns
# above. Same investigation found a 0.43 difference (real -0.02, synthetic
# 0.41) undetected by every other check in this file.
ABS_AUTOCORR_DIFF_WARNING_THRESHOLD = 0.25


def _return_stats(x: Annotated[torch.Tensor, "[Batch] terminal log-returns"]) -> Dict[str, float]:
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "skewness": skewness(x),
        "excess_kurtosis": excess_kurtosis(x),
    }


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


def _plot_terminal_return_distribution(
    real_returns: torch.Tensor, synthetic_returns: torch.Tensor, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_SURFACE)
    bins = 40
    ax.hist(
        real_returns.detach().numpy(), bins=bins, alpha=0.55, color=COLOR_REAL, label="Real", zorder=2
    )
    ax.hist(
        synthetic_returns.detach().numpy(),
        bins=bins,
        alpha=0.55,
        color=COLOR_SYNTHETIC,
        label="Synthetic",
        zorder=2,
    )
    ax.set_xlabel("Terminal Log-Return: log(S_T / S_0)")
    ax.set_ylabel("Frequency")
    ax.set_title("Real vs. Synthetic Terminal Return Distribution")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def _plot_sample_paths(
    real_prices: torch.Tensor,
    synthetic_prices: torch.Tensor,
    path: Path,
    n_paths: Annotated[int, "number of sample paths to overlay per panel"] = 20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_SURFACE, sharey=True)
    for ax, prices, color, title in (
        (axes[0], real_prices, COLOR_REAL, "Real"),
        (axes[1], synthetic_prices, COLOR_SYNTHETIC, "Synthetic"),
    ):
        for i in range(min(n_paths, prices.shape[0])):
            ax.plot(
                prices[i, :, 0].detach().numpy(), color=color, alpha=0.5, linewidth=1.0, zorder=2
            )
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        _style_axes(ax)
    axes[0].set_ylabel("Normalized Price")
    fig.suptitle(f"{n_paths} Sample Paths: Real vs. Synthetic", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def validate_generator_fidelity(
    real_prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real (or reference) price paths"],
    synthetic_prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] GAN-generated price paths"],
    output_dir: Annotated[Path, "directory for plots and JSON summary"] = Path("results"),
) -> Annotated[Dict, "fidelity summary, also written to <output_dir>/gan_fidelity_summary.json"]:
    """Compares real and synthetic price paths to catch generative failures.

    Checks seven independent, complementary failure modes -- four on the
    terminal/cumulative return distribution, three on per-step path dynamics:

    - Diversity ratio (synthetic terminal-return std / real terminal-return
      std): mode collapse -- the generator producing near-identical paths
      regardless of noise -- shows up as this ratio collapsing toward zero.
      Badly over-dispersed synthetic data (ratio >> 1) is checked too, a
      distinct failure mode this checker used to miss entirely (see
      RESULTS.md's TimeGAN section for the 214-224%-diversity checkpoint
      that printed "OK" before this check existed).
    - Mean bias (in real standard deviations): the generator can have
      perfectly healthy diversity while still centering its whole
      distribution on the wrong location (e.g. a persistent decline that
      isn't in the real data) -- diversity alone would miss this.
    - Skew and excess-kurtosis mismatches: the generator can have the right
      center *and* the right spread while still missing real markets' tail
      asymmetry (crash risk) and fat tails entirely. This is the failure
      mode that let an earlier checkpoint pass the mean/diversity checks
      while still producing policies with badly underestimated tail risk
      once stress-tested.
    - Per-step volatility ratio, and lag-1 autocorrelation of both signed
      and absolute per-step returns: all four checks above only ever
      inspect the *terminal* price, and it turns out a generator can nail
      every one of them while taking a wildly unrealistic route to get
      there -- much larger, more momentum-heavy, more clustered per-step
      moves that still compound back toward a realistic endpoint. This is
      exactly what let a terminal-"OK" TimeGAN checkpoint (diversity
      110.6%, skew/kurtosis within threshold) produce catastrophic-tail
      policies: 2x real per-step volatility, 7.7x more frequent large
      single-step moves, invisible to every check above. See RESULTS.md's
      "Investigating why the best-fidelity generator produced the worst
      policies" writeup for the full investigation this is based on.

    All seven run automatically after every `train_gan.py` run, before any
    of them can cascade into a policy trained against the generator.
    """
    real_returns = terminal_log_return(real_prices)
    synthetic_returns = terminal_log_return(synthetic_prices)

    real_stats = _return_stats(real_returns)
    synthetic_stats = _return_stats(synthetic_returns)
    diversity_ratio = (
        synthetic_stats["std"] / real_stats["std"] if real_stats["std"] > 0 else float("nan")
    )
    mean_bias = synthetic_stats["mean"] - real_stats["mean"]
    mean_bias_in_std = mean_bias / real_stats["std"] if real_stats["std"] > 0 else float("nan")
    skew_diff = synthetic_stats["skewness"] - real_stats["skewness"]
    kurtosis_diff = synthetic_stats["excess_kurtosis"] - real_stats["excess_kurtosis"]

    # Path-dynamics statistics -- computed on per-STEP returns, not the
    # terminal return the four checks above use.
    real_step = step_log_returns(real_prices)
    synthetic_step = step_log_returns(synthetic_prices)
    real_step_std = real_step.std().item()
    synthetic_step_std = synthetic_step.std().item()
    step_vol_ratio = synthetic_step_std / real_step_std if real_step_std > 0 else float("nan")
    real_signed_autocorr = lag1_autocorrelation(real_step)
    synthetic_signed_autocorr = lag1_autocorrelation(synthetic_step)
    signed_autocorr_diff = synthetic_signed_autocorr - real_signed_autocorr
    real_abs_autocorr = lag1_autocorrelation(real_step.abs())
    synthetic_abs_autocorr = lag1_autocorrelation(synthetic_step.abs())
    abs_autocorr_diff = synthetic_abs_autocorr - real_abs_autocorr

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_terminal_return_distribution(
        real_returns, synthetic_returns, output_dir / "gan_fidelity_terminal_returns.png"
    )
    _plot_sample_paths(real_prices, synthetic_prices, output_dir / "gan_fidelity_sample_paths.png")

    problems = []
    if diversity_ratio < DIVERSITY_WARNING_THRESHOLD:
        problems.append(
            f"diversity is only {diversity_ratio:.1%} of real (threshold "
            f"{DIVERSITY_WARNING_THRESHOLD:.0%}) -- likely mode collapse"
        )
    elif diversity_ratio > DIVERSITY_OVERSHOOT_WARNING_THRESHOLD:
        problems.append(
            f"diversity is {diversity_ratio:.1%} of real (threshold "
            f"{DIVERSITY_OVERSHOOT_WARNING_THRESHOLD:.0%}) -- badly over-dispersed, a distinct "
            "failure mode from mode collapse but just as capable of producing a degenerate "
            "downstream policy"
        )
    if abs(mean_bias_in_std) > MEAN_BIAS_WARNING_THRESHOLD_STD:
        problems.append(
            f"mean is {mean_bias_in_std:+.1f} real std devs off ({synthetic_stats['mean']:+.4f} "
            f"vs. real {real_stats['mean']:+.4f}) -- generator learned the wrong distribution location"
        )
    if abs(skew_diff) > SKEW_WARNING_THRESHOLD:
        problems.append(
            f"skewness off by {skew_diff:+.2f} (synthetic {synthetic_stats['skewness']:+.2f} vs. "
            f"real {real_stats['skewness']:+.2f}) -- generator isn't capturing real tail asymmetry"
        )
    if abs(kurtosis_diff) > KURTOSIS_WARNING_THRESHOLD:
        problems.append(
            f"excess kurtosis off by {kurtosis_diff:+.2f} (synthetic "
            f"{synthetic_stats['excess_kurtosis']:+.2f} vs. real {real_stats['excess_kurtosis']:+.2f}) "
            "-- generator isn't capturing real fat-tail risk"
        )
    if step_vol_ratio == step_vol_ratio and (  # NaN-safe: NaN != NaN
        step_vol_ratio < STEP_VOL_RATIO_LOW_THRESHOLD or step_vol_ratio > STEP_VOL_RATIO_HIGH_THRESHOLD
    ):
        problems.append(
            f"per-step volatility is {step_vol_ratio:.1%} of real (synthetic std "
            f"{synthetic_step_std:.5f} vs. real {real_step_std:.5f}, threshold "
            f"{STEP_VOL_RATIO_LOW_THRESHOLD:.0%}-{STEP_VOL_RATIO_HIGH_THRESHOLD:.0%}) -- the terminal "
            "distribution can still look fine if this compounds back toward a realistic endpoint, "
            "but recurrent policies consume the whole path and are sensitive to this directly"
        )
    if signed_autocorr_diff == signed_autocorr_diff and abs(signed_autocorr_diff) > SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD:
        problems.append(
            f"per-step return autocorrelation off by {signed_autocorr_diff:+.2f} (synthetic "
            f"{synthetic_signed_autocorr:+.2f} vs. real {real_signed_autocorr:+.2f}) -- generator has "
            "unrealistic momentum/mean-reversion structure invisible to the terminal-distribution checks above"
        )
    if abs_autocorr_diff == abs_autocorr_diff and abs(abs_autocorr_diff) > ABS_AUTOCORR_DIFF_WARNING_THRESHOLD:
        problems.append(
            f"volatility clustering off by {abs_autocorr_diff:+.2f} (synthetic |return| autocorrelation "
            f"{synthetic_abs_autocorr:+.2f} vs. real {real_abs_autocorr:+.2f}) -- generator doesn't "
            "capture real markets' ARCH-effect clustering of large moves"
        )

    if problems:
        verdict = "WARNING: " + "; ".join(problems) + ". Policies trained against this generator may fail badly out of sample."
    else:
        verdict = (
            f"OK: diversity is {diversity_ratio:.1%} of real, mean bias is {mean_bias_in_std:+.1f} "
            f"real std devs, skew diff {skew_diff:+.2f}, kurtosis diff {kurtosis_diff:+.2f}, "
            f"step-vol ratio {step_vol_ratio:.1%}, autocorrelation diffs {signed_autocorr_diff:+.2f}/"
            f"{abs_autocorr_diff:+.2f} (signed/abs)."
        )
    print(verdict)

    summary = {
        "real": real_stats,
        "synthetic": synthetic_stats,
        "diversity_ratio": diversity_ratio,
        "diversity_warning_threshold": DIVERSITY_WARNING_THRESHOLD,
        "diversity_overshoot_warning_threshold": DIVERSITY_OVERSHOOT_WARNING_THRESHOLD,
        "mean_bias_in_std": mean_bias_in_std,
        "mean_bias_warning_threshold_std": MEAN_BIAS_WARNING_THRESHOLD_STD,
        "skew_diff": skew_diff,
        "skew_warning_threshold": SKEW_WARNING_THRESHOLD,
        "kurtosis_diff": kurtosis_diff,
        "kurtosis_warning_threshold": KURTOSIS_WARNING_THRESHOLD,
        "step_vol_ratio": step_vol_ratio,
        "step_vol_ratio_low_threshold": STEP_VOL_RATIO_LOW_THRESHOLD,
        "step_vol_ratio_high_threshold": STEP_VOL_RATIO_HIGH_THRESHOLD,
        "signed_autocorr_diff": signed_autocorr_diff,
        "signed_autocorr_diff_warning_threshold": SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD,
        "abs_autocorr_diff": abs_autocorr_diff,
        "abs_autocorr_diff_warning_threshold": ABS_AUTOCORR_DIFF_WARNING_THRESHOLD,
        "verdict": verdict,
    }
    with open(output_dir / "gan_fidelity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check WGAN-GP market generator fidelity against real (or reference) data."
    )
    parser.add_argument("--generator-checkpoint", type=str, default="checkpoints/market_gan.pt")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--ticker", type=str, default="^GSPC")
    parser.add_argument("--data-start", type=str, default="1950-01-03")
    parser.add_argument("--data-end", type=str, default="2021-01-25")
    parser.add_argument("--price-column", type=str, default="Adj Close")
    parser.add_argument("--data-cache-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.generator_checkpoint, map_location="cpu", weights_only=False)
    gen_args = checkpoint["args"]
    generator = Generator(
        noise_dim=gen_args["noise_dim"],
        hidden_dim=gen_args["hidden_dim"],
        num_layers=gen_args["num_layers"],
        initial_price=gen_args["s0"],
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    seq_len = gen_args["seq_len"]

    if gen_args.get("data_source") == "yfinance":
        loader = HistoricalPriceLoader(
            ticker=args.ticker,
            start=args.data_start,
            end=args.data_end,
            price_column=args.price_column,
            cache_dir=Path(args.data_cache_dir),
        )
        real_prices = loader.sample(args.batch_size, seq_len, initial_price=gen_args["s0"])
    else:
        real_prices = sample_real_prices(
            args.batch_size, seq_len, s0=gen_args["s0"], vol=gen_args.get("vol", 0.2)
        )

    with torch.no_grad():
        z = generator.sample_noise(args.batch_size, seq_len)
        synthetic_prices = generator(z)

    validate_generator_fidelity(real_prices, synthetic_prices, output_dir=Path(args.output_dir))
