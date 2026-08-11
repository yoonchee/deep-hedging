"""Tests for the GAN fidelity check (src/generator/validate.py)."""

import pytest
import torch

from generator.validate import (
    ABS_AUTOCORR_DIFF_WARNING_THRESHOLD,
    DIVERSITY_OVERSHOOT_WARNING_THRESHOLD,
    DIVERSITY_WARNING_THRESHOLD,
    KURTOSIS_WARNING_THRESHOLD,
    SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD,
    SKEW_WARNING_THRESHOLD,
    STEP_VOL_RATIO_HIGH_THRESHOLD,
    validate_generator_fidelity,
)


def _make_diverse_paths(
    batch_size: int, seq_len: int, seed: int, drift: float = 0.0
) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.randn(batch_size, seq_len - 1) * 0.1 + drift
    log_prices = torch.cat([torch.zeros(batch_size, 1), torch.cumsum(z, dim=1)], dim=1)
    return torch.exp(log_prices).unsqueeze(-1)


def _make_fat_tailed_paths(
    batch_size: int, seq_len: int, seed: int, crash_prob: float = 0.08, crash_size: float = -1.5
) -> torch.Tensor:
    # A rare, single per-path crash (not many small per-step jumps, which
    # wash out into just a wider Gaussian via the CLT): most paths look
    # ordinary, but a minority carry one large negative shock -- real
    # markets' crash risk, giving strong negative skew and fat tails
    # (positive excess kurtosis), unlike a pure Gaussian generator.
    torch.manual_seed(seed)
    z = torch.randn(batch_size, seq_len - 1) * 0.1
    has_crash = torch.rand(batch_size) < crash_prob
    crash_step = torch.randint(0, seq_len - 1, (batch_size,))
    crash_return = torch.zeros(batch_size, seq_len - 1)
    crashing = has_crash.nonzero(as_tuple=True)[0]
    crash_return[crashing, crash_step[crashing]] = crash_size
    log_prices = torch.cat(
        [torch.zeros(batch_size, 1), torch.cumsum(z + crash_return, dim=1)], dim=1
    )
    return torch.exp(log_prices).unsqueeze(-1)


def _make_paths_with_inflated_step_vol(
    batch_size: int, seq_len: int, seed: int, extra_step_swing: float = 0.3
) -> torch.Tensor:
    # Same terminal distribution as _make_diverse_paths: every consecutive
    # PAIR of steps (z_2k, z_2k+1) is replaced by (half+swing, half-swing),
    # where half = (z_2k + z_2k+1) / 2 -- preserving their two-step sum
    # exactly (so the terminal price is bit-for-bit identical to the
    # unperturbed version) while inflating per-step volatility and
    # injecting strong negative lag-1 autocorrelation within each pair.
    # This is the exact failure mode RESULTS.md's investigation found in a
    # real TimeGAN checkpoint: terminal statistics indistinguishable from
    # real, per-step dynamics badly miscalibrated.
    torch.manual_seed(seed)
    z = torch.randn(batch_size, seq_len - 1) * 0.1
    n_pairs = (seq_len - 1) // 2
    pair_sum = z[:, : 2 * n_pairs : 2] + z[:, 1 : 2 * n_pairs : 2]
    half = pair_sum / 2
    sign = torch.where(torch.rand_like(half) < 0.5, 1.0, -1.0)
    swing = extra_step_swing * sign
    z_new = z.clone()
    z_new[:, : 2 * n_pairs : 2] = half + swing
    z_new[:, 1 : 2 * n_pairs : 2] = half - swing
    log_prices = torch.cat([torch.zeros(batch_size, 1), torch.cumsum(z_new, dim=1)], dim=1)
    return torch.exp(log_prices).unsqueeze(-1)


def _make_paths_with_autocorrelation_only(batch_size: int, seq_len: int, seed: int) -> torch.Tensor:
    # Forces a strict +,-,+,-,... sign alternation onto each step's
    # magnitude (|z_t|), rather than its original random sign -- Var(z_new)
    # only depends on the magnitudes (E[z_new^2] = E[z^2] regardless of
    # sign pattern), so per-step std is preserved almost exactly, but every
    # consecutive pair now has opposite sign by construction, injecting
    # strong negative lag-1 autocorrelation in near-isolation from every
    # other statistic in this file.
    torch.manual_seed(seed)
    z = torch.randn(batch_size, seq_len - 1) * 0.1
    sign = torch.tensor([(-1.0) ** t for t in range(seq_len - 1)])
    z_new = z.abs() * sign
    log_prices = torch.cat([torch.zeros(batch_size, 1), torch.cumsum(z_new, dim=1)], dim=1)
    return torch.exp(log_prices).unsqueeze(-1)


def _make_collapsed_paths(batch_size: int, seq_len: int, decline: float = -0.6) -> torch.Tensor:
    # Every path is (almost) identical: a fixed linear decline plus tiny noise.
    base = torch.linspace(0.0, decline, seq_len)
    log_prices = base.unsqueeze(0).expand(batch_size, -1) + torch.randn(batch_size, seq_len) * 1e-4
    return torch.exp(log_prices).unsqueeze(-1)


def test_validate_generator_fidelity_flags_mode_collapse(tmp_path) -> None:
    real = _make_diverse_paths(batch_size=500, seq_len=30, seed=0)
    synthetic = _make_collapsed_paths(batch_size=500, seq_len=30)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert summary["diversity_ratio"] < DIVERSITY_WARNING_THRESHOLD
    assert "WARNING" in summary["verdict"]


def test_validate_generator_fidelity_flags_diversity_overshoot(tmp_path) -> None:
    # This repo hit this exact failure mode: a mode-collapse fix (sigmoid ->
    # tanh latents) overshot to 214-224% of real diversity, and this checker
    # printed "OK" the whole time because only the low-diversity side was
    # ever checked (see RESULTS.md's TimeGAN section). Badly over-dispersed
    # synthetic data is just as capable of producing a degenerate downstream
    # policy as mode collapse, only via the opposite mechanism (training on
    # tail risk the test distribution doesn't have).
    real = _make_diverse_paths(batch_size=500, seq_len=30, seed=0)
    synthetic = _make_diverse_paths(batch_size=500, seq_len=30, seed=1)
    # Scale up synthetic's spread directly (bypassing the noise-generation
    # helper, which only controls per-step scale, not terminal spread
    # precisely) to land comfortably over the threshold.
    synthetic = synthetic.pow(3)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert summary["diversity_ratio"] > DIVERSITY_OVERSHOOT_WARNING_THRESHOLD
    assert "WARNING" in summary["verdict"]
    assert "over-dispersed" in summary["verdict"]


def test_validate_generator_fidelity_flags_mean_bias_with_healthy_diversity(tmp_path) -> None:
    # Same per-step noise scale as real (so diversity is comparable), but
    # with an added constant drift -- diversity alone would call this
    # healthy; the mean-bias check must not.
    real = _make_diverse_paths(batch_size=500, seq_len=30, seed=0)
    synthetic = _make_diverse_paths(batch_size=500, seq_len=30, seed=1, drift=-0.1)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert summary["diversity_ratio"] >= DIVERSITY_WARNING_THRESHOLD
    assert abs(summary["mean_bias_in_std"]) > 2.0
    assert "WARNING" in summary["verdict"]
    assert "wrong distribution location" in summary["verdict"]


def test_validate_generator_fidelity_passes_for_comparable_diversity(tmp_path) -> None:
    real = _make_diverse_paths(batch_size=500, seq_len=30, seed=0)
    synthetic = _make_diverse_paths(batch_size=500, seq_len=30, seed=1)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert summary["diversity_ratio"] >= DIVERSITY_WARNING_THRESHOLD
    assert "OK" in summary["verdict"]


def test_validate_generator_fidelity_flags_tail_shape_mismatch(tmp_path) -> None:
    # Real data has crash risk (negative skew, fat tails); synthetic is a
    # plain symmetric Gaussian -- mean/diversity checks alone would miss
    # this entirely, since the generator can still get the center and
    # spread roughly right (this is exactly what happened with the actual
    # market_gan.pt checkpoint before this check existed).
    real = _make_fat_tailed_paths(batch_size=3000, seq_len=30, seed=0)
    synthetic = _make_diverse_paths(batch_size=3000, seq_len=30, seed=1)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert (
        abs(summary["skew_diff"]) > SKEW_WARNING_THRESHOLD
        or abs(summary["kurtosis_diff"]) > KURTOSIS_WARNING_THRESHOLD
    )
    assert "WARNING" in summary["verdict"]
    assert (
        "tail asymmetry" in summary["verdict"] or "fat-tail risk" in summary["verdict"]
    )


def test_validate_generator_fidelity_flags_step_vol_mismatch_despite_matching_terminal_distribution(
    tmp_path,
) -> None:
    # The core scenario this check exists for: a generator whose terminal
    # distribution is indistinguishable from real (same diversity ratio,
    # mean, skew, kurtosis -- every check above would print "OK") but whose
    # per-step path dynamics are badly miscalibrated. This is exactly what
    # RESULTS.md's investigation found in a real, terminal-"OK" TimeGAN
    # checkpoint (2x real per-step volatility) that produced catastrophic-
    # tail policies -- see the "Investigating why the best-fidelity
    # generator produced the worst policies" writeup.
    real = _make_diverse_paths(batch_size=1000, seq_len=30, seed=0)
    synthetic = _make_paths_with_inflated_step_vol(batch_size=1000, seq_len=30, seed=1)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    # Terminal-distribution checks stay clean -- this is the whole point.
    assert DIVERSITY_WARNING_THRESHOLD <= summary["diversity_ratio"] <= DIVERSITY_OVERSHOOT_WARNING_THRESHOLD
    assert abs(summary["skew_diff"]) <= SKEW_WARNING_THRESHOLD
    assert abs(summary["kurtosis_diff"]) <= KURTOSIS_WARNING_THRESHOLD
    # But the path-dynamics check catches what they all missed.
    assert summary["step_vol_ratio"] > STEP_VOL_RATIO_HIGH_THRESHOLD
    assert "WARNING" in summary["verdict"]
    assert "per-step volatility" in summary["verdict"]


def test_validate_generator_fidelity_flags_autocorrelation_mismatch_in_isolation(tmp_path) -> None:
    # Same per-step std as real (checked directly below) -- only the sign
    # pattern differs (strict alternation vs. random), isolating the
    # *signed*-return autocorrelation check from every other statistic in
    # this file, including the |return| (volatility-clustering) check:
    # flipping signs while keeping magnitudes in their original order and
    # relative sequence leaves |return| -- and its own autocorrelation --
    # completely untouched, a clean demonstration that momentum and
    # volatility-clustering are independent path-dynamics failure modes.
    real = _make_diverse_paths(batch_size=1000, seq_len=30, seed=0)
    synthetic = _make_paths_with_autocorrelation_only(batch_size=1000, seq_len=30, seed=0)

    summary = validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert summary["step_vol_ratio"] == pytest.approx(1.0, abs=0.01)
    assert abs(summary["signed_autocorr_diff"]) > SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD
    assert abs(summary["abs_autocorr_diff"]) < ABS_AUTOCORR_DIFF_WARNING_THRESHOLD
    assert "WARNING" in summary["verdict"]
    assert "momentum" in summary["verdict"] or "autocorrelation" in summary["verdict"]


def test_validate_generator_fidelity_writes_outputs(tmp_path) -> None:
    real = _make_diverse_paths(batch_size=200, seq_len=20, seed=0)
    synthetic = _make_diverse_paths(batch_size=200, seq_len=20, seed=1)

    validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert (tmp_path / "gan_fidelity_terminal_returns.png").exists()
    assert (tmp_path / "gan_fidelity_sample_paths.png").exists()
    assert (tmp_path / "gan_fidelity_summary.json").exists()
