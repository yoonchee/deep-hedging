"""Tests for the GAN fidelity check (src/generator/validate.py)."""

import torch

from generator.validate import (
    DIVERSITY_OVERSHOOT_WARNING_THRESHOLD,
    DIVERSITY_WARNING_THRESHOLD,
    KURTOSIS_WARNING_THRESHOLD,
    SKEW_WARNING_THRESHOLD,
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


def test_validate_generator_fidelity_writes_outputs(tmp_path) -> None:
    real = _make_diverse_paths(batch_size=200, seq_len=20, seed=0)
    synthetic = _make_diverse_paths(batch_size=200, seq_len=20, seed=1)

    validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert (tmp_path / "gan_fidelity_terminal_returns.png").exists()
    assert (tmp_path / "gan_fidelity_sample_paths.png").exists()
    assert (tmp_path / "gan_fidelity_summary.json").exists()
