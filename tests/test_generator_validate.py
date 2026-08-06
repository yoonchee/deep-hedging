"""Tests for the GAN fidelity check (src/generator/validate.py)."""

import torch

from generator.validate import DIVERSITY_WARNING_THRESHOLD, validate_generator_fidelity


def _make_diverse_paths(
    batch_size: int, seq_len: int, seed: int, drift: float = 0.0
) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.randn(batch_size, seq_len - 1) * 0.1 + drift
    log_prices = torch.cat([torch.zeros(batch_size, 1), torch.cumsum(z, dim=1)], dim=1)
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


def test_validate_generator_fidelity_writes_outputs(tmp_path) -> None:
    real = _make_diverse_paths(batch_size=200, seq_len=20, seed=0)
    synthetic = _make_diverse_paths(batch_size=200, seq_len=20, seed=1)

    validate_generator_fidelity(real, synthetic, output_dir=tmp_path)

    assert (tmp_path / "gan_fidelity_terminal_returns.png").exists()
    assert (tmp_path / "gan_fidelity_sample_paths.png").exists()
    assert (tmp_path / "gan_fidelity_summary.json").exists()
