"""Tests for the Part I frictionless replication study (src/backtester/replicate_part1.py)."""

import torch

from backtester.replicate_part1 import GBMDataSource, run_part1_replication


def test_gbm_data_source_matches_generator_protocol() -> None:
    source = GBMDataSource(s0=100.0, vol=0.15, dt=1.0 / 360)
    z = source.sample_noise(batch_size=16, seq_len=10)
    prices = source(z)

    assert prices.shape == (16, 10, 1)
    assert torch.all(prices > 0)


def test_run_part1_replication_end_to_end(tmp_path) -> None:
    # Tiny settings: this test checks the pipeline runs and produces the
    # expected structure, not that training has converged.
    result = run_part1_replication(
        architectures=("mlp", "gru"),
        alphas=(0.5,),
        train_epochs=5,
        train_batch_size=64,
        test_batch_size=64,
        seq_len=10,
        output_dir=tmp_path,
    )

    assert "0.5" in result["alphas"]
    strategies = result["alphas"]["0.5"]
    assert set(strategies.keys()) == {"Black-Scholes Delta", "MLP", "GRU"}
    for stats in strategies.values():
        assert set(stats.keys()) == {"mean_pnl", "cvar_pnl", "skewness", "excess_kurtosis"}

    alpha_dir = tmp_path / "alpha_0_5"
    assert (alpha_dir / "pnl_distribution.png").exists()
    assert (alpha_dir / "pnl_boxplot.png").exists()
    assert (alpha_dir / "delta_convexity.png").exists()
    assert (tmp_path / "part1_summary.json").exists()
