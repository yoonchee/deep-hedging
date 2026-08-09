"""Tests for the Part I frictionless replication study (src/backtester/replicate_part1.py)."""

import torch

from backtester import replicate_part1
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


def test_run_part1_replication_wires_rnn_lr_separately_from_shared_lr(tmp_path, monkeypatch) -> None:
    # Regression test for the Basic RNN weight-blowup fix (RESULTS.md,
    # "Basic RNN's Part I gap, root-caused and fixed"): `architecture_lr`
    # must resolve to `rnn_lr` for architecture="rnn" and to `lr` for every
    # other architecture. A silently inverted or dropped branch here would
    # still produce plausible-looking output at these tiny epoch counts, so
    # this asserts on the actual lr each architecture's training call
    # receives rather than on downstream training behavior.
    seen_lrs: dict = {}
    original_train_policy = replicate_part1.train_policy

    def spy_train_policy(architecture, alpha, environment, data_source, **kwargs):
        seen_lrs[architecture] = kwargs["lr"]
        return original_train_policy(architecture, alpha, environment, data_source, **kwargs)

    monkeypatch.setattr(replicate_part1, "train_policy", spy_train_policy)

    run_part1_replication(
        architectures=("mlp", "rnn"),
        alphas=(0.5,),
        train_epochs=2,
        train_batch_size=16,
        test_batch_size=16,
        seq_len=5,
        lr=7e-2,
        rnn_lr=3e-4,
        output_dir=tmp_path,
    )

    assert seen_lrs == {"mlp": 7e-2, "rnn": 3e-4}
