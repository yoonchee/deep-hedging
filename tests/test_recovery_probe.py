"""Coverage for `backtester/recovery_probe.py` and the finding it supports.

RESULTS.md ("Where GRU's seed variance comes from") uses the probe's
`mean_lag_steps` as a training-free predictor of a checkpoint's 500,000-path
tail risk. The probe's committed output (`sweep_data/PROBE_recovery_lag.json`)
plus the committed stress records make that claim checkable without the
gitignored checkpoints, which is what the correlation test below does.
"""

import json
from pathlib import Path

import pytest
import torch

from backtester.recovery_probe import (
    DEFAULT_DEPTHS,
    DIP_END_STEP,
    RALLY_TARGET,
    SEQ_LEN,
    build_probe_paths,
    recovery_lag,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DATA_DIR = REPO_ROOT / "sweep_data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def _spearman(x: list, y: list) -> float:
    def rank(values: list) -> list:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = position + 1
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def test_probe_paths_have_the_down_then_rally_shape_the_diagnosis_describes() -> None:
    depths = [-0.5, -1.0]
    # [Batch, Time_Steps, 1]
    prices = build_probe_paths(depths, strike=1.0)
    assert prices.shape == (len(depths), SEQ_LEN, 1)

    # [Batch, Time_Steps, 1] -> [Batch, Time_Steps] log-moneyness
    logm = prices[:, :, 0].log()
    for row, depth in enumerate(depths):
        assert logm[row, 0].item() == pytest.approx(0.0, abs=1e-6), "probe starts at-the-money"
        assert logm[row, DIP_END_STEP].item() == pytest.approx(depth, abs=1e-5), "dip bottoms out on schedule"
        assert logm[row, -1].item() == pytest.approx(RALLY_TARGET, abs=1e-4), "rally reaches the target"
        assert torch.all(logm[row, : DIP_END_STEP + 1].diff() <= 0), "down-move is monotonic"
        assert torch.all(logm[row, DIP_END_STEP:].diff() >= 0), "rally is monotonic"


def test_deeper_shocks_are_swept_past_the_generators_realistic_range() -> None:
    # The measured failure threshold sits near -0.7; the sweep has to run well
    # past it or the metric saturates and loses its resolution between seeds.
    assert min(DEFAULT_DEPTHS) <= -2.0
    assert max(DEFAULT_DEPTHS) == 0.0


def test_recovery_lag_reads_a_real_checkpoint() -> None:
    checkpoints = sorted(CHECKPOINT_DIR.glob("hedging_agent_gru*.pt"))
    if not checkpoints:
        pytest.skip("no GRU checkpoint on disk (gitignored)")

    record = recovery_lag(checkpoints[0])
    assert 0.0 <= record["mean_lag_steps"] <= SEQ_LEN
    assert len(record["final_delta_curve"]) == len(DEFAULT_DEPTHS)


def test_recovery_lag_rejects_a_non_recurrent_checkpoint() -> None:
    mlp = CHECKPOINT_DIR / "hedging_agent.pt"
    if not mlp.exists():
        pytest.skip("no MLP checkpoint on disk (gitignored)")
    # The probe feeds a whole price path; a per-step MLP policy cannot consume one.
    with pytest.raises(ValueError):
        recovery_lag(mlp)


@pytest.mark.parametrize("arm", ["gru_wgan_baseline", "gru_wgan_gradclip", "gru_tg_baseline", "gru_tg_clip"])
def test_probe_lag_ranks_gru_seeds_by_measured_tail_risk(arm: str) -> None:
    """The finding itself: within a GRU arm, the training-free lag tracks the
    seed's 500,000-path CVaR99. Checked against committed data, so it holds a
    published claim rather than re-deriving it from checkpoints.
    """
    probe = json.loads((SWEEP_DATA_DIR / "PROBE_recovery_lag.json").read_text())
    stress = {}
    for result_file in SWEEP_DATA_DIR.glob("RESULT_gru_*5seed.json"):
        stress.update(json.loads(result_file.read_text()))

    names = sorted(k for k in probe if k.startswith(arm) and k in stress)
    assert len(names) == 5, f"expected 5 seeds for {arm}, found {len(names)}"

    lags = [probe[name]["mean_lag_steps"] for name in names]
    cvar99 = [stress[name]["cvar_99"] for name in names]
    # Positive and strong, but not asserted at 1.0: seeds whose lags sit within
    # ~2 steps of each other are not reliably separated (see RESULTS.md's
    # sensitivity check), and the WGAN-GP baseline arm has three such seeds.
    assert _spearman(lags, cvar99) >= 0.7
