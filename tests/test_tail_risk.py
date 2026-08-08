"""Regression coverage for RESULTS.md's "Catastrophic tail risk, invisible
below ~500,000 test paths" finding.

Mean wealth, CVaR, skew, and kurtosis all looked ordinary for the affected
checkpoints -- only a direct count of paths below a fixed loss threshold
caught it (see `evaluate.py::tail_risk_summary`). This file has two jobs:

1. `test_tail_risk_summary_counts_*`: a checkpoint-independent unit test of
   the counting logic itself, so it's always exercised regardless of what's
   trained locally.
2. `test_known_good_checkpoints_*` / `test_known_bad_checkpoints_*`: a
   reproduction of RESULTS.md's own checkpoint scan (previously run ad hoc
   via the shell, not committed as tests -- see `evaluate.py::scan_checkpoint_tail_risk`),
   guarding known-good checkpoints against silently regressing and
   documenting the checkpoints with the currently-unfixed failure. Trained
   checkpoints are gitignored (`checkpoints/`), so these skip cleanly on a
   fresh clone or in CI and only run where checkpoints already exist.
"""

from pathlib import Path

import pytest
import torch

from backtester.evaluate import scan_checkpoint_tail_risk, tail_risk_summary

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Same seed as RESULTS.md's own scan and the main stress-test table
# (evaluate.py's __main__). 50,000 paths (not the paper-scale 500,000 used
# in RESULTS.md) keeps this fast while still reliably reproducing the
# pattern -- verified directly: at this seed/scale every "known-bad"
# checkpoint below already shows a nonzero tail count. This is a
# sensitivity limit, not a guarantee: RESULTS.md's measured catastrophic-path
# rates range from ~1-in-650 to ~1-in-15,000 depending on the checkpoint, so
# a *newly* regressed checkpoint at the low end of that range could still
# read as clean at 50,000 paths (expected count < 4) and pass here even
# though it wouldn't at the full 500,000-path scale.
_SCAN_BATCH_SIZE = 50_000
_SCAN_SEED = 42

# RESULTS.md's "Catastrophic tail risk" table: checkpoints with 0/500,000
# paths below -50 (a loss > 25x the option premium). "MLP (alpha=0.997)" and
# "MLP (alpha=0.99)" joined this list after mechanism (a) was root-caused and
# fixed (sigmoid output saturation under CVaR's (1/(1-alpha)) gradient
# amplification -- see RESULTS.md's mechanism (a) writeup) via
# PolicyTrainer's grad_clip_norm, newly wired up to train_policy.py's
# --grad-clip-norm CLI flag; it was never wired up before, so every prior
# checkpoint in this repo trained with clipping disabled.
_KNOWN_GOOD_CHECKPOINTS = [
    "MLP", "Basic RNN", "LSTM", "MLP (TimeGAN)", "MLP (alpha=0.997)", "MLP (alpha=0.99)",
]

# Same table's checkpoints with a confirmed, not-yet-fixed catastrophic tail.
# GRU (WGAN-GP): checked directly against the same saturation diagnostic that
# found and fixed mechanism (a) -- its delta span is a healthy 1.0000 at
# every timestep checked, so this is confirmed NOT sigmoid saturation; a
# distinct, still-uncharacterized mechanism. Basic RNN/LSTM/GRU (TimeGAN):
# mechanism (b) (TimeGAN-trained recurrent policies generalizing badly to
# price extremes) for LSTM/GRU (also checked: healthy delta span, not
# saturated). Basic RNN (TimeGAN) specifically was further diagnosed and
# turned out to be a third, distinct mechanism: its vanilla-RNN hidden state
# is saturated to tanh's +-1.0 bound regardless of input (constant delta
# output) -- confirmed not fixed by grad_clip_norm, orthogonal_init, or both
# together (all tested; see RESULTS.md). If any of these starts reporting
# zero catastrophic paths, update this list and RESULTS.md's Known
# Limitations item 5 together.
_KNOWN_BAD_CHECKPOINTS = ["GRU", "Basic RNN (TimeGAN)", "LSTM (TimeGAN)", "GRU (TimeGAN)"]

_checkpoints_available = pytest.mark.skipif(
    not CHECKPOINT_DIR.exists() or not any(CHECKPOINT_DIR.glob("hedging_agent*.pt")),
    reason="no trained checkpoints found in checkpoints/ (gitignored; train locally with train_policy.py first)",
)


def test_tail_risk_summary_counts_and_fractions_correctly() -> None:
    wealth = torch.tensor([-100.0, -60.0, -20.0, -5.0, 0.0, 3.0])

    summary = tail_risk_summary(wealth, thresholds=(-50.0, -10.0))

    assert summary["worst_loss"] == pytest.approx(-100.0)
    assert summary["below_-50_count"] == 2
    assert summary["below_-50_fraction"] == pytest.approx(2 / 6)
    assert summary["below_-10_count"] == 3
    assert summary["below_-10_fraction"] == pytest.approx(3 / 6)


def test_tail_risk_summary_is_zero_when_no_path_crosses_threshold() -> None:
    wealth = torch.full((1000,), -1.0)

    summary = tail_risk_summary(wealth, thresholds=(-50.0,))

    assert summary["worst_loss"] == pytest.approx(-1.0)
    assert summary["below_-50_count"] == 0
    assert summary["below_-50_fraction"] == 0.0


@pytest.fixture(scope="module")
def tail_risk_scan() -> dict:
    if not CHECKPOINT_DIR.exists() or not any(CHECKPOINT_DIR.glob("hedging_agent*.pt")):
        pytest.skip("no trained checkpoints found in checkpoints/ (gitignored; train locally with train_policy.py first)")
    return scan_checkpoint_tail_risk(
        checkpoint_dir=CHECKPOINT_DIR, batch_size=_SCAN_BATCH_SIZE, seed=_SCAN_SEED
    )


@_checkpoints_available
@pytest.mark.parametrize("name", _KNOWN_GOOD_CHECKPOINTS)
def test_known_good_checkpoints_have_no_catastrophic_tail(tail_risk_scan: dict, name: str) -> None:
    if name not in tail_risk_scan:
        pytest.skip(f"no checkpoint loaded for {name!r}")

    summary = tail_risk_scan[name]

    # RESULTS.md's clean checkpoints show 0/500,000 below -50 and only a
    # handful (3-5) below -10; a small margin above that keeps this from
    # being flaky at the smaller batch size used here.
    assert summary["below_-50_count"] == 0, (
        f"{name} now shows paths losing more than 50x the option premium -- "
        f"this previously distinguished the clean checkpoints from the ones "
        f"with the catastrophic-tail bug (RESULTS.md, Known limitations item 5)"
    )
    assert summary["below_-10_fraction"] < 0.001


@_checkpoints_available
@pytest.mark.parametrize("name", _KNOWN_BAD_CHECKPOINTS)
def test_known_bad_checkpoints_still_show_documented_tail_risk(tail_risk_scan: dict, name: str) -> None:
    if name not in tail_risk_scan:
        pytest.skip(f"no checkpoint loaded for {name!r}")

    summary = tail_risk_scan[name]

    # This is a canary, not a guard: it currently documents an open, known
    # bug (RESULTS.md's "Catastrophic tail risk" section). If this starts
    # failing, the bug was fixed -- move `name` to _KNOWN_GOOD_CHECKPOINTS
    # and update RESULTS.md rather than deleting this test.
    assert summary["below_-50_count"] > 0, (
        f"{name} no longer shows the catastrophic tail documented in RESULTS.md -- "
        f"if this is a genuine fix, update RESULTS.md's Known limitations item 5 "
        f"and move this checkpoint to the known-good list above"
    )


@_checkpoints_available
def test_alpha_0997_checkpoint_no_longer_shows_degenerate_never_hedge_policy(tail_risk_scan: dict) -> None:
    name = "MLP (alpha=0.997)"
    if name not in tail_risk_scan:
        pytest.skip(f"no checkpoint loaded for {name!r}")

    summary = tail_risk_scan[name]

    # This checkpoint used to be a fully degenerate never-hedge policy
    # (mean_transaction_cost exactly 0, the confirmed signature of a single
    # CVaR-amplified gradient step saturating the MLP's sigmoid output layer
    # so hard that its local derivative underflows to exactly 0.0 in
    # float32 -- see RESULTS.md's mechanism (a) writeup). Retrained with
    # PolicyTrainer's grad_clip_norm (now exposed via --grad-clip-norm),
    # which keeps every step small enough to never reach that saturated
    # regime. If this starts failing, the fix regressed -- don't just widen
    # the threshold, re-check delta span / mean_transaction_cost directly.
    assert summary["below_-50_count"] == 0
    assert summary["mean_transaction_cost"] > 1e-4
