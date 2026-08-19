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

from backtester.evaluate import _load_policy_checkpoint, scan_checkpoint_tail_risk, tail_risk_summary

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
#
# GRU (WGAN-GP) does NOT belong here despite also being grad_clip_norm-fixed:
# its fix is a substantial improvement (34/500,000 -> 4/500,000 catastrophic
# paths), not a full 0/500,000 close, so it would fail this list's
# below_-50_count == 0 assertion. See test_gru_checkpoint_substantially_
# improved_but_not_fully_clean below and RESULTS.md's "Fix attempt" writeup.
# Note for anyone regenerating checkpoints from scratch: train_policy.py's
# --grad-clip-norm default is still None, so `--architecture gru` with no
# extra flags reproduces the *pre-fix*, more-catastrophic checkpoint, not
# the one currently at checkpoints/hedging_agent_gru.pt. Pass
# --grad-clip-norm 1.0 explicitly to reproduce the promoted one. Same trap
# for GRU (TimeGAN): `--architecture gru --generator-type timegan` with no
# extra flags reproduces the *pre-fix* checkpoint (preserved as
# checkpoints/hedging_agent_gru_timegan.pt.bak-pre-moneyness-clip-fix), not
# the promoted one -- pass `--moneyness-clip -0.15 0.10` explicitly.
#
# LSTM (TimeGAN) joined this list once mechanism (b)'s velocity-hysteresis
# fix was found and promoted (RESULTS.md's "Fix attempt continued" writeup):
# `--slow-ramp-fraction 0.05` (5-seed validated, plus a dose sweep showing
# 0.05 -- not the naively-better-looking-at-one-seed 0.10 -- is the robust
# choice). `--architecture lstm --generator-type timegan` with no extra
# flags reproduces the *pre-fix* checkpoint, not the promoted one -- pass
# `--slow-ramp-fraction 0.05` explicitly. Note this checkpoint was trained
# against a freshly-retrained TimeGAN generator (checkpoints/timegan.pt),
# not the exact one the other rows in RESULTS.md's attempt-4 table used
# (that one wasn't preserved) -- see the promoted-checkpoint caveat there.
#
# "MLP (TimeGAN)" was removed from this list after a from-scratch rebuild:
# against the surviving checkpoints/timegan.pt it shows 0/50,000 below -50
# (still not catastrophic) but 0.86% of paths below -10, ~9x this list's
# below_-10_fraction bound. That is not a code regression -- RESULTS.md
# records that attempt 4's generator, which the documented "clean, no tail
# risk" row was measured against, was never preserved. See
# test_mlp_timegan_is_no_longer_clean_against_the_surviving_generator below.
_KNOWN_GOOD_CHECKPOINTS = [
    "MLP", "Basic RNN", "LSTM", "MLP (alpha=0.997)", "MLP (alpha=0.99)",
    "LSTM (TimeGAN)",
]

# Same table's checkpoints with a confirmed, not-yet-fixed catastrophic tail.
# GRU (WGAN-GP) is NOT here either, despite still being catastrophic (see
# above) -- its post-fix catastrophic rate (4/500,000, worst loss -137.5) is
# below this file's 50,000-path scan's sensitivity: verified directly, a
# seed=42 scan at that scale reads 0/50,000 below -50 for the current
# checkpoint, which would make this list's below_-50_count > 0 assertion
# fail. Basic RNN/GRU (TimeGAN): mechanism (b) (TimeGAN-trained recurrent
# policies generalizing badly to price extremes) for GRU (checked: healthy
# delta span, not saturated). Basic RNN (TimeGAN) specifically was further
# diagnosed and turned out to be a third, distinct mechanism: its
# vanilla-RNN hidden state is saturated to tanh's +-1.0 bound regardless of
# input (constant delta output) -- confirmed not fixed by grad_clip_norm,
# orthogonal_init, or both together (all tested; see RESULTS.md). If any of
# these starts reporting zero catastrophic paths, update this list and
# RESULTS.md's Known Limitations item 5 together.
# "GRU" joined this list once its promoted --grad-clip-norm 1.0 fix was
# retracted (see below): the production checkpoint is now a plain default
# training run, which shows a catastrophic tail like the TimeGAN entries.
_KNOWN_BAD_CHECKPOINTS = ["Basic RNN (TimeGAN)", "GRU", "GRU (TimeGAN)"]

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


@_checkpoints_available
def test_mlp_timegan_is_no_longer_clean_against_the_surviving_generator(tail_risk_scan: dict) -> None:
    name = "MLP (TimeGAN)"
    if name not in tail_risk_scan:
        pytest.skip(f"no checkpoint loaded for {name!r}")

    summary = tail_risk_scan[name]

    # RESULTS.md documents this checkpoint as "clean, no tail risk", measured
    # against TimeGAN attempt 4's generator -- which was never preserved.
    # Rebuilt against the surviving checkpoints/timegan.pt it is neither
    # clean nor catastrophic: no path loses more than 50x the premium, but
    # ~0.9% of paths lose more than 10x, against the known-good list's
    # <0.1% bound. This test pins that intermediate state so it is visible
    # rather than silently absent, and so a genuine change in either
    # direction fails loudly.
    #
    # Scope note: the ~0.9% is this checkpoint's seed, not the architecture's
    # behaviour. A later 5-seed re-anchoring (RESULTS.md, "Re-anchoring all
    # four TimeGAN rows to the surviving generator") found the rate spans
    # 0.012%-0.877% and that seed 0 is the worst of the five -- two seeds land
    # inside the known-good bound. The catastrophic-path assertion below is
    # the seed-independent claim (0/500,000 at every seed measured); the
    # below_-10 band is a pin on this specific promoted checkpoint.
    assert summary["below_-50_count"] == 0, (
        "MLP (TimeGAN) has developed a catastrophic tail -- it previously had "
        "none at any generator; move it to _KNOWN_BAD_CHECKPOINTS and update "
        "RESULTS.md's TimeGAN table"
    )
    assert 0.001 < summary["below_-10_fraction"] < 0.02, (
        "MLP (TimeGAN)'s below_-10 rate left the band measured against the "
        "surviving generator -- if it dropped under 0.001 the checkpoint is "
        "clean again and belongs back in _KNOWN_GOOD_CHECKPOINTS"
    )


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "hedging_agent_gru_timegan.pt").exists(),
    reason="checkpoints/hedging_agent_gru_timegan.pt not found (gitignored; train locally first)",
)
def test_promoted_gru_timegan_checkpoint_has_no_moneyness_clip() -> None:
    # This assertion is deliberately the inverse of what it used to be.
    # --moneyness-clip was promoted for GRU (TimeGAN) on a single seed; a
    # 5-seed paired rerun found it improves 1/5 seeds, doubles mean CVaR99
    # (12.14 -> 24.97) and nearly triples mean catastrophic paths (150 ->
    # 411.8), and at seed 0 collapses the policy to never-hedge outright.
    # The fix is retracted and the production checkpoint is a plain default
    # run (the pre-retraction one is preserved as
    # hedging_agent_gru_timegan.pt.bak-moneynessclip-promoted). This guards
    # against it being silently re-promoted -- if you have evidence the clip
    # helps, put the multi-seed numbers in RESULTS.md before flipping this.
    policy, _, _ = _load_policy_checkpoint(CHECKPOINT_DIR / "hedging_agent_gru_timegan.pt")
    assert policy.moneyness_clip is None


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "hedging_agent_rnn_timegan.pt").exists(),
    reason="checkpoints/hedging_agent_rnn_timegan.pt not found (gitignored; train locally first)",
)
def test_promoted_basic_rnn_timegan_checkpoint_carries_the_lr_and_clip_fix() -> None:
    # Basic RNN (TimeGAN)'s promoted fix is the *stack*: --lr 1e-3 removes the
    # tanh saturation that made the network input-insensitive, and only then
    # can --moneyness-clip do anything at all (clipping an input the hidden
    # state ignores is provably a no-op -- that is why the clip was
    # incorrectly ruled out for this architecture the first time). 5/5 seeds
    # improved on every risk metric; see RESULTS.md. Both halves must
    # round-trip through _load_policy_checkpoint, so assert both.
    policy, _, policy_args = _load_policy_checkpoint(CHECKPOINT_DIR / "hedging_agent_rnn_timegan.pt")
    assert policy.moneyness_clip == (-0.15, 0.10)
    assert policy_args["lr"] == pytest.approx(1e-3)
