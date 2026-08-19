"""Coverage for `--data-seed` and the finding it produced.

`train_policy.py`'s single `--seed` sets both the policy initialization and
the training noise stream. `--data-seed` splits them, which is what let
RESULTS.md ("Why a seed lands at a given severity") rule out both factors as
main effects. These tests pin the split itself and the published result.
"""

import json
from pathlib import Path

import pytest
import torch

from policy.hedging_agent import RecurrentHedgingAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DATA_DIR = REPO_ROOT / "sweep_data"

# A run counts as having collapsed above this probe lag. Chosen from the
# measured separation, not fitted: across the 14 checkpoints trained for the
# decomposition, lag > 5 gives CVaR99 17-37 and lag <= 5 gives 1.7-10.3.
COLLAPSE_LAG_STEPS = 5.0


def test_initialization_depends_only_on_the_init_seed() -> None:
    """The premise of the split: --seed alone fixes the starting weights, so
    two runs differing only in --data-seed begin from the same policy.
    """
    def build(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        policy = RecurrentHedgingAgent(
            cell_type="gru", hidden_dim=64, num_layers=2, strike=1.0,
            implied_vol=0.2, time_to_maturity=1.0 * (30 - 1),
        )
        # [3 * hidden_dim, hidden_dim] -> flat copy of layer 0's recurrent weights
        return policy.rnn.weight_hh_l0.detach().clone()

    assert torch.equal(build(3), build(3)), "same seed must reproduce the same init"
    assert not torch.equal(build(3), build(4)), "different seeds must differ"


def test_train_policy_exposes_data_seed() -> None:
    source = (REPO_ROOT / "src" / "policy" / "train_policy.py").read_text()
    assert '"--data-seed"' in source
    # Re-seeding must happen after the policy is built, or it would not be a split.
    assert source.index("torch.manual_seed(args.data_seed)") > source.index("RecurrentHedgingAgent(")


def test_severity_follows_neither_the_init_seed_nor_the_data_seed() -> None:
    """The published negative result, checked against committed data rather
    than re-derived from the gitignored checkpoints.
    """
    probe = json.loads((SWEEP_DATA_DIR / "PROBE_seed_decomposition.json").read_text())
    lags = {name: record["mean_lag_steps"] for name, record in probe.items()}
    assert len(lags) == 9, "the factorial is 3x3"

    collapsed = {name for name, lag in lags.items() if lag > COLLAPSE_LAG_STEPS}
    assert len(collapsed) == 1, f"expected exactly one collapsed cell, got {sorted(collapsed)}"
    (cell,) = collapsed
    init, data = cell.replace("gru_tg_", "").split("_")

    # Neither factor is a main effect: the collapsed cell's own row and column
    # are otherwise clean, so no init value and no data value predicts collapse.
    row = [lag for name, lag in lags.items() if name.split("_")[2] == init and name != cell]
    column = [lag for name, lag in lags.items() if name.split("_")[3] == data and name != cell]
    assert all(lag <= COLLAPSE_LAG_STEPS for lag in row), "collapse would follow the initialization"
    assert all(lag <= COLLAPSE_LAG_STEPS for lag in column), "collapse would follow the data draw"


def test_probe_detects_the_collapse_mode_out_of_sample() -> None:
    """Detection is the claim that survived out-of-sample; ranking is not.
    Both the factorial and the base-rate control postdate the probe.
    """
    lags, cvar99 = {}, {}
    for stem in ("seed_decomposition", "gru_tg_baserate"):
        lags.update({
            k: v["mean_lag_steps"]
            for k, v in json.loads((SWEEP_DATA_DIR / f"PROBE_{stem}.json").read_text()).items()
        })
        cvar99.update({
            k: v["cvar_99"]
            for k, v in json.loads((SWEEP_DATA_DIR / f"RESULT_{stem}.json").read_text()).items()
        })

    collapsed = [cvar99[k] for k in lags if lags[k] > COLLAPSE_LAG_STEPS]
    survived = [cvar99[k] for k in lags if lags[k] <= COLLAPSE_LAG_STEPS]
    assert collapsed and survived
    assert min(collapsed) > max(survived), (
        "the probe's one surviving claim is that lag separates the collapse mode; "
        "if this fails, RESULTS.md's detection claim needs revising too"
    )
