"""Coverage for `backtester/stress_eval.py`, the harness behind every
multi-seed table in RESULTS.md and every record in `sweep_data/`.

The script itself was originally written ad hoc outside the repo and lost
when its scratch directory was cleared -- which is the same failure mode as
the unpreserved generator behind RESULTS.md's TimeGAN rows. It is committed
now, and these tests pin the two properties that make the committed records
re-derivable: the record schema, and the scenario constants.

The trained checkpoints are gitignored, so the checkpoint-driven test skips
cleanly on a fresh clone; the schema test reads `sweep_data/` and always runs.
"""

import json
from pathlib import Path

import pytest

from backtester.stress_eval import CONDITION_ARGS, stress_evaluate_checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DATA_DIR = REPO_ROOT / "sweep_data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Small enough to stay fast; this test checks the record's shape, not its
# tail-risk sensitivity (see test_tail_risk.py for the scale discussion).
_SMOKE_BATCH_SIZE = 2_000


@pytest.mark.parametrize("result_file", sorted(SWEEP_DATA_DIR.glob("RESULT_*.json")))
def test_harness_still_emits_every_field_the_committed_records_carry(result_file: Path) -> None:
    """The committed sweep records are the durable version of results whose
    checkpoints are gitignored. If the harness stops emitting a field they
    contain, a rerun silently produces records that no longer line up with
    the ones RESULTS.md's tables were built from.
    """
    records = json.loads(result_file.read_text())
    assert records, f"{result_file.name} is empty"

    for name, record in records.items():
        missing = {"checkpoint", "premium", "train_args", "cvar_99", "worst_loss"} - set(record)
        assert not missing, f"{result_file.name}:{name} missing {missing}"
        assert set(record["train_args"]) == set(CONDITION_ARGS), (
            f"{result_file.name}:{name} train_args keys drifted from CONDITION_ARGS"
        )


def test_stress_evaluate_produces_a_complete_record_for_a_real_checkpoint() -> None:
    checkpoints = sorted(CHECKPOINT_DIR.glob("hedging_agent*.pt"))
    if not checkpoints:
        pytest.skip("no trained checkpoints on disk (they are gitignored)")

    results = stress_evaluate_checkpoints(
        [checkpoints[0]], batch_size=_SMOKE_BATCH_SIZE, include_premium=False
    )

    record = results[checkpoints[0].stem]
    assert record["premium"] == 0.0
    assert record["checkpoint"] == str(checkpoints[0])
    assert set(record["train_args"]) == set(CONDITION_ARGS)
    assert record["train_args"]["architecture"] in {"mlp", "rnn", "lstm", "gru"}
    # Counting logic is unit-tested in test_tail_risk.py; here just assert the
    # counts are internally consistent with the batch actually evaluated.
    assert 0 <= record["below_-50_count"] <= _SMOKE_BATCH_SIZE
    assert record["worst_loss"] <= record["mean_wealth"]
