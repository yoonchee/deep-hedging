"""Training-free recovery-lag probe for recurrent hedging policies.

RESULTS.md's GRU (WGAN-GP) diagnosis found that its catastrophic paths all
share one shape -- an early downward move followed by a large rally -- and
that the defect is a *lag* in delta recovering after the down-move rather
than a permanent collapse. This module turns that diagnosis into a scalar
measured from a handful of synthetic paths, cheaply enough to run on every
checkpoint of a sweep.

The scalar is the number of steps delta spends below 0.5 after the shock,
averaged over a sweep of shock depths. It is deliberately *duration*, not
final delta: a policy that recovers late still misses most of the hedge P&L,
because the price path is exponential and its largest absolute increments
land early in the rally. Final delta saturates at 1.0 for most checkpoints
and loses the resolution that the lag keeps.

See RESULTS.md, "Where GRU's seed variance comes from", for the validation
against measured 500,000-path tail risk.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Dict, List, Sequence, Tuple

import torch

# Allow `python src/backtester/recovery_probe.py` to resolve sibling packages
# the same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backtester.evaluate import _load_policy_checkpoint  # noqa: E402

SEQ_LEN: int = 30
"""The paper's own path length; the lag is only meaningful relative to it."""

DIP_END_STEP: int = 10
"""Step at which the down-move bottoms out, matching the worst-path shape."""

RALLY_TARGET: float = 4.87
"""Final log-moneyness of the rally -- the actual worst path's, from the scan."""

DEFAULT_DEPTHS: Tuple[float, ...] = tuple(round(-0.1 * i, 2) for i in range(26))
"""0.0 down to -2.5: spans the generator's realistic range and well past it."""


def build_probe_paths(
    depths: Annotated[Sequence[float], "log-moneyness levels the down-move bottoms out at"],
    strike: Annotated[float, "option strike K; probe starts at-the-money"] = 1.0,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] one down-then-rally path per depth"]:
    """Builds one path per depth: linear in log-moneyness down to `depth` by
    `DIP_END_STEP`, then a smooth ramp up to `RALLY_TARGET` at the horizon.
    """
    paths: List[torch.Tensor] = []
    for depth in depths:
        # [DIP_END_STEP + 1] + [SEQ_LEN - DIP_END_STEP - 1] -> [Time_Steps]
        down = torch.linspace(0.0, depth, DIP_END_STEP + 1)
        up = torch.linspace(depth, RALLY_TARGET, SEQ_LEN - DIP_END_STEP)[1:]
        paths.append(torch.cat([down, up]))

    # [Batch, Time_Steps] log-moneyness -> [Batch, Time_Steps, 1] prices
    return (strike * torch.stack(paths).exp()).unsqueeze(-1)


def recovery_lag(
    checkpoint_path: Annotated[Path, "a recurrent policy checkpoint from train_policy.py"],
    depths: Sequence[float] = DEFAULT_DEPTHS,
    recovered_delta: Annotated[float, "delta above which the hedge counts as recovered"] = 0.5,
) -> Annotated[
    Dict[str, object],
    "mean_lag_steps (the predictor), plus the per-depth final-delta curve behind it",
]:
    """Loads a checkpoint and measures how long its delta stays un-recovered
    after a downward shock, averaged over shock depths.
    """
    loaded = _load_policy_checkpoint(checkpoint_path)
    if loaded is None:
        raise FileNotFoundError(checkpoint_path)
    policy, sequence_policy, train_args = loaded
    if not sequence_policy:
        raise ValueError(f"{checkpoint_path} is not a recurrent policy; the probe needs a price path")

    policy.eval()
    prices = build_probe_paths(depths, strike=train_args["strike"])
    with torch.no_grad():
        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps - 1, 1]
        deltas = policy(prices)

    # [Batch, Time_Steps - 1, 1] -> [Batch] steps below the recovery level after the dip
    lag = (deltas[:, DIP_END_STEP:, 0] < recovered_delta).sum(dim=1)
    # [Batch, Time_Steps - 1, 1] -> [Batch] delta at the final decision step
    final = deltas[:, -1, 0]

    return {
        "mean_lag_steps": lag.float().mean().item(),
        "max_lag_steps": int(lag.max().item()),
        "final_delta_curve": {str(d): round(f, 4) for d, f in zip(depths, final.tolist())},
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = {path.stem: recovery_lag(path) for path in args.checkpoints}
    for name, record in results.items():
        print(f"{name:<26} mean_lag={record['mean_lag_steps']:5.1f} steps", file=sys.stderr)

    payload = json.dumps(results, indent=2)
    print(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")


if __name__ == "__main__":
    main()
