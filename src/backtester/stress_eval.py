"""Stress-evaluates an explicit list of policy checkpoints on the shared
500,000-path regime-switching scenario, and writes one JSON record per
checkpoint.

This is the harness behind every multi-seed table in RESULTS.md and behind the
raw records committed in `sweep_data/`. `evaluate.py::scan_checkpoint_tail_risk`
answers a different question -- it sweeps whatever production checkpoints happen
to be in a directory, keyed by display name -- whereas a sweep needs named
checkpoints passed in explicitly, each carrying the training args it was saved
with, so the condition a row belongs to is recoverable from the data rather than
only from its filename.

The scenario parameters are fixed to the ones used throughout RESULTS.md
(500,000 paths, seed 42, low_vol=0.15 / high_vol=0.60 / switch_prob=0.10,
30 bps proportional fee), so records produced here are directly comparable to
that document's other tables.

Usage:
    python src/backtester/stress_eval.py checkpoints/sweeps/gru_wgan_*.pt \
        --out sweep_data/RESULT_gru_wgan_5seed.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import torch

# Allow `python src/backtester/stress_eval.py` to resolve sibling packages the
# same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from backtester.evaluate import (  # noqa: E402
    _load_policy_checkpoint,
    _summarize_strategy,
    simulate_regime_switching_paths,
    tail_risk_summary,
)
from environment.market_env import MarketEnvironment, estimate_premium_monte_carlo  # noqa: E402

# The subset of a checkpoint's saved training args that identifies which sweep
# condition it belongs to. Deliberately a fixed list rather than the whole args
# dict: these are the knobs the multi-seed sweeps vary, and pinning them keeps
# the committed records diffable across runs even as unrelated CLI flags come
# and go.
CONDITION_ARGS: Tuple[str, ...] = (
    "architecture",
    "seed",
    "lr",
    "grad_clip_norm",
    "moneyness_clip",
    "slow_ramp_fraction",
    "use_bs_baseline",
    "generator_type",
    "cvar_alpha",
)


def stress_evaluate_checkpoints(
    checkpoint_paths: Annotated[List[Path], "checkpoints saved by policy/train_policy.py"],
    batch_size: Annotated[int, "500,000 matches the scale that first surfaced this failure mode"] = 500_000,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    proportional_fee: Annotated[float, "kappa; e.g. 0.003 = 30 bps"] = 0.003,
    low_vol: Annotated[float, "stress regime: calm-period volatility"] = 0.15,
    high_vol: Annotated[float, "stress regime: turbulent-period volatility"] = 0.60,
    switch_prob: Annotated[float, "stress regime: per-step switch probability"] = 0.10,
    dt: Annotated[float, "time increment per step"] = 1.0,
    seed: Annotated[int, "matches the seed used for RESULTS.md's own scan and main stress-test table"] = 42,
    thresholds: Annotated[Tuple[float, ...], "wealth levels to count paths below"] = (-50.0, -10.0),
    include_premium: bool = True,
) -> Annotated[
    Dict[str, Dict[str, Any]],
    "{checkpoint stem: summary + tail counts + premium + train_args}",
]:
    """Evaluates every checkpoint on one shared set of stress paths.

    Paths and premium are drawn once and reused across checkpoints (they depend
    only on the scenario and the strike, both fixed here), so every record in a
    sweep sees the identical test set -- the paired comparison the multi-seed
    tables rely on.
    """
    device = torch.device("cpu")
    results: Dict[str, Dict[str, Any]] = {}
    prices: torch.Tensor = None  # [Batch, Time_Steps, 1], built lazily from the first strike
    premium = 0.0
    scenario_key: Tuple[float, float] = None

    for checkpoint_path in checkpoint_paths:
        loaded = _load_policy_checkpoint(checkpoint_path)
        if loaded is None:
            print(f"skipping missing checkpoint: {checkpoint_path}", file=sys.stderr)
            continue
        policy, sequence_policy, train_args = loaded
        strike = train_args["strike"]
        implied_vol = train_args["implied_vol"]

        # Rebuild the shared test set only if a checkpoint changes the scenario
        # it has to be priced under; in a normal sweep this happens once.
        if scenario_key != (strike, implied_vol):
            scenario_key = (strike, implied_vol)
            path_generator = torch.Generator().manual_seed(seed)
            prices = simulate_regime_switching_paths(
                batch_size, seq_len, s0=strike, dt=dt, low_vol=low_vol, high_vol=high_vol,
                switch_prob=switch_prob, generator=path_generator,
            )
            premium = 0.0
            if include_premium:
                premium_generator = torch.Generator().manual_seed(seed + 1)
                premium = estimate_premium_monte_carlo(
                    lambda n, s=strike: simulate_regime_switching_paths(
                        n, seq_len, s0=s, dt=dt, low_vol=low_vol, high_vol=high_vol,
                        switch_prob=switch_prob, generator=premium_generator,
                    ),
                    strike=strike,
                )
            environment = MarketEnvironment(
                strike=strike, proportional_fee=proportional_fee, dt=dt, premium=premium
            )

        policy.to(device)
        policy.eval()
        with torch.no_grad():
            # [Batch, Time_Steps, 1] -> ([Batch], [Batch]) terminal wealth, total cost
            wealth, cost = environment.simulate_with_costs(
                policy, prices, implied_vol, sequence_policy=sequence_policy, chunk_size=50_000
            )

        record: Dict[str, Any] = _summarize_strategy(wealth, cost)
        record.update(tail_risk_summary(wealth, thresholds))
        record["premium"] = premium
        record["checkpoint"] = str(checkpoint_path)
        record["train_args"] = {key: train_args.get(key) for key in CONDITION_ARGS}
        results[checkpoint_path.stem] = record
        print(
            f"{checkpoint_path.stem}: cvar_99={record['cvar_99']:.4f} "
            f"worst={record['worst_loss']:.1f} below_-50={record['below_-50_count']}",
            file=sys.stderr,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint files to evaluate")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout only)")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-premium", action="store_true", help="omit the P0 wealth shift")
    args = parser.parse_args()

    results = stress_evaluate_checkpoints(
        args.checkpoints,
        batch_size=args.batch_size,
        seed=args.seed,
        include_premium=not args.no_premium,
    )
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")


if __name__ == "__main__":
    main()
