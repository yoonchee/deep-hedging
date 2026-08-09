"""Part I frictionless replication study (src/backtester/replicate_part1.py).

Reproduces Kim (2021) "Deep Hedging, Generative Adversarial Networks, and
Beyond", Part I: a parametrically generated GBM underlying, NO transaction
costs, comparing Black-Scholes against RNN/LSTM/GRU direct policy-search RL
agents (plus our MLP) at several risk-aversion levels (Table 1 params:
S_0=100, K=100 ATM, r=0, vol=0.15, T=1/12, 30 time steps, batch_size=1000,
Adam, alpha in {0.5, 0.75, 0.99}).

Unlike the stress-test backtester (evaluate.py), this trains directly
against the known GBM process -- no adversarial market generator -- since
that's exactly what the paper's Part I isolates: can the RL agents replicate
Black-Scholes under ideal, frictionless conditions before anything realistic
(transaction costs, a learned market model) is introduced.

Includes the option premium P_0 (the analytic Black-Scholes price at these
exact params) in the wealth objective -- see math_spec.md section 1.1. This
is the one experiment in the repo where P_0 is included: constant vol and
r=0 make the closed-form price exact, unlike the regime-switching/GAN-driven
settings elsewhere, which have no closed form and still omit it.

Run directly as a CLI:

    python src/backtester/replicate_part1.py
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.black_scholes import BlackScholesDeltaPolicy, black_scholes_call_price  # noqa: E402
from backtester.plotting import (  # noqa: E402
    plot_boxplot_comparison,
    plot_delta_convexity,
    plot_pnl_distribution,
)
from common.stats import excess_kurtosis, skewness  # noqa: E402
from environment.market_env import MarketEnvironment  # noqa: E402
from generator.data import sample_real_prices  # noqa: E402
from loss.cvar import CVaRLoss  # noqa: E402
from policy.hedging_agent import HedgingAgent, RecurrentHedgingAgent  # noqa: E402
from policy.train_policy import PolicyTrainer  # noqa: E402

ARCHITECTURE_DISPLAY_NAMES: Dict[str, str] = {
    "mlp": "MLP",
    "rnn": "Basic RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


class GBMDataSource:
    """Adapts the fixed-parameter GBM sampler to PolicyTrainer's Generator interface.

    PolicyTrainer expects a `generator`-shaped object: `.sample_noise(batch,
    seq_len)` followed by `__call__(z)`. GBM sampling needs no external noise
    input at all, but implementing this two-step protocol lets Part I reuse
    PolicyTrainer's training loop unchanged instead of duplicating it.
    """

    def __init__(
        self,
        s0: Annotated[float, "initial asset price S_0"],
        vol: Annotated[float, "GBM volatility"],
        dt: Annotated[float, "time increment per step"],
    ) -> None:
        self.s0 = s0
        self.vol = vol
        self.dt = dt

    def to(self, device: torch.device) -> "GBMDataSource":
        return self

    def sample_noise(
        self, batch_size: int, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        # Shape carrier only -- see class docstring.
        return torch.empty(batch_size, seq_len)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = z.shape[0], z.shape[1]
        return sample_real_prices(batch_size, seq_len, s0=self.s0, vol=self.vol, dt=self.dt)


def build_policy(
    architecture: Annotated[str, "'mlp', 'rnn', 'lstm', or 'gru'"],
    hidden_dim: Annotated[int, "MLP hidden layer width"],
    num_hidden_layers: Annotated[int, "MLP hidden layer count"],
    rnn_hidden_dim: Annotated[int, "recurrent hidden state size"],
    rnn_output_hidden_dims: Annotated[List[int], "FC head widths after the RNN, e.g. [64, 64]"],
    strike: Annotated[float, "option strike K, used to normalize the price input to ~1 scale"],
    implied_vol: Annotated[float, "implied volatility, used to scale the RNN's log-moneyness input"] = 0.2,
    time_to_maturity: Annotated[float, "T, used to scale the RNN's log-moneyness input"] = 1.0,
) -> Annotated[Tuple[Any, bool], "(policy, sequence_policy)"]:
    if architecture == "mlp":
        return (
            HedgingAgent(
                hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, strike=strike
            ),
            False,
        )
    return (
        RecurrentHedgingAgent(
            cell_type=architecture,
            hidden_dim=rnn_hidden_dim,
            num_layers=1,
            output_hidden_dims=rnn_output_hidden_dims,
            strike=strike,
            implied_vol=implied_vol,
            time_to_maturity=time_to_maturity,
        ),
        True,
    )


def train_policy(
    architecture: str,
    alpha: float,
    environment: MarketEnvironment,
    data_source: GBMDataSource,
    implied_vol: float,
    epochs: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: torch.device,
    hidden_dim: int,
    num_hidden_layers: int,
    rnn_hidden_dim: int,
    rnn_output_hidden_dims: List[int],
    strike: Annotated[float, "option strike K, used to normalize the price input to ~1 scale"],
    time_to_maturity: Annotated[float, "T, used to scale the RNN's log-moneyness input"] = 1.0,
    log_every: int = 0,
) -> Annotated[Tuple[Any, bool], "(trained policy, sequence_policy)"]:
    policy, sequence_policy = build_policy(
        architecture,
        hidden_dim,
        num_hidden_layers,
        rnn_hidden_dim,
        rnn_output_hidden_dims,
        strike,
        implied_vol=implied_vol,
        time_to_maturity=time_to_maturity,
    )
    cvar_loss = CVaRLoss(alpha=alpha)
    trainer = PolicyTrainer(
        policy,
        environment,
        data_source,
        cvar_loss,
        implied_vol=implied_vol,
        lr=lr,
        device=device,
        sequence_policy=sequence_policy,
    )
    for epoch in range(1, epochs + 1):
        stats = trainer.train_step(batch_size, seq_len)
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                f"    epoch {epoch:4d}/{epochs}  cvar_loss={stats['loss']:.4f}  "
                f"mean_wealth={stats['mean_wealth']:.4f}"
            )
    return policy, sequence_policy


def summarize_pnl(
    wealth: Annotated[torch.Tensor, "[Batch] terminal wealth"],
    alpha: Annotated[float, "CVaR confidence level to report"],
) -> Dict[str, float]:
    losses = -wealth
    var = torch.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean().item() if tail.numel() > 0 else var.item()
    return {
        "mean_pnl": wealth.mean().item(),
        "cvar_pnl": cvar,
        "skewness": skewness(wealth),
        "excess_kurtosis": excess_kurtosis(wealth),
    }


def run_part1_replication(
    architectures: Annotated[Sequence[str], "which policy architectures to train"] = (
        "mlp",
        "rnn",
        "lstm",
        "gru",
    ),
    alphas: Annotated[Sequence[float], "CVaR risk-aversion levels, per paper Table 1"] = (
        0.5,
        0.75,
        0.99,
    ),
    s0: Annotated[float, "initial underlying value"] = 100.0,
    strike: Annotated[float, "strike price K (at-the-money)"] = 100.0,
    vol: Annotated[float, "GBM volatility"] = 0.15,
    time_to_maturity: Annotated[float, "T, in years (1/12 = one month)"] = 1.0 / 12.0,
    seq_len: Annotated[int, "number of price observations per path"] = 30,
    train_epochs: Annotated[
        int,
        "gradient steps per (architecture, alpha) pair. Paper Table 1 states '50 "
        "epochs' over a fixed 500,000-scenario training set with batch_size=1000 -- "
        "standard ML epoch semantics (one epoch = one pass over the dataset) make "
        "that 50 * (500,000 / 1,000) = 25,000 gradient steps, not 50 in this "
        "codebase's per-step convention (each 'epoch' here is one gradient step on "
        "a freshly-sampled batch, since GBM data is cheap to generate on the fly "
        "rather than drawn from a fixed pre-generated pool). 25,000 is the correct "
        "unit-matched default; an earlier 500 was a first attempt at closing the "
        "gap before this reconciliation, itself only ~2% of the paper's true budget.",
    ] = 25_000,
    train_batch_size: Annotated[int, "training batch size, per paper Table 1"] = 1000,
    test_batch_size: Annotated[
        int, "out-of-sample evaluation batch size, matches paper Table 1's 500,000-scenario test set"
    ] = 500_000,
    lr: Annotated[float, "Adam learning rate, used for every architecture except 'rnn' (see rnn_lr)"] = 1e-2,
    rnn_lr: Annotated[
        float,
        "Adam learning rate specifically for architecture='rnn'. At the shared "
        "lr=1e-2, Basic RNN's recurrent weight norm blows up during training "
        "(direct inspection: weight_hh norm 7 -> 40-79, hidden-state saturation "
        "0% -> 93-95%, typically within the first ~4,000-4,500 steps) and never "
        "recovers, giving a CVaR 43-59% worse than the paper's own RNN figure "
        "-- confirmed not fixed by --grad-clip-norm or --orthogonal-init, which "
        "both converge to nearly the same pathological weight norm regardless. "
        "1e-3 (10x lower) eliminates the blowup entirely (verified across 4 "
        "seeds: CVaR within 1-2 percentage points of the paper at every alpha, "
        "vs. 32-59% at lr=1e-2) -- LSTM/GRU/MLP don't show this failure at the "
        "shared 1e-2 default, consistent with their gating (LSTM/GRU) or lack "
        "of a compounding hidden state (MLP) providing implicit stability a "
        "vanilla RNN's recurrence doesn't have. See RESULTS.md.",
    ] = 1e-3,
    hidden_dim: Annotated[int, "MLP hidden layer width"] = 32,
    num_hidden_layers: Annotated[int, "MLP hidden layer count"] = 2,
    rnn_hidden_dim: Annotated[int, "recurrent hidden state size (paper: 128)"] = 128,
    rnn_output_hidden_dims: Annotated[
        Sequence[int],
        "FC head after the RNN. Empirically, reading the paper's '128, 64, "
        "64, 1' node counts as RNN(128) -> FC(64) -> FC(64) -> 1 causes "
        "severe training failure at our scale (dead ReLU units after the "
        "recurrence leave Basic RNN/LSTM permanently input-insensitive, "
        "confirmed by direct hidden-state inspection); a single linear "
        "readout (the default, empty tuple) trains all three cell types "
        "successfully and even improves GRU's result. Kept configurable "
        "for anyone who wants to reproduce the failure or try their own head.",
    ] = (),
    output_dir: Annotated[
        Union[str, Path], "directory for per-alpha plots and the summary JSON"
    ] = "results/part1_replication",
    seed: Annotated[int, "base random seed"] = 0,
    device: Optional[torch.device] = None,
) -> Annotated[Dict, "per-alpha summary, also written to <output_dir>/part1_summary.json"]:
    device = device or torch.device("cpu")
    dt = time_to_maturity / (seq_len - 1)
    # P_0: constant vol and r=0 make the closed-form Black-Scholes price exact
    # here, unlike the regime-switching/GAN-driven settings elsewhere in this
    # repo where no closed form exists -- see math_spec.md section 1.1 and
    # RESULTS.md for why this term was missing and what adding it changes.
    # A constant additive shift to wealth doesn't change the CVaR-minimizing
    # policy (same argmin, gradient unaffected), only the reported numbers.
    premium = black_scholes_call_price(S0=s0, K=strike, tau=time_to_maturity, sigma=vol)
    environment = MarketEnvironment(strike=strike, proportional_fee=0.0, dt=dt, premium=premium)
    data_source = GBMDataSource(s0=s0, vol=vol, dt=dt)
    bs_policy = BlackScholesDeltaPolicy(strike=strike)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    for alpha in alphas:
        print(f"=== alpha = {alpha} ===")
        policies: Dict[str, Tuple[Any, bool]] = {}
        for architecture in architectures:
            torch.manual_seed(seed)
            display_name = ARCHITECTURE_DISPLAY_NAMES[architecture]
            architecture_lr = rnn_lr if architecture == "rnn" else lr
            print(f"  Training {display_name} (alpha={alpha}, {train_epochs} epochs, lr={architecture_lr})...")
            policy, sequence_policy = train_policy(
                architecture,
                alpha,
                environment,
                data_source,
                implied_vol=vol,
                epochs=train_epochs,
                batch_size=train_batch_size,
                seq_len=seq_len,
                lr=architecture_lr,
                device=device,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_output_hidden_dims=list(rnn_output_hidden_dims),
                time_to_maturity=time_to_maturity,
                strike=strike,
            )
            policies[display_name] = (policy, sequence_policy)

        # Fresh out-of-sample GBM test data (an unlimited i.i.d. generator,
        # so this is out-of-sample by construction -- no held-out split needed).
        test_prices = sample_real_prices(test_batch_size, seq_len, s0=s0, vol=vol, dt=dt)

        all_strategies: Dict[str, Tuple[Any, bool]] = {"Black-Scholes Delta": (bs_policy, False)}
        all_strategies.update(policies)

        wealth_by_strategy: Dict[str, torch.Tensor] = {}
        summary_by_strategy: Dict[str, Dict[str, float]] = {}
        for name, (policy_obj, sequence_policy) in all_strategies.items():
            if hasattr(policy_obj, "eval"):
                policy_obj.eval()
            with torch.no_grad():
                # Chunked: a single-pass forward through an RNN/LSTM/GRU
                # policy at the paper's full 500,000-path test scale hits a
                # severe (not just linear) CPU slowdown -- see
                # MarketEnvironment.simulate's chunk_size docstring.
                wealth = environment.simulate(
                    policy_obj, test_prices, vol, sequence_policy=sequence_policy, chunk_size=50_000
                )
            wealth_by_strategy[name] = wealth
            summary_by_strategy[name] = summarize_pnl(wealth, alpha)

        alpha_dir = output_path / f"alpha_{alpha}".replace(".", "_")
        alpha_dir.mkdir(parents=True, exist_ok=True)
        plot_pnl_distribution(
            wealth_by_strategy,
            alpha_dir / "pnl_distribution.png",
            title=f"PnL Distribution (alpha = {alpha})",
        )
        plot_boxplot_comparison(
            wealth_by_strategy,
            alpha_dir / "pnl_boxplot.png",
            title=f"PnL Comparison Across Strategies (alpha = {alpha})",
        )
        plot_delta_convexity(
            all_strategies, strike, vol, n_steps=seq_len - 1, dt=dt,
            path=alpha_dir / "delta_convexity.png",
        )

        all_summaries[str(alpha)] = summary_by_strategy
        print(json.dumps(summary_by_strategy, indent=2))

    result = {
        "config": {
            "s0": s0,
            "strike": strike,
            "vol": vol,
            "time_to_maturity": time_to_maturity,
            "seq_len": seq_len,
            "proportional_fee": 0.0,
            "premium": premium,
            "train_epochs": train_epochs,
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
            "lr": lr,
            "rnn_lr": rnn_lr,
            "architectures": list(architectures),
            "rnn_hidden_dim": rnn_hidden_dim,
            "rnn_output_hidden_dims": list(rnn_output_hidden_dims),
        },
        "alphas": all_summaries,
    }
    with open(output_path / "part1_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Replicate the paper's Part I frictionless GBM experiment."
    )
    parser.add_argument(
        "--epochs", type=int, default=25_000,
        help="gradient steps per (arch, alpha) -- paper Table 1's 50 epochs * 500 batches/epoch over a fixed 500k-scenario set",
    )
    parser.add_argument("--train-batch-size", type=int, default=1000)
    parser.add_argument("--test-batch-size", type=int, default=500_000, help="matches paper Table 1's 500,000-scenario test set")
    parser.add_argument("--lr", type=float, default=1e-2, help="used for every architecture except rnn (see --rnn-lr)")
    parser.add_argument(
        "--rnn-lr", type=float, default=1e-3,
        help="Basic RNN-specific learning rate -- 1e-2 (matching every other architecture) causes a "
        "training-time weight-norm blowup unique to the vanilla RNN cell; see run_part1_replication's "
        "rnn_lr docstring and RESULTS.md. Pass --rnn-lr 1e-2 to reproduce the old, pre-fix numbers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/part1_replication")
    args = parser.parse_args()

    final_result = run_part1_replication(
        train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        rnn_lr=args.rnn_lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("Saved summary to", Path(args.output_dir) / "part1_summary.json")
