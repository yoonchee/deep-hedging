"""Trains the Deep Hedging policy (src/policy/hedging_agent.py) to minimize
the CVaR (src/loss/cvar.py) of terminal wealth over synthetic asset price
paths produced by the Market Generator (src/generator/market_gan.py).

Run directly as a training CLI:

    python src/policy/train_policy.py --epochs 100
"""

import sys
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Protocol, Union

import torch

# Allow `python src/policy/train_policy.py` to resolve sibling packages the
# same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.black_scholes import BlackScholesDeltaPolicy  # noqa: E402
from environment.market_env import MarketEnvironment  # noqa: E402
from generator.market_gan import Generator  # noqa: E402
from generator.train_timegan import load_timegan_price_generator  # noqa: E402
from loss.cvar import CVaRLoss  # noqa: E402
from policy.hedging_agent import HedgingAgent, RecurrentHedgingAgent  # noqa: E402


class MarketGenerator(Protocol):
    """Structural type for anything PolicyTrainer can treat as a fixed market
    simulator: market_gan.Generator (single-feature WGAN-GP) or
    generator.train_timegan.TimeGANPriceGenerator (TimeGAN adapter).
    """

    def sample_noise(
        self, batch_size: int, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor: ...

    def __call__(self, z: torch.Tensor) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price path"]: ...


class PolicyTrainer:
    """Single training-step encapsulation for the CVaR-hedging policy."""

    def __init__(
        self,
        policy: Union[HedgingAgent, RecurrentHedgingAgent],
        environment: MarketEnvironment,
        generator: MarketGenerator,
        cvar_loss: CVaRLoss,
        implied_vol: Annotated[float, "implied volatility fed into the policy state"] = 0.2,
        lr: Annotated[float, "Adam learning rate for policy params and CVaR threshold h"] = 1e-3,
        device: Optional[torch.device] = None,
        sequence_policy: Annotated[
            bool, "True for a RecurrentHedgingAgent (whole-path) policy"
        ] = False,
        grad_clip_norm: Annotated[
            Optional[float],
            "if set, clips the policy's gradient norm to this value before each "
            "optimizer step -- combats early gradient explosion permanently "
            "saturating recurrent gates (a classic RNN/LSTM training failure: "
            "the policy gets stuck outputting a near-constant value from the "
            "first few steps onward, with zero further movement no matter how "
            "many epochs follow). None (default) disables clipping.",
        ] = None,
        use_bs_baseline: Annotated[
            bool,
            "if set, trains on CVaR of (policy_wealth - black_scholes_wealth) computed "
            "on the SAME sampled price batch, instead of raw policy_wealth -- a "
            "control-variate variance-reduction technique (the paper's suggested "
            "'actor-critic baseline', adapted to this codebase's direct-backprop "
            "training rather than REINFORCE: Black-Scholes' wealth on the same path "
            "serves as a zero-approximation-error state-value baseline). Since both "
            "wealths are computed on the identical market draw, subtracting them "
            "cancels shared market-driven variance, leaving a lower-variance signal "
            "for which paths CVaR's sparse gradient selects as 'worst' batch to "
            "batch -- see RESULTS.md. mean_wealth in train_step's return is always "
            "the RAW policy wealth regardless of this flag, for comparability.",
        ] = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.generator = generator.to(self.device)
        self.environment = environment
        self.cvar_loss = cvar_loss.to(self.device)
        self.implied_vol = implied_vol
        self.sequence_policy = sequence_policy
        self.grad_clip_norm = grad_clip_norm
        self.use_bs_baseline = use_bs_baseline
        self._bs_policy = BlackScholesDeltaPolicy(strike=environment.strike) if use_bs_baseline else None

        params = list(self.policy.parameters()) + list(self.cvar_loss.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(
        self,
        batch_size: Annotated[int, "number of synthetic paths per step"],
        seq_len: Annotated[int, "number of price observations per path"],
    ) -> Annotated[
        Dict[str, float],
        "{'loss': CVaR loss, 'mean_wealth': mean terminal wealth, 'grad_norm': pre-clip policy grad norm}",
    ]:
        # The generator acts as a fixed, pretrained market simulator here: no
        # gradient is needed through its parameters when training the policy.
        with torch.no_grad():
            z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
            prices = self.generator(z)  # [Batch, seq_len, 1]

        wealth = self.environment.simulate(
            self.policy, prices, self.implied_vol, sequence_policy=self.sequence_policy
        )  # [Batch]

        if self.use_bs_baseline:
            # No gradient needed through the fixed analytic baseline; only the
            # policy's own wealth term carries gradient into loss.backward().
            with torch.no_grad():
                bs_wealth = self.environment.simulate(
                    self._bs_policy, prices, self.implied_vol, sequence_policy=False
                )
            loss = self.cvar_loss(wealth - bs_wealth)
        else:
            loss = self.cvar_loss(wealth)

        self.optimizer.zero_grad()
        loss.backward()

        # Always measured (even with clipping off) so callers can diagnose
        # whether explosion is actually happening.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.grad_clip_norm or float("inf")
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_wealth": wealth.mean().item(),
            "grad_norm": grad_norm.item(),
        }


def _load_or_init_generator(
    checkpoint_path: Annotated[Path, "path to a checkpoint saved by generator/train_gan.py"],
    strike: Annotated[float, "option strike price, used as S_0 for a fallback generator"],
    noise_dim: Annotated[int, "fallback generator noise dim if no checkpoint is found"],
    hidden_dim: Annotated[int, "fallback generator hidden dim if no checkpoint is found"],
    num_layers: Annotated[int, "fallback generator layer count if no checkpoint is found"],
) -> Generator:
    """Loads a pretrained market Generator, or falls back to an untrained one.

    The policy is trained against a *fixed* market simulator (see
    `PolicyTrainer.train_step`), so ideally this loads the checkpoint saved
    by `generator/train_gan.py --epochs ...`. If none exists yet, an
    untrained generator is used instead so the CLI still runs end-to-end.
    """
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        gen_args = checkpoint["args"]
        generator = Generator(
            noise_dim=gen_args["noise_dim"],
            hidden_dim=gen_args["hidden_dim"],
            num_layers=gen_args["num_layers"],
            initial_price=gen_args["s0"],
        )
        generator.load_state_dict(checkpoint["generator_state_dict"])
        for param in generator.parameters():
            param.requires_grad_(False)
        print(f"Loaded pretrained generator from {checkpoint_path}")
        return generator

    print(
        f"No generator checkpoint found at {checkpoint_path}; using an untrained "
        f"generator (noise_dim={noise_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}). "
        "Run `python src/generator/train_gan.py` first for a realistic market simulator."
    )
    return Generator(
        noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=num_layers, initial_price=strike
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the Deep Hedging policy via CVaR minimization.")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--strike", type=float, default=1.0, help="option strike price K")
    parser.add_argument("--proportional-fee-bps", type=float, default=30.0, help="kappa in basis points")
    parser.add_argument("--implied-vol", type=float, default=0.2)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["mlp", "rnn", "lstm", "gru"],
        default="mlp",
        help="mlp = feed-forward HedgingAgent (delta_{t-1} fed explicitly); "
        "rnn/lstm/gru = RecurrentHedgingAgent, a true recurrent policy over the whole path",
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="policy hidden layer width (mlp)")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="policy hidden layer count (mlp)")
    parser.add_argument("--rnn-hidden-dim", type=int, default=64, help="recurrent hidden state size (rnn/lstm/gru)")
    parser.add_argument("--rnn-num-layers", type=int, default=2, help="stacked recurrent layers (rnn/lstm/gru)")
    parser.add_argument("--noise-dim", type=int, default=8, help="fallback generator noise dim")
    parser.add_argument("--gen-hidden-dim", type=int, default=64, help="fallback generator hidden dim")
    parser.add_argument("--gen-num-layers", type=int, default=2, help="fallback generator layer count")
    parser.add_argument(
        "--generator-checkpoint",
        type=str,
        default="checkpoints/market_gan.pt",
        help="checkpoint saved by generator/train_gan.py (--generator-type wgan)",
    )
    parser.add_argument(
        "--generator-type",
        type=str,
        choices=["wgan", "timegan"],
        default="wgan",
        help="wgan = single-feature WGAN-GP (generator/train_gan.py); "
        "timegan = multi-variate TimeGAN (generator/train_timegan.py), sliced to its price channel",
    )
    parser.add_argument(
        "--timegan-checkpoint",
        type=str,
        default="checkpoints/timegan.pt",
        help="checkpoint saved by generator/train_timegan.py (--generator-type timegan)",
    )
    parser.add_argument(
        "--use-bs-baseline",
        action="store_true",
        help="train on CVaR of (policy_wealth - black_scholes_wealth) instead of raw "
        "policy_wealth -- a control-variate variance-reduction technique for CVaR's "
        "sparse gradient (see PolicyTrainer's use_bs_baseline and RESULTS.md)",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to save the trained policy weights "
        "(default: checkpoints/hedging_agent.pt for mlp, checkpoints/hedging_agent_<architecture>.pt otherwise)",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        default=None,
        help="comma-separated CVaR alphas to train separate checkpoints for "
        "(e.g. '0.5,0.75,0.9,0.95,0.99,0.995,0.997'), overriding --cvar-alpha and --checkpoint; "
        "saves to checkpoints/hedging_agent_<architecture>_alpha<alpha>.pt per value",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generator_type == "timegan":
        generator = load_timegan_price_generator(Path(args.timegan_checkpoint))
        print(f"Loaded pretrained TimeGAN from {args.timegan_checkpoint}")
        suffix = "_timegan"
    else:
        generator = _load_or_init_generator(
            Path(args.generator_checkpoint),
            strike=args.strike,
            noise_dim=args.noise_dim,
            hidden_dim=args.gen_hidden_dim,
            num_layers=args.gen_num_layers,
        )
        suffix = ""

    if args.alpha_sweep is not None:
        alphas = [float(a) for a in args.alpha_sweep.split(",")]
        for alpha in alphas:
            alpha_str = f"{alpha:.4g}".replace(".", "_")
            checkpoint_path = Path(
                f"checkpoints/hedging_agent_{args.architecture}_alpha{alpha_str}{suffix}.pt"
            )
            _train_and_save(args, alpha, checkpoint_path, generator, device)
        return

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(
        f"checkpoints/hedging_agent{suffix}.pt"
        if args.architecture == "mlp"
        else f"checkpoints/hedging_agent_{args.architecture}{suffix}.pt"
    )
    _train_and_save(args, args.cvar_alpha, checkpoint_path, generator, device)


def _train_and_save(
    args: Annotated[Any, "parsed argparse Namespace"],
    alpha: Annotated[float, "CVaR confidence level for this training run"],
    checkpoint_path: Path,
    generator: MarketGenerator,
    device: torch.device,
) -> None:
    if args.architecture == "mlp":
        policy = HedgingAgent(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            strike=args.strike,
        )
        sequence_policy = False
    else:
        policy = RecurrentHedgingAgent(
            cell_type=args.architecture,
            hidden_dim=args.rnn_hidden_dim,
            num_layers=args.rnn_num_layers,
            strike=args.strike,
            implied_vol=args.implied_vol,
            time_to_maturity=args.dt * (args.seq_len - 1),
        )
        sequence_policy = True

    environment = MarketEnvironment(
        strike=args.strike, proportional_fee=args.proportional_fee_bps / 1e4, dt=args.dt
    )
    cvar_loss = CVaRLoss(alpha=alpha)

    trainer = PolicyTrainer(
        policy,
        environment,
        generator,
        cvar_loss,
        implied_vol=args.implied_vol,
        lr=args.lr,
        device=device,
        sequence_policy=sequence_policy,
        use_bs_baseline=args.use_bs_baseline,
    )

    print(
        f"Training Deep Hedging policy ({args.architecture}, alpha={alpha}) "
        f"for {args.epochs} epochs on {device}..."
    )
    for epoch in range(1, args.epochs + 1):
        stats = trainer.train_step(args.batch_size, args.seq_len)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}/{args.epochs}  "
                f"cvar_loss={stats['loss']:.4f}  mean_wealth={stats['mean_wealth']:.4f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = dict(vars(args))
    saved_args["cvar_alpha"] = alpha  # reflect the actual alpha used for this checkpoint
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "cvar_h": cvar_loss.h.item(),
            "args": saved_args,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
