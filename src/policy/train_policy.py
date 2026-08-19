"""Trains the Deep Hedging policy (src/policy/hedging_agent.py) to minimize
the CVaR (src/loss/cvar.py) of terminal wealth over synthetic asset price
paths produced by the Market Generator (src/generator/market_gan.py).

Run directly as a training CLI:

    python src/policy/train_policy.py --epochs 100
"""

import sys
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Protocol, Tuple, Union

import torch

# Allow `python src/policy/train_policy.py` to resolve sibling packages the
# same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.black_scholes import BlackScholesDeltaPolicy  # noqa: E402
from common.checkpoints import checkpoint_filename  # noqa: E402
from common.device import select_device  # noqa: E402
from environment.market_env import MarketEnvironment, estimate_premium_monte_carlo  # noqa: E402
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
        slow_ramp_fraction: Annotated[
            float,
            "fraction of each training batch to replace with synthetic price paths "
            "whose standardized log-moneyness ramps slowly through slow_ramp_zone, "
            "instead of being sampled from the generator. Targets RESULTS.md's LSTM "
            "(TimeGAN) velocity-hysteresis finding: the recurrent policy's hidden "
            "state gets stuck in a degenerate state when log-moneyness crosses this "
            "zone faster than ~slow_ramp_step per step, and real TimeGAN-generated "
            "paths essentially never cross it slowly, so the policy never sees a "
            "well-behaved example to learn correct behavior from there. 0.0 "
            "(default) is a no-op. Only applied for a RecurrentHedgingAgent policy "
            "(silently ignored for HedgingAgent, which has no log-moneyness input).",
        ] = 0.0,
        slow_ramp_zone: Annotated[
            Tuple[float, float],
            "(lo, hi) standardized log-moneyness range the synthetic ramp paths "
            "cross, per the transition boundary measured in RESULTS.md's mechanism "
            "(b) follow-up (a single continuous ramp collapsed delta from 0.96 to "
            "0.00001 between log-moneyness 0.052 and 0.093).",
        ] = (0.08, 0.14),
        slow_ramp_step: Annotated[
            float,
            "per-step change in standardized log-moneyness for the synthetic ramp. "
            "0.0129 is the largest step size RESULTS.md's velocity-isolated probe "
            "found the policy recovers from within a 40-step hold (0.0180 does "
            "not), so this default stays inside the empirically-verified 'safe' "
            "regime.",
        ] = 0.0129,
        smoothness_penalty_weight: Annotated[
            float,
            "weight on an auxiliary loss term penalizing the recurrent policy's "
            "output sensitivity d(delta)/d(log-moneyness), computed directly on "
            "each training batch's own sampled paths (not a synthetic probe, so "
            "it isn't subject to the trajectory-shape confound slow-ramp "
            "augmentation is). Targets RESULTS.md's untried Lipschitz-penalty "
            "candidate for LSTM (TimeGAN)'s velocity-hysteresis failure -- unlike "
            "the zone-restricted framing in that writeup, this is a global "
            "penalty (every time step, not just a hand-picked input range), since "
            "a fixed numeric zone was found not to transfer across differently- "
            "calibrated TimeGAN checkpoints (each one's own natural log-moneyness "
            "range differs). 0.0 (default) is a no-op. Only applied for a "
            "RecurrentHedgingAgent policy.",
        ] = 0.0,
    ) -> None:
        self.device = device or select_device()
        if smoothness_penalty_weight > 0.0 and self.device.type == "mps":
            raise ValueError(
                "smoothness_penalty_weight's double-backward through an nn.LSTM "
                "(torch.autograd.grad(..., create_graph=True)) is not supported on "
                "MPS as of torch 2.8 ('derivative for lstm_mps_backward is not "
                "implemented') -- pass device=torch.device('cpu') (or --device cpu "
                "on the CLI) when using this flag."
            )
        self.policy = policy.to(self.device)
        self.generator = generator.to(self.device)
        self.environment = environment
        self.cvar_loss = cvar_loss.to(self.device)
        self.implied_vol = implied_vol
        self.sequence_policy = sequence_policy
        self.grad_clip_norm = grad_clip_norm
        self.use_bs_baseline = use_bs_baseline
        self._bs_policy = BlackScholesDeltaPolicy(strike=environment.strike) if use_bs_baseline else None
        self.slow_ramp_fraction = slow_ramp_fraction
        self.slow_ramp_zone = slow_ramp_zone
        self.slow_ramp_step = slow_ramp_step
        self.smoothness_penalty_weight = smoothness_penalty_weight

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
            if self.slow_ramp_fraction > 0.0:
                prices = self._inject_slow_ramp_paths(prices)

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

        if self.smoothness_penalty_weight > 0.0 and isinstance(self.policy, RecurrentHedgingAgent):
            loss = loss + self.smoothness_penalty_weight * self._compute_smoothness_penalty(prices)

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

    def _inject_slow_ramp_paths(
        self,
        prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] generator-sampled price paths"],
    ) -> Annotated[
        torch.Tensor,
        "[Batch, Time_Steps, 1] -- the first round(slow_ramp_fraction * Batch) rows "
        "replaced with synthetic slow log-moneyness ramps, the rest untouched",
    ]:
        """Builds synthetic price paths whose standardized log-moneyness ramps
        from 0 to a random target inside slow_ramp_zone at slow_ramp_step per
        step, then holds near that target (with small jitter) for the rest of
        the path -- see RESULTS.md's velocity-isolated hold-at-fixed-level
        probe, which this construction directly mirrors.
        """
        if not isinstance(self.policy, RecurrentHedgingAgent):
            return prices

        batch_size, seq_len, _ = prices.shape
        n = int(round(self.slow_ramp_fraction * batch_size))
        if n == 0:
            return prices

        lo, hi = self.slow_ramp_zone
        # [n] target landing log-moneyness inside the zone, random sign (only the
        # positive zone was empirically diagnosed; mirroring to the negative side
        # is an untested extrapolation, flagged in RESULTS.md).
        magnitude = lo + (hi - lo) * torch.rand(n, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        target = magnitude * sign  # [n]

        # [n] -> [n, 1] (ramp length in steps, at least 1, capped to the path length)
        ramp_len = (target.abs() / self.slow_ramp_step).ceil().clamp(min=1, max=seq_len - 1)
        ramp_len = ramp_len.unsqueeze(1)
        step_frac = target.unsqueeze(1) / ramp_len  # [n, 1]

        # [1, Time] -> [n, Time] (linear ramp 0 -> target over ramp_len steps, then
        # held constant at target for every later step)
        steps = torch.arange(seq_len, device=self.device, dtype=prices.dtype).unsqueeze(0)
        ramp_values = steps * step_frac
        log_moneyness = torch.where(steps <= ramp_len, ramp_values, target.unsqueeze(1).expand(n, seq_len))

        # small jitter once holding, so the network doesn't just memorize an
        # exactly flat post-ramp signal
        hold_mask = (steps > ramp_len).to(prices.dtype)
        log_moneyness = log_moneyness + hold_mask * 0.01 * torch.randn(n, seq_len, device=self.device)

        # [n, Time] -> [n, Time, 1] (invert RecurrentHedgingAgent.forward's
        # log(S_t/K)/moneyness_scale transform back into a raw price path)
        moneyness_scale = self.policy.moneyness_scale
        strike = self.policy.strike
        ramp_prices = (strike * torch.exp(log_moneyness * moneyness_scale)).unsqueeze(-1)

        prices = prices.clone()
        prices[:n] = ramp_prices.to(prices.dtype)
        return prices

    def _compute_smoothness_penalty(
        self,
        prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] this step's sampled price paths"],
    ) -> Annotated[torch.Tensor, "scalar: mean squared d(delta)/d(log-moneyness) over the whole batch"]:
        """Penalizes the recurrent policy's output sensitivity to its own
        standardized log-moneyness input, directly on this step's real
        training batch -- a differentiable proxy for the transition
        steepness RESULTS.md's mechanism (b) diagnosis identified as LSTM
        (TimeGAN)'s failure. Computed via the chain rule from
        d(delta)/d(price) (autograd) rather than by threading a second input
        path through RecurrentHedgingAgent.forward, so the policy's forward
        pass itself needs no changes.
        """
        prices_for_grad = prices.detach().clone().requires_grad_(True)
        delta_path = self.policy(prices_for_grad)  # [Batch, Time_Steps - 1, 1]
        (grad_wrt_price,) = torch.autograd.grad(
            delta_path.sum(), prices_for_grad, create_graph=True
        )

        # d(delta)/d(log_moneyness) = d(delta)/d(price) * price * moneyness_scale,
        # since log_moneyness = log(price / strike) / moneyness_scale
        # [Batch, Time_Steps - 1, 1] (only the prices RecurrentHedgingAgent.forward
        # actually uses -- prices[:, :-1, :] -- carry nonzero gradient here)
        used_prices = prices_for_grad[:, :-1, :]
        sensitivity = grad_wrt_price[:, :-1, :] * used_prices * self.policy.moneyness_scale
        return (sensitivity**2).mean()


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
    parser.add_argument(
        "--epochs", type=int, default=25_000,
        help="gradient steps -- paper Table 3's 50 epochs * (500,000-scenario dataset / 1,000 batch) "
        "= 25,000 gradient steps in this codebase's per-step training convention (see replicate_part1.py)",
    )
    parser.add_argument("--batch-size", type=int, default=1_000, help="paper Table 3")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--strike", type=float, default=1.0, help="option strike price K")
    parser.add_argument("--proportional-fee-bps", type=float, default=30.0, help="kappa in basis points")
    parser.add_argument("--implied-vol", type=float, default=0.2)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--disable-premium",
        action="store_true",
        help="omit P0 from the wealth objective (pre-P0 behavior); by default P0 is estimated via Monte Carlo through the generator, see math_spec.md section 1.1",
    )
    parser.add_argument(
        "--premium-paths",
        type=int,
        default=500_000,
        help="Monte Carlo path count for the P0 estimate (chunked internally; see environment/market_env.py::estimate_premium_monte_carlo)",
    )
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
    parser.add_argument(
        "--orthogonal-init",
        action="store_true",
        help="orthogonally initialize RecurrentHedgingAgent's recurrent weight matrices "
        "(see RecurrentHedgingAgent's orthogonal_init) -- was implemented but never wired "
        "to this CLI, like --grad-clip-norm; see RESULTS.md's mechanism (a) writeup, "
        "'Extending the fix' subsection, for why neither fixed Basic RNN (TimeGAN)'s "
        "recurrent hidden-state saturation on its own.",
    )
    parser.add_argument(
        "--moneyness-clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="(rnn/lstm/gru only) clamp the standardized log-moneyness input to [LO, HI] "
        "before it reaches the recurrent cell (see RecurrentHedgingAgent's moneyness_clip). "
        "Default None is a no-op. Motivated by RESULTS.md mechanism (b): a bounded "
        "generator's training data only covers a finite input range, and this network has "
        "no principled behavior once its input leaves that range -- e.g. "
        "'--moneyness-clip -0.15 0.10' for TimeGAN, just inside its own measured "
        "training-distribution boundary.",
    )
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
        "--timegan-output-scale",
        type=float,
        default=1.0,
        help="widen TimeGAN's generated price distribution by this factor before "
        "training (see TimeGANPriceGenerator.output_scale) -- addresses mechanism "
        "(b) (RESULTS.md): TimeGAN's own training distribution (real ^GSPC-bounded, "
        "log-moneyness ~+-0.13-0.17) is far narrower than the stress test's extreme "
        "paths, and LSTM/GRU (TimeGAN) hit a sharp delta-collapse cliff just inside "
        "that boundary. 1.0 (default) is a no-op, reproducing every existing TimeGAN "
        "checkpoint's training distribution exactly.",
    )
    parser.add_argument(
        "--slow-ramp-fraction",
        type=float,
        default=0.0,
        help="(rnn/lstm/gru only) replace this fraction of each training batch with "
        "synthetic price paths whose standardized log-moneyness ramps slowly through "
        "--slow-ramp-zone (see PolicyTrainer's slow_ramp_fraction). Default 0.0 is a "
        "no-op. Motivated by RESULTS.md's LSTM (TimeGAN) velocity-hysteresis finding: "
        "real TimeGAN paths essentially never cross this zone slowly, so the policy "
        "never sees a well-behaved example there to learn from -- "
        "e.g. '--slow-ramp-fraction 0.1' for LSTM (TimeGAN).",
    )
    parser.add_argument(
        "--slow-ramp-zone",
        type=float,
        nargs=2,
        default=(0.08, 0.14),
        metavar=("LO", "HI"),
        help="standardized log-moneyness range the synthetic slow-ramp paths cross "
        "(see --slow-ramp-fraction); default matches RESULTS.md's measured LSTM "
        "(TimeGAN) transition boundary.",
    )
    parser.add_argument(
        "--slow-ramp-step",
        type=float,
        default=0.0129,
        help="per-step change in standardized log-moneyness for the synthetic ramp "
        "(see --slow-ramp-fraction); default is the largest step size RESULTS.md's "
        "velocity probe found LSTM (TimeGAN) recovers from.",
    )
    parser.add_argument(
        "--smoothness-penalty-weight",
        type=float,
        default=0.0,
        help="(rnn/lstm/gru only) weight on an auxiliary loss term penalizing the "
        "policy's output sensitivity d(delta)/d(log-moneyness) on each training "
        "batch's own sampled paths (see PolicyTrainer's smoothness_penalty_weight). "
        "Default 0.0 is a no-op. A global alternative to --slow-ramp-fraction for "
        "RESULTS.md's LSTM (TimeGAN) velocity-hysteresis finding, not tied to a "
        "hand-picked input zone -- e.g. '--smoothness-penalty-weight 0.01'.",
    )
    parser.add_argument(
        "--use-bs-baseline",
        action="store_true",
        help="train on CVaR of (policy_wealth - black_scholes_wealth) instead of raw "
        "policy_wealth -- a control-variate variance-reduction technique for CVaR's "
        "sparse gradient (see PolicyTrainer's use_bs_baseline and RESULTS.md)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="clip the policy's gradient norm to this value before each optimizer step "
        "(see PolicyTrainer's grad_clip_norm). Was implemented but never wired to this "
        "CLI, so every checkpoint in this repo has always trained with clipping "
        "disabled -- see RESULTS.md's mechanism (a) writeup for why this matters even "
        "for the feed-forward MLP, not just recurrent architectures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="force a specific device ('cpu', 'cuda', 'mps'). Default (None) "
        "auto-detects the fastest available (cuda > mps > cpu) -- see "
        "common/device.py for the benchmarked rationale. Force 'cpu' when "
        "using --smoothness-penalty-weight, whose double-backward through an "
        "nn.LSTM is unsupported on MPS.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="also save an intermediate checkpoint every N gradient steps, alongside the "
        "final one, as <checkpoint>.step<N>.pt. 0 (default) disables. Added to locate "
        "*when* during training a run's tail-risk severity is decided -- see RESULTS.md.",
    )
    parser.add_argument(
        "--data-seed", type=int, default=None,
        help="re-seed the RNG with this value immediately before the training loop, so "
        "--seed governs only policy initialization (and the Monte Carlo premium estimate) "
        "while this governs the training noise draws. Default (None) leaves the single-seed "
        "behaviour untouched. Added to attribute GRU's cross-seed tail-risk severity to "
        "initialization vs. data order -- see RESULTS.md.",
    )
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
    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.generator_type == "timegan":
        generator = load_timegan_price_generator(
            Path(args.timegan_checkpoint), output_scale=args.timegan_output_scale
        )
        print(
            f"Loaded pretrained TimeGAN from {args.timegan_checkpoint}"
            + (f" (output_scale={args.timegan_output_scale})" if args.timegan_output_scale != 1.0 else "")
        )
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

    # Move the (frozen, pretrained) generator to the target device now, not
    # just inside PolicyTrainer.__init__ -- _train_and_save's premium
    # estimation runs the generator before PolicyTrainer is ever
    # constructed, and would otherwise crash with a device mismatch on any
    # non-CPU device (surfaced by MPS; the same latent bug applied to CUDA,
    # just never exercised since every run before this used --device cpu
    # implicitly).
    generator = generator.to(device)

    if args.alpha_sweep is not None:
        alphas = [float(a) for a in args.alpha_sweep.split(",")]
        for alpha in alphas:
            checkpoint_path = Path("checkpoints") / checkpoint_filename(
                args.architecture, alpha=alpha, suffix=suffix
            )
            _train_and_save(args, alpha, checkpoint_path, generator, device)
        return

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(
        "checkpoints"
    ) / checkpoint_filename(args.architecture, suffix=suffix)
    _train_and_save(args, args.cvar_alpha, checkpoint_path, generator, device)


def _save_checkpoint(
    checkpoint_path: Path,
    policy: Annotated[Any, "the policy being trained"],
    cvar_loss: CVaRLoss,
    args: Annotated[Any, "parsed argparse Namespace"],
    alpha: Annotated[float, "the CVaR level this run actually used"],
    premium: Annotated[float, "the Monte Carlo P0 estimate this run actually used"],
) -> None:
    """Writes the checkpoint payload every loader in this repo expects.

    Shared by the final save and by --checkpoint-every's intermediate saves, so
    an intermediate checkpoint is loadable by exactly the same code paths (the
    stress harness, the recovery probe) as a finished one.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = dict(vars(args))
    saved_args["cvar_alpha"] = alpha  # reflect the actual alpha used for this checkpoint
    saved_args["premium"] = premium  # the actual Monte Carlo estimate used, not just the CLI request
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "cvar_h": cvar_loss.h.item(),
            "args": saved_args,
        },
        checkpoint_path,
    )


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
            orthogonal_init=args.orthogonal_init,
            moneyness_clip=tuple(args.moneyness_clip) if args.moneyness_clip else None,
        )
        sequence_policy = True

    premium = 0.0
    if not args.disable_premium:
        premium = estimate_premium_monte_carlo(
            lambda n: generator(generator.sample_noise(n, args.seq_len, device=device)),
            strike=args.strike,
            num_paths=args.premium_paths,
        )
        print(
            f"P0 (premium) estimated via Monte Carlo over {args.premium_paths} paths "
            f"through the generator: {premium:.4f}"
        )

    environment = MarketEnvironment(
        strike=args.strike, proportional_fee=args.proportional_fee_bps / 1e4, dt=args.dt, premium=premium
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
        grad_clip_norm=args.grad_clip_norm,
        slow_ramp_fraction=args.slow_ramp_fraction,
        slow_ramp_zone=tuple(args.slow_ramp_zone),
        slow_ramp_step=args.slow_ramp_step,
        smoothness_penalty_weight=args.smoothness_penalty_weight,
    )

    print(
        f"Training Deep Hedging policy ({args.architecture}, alpha={alpha}) "
        f"for {args.epochs} epochs on {device}..."
    )

    # Split the single seed into its two roles. Everything above -- policy
    # initialization and the premium estimate -- has already consumed the
    # --seed stream; re-seeding here makes every subsequent draw (the training
    # scenarios) depend on --data-seed alone. Note this shifts the noise stream
    # even when --data-seed equals --seed, so a run with both set is not
    # bit-identical to the same run with neither.
    if getattr(args, "data_seed", None) is not None:
        torch.manual_seed(args.data_seed)
        print(f"Re-seeded training noise with data_seed={args.data_seed}")

    for epoch in range(1, args.epochs + 1):
        stats = trainer.train_step(args.batch_size, args.seq_len)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}/{args.epochs}  "
                f"cvar_loss={stats['loss']:.4f}  mean_wealth={stats['mean_wealth']:.4f}"
            )

        if args.checkpoint_every and epoch % args.checkpoint_every == 0 and epoch != args.epochs:
            _save_checkpoint(
                checkpoint_path.with_suffix(f".step{epoch}.pt"), policy, cvar_loss, args, alpha, premium
            )

    _save_checkpoint(checkpoint_path, policy, cvar_loss, args, alpha, premium)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
