"""WGAN-GP training step for the Market Generator.

Implements the critic loss from math_spec.md, section 4:

    L_D = E[D(real)] - E[D(fake)] - lambda * E[(||grad D(interp)|| - 1)^2]

The discriminator is trained to maximize L_D, i.e. minimize -L_D. The
generator is trained with the standard WGAN objective to maximize E[D(fake)].

Run directly as a training CLI:

    python src/generator/train_gan.py --epochs 50
    python src/generator/train_gan.py --epochs 50 --data-source yfinance
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import torch

# Allow `python src/generator/train_gan.py` to resolve `generator.market_gan`
# the same way pytest's `pythonpath = src` does, regardless of invocation
# style (direct script path vs. `python -m generator.train_gan`).
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.device import select_device  # noqa: E402
from common.stats import excess_kurtosis, excess_kurtosis_tensor, skewness, skewness_tensor, terminal_log_return  # noqa: E402
from generator.data import HistoricalPriceLoader, sample_real_prices  # noqa: E402
from generator.market_gan import Discriminator, Generator  # noqa: E402
from generator.validate import validate_generator_fidelity  # noqa: E402


def gradient_penalty(
    discriminator: Discriminator,
    real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"],
    fake: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] generated price paths"],
    device: torch.device,
) -> Annotated[torch.Tensor, "scalar gradient penalty E[(||grad D(interp)|| - 1)^2]"]:
    batch_size = real.size(0)

    # [Batch, 1, 1] -> [Batch, Time_Steps, 1] (broadcast interpolation weight)
    epsilon = torch.rand(batch_size, 1, 1, device=device).expand_as(real)

    # [Batch, Time_Steps, 1] (x_hat sampled along straight lines real -> fake)
    interpolates = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

    # [Batch, Time_Steps, 1] -> [Batch, 1]
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # [Batch, Time_Steps, 1] -> [Batch, Time_Steps] -> [Batch]
    grad_norm = gradients.reshape(batch_size, -1).norm(2, dim=1)

    # [Batch] -> scalar
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


class WGANGPTrainer:
    """Encapsulates a single WGAN-GP training iteration for the Market GAN."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        lr: Annotated[float, "learning rate for both Adam optimizers"] = 1e-4,
        betas: Annotated[tuple, "Adam beta coefficients"] = (0.5, 0.9),
        lambda_gp: Annotated[float, "gradient penalty coefficient (lambda)"] = 10.0,
        n_critic: Annotated[int, "number of critic updates per generator update"] = 5,
        lambda_moment: Annotated[
            float, "weight on the skew/kurtosis moment-matching penalty (0 disables it)"
        ] = 1.0,
        target_skewness: Annotated[
            Optional[float], "real-data terminal log-return skewness to match; None disables moment loss"
        ] = None,
        target_excess_kurtosis: Annotated[
            Optional[float], "real-data terminal log-return excess kurtosis to match; None disables moment loss"
        ] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or select_device()
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.lambda_moment = lambda_moment
        self.target_skewness = target_skewness
        self.target_excess_kurtosis = target_excess_kurtosis

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def train_discriminator_step(
        self, real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"]
    ) -> Annotated[float, "critic loss value for this step (-L_D)"]:
        real = real.to(self.device)
        batch_size, seq_len, _ = real.shape

        z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
        fake = self.generator(z).detach()

        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        gp = gradient_penalty(self.discriminator, real, fake, self.device)

        # Minimize -L_D = E[D(fake)] - E[D(real)] + lambda * gradient_penalty
        loss_d = d_fake.mean() - d_real.mean() + self.lambda_gp * gp

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d.item()

    def train_generator_step(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
    ) -> Annotated[dict, "{'loss_g': float, 'loss_adv': float, 'loss_moment': float}"]:
        z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
        fake = self.generator(z)

        # Maximize E[D(fake)] <=> minimize -E[D(fake)]
        loss_adv = -self.discriminator(fake).mean()

        # Moment-matching penalty: nudges the generator's terminal log-return
        # skew/kurtosis toward the real data's, since the adversarial loss
        # alone under-penalizes missing rare tail events (see RESULTS.md,
        # "The GAN fidelity story" -- the critic can be fooled by matching
        # mean/variance while ignoring tail shape entirely).
        loss_moment = torch.zeros((), device=self.device)
        if self.target_skewness is not None and self.target_excess_kurtosis is not None:
            fake_returns = terminal_log_return(fake)
            fake_skew = skewness_tensor(fake_returns)
            fake_kurtosis = excess_kurtosis_tensor(fake_returns)
            loss_moment = (fake_skew - self.target_skewness) ** 2 + (
                fake_kurtosis - self.target_excess_kurtosis
            ) ** 2

        loss_g = loss_adv + self.lambda_moment * loss_moment

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        return {"loss_g": loss_g.item(), "loss_adv": loss_adv.item(), "loss_moment": loss_moment.item()}

    def train_step(
        self, real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"]
    ) -> Annotated[dict, "{'loss_d': float, 'loss_g': float, 'loss_adv': float, 'loss_moment': float}"]:
        batch_size, seq_len, _ = real.shape

        loss_d = None
        for _ in range(self.n_critic):
            loss_d = self.train_discriminator_step(real)

        generator_stats = self.train_generator_step(batch_size, seq_len)
        return {"loss_d": loss_d, **generator_stats}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the WGAN-GP market generator.")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-critic", type=int, default=5, help="critic updates per generator update")
    parser.add_argument("--lambda-gp", type=float, default=10.0, help="gradient penalty coefficient")
    parser.add_argument(
        "--lambda-moment",
        type=float,
        default=1.0,
        help="weight on the skew/kurtosis moment-matching penalty (see RESULTS.md tail-shape gap)",
    )
    parser.add_argument(
        "--disable-moment-loss",
        action="store_true",
        help="train with the plain adversarial loss only, no moment-matching penalty",
    )
    parser.add_argument(
        "--moment-target-batch-size",
        type=int,
        default=5000,
        help="real-data sample size used once, up front, to fix the target skew/kurtosis",
    )
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--s0", type=float, default=1.0, help="initial price of 'real' training data")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["synthetic", "yfinance"],
        default="synthetic",
        help="synthetic = single-regime GBM placeholder (offline, deterministic); "
        "yfinance = real historical OHLCV data via generator.data.HistoricalPriceLoader",
    )
    parser.add_argument("--vol", type=float, default=0.2, help="volatility of 'real' training data (synthetic source)")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Yahoo Finance ticker (yfinance source)")
    parser.add_argument("--data-start", type=str, default="1950-01-03", help="history start date (yfinance source)")
    parser.add_argument("--data-end", type=str, default="2021-01-25", help="history end date (yfinance source)")
    parser.add_argument(
        "--price-column", type=str, default="Adj Close", help="OHLCV column used as the price series (yfinance source)"
    )
    parser.add_argument("--data-cache-dir", type=str, default="data", help="local CSV cache directory (yfinance source)")
    parser.add_argument(
        "--skip-fidelity-check",
        action="store_true",
        help="skip the post-training real-vs-synthetic fidelity check (generator.validate)",
    )
    parser.add_argument(
        "--fidelity-output-dir", type=str, default="results", help="directory for the fidelity check's plots/JSON"
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/market_gan.pt",
        help="path to save trained generator/discriminator weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="force a specific device ('cpu', 'cuda', 'mps'). Default (None) "
        "auto-detects the fastest available (cuda > mps > cpu) -- see "
        "common/device.py.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    generator = Generator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        initial_price=args.s0,
    )
    discriminator = Discriminator(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    if args.data_source == "yfinance":
        loader = HistoricalPriceLoader(
            ticker=args.ticker,
            start=args.data_start,
            end=args.data_end,
            price_column=args.price_column,
            cache_dir=Path(args.data_cache_dir),
        )
        print(
            f"Using real historical data: {args.ticker} [{args.data_start}, {args.data_end}], "
            f"{loader.prices.shape[0]} observations (cache: {loader.cache_path})"
        )

        def sample_real(batch_size: int, seq_len: int) -> torch.Tensor:
            return loader.sample(batch_size, seq_len, initial_price=args.s0)

    else:
        print("Using synthetic single-regime GBM 'real' data (offline placeholder).")

        def sample_real(batch_size: int, seq_len: int) -> torch.Tensor:
            return sample_real_prices(batch_size, seq_len, s0=args.s0, vol=args.vol)

    target_skewness = None
    target_excess_kurtosis = None
    if not args.disable_moment_loss:
        moment_target_real = sample_real(args.moment_target_batch_size, args.seq_len)
        moment_target_returns = terminal_log_return(moment_target_real)
        target_skewness = skewness(moment_target_returns)
        target_excess_kurtosis = excess_kurtosis(moment_target_returns)
        print(
            f"Moment-matching enabled (lambda={args.lambda_moment}): target skewness "
            f"{target_skewness:+.2f}, target excess kurtosis {target_excess_kurtosis:+.2f} "
            f"(from {args.moment_target_batch_size} real paths)"
        )
    else:
        print("Moment-matching disabled (--disable-moment-loss): plain adversarial loss only.")

    trainer = WGANGPTrainer(
        generator,
        discriminator,
        lr=args.lr,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        lambda_moment=args.lambda_moment,
        target_skewness=target_skewness,
        target_excess_kurtosis=target_excess_kurtosis,
        device=device,
    )

    print(f"Training WGAN-GP market generator for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        real = sample_real(args.batch_size, args.seq_len)
        stats = trainer.train_step(real)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}/{args.epochs}  loss_d={stats['loss_d']:.4f}  "
                f"loss_g={stats['loss_g']:.4f}  loss_adv={stats['loss_adv']:.4f}  "
                f"loss_moment={stats['loss_moment']:.4f}"
            )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    if not args.skip_fidelity_check:
        print("Running real-vs-synthetic fidelity check...")
        generator.eval()
        with torch.no_grad():
            # Skew/kurtosis are high-order moments with high sampling
            # variance -- args.batch_size (the WGAN minibatch, e.g. 64) is
            # far too small to estimate them reliably, so this reuses the
            # larger moment_target_batch_size sample instead.
            real_check = sample_real(args.moment_target_batch_size, args.seq_len)
            z = generator.sample_noise(args.moment_target_batch_size, args.seq_len)
            synthetic_check = generator(z)
        validate_generator_fidelity(
            real_check, synthetic_check, output_dir=Path(args.fidelity_output_dir)
        )


if __name__ == "__main__":
    main()
