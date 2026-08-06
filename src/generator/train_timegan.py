"""3-phase TimeGAN training (src/generator/train_timegan.py).

Implements Yoon et al. (2019)'s training procedure for the networks in
``timegan.py`` (math_spec.md, section 5):

    Phase 1 -- autoencoder pretraining:  Embedder + Recovery, reconstruction loss
    Phase 2 -- supervised pretraining:   Supervisor, next-step latent loss on real data
    Phase 3 -- joint adversarial training: D (WGAN-GP critic), G+S (adversarial +
               supervised + moment-matching on the price channel), and a small
               continued E+R reconstruction update, alternated each iteration.

Run directly as a training CLI:

    python src/generator/train_timegan.py --phase1-epochs 50 --phase2-epochs 50 --phase3-epochs 100
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import torch.nn as nn

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.stats import excess_kurtosis, excess_kurtosis_tensor, skewness, skewness_tensor, terminal_log_return  # noqa: E402
from generator.data import HistoricalPriceLoader, MinMaxScaler, sample_real_prices  # noqa: E402
from generator.timegan import TimeGAN  # noqa: E402
from generator.train_gan import gradient_penalty  # noqa: E402
from generator.validate import validate_generator_fidelity  # noqa: E402


class TimeGANTrainer:
    """Encapsulates the 3-phase TimeGAN training procedure for one TimeGAN instance."""

    def __init__(
        self,
        timegan: TimeGAN,
        lr: Annotated[float, "Adam learning rate, all optimizer groups"] = 1e-3,
        betas: Annotated[tuple, "Adam beta coefficients"] = (0.5, 0.9),
        lambda_gp: Annotated[float, "gradient penalty coefficient (phase 3)"] = 10.0,
        n_critic: Annotated[int, "discriminator updates per generator update (phase 3)"] = 5,
        lambda_supervised: Annotated[float, "weight on the fake-data supervised loss (phase 3)"] = 1.0,
        lambda_recon_joint: Annotated[float, "weight on the continued E+R reconstruction update (phase 3)"] = 0.1,
        lambda_moment: Annotated[float, "weight on the skew/kurtosis moment-matching penalty (0 disables it)"] = 1.0,
        target_skewness: Annotated[
            Optional[float], "real price-channel terminal log-return skewness to match; None disables moment loss"
        ] = None,
        target_excess_kurtosis: Annotated[
            Optional[float], "real price-channel terminal log-return excess kurtosis to match; None disables it"
        ] = None,
        price_min: Annotated[Optional[float], "MinMaxScaler min for the price channel, needed to invert it"] = None,
        price_max: Annotated[Optional[float], "MinMaxScaler max for the price channel, needed to invert it"] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timegan = timegan.to(self.device)
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.lambda_supervised = lambda_supervised
        self.lambda_recon_joint = lambda_recon_joint
        self.lambda_moment = lambda_moment
        self.target_skewness = target_skewness
        self.target_excess_kurtosis = target_excess_kurtosis
        self.price_min = price_min
        self.price_max = price_max
        self.price_index = timegan.price_index

        self.mse = nn.MSELoss()

        er_params = list(self.timegan.embedder.parameters()) + list(self.timegan.recovery.parameters())
        gs_params = list(self.timegan.generator.parameters()) + list(self.timegan.supervisor.parameters())

        self.optimizer_er = torch.optim.Adam(er_params, lr=lr, betas=betas)
        self.optimizer_s_pretrain = torch.optim.Adam(self.timegan.supervisor.parameters(), lr=lr, betas=betas)
        self.optimizer_gs = torch.optim.Adam(gs_params, lr=lr, betas=betas)
        self.optimizer_d = torch.optim.Adam(self.timegan.discriminator.parameters(), lr=lr, betas=betas)

    # ---- Phase 1: autoencoder pretraining ----

    def pretrain_autoencoder_step(
        self, x_real: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [-1, 1]"]
    ) -> Annotated[float, "reconstruction loss for this step"]:
        x_real = x_real.to(self.device)
        h = self.timegan.embedder(x_real)
        x_tilde = self.timegan.recovery(h)
        loss = self.mse(x_tilde, x_real)

        self.optimizer_er.zero_grad()
        loss.backward()
        self.optimizer_er.step()
        return loss.item()

    # ---- Phase 2: supervised pretraining ----

    def pretrain_supervisor_step(
        self, x_real: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [-1, 1]"]
    ) -> Annotated[float, "supervised loss for this step"]:
        x_real = x_real.to(self.device)
        with torch.no_grad():
            h_real = self.timegan.embedder(x_real)

        h_supervised = self.timegan.supervisor(h_real)
        loss = self.mse(h_supervised[:, :-1, :], h_real[:, 1:, :])

        self.optimizer_s_pretrain.zero_grad()
        loss.backward()
        self.optimizer_s_pretrain.step()
        return loss.item()

    # ---- Phase 3: joint adversarial training ----

    def _invert_price_channel(
        self, price_scaled: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price channel, in [-1, 1]"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price channel, price-ratio scale"]:
        return (price_scaled + 1.0) / 2.0 * (self.price_max - self.price_min) + self.price_min

    def train_discriminator_step(
        self, x_real: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [-1, 1]"]
    ) -> Annotated[float, "critic loss value for this step (-L_D)"]:
        x_real = x_real.to(self.device)
        batch_size, seq_len, _ = x_real.shape

        with torch.no_grad():
            h_real = self.timegan.embedder(x_real)
            z = self.timegan.sample_noise(batch_size, seq_len, device=self.device)
            h_hat = self.timegan.supervisor(self.timegan.generator(z))

        d_real = self.timegan.discriminator(h_real)
        d_fake = self.timegan.discriminator(h_hat)
        gp = gradient_penalty(self.timegan.discriminator, h_real, h_hat, self.device)

        # Minimize -L_D = E[D(fake)] - E[D(real)] + lambda * gradient_penalty
        loss_d = d_fake.mean() - d_real.mean() + self.lambda_gp * gp

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d.item()

    def train_generator_supervisor_step(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
    ) -> Annotated[dict, "{'loss': float, 'loss_adv': float, 'loss_supervised': float, 'loss_moment': float}"]:
        z = self.timegan.sample_noise(batch_size, seq_len, device=self.device)
        h_hat = self.timegan.generator(z)
        h_hat_supervised = self.timegan.supervisor(h_hat)

        # Maximize E[D(fake)] <=> minimize -E[D(fake)]
        loss_adv = -self.timegan.discriminator(h_hat_supervised).mean()

        # Keeps G's own stepwise dynamics matching the Supervisor's real-data-
        # trained prediction as G updates (the core TimeGAN mechanism).
        loss_supervised = self.mse(h_hat_supervised[:, :-1, :], h_hat[:, 1:, :])

        loss_moment = torch.zeros((), device=self.device)
        if self.target_skewness is not None and self.target_excess_kurtosis is not None:
            x_hat = self.timegan.recovery(h_hat_supervised)
            price_scaled = x_hat[..., self.price_index : self.price_index + 1]
            price = self._invert_price_channel(price_scaled)
            fake_returns = terminal_log_return(price)
            fake_skew = skewness_tensor(fake_returns)
            fake_kurtosis = excess_kurtosis_tensor(fake_returns)
            loss_moment = (fake_skew - self.target_skewness) ** 2 + (
                fake_kurtosis - self.target_excess_kurtosis
            ) ** 2

        loss = loss_adv + self.lambda_supervised * loss_supervised + self.lambda_moment * loss_moment

        self.optimizer_gs.zero_grad()
        loss.backward()
        self.optimizer_gs.step()
        return {
            "loss": loss.item(),
            "loss_adv": loss_adv.item(),
            "loss_supervised": loss_supervised.item(),
            "loss_moment": loss_moment.item(),
        }

    def train_embedder_recovery_joint_step(
        self, x_real: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [-1, 1]"]
    ) -> Annotated[float, "weighted reconstruction loss for this step"]:
        x_real = x_real.to(self.device)
        h = self.timegan.embedder(x_real)
        x_tilde = self.timegan.recovery(h)
        loss = self.lambda_recon_joint * self.mse(x_tilde, x_real)

        self.optimizer_er.zero_grad()
        loss.backward()
        self.optimizer_er.step()
        return loss.item()

    def train_step_phase3(
        self, x_real: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [-1, 1]"]
    ) -> Annotated[dict, "{'loss_d', 'loss_er', 'loss', 'loss_adv', 'loss_supervised', 'loss_moment'}"]:
        batch_size, seq_len, _ = x_real.shape

        loss_d = None
        for _ in range(self.n_critic):
            loss_d = self.train_discriminator_step(x_real)

        gs_stats = self.train_generator_supervisor_step(batch_size, seq_len)
        loss_er = self.train_embedder_recovery_joint_step(x_real)

        return {"loss_d": loss_d, "loss_er": loss_er, **gs_stats}


class TimeGANPriceGenerator(nn.Module):
    """Adapts a trained TimeGAN to the single-feature Generator protocol
    policy/train_policy.py::PolicyTrainer expects (sample_noise(...) /
    __call__(z) -> [Batch, Time_Steps, 1]) -- generates the full
    multi-feature path and slices out the price channel, inverse-scaled
    back to price-ratio scale. The same adapter pattern already used for
    GBMDataSource in backtester/replicate_part1.py.
    """

    def __init__(self, timegan: TimeGAN, scaler: "MinMaxScaler") -> None:
        super().__init__()
        self.timegan = timegan
        self.scaler = scaler
        self.price_index = timegan.price_index

    def sample_noise(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
        device: Optional[torch.device] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]:
        return self.timegan.sample_noise(batch_size, seq_len, device=device)

    def forward(
        self, z: Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] price-ratio-scale price path"]:
        # [Batch, Time_Steps, noise_dim] -> [Batch, Time_Steps, F], in [-1, 1]
        x_hat_scaled = self.timegan.generate(z)

        # [Batch, Time_Steps, F] -> [Batch, Time_Steps, F], price-ratio scale
        x_hat = self.scaler.inverse_transform(x_hat_scaled)

        # [Batch, Time_Steps, F] -> [Batch, Time_Steps, 1]
        return x_hat[..., self.price_index : self.price_index + 1]


def load_timegan_price_generator(
    checkpoint_path: Annotated[Path, "checkpoint saved by this module's --checkpoint"],
) -> TimeGANPriceGenerator:
    """Loads a trained TimeGAN checkpoint as a frozen TimeGANPriceGenerator,
    ready to be passed to policy/train_policy.py::PolicyTrainer in place of
    a market_gan.Generator.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    feature_columns = checkpoint["feature_columns"]
    price_index = checkpoint["price_index"]

    timegan = TimeGAN(
        feature_dim=len(feature_columns),
        hidden_dim=train_args["hidden_dim"],
        noise_dim=train_args["noise_dim"],
        num_layers=train_args["num_layers"],
        feature_columns=feature_columns,
        price_index=price_index,
    )
    timegan.load_state_dict(checkpoint["timegan_state_dict"])
    for param in timegan.parameters():
        param.requires_grad_(False)

    scaler = MinMaxScaler(len(feature_columns))
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return TimeGANPriceGenerator(timegan, scaler)


def _synthetic_multivariate_prices(
    batch_size: Annotated[int, "number of paths"],
    seq_len: Annotated[int, "number of time steps per path"],
    s0: Annotated[float, "initial asset price S_0"] = 1.0,
    vol: Annotated[float, "single-regime GBM volatility for the Close channel"] = 0.2,
) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 5] synthetic (Open, High, Low, Close, Volume) paths"]:
    """Offline, deterministic multi-feature placeholder (--data-source synthetic).

    Derives plausible Open/High/Low/Volume around the existing single-regime
    GBM Close series (generator.data.sample_real_prices) for fast, network-free
    smoke testing -- not intended as a realistic OHLCV model. For actual
    historical data, see HistoricalPriceLoader.sample_multivariate.
    """
    close = sample_real_prices(batch_size, seq_len, s0=s0, vol=vol).squeeze(-1)  # [Batch, Time_Steps]

    # [Batch, Time_Steps] -> [Batch, Time_Steps] (yesterday's close as today's open)
    open_ = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    high = torch.maximum(open_, close) * (1.0 + torch.rand_like(close) * 0.01)
    low = torch.minimum(open_, close) * (1.0 - torch.rand_like(close) * 0.01)
    volume = torch.exp(torch.randn(batch_size, seq_len) * 0.3 + 10.0)

    # [Batch, Time_Steps] x5 -> [Batch, Time_Steps, 5]
    return torch.stack([open_, high, low, close, volume], dim=-1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TimeGAN, the paper's Part II multi-variate market generator.")
    parser.add_argument("--phase1-epochs", type=int, default=200, help="autoencoder pretraining epochs")
    parser.add_argument("--phase2-epochs", type=int, default=200, help="supervisor pretraining epochs")
    parser.add_argument("--phase3-epochs", type=int, default=500, help="joint adversarial training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=24, help="latent dimension H")
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2, help="stacked recurrent layers, all five networks")
    parser.add_argument("--n-critic", type=int, default=5, help="critic updates per generator update (phase 3)")
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--lambda-supervised", type=float, default=1.0)
    parser.add_argument("--lambda-recon-joint", type=float, default=0.1)
    parser.add_argument("--lambda-moment", type=float, default=1.0)
    parser.add_argument("--disable-moment-loss", action="store_true")
    parser.add_argument("--moment-target-batch-size", type=int, default=5000)
    parser.add_argument("--s0", type=float, default=1.0, help="initial asset price of 'real' training data")
    parser.add_argument("--vol", type=float, default=0.2, help="volatility of 'real' training data (synthetic source)")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["synthetic", "yfinance"],
        default="synthetic",
        help="synthetic = single-regime-GBM-derived OHLCV placeholder (offline, deterministic); "
        "yfinance = real historical OHLCV data via generator.data.HistoricalPriceLoader.sample_multivariate",
    )
    parser.add_argument(
        "--feature-columns", type=str, default="Open,High,Low,Close,Volume", help="comma-separated feature names"
    )
    parser.add_argument("--price-column", type=str, default="Close", help="which feature column is 'the price'")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Yahoo Finance ticker (yfinance source)")
    parser.add_argument("--data-start", type=str, default="1950-01-03", help="history start date (yfinance source)")
    parser.add_argument("--data-end", type=str, default="2021-01-25", help="history end date (yfinance source)")
    parser.add_argument("--data-cache-dir", type=str, default="data", help="local CSV cache directory (yfinance source)")
    parser.add_argument(
        "--skip-fidelity-check", action="store_true", help="skip the post-training price-channel fidelity check"
    )
    parser.add_argument("--fidelity-output-dir", type=str, default="results/timegan")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/timegan.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns = args.feature_columns.split(",")
    feature_dim = len(feature_columns)
    price_index = feature_columns.index(args.price_column)

    timegan = TimeGAN(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        num_layers=args.num_layers,
        feature_columns=feature_columns,
        price_index=price_index,
    )

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
        rescale_columns = [c for c in feature_columns if c != "Volume"]

        def sample_real_raw(batch_size: int, seq_len: int) -> torch.Tensor:
            return loader.sample_multivariate(
                batch_size,
                seq_len,
                feature_columns=feature_columns,
                price_column=args.price_column,
                rescale_columns=rescale_columns,
                initial_price=args.s0,
            )

    else:
        print("Using synthetic single-regime-GBM-derived OHLCV 'real' data (offline placeholder).")

        def sample_real_raw(batch_size: int, seq_len: int) -> torch.Tensor:
            return _synthetic_multivariate_prices(batch_size, seq_len, s0=args.s0, vol=args.vol)

    scaler = MinMaxScaler(feature_dim)
    scaler.fit(sample_real_raw(max(args.moment_target_batch_size, args.batch_size), args.seq_len))

    def sample_real(batch_size: int, seq_len: int) -> torch.Tensor:
        return scaler.transform(sample_real_raw(batch_size, seq_len))

    target_skewness: Optional[float] = None
    target_excess_kurtosis: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    if not args.disable_moment_loss:
        moment_target_raw = sample_real_raw(args.moment_target_batch_size, args.seq_len)
        moment_target_price = moment_target_raw[..., price_index : price_index + 1]
        moment_target_returns = terminal_log_return(moment_target_price)
        target_skewness = skewness(moment_target_returns)
        target_excess_kurtosis = excess_kurtosis(moment_target_returns)
        price_min = scaler.min_vals[price_index].item()
        price_max = scaler.max_vals[price_index].item()
        print(
            f"Moment-matching enabled (lambda={args.lambda_moment}): target skewness "
            f"{target_skewness:+.2f}, target excess kurtosis {target_excess_kurtosis:+.2f} "
            f"(from {args.moment_target_batch_size} real paths)"
        )
    else:
        print("Moment-matching disabled (--disable-moment-loss).")

    trainer = TimeGANTrainer(
        timegan,
        lr=args.lr,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        lambda_supervised=args.lambda_supervised,
        lambda_recon_joint=args.lambda_recon_joint,
        lambda_moment=args.lambda_moment,
        target_skewness=target_skewness,
        target_excess_kurtosis=target_excess_kurtosis,
        price_min=price_min,
        price_max=price_max,
        device=device,
    )

    print(f"Phase 1: autoencoder pretraining for {args.phase1_epochs} epochs on {device}...")
    for epoch in range(1, args.phase1_epochs + 1):
        x_real = sample_real(args.batch_size, args.seq_len)
        loss = trainer.pretrain_autoencoder_step(x_real)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.phase1_epochs:
            print(f"  epoch {epoch:4d}/{args.phase1_epochs}  recon_loss={loss:.4f}")

    print(f"Phase 2: supervisor pretraining for {args.phase2_epochs} epochs...")
    for epoch in range(1, args.phase2_epochs + 1):
        x_real = sample_real(args.batch_size, args.seq_len)
        loss = trainer.pretrain_supervisor_step(x_real)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.phase2_epochs:
            print(f"  epoch {epoch:4d}/{args.phase2_epochs}  supervised_loss={loss:.4f}")

    print(f"Phase 3: joint adversarial training for {args.phase3_epochs} epochs...")
    for epoch in range(1, args.phase3_epochs + 1):
        x_real = sample_real(args.batch_size, args.seq_len)
        stats = trainer.train_step_phase3(x_real)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.phase3_epochs:
            print(
                f"  epoch {epoch:4d}/{args.phase3_epochs}  loss_d={stats['loss_d']:.4f}  "
                f"loss_adv={stats['loss_adv']:.4f}  loss_supervised={stats['loss_supervised']:.4f}  "
                f"loss_moment={stats['loss_moment']:.4f}  loss_er={stats['loss_er']:.4f}"
            )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "timegan_state_dict": timegan.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "feature_columns": feature_columns,
            "price_index": price_index,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    if not args.skip_fidelity_check:
        print("Running real-vs-synthetic fidelity check on the price channel...")
        timegan.eval()
        with torch.no_grad():
            real_check_raw = sample_real_raw(args.moment_target_batch_size, args.seq_len)
            real_check_price = real_check_raw[..., price_index : price_index + 1]

            z = timegan.sample_noise(args.moment_target_batch_size, args.seq_len, device=device)
            synthetic_scaled = timegan.generate(z).cpu()
            synthetic_check_full = scaler.inverse_transform(synthetic_scaled)
            synthetic_check_price = synthetic_check_full[..., price_index : price_index + 1]

        validate_generator_fidelity(
            real_check_price, synthetic_check_price, output_dir=Path(args.fidelity_output_dir)
        )


if __name__ == "__main__":
    main()
