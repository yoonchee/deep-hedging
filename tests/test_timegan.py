"""Tests for TimeGAN (src/generator/timegan.py, src/generator/train_timegan.py)."""

import torch

from common.stats import excess_kurtosis, skewness, terminal_log_return
from generator.timegan import Embedder, LatentDiscriminator, Recovery, Supervisor, TimeGAN, TimeGANGenerator
from generator.train_timegan import (
    MinMaxScaler,
    TimeGANPriceGenerator,
    TimeGANTrainer,
    _synthetic_multivariate_prices,
    load_timegan_price_generator,
)


def test_embedder_output_shape_and_bounded() -> None:
    batch, seq, feature_dim, hidden_dim = 8, 20, 5, 16
    embedder = Embedder(feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=1)
    x = torch.rand(batch, seq, feature_dim)

    h = embedder(x)

    assert h.shape == (batch, seq, hidden_dim)
    assert torch.all(h >= -1.0) and torch.all(h <= 1.0)


def test_recovery_output_shape_and_bounded() -> None:
    batch, seq, feature_dim, hidden_dim = 8, 20, 5, 16
    recovery = Recovery(feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=1)
    h = torch.rand(batch, seq, hidden_dim)

    x_tilde = recovery(h)

    assert x_tilde.shape == (batch, seq, feature_dim)
    assert torch.all(x_tilde >= -1.0) and torch.all(x_tilde <= 1.0)


def test_timegan_generator_output_shape_and_bounded() -> None:
    batch, seq, noise_dim, hidden_dim = 8, 20, 8, 16
    generator = TimeGANGenerator(noise_dim=noise_dim, hidden_dim=hidden_dim, num_layers=1)
    z = generator.sample_noise(batch, seq)

    h_hat = generator(z)

    assert h_hat.shape == (batch, seq, hidden_dim)
    assert torch.all(h_hat >= -1.0) and torch.all(h_hat <= 1.0)


def test_supervisor_output_shape_and_bounded() -> None:
    batch, seq, hidden_dim = 8, 20, 16
    supervisor = Supervisor(hidden_dim=hidden_dim, num_layers=1)
    h = torch.rand(batch, seq, hidden_dim)

    h_supervised = supervisor(h)

    assert h_supervised.shape == (batch, seq, hidden_dim)
    assert torch.all(h_supervised >= -1.0) and torch.all(h_supervised <= 1.0)


def test_latent_discriminator_output_is_per_timestep() -> None:
    batch, seq, hidden_dim = 8, 20, 16
    discriminator = LatentDiscriminator(hidden_dim=hidden_dim, num_layers=1)
    h = torch.rand(batch, seq, hidden_dim)

    score = discriminator(h)

    assert score.shape == (batch, seq, 1)


def test_timegan_generate_output_shape_and_bounded() -> None:
    batch, seq, feature_dim = 8, 20, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=16, noise_dim=8, num_layers=1)
    z = timegan.sample_noise(batch, seq)

    x_hat = timegan.generate(z)

    assert x_hat.shape == (batch, seq, feature_dim)
    assert torch.all(x_hat >= -1.0) and torch.all(x_hat <= 1.0)


def test_minmax_scaler_round_trip() -> None:
    scale = torch.tensor([1.0, 2.0, 0.5, 3.0, 1000.0])
    offset = torch.tensor([0.0, 10.0, -5.0, 0.0, 500.0])
    x = torch.rand(100, 30, 5) * scale + offset

    scaler = MinMaxScaler(feature_dim=5)
    scaler.fit(x)
    scaled = scaler.transform(x)

    assert scaled.min() >= -1 - 1e-5 and scaled.max() <= 1 + 1e-5

    recovered = scaler.inverse_transform(scaled)
    assert torch.allclose(recovered, x, atol=1e-4)


def test_synthetic_multivariate_prices_shape_and_positive() -> None:
    batch, seq = 16, 30
    x = _synthetic_multivariate_prices(batch, seq)

    assert x.shape == (batch, seq, 5)
    assert torch.all(x > 0)


def test_pretrain_autoencoder_step_reduces_reconstruction_loss() -> None:
    torch.manual_seed(0)
    batch, seq, feature_dim = 32, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1)
    trainer = TimeGANTrainer(timegan, lr=1e-2, device=torch.device("cpu"))

    x_real = torch.rand(batch, seq, feature_dim)
    early = trainer.pretrain_autoencoder_step(x_real)
    later = early
    for _ in range(50):
        later = trainer.pretrain_autoencoder_step(x_real)

    assert later < early


def test_pretrain_supervisor_step_runs() -> None:
    batch, seq, feature_dim = 8, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1)
    trainer = TimeGANTrainer(timegan, device=torch.device("cpu"))

    loss = trainer.pretrain_supervisor_step(torch.rand(batch, seq, feature_dim))

    assert isinstance(loss, float)


def test_train_step_phase3_runs_and_returns_expected_keys() -> None:
    batch, seq, feature_dim = 4, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1)
    trainer = TimeGANTrainer(timegan, n_critic=1, device=torch.device("cpu"))

    stats = trainer.train_step_phase3(torch.rand(batch, seq, feature_dim))

    for key in ("loss_d", "loss_er", "loss", "loss_adv", "loss_supervised", "loss_moment"):
        assert key in stats


def test_moment_loss_is_zero_without_target() -> None:
    batch, seq, feature_dim = 4, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1)
    trainer = TimeGANTrainer(timegan, n_critic=1, device=torch.device("cpu"))

    stats = trainer.train_step_phase3(torch.rand(batch, seq, feature_dim))

    assert stats["loss_moment"] == 0.0


def test_moment_loss_is_nonzero_with_target() -> None:
    batch, seq, feature_dim = 4, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1, price_index=3)
    trainer = TimeGANTrainer(
        timegan,
        n_critic=1,
        target_skewness=-0.9,
        target_excess_kurtosis=4.2,
        price_min=0.5,
        price_max=1.5,
        device=torch.device("cpu"),
    )

    stats = trainer.train_step_phase3(torch.rand(batch, seq, feature_dim))

    assert stats["loss_moment"] > 0.0


def test_timegan_price_generator_matches_generator_protocol() -> None:
    batch, seq, feature_dim = 8, 15, 5
    timegan = TimeGAN(feature_dim=feature_dim, hidden_dim=12, noise_dim=8, num_layers=1, price_index=3)
    scaler = MinMaxScaler(feature_dim=feature_dim)
    scaler.fit(_synthetic_multivariate_prices(50, seq))

    adapter = TimeGANPriceGenerator(timegan, scaler)
    z = adapter.sample_noise(batch, seq)
    prices = adapter(z)

    assert prices.shape == (batch, seq, 1)


def test_load_timegan_price_generator_round_trips_checkpoint(tmp_path) -> None:
    feature_columns = ["Open", "High", "Low", "Close", "Volume"]
    price_index = 3
    timegan = TimeGAN(
        feature_dim=5, hidden_dim=12, noise_dim=8, num_layers=1, feature_columns=feature_columns, price_index=price_index
    )
    scaler = MinMaxScaler(feature_dim=5)
    scaler.fit(_synthetic_multivariate_prices(50, 15))

    checkpoint_path = tmp_path / "timegan.pt"
    torch.save(
        {
            "timegan_state_dict": timegan.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "feature_columns": feature_columns,
            "price_index": price_index,
            "args": {"hidden_dim": 12, "noise_dim": 8, "num_layers": 1},
        },
        checkpoint_path,
    )

    adapter = load_timegan_price_generator(checkpoint_path)
    z = adapter.sample_noise(4, 15)
    prices = adapter(z)

    assert prices.shape == (4, 15, 1)
    for param in adapter.timegan.parameters():
        assert not param.requires_grad


def test_moment_loss_pulls_price_channel_skew_kurtosis_toward_target() -> None:
    torch.manual_seed(0)
    seq = 20
    price_index = 3  # Open, High, Low, Close, Volume
    target_skew, target_kurtosis = -0.9, 4.2

    real_raw = _synthetic_multivariate_prices(200, seq)
    scaler = MinMaxScaler(feature_dim=5)
    scaler.fit(real_raw)
    price_min = scaler.min_vals[price_index].item()
    price_max = scaler.max_vals[price_index].item()

    timegan = TimeGAN(feature_dim=5, hidden_dim=12, noise_dim=8, num_layers=1, price_index=price_index)
    trainer = TimeGANTrainer(
        timegan,
        n_critic=1,
        lr=3e-3,
        lambda_moment=10.0,
        target_skewness=target_skew,
        target_excess_kurtosis=target_kurtosis,
        price_min=price_min,
        price_max=price_max,
        device=torch.device("cpu"),
    )

    def moment_gap() -> float:
        with torch.no_grad():
            z = timegan.sample_noise(1000, seq)
            synthetic_scaled = timegan.generate(z)
            price_scaled = synthetic_scaled[..., price_index : price_index + 1]
            price = (price_scaled + 1.0) / 2.0 * (price_max - price_min) + price_min
            returns = terminal_log_return(price)
            return abs(skewness(returns) - target_skew) + abs(excess_kurtosis(returns) - target_kurtosis)

    gap_before = moment_gap()

    # tanh-bounded latents converge more slowly here than the earlier
    # sigmoid version did -- 250 steps needed for a reliable margin, vs 100
    # before (see RESULTS.md's TimeGAN section for why tanh replaced sigmoid).
    x_real = scaler.transform(real_raw[:32])
    for _ in range(250):
        trainer.train_step_phase3(x_real)

    gap_after = moment_gap()
    assert gap_after < gap_before
