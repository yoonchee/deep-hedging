"""Tests for the Market Generator (src/generator/market_gan.py)."""

import torch

from common.stats import excess_kurtosis, skewness, terminal_log_return
from generator.market_gan import Discriminator, Generator
from generator.train_gan import WGANGPTrainer


def test_generator_output_shape() -> None:
    batch_size, seq_len, noise_dim = 16, 30, 8
    generator = Generator(noise_dim=noise_dim, hidden_dim=32, num_layers=2)

    z = generator.sample_noise(batch_size, seq_len)
    prices = generator(z)

    assert prices.shape == (batch_size, seq_len, 1)


def test_generator_output_strictly_positive() -> None:
    batch_size, seq_len, noise_dim = 32, 50, 8
    generator = Generator(noise_dim=noise_dim, hidden_dim=32, num_layers=2)

    # Use an untrained (randomly initialized) generator and extreme noise to
    # stress-test that positivity holds regardless of raw network output.
    z = torch.randn(batch_size, seq_len, noise_dim) * 10.0
    prices = generator(z)

    assert torch.all(prices > 0)


def test_discriminator_output_shape() -> None:
    batch_size, seq_len = 16, 30
    generator = Generator(noise_dim=8, hidden_dim=32, num_layers=2)
    discriminator = Discriminator(input_dim=1, hidden_dim=32, num_layers=2)

    z = generator.sample_noise(batch_size, seq_len)
    fake_prices = generator(z)
    score = discriminator(fake_prices)

    assert score.shape == (batch_size, 1)


def test_wgan_gp_train_step_runs() -> None:
    batch_size, seq_len = 4, 20
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    discriminator = Discriminator(input_dim=1, hidden_dim=16, num_layers=1)
    trainer = WGANGPTrainer(
        generator, discriminator, device=torch.device("cpu"), n_critic=1
    )

    real = torch.randn(batch_size, seq_len, 1).abs() + 1.0
    losses = trainer.train_step(real)

    assert "loss_d" in losses and "loss_g" in losses
    assert isinstance(losses["loss_d"], float)
    assert isinstance(losses["loss_g"], float)


def test_moment_loss_is_zero_without_target() -> None:
    batch_size, seq_len = 4, 20
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    discriminator = Discriminator(input_dim=1, hidden_dim=16, num_layers=1)
    trainer = WGANGPTrainer(generator, discriminator, device=torch.device("cpu"), n_critic=1)

    real = torch.randn(batch_size, seq_len, 1).abs() + 1.0
    losses = trainer.train_step(real)

    assert "loss_adv" in losses and "loss_moment" in losses
    assert losses["loss_moment"] == 0.0


def test_moment_loss_is_nonzero_with_target() -> None:
    batch_size, seq_len = 4, 20
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    discriminator = Discriminator(input_dim=1, hidden_dim=16, num_layers=1)
    trainer = WGANGPTrainer(
        generator,
        discriminator,
        device=torch.device("cpu"),
        n_critic=1,
        target_skewness=-0.9,
        target_excess_kurtosis=4.2,
    )

    real = torch.randn(batch_size, seq_len, 1).abs() + 1.0
    losses = trainer.train_step(real)

    assert losses["loss_moment"] > 0.0


def test_moment_loss_pulls_generator_skew_kurtosis_toward_target() -> None:
    torch.manual_seed(0)
    batch_size, seq_len = 32, 20
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    discriminator = Discriminator(input_dim=1, hidden_dim=16, num_layers=1)

    target_skew, target_kurtosis = -0.9, 4.2
    trainer = WGANGPTrainer(
        generator,
        discriminator,
        device=torch.device("cpu"),
        n_critic=1,
        lambda_moment=5.0,
        target_skewness=target_skew,
        target_excess_kurtosis=target_kurtosis,
    )

    def moment_gap() -> float:
        with torch.no_grad():
            z = generator.sample_noise(2000, seq_len)
            returns = terminal_log_return(generator(z))
            return abs(skewness(returns) - target_skew) + abs(excess_kurtosis(returns) - target_kurtosis)

    gap_before = moment_gap()

    real = torch.randn(batch_size, seq_len, 1).abs() + 1.0
    for _ in range(150):
        trainer.train_step(real)

    gap_after = moment_gap()
    assert gap_after < gap_before
