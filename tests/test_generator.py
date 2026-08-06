"""Tests for the Market Generator (src/generator/market_gan.py)."""

import torch

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
