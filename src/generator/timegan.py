"""TimeGAN networks for multi-variate synthetic market path generation.

Implements Yoon et al. (2019), "Time-series Generative Adversarial
Networks" -- the architecture Kim (2021) uses as Part II's market
simulator, in place of the single-feature WGAN-GP in ``market_gan.py``.
Five networks operating in a shared bounded latent space (see
``train_timegan.py`` for the 3-phase training procedure, math_spec.md
section 5 for the loss formulas):

    Embedder   E: X (real, [Batch,Time,F])      -> H_real [Batch,Time,H]
    Recovery   R: H (any)                       -> X_tilde [Batch,Time,F]
    Generator  G: Z (noise, [Batch,Time,Z_dim])  -> H_hat [Batch,Time,H]
    Supervisor S: H (any)                       -> next-step H prediction
    Discriminator D: H (any)                    -> per-step realism score

Embedder/Recovery/Generator/Supervisor all end in sigmoid, bounding every
latent code (and the reconstructed/generated features) to [0,1] -- this is
also why the recovered price channel is automatically non-negative with no
special-cased activation, as long as the real data fed to Embedder was
itself min-max scaled to [0,1] first (see data.py::MinMaxScaler).
"""

from typing import Annotated, Optional

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """Maps real multi-feature paths to bounded latent codes H_real."""

    def __init__(
        self,
        feature_dim: Annotated[int, "number of input features F (e.g. 5 for O,H,L,C,V)"],
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        num_layers: Annotated[int, "stacked GRU layers"] = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: Annotated[torch.Tensor, "[Batch, Time_Steps, F] real path, features in [0, 1]"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, H] latent code H_real, in [0, 1]"]:
        # [Batch, Time_Steps, F] -> [Batch, Time_Steps, H]
        hidden_states, _ = self.rnn(x)
        return torch.sigmoid(self.output_layer(hidden_states))


class Recovery(nn.Module):
    """Maps latent codes back to reconstructed multi-feature paths."""

    def __init__(
        self,
        feature_dim: Annotated[int, "number of output features F"],
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        num_layers: Annotated[int, "stacked GRU layers"] = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(
        self, h: Annotated[torch.Tensor, "[Batch, Time_Steps, H] latent code, in [0, 1]"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, F] reconstructed path, in [0, 1]"]:
        # [Batch, Time_Steps, H] -> [Batch, Time_Steps, F]
        hidden_states, _ = self.rnn(h)
        return torch.sigmoid(self.output_layer(hidden_states))


class TimeGANGenerator(nn.Module):
    """Maps noise to synthetic latent codes H_hat, in the same bounded space as Embedder's output."""

    def __init__(
        self,
        noise_dim: Annotated[int, "dimensionality of the per-step noise vector"] = 8,
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        num_layers: Annotated[int, "stacked GRU layers"] = 2,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.rnn = nn.GRU(input_size=noise_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, z: Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, H] synthetic latent code H_hat, in [0, 1]"]:
        # [Batch, Time_Steps, noise_dim] -> [Batch, Time_Steps, H]
        hidden_states, _ = self.rnn(z)
        return torch.sigmoid(self.output_layer(hidden_states))

    def sample_noise(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
        device: Optional[torch.device] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]:
        return torch.randn(batch_size, seq_len, self.noise_dim, device=device)


class Supervisor(nn.Module):
    """Predicts the next-step latent code from the sequence so far.

    Trained via MSE(H[:, 1:, :], Supervisor(H)[:, :-1, :]) -- the core
    TimeGAN mechanism that teaches the generator real stepwise dynamics
    instead of just matching a static terminal distribution.
    """

    def __init__(
        self,
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        num_layers: Annotated[int, "stacked GRU layers"] = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, h: Annotated[torch.Tensor, "[Batch, Time_Steps, H] latent code (real H or fake H_hat)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, H] next-step latent code prediction, in [0, 1]"]:
        # [Batch, Time_Steps, H] -> [Batch, Time_Steps, H]
        hidden_states, _ = self.rnn(h)
        return torch.sigmoid(self.output_layer(hidden_states))


class LatentDiscriminator(nn.Module):
    """Per-timestep WGAN-GP critic over latent codes.

    Unlike market_gan.Discriminator (which pools to a single sequence-level
    score), this returns a score at every timestep -- TimeGAN's own
    discriminator classifies local realism at each step, not just the
    sequence as a whole.
    """

    def __init__(
        self,
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        num_layers: Annotated[int, "stacked LSTM layers"] = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self, h: Annotated[torch.Tensor, "[Batch, Time_Steps, H] latent code (real H or fake H_hat)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] per-step realism score D(h_t)"]:
        # [Batch, Time_Steps, H] -> [Batch, Time_Steps, H]
        hidden_states, _ = self.rnn(h)

        # [Batch, Time_Steps, H] -> [Batch, Time_Steps, 1]
        return self.output_layer(hidden_states)


class TimeGAN(nn.Module):
    """Container for the five TimeGAN networks plus end-to-end generation.

    ``feature_columns``/``price_index`` record which output channel of the
    F-feature path is "the price" that downstream hedging code (which only
    ever consumes a single-feature [Batch, Time, 1] tensor) should read --
    see generator/train_timegan.py::TimeGANPriceGenerator.
    """

    def __init__(
        self,
        feature_dim: Annotated[int, "number of features F (e.g. 5 for O,H,L,C,V)"],
        hidden_dim: Annotated[int, "latent dimension H"] = 24,
        noise_dim: Annotated[int, "generator noise dimension"] = 8,
        num_layers: Annotated[int, "stacked recurrent layers, all five networks"] = 2,
        feature_columns: Annotated[Optional[list], "feature names, in output-channel order"] = None,
        price_index: Annotated[int, "output channel index that is 'the price' (e.g. Close)"] = 0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.feature_columns = feature_columns
        self.price_index = price_index

        self.embedder = Embedder(feature_dim, hidden_dim, num_layers)
        self.recovery = Recovery(feature_dim, hidden_dim, num_layers)
        self.generator = TimeGANGenerator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = LatentDiscriminator(hidden_dim, num_layers)

    def generate(
        self, z: Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, F] synthetic multi-feature path, in [0, 1]"]:
        # [Batch, Time_Steps, noise_dim] -> [Batch, Time_Steps, H] -> [Batch, Time_Steps, H] -> [Batch, Time_Steps, F]
        h_hat = self.generator(z)
        h_hat_supervised = self.supervisor(h_hat)
        return self.recovery(h_hat_supervised)

    def sample_noise(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
        device: Optional[torch.device] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]:
        return self.generator.sample_noise(batch_size, seq_len, device=device)
