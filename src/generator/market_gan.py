"""Market Generator networks for synthetic asset price path simulation.

Implements the generator/discriminator pair used by the WGAN-GP training
step in ``train_gan.py`` (see math_spec.md, section 4).
"""

from typing import Annotated, Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    """GRU-based generator mapping noise to synthetic price paths.

    The network outputs per-step log-return increments and integrates them
    via a cumulative sum before exponentiating, which guarantees the
    resulting price path is strictly positive (S_t = S_0 * exp(cumsum) > 0)
    regardless of the sign of the raw network output.
    """

    def __init__(
        self,
        noise_dim: Annotated[int, "dimensionality of the per-step noise vector"] = 8,
        hidden_dim: Annotated[int, "GRU hidden state size"] = 64,
        num_layers: Annotated[int, "number of stacked GRU layers"] = 2,
        initial_price: Annotated[float, "S_0, starting price of generated paths"] = 1.0,
        return_scale: Annotated[float, "bound applied to per-step log-returns via tanh"] = 0.1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.initial_price = initial_price
        self.return_scale = return_scale

        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self, z: Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, 1] strictly positive price path"]:
        # [Batch, Time_Steps, noise_dim] -> [Batch, Time_Steps, hidden_dim]
        hidden_states, _ = self.rnn(z)

        # [Batch, Time_Steps, hidden_dim] -> [Batch, Time_Steps, 1]
        raw_increments = self.output_layer(hidden_states)

        # Bound increments so cumulative sum (and exp) stays numerically stable.
        log_returns = torch.tanh(raw_increments) * self.return_scale

        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps, 1] (cumulative log-return)
        cumulative_log_returns = torch.cumsum(log_returns, dim=1)

        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps, 1] (S_t = S_0 * exp(sum of log-returns))
        prices = self.initial_price * torch.exp(cumulative_log_returns)
        return prices

    def sample_noise(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
        device: Optional[torch.device] = None,
    ) -> Annotated[torch.Tensor, "[Batch, Time_Steps, noise_dim] ~ N(0, I)"]:
        return torch.randn(batch_size, seq_len, self.noise_dim, device=device)


class Discriminator(nn.Module):
    """LSTM-based critic evaluating temporal realism of price paths.

    Used as the WGAN-GP critic D(x): no output activation (unbounded real
    valued score), consistent with the Wasserstein critic loss in
    math_spec.md section 4.
    """

    def __init__(
        self,
        input_dim: Annotated[int, "number of features per time step (price = 1)"] = 1,
        hidden_dim: Annotated[int, "LSTM hidden state size"] = 64,
        num_layers: Annotated[int, "number of stacked LSTM layers"] = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real or generated price path"]
    ) -> Annotated[torch.Tensor, "[Batch, 1] scalar realism score D(x)"]:
        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps, hidden_dim]
        hidden_states, _ = self.rnn(x)

        # [Batch, Time_Steps, hidden_dim] -> [Batch, hidden_dim] (last time step)
        last_hidden = hidden_states[:, -1, :]

        # [Batch, hidden_dim] -> [Batch, 1]
        score = self.output_layer(last_hidden)
        return score
