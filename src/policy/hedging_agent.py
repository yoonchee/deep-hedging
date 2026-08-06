"""Deep Hedging policy networks (src/policy/hedging_agent.py).

Two families:

- `HedgingAgent`: a feed-forward MLP shared across time steps, with
  delta_{t-1} fed back in as an explicit input feature (Buehler et al.'s
  `delta_k = f(I_k, delta_{k-1})` formulation).
- `RecurrentHedgingAgent`: a genuine RNN/LSTM/GRU that consumes the whole
  price path in one pass and lets its own hidden state carry history and
  elapsed time implicitly, matching Kim (2021) "Deep Hedging, Generative
  Adversarial Networks, and Beyond", which compares exactly these three
  cell types against Black-Scholes.
"""

from typing import Annotated, List, Literal

import torch
import torch.nn as nn


class HedgingAgent(nn.Module):
    """Feed-forward hedging policy: state -> delta_t in [0, 1].

    State at step t: (S_t, delta_{t-1}, T - t, implied_vol).
    """

    STATE_DIM = 4  # S_t, delta_{t-1}, time_to_maturity, implied_vol

    def __init__(
        self,
        hidden_dim: Annotated[int, "width of each hidden layer"] = 32,
        num_hidden_layers: Annotated[int, "number of hidden layers"] = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = self.STATE_DIM
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: Annotated[
            torch.Tensor, "[Batch, 4] = (S_t, delta_{t-1}, T - t, implied_vol)"
        ],
    ) -> Annotated[torch.Tensor, "[Batch, 1] next hedge ratio delta_t in [0, 1]"]:
        # [Batch, 4] -> [Batch, 1]
        raw_output = self.net(state)

        # [Batch, 1] -> [Batch, 1] (constrain hedge ratio to [0, 1])
        delta_t = torch.sigmoid(raw_output)
        return delta_t


class RecurrentHedgingAgent(nn.Module):
    """True recurrent hedging policy: full price path -> full hedge-ratio path.

    Unlike `HedgingAgent`, this network never sees delta_{t-1}, T-t, or
    implied_vol as explicit inputs -- it sees only the raw price sequence,
    and its own recurrent hidden state implicitly encodes elapsed time and
    prior decisions. One forward pass over the whole path produces every
    delta_t at once, which `MarketEnvironment` consumes via its
    `sequence_policy=True` mode (see src/environment/market_env.py).
    """

    CELL_TYPES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(
        self,
        cell_type: Annotated[
            Literal["rnn", "lstm", "gru"], "which recurrent cell to use"
        ] = "gru",
        hidden_dim: Annotated[int, "RNN hidden state size"] = 64,
        num_layers: Annotated[int, "number of stacked recurrent layers"] = 2,
    ) -> None:
        super().__init__()
        if cell_type not in self.CELL_TYPES:
            raise ValueError(
                f"cell_type must be one of {sorted(self.CELL_TYPES)}, got {cell_type!r}"
            )
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_cls = self.CELL_TYPES[cell_type]
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] full price path S_0..S_N"],
    ) -> Annotated[
        torch.Tensor, "[Batch, Time_Steps - 1, 1] hedge ratios delta_0..delta_{N-1} in [0, 1]"
    ]:
        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps - 1, 1] (no decision needed at S_N)
        inputs = prices[:, :-1, :]

        # [Batch, Time_Steps - 1, 1] -> [Batch, Time_Steps - 1, hidden_dim]
        hidden_states, _ = self.rnn(inputs)

        # [Batch, Time_Steps - 1, hidden_dim] -> [Batch, Time_Steps - 1, 1]
        raw_output = self.output_layer(hidden_states)
        delta_path = torch.sigmoid(raw_output)
        return delta_path
