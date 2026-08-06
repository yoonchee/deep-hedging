"""Deep Hedging policy network (src/policy/hedging_agent.py).

A single feed-forward network shared across time steps, mapping the current
market state to the next hedge ratio delta_t. Weight sharing across steps is
what lets gradients from the terminal wealth (see src/environment/market_env.py)
propagate back through every step's decision.
"""

from typing import Annotated, List

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
