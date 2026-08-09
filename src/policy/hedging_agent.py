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

import math
from typing import Annotated, List, Literal, Optional, Tuple

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
        strike: Annotated[
            float,
            "option strike K; S_t is divided by this before the network sees "
            "it, so the input is always moneyness-scale (~1) regardless of "
            "the underlying's raw price level. Default 1.0 is a no-op for "
            "already-normalized prices (S_0 = strike = 1.0, used throughout "
            "this project's stress-test pipeline).",
        ] = 1.0,
    ) -> None:
        super().__init__()
        self.strike = strike
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
        # [Batch, 4] -> [Batch, 1] x4 (unpack so S_t alone can be rescaled)
        S_t, delta_prev, time_to_maturity, implied_vol = state.split(1, dim=-1)

        # [Batch, 1] x4 -> [Batch, 4] (S_t rescaled to moneyness, ~1 scale)
        normalized_state = torch.cat(
            [S_t / self.strike, delta_prev, time_to_maturity, implied_vol], dim=-1
        )

        # [Batch, 4] -> [Batch, 1]
        raw_output = self.net(normalized_state)

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
    # Number of stacked weight matrices inside a single weight_hh_l* tensor
    # for each cell type (PyTorch concatenates per-gate matrices along dim 0).
    NUM_GATES = {"rnn": 1, "gru": 3, "lstm": 4}

    def __init__(
        self,
        cell_type: Annotated[
            Literal["rnn", "lstm", "gru"], "which recurrent cell to use"
        ] = "gru",
        hidden_dim: Annotated[int, "RNN hidden state size"] = 64,
        num_layers: Annotated[int, "number of stacked recurrent layers"] = 2,
        output_hidden_dims: Annotated[
            Optional[List[int]],
            "widths of FC layers between the RNN and the final delta output; "
            "None = a single linear layer (default). E.g. [64, 64] with "
            "hidden_dim=128 approximates Kim (2021)'s stated '128, 64, 64, 1' "
            "node counts, since nn.RNN/LSTM/GRU require uniform width per "
            "recurrent layer -- this reads it as RNN(128) -> FC(64) -> FC(64) -> 1.",
        ] = None,
        strike: Annotated[
            float,
            "option strike K, used in the log-moneyness input transform "
            "log(S_t / K); K itself only recenters the ratio, it doesn't fix "
            "the DC-dominance problem alone (see implied_vol/time_to_maturity "
            "below). Default 1.0 matches this project's normalized-price "
            "convention (S_0 = strike = 1.0).",
        ] = 1.0,
        orthogonal_init: Annotated[
            bool,
            "orthogonally initialize the recurrent (hidden-to-hidden) weight "
            "matrices, per gate -- a standard RNN-stabilization trick: an "
            "orthogonal matrix preserves gradient/activation norm through "
            "repeated application, which random (uniform-initialized) "
            "matrices don't, and can otherwise leave the network stuck at an "
            "input-insensitive constant output no matter how long it trains.",
        ] = False,
        implied_vol: Annotated[
            float,
            "implied volatility, used only to scale the log-moneyness input "
            "(see time_to_maturity below) -- NOT fed to the network as a "
            "per-step feature; it's a fixed hyperparameter of this policy "
            "instance, matching whatever implied_vol PolicyTrainer uses "
            "during training and stress-testing so the same instance is "
            "used consistently. Default 0.2 matches this project's typical "
            "CLI default (--implied-vol).",
        ] = 0.2,
        time_to_maturity: Annotated[
            float,
            "T, the option's total time to maturity, used only to scale the "
            "log-moneyness input: dividing log(S_t/K) by implied_vol * "
            "sqrt(T) turns a signal whose raw variation is a tiny fraction "
            "of S_t's magnitude (e.g. ~3% for Part I's S0=K=100, vol=0.15, "
            "T=1/12) into an O(1)-scaled quantity the RNN can actually learn "
            "from -- dividing raw price by strike alone (the previous "
            "approach) is scale-invariant to this signal-to-DC ratio and "
            "does not fix it. See RESULTS.md's RNN/LSTM diagnostic for the "
            "measured before/after. Default 1.0 is a generic placeholder for "
            "callers (e.g. shape/gradient tests) that don't care about "
            "matching a specific market scenario.",
        ] = 1.0,
        moneyness_clip: Annotated[
            Optional[Tuple[float, float]],
            "(lo, hi) bounds clamped onto the standardized log-moneyness input "
            "before it reaches the recurrent cell. None (default) is a no-op. "
            "Motivation: a bounded generator's training data only ever covers "
            "a finite input range, and this network has no principled behavior "
            "once its input leaves that range (see RESULTS.md mechanism (b)) "
            "-- widening the generator's training distribution just relocates "
            "that edge rather than removing it. Clamping the input instead "
            "means an arbitrarily extreme real price presents the same "
            "in-range value the network was actually trained on. Set to just "
            "inside the generator's own measured training boundary. Applied "
            "during training too (not just at inference) so the network's "
            "weights adapt to the clipped distribution rather than encounter "
            "it only at test time.",
        ] = None,
    ) -> None:
        super().__init__()
        if cell_type not in self.CELL_TYPES:
            raise ValueError(
                f"cell_type must be one of {sorted(self.CELL_TYPES)}, got {cell_type!r}"
            )
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.strike = strike
        self.implied_vol = implied_vol
        self.time_to_maturity = time_to_maturity
        self.moneyness_scale = implied_vol * math.sqrt(time_to_maturity)
        self.moneyness_clip = moneyness_clip

        rnn_cls = self.CELL_TYPES[cell_type]
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        if orthogonal_init:
            self._orthogonal_init_recurrent_weights()

        output_layers: List[nn.Module] = []
        in_dim = hidden_dim
        for width in output_hidden_dims or []:
            output_layers.append(nn.Linear(in_dim, width))
            output_layers.append(nn.ReLU())
            in_dim = width
        output_layers.append(nn.Linear(in_dim, 1))
        self.output_layer = nn.Sequential(*output_layers)

    def _orthogonal_init_recurrent_weights(self) -> None:
        num_gates = self.NUM_GATES[self.cell_type]
        for name, param in self.rnn.named_parameters():
            if "weight_hh" not in name:
                continue
            # [num_gates * hidden_dim, hidden_dim] -> orthogonal_() per gate,
            # not on the whole concatenated block (which wouldn't give each
            # individual gate's matrix orthogonal structure).
            for gate_idx in range(num_gates):
                start, end = gate_idx * self.hidden_dim, (gate_idx + 1) * self.hidden_dim
                nn.init.orthogonal_(param.data[start:end])

    def forward(
        self,
        prices: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] full price path S_0..S_N"],
    ) -> Annotated[
        torch.Tensor, "[Batch, Time_Steps - 1, 1] hedge ratios delta_0..delta_{N-1} in [0, 1]"
    ]:
        # [Batch, Time_Steps, 1] -> [Batch, Time_Steps - 1, 1] (no decision needed at S_N)
        inputs = prices[:, :-1, :]

        # [Batch, Time_Steps - 1, 1] -> [Batch, Time_Steps - 1, 1] (standardized
        # log-moneyness: log(S_t/K) / (implied_vol * sqrt(T)), O(1)-scaled)
        inputs = torch.log(inputs / self.strike) / self.moneyness_scale

        # [Batch, Time_Steps - 1, 1] -> [Batch, Time_Steps - 1, 1] (clamp to the
        # generator's own training-distribution boundary, if configured -- see
        # moneyness_clip docstring)
        if self.moneyness_clip is not None:
            lo, hi = self.moneyness_clip
            inputs = inputs.clamp(min=lo, max=hi)

        # [Batch, Time_Steps - 1, 1] -> [Batch, Time_Steps - 1, hidden_dim]
        hidden_states, _ = self.rnn(inputs)

        # [Batch, Time_Steps - 1, hidden_dim] -> [Batch, Time_Steps - 1, 1]
        raw_output = self.output_layer(hidden_states)
        delta_path = torch.sigmoid(raw_output)
        return delta_path
