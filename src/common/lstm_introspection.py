"""Manual LSTM unroll for per-step diagnostics (src/common/lstm_introspection.py).

`nn.LSTM.forward()` returns only the final layer's per-step hidden state and
the last layer's final (hidden, cell) pair -- it never exposes intermediate
layers' per-step cell state or gate activations, which RESULTS.md's
mechanism (b) diagnosis needs to distinguish "hidden state saturated" from
"a few units moving in coordination" from "a level- vs. velocity-triggered
transition" for LSTM (TimeGAN)'s stress-test failure. This re-implements
`nn.LSTM`'s forward pass step by step, using the same trained weights, to
expose that internal state -- kept here rather than only in a scratch
script so the RESULTS.md claims that depend on it stay reproducible from
the repo. See `unroll_lstm_with_gates`'s docstring for the equations and
`tests/test_lstm_introspection.py` for the numerical check against
`nn.LSTM`'s real forward pass.
"""

from typing import Annotated, Dict

import torch
import torch.nn as nn


def unroll_lstm_with_gates(
    rnn: Annotated[nn.LSTM, "a trained (batch_first=True) nn.LSTM, any num_layers"],
    inputs: Annotated[torch.Tensor, "[Batch, Time, input_size]"],
) -> Annotated[
    Dict[int, Dict[str, torch.Tensor]],
    "{layer_index: {'h', 'c', 'i', 'f', 'g', 'o'}}, each [Batch, Time, hidden_size] -- "
    "h/c are the hidden/cell state, i/f/g/o the input/forget/candidate/output gates, "
    "all per time step (unlike rnn.forward()'s single final (h_n, c_n))",
]:
    """Standard LSTM recurrence, unrolled manually so every step's gates and
    cell state are visible, not just the final one nn.LSTM.forward() returns.

    Per layer, per step t (PyTorch's gate order: input, forget, cell, output):
        gates = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh
        i_t, f_t, g_t, o_t = split(gates, 4)
        i_t, f_t, o_t = sigmoid(i_t), sigmoid(f_t), sigmoid(o_t)
        g_t = tanh(g_t)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """
    batch, seq_len, _ = inputs.shape
    hidden = rnn.hidden_size
    layer_input = inputs
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in range(rnn.num_layers):
        w_ih = getattr(rnn, f"weight_ih_l{layer}")
        w_hh = getattr(rnn, f"weight_hh_l{layer}")
        b_ih = getattr(rnn, f"bias_ih_l{layer}")
        b_hh = getattr(rnn, f"bias_hh_l{layer}")

        # [Batch, hidden] x2, the recurrent state carried across steps
        h_t = torch.zeros(batch, hidden, dtype=inputs.dtype)
        c_t = torch.zeros(batch, hidden, dtype=inputs.dtype)
        hs, cs, gi, gf, gg, go = [], [], [], [], [], []
        for t in range(seq_len):
            # [Batch, input_size] -> [Batch, 4*hidden] (stacked i/f/g/o pre-activations)
            x_t = layer_input[:, t, :]
            gates = x_t @ w_ih.T + b_ih + h_t @ w_hh.T + b_hh
            i_g, f_g, g_g, o_g = gates.chunk(4, dim=-1)
            i_g, f_g, o_g = torch.sigmoid(i_g), torch.sigmoid(f_g), torch.sigmoid(o_g)
            g_g = torch.tanh(g_g)
            c_t = f_g * c_t + i_g * g_g
            h_t = o_g * torch.tanh(c_t)
            hs.append(h_t); cs.append(c_t)
            gi.append(i_g); gf.append(f_g); gg.append(g_g); go.append(o_g)
        # [Time] x [Batch, hidden] -> [Batch, Time, hidden], per stacked tensor
        layer_input = torch.stack(hs, dim=1)
        result[layer] = dict(
            h=torch.stack(hs, dim=1), c=torch.stack(cs, dim=1),
            i=torch.stack(gi, dim=1), f=torch.stack(gf, dim=1),
            g=torch.stack(gg, dim=1), o=torch.stack(go, dim=1),
        )
    return result
