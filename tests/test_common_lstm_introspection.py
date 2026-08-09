"""Tests for the manual LSTM unroll (src/common/lstm_introspection.py).

The whole point of this module is that its output must match nn.LSTM's real
forward pass exactly (it exists only to expose intermediate state nn.LSTM's
own forward() doesn't return) -- so the primary test is a direct numerical
comparison against nn.LSTM.forward(), at both 1 and 2 layers, matching the
RESULTS.md mechanism (b) diagnosis that depends on this equivalence holding.
"""

import torch
import torch.nn as nn

from common.lstm_introspection import unroll_lstm_with_gates


def test_unroll_matches_nn_lstm_single_layer() -> None:
    torch.manual_seed(0)
    rnn = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
    inputs = torch.randn(4, 10, 1)

    with torch.no_grad():
        real_hidden, (h_n, c_n) = rnn(inputs)
        manual = unroll_lstm_with_gates(rnn, inputs)

    assert torch.allclose(real_hidden, manual[0]["h"], atol=1e-6)
    assert torch.allclose(h_n[0], manual[0]["h"][:, -1, :], atol=1e-6)
    assert torch.allclose(c_n[0], manual[0]["c"][:, -1, :], atol=1e-6)


def test_unroll_matches_nn_lstm_two_layers() -> None:
    torch.manual_seed(1)
    rnn = nn.LSTM(input_size=1, hidden_size=8, num_layers=2, batch_first=True)
    inputs = torch.randn(4, 12, 1)

    with torch.no_grad():
        real_hidden, (h_n, c_n) = rnn(inputs)
        manual = unroll_lstm_with_gates(rnn, inputs)

    # real_hidden is the TOP layer's per-step hidden state
    assert torch.allclose(real_hidden, manual[rnn.num_layers - 1]["h"], atol=1e-6)
    for layer in range(rnn.num_layers):
        assert torch.allclose(h_n[layer], manual[layer]["h"][:, -1, :], atol=1e-6)
        assert torch.allclose(c_n[layer], manual[layer]["c"][:, -1, :], atol=1e-6)


def test_unroll_gates_are_in_valid_ranges() -> None:
    torch.manual_seed(2)
    rnn = nn.LSTM(input_size=1, hidden_size=8, num_layers=1, batch_first=True)
    inputs = torch.randn(4, 10, 1)

    with torch.no_grad():
        manual = unroll_lstm_with_gates(rnn, inputs)

    for gate in ["i", "f", "o"]:
        assert torch.all(manual[0][gate] >= 0.0) and torch.all(manual[0][gate] <= 1.0)
    assert torch.all(manual[0]["g"] >= -1.0) and torch.all(manual[0]["g"] <= 1.0)
    assert torch.all(manual[0]["h"] >= -1.0) and torch.all(manual[0]["h"] <= 1.0)
