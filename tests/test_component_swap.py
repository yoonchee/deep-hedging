"""Coverage for `backtester/component_swap.py`."""

import pytest
import torch

from backtester.component_swap import build_hybrid_state_dict


def _state(value: float) -> dict:
    return {
        "rnn.weight_hh_l0": torch.full((4, 4), value),
        "output_layer.0.weight": torch.full((1, 4), value),
        "output_layer.0.bias": torch.full((1,), value),
    }


def test_hybrid_takes_only_the_named_prefixes() -> None:
    hybrid = build_hybrid_state_dict(_state(1.0), _state(2.0), ["output_layer"])
    assert torch.all(hybrid["rnn.weight_hh_l0"] == 1.0), "untaken parameters come from base"
    assert torch.all(hybrid["output_layer.0.weight"] == 2.0), "taken parameters come from donor"
    assert torch.all(hybrid["output_layer.0.bias"] == 2.0)


def test_hybrid_can_target_a_single_parameter() -> None:
    hybrid = build_hybrid_state_dict(_state(1.0), _state(2.0), ["output_layer.0.weight"])
    assert torch.all(hybrid["output_layer.0.weight"] == 2.0)
    assert torch.all(hybrid["output_layer.0.bias"] == 1.0), "the bias must stay with base"


def test_hybrid_result_is_a_copy_not_a_view() -> None:
    base, donor = _state(1.0), _state(2.0)
    hybrid = build_hybrid_state_dict(base, donor, ["output_layer"])
    hybrid["rnn.weight_hh_l0"] += 5.0
    hybrid["output_layer.0.weight"] += 5.0
    assert torch.all(base["rnn.weight_hh_l0"] == 1.0), "mutating the hybrid must not touch its sources"
    assert torch.all(donor["output_layer.0.weight"] == 2.0)


@pytest.mark.parametrize("prefixes", [["nonexistent"], ["rnn", "output_layer"]])
def test_degenerate_swaps_are_rejected(prefixes: list) -> None:
    # Neither a no-op nor a full copy is a swap; both are silent-mistake shapes.
    with pytest.raises(ValueError):
        build_hybrid_state_dict(_state(1.0), _state(2.0), prefixes)


def test_mismatched_architectures_are_rejected() -> None:
    smaller = {"rnn.weight_hh_l0": torch.zeros(4, 4)}
    with pytest.raises(ValueError):
        build_hybrid_state_dict(_state(1.0), smaller, ["rnn"])
