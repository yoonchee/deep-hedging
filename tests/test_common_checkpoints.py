"""Tests for the canonical checkpoint filename convention (src/common/checkpoints.py).

Guards the single source of truth that replaced three independent
implementations of this naming scheme (see the module's own docstring for
the two real bugs that gap caused).
"""

from common.checkpoints import checkpoint_filename


def test_mlp_default_path_has_no_architecture_suffix() -> None:
    assert checkpoint_filename("mlp") == "hedging_agent.pt"


def test_non_mlp_default_path_includes_architecture() -> None:
    assert checkpoint_filename("rnn") == "hedging_agent_rnn.pt"
    assert checkpoint_filename("lstm") == "hedging_agent_lstm.pt"
    assert checkpoint_filename("gru") == "hedging_agent_gru.pt"


def test_timegan_suffix_applies_to_every_architecture() -> None:
    assert checkpoint_filename("mlp", suffix="_timegan") == "hedging_agent_timegan.pt"
    assert checkpoint_filename("rnn", suffix="_timegan") == "hedging_agent_rnn_timegan.pt"


def test_alpha_sweep_path_encodes_alpha_and_ignores_mlp_special_case() -> None:
    # alpha=0.997 -> "0_997"; the mlp-has-no-architecture-name special case
    # only applies to the alpha=None default path, not alpha-sweep paths.
    assert checkpoint_filename("mlp", alpha=0.997) == "hedging_agent_mlp_alpha0_997.pt"
    assert checkpoint_filename("rnn", alpha=0.5) == "hedging_agent_rnn_alpha0_5.pt"


def test_alpha_sweep_path_with_timegan_suffix() -> None:
    # This combination -- an alpha-sweep checkpoint trained against TimeGAN
    # -- was previously unreachable: evaluate.py::load_alpha_sweep_checkpoints
    # had no suffix parameter, so it could never find a filename like this
    # one even though train_policy.py could produce it.
    assert (
        checkpoint_filename("mlp", alpha=0.995, suffix="_timegan")
        == "hedging_agent_mlp_alpha0_995_timegan.pt"
    )


def test_alpha_string_formatting_matches_every_paper_part_ii_grid_value() -> None:
    # The paper's Part II alpha grid, per RESULTS.md -- these are the exact
    # values --alpha-sweep is run with in practice; each must round-trip
    # through the "%.4g then replace '.' with '_'" formatting without
    # collisions or unexpected truncation.
    expected = {
        0.5: "hedging_agent_mlp_alpha0_5.pt",
        0.75: "hedging_agent_mlp_alpha0_75.pt",
        0.9: "hedging_agent_mlp_alpha0_9.pt",
        0.95: "hedging_agent_mlp_alpha0_95.pt",
        0.99: "hedging_agent_mlp_alpha0_99.pt",
        0.995: "hedging_agent_mlp_alpha0_995.pt",
        0.997: "hedging_agent_mlp_alpha0_997.pt",
    }
    for alpha, filename in expected.items():
        assert checkpoint_filename("mlp", alpha=alpha) == filename
