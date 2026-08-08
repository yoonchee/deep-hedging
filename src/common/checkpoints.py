"""Canonical trained-policy checkpoint filename convention (src/common/checkpoints.py).

Single source of truth, shared between `policy/train_policy.py` (the writer)
and `backtester/evaluate.py` (the readers) -- before this existed, the same
naming scheme (`hedging_agent_<architecture>[_alpha<alpha>][<suffix>].pt`)
was implemented independently in three places with no way to ask "what path
will this write to?" from outside the writer's own CLI branching logic. That
gap caused two real problems: `evaluate.py::load_alpha_sweep_checkpoints`
never grew a `suffix` parameter, so it could never find a TimeGAN-trained
alpha-sweep checkpoint even though the writer could produce one; and a bare
`--cvar-alpha 0.997` retrain (intended for `hedging_agent_mlp_alpha0_997.pt`)
silently overwrote `hedging_agent.pt`, the main-table checkpoint, because
there was no single function to check the intended path against before
running (see RESULTS.md's mechanism (a) writeup).
"""

from typing import Annotated, Optional


def checkpoint_filename(
    architecture: Annotated[str, "'mlp', 'rnn', 'lstm', or 'gru'"],
    alpha: Annotated[
        Optional[float],
        "CVaR alpha, for a --alpha-sweep checkpoint; None (default) for the "
        "single-run path used by --cvar-alpha without --alpha-sweep",
    ] = None,
    suffix: Annotated[
        str, "'_timegan' for TimeGAN-trained checkpoints, '' (default) for WGAN-GP"
    ] = "",
) -> Annotated[
    str,
    "filename only, e.g. 'hedging_agent_rnn_alpha0_997_timegan.pt' -- "
    "caller joins this with its own checkpoint directory",
]:
    if alpha is not None:
        alpha_str = f"{alpha:.4g}".replace(".", "_")
        return f"hedging_agent_{architecture}_alpha{alpha_str}{suffix}.pt"
    if architecture == "mlp":
        return f"hedging_agent{suffix}.pt"
    return f"hedging_agent_{architecture}{suffix}.pt"
