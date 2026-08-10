"""Shared training-device selection (src/common/device.py).

Every training CLI in this repo previously hardcoded
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`, which never
considers Apple Silicon's MPS backend. Benchmarked directly on an M5 Pro
(`RecurrentHedgingAgent`, hidden_dim=64, batch=1000, seq_len=30, matching
`train_policy.py`'s paper-scale defaults): MPS is ~5.5x faster than CPU for
this workload (14ms/step vs. 79ms/step), which was the dominant cost in
every long training run in this project's history (single paper-scale LSTM
policy runs took 78 minutes on CPU). TimeGAN's own training loop
(`train_timegan.py`, batch=178 -- much smaller than the policy's batch=1000)
is the exception: MPS is actually ~1.3x *slower* there, dispatch overhead
dominating at that batch size. Net effect on the full pipeline (TimeGAN +
policy) still strongly favors MPS, since policy training is the far larger
cost. `PolicyTrainer.smoothness_penalty_weight`'s double-backward
(`torch.autograd.grad(..., create_graph=True)` through an `nn.LSTM`) is not
supported on MPS at all as of torch 2.8 (`derivative for lstm_mps_backward
is not implemented`) -- callers combining that flag with an MPS device get
a clear error at construction time (see `PolicyTrainer.__init__`) rather
than a mid-training crash.
"""

from typing import Annotated, Optional

import torch


def select_device(
    override: Annotated[
        Optional[str],
        "explicit device string ('cpu', 'cuda', 'mps', ...) -- bypasses "
        "auto-detection entirely when set. None (default) auto-detects.",
    ] = None,
) -> torch.device:
    """cuda > mps > cpu, the fastest available accelerator by default."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
