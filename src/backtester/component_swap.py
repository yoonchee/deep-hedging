"""Build hybrid checkpoints that take some parameter groups from one
checkpoint and the rest from another.

Written to localise GRU's severity transition: transplanting a severe run's
post-transition readout layer onto its own pre-transition network, and the
reverse, separates which half of the network carries the collapse. The hybrids
are ordinary checkpoints, so the stress harness and the recovery probe consume
them unchanged.

See RESULTS.md, "What moves at the transition", for the result.

Usage:
    python src/backtester/component_swap.py base.pt donor.pt \
        --take output_layer --out hybrids/rnnPRE_outPOST.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Annotated, Dict, Sequence

import torch

# Allow `python src/backtester/component_swap.py` to resolve sibling packages
# the same way pytest's `pythonpath = src` does, regardless of invocation style.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def build_hybrid_state_dict(
    base: Annotated[Dict[str, torch.Tensor], "state_dict supplying every parameter not taken"],
    donor: Annotated[Dict[str, torch.Tensor], "state_dict supplying the taken parameters"],
    take_prefixes: Annotated[Sequence[str], "parameter-name prefixes to take from the donor"],
) -> Annotated[Dict[str, torch.Tensor], "hybrid state_dict with base's keys"]:
    """Takes every parameter whose name starts with one of `take_prefixes` from
    `donor` and the rest from `base`.
    """
    if set(base) != set(donor):
        raise ValueError("checkpoints have different parameter sets; they are not the same architecture")

    hybrid: Dict[str, torch.Tensor] = {}
    taken = 0
    for name, tensor in base.items():
        if any(name.startswith(prefix) for prefix in take_prefixes):
            hybrid[name] = donor[name].clone()
            taken += 1
        else:
            hybrid[name] = tensor.clone()

    if taken == 0:
        raise ValueError(f"no parameter matched {list(take_prefixes)}; nothing would be swapped")
    if taken == len(base):
        raise ValueError(f"{list(take_prefixes)} matched every parameter; the result is just the donor")
    return hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="checkpoint supplying the untaken parameters")
    parser.add_argument("donor", type=Path, help="checkpoint supplying the taken parameters")
    parser.add_argument("--take", nargs="+", required=True, help="parameter-name prefixes, e.g. output_layer rnn.weight_hh_l0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    hybrid = build_hybrid_state_dict(base["policy_state_dict"], donor["policy_state_dict"], args.take)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Carry the base checkpoint's args so the hybrid loads under the same
    # architecture/strike/implied-vol the swap was defined against.
    torch.save({"policy_state_dict": hybrid, "cvar_h": base["cvar_h"], "args": base["args"]}, args.out)
    print(f"wrote {args.out} (took {args.take} from {args.donor.name})")


if __name__ == "__main__":
    main()
