"""Convert a PyTorch policy checkpoint to the shared checkpoint container.

    uv run python -m manabot.backends.convert CHECKPOINT.pt

Reads the `.pt` file with PyTorch and writes `<stem>.safetensors` beside it
(`manabot.policy.checkpoint`), which every framework backend loads. Every
torch tensor becomes one canonical parameter or lookup buffer, or conversion
fails naming the key; a configuration the backends do not implement is
refused with every unsupported field named.
"""

import argparse
from pathlib import Path

import torch

from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.policy.checkpoint import (
    UnsupportedPolicy,
    checkpoint_metadata,
    from_torch_state,
    write_container,
)


def convert(checkpoint: Path) -> tuple[Path, int]:
    """Write the container for a PyTorch checkpoint; return it and the number
    of torch tensors read."""
    agent, space = load_checkpoint(str(checkpoint))
    record = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.detach().numpy() for k, v in agent.state_dict().items()}
    container = from_torch_state(
        agent.config,
        space.encoder.config,
        state,
        checkpoint_metadata(record, checkpoint),
    )
    target = checkpoint.with_suffix(".safetensors")
    write_container(target, container)
    return target, len(state)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint.suffix == ".safetensors":
        parser.error(f"{args.checkpoint} is already a container; pass its .pt")
    try:
        target, tensors = convert(args.checkpoint)
    except (FileNotFoundError, UnsupportedPolicy) as error:
        parser.error(str(error))
    print(f"read   {args.checkpoint}  ({tensors} tensors)")
    print(f"wrote  {target}")


if __name__ == "__main__":
    main()
