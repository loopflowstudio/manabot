"""Convert a PyTorch policy checkpoint to the shared checkpoint container.

    uv run python -m manabot.backends.convert CHECKPOINT.pt

Reads the `.pt` file with PyTorch and writes `<stem>.safetensors` beside it
(`manabot.policy.checkpoint`), which every framework backend loads. Every
torch tensor becomes one canonical parameter or lookup buffer, or conversion
fails naming the key. A backend refuses configurations it does not implement
when loading the container.
"""

import argparse
from dataclasses import replace
from pathlib import Path

from manabot.backends.torch.checkpoint import read_written, restore_policy
from manabot.policy.checkpoint import (
    UnsupportedPolicy,
    admit_checkpoint,
    write_container,
)


def convert(checkpoint: Path) -> tuple[Path, int]:
    """Validate on this runtime and write the writer's original tensors and metadata."""
    container = read_written(checkpoint)
    restore_policy(admit_checkpoint(container))
    target = checkpoint.with_suffix(".safetensors")
    write_container(
        target,
        replace(
            container,
            metadata={**container.metadata, "converted_from": str(checkpoint)},
        ),
    )
    return target, len(container.buffers) + len(container.parameters)


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
