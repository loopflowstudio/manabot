"""Convert a PyTorch policy checkpoint for the MLX port.

    uv run --extra mlx python -m manabot.model.mlx.convert CHECKPOINT.pt

Writes `<stem>.mlx.safetensors` beside the checkpoint
(`manabot.model.mlx.checkpoint.convert`).
"""

import argparse
from pathlib import Path

from manabot.model.mlx.checkpoint import UnsupportedForMLX, convert


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        target, tensors = convert(args.checkpoint)
    except (FileNotFoundError, UnsupportedForMLX) as error:
        parser.error(str(error))
    print(f"read   {args.checkpoint}  ({tensors} tensors)")
    print(f"wrote  {target}")


if __name__ == "__main__":
    main()
