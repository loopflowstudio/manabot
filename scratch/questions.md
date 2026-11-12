# MTG-293 assumptions

- Torch 2.15.0 Linux wheels exist only on the cu132 index (checked
  download.pytorch.org 2026-11-11: cu128 stops at 2.11.0, cu130 at 2.14.1, cu131
  and cu133 have none). Per the Task's fallback, the training runtime moves to
  `https://download.pytorch.org/whl/cu132` and `torch==2.15.0+cu132`. CUDA 13
  needs NVIDIA driver 580+ and compute capability 7.5+; not re-qualified on
  rented GPUs. `tests/academy/test_play_launcher.py:321` asserts `+cu128`.
- Baseline on Apple Silicon with Torch < 2.15 and no `--device` should fail
  with the minimum-version message, not the generic "requires CUDA or Apple
  MPS" message (docs/training.md says so). Simplest: baseline checks the MPS
  version admission when CUDA is absent and MPS is available.
- The minimum-version constant lives in `manabot/training/mps.py`, not
  `manabot/recipes/device.py`, because execution admission (training) must not
  import recipes; `accelerator()` reads it from there.
