# manabot

The trainable agent of Etude Fantasia: environment wrappers, models, search,
training, and verification. You train *a* manabot; this package is how.

## Quickstart

```bash
uv run manabot train              # --preset local: bounded laptop run
uv run manabot train --preset simple    # full PPO run (CUDA, W&B)
uv run manabot sim --preset sim --set sim.hero=attention --set sim.villain=simple
```

The default `local` preset is the certified laptop path: it trains a small
manabot on CPU in under a minute, needs no W&B account or CUDA, and saves
checkpoints to `.runs/local/step_N.pt`. The `simple` and `attention` presets
are real training runs: they expect a CUDA machine (in practice Ubuntu on
AWS — see [ops/](../ops/README.md)) and track to the `manabot` Weights &
Biases project. Simulation pulls trained models from W&B and runs locally on
CPU at small scales.

Override any hyperparameter with `--set dotted.path=value`; presets live in
`manabot/config/presets.py`.

To face what you trained, launch `./scripts/play`, choose the **Checkpoint**
opponent, and enter your `.runs/local/step_*.pt` path — the server loads the
agent behind the same experience protocol as every other villain.

## You are in world w2

An observation/action-shape change is a world version, and artifacts are only
comparable within a world. The current world is **w2**; the live baselines
and the porting rules are in [WORLDS.md](../WORLDS.md). Before comparing
against or reusing any checkpoint, check its world tag.

## Architecture

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   (`VectorEnv`, `ObservationSpace`, `Match`, `Reward`)
2. **`manabot.model`**: trainable policy and value models (`Agent`, PPO
   `Trainer`)
3. **`manabot.sim`**: search, teacher data generation, and game simulation
   (`Player`, `Sim`, determinized PUCT, Teacher-1 evidence)
4. **`manabot.semantic`**: semantic-program encoders and structural
   representation experiments
5. **`manabot.verify`**: competency scenarios, behavioral probes, and the
   run-provenance store
6. **`manabot.infra`**: experiment tracking (W&B/TensorBoard), `Hypers`
   config models, profiling

Experiment-specific driver scripts live in
[experiments/runners/](../experiments/runners/), not here — `manabot/` keeps
only reusable instruments. The experiment discipline and ledger are in
[experiments/README.md](../experiments/README.md).

## Research program

[RESEARCH.md](RESEARCH.md) is the durable map from runnable manabots to a
bounded superhuman claim. It records the builder's loop, the world-pinned skill
rating we are establishing, accepted evidence, and the value-learning,
teacher/student, belief-modeling, and semantic-transfer frontiers. Live work
and ownership remain in Linear; frozen predictions and results remain in
`experiments/`.

## Style

```python
"""
filename.py
One-line purpose of file

Instructions for collaborators on how to approach understanding and editing.
"""

# Standard library
import os

# Third-party imports
from torch import Tensor

# First-party imports
from manabot.env import ObservationSpace

# Local imports
from .sibling import Thing
```
