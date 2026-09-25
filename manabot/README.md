# manabot

The trainable agent of Etude Fantasia: environment wrappers, models, search,
training, and verification. You train *a* manabot; this package is how.

## Quickstart

```bash
uv run manabot train              # --preset local: bounded laptop run
uv run manabot train --preset simple    # full PPO run (CUDA, W&B)
uv run manabot sim --preset sim --set sim.hero=attention --set sim.villain=simple
uv run manabot belief-demo        # engine-derived belief/intervention proof
uv run manabot belief-learn-demo  # held-out frozen-population belief proof
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

For an identified Allies/Lessons opponent, use the bounded
[train/export/play workflow](../docs/local-trained-challenger.md). It uses the
existing search-distillation trainer and a server-pinned candidate; players see
the bot's identity in the ordinary table. The `local` PPO preset above remains
a separate default-deck training smoke.

Build named curated matches with `MatchHypers.authored(pack_key, hero_deck,
villain_deck)` so both main decks and sideboards come from managym's compiled
setup. Custom `hero_sideboard` / `villain_sideboard` maps default to empty and
require positive integer counts. `Match.swapped()` moves both lists together;
`Env.reset(options={"match": match})` also replaces the setup used by auto-reset.

## World identity

An observation/action-shape change is a world version, and artifacts are only
comparable within a world. The **w2** baselines and porting rules are in
[WORLDS.md](../WORLDS.md). Full Learn changes that world's rules and inputs;
the [Learn integration record](../docs/rules/learn-lesson.md) tracks the
unfinished w3 declaration and ordinary checkpoint binding. Existing w2
checkpoints do not certify the corrected rules, even if they load or their
dimensions match.

## Architecture

manabot owns memory, beliefs, planning, policy/value learning, teacher data,
self-play, evaluation, and promotion evidence over the authoritative managym
world. The cross-package contracts and convergence status are in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

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
7. **`manabot.belief`**: canonical world distributions, viewer history,
   supervised exact-world learning, and the `ManabotPlayer` lifecycle

Belief input is opt-in: schema-bound checkpoints load through the ordinary
`checkpoint` player and update belief before each policy/value decision.
`belief-demo` uses a freshly initialized policy to prove intervention wiring;
`belief-learn-demo` trains a belief scorer on one scripted population and holds
out whole episodes. Neither demonstrates strategic strength or opponent
transfer. The generic PPO and teacher-shard trainers do not yet produce the
semantic belief inputs required to train a belief-enabled policy. Historical
positional-condition checkpoints are rejected rather than reinterpreted.

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
