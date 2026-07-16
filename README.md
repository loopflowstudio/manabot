# Etude Fantasia

**Etude Fantasia** is an AI-native research game for
[Magic: The Gathering](https://magic.wizards.com/): exact, creator-selected
matchups for developing strong agents and studying their decisions through a
finished play experience. **Etude** is its short name and the repository's
machine identity.

Etude Fantasia trains and studies a **manabot**. This repository contains:

- **Etude Fantasia**: the authored play, replay, and study experience
- **Manabot** (`manabot`, Python): the trainable agent, search and learning
  library, Gymnasium wrappers, and experiment tracking
- **managym** (Rust): the deterministic game and search environment, with PyO3
  Python bindings

## Installation

```bash
# Clone the repo
git clone git@github.com:loopflowstudio/etude.git
cd etude

# Install locked Python dependencies
uv sync --python 3.12 --extra dev

# Install the local managym extension
uv run --python 3.12 --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python

# Or install and launch the exact browser matchup in one command
./scripts/play
```

## Training

Manabots are primarily trained on Ubuntu machines in AWS and require W&B credentials.

```bash
uv run manabot train --preset simple
# or: uv run python manabot/model/train.py --preset simple
```

## Simulation

Simulation pulls models from wandb. At small scales this can be done locally on CPU machines.

```bash
uv run manabot sim --preset sim --set sim.hero=attention --set sim.villain=simple
# or: uv run python manabot/sim/sim.py --preset sim --set sim.num_games=10
```

## Testing

```bash
# Rust checks
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cd ..

# Install managym into the uv-managed environment
uv run --python 3.12 --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python

# Python tests (full + integration slice)
uv run --extra dev pytest tests/
uv run --extra dev pytest tests/env/ tests/agent/ -v
```

## Architecture

### Manabot (Python)

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   - `VectorEnv`: vectorized environment backed by `managym.VectorEnv`
   - `ObservationSpace`: Observation space encoding
   - `Match`: Game configuration (decklists, etc.)
   - `Reward`: Reward function

2. **`manabot.model`**: trainable policy and value models
   - `Agent`: Shared value/policy network
   - `Trainer`: PPO trainer

3. **`manabot.sim`**: search, data generation, and game simulation
   - `Player`: Agent implementations (learned or random)
   - `Sim`: Multi-game simulation runner

4. **`manabot.infra`**: training infrastructure
   - `Experiment`: W&B/TensorBoard tracking
   - `Hypers`: Pydantic config model
   - `Profiler`: Performance profiling

### managym (Rust)

1. **`managym/src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`managym/src/flow/`**: Game progression (turns, priority, combat)
3. **`managym/src/state/`**: Core game state (cards, players, zones, mana)
4. **`managym/src/cardsets/`**: Card implementations
5. **`managym/src/infra/`**: Logging and profiler infrastructure
6. **`managym/src/python/`**: PyO3 bindings and Rust→Python conversions

Dependencies flow: python → agent → flow → state/infra

## Style Guide

### Python (Manabot)

```python
"""
filename.py
One-line purpose of file

Instructions for collaborators on how to approach understanding and editing.
"""

# Standard library
import os
from typing import Dict, List

# Third-party imports
from torch import Tensor

# Manabot imports
from manabot.env import ObservationSpace

# Local imports
from .sibling import Thing
```

### Rust (managym)

```rust
// filename.rs
// One-line purpose of file

use crate::flow::game::Game;
use crate::state::player::PlayerId;
```

Prefer explicit types and focused modules. Keep game behavior in enums +
`match` expressions instead of inheritance-like abstractions.

## Naming

Etude Fantasia is the full project and product name; Etude is the short name.
Manabot is not a former name to erase: it is the agent Etude trains and the
Python library that implements its learning and search systems. Existing
`manabot.*` contract/schema identifiers and experiment receipts therefore keep
their names and remain reproducible.

## LLM Collaboration

When working with this codebase:
- Avoid transient comments that denote changes
- Pay attention to file headers and README content
- Propose small, iterative changes
- End responses with full implementations, clarifying questions, and notes on what was left out
