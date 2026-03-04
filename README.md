# manabot

A reinforcement learning framework for [Magic: The Gathering](https://magic.wizards.com/), using PPO as the core training algorithm.

This repository contains both:
- **manabot** (Python): RL training framework, Gymnasium environment wrapper, experiment tracking
- **managym** (Rust): Game engine with PyO3 Python bindings

## Installation

```bash
# Clone the repo
git clone git@github.com:loopflowstudio/manabot.git
cd manabot

# Create a local environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install managym (Rust extension)
pip install -e managym

# Install manabot
pip install -e ".[dev]"
```

## Training

Manabot is primarily trained on Ubuntu machines in AWS and requires wandb credentials.

```bash
manabot train --preset simple
# or: python manabot/model/train.py --preset simple
```

## Simulation

Simulation pulls models from wandb. At small scales this can be done locally on CPU machines.

```bash
manabot sim --preset sim --set sim.hero=attention --set sim.villain=simple
# or: python manabot/sim/sim.py --preset sim --set sim.num_games=10
```

## Testing

```bash
# Rust checks
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cd ..

# Install managym into the active venv
pip install -e managym

# Python tests (full + integration slice)
pytest tests/
pytest tests/env/ tests/agent/ -v
```

## Architecture

### manabot (Python)

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   - `VectorEnv`: AsyncVectorEnv-based interface
   - `ObservationSpace`: Observation space encoding
   - `Match`: Game configuration (decklists, etc.)
   - `Reward`: Reward function

2. **`manabot.model`**: PPO implementation
   - `Agent`: Shared value/policy network
   - `Trainer`: PPO trainer

3. **`manabot.sim`**: Game simulation
   - `Player`: Agent implementations (learned or random)
   - `Sim`: Multi-game simulation runner

4. **`manabot.infra`**: Infrastructure
   - `Experiment`: W&B/TensorBoard tracking
   - `Hypers`: dataclass config model
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

### Python (manabot)

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

# manabot imports
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

## LLM Collaboration

When working with this codebase:
- Avoid transient comments that denote changes
- Pay attention to file headers and README content
- Propose small, iterative changes
- End responses with full implementations, clarifying questions, and notes on what was left out
