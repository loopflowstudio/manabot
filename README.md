# manabot

A reinforcement learning framework for [Magic: The Gathering](https://magic.wizards.com/), using PPO as the core training algorithm.

This repository contains both:
- **manabot** (Python): RL training framework, Gymnasium environment wrapper, experiment tracking
- **managym** (C++): Game engine with pybind11 Python bindings

## Installation

```bash
# Clone the repo
git clone git@github.com:loopflowstudio/manabot.git
cd manabot

# Install managym (C++ module)
pip install -e managym

# Install manabot
pip install -e .
```

## Training

Manabot is primarily trained on Ubuntu machines in AWS and requires wandb credentials.

```bash
# Run training
python manabot/ppo/train.py --config-name simple
```

## Simulation

Simulation pulls models from wandb. At small scales this can be done locally on CPU machines.

```bash
python sim/sim.py --hero attention --villain simple
```

## Testing

```bash
# Python tests
pytest tests/

# C++ tests
mkdir -p build && cd build
cmake ..
make run_tests

# Run specific C++ tests
./managym_test --gtest_filter=TestRegex.* --log=priority,turn,test

# Rust engine tests (stage-01 migration path)
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Architecture

### manabot (Python)

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   - `VectorEnv`: AsyncVectorEnv-based interface
   - `ObservationSpace`: Observation space encoding
   - `Match`: Game configuration (decklists, etc.)
   - `Reward`: Reward function

2. **`manabot.ppo`**: PPO implementation
   - `Agent`: Shared value/policy network
   - `Trainer`: PPO trainer

3. **`manabot.sim`**: Game simulation
   - `Player`: Agent implementations (learned or random)
   - `Sim`: Multi-game simulation runner

4. **`manabot.infra`**: Infrastructure
   - `Experiment`: W&B/TensorBoard tracking
   - `Hypers`: Hydra config management
   - `Profiler`: Performance profiling

### managym (C++)

1. **`managym/agent/`**: RL agent API and pybind11 bindings
2. **`managym/flow/`**: Game state progression (turns, priority, combat)
3. **`managym/state/`**: Game state (cards, players, zones)
4. **`managym/cardsets/`**: Card implementations
5. **`managym/infra/`**: Logging and profiling

Dependencies flow: agent → flow → state → infra

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

### C++ (managym)

```cpp
// filename.h/.cpp
// One-line purpose of file
//
// EDITING INSTRUCTIONS:
// Instructions for collaborators on how to approach editing.

#include "me.h"           // Corresponding header
#include "sibling.h"      // Same directory
#include "managym/other.h" // Other managym headers
#include <3rdparty.h>     // Third party
#include <std>            // Standard library
```

Objects use `struct` with single ownership via `std::unique_ptr<T>`. Other references are raw pointers.

### DO NOT USE in C++

- std::shared_ptr, templates/metaprogramming, macros, manual memory management
- dynamic_cast, multiple inheritance, auto typing (strongly avoid)

## LLM Collaboration

When working with this codebase:
- Avoid transient comments that denote changes
- Pay attention to file headers and README content
- Propose small, iterative changes
- End responses with full implementations, clarifying questions, and notes on what was left out
