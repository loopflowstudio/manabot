# manabot

A reinforcement learning framework for [Magic: The Gathering](https://magic.wizards.com/), built on top of [managym](https://github.com/jacklionheart/managym), using PPO as the core training algorithm.

### Training

Manabot is currently trained primarily on Ubuntu machines in aws and requires wandb credentials.

```bash
# Clone the repo
git clone git@github.com:jacklionheart/manabot.git
cd manabot
# Update python for AWS Ubuntu deep learning AMIs
ops/machine.sh
# Install managym
pip install -e managym
# Install other dependencies
pip install -e .
# Run training
python manabot/ppo/train.py --config-name simple
```

### Simulation

Simulation pulls models from wandb and so similarly requires wandb credentials, but at small scales
can easily be done locally on CPU machines (though the model inference will dominate the env time).

```bash
# Assumes manabot and managym installed as above
python sim/sim.py --hero attention --villain simple
```

### Testing

```bash
pytest tests/
```

### Architecture

manabot is organized into these major components:

1. **`manabot.env`**: 
   - `VectorEnv`: gymnasium.AsyncVectorEnv-based interface around managym
   - `ObservationSpace`: dataclass describing the observation space
   - `Match`: dataclass describing the game of magic to be played (decklists, etc.)
   - `Reward`: dataclass describing the reward function

2. **`manabot.ppo`**: PPO implementation for training model  
   - `Agent`: Shared Value/Policy network
   - `Trainer`: PPO trainer for learning network weights

3. **`manabot.sim`**: Simulate games of Magic: the Gathering using trained models
   -- `Player`: An agent for playing Magic (either from a learned model, or random/trivial implementations)
   -- `Sim`: A simulation of many games of magic of two specific players against each other

3. **`manabot.infra`**: Software infrastructure
   - `Experiment`: Experiment tracking with wandb/tensorboard
   - `Hypers`: Hydra-compatible hyperparameter management and configuration
   - `Profiler`: Performance profiler
   - `log.py`: Unified logging management .data`.

### Some Major Design Decisions

- **managym vs. manabot Split**  
  - **managym** (C++): Handles low-level game logic. Eventually: add native-Cpp inference for faster rollouts
  - **manabot** (Python): Torch-backed optimization, and, for now, inference for rollouts.

- **Dynamic Discrete Action Space**  
  - Limited number of action slots whose meaning can vary significantly from step to step.  
  - The “meaning” of each action slot is included in the observation tensor for context.

## Style Guide

### Code Organization

Files should follow this template:

```python

"""
filename.py
One-line purpose of file

Instructions for collaborators (both human and LLM) on how to approach understanding and editing the code.
Keep this section focused and impactful.
"""

# Standard library
import os
from typing import Dict, List

# Third-party imports
from import torch import Tensor

# manabot imports
from manabot.env import ObservationSpace

# Local imports
from .sibling import Thing
```

### Documentation Principles

- Public APIs should have docstrings focused on clarifying behavior and resolving ambiguities
- Implementation details and design rationale belong in comments or READMEs
- Use organizational comments liberally:

```python
# -----------------------------------------------------------------------------
# Organizational Header
# -----------------------------------------------------------------------------
```

## Code Style

- Follow PEP8
- Use type hints as much as possible
- Use dataclasses where appropriate

When working with this codebase, LLMs should:
- Avoid comments that are transient/denote changes. Imagine the code will be directly copied into the codebase for eternity.
- Pay special attention to file headers and README content for context
- Propose small, iterative changes
- End responses with:
  1. Full implementations of changed files that can be copied into the codebase 
  2. Questions that could clarify intent
  3. Notes on what was intentionally left out
