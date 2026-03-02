# 01: Opponent Pool + Frozen Checkpoint Training

## Finish line

Training loop supports playing against frozen past checkpoints sampled
from a pool. Elo tracking shows monotonic improvement.

## Changes

### 1. Checkpoint pool

Periodically save policy checkpoints during training (every N updates).
Maintain a pool of the last K checkpoints. Each training game samples
an opponent from the pool (uniform random, or weighted toward recent).

```python
class OpponentPool:
    def __init__(self, max_size=20):
        self.checkpoints = []

    def add(self, state_dict, step, elo=None): ...
    def sample(self) -> state_dict: ...
```

### 2. Opponent-as-env integration

Extend `SingleAgentEnv` from first-light to accept a policy callable
as the opponent. The opponent policy is loaded from a checkpoint and
runs inference inside the env wrapper (same as PassivePolicy/RandomPolicy
but using a neural network).

### 3. Elo tracking

After each training epoch, evaluate the current policy against every
opponent in the pool (small number of games each). Compute Elo ratings.
Log to wandb.

Reference opponent set (always in the pool, never removed):
- RandomPlayer
- PassivePlayer
- Best first-light checkpoint

## Done when

- Elo vs reference set increases over training
- Win rate vs random doesn't drop below 60%
- Pool cycling works (old checkpoints evicted, new ones added)
