# 03: Evaluation Discipline

## Finish line

Comprehensive evaluation pipeline that runs automatically and produces
actionable metrics. Evaluation is separated from training — it never
affects the training loop, only observes.

## Changes

### 1. Evaluation harness

Standalone evaluation script that takes any two policies and plays
N games between them:

```bash
python -m manabot.eval \
  --hero checkpoint_v42.pt \
  --villain random \
  --games 1000 \
  --output eval_results.json
```

Supports all opponent types: checkpoint, random, passive, pool.

### 2. Automated eval during training

Every N updates, pause training (or run async) and evaluate:
- Current policy vs RandomPlayer (100 games)
- Current policy vs best-so-far checkpoint (100 games)
- Current policy vs 3 sampled pool opponents (50 games each)

Results logged to wandb as a structured table.

### 3. Behavioral analysis

Beyond win rate, track per-game:
- Action-type distribution (how often does it play lands, cast, attack, pass?)
- Mean game length
- Life total trajectories
- Creature count over time

These reveal *how* the agent is winning, not just *whether*. An agent
that wins 90% by timeout (never attacking) is very different from one
that wins by turn 6 aggro.

## Done when

- Eval harness runs standalone and during training
- Behavioral metrics show clear strategic profile of the trained agent
- Dashboard in wandb shows eval progression over training
