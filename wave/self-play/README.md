# Self-Play

## Vision

Move from beating fixed opponents to training against itself. The agent
should improve through competition — not just exploit a static opponent's
weaknesses.

This is the hardest transition in game AI training. PPO in imperfect
information games can cycle rather than converge. The wave builds the
infrastructure to detect and prevent that.

### Not here

- Game-theoretic equilibrium (NFSP, exploitability) — that's the scale wave
- Complex card pools — keep using simple decks until self-play is stable
- Distributed training

## Goals

1. Agent trained via self-play beats the best fixed-opponent agent from first-light
2. Win rate vs random doesn't regress during self-play training
3. Elo tracking shows monotonic improvement against reference opponents
4. No strategy cycling detected over 50M+ steps

## Risks

- Strategy cycling: agent learns rock, then scissors, then paper, repeat
- Catastrophic forgetting: self-play agent loses ability to beat random
- Computational cost: self-play doubles the inference load per game

## Metrics

- Elo vs reference opponent pool (primary)
- Win rate vs RandomPlayer (must not regress below 60%)
- Win rate vs best first-light checkpoint
- Strategy diversity (entropy of action-type distribution over evaluation games)
- Training reward variance (high variance = cycling signal)
