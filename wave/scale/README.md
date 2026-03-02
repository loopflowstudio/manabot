# Scale

## Vision

Remove throughput bottlenecks and add game-theoretic rigor. Two
concerns that converge at scale: you need fast training to run enough
games for equilibrium-seeking algorithms, and you need equilibrium-seeking
algorithms to make those games count.

### Not here

- New game engines or card pool expansion
- Fundamental algorithm changes (still PPO-based, with extensions)

## Goals

1. 10x throughput over first-light baseline
2. Multi-GPU training works
3. NFSP or policy-averaging prevents self-play cycling at scale
4. Exploitability measurement gives a ground-truth quality signal

## Risks

- Distributed training introduces subtle synchronization bugs
- NFSP adds significant implementation complexity
- Exploitability computation may be intractable for full MTG
  (approximations needed)

## Metrics

- Training throughput (SPS) — target 10x first-light
- GPU utilization during training
- Exploitability estimate (lower = closer to equilibrium)
- Wall-clock time to reach target Elo
