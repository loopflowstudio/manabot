# First Light

## Vision

Get the manabot training platform to the point where an agent demonstrably
learns something — plays lands, casts creatures, attacks, beats a passive
opponent. Not impressive play. Just clear, measurable learning signal where
none existed before.

The current system has never shown clean convergence. This wave fixes the
known bugs, simplifies the architecture to remove confounding complexity,
and builds a verification ladder that proves each layer works before adding
the next.

### Not here

- Self-play training (comes after we can beat random)
- Attention mechanism tuning (turn it off, prove learning without it)
- Rust engine integration (parallel effort, this wave works with existing C++ engine)
- Distributed training / multi-GPU
- Complex card interactions beyond vanilla creatures

## Goals

1. Fix every confirmed PPO bug so the training loop matches CleanRL's
   known-correct implementation
2. Reduce the system to single-agent training against a passive opponent
3. Clean the observation space so features are meaningful and normalized
4. Build a verification ladder: trivial reward → memorization → beat passive → beat random
5. Add auxiliary prediction heads for dense training signal

## Risks

- **Env wrapper complexity.** Wrapping the existing C++ env to auto-step
  the opponent may introduce subtle bugs (observation alignment, done
  signal propagation). Needs careful testing.
- **Observation changes break existing checkpoints.** Any change to the
  observation encoding invalidates saved models. Fine — we don't have
  working models anyway.
- **Passive opponent is too easy.** The agent might learn a degenerate
  policy that only works against passive (e.g., never learns blocking
  because it never needs to). Acceptable — that's what the "beat random"
  step is for.
- **C++ engine bugs masked by training bugs.** Once training is correct,
  we may discover env-side issues. The Rust rewrite will eventually
  replace this, but we need the C++ engine to work for now.

## Metrics

- Win rate vs. passive opponent (target: >90%)
- Win rate vs. random opponent (target: >60%)
- Explained variance (target: >0.5 after 1M steps)
- Entropy decay rate (should decrease steadily, not collapse)
- Mean episode length vs. passive (should decrease as agent learns to attack)
- Per-action-type frequency (should show increasing land/creature/attack rate)
