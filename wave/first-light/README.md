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
- Rust engine work (now on main; this wave builds on the Rust engine but doesn't modify it)
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

- **Env wrapper complexity.** SingleAgentEnv auto-steps the opponent,
  which can introduce subtle bugs (observation alignment, done signal
  propagation). The wrapper exists and has tests, but edge cases may
  surface during training.
- **Observation changes break existing checkpoints.** Any change to the
  observation encoding invalidates saved models. Fine — we don't have
  working models anyway.
- **Passive opponent is too easy.** The agent might learn a degenerate
  policy that only works against passive (e.g., never learns blocking
  because it never needs to). Acceptable — that's what the "beat random"
  step is for.
- **Rust engine behavior differences.** The Rust engine is on main but
  may produce different game states than expected (card ordering, timing,
  edge cases). Training will surface these if they exist.

## Metrics

- Win rate vs. passive opponent (target: >90%)
- Win rate vs. random opponent (target: >60%)
- Explained variance (target: >0.5 after 1M steps)
- Entropy decay rate (should decrease steadily, not collapse)
- Mean episode length vs. passive (should decrease as agent learns to attack)
- Per-action-type frequency (should show increasing land/creature/attack rate)

## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [37 PPO implementation details (ICLR blog)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/)
- [PufferLib docs](https://puffer.ai/docs.html)
- [Sample Factory architecture](https://www.samplefactory.dev/06-architecture/overview/)
- [KataGo paper](https://arxiv.org/abs/1902.10565)
- [KataGo methods doc](https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md)
- [OpenSpiel](https://github.com/google-deepmind/open_spiel)
- [NFSP paper](https://arxiv.org/abs/1603.01121)
- [MAPPO paper](https://arxiv.org/abs/2103.01955)
- [OpenAI Five](https://arxiv.org/abs/1912.06680)
- [IMPALA](https://arxiv.org/abs/1802.01561)
