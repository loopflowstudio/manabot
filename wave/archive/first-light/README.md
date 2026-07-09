# First Light

> **Status: closed. Superseded by [wave/search](../search/README.md).**
>
> Goal 5 (auxiliary prediction heads) is superseded, not abandoned. Aux heads
> were the strategy-neutral fix for credit assignment; lookahead is a better
> one. They remain the fallback if the search wave's goal-4 gate fails.
>
> **Read the findings below with a large caveat.** Every one of them was
> derived on the deck at `manabot/infra/hypers.py:13` — `Mountain 12, Forest 12,
> Llanowar Elves 18, Grey Ogre 18`. No instants, no removal, no counterspells,
> no card draw, no mass removal. That is a vanilla creature-combat game, not
> Magic: the only legal decisions are play-land, play-creature, attack, block.
> The engine was never the limitation (`cardsets/alpha.rs:115` implements
> Lightning Bolt; `:127` Counterspell; `agent/action.rs:8` carries
> `DeclareBlocker` and `ChooseTarget`) — the deck was.
>
> **Update 2026-07-09: findings 1 and 2 are refuted, not merely caveated.**
> Terminal-only reward fails replication as a failure: on this wave's own
> STANDARD_DECK with current code it learns cleanly (60.8–66.3% seat-balanced
> vs random, 3/3 seeds, no pass-collapse), and on the interactive deck it
> *beats* the shaped recipe by ~15 points while the shaping produced
> cast-everything policies, one net-harmful. The original pass-collapse
> observation (unsourced, pre-bugfix code, hero-on-play single-seed evals)
> appears to have been an artifact. See `reports/exp-04-potential-shaping.md`
> addenda.

## Vision

Get the manabot training platform to the point where an agent demonstrably
learns something — plays lands, casts creatures, attacks, beats a random
opponent. Not impressive play. Just clear, measurable learning signal where
none existed before.

### Not here

- Self-play training (comes after we can beat random)
- Attention mechanism tuning (turn it off, prove learning without it)
- Rust engine work (this wave builds on the Rust engine but doesn't modify it)
- Distributed training / multi-GPU
- Complex card interactions beyond vanilla creatures

## Strategy

The first-light wave has established that:

1. **Pure terminal reward fails.** PPO cannot assign credit to "play land on
   turn 1" over ~200 steps of sparse +1/-1 signal. Training with terminal-only
   reward produces pass-collapse — the agent learns to stop playing lands,
   which breaks the entire gameplay chain.

2. **Reward shaping is required.** Intermediate rewards for land play, creature
   play, and opponent life loss are now the baseline recipe. This is not a
   preference — it is what the current system requires to learn.

3. **Stochastic eval reveals policy dynamics.** Deterministic (argmax) eval
   was hiding everything — an untrained policy's tiny logit differences meant
   argmax always picked pass. Stochastic eval shows the real learning signal.

4. **The causal chain is the diagnostic surface.** `landed_when_able`,
   `cast_when_able`, and pass-vs-land choice distributions tell you whether
   learning is real. Win rate alone is too noisy at small eval sizes.

Findings 1 and 2 are scoped to the vanilla-creature deck (see Status, above) and
carry no weight on a deck with interaction in it. Findings 3 and 4 are about the
measurement apparatus and survive the deck change intact — stochastic eval and
the causal-chain metrics remain the right instruments.

## Goals

1. ~~Fix PPO bugs~~ done
2. ~~Single-agent training against passive/random~~ done
3. ~~Clean observation space~~ done
4. ~~Verification harness~~ done
5. ~~Add auxiliary prediction heads for dense training signal~~ superseded by
   [wave/search](../search/README.md) goals 3-5: search generates the dense
   signal aux heads were meant to approximate, and it does so without injecting
   a strategic prior. Retained as the fallback if search fails its goal-4 gate.

## Risks

- **Reward shaping distortion.** Current shaping values (land=0.03,
  creature=0.06, life_loss=0.01) are hand-tuned. They could incentivize
  degenerate strategies (e.g., prioritize land play over attacking). The
  harness's causal-chain metrics should detect this.
- **Auxiliary heads might not help enough.** Dense gradient signal from aux
  heads may improve encoder quality without fixing the core credit assignment
  problem for the policy head. If so, curriculum or entropy tuning are
  fallback directions.
- **Eval noise at small game counts.** 50-game evals swing 0%-84% win rate.
  The harness uses 200-game evals with confidence intervals, but stochastic
  policy + stochastic opponent means high variance remains.

## Metrics

- Win rate vs. random opponent (target: >60%, improved over untrained 72% baseline)
- Win rate vs. passive opponent (target: >90%)
- Explained variance (target: >0.5 after 1M steps)
- `landed_when_able` (target: >= untrained baseline ~44%)
- `cast_when_able` (target: maintain or improve)
- `pass_land_pass_rate` (target: decreasing — agent should prefer land over pass)

## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [37 PPO implementation details (ICLR blog)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/)
- [KataGo paper](https://arxiv.org/abs/1902.10565)
- [KataGo methods doc](https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md)
