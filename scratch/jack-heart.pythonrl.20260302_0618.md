# Manabot Architecture Review → First Light Wave

## What this is

Full audit of the manabot PPO training platform, now driving the
`wave/first-light/` plan.

This branch implements **stage 01: Fix PPO Correctness Bugs**.
See `wave/first-light/01-fix-ppo-bugs.md` for the spec.

---

## Part 1: Architecture Overview

### The Stack

```
VectorEnv (AsyncVectorEnv, N parallel games)
  → ObservationEncoder (Python/numpy, per-env)
    → Agent (PyTorch)
      → Typed object embeddings (player, card, permanent, action)
      → GameObjectAttention (multi-head self-attention over all objects)
      → Focus-augmented action representations
      → Policy head (per-action logits)
      → Value head (max-pooled over objects)
    → MultiAgentBuffer (per env × per player PPOBuffers)
      → GAE advantage computation
      → Flattened into single batch
    → PPO clipped update (multiple epochs, minibatches)
```

### What's Good

1. **Typed object embeddings.** Separate projection layers for players,
   cards, permanents, and actions. This is sound — different object types
   have different feature semantics, so separate encoders make sense.

2. **Focus object mechanism.** Actions in MTG target specific game objects.
   The pointer-network-style approach of looking up target object embeddings
   and concatenating them with the action embedding is architecturally correct
   and genuinely creative. Similar to how AlphaStar handles targeted abilities.

3. **Perspective vector.** Adding/subtracting a learned vector based on
   object ownership is a clean way to encode "mine vs. theirs" without
   explicit masking. Lightweight and elegant.

4. **Attention masking verification.** The `verify_attention_masking` test
   that injects noise into masked positions and checks output invariance is
   a great debugging tool. Shows good defensive engineering instinct.

5. **Gradient monitoring.** Per-component gradient norm tracking with
   exploding/vanishing detection. Good infrastructure for debugging.

6. **Action masking.** Correctly masks invalid actions with -1e8 before
   softmax. Applied in both rollout and optimization steps (via stored
   observations). This is correct.

7. **GAE implementation.** The reverse-loop GAE with proper terminal masking
   matches CleanRL's reference implementation.

---

## Part 2: Confirmed Bugs

### Bug 1: Advantage normalization configured but never applied

`hypers.norm_adv = True` exists but is never checked in `_optimize_step`.
Line 335 of train.py:

```python
mb_advantages = advantages[mb_inds]
# Missing: if self.hypers.norm_adv:
#     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
```

**Impact: HIGH.** Per-minibatch advantage normalization is implementation
detail #9 in the "37 Details of PPO" paper. Without it, advantage magnitudes
vary wildly across updates, causing unstable policy gradients. This alone
could explain failure to converge.

### Bug 2: Batch size mismatch with multi-agent buffer

```python
batch_size = hypers.num_envs * hypers.num_steps  # = 16 * 128 = 2048
inds = np.arange(batch_size)                     # Fixed size
# But the flattened multi-agent buffer may have MORE or FEWER transitions
```

The multi-agent buffer stores transitions per (env, player). Since only one
player acts per step, the flattened buffer has ~`num_envs * num_steps`
transitions total (split across two players), but the exact count varies.
Using a fixed `batch_size` for indexing will either miss transitions or
access out-of-bounds indices.

**Impact: HIGH.** Could silently produce incorrect training batches.

### Bug 3: Policy head initialization too large

```python
# Current (agent.py:72):
layer_init(nn.Linear(embed_dim, 1))  # gain=1 (default)

# CleanRL reference:
layer_init(nn.Linear(hidden_size, num_actions), std=0.01)
```

With `gain=1`, the initial policy output logits are large, causing the
initial action distribution to be far from uniform. The agent starts
overconfident, which kills early exploration.

**Impact: MEDIUM.** Makes early training unstable and reduces exploration.

### Bug 4: Categorical(probs=probs) instead of Categorical(logits=logits)

```python
# Current (agent.py:200-202):
probs = torch.softmax(logits, dim=-1)
dist = torch.distributions.Categorical(probs=probs)

# Better:
dist = torch.distributions.Categorical(logits=logits)
```

Computing softmax manually then passing probs is numerically less stable
than letting Categorical handle the log-softmax internally. Can cause NaN
gradients with extreme logits.

**Impact: LOW-MEDIUM.** Usually fine, but can cause NaN in edge cases.

---

## Part 3: Design Problems

### Problem 1: Self-play with shared policy (contradictory gradients)

Both players use the same `Agent`. Both players' transitions feed into the
same PPO update. When player 0 wins:
- Player 0's transitions say "do more of this" (positive advantage)
- Player 1's transitions say "do less of this" (negative advantage)
- But they're the same policy, so the gradients partially cancel

This makes learning very slow at best, and unstable at worst. The average
advantage across the batch is approximately zero.

**What reference systems do:**
- KataGo: trains against self, but uses MCTS to produce supervised targets
  (no gradient conflict)
- AlphaStar: league-based training with frozen opponents
- OpenAI Five: trains both sides but with careful opponent sampling
- PufferLib/CleanRL: train against a fixed opponent first

**Fix:** Train only the hero's transitions. Use a frozen checkpoint (or
random player) as the villain. Graduate to self-play after beating random.

### Problem 2: Extremely sparse reward

```python
win_reward = +100.0
lose_reward = -100.0
# All other steps: 0 (or small cpp_reward)
```

A game of MTG can last hundreds of steps. The value function must learn to
predict the game outcome from the current board state with zero intermediate
signal. This is very hard — the effective credit assignment horizon is the
entire game length.

**What reference systems do:**
- KataGo: trains on score (margin of victory), plus auxiliary targets
  (ownership, score distribution) that provide dense signal
- The "37 Details" paper: Atari clips rewards to {-1, 0, +1} per step
- AlphaStar: shaped rewards for intermediate objectives

**Fix options:**
1. Add life total differential as a shaped reward
2. Add auxiliary prediction targets (predict game length, life delta, etc.)
3. At minimum, predict margin (life differential at game end) not just
   win/loss binary

### Problem 3: No game phase in observation

`PhaseEnum` and `StepEnum` are defined in observation.py but **never
encoded in the observation**. The agent cannot distinguish:
- Main phase (can play lands, cast spells)
- Combat phase (must declare attackers/blockers)
- Beginning/end phases (mostly automatic)

The correct action depends entirely on what phase it is. Without this
information, the agent cannot learn phase-appropriate behavior.

**Fix:** One-hot encode the current phase and step into the player features.

### Problem 4: Raw integer IDs in observations

The observation includes:
- `player.id` — arbitrary integer, changes per game
- `card.owner_id` — raw integer
- `perm.controller_id` — raw integer

The network treats these as continuous values and tries to learn
correlations with them. But they don't generalize across games. This adds
noise to training.

**Fix:** Remove `id`, replace `owner_id`/`controller_id` with a boolean
"is mine" flag.

### Problem 5: Observation features at wildly different scales

| Feature | Typical range |
|---------|--------------|
| player_index | 0-1 |
| life | -20 to 40 |
| is_active | 0-1 |
| zone_counts | 0-60 |
| power | 0-20 |
| toughness | 0-20 |
| mana_cost | 0-15 |
| card type flags | 0-1 |

Features with larger magnitudes dominate gradient signal. The embedding
layers partly compensate, but explicit normalization is more reliable.

**Fix:** Normalize continuous features. life/20, power/10, toughness/10,
mana_cost/10, zone_counts/deck_size.

### Problem 6: Massive padding waste

```python
max_cards_per_player = 100
max_permanents_per_player = 50
```

A typical game state has ~7 cards in hand and ~5 permanents. That means
~90% of slots are zero-padding. The attention mechanism processes all 302
object slots (2 players + 200 cards + 100 permanents) regardless. This
wastes compute and dilutes the attention signal.

**Fix:** Profile actual game statistics and reduce to realistic maximums
(e.g., 20 cards, 15 permanents). Still pad, but much less.

---

## Part 4: Performance Problems

### Perf 1: Python observation encoding

Every observation is encoded in Python with per-card and per-permanent
loops:
```python
for i, cid in enumerate(sorted_ids):
    feat[i] = self._encode_card_features(cards[cid])
```

With 16 envs each producing observations every step, this is thousands of
Python function calls per rollout step.

**Fix:** Move encoding to C++/Rust side. The new Rust engine should return
pre-encoded tensors.

### Perf 2: AsyncVectorEnv with pickle serialization

`gymnasium.vector.AsyncVectorEnv` uses subprocess pipes with pickle
serialization. Every observation dict gets pickled/unpickled per step per
env.

**What PufferLib does:** Shared memory buffers with zero-copy access.
**What Sample Factory does:** POSIX shared memory with lightweight signaling.

**Fix:** For now, consider `SyncVectorEnv` if the C++ engine is fast enough
(sub-millisecond steps don't need async). For production, implement shared
memory.

### Perf 3: Dict observations in the training loop

Every step produces a dict of 14 tensors. These dicts are stored in Python
lists, then concatenated. The PPO update indexes into them with
dict comprehensions per minibatch.

**What CleanRL/PufferLib do:** Pre-allocated flat tensor buffers. No dicts
in the hot loop.

**Fix:** Flatten observations at the env boundary. Store as a single
contiguous tensor. Unflatten inside the policy network.

### Perf 4: Per-env Python loop in buffer routing

```python
for i in range(num_envs):          # Python loop
    pid = int(actor_ids[i].item())  # Tensor → Python
    key = (i, pid)
    single_obs = {k: v[i] for k, v in obs.items()}  # Dict comprehension
    self.buffers[key].store(...)
```

This is 16 iterations of Python dict manipulation per rollout step.

**Fix:** Eliminate per-player buffers entirely. Store all transitions in a
single flat buffer with the actor_id as a column. Filter during advantage
computation if needed.

### Perf 5: Attention over 302 object slots

The attention mechanism processes `2 + 200 + 100 = 302` object slots per
forward pass. With 4 heads and hidden_dim=64, the attention matrix is
302x302 per head per batch element. Most slots are padding.

This is the heaviest part of the forward pass and most of the computation
is wasted on padding.

**Fix:** Reduce max slots (see Problem 6). Consider whether attention is
needed at all for the initial "learn anything" milestone.

---

## Part 5: Lessons from Reference Systems

### CleanRL (the known-correct baseline)

The gold standard PPO implementation. Key differences from manabot:

| Detail | CleanRL | Manabot |
|--------|---------|---------|
| Advantage normalization | Per minibatch | Missing |
| Policy head init | gain=0.01 | gain=1.0 |
| Value head init | gain=1.0 | gain=1.0 |
| Buffer | Pre-allocated tensors | Python lists + cat |
| Batch indexing | Derived from actual data | Hardcoded from config |
| Observation format | Flat tensor | Dict of 14 tensors |
| Categorical | logits= | probs= (less stable) |
| clip_coef | 0.2 (basic) / 0.1 (atari) | 0.1 |

### PufferLib (high-perf RL with simplicity)

Key architectural patterns to adopt:
- **encode/decode/value policy interface** — single encoder pass, two heads
- **Flat observation buffers** in shared memory
- **Multi-agent as batch dimension** — treat each agent as a separate env
  slot rather than maintaining per-agent buffers
- **No custom CUDA kernels needed** — just get the data layout right

### KataGo (what made game AI work)

The most provocative insight: KataGo doesn't use PPO at all. It's
supervised learning on MCTS search targets. Full MCTS is infeasible for
MTG (hidden information, massive branching), but the transferable ideas:

1. **Auxiliary prediction targets** — predict more than just win/loss.
   Dense signal dramatically improves training efficiency.
2. **Score-based value targets** — predict margin, not just binary outcome.
3. **Playout cap randomization** — not directly applicable without MCTS,
   but the principle (spend compute where it generates the best training
   signal) transfers.
4. **Global pooling** — whole-state reasoning without full attention cost.

### OpenSpiel (imperfect information games)

Key patterns for MTG:
- **Fixed global action space + masking** — the proven approach for
  variable action spaces. Manabot already does this correctly.
- **Information state vs. observation** — the agent should only see what
  a human player would see. Verify no hidden info leaks.
- **For PPO in imperfect info games:** PPO does NOT converge to Nash
  equilibrium. It finds a best response to the current opponent.
  Self-play with PPO can cycle rather than converge. Opponent pool
  sampling is essential for stability.
- **Consider NFSP** as a future alternative — theoretically grounded
  for imperfect information games.

### Sample Factory (async throughput)

Less relevant for our case (env is fast C++/Rust), but the shared
memory and zero-copy patterns are worth understanding for production
scaling.

---

## Part 6: The Verification Ladder

Before redesigning anything, we need a debugging methodology. Based on RL
best practices, the correct order is:

### Step 0: Random baseline
Run 1000 games with `RandomPlayer` vs `RandomPlayer`. Verify ~50% win
rate. This validates environment symmetry.

### Step 1: Trivial reward
Set `reward.trivial=True` (reward=1.0 every step). Train. The value
function should learn to predict `1 / (1 - gamma)` = 100. Explained
variance should approach 1.0 within a few hundred updates. If it doesn't,
there's a bug in the training loop independent of the game.

### Step 2: Memorization test
`num_envs=1`, fixed seed, tiny deck (20 Mountains). The game plays out
almost identically each time. The agent should learn to win consistently
because it can memorize the optimal play.

### Step 3: Beat random
Train against `RandomPlayer` with real reward. Win rate should exceed 60%
and trend upward.

### Step 4: Self-play
Only after step 3 succeeds.

---

## Part 7: Prioritized Fix List

Ordered by expected impact on convergence, from highest to lowest:

### Tier 1: Must fix before any training makes sense

1. **Add advantage normalization** — the `norm_adv` flag exists, just
   wire it up. ~3 lines of code.
2. **Fix batch size mismatch** — compute actual batch size from flattened
   buffer, not from config.
3. **Train against fixed opponent first** — use `RandomPlayer` as villain.
   Only train hero's transitions.

### Tier 2: Likely needed for convergence

4. **Add game phase/step to observation** — one-hot encode, add to player
   features.
5. **Fix policy head initialization** — gain=0.01 on the final linear layer.
6. **Normalize observation features** — divide by reasonable maximums.
7. **Remove raw integer IDs** — replace with relational features.

### Tier 3: Improves training quality

8. **Add reward shaping** — life differential, board presence, damage dealt.
9. **Use Categorical(logits=) not Categorical(probs=)** — numerical stability.
10. **Log normalized entropy** — entropy / log(num_valid_actions).
11. **Add explained variance tracking** — already computed but verify it's
    correct with the multi-agent buffer.
12. **Reduce observation padding** — profile actual game sizes, shrink maxes.

### Tier 4: Performance (do after correctness)

13. **Flatten observation buffers** — single tensor, not dict of lists.
14. **Pre-allocate rollout buffer** — fixed-size tensors, not append.
15. **Eliminate per-env Python loop in buffer routing** — vectorize.
16. **Move observation encoding to Rust** — will come with engine rewrite.
17. **Evaluate attention necessity** — try without for initial milestone.

---

## Part 8: Minimal Redesign Target

> "How do we get the smallest working thing that demonstrably learns
> something, like e.g. always attack?"

### The simplest possible setup

- **Opponent:** `RandomPlayer` (frozen, not self-play)
- **Deck:** Tiny — Mountains + Grey Ogres (vanilla 2/2 creatures)
- **Attention:** OFF (just embeddings → policy/value heads)
- **Reward:** +1 for win, -1 for loss (simpler than +100/-100)
- **Observation:** Minimal — current life totals, creature counts on
  battlefield, current phase, number of valid actions
- **Action space:** The existing masked action space (already correct)

### What "learning something" looks like

1. Agent learns to play lands (better than random)
2. Agent learns to cast creatures (better than random)
3. Agent learns to attack with creatures (better than random)
4. Win rate vs. random exceeds 60%

### What to measure

- Win rate vs. RandomPlayer (the north star)
- Explained variance (is the value function learning?)
- Entropy over time (is the policy specializing?)
- Mean episode length (shorter = more decisive play)
- Per-action-type frequency (is it learning to prefer certain actions?)

---

## Open Questions

1. **How fast is the current C++ env per step?** Need to know if
   vectorization overhead matters or if sync stepping is fine.
2. **What does the Rust engine rewrite change about the observation
   interface?** Should we design the new obs encoding for the Rust API?
3. **Is the multi-agent buffer approach worth keeping at all?** PufferLib's
   "agents as batch slots" approach is simpler and more standard.
4. **Should we consider MCTS + supervised learning instead of PPO?** KataGo's
   approach is much more stable, but requires solving the hidden-information
   MCTS problem (Information Set MCTS / ISMCTS exists but is less proven).
