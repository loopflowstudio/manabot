# 03: Observation Cleanup

## Problem

The agent can't learn effectively because its observations are noisy,
unnormalized, and missing critical game state. Three specific failures:

1. **No phase/step encoding.** The agent doesn't know what phase of the
   turn it's in. It can't learn that "play land" only makes sense in
   main phases, or that "declare attacker" only appears in combat. This
   information exists in `obs.turn.phase` and `obs.turn.step` but is
   never encoded.

2. **Raw integer IDs as features.** `player.id`, `player.player_index`,
   `card.owner_id`, `perm.controller_id` are arbitrary runtime integers
   (e.g., 1, 2). The network treats these as continuous values. They
   don't generalize and inject noise.

3. **No normalization.** Life (0–20), zone counts (0–60), power (0–10),
   damage (0–N) all enter the network at wildly different scales. PPO
   is sensitive to observation scale — this makes value estimation
   unstable.

A secondary issue: padding is 4x larger than needed. 302 objects when
~72 suffice for the test deck. This doesn't block learning but wastes
compute, especially if attention is turned on.

**Wave goal 3**: "Clean the observation space (meaningful, normalized
features)."

## Approach

One shot. All four changes land together because they're interdependent
— normalizing half the features while leaving raw IDs creates a
confusing observation space. No checkpoint compatibility to worry about.

### Feature changes

**Player features** (12 → 26 floats, no validity flag):

| # | Feature | Encoding |
|---|---------|----------|
| 0 | life | `life / 20.0` |
| 1 | is_active | bool (1.0 if this player's turn) |
| 2–8 | zone_counts | `count / 60.0` per zone |
| 9–13 | phase | one-hot (5 values: BEGINNING..ENDING) |
| 14–25 | step | one-hot (12 values: BEGINNING_UNTAP..ENDING_CLEANUP) |

Removed: `player_index` (redundant — perspective is structural),
`player.id` (arbitrary runtime int), `is_agent` (redundant — agent
objects are always in the `agent_*` slots).

Phase and step go in *both* player vectors. Same embedding handles
both, and the redundancy is harmless (17 extra floats).

**Card features** (18 → 18 floats, last = validity):

| # | Feature | Encoding |
|---|---------|----------|
| 0–6 | zone | one-hot (7 zones) |
| 7 | is_mine | 1.0 if in `agent_cards`, 0.0 if in `opponent_cards` |
| 8 | power | `power / 10.0` |
| 9 | toughness | `toughness / 10.0` |
| 10 | mana_value | `mana_value / 10.0` |
| 11–16 | card types | bools: land, creature, artifact, enchantment, planeswalker, battle |
| 17 | is_valid | 1.0 if slot occupied |

Changed: `owner_id` (raw int) → `is_mine` (bool derived from partition,
not from the ID field). Continuous features normalized.

**Permanent features** (5 → 5 floats, last = validity):

| # | Feature | Encoding |
|---|---------|----------|
| 0 | is_mine | 1.0 if in `agent_permanents`, 0.0 if in `opponent_permanents` |
| 1 | tapped | bool |
| 2 | damage | `damage / 10.0` |
| 3 | is_summoning_sick | bool |
| 4 | is_valid | 1.0 if slot occupied |

Changed: `controller_id` (raw int) → `is_mine` (bool).

**Action features** — unchanged (5 type one-hot + validity = 6).

### Padding reduction

| Parameter | Old | New |
|-----------|-----|-----|
| max_cards_per_player | 100 | 20 |
| max_permanents_per_player | 50 | 15 |
| max_actions | 10 | 10 |
| **total objects** | **302** | **72** |

Add truncation warnings for cards/permanents (actions already warn), then
run 100+ games with the test deck and verify none of the three warnings fire.
If they do, bump the limit that fired.

Before finalizing `20/15`, run a Rust-first validation protocol (using raw
`managym` observation lengths from `managym/src/agent/observation.rs` output,
before Python truncation):

1. Log per-step raw counts:
   - `len(obs.agent_cards)`, `len(obs.opponent_cards)`
   - `len(obs.agent_permanents)`, `len(obs.opponent_permanents)`
   - `len(obs.action_space.actions)`
2. Compute overflow flags for candidate caps (`20/15/10`).
3. Run stress scenarios:
   - passive vs passive
   - random vs random
   - first-light training deck/policy setup
   - enough games to observe tails (target: 5k–10k episodes total)
4. Analyze overflow bias:
   - overall overflow rate
   - overflow by turn bucket (early/mid/late)
   - overflow by game outcome (win/loss)
5. A/B check truncation impact:
   - train/eval with `100/50` baseline vs `20/15` candidate (same seeds)
   - compare win rate + learning curve shape

Pass criteria for keeping `20/15`:
- overflow is near-zero (<0.1% of steps) for cards/permanents/actions,
- no strong outcome skew (overflow rates for wins vs losses within 2x),
- no material A/B regression (final win rate delta ≤ 2 percentage points).

### Downstream changes

**Agent model** (`agent.py`):
- `player_embedding`: input dim 12 → 26
- `card_embedding`: unchanged (dim 18)
- `perm_embedding`: unchanged (dim 5)
- Object tensor shape: `(B, 302, H)` → `(B, 72, H)`
- Focus indices range shrinks accordingly

**SingleAgentEnv** (`single_agent_env.py`):
- `_is_opponent()` currently reads `obs["agent_player"][0, 0]` which
  was `player_index`. After removal, this breaks.
- Fix: do **not** add per-step info payload. Keep actor identity as runtime
  state from raw managym observation (`self.inner.last_cpp_obs.agent.player_index`)
  and have `SingleAgentEnv` route turns from that.
- Signature change: `_is_opponent()` and `_skip_opponent()` no longer rely on
  encoded observation feature positions.

**`get_agent_indices()`** (`observation.py`):
- Currently reads `obs["agent_player"][:, 0, 0]` (was `player_index`).
- After cleanup, position 0 is `life / 20.0`. This function is wrong.
- Remove this helper and migrate call sites to raw env observation state
  (`env.last_cpp_obs.agent.player_index`) rather than encoded tensors.

**Simulation fallback parsing** (`sim.py`):
- `_extract_life()` currently reads index 2 (old life slot).
- After cleanup, life is at index 0. Update fallback parsing.

**Tests** — all observation shape assertions, validity mask tests, and
agent forward pass tests need updated expected dimensions.

### Dimension summary

```
             old_dim  new_dim  old_count  new_count  old_total  new_total
player           12       26         2          2         24         52
card             18       18       200         40       3600        720
permanent         5        5       100         30        500        150
action            6        6        10         10         60         60
                                                        4184       982
```

~4x reduction in observation tensor size.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Learned normalization (BatchNorm at input) | Adapts to actual data distribution | Adds training instability early on, hides bad feature design, harder to debug. Manual normalization with known constants is simpler and sufficient for this deck. |
| Step-only encoding (skip phase) | 12 floats instead of 17 extra | Phase gives a coarser signal the agent can learn from first. Redundancy is cheap. Keep both. |
| Incremental rollout (normalize first, add phase later) | Lower risk per change | Changes are interdependent. Half-cleaned observations don't help learning. No checkpoints to protect. Ship it all. |
| Keep `is_agent` in player features | Might help opponent modeling | In SingleAgentEnv, agent_player is always the hero. `is_agent` is a constant — zero information. Remove it. |

## Key decisions

**`is_mine` derived from partition, not from ID fields.** `agent_cards`
always have `is_mine = 1.0`, `opponent_cards` always `0.0`. Don't read
`owner_id` and compare against `agent.id` — that reintroduces raw IDs
into the logic. The partition is the source of truth.

**Phase/step in both player vectors.** Both players share the same
phase/step. Putting it only in one breaks the shared player embedding.
17 redundant floats is nothing compared to the clarity of uniform
dimensions.

**`_is_opponent` reads raw env state, not model features.** The encoded
observation should contain only features useful for the neural network.
Env wrapper routing logic reads from `last_cpp_obs` (Rust-backed managym
observation), avoiding per-step info payload growth.

**Normalization constants are hardcoded, not learned.** `/20.0` for
life, `/60.0` for zone counts, `/10.0` for power/toughness/damage/mana.
These are reasonable for the current deck and slightly generous for
future cards. Exact [0,1] clamping is not important — approximate
normalization is sufficient.

## Scope

- In scope: phase/step encoding, normalization, ID removal, padding
  reduction, all downstream dimension updates, SingleAgentEnv fix,
  test updates
- Out of scope: per-color mana cost encoding (not useful for current
  mono-color decks), additional CardType booleans (6 unused ones exist
  but add noise for this deck), turn number encoding, any changes to
  the managym engine itself

## Done when

```bash
pytest tests/ -v
```

- Phase and step are correctly one-hot encoded in player features
- All continuous features are approximately in [0, 1]
- No raw integer IDs appear in any observation tensor
- Observation object count is 72 (down from 302)
- No action/card/permanent truncation warnings fire during a 100-game validation run
- Agent forward pass works with new dimensions
- `SingleAgentEnv` correctly identifies opponent turns from raw env state
- `get_agent_indices` is fixed or removed
