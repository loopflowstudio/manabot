# 03: Observation Cleanup

Fix the observation space so features are meaningful, normalized, and
don't include noise. The engine already exposes phase/step in TurnData —
we just need to encode it.

## Finish line

Observations include game phase/step, all continuous features are
normalized to ~[0, 1], raw integer IDs are removed, and padding is
reduced to match actual game statistics.

## Changes

### 1. Encode game phase and step

Add to player features (or as a separate observation key):

```python
# One-hot phase (5 values: BEGINNING through ENDING)
phase_onehot = np.zeros(5, dtype=np.float32)
phase_onehot[obs.turn.phase] = 1.0

# One-hot step (12 values: BEGINNING_UNTAP through ENDING_CLEANUP)
step_onehot = np.zeros(12, dtype=np.float32)
step_onehot[obs.turn.step] = 1.0
```

This gives the agent the critical information about what phase of the
turn it's in, which determines which actions make sense.

### 2. Normalize continuous features

```python
# Player features
life = player.life / 20.0
zone_counts = [count / 60.0 for count in player.zone_counts]

# Card features
power = card.power / 10.0
toughness = card.toughness / 10.0
mana_cost = card.mana_cost.mana_value / 10.0

# Permanent features
damage = perm.damage / 10.0
```

### 3. Remove raw integer IDs

Remove from player features:
- `player.id` — arbitrary, doesn't generalize
- `player.player_index` — redundant with perspective mechanism

Remove from card features:
- `card.owner_id` — replace with boolean `is_mine` (1.0 if owner is
  agent, 0.0 if opponent)

Remove from permanent features:
- `perm.controller_id` — replace with boolean `is_mine`

### 4. Reduce padding maximums

Profile actual game statistics with the test deck (Mountains + Grey
Ogres) and set realistic maximums:

```python
max_cards_per_player = 20      # was 100
max_permanents_per_player = 15  # was 50
max_actions = 10               # keep as-is (already reasonable)
```

This reduces the object count from 302 to 72, which is a 4x reduction
in attention compute if attention is ever turned on.

### 5. Update observation dimensions

All downstream code (agent embeddings, shapes, tests) must be updated
to match the new feature dimensions. The `player_dim`, `card_dim`, and
`permanent_dim` will change.

## Constraints

- The encoding order must remain consistent with the object_to_index
  mapping (agent player → opponent player → agent cards → ...)
- Validity masks must still work correctly with reduced padding
- The observation must not leak hidden information (opponent's hand
  contents should not be visible)
- Action-space truncation warning already exists in the observation
  encoder (added in stage 01) and fires when actions exceed
  `max_actions`. The env wrapper also emits `info["action_space_truncated"]`.
  If `max_actions=10` stays, verify this never fires with the test
  deck during profiling (section 4). If it does, the reduced padding
  is too aggressive.

## Done when

```bash
pytest tests/ -v
# - Phase and step are correctly one-hot encoded
# - All continuous features are in [0, 1] range (approximately)
# - No raw integer IDs in observations
# - Observation tensor is smaller (fewer padding slots)
# - Agent forward pass works with new observation shapes
# - Value estimates are in reasonable range with normalized inputs
```
