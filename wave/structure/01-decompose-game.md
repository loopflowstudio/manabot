# 01: Decompose game.rs

## Finish line

`game.rs` is a thin coordinator under 400 lines. Card-specific logic,
combat resolution, mana production, zone movement, SBAs, and spell
resolution each live in their own module. No method in `game.rs` knows
a card name.

## Context

At 14c0743, `game.rs` is 1558 lines containing 50+ methods spanning
every concern: game loop, priority, action dispatch, spell resolution,
combat damage (with first strike, double strike, trample, deathtouch,
lifelink, menace), mana production/spending, zone movement, targeting,
card-specific resolution (`resolve_lightning_bolt`, `resolve_counterspell`,
`is_legal_target_for_bolt`), and SBAs.

This makes the file hard to navigate, hard to review, and creates
implicit coupling between unrelated concerns.

## Changes

### 1. Extract resolution.rs

Move from game.rs:
- `resolve_top_of_stack()`
- `resolve_spell_object()`
- `resolve_lightning_bolt()`
- `resolve_counterspell()`
- `resolve_activated_ability()`
- `counter_spell()`
- `find_spell_on_stack_index()`
- `is_legal_target_for_bolt()`

These become `impl Game` methods in `flow/resolution.rs`. This file
already exists from 965843b (for triggered ability resolution) — extend
it with spell resolution.

### 2. Extract combat.rs

Move from game.rs:
- `resolve_combat_damage()`
- `resolve_combat_damage_pass()`
- `combat_has_first_or_double_strike()`
- `creature_deals_damage_in_pass()`
- `apply_player_damage()`
- `apply_permanent_damage()`
- `source_has_lifelink()`
- `source_has_deathtouch()`
- `gain_life()`
- `eligible_attackers()`
- `eligible_blockers()`
- `blocker_can_block_attacker()`
- `cleanup_illegal_menace_blocks()`
- `declare_attacker()`
- `declare_blocker()`

These become `impl Game` methods in `flow/combat.rs`. The existing
`combat.rs` has only `CombatState` — it gains the methods that operate
on it.

### 3. Extract mana.rs (flow-level)

Move from game.rs:
- `pay_spell_cost()`
- `produce_mana()`
- `spend_mana()`
- `producible_mana()`
- `invalidate_mana_cache()` (if it exists)
- `can_pay_mana_cost()`

These become `impl Game` methods in `flow/mana.rs`. Distinct from
`state/mana.rs` which defines `ManaCost`/`Mana` data types.

### 4. Extract zones.rs (flow-level)

Move from game.rs:
- `move_card()`
- `battlefield_permanents()`

These become `impl Game` methods in `flow/zones.rs`. Distinct from
`state/zone.rs` which defines `ZoneManager`.

### 5. Extract sba.rs

Move from game.rs:
- `perform_state_based_actions()`
- `lose_game()`
- `is_game_over()`
- `winner_index()`

### 6. Extract actions.rs (flow-level)

Move from game.rs:
- `execute_action()`
- `compute_player_actions()`
- `priority_action_for_card()`
- `priority_activate_ability_actions()`
- `pending_choice_action_space()`
- `legal_targets_for_spell()`
- `can_player_act()`

### 7. game.rs remainder

What stays:
- `Game::new()`
- `Game::step()` — entry point from agent
- `Game::play()` — self-play loop
- `Game::tick()` — main loop
- `Game::turn_tick()` — turn progression
- `Game::on_step_start()` / `Game::on_step_end()`
- `Game::perform_turn_based_actions()`
- `Game::tick_priority()` — priority loop (delegates to actions/resolution)
- `Game::drain_events()` / `Game::emit()`
- Helper accessors (`active_player`, `non_active_player`, etc.)
- `clear_mana_pools()`, `clear_damage()`, `clear_temporary_modifiers()`
- `untap_all_permanents()`, `mark_permanents_not_summoning_sick()`
- `draw_cards()`

## Approach

All extractions are `impl Game` blocks in new files. No new traits, no
new structs, no behavior changes. Methods keep the same signatures. The
only changes are `use` imports and visibility (`pub` → `pub(crate)` where
appropriate).

Order: resolution → combat → mana → zones → sba → actions. Each
extraction is a standalone commit that passes all tests.

## Done when

- `game.rs` is under 400 lines
- `cargo test` passes
- `pytest tests/env/` passes
- No method in `game.rs` references a card by name
- No `resolve_lightning_bolt` or `is_legal_target_for_bolt` in game.rs
