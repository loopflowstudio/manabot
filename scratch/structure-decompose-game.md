# 01: Decompose game.rs

**Status: Complete** (landed in `52ac9ca`)

`game.rs` is 52 lines containing only `GameState` and `Game` struct
definitions. All logic extracted to flow/ modules: action, combat,
combat_actions, damage, event, identity, mana, play, priority,
resolution, sba, setup, tick, trigger, triggers, turn, zones.
