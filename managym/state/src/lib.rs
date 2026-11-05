//! The engine's rules vocabulary and game-state types: what a card definition
//! is written in (`card`, `ability`, `effect`, `predicate`, `mana`) and what a
//! match holds (`permanent`, `zone`, `stack_object`, `game_object`). Card sets
//! (`managym-cards`) depend only on this crate, and the engine (`managym`)
//! re-exports it as `managym::state`, adding the players and decks it binds to
//! Python and the canonical match-state hash over its own flow state.

pub mod ability;
pub mod binding;
pub mod card;
pub mod effect;
pub mod game_object;
pub mod mana;
pub mod permanent;
pub mod predicate;
pub mod stack_object;
pub mod target;
pub mod zone;

/// Serde `skip_serializing_if` for counts whose default is omitted.
pub fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}
