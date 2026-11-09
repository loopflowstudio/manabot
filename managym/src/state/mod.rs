//! The rules vocabulary and game-state types live in the `managym-state`
//! crate, re-exported here so the engine addresses them as `crate::state::…`.
//! `player` and `deck` stay here with their Python bindings, and `hash`
//! canonicalizes everything with the engine's own flow state.

pub use managym_state::*;

pub mod deck;
pub mod hash;
pub mod player;
