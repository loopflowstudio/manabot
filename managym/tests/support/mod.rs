//! The scenario API shared by the rules and card test binaries, organized by
//! purpose: `game` builds a game (decks, hand, library, battlefield,
//! graveyard), `play` drives it (cast, activate, choose, target, pass and
//! resolve, combat), and `inspect` reads it (zones, permanents, counters,
//! offered actions, observation contracts). Each binary uses a subset, so
//! unused helpers are expected.
#![allow(dead_code)]

pub mod game;
pub mod inspect;
pub mod play;
