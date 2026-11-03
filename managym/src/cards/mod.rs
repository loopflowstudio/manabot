//! What a running engine knows about its cards beyond their rules: the one
//! catalog every match runs against, which cards are supported, and how the
//! policy reads a card.
//!
//! Card rules are the engine's own [`CardDefinition`](crate::state::card::CardDefinition)
//! values written in `cardsets/`, one file per printed set; nothing here
//! interprets them beyond the catalog's one normalization (ward's trigger).
//! - [`content_pack`]: the immutable catalog every match runs against.
//! - [`setups`]: the authored decklists over that catalog.
//! - [`pool`]: the world pool's descriptor rows with support (eligibility).
//! - [`deck_admission`]: development construction admission of a submitted deck.
//! - [`decklist`]: decklist files, name resolution and import.
//! - [`tokens`]: learning tokens derived from the definitions themselves.
//! - `fixtures`: rules fixtures registered in the catalog that are not
//!   printed cards.

pub mod content_pack;
pub mod deck_admission;
pub mod decklist;
mod fixtures;
mod pool;
pub mod setups;
pub mod tokens;

pub use pool::{content_pool, world_pool, ContentPool, Eligibility, PoolDefinition};

/// A content catalog request that cannot be satisfied: an unknown pool,
/// setup or card, or a definition the learning tokens cannot encode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentError(pub String);

impl std::fmt::Display for ContentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContentError {}
