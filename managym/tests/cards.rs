//! Card behavior tests, one module per printed set in `src/cardsets/`.
//!
//! Engine rules that are not about one card stay in `rules/`. These tests
//! document and protect behavior; they never decide whether a card is
//! supported (the set file defining it does).
#[allow(dead_code)]
#[path = "rules/helpers.rs"]
mod helpers;

#[path = "cards/support.rs"]
mod support;

#[path = "cards/tla.rs"]
mod tla;
