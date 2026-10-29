//! The card language and the content packs built from it.
//!
//! - [`authoring`]: typed, symbolic card definitions that set files in
//!   `cardsets/` are written in.
//! - [`content_pack`]: the immutable catalog a match runs against.
//! - `native`: the few hand-registered cards with no typed definition yet.

pub mod authoring;
pub mod content_pack;
mod native;
