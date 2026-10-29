//! The card language and the content packs built from it.
//!
//! - [`authoring`]: typed, symbolic card definitions that set files in
//!   `cardsets/` are written in.
//! - [`content_pack`]: the immutable catalog a match runs against.
//! - `fixtures`: rules fixtures registered in the default pack that are not
//!   printed cards.

pub mod authoring;
pub mod content_pack;
mod fixtures;
