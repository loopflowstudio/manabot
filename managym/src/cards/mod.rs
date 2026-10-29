//! The content packs a match runs against.
//!
//! - [`content_pack`]: the immutable catalog a match runs against.
//! - `fixtures`: rules fixtures registered in the default pack that are not
//!   printed cards.
//!
//! Cards themselves are written in `cardsets/`, one file per printed set.

pub mod content_pack;
mod fixtures;
