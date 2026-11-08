//! Exact captured subjects and the private selection facts they retain.
//! Bindings travel with their trigger occurrence, including after its selecting
//! frame finishes. Sampling may change an unknown witness, never its constraints.

use crate::{
    game_object::{ObjectRef, PlayerId},
    predicate::CardPredicate,
    zone::ZoneType,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ZoneSelection {
    pub owner: PlayerId,
    pub zone: ZoneType,
    pub predicate: CardPredicate,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectBinding {
    pub object: ObjectRef,
    /// Membership and predicate at capture, not a predicate re-read after movement.
    pub selection: Option<ZoneSelection>,
    /// Seats that knew this exact witness at capture. Definition-only hand
    /// knowledge does not identify a physical copy.
    pub known_to: [bool; 2],
}
