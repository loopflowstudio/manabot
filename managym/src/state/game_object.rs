use std::collections::BTreeMap;

use super::card::CardDefId;
use super::zone::ZoneType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ObjectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct CardId(pub usize);

/// Stable match-local identity of a physical card or token.
///
/// `CardId` is the current storage authority for physical cards. Keeping the
/// rules identity in a distinct type prevents continuations from silently
/// treating that storage index as the current rules object after a zone
/// change. The adapter can move when the card-instance layout changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct EntityId(pub usize);

impl From<CardId> for EntityId {
    fn from(card: CardId) -> Self {
        Self(card.0)
    }
}

impl From<EntityId> for CardId {
    fn from(entity: EntityId) -> Self {
        Self(entity.0)
    }
}

/// Monotonic rules-object generation for one physical entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct Incarnation(pub u32);

impl Incarnation {
    pub const INITIAL: Self = Self(0);

    pub fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// An exact rules object. A zone change preserves `entity` but advances
/// `incarnation` (CR 400.7), so old continuations cannot bind to a later
/// object represented by the same physical card.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ObjectRef {
    pub entity: EntityId,
    pub incarnation: Incarnation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ObjectLookupError {
    MissingEntity,
    StaleIncarnation,
    WrongZone,
}

/// Battlefield facts retained for a departed exact object.
///
/// This deliberately stores identity and mutable facts, not a cloned card
/// definition. `definition_id` resolves through the match's immutable
/// `ContentPack` and does not clone card semantics into mutable state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectLki {
    pub keywords: crate::state::card::Keywords,
    /// Its creature types as it last existed (CR 608.2h), sorted; read when
    /// a departed target's partners must share one (Secret Tunnel).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub creature_types: Vec<String>,
    /// The battlefield slot it occupied, so a departed `Target::Permanent`
    /// finds its last known information.
    pub permanent: PermanentId,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub animated_creature: bool,
    /// The object had lost all abilities, so it triggers nothing as it leaves.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub abilities_lost: bool,
    /// It had been dealt damage this turn.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Its power as it last existed ("that creature's power").
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub power: i32,
    /// Its counters as it last existed ("put its counters on ...").
    #[serde(skip_serializing_if = "crate::state::permanent::Counters::is_empty")]
    pub counters: crate::state::permanent::Counters,
    pub object_ref: ObjectRef,
    pub card: CardId,
    pub from_zone: ZoneType,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub definition_id: CardDefId,
    pub presentation_id: ObjectId,
}

/// Last-known battlefield facts of departed exact objects, keyed by object,
/// with each entity's latest departed object and each battlefield slot's
/// object at hand: a card leaves and re-enters the battlefield as new
/// objects, and zone-change matching looks back to its most recent one.
/// Iterates in object order; the canonical state hash serializes that.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectLkiLog {
    by_object: BTreeMap<ObjectRef, ObjectLki>,
    /// The latest departed object of each entity, by entity index.
    latest_by_entity: Vec<Option<ObjectRef>>,
    /// The departed object that occupied each battlefield slot, by
    /// `PermanentId`; a slot is never reused.
    by_permanent: Vec<Option<ObjectRef>>,
}

impl ObjectLkiLog {
    pub fn insert(&mut self, lki: ObjectLki) {
        let object = lki.object_ref;
        let entity = object.entity.0;
        if self.latest_by_entity.len() <= entity {
            self.latest_by_entity.resize(entity + 1, None);
        }
        if self.latest_by_entity[entity].is_none_or(|latest| latest < object) {
            self.latest_by_entity[entity] = Some(object);
        }
        let slot = lki.permanent.0;
        if self.by_permanent.len() <= slot {
            self.by_permanent.resize(slot + 1, None);
        }
        self.by_permanent[slot].get_or_insert(object);
        self.by_object.insert(object, lki);
    }

    pub fn get(&self, object: ObjectRef) -> Option<&ObjectLki> {
        self.by_object.get(&object)
    }

    pub fn contains(&self, object: ObjectRef) -> bool {
        self.by_object.contains_key(&object)
    }

    /// The card's most recent departed battlefield object.
    pub fn latest_for_card(&self, card: CardId) -> Option<&ObjectLki> {
        self.latest_by_entity
            .get(card.0)
            .copied()
            .flatten()
            .and_then(|object| self.by_object.get(&object))
    }

    /// The departed object that occupied a battlefield slot.
    pub fn for_permanent(&self, permanent: PermanentId) -> Option<&ObjectLki> {
        self.by_permanent
            .get(permanent.0)
            .copied()
            .flatten()
            .and_then(|object| self.by_object.get(&object))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ObjectRef, &ObjectLki)> {
        self.by_object.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PermanentId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PlayerId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum Target {
    Player(PlayerId),
    Permanent(PermanentId),
    StackSpell(CardId),
    /// An activated or triggered ability on the stack, by its stack object id.
    StackAbility(ObjectId),
    GraveyardCard(ObjectRef),
}

// Typed index wrappers — prevent accidental cross-collection indexing.

#[derive(Clone, Debug, serde::Serialize)]
pub struct CardVec<T>(pub Vec<T>);

impl<T> Default for CardVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> std::ops::Deref for CardVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}
impl<T> std::ops::DerefMut for CardVec<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}
impl<T> std::ops::Index<CardId> for CardVec<T> {
    type Output = T;
    fn index(&self, id: CardId) -> &T {
        &self.0[id.0]
    }
}
impl<T> std::ops::Index<&CardId> for CardVec<T> {
    type Output = T;
    fn index(&self, id: &CardId) -> &T {
        &self.0[id.0]
    }
}
impl<T> std::ops::IndexMut<CardId> for CardVec<T> {
    fn index_mut(&mut self, id: CardId) -> &mut T {
        &mut self.0[id.0]
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PermanentVec<T>(pub Vec<T>);

impl<T> Default for PermanentVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> std::ops::Deref for PermanentVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}
impl<T> std::ops::DerefMut for PermanentVec<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}
impl<T> std::ops::Index<PermanentId> for PermanentVec<T> {
    type Output = T;
    fn index(&self, id: PermanentId) -> &T {
        &self.0[id.0]
    }
}
impl<T> std::ops::Index<&PermanentId> for PermanentVec<T> {
    type Output = T;
    fn index(&self, id: &PermanentId) -> &T {
        &self.0[id.0]
    }
}
impl<T> std::ops::IndexMut<PermanentId> for PermanentVec<T> {
    fn index_mut(&mut self, id: PermanentId) -> &mut T {
        &mut self.0[id.0]
    }
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct IdGenerator {
    next_id: u32,
}

impl IdGenerator {
    pub fn next_id(&mut self) -> ObjectId {
        let out = ObjectId(self.next_id);
        self.next_id += 1;
        out
    }

    pub fn watermark(&self) -> u32 {
        self.next_id
    }
}
