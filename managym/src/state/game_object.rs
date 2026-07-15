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
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize,
)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectLki {
    pub object_ref: ObjectRef,
    pub card: CardId,
    pub from_zone: ZoneType,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub definition_id: CardDefId,
    pub presentation_id: ObjectId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PermanentId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PlayerId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Target {
    Player(PlayerId),
    Permanent(PermanentId),
    StackSpell(CardId),
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
