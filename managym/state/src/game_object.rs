//! Runtime objects, checked card ids, and exact identities across zone changes.
//!
//! `CardVec` owns the immutable physical-card roster; `ObjectVec` owns runtime
//! rows. `ObjectId` names one appearance. `PublicId` joins observations and
//! orders timestamps.

use std::{collections::BTreeMap, num::NonZeroU32, sync::Arc};

use super::zone::ZoneType;
use super::{
    ability::{Effect, TargetRequirement, TargetSpec},
    card::{CardDefId, Keyword, PrintedCharacteristics},
    mana::ManaCost,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PublicId(pub u32);

/// One zone appearance of a card, token or copy (CR 109, 400.7). It goes
/// stale when that object changes zones; only the zone move's returned id
/// names the successor.
///
/// `slot` is private storage, shared by a card's successive appearances and
/// never reused by another card or token. It orders ids as creation did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId {
    slot: u32,
    generation: u32,
}

const _: () = assert!(std::mem::size_of::<ObjectId>() == 8);

impl ObjectId {
    /// Storage only: the object store and zone manager name the appearance
    /// they hold. Rules code takes ids from the game, never from numbers.
    #[doc(hidden)]
    #[inline]
    pub(crate) fn in_slot(slot: usize, generation: u32) -> Self {
        Self {
            slot: u32::try_from(slot).expect("object store exceeds u32"),
            generation,
        }
    }

    /// Storage row, for the stores, the state hash and viewer captures.
    #[doc(hidden)]
    #[inline]
    pub fn slot(self) -> usize {
        self.slot as usize
    }

    /// How many zone changes preceded this appearance in its slot.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// Recorded bytes name an appearance by its slot alone wherever a bare row
/// was recorded before the identity split; `exact` writes both halves.
impl serde::Serialize for ObjectId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.slot)
    }
}

/// Values whose object ids are recorded whole, as `{"entity","incarnation"}`.
pub trait ExactIds {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

pub(crate) struct Exact<'a, T: ?Sized>(pub(crate) &'a T);

impl<T: ExactIds + ?Sized> serde::Serialize for Exact<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize_exact(serializer)
    }
}

/// `#[serde(serialize_with = "exact")]` for a field holding whole object ids.
pub fn exact<T: ExactIds + ?Sized, S: serde::Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    value.serialize_exact(serializer)
}

impl ExactIds for ObjectId {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut id = serializer.serialize_struct("ObjectId", 2)?;
        id.serialize_field("entity", &self.slot)?;
        id.serialize_field("incarnation", &self.generation)?;
        id.end()
    }
}

impl<T: ExactIds> ExactIds for Option<T> {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Some(value) => serializer.serialize_some(&Exact(value)),
            None => serializer.serialize_none(),
        }
    }
}

impl<T: ExactIds> ExactIds for [T] {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter().map(Exact))
    }
}

impl<T: ExactIds> ExactIds for Vec<T> {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize_exact(serializer)
    }
}

impl<T: ExactIds, U: serde::Serialize> ExactIds for (T, U) {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&(Exact(&self.0), &self.1), serializer)
    }
}

impl<T: ExactIds, U: serde::Serialize, V: serde::Serialize> ExactIds for (T, U, V) {
    fn serialize_exact<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&(Exact(&self.0), &self.1, &self.2), serializer)
    }
}

/// One source identity and whether its creator captured an exact reference.
/// Presence remains recorded even for spell and graveyard sources; it is not
/// inferred from the source's zone at serialization time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbilitySource {
    object: ObjectId,
    captured: bool,
}
impl AbilitySource {
    pub fn new(object: ObjectId, reference: Option<ObjectId>) -> Self {
        assert!(
            reference.is_none_or(|id| id.slot() == object.slot()),
            "source slots differ"
        );
        Self {
            object: reference.unwrap_or(object),
            captured: reference.is_some(),
        }
    }
    pub fn object(self) -> ObjectId {
        self.object
    }
    pub fn reference(self) -> Option<ObjectId> {
        self.captured.then_some(self.object)
    }
    pub fn optional(object: Option<ObjectId>, reference: Option<ObjectId>) -> Option<Self> {
        assert!(
            object.is_some() || reference.is_none(),
            "reference requires a source"
        );
        object.map(|object| Self::new(object, reference))
    }
}
impl serde::Serialize for AbilitySource {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("source_card", &self.object)?;
        map.serialize_entry("source_ref", &Exact(&self.reference()))?;
        map.end()
    }
}
/// V16 frame projection of the same source identity; an absent source emits
/// both null fields rather than omitting them.
pub fn serialize_frame_source<S: serde::Serializer>(
    source: &Option<AbilitySource>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(2))?;
    map.serialize_entry("source", &source.map(AbilitySource::object))?;
    map.serialize_entry(
        "source_ref",
        &Exact(&source.and_then(AbilitySource::reference)),
    )?;
    map.end()
}

/// A physical-card roster index (CR 108.2), independent of appearance ids.
///
/// Internally one-based so an optional card association is also four bytes.
/// Serialization retains the zero-based number in saved decisions and rosters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CardId(NonZeroU32);

impl CardId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let one_based = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("physical-card roster exceeds u32");
        Self(one_based)
    }

    /// Zero-based roster position, which is also the card's storage slot.
    #[inline]
    pub(crate) fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl serde::Serialize for CardId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.0.get() - 1)
    }
}

/// A physical Magic card (CR 108), retained independently of its appearances.
/// The immutable roster is built at setup, including outside-game cards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Card {
    pub owner: PlayerId,
    pub definition: CardDefId,
    pub public_id: PublicId,
}

impl Card {
    /// The physical card a freshly instantiated card object stands for.
    pub fn of(object: &Object) -> Self {
        Self {
            owner: object.owner,
            definition: object.definition_id,
            public_id: object.id,
        }
    }
}

/// Flat physical-card roster, frozen after setup and shared by every clone
/// of the game (search clones it per node). Tokens and copies belong solely
/// to the object store.
#[derive(Clone, Debug, Default)]
pub struct CardVec(Arc<[Card]>);

impl CardVec {
    pub fn new(cards: Vec<Card>) -> Self {
        Self(cards.into())
    }

    /// A private world's physical roster follows its assigned printings.
    pub fn assign_printing(&mut self, card: CardId, definition: CardDefId) {
        Arc::make_mut(&mut self.0)[card.index()].definition = definition;
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Index<CardId> for CardVec {
    type Output = Card;

    #[inline]
    fn index(&self, card: CardId) -> &Card {
        &self.0[card.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ObjectLookupError {
    /// No object was ever stored there.
    Missing,
    /// The object changed zones or ceased to exist (CR 400.7).
    Stale,
    WrongZone,
}

/// The values copied from one exact appearance, before new exceptions apply.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CopiableValues {
    pub definition_id: Option<CardDefId>,
    pub definition: crate::card::ObjectDefinition,
    /// A transforming source's raw faces; a token copy of it is a
    /// transforming token (CR 707.8a).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faces: Option<TransformingFaces>,
    pub copy: Option<crate::effect::CopyEffect>,
}

/// Effective characteristics and permanent status used by battlefield predicates.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BattlefieldFacts {
    pub types: crate::card::CardTypes,
    pub subtypes: Vec<String>,
    pub supertypes: Vec<String>,
    pub colors: crate::mana::Colors,
    pub name: String,
    pub mana_value: u8,
    pub is_legendary: bool,
    pub token: bool,
    pub attacking: bool,
    pub in_combat: bool,
    pub tapped: bool,
}

/// Battlefield facts retained for a departed exact object.
///
/// Ordinary copiable values resolve through the immutable catalog. A departure
/// with a distinct form retains its shared definition and copy exceptions.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectLki {
    /// Ordinary appearances use their immutable catalog row directly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copiable: Option<Arc<CopiableValues>>,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub transforms: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<crate::card::CardDefId>,
    /// Effective grants retained at the event boundary, including after ability loss.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub granted_abilities: Box<[crate::ability::EffectiveTrigger]>,
    pub keywords: crate::card::Keywords,
    /// Its creature types as it last existed (CR 608.2h), sorted; read when
    /// a departed target's partners must share one (Secret Tunnel).
    #[serde(skip_serializing_if = "crate::subtype::CreatureTypes::is_empty")]
    pub creature_types: crate::subtype::CreatureTypes,
    /// The battlefield slot it occupied, so a departed `Target::Permanent`
    /// finds its last known information.
    pub permanent: PermanentId,
    pub battlefield: Arc<BattlefieldFacts>,
    #[serde(serialize_with = "exact")]
    pub attached_to: Option<ObjectId>,
    /// Second component's slot appearance immediately before the meld separates.
    #[serde(skip_serializing_if = "Option::is_none", serialize_with = "exact")]
    pub meld_partner: Option<ObjectId>,
    /// The object had lost all abilities, so it triggers nothing as it leaves.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub abilities_lost: bool,
    /// It had been dealt damage this turn.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Its power as it last existed ("that creature's power").
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub power: i32,
    /// Its toughness as it last existed ("its toughness").
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub toughness: i32,
    /// Its counters as it last existed ("put its counters on ...").
    #[serde(skip_serializing_if = "crate::permanent::Counters::is_empty")]
    pub counters: crate::permanent::Counters,
    /// The departed appearance; recorded as `object_ref` and its slot as `card`.
    #[serde(flatten, serialize_with = "serialize_lki_object")]
    pub object: ObjectId,
    pub from_zone: ZoneType,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub definition_id: CardDefId,
    pub presentation_id: PublicId,
}

fn serialize_lki_object<S: serde::Serializer>(
    object: &ObjectId,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(2))?;
    map.serialize_entry("object_ref", &Exact(object))?;
    map.serialize_entry("card", object)?;
    map.end()
}

/// Last-known battlefield facts of departed exact objects, keyed by object,
/// with each battlefield slot's departed object at hand. Zone-change
/// events carry their own departure identity, never a latest-by-row query.
/// Iterates in object order; the canonical state hash serializes that.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectLkiLog {
    by_object: BTreeMap<ObjectId, ObjectLki>,
    indexes: Box<DepartureIndexes>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct DepartureIndexes {
    /// Departures still carrying extra copiable values.
    forms: std::collections::BTreeSet<ObjectId>,
    /// The departed object that occupied each battlefield slot, by
    /// `PermanentId`; a slot is never reused.
    by_permanent: Vec<Option<ObjectId>>,
}

impl ObjectLkiLog {
    pub fn insert(&mut self, lki: ObjectLki) {
        let object = lki.object;
        let slot = lki.permanent.0;
        if self.indexes.by_permanent.len() <= slot {
            self.indexes.by_permanent.resize(slot + 1, None);
        }
        self.indexes.by_permanent[slot].get_or_insert(object);
        if lki.copiable.is_some() {
            self.indexes.forms.insert(object);
        }
        self.by_object.insert(object, lki);
    }

    /// Release extra values once no resolution can read the departed form.
    /// Only visits departures that still own a snapshot.
    pub fn retain_forms(&mut self, mut keep: impl FnMut(ObjectId) -> bool) {
        self.indexes.forms.retain(|object| {
            if keep(*object) {
                return true;
            }
            self.by_object
                .get_mut(object)
                .expect("indexed departure")
                .copiable = None;
            false
        });
    }

    pub fn get(&self, object: ObjectId) -> Option<&ObjectLki> {
        self.by_object.get(&object)
    }

    pub fn contains(&self, object: ObjectId) -> bool {
        self.by_object.contains_key(&object)
    }

    /// The departed object that occupied a battlefield slot.
    pub fn for_permanent(&self, permanent: PermanentId) -> Option<&ObjectLki> {
        self.indexes
            .by_permanent
            .get(permanent.0)
            .copied()
            .flatten()
            .and_then(|object| self.by_object.get(&object))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ObjectId, &ObjectLki)> {
        self.by_object.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct PermanentId(pub usize);

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct PlayerId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum Target {
    Player(PlayerId),
    Permanent(PermanentId),
    StackSpell(ObjectId),
    /// An activated or triggered ability on the stack, by its stack object id.
    StackAbility(PublicId),
    GraveyardCard(#[serde(serialize_with = "exact")] ObjectId),
}

// Typed index wrappers — prevent accidental cross-collection indexing.

#[derive(Clone, Debug, serde::Serialize)]
pub struct ObjectVec<T>(pub Vec<T>);

impl<T> Default for ObjectVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> std::ops::Deref for ObjectVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}
impl<T> std::ops::DerefMut for ObjectVec<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}
impl ObjectVec<Object> {
    /// Retained printing, independent of the slot's current copy or face.
    /// This is definition provenance, never a lookup of a departed object.
    pub fn retained_base(&self, object: ObjectId) -> Option<(CardDefId, &PrintedCharacteristics)> {
        self.get(object.slot()).map(Object::retained_base)
    }

    /// Viewer identity and ownership are stable across appearances.
    pub fn provenance(&self, object: ObjectId) -> Option<(PublicId, PlayerId, ObjectKind)> {
        self.get(object.slot())
            .map(|row| (row.id, row.owner, row.kind()))
    }

    /// Narrow a storage id once; missing rows, tokens and spell copies are not cards.
    #[inline]
    pub fn card_id(&self, object: ObjectId) -> Option<CardId> {
        self.get(object.slot())
            .filter(|value| value.is_card())
            .map(|_| CardId::from_index(object.slot()))
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
    pub fn next_id(&mut self) -> PublicId {
        let out = PublicId(self.next_id);
        self.next_id += 1;
        out
    }

    pub fn watermark(&self) -> u32 {
        self.next_id
    }
}

// ---------------------------------------------------------------------------
// Runtime objects

/// The kind of object a runtime object is (CR 109.1). An object keeps its
/// kind for its whole life. A copy of a permanent spell resolves into a new
/// token appearance (CR 111.12, 608.3f); the zone mint sets that successor kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ObjectKind {
    /// A Magic card (CR 108.2), from a card definition.
    Card,
    /// A token (CR 111.1), from a [`TokenDefinition`](crate::card::TokenDefinition) or a copy effect. It
    /// ceases to exist in any zone but the battlefield (CR 111.7, 704.5d).
    Token,
    /// A copy of a spell (CR 707.10), made on the stack. It ceases to exist
    /// in any zone but the stack (CR 704.5e).
    SpellCopy,
}

/// A raw face: a catalog row id and its printed characteristics, before any
/// copy exception.
pub type Face = (CardDefId, Arc<PrintedCharacteristics>);

/// Both raw faces of a transforming double-faced object (CR 712.1): a
/// transforming double-faced card, or a transforming token made as a copy of
/// one (CR 707.8a). Copy exceptions are never written into these faces; each
/// face transition applies them to the face it shows (CR 707.9b).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TransformingFaces {
    pub front: Face,
    pub back: Face,
    /// The back face is up (CR 712.8).
    pub back_up: bool,
}

impl TransformingFaces {
    /// The face that is up.
    pub fn shown(&self) -> &Face {
        if self.back_up {
            &self.back
        } else {
            &self.front
        }
    }

    /// The same faces turned over (CR 701.27a).
    pub fn turned_over(&self) -> Self {
        Self {
            back_up: !self.back_up,
            ..self.clone()
        }
    }
}

/// What an object is beyond its shown printed face, when that differs from
/// an ordinary card or token resting on its own printing: the physical
/// printing behind a face-down, melded or entry-copy form, a transforming
/// object's raw faces, and the copy effect behind its copiable values.
///
/// Boxed on [`Object::form`] so ordinary objects allocate nothing. Melded zone
/// membership belongs to `ZoneManager`, not here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BattlefieldForm {
    /// The object's own printing while it shows another face (a manifest's
    /// 2/2, a meld result, an entry copy), restored by its next zone move
    /// (CR 400.7, 707.5, 708.9, 712.4c).
    backing: Option<Face>,
    /// A transforming object's raw faces. A double-faced card that has not
    /// transformed reads them from the catalog instead.
    faces: Option<TransformingFaces>,
    /// Face down (CR 708.2): `backing` holds the hidden printing.
    face_down: bool,
    /// The copy effect that made this object a copy (CR 707.2); the shown
    /// face is then the copiable values that effect produced.
    copy: Option<crate::effect::CopyEffect>,
}

impl BattlefieldForm {
    pub fn backing(&self) -> Option<&Face> {
        self.backing.as_ref()
    }

    pub fn faces(&self) -> Option<&TransformingFaces> {
        self.faces.as_ref()
    }

    pub fn copy(&self) -> Option<&crate::effect::CopyEffect> {
        self.copy.as_ref()
    }

    pub fn is_face_down(&self) -> bool {
        self.face_down
    }
}

/// One object in a game: a card, a token or a copy of a spell, told apart
/// by [`Object::kind`]. Its shown characteristics are read through `Deref`;
/// zone state (a `Permanent`, a stack entry) lives beside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: PublicId,
    kind: ObjectKind,
    pub definition_id: CardDefId,
    /// Compatibility name for the existing observation ABI. This is always
    /// the numeric representation of `definition_id`.
    pub registry_key: PublicId,
    pub owner: PlayerId,
    /// The face it shows, with any copy exceptions applied.
    definition: Arc<PrintedCharacteristics>,
    /// Physical backing, raw faces, orientation and copy effect, when they
    /// differ from an ordinary object on its own printing. Fixtures may take
    /// and restore it; rules code changes it only through `Object` methods.
    pub form: Option<Box<BattlefieldForm>>,
}

impl Object {
    /// An object of a catalog row: a token for a token definition's row, a
    /// card otherwise.
    pub fn from_definition(
        id: PublicId,
        owner: PlayerId,
        definition_id: CardDefId,
        definition: Arc<PrintedCharacteristics>,
        is_token: bool,
    ) -> Self {
        Self {
            id,
            kind: if is_token {
                ObjectKind::Token
            } else {
                ObjectKind::Card
            },
            definition_id,
            registry_key: PublicId(definition_id.0),
            owner,
            definition,
            form: None,
        }
    }

    /// A token with copiable values (CR 707.2, 111.1): the shown face
    /// `definition`, the raw `faces` of a transforming source (CR 707.8a) and
    /// the source's copy effect. `exceptions` then apply after the source's
    /// own (CR 707.9b). Token status is not copiable; it is this object's kind.
    pub fn token_with_copiable_values(
        id: PublicId,
        owner: PlayerId,
        (definition_id, definition): Face,
        faces: Option<TransformingFaces>,
        copy: Option<crate::effect::CopyEffect>,
        exceptions: &[crate::effect::CopyException],
    ) -> Self {
        let mut token = Self::from_definition(id, owner, definition_id, definition, true);
        token.form = Some(Box::new(BattlefieldForm {
            faces,
            copy,
            ..Default::default()
        }));
        token.apply_copy_exceptions(exceptions);
        token
    }

    /// Enter as a copy of `source` (CR 707.5): take its shown copiable values
    /// and copy chain, then `exceptions` (CR 707.9b). The copy is backed by
    /// this object's own single printing, so it cannot transform and its mana
    /// value is the copied face's (CR 712.8e, 712.9). Token status is not
    /// copiable. The printing returns at [`Object::mint_appearance`].
    pub fn become_copy_of(&mut self, source: &Object, exceptions: &[crate::effect::CopyException]) {
        let (id, row) = self.retained_definition();
        let backing = (id, Arc::clone(row));
        self.definition_id = source.definition_id;
        self.registry_key = PublicId(source.definition_id.0);
        self.definition = source.copiable_definition();
        self.form = Some(Box::new(BattlefieldForm {
            backing: Some(backing),
            copy: source.copy().cloned(),
            ..Default::default()
        }));
        self.apply_copy_exceptions(exceptions);
    }

    /// Record the copy's root and apply new exceptions after inherited ones.
    pub fn apply_copy_exceptions(&mut self, exceptions: &[crate::effect::CopyException]) {
        for exception in exceptions {
            exception.apply(self);
        }
        let of = self.definition_id;
        self.form
            .get_or_insert_with(Default::default)
            .copy
            .get_or_insert_with(|| crate::effect::CopyEffect {
                of,
                exceptions: Vec::new(),
            })
            .exceptions
            .extend_from_slice(exceptions);
    }

    /// The copy effect that made this object a copy (CR 707.2).
    pub fn copy(&self) -> Option<&crate::effect::CopyEffect> {
        self.form.as_deref().and_then(BattlefieldForm::copy)
    }

    /// Raw faces held by the form: a transformed card or a transforming token.
    /// A double-faced card on its front face reads its faces from the catalog.
    pub fn transforming_faces(&self) -> Option<&TransformingFaces> {
        self.form.as_deref().and_then(BattlefieldForm::faces)
    }

    /// Face down (CR 708.2).
    pub fn is_face_down(&self) -> bool {
        self.form
            .as_deref()
            .is_some_and(BattlefieldForm::is_face_down)
    }

    /// The base that a zone move restores, including fixture edits. It never
    /// follows an entry-copy or transformed face installed on a later appearance.
    pub fn retained_base(&self) -> (CardDefId, &PrintedCharacteristics) {
        let (id, row) = self.retained_definition();
        (id, &**row)
    }

    /// Assign a printing in a sampled world while preserving its current appearance.
    pub fn assign_hidden_printing(&mut self, id: CardDefId, row: Arc<PrintedCharacteristics>) {
        match self
            .form
            .as_deref_mut()
            .and_then(|form| form.backing.as_mut())
        {
            Some(backing) => *backing = (id, row),
            None => self.show(id, row),
        }
    }

    pub fn retained_definition(&self) -> (CardDefId, &Arc<PrintedCharacteristics>) {
        let form = self.form.as_deref();
        form.and_then(|form| form.backing.as_ref())
            .or(form
                .and_then(|form| form.faces.as_ref())
                .map(|faces| &faces.front))
            .map_or((self.definition_id, &self.definition), |(id, row)| {
                (*id, row)
            })
    }

    /// Mint the values of a new zone appearance (CR 400.7), consuming the
    /// retained base in place rather than loading another definition. Only
    /// the zone-move authority calls this, alongside advancing identity.
    ///
    /// `entry_copy` assigns the prospective values already installed by an
    /// enters-as-copy replacement to the new entry (CR 614.1c, 614.12).
    /// Otherwise the old appearance's backing or front face returns. A token or
    /// spell copy starts with its own copiable base, not a physical printing.
    pub fn mint_appearance(&mut self, from: Option<ZoneType>, to: ZoneType, entry_copy: bool) {
        debug_assert!(!entry_copy || (to == ZoneType::Battlefield && self.copy().is_some()));
        if !entry_copy && (self.is_card() || from == Some(ZoneType::Battlefield)) {
            if let Some(form) = self.form.as_deref_mut() {
                if let Some((id, definition)) = form.backing.take() {
                    // An entry copy, meld or face-down form ends with its
                    // appearance (CR 707.5, 708.9, 712.4c).
                    self.form = None;
                    self.show(id, definition);
                } else if let Some(faces) = form.faces.as_ref().filter(|faces| faces.back_up) {
                    // CR 712.8a. A token made as a copy keeps its inherent
                    // copy exceptions on its front face, just as on a transform.
                    let front = faces.turned_over();
                    self.show_transforming_face(front);
                }
            }
        }
        // CR 111.12, 608.3f: the resolving copy's successor is a token.
        if to == ZoneType::Battlefield && self.kind == ObjectKind::SpellCopy {
            self.kind = ObjectKind::Token;
        }
    }

    /// A fresh token with the source's copiable values (CR 707.2, 111.1).
    pub fn token_copy(source: Self, id: PublicId, owner: PlayerId) -> Self {
        Self {
            id,
            owner,
            kind: ObjectKind::Token,
            ..source
        }
    }

    /// A fresh copy of a spell on the stack (CR 707.10), never a card.
    pub fn spell_copy(source: Self, id: PublicId, owner: PlayerId) -> Self {
        Self {
            id,
            owner,
            kind: ObjectKind::SpellCopy,
            ..source
        }
    }

    #[inline]
    pub fn kind(&self) -> ObjectKind {
        self.kind
    }

    /// Whether it is a Magic card (CR 108.2): not a token or a copy. Effects
    /// that say "card" see only these.
    #[inline]
    pub fn is_card(&self) -> bool {
        self.kind == ObjectKind::Card
    }

    /// Its mana cost for mana value: a transformed permanent's is its front
    /// face's (CR 712.8e).
    pub fn mana_value_cost(&self) -> Option<&ManaCost> {
        match self.transforming_faces() {
            Some(faces) if faces.back_up => faces.front.1.mana_cost.as_ref(),
            _ => self.mana_cost.as_ref(),
        }
    }

    /// Whether its back face is up (CR 712.8).
    pub fn is_transformed(&self) -> bool {
        self.transforming_faces().is_some_and(|faces| faces.back_up)
    }

    /// Show the up face of `faces`, applying the copy effect's exceptions
    /// once to that raw face (CR 707.8a, 707.9b, 712.8). Whether the object
    /// may transform at all is the caller's question (CR 712.9, 712.10).
    pub fn show_transforming_face(&mut self, faces: TransformingFaces) {
        let (id, row) = faces.shown().clone();
        self.form.get_or_insert_with(Default::default).faces = Some(faces);
        self.show(id, row);
        self.settle_form();
    }

    /// Show a meld result (CR 712.4c), keeping the primary card's printing
    /// for its next zone move.
    pub fn show_melded(&mut self, id: CardDefId, definition: Arc<PrintedCharacteristics>) {
        self.show_over_backing(id, definition);
    }

    /// Turn face down as `definition`, the face-down characteristics
    /// (CR 708.2), hiding this object's printing behind it.
    pub fn turn_face_down(&mut self, definition: Arc<PrintedCharacteristics>) {
        self.show_over_backing(CardDefId::ANONYMOUS, definition);
        self.form.as_deref_mut().expect("backed form").face_down = true;
    }

    /// Turn a face-down object face up in place, showing its hidden printing
    /// (CR 708.8). False when it is not face down.
    pub fn turn_face_up(&mut self) -> bool {
        if !self.is_face_down() {
            return false;
        }
        let form = self.form.as_deref_mut().expect("face-down form");
        let (id, definition) = form.backing.take().expect("face down over a printing");
        form.face_down = false;
        self.show(id, definition);
        self.settle_form();
        true
    }

    fn show_over_backing(&mut self, id: CardDefId, definition: Arc<PrintedCharacteristics>) {
        let backing = (self.definition_id, Arc::clone(&self.definition));
        self.form.get_or_insert_with(Default::default).backing = Some(backing);
        self.show(id, definition);
    }

    /// Show one face. Its kind stays (a double-faced token is a token on
    /// either face), and the copy effect's exceptions apply to it once,
    /// keeping the copy's root (CR 707.8a, 707.9b).
    fn show(&mut self, id: CardDefId, definition: Arc<PrintedCharacteristics>) {
        self.definition_id = id;
        self.registry_key = PublicId(id.0);
        self.definition = definition;
        if let Some(copy) = self.form.as_deref_mut().and_then(|form| form.copy.take()) {
            for exception in &copy.exceptions {
                exception.apply(self);
            }
            self.form.as_deref_mut().expect("copy form").copy = Some(copy);
        }
    }

    /// Drop a form that says nothing the catalog does not: a double-faced
    /// card back on its front face with no copy, backing or orientation.
    fn settle_form(&mut self) {
        let ordinary = self.form.as_deref().is_some_and(|form| {
            form.backing.is_none()
                && !form.face_down
                && form.copy.is_none()
                && form
                    .faces
                    .as_ref()
                    .is_none_or(|faces| !faces.back_up && self.kind == ObjectKind::Card)
        });
        if ordinary {
            self.form = None;
        }
    }

    /// Snapshot the current copiable face without copying its ability programs.
    pub fn copiable_definition(&self) -> Arc<PrintedCharacteristics> {
        Arc::clone(self.copiable_face())
    }

    pub fn copiable_face(&self) -> &Arc<PrintedCharacteristics> {
        &self.definition
    }

    pub fn shares_definition_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.definition, &other.definition)
    }

    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
    }

    pub fn is_legendary(&self) -> bool {
        self.supertypes
            .iter()
            .any(|kind| kind.eq_ignore_ascii_case("legendary"))
    }

    /// CR 117.1a, 702.8 — Instants and cards with flash use instant timing.
    /// Printed instant timing; a conditional `StaticAbility::FlashIf` is checked by
    /// `Game::has_instant_timing`.
    pub fn is_instant_speed(&self) -> bool {
        self.types.is_instant() || self.keywords.has(Keyword::Flash)
    }

    /// Modes are announced before targets (CR 601.2b-c). Only spell-level
    /// modal effects participate; choices inside resolving abilities do not.
    pub fn spell_modes(&self) -> Option<&[Vec<Effect>]> {
        self.spell_effects.iter().find_map(|effect| match effect {
            Effect::Modal { modes } => Some(modes.as_slice()),
            _ => None,
        })
    }

    /// An Aura's enchant restriction (CR 303.4a, 702.5c): what its spell
    /// targets and what its attached object must keep matching (CR 704.5m).
    pub fn enchant(&self) -> Option<&TargetSpec> {
        if !self.has_subtype("Aura") {
            return None;
        }
        self.spell_effects.iter().find_map(|effect| match effect {
            Effect::AttachSourceTo { what } => what.target_spec(),
            _ => None,
        })
    }

    pub fn spell_effects_for_mode(&self, mode: Option<usize>) -> Vec<Effect> {
        self.spell_effects
            .iter()
            .flat_map(|effect| match (effect, mode) {
                (Effect::Modal { modes }, Some(index)) => {
                    modes.get(index).map(Vec::as_slice).unwrap_or_default()
                }
                _ => std::slice::from_ref(effect),
            })
            .cloned()
            .collect()
    }

    pub fn target_requirements_for_mode(&self, mode: Option<usize>) -> Vec<TargetRequirement> {
        if !self.targeting.is_empty() {
            return self.targeting.clone();
        }
        crate::ability::effect_target_requirements(&self.spell_effects_for_mode(mode))
    }

    /// Explicit clauses bind distinct targets; repeated effects on "that"
    /// target share the inferred single clause.
    pub fn target_requirements(&self) -> Vec<TargetRequirement> {
        self.target_requirements_for_mode(None)
    }
}

impl std::ops::Deref for Object {
    type Target = PrintedCharacteristics;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

/// Preserve the existing Rust scenario-test seam without making ordinary
/// match clones copy definitions. Mutating printed characteristics detaches
/// only that physical card from the shared pack definition.
impl std::ops::DerefMut for Object {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.definition)
    }
}

impl std::fmt::Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{name: {}}}", self.name)
    }
}
