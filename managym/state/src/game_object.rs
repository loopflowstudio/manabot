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
    pub fn in_slot(slot: usize, generation: u32) -> Self {
        Self {
            slot: u32::try_from(slot).expect("object store exceeds u32"),
            generation,
        }
    }

    /// Storage row, for the stores, the state hash and viewer captures.
    #[doc(hidden)]
    pub fn slot(self) -> usize {
        self.slot as usize
    }

    /// How many zone changes preceded this appearance in its slot.
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

struct Exact<'a, T: ?Sized>(&'a T);

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

/// A physical-card roster index (CR 108.2), independent of appearance ids.
///
/// Internally one-based so an optional card association is also four bytes.
/// Serialization retains the zero-based number in saved decisions and rosters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CardId(NonZeroU32);

impl CardId {
    fn from_index(index: usize) -> Self {
        let one_based = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(NonZeroU32::new)
            .expect("physical-card roster exceeds u32");
        Self(one_based)
    }

    /// Zero-based roster position, which is also the card's storage slot.
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

/// Flat physical-card roster, frozen after setup. Tokens and copies belong
/// solely to the object store.
#[derive(Clone, Debug, Default)]
pub struct CardVec(Vec<Card>);

impl CardVec {
    pub fn new(cards: Vec<Card>) -> Self {
        Self(cards)
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

/// Battlefield facts retained for a departed exact object.
///
/// This deliberately stores identity and mutable facts, not a cloned card
/// definition. `definition_id` resolves through the match's immutable
/// `ContentPack` and does not clone card semantics into mutable state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectLki {
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub transforms: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<crate::card::CardDefId>,
    /// Effective grants retained at the event boundary, including after ability loss.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub granted_abilities: Vec<crate::ability::EffectiveTrigger>,
    pub keywords: crate::card::Keywords,
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
    /// The departed object that occupied each battlefield slot, by
    /// `PermanentId`; a slot is never reused.
    by_permanent: Vec<Option<ObjectId>>,
}

impl ObjectLkiLog {
    pub fn insert(&mut self, lki: ObjectLki) {
        let object = lki.object;
        let slot = lki.permanent.0;
        if self.by_permanent.len() <= slot {
            self.by_permanent.resize(slot + 1, None);
        }
        self.by_permanent[slot].get_or_insert(object);
        self.by_object.insert(object, lki);
    }

    pub fn get(&self, object: ObjectId) -> Option<&ObjectLki> {
        self.by_object.get(&object)
    }

    pub fn contains(&self, object: ObjectId) -> bool {
        self.by_object.contains_key(&object)
    }

    /// The departed object that occupied a battlefield slot.
    pub fn for_permanent(&self, permanent: PermanentId) -> Option<&ObjectLki> {
        self.by_permanent
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
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
impl<T> std::ops::Index<ObjectId> for ObjectVec<T> {
    type Output = T;
    fn index(&self, id: ObjectId) -> &T {
        &self.0[id.slot()]
    }
}
impl<T> std::ops::Index<&ObjectId> for ObjectVec<T> {
    type Output = T;
    fn index(&self, id: &ObjectId) -> &T {
        &self.0[id.slot()]
    }
}
impl<T> std::ops::IndexMut<ObjectId> for ObjectVec<T> {
    fn index_mut(&mut self, id: ObjectId) -> &mut T {
        &mut self.0[id.slot()]
    }
}

impl ObjectVec<Object> {
    /// Narrow a storage id once; missing rows, tokens and spell copies are not cards.
    pub fn card_id(&self, object: ObjectId) -> Option<CardId> {
        self.get(object.slot())
            .filter(|value| value.is_card())
            .map(|_| CardId::from_index(object.slot()))
    }
}

impl std::ops::Index<CardId> for ObjectVec<Object> {
    type Output = Object;
    fn index(&self, card: CardId) -> &Object {
        let object = &self.0[card.index()];
        debug_assert!(object.is_card(), "CardId must still name a card");
        object
    }
}
impl std::ops::Index<&CardId> for ObjectVec<Object> {
    type Output = Object;
    fn index(&self, card: &CardId) -> &Object {
        &self[*card]
    }
}
impl std::ops::IndexMut<CardId> for ObjectVec<Object> {
    fn index_mut(&mut self, card: CardId) -> &mut Object {
        let object = &mut self.0[card.index()];
        debug_assert!(object.is_card(), "CardId must still name a card");
        object
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
/// kind for its whole life; the one change is a copy of a permanent spell,
/// which becomes a token as it resolves (CR 111.12, 608.3f).
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

/// One object in a game: a card, a token or a copy of a spell, told apart
/// by [`Object::kind`]. Its characteristics are its definition's, read
/// through `Deref`; zone state (a `Permanent`, a stack entry) lives beside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: PublicId,
    kind: ObjectKind,
    pub definition_id: CardDefId,
    /// Compatibility name for the existing observation ABI. This is always
    /// the numeric representation of `definition_id`.
    pub registry_key: PublicId,
    pub owner: PlayerId,
    /// The copy effect that made this object a copy (CR 707.2); its
    /// definition is then the copiable values that effect produced.
    pub copy: Option<crate::effect::CopyEffect>,
    definition: Arc<PrintedCharacteristics>,
    /// The front face's row while the back face is up (CR 712.8): set by
    /// [`Object::turn_back_face_up`], cleared by [`Object::turn_front_face_up`].
    front: Option<(CardDefId, Arc<PrintedCharacteristics>)>,
    /// The printed card while it is a copy it entered as (CR 707.5), restored
    /// by [`Object::mint_appearance`] at the next zone change.
    printed: Option<(CardDefId, Arc<PrintedCharacteristics>)>,
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
            copy: None,
            definition,
            front: None,
            printed: None,
        }
    }

    /// Enter as a copy of `source` (CR 707.5): take its copiable values,
    /// face and copy chain, then `exceptions` (CR 707.9b). Token status is
    /// not copiable, so the object stays a card or token as it was; its
    /// printed row is kept for [`Object::mint_appearance`].
    pub fn become_copy_of(&mut self, source: &Object, exceptions: &[crate::effect::Modification]) {
        let printed = self
            .printed
            .take()
            .unwrap_or_else(|| (self.definition_id, Arc::clone(&self.definition)));
        self.definition_id = source.definition_id;
        self.registry_key = PublicId(source.definition_id.0);
        self.definition = source.copiable_definition();
        self.front = source.front.clone();
        for exception in exceptions {
            exception.apply_copiable(self);
        }
        let mut copy = source.copy.clone().unwrap_or(crate::effect::CopyEffect {
            of: source.definition_id,
            exceptions: Vec::new(),
        });
        copy.exceptions.extend(exceptions.iter().cloned());
        self.copy = Some(copy);
        self.printed = Some(printed);
    }

    /// Mint the values of a new zone appearance (CR 400.7), consuming the
    /// retained base in place rather than loading another definition. Only
    /// the zone-move authority calls this, alongside advancing identity.
    ///
    /// `entry_copy` assigns the prospective values already installed by an
    /// enters-as-copy replacement to the new entry (CR 614.1c, 614.12).
    /// Otherwise the old appearance's copy/face is discarded. A token or
    /// spell copy starts with its own copiable base, not a physical printing.
    pub fn mint_appearance(&mut self, from: Option<ZoneType>, to: ZoneType, entry_copy: bool) {
        debug_assert!(!entry_copy || (to == ZoneType::Battlefield && self.copy.is_some()));
        if !entry_copy && (self.is_card() || from == Some(ZoneType::Battlefield)) {
            if let Some((id, definition)) = self.printed.take() {
                // An entry copy ends with its appearance (CR 707.5).
                self.copy = None;
                self.front = None;
                self.show(id, definition);
            } else {
                // CR 712.8a. A token made as a copy keeps its inherent copy
                // exceptions on its front face, just as on a transform.
                self.turn_front_face_up();
            }
        }
        // CR 111.12, 608.3f: the resolving copy's successor is a token.
        if to == ZoneType::Battlefield && self.kind == ObjectKind::SpellCopy {
            self.kind = ObjectKind::Token;
        }
    }

    /// The same object as `kind`: a token made by a copy effect (CR 707.2,
    /// 111.1) or a copy of a spell (CR 707.10), before it enters the game.
    pub fn with_kind(self, kind: ObjectKind) -> Self {
        Self { kind, ..self }
    }

    pub fn kind(&self) -> ObjectKind {
        self.kind
    }

    /// Whether it is a Magic card (CR 108.2): not a token or a copy. Effects
    /// that say "card" see only these.
    pub fn is_card(&self) -> bool {
        self.kind == ObjectKind::Card
    }

    /// Its mana cost for mana value: a transformed permanent's is its front
    /// face's (CR 712.8e).
    pub fn mana_value_cost(&self) -> Option<&ManaCost> {
        match &self.front {
            Some((_, front)) => front.mana_cost.as_ref(),
            None => self.mana_cost.as_ref(),
        }
    }

    /// Whether its back face is up (CR 712.8).
    pub fn is_transformed(&self) -> bool {
        self.front.is_some()
    }

    /// The front face's row while the back face is up.
    pub fn front_face(&self) -> Option<&(CardDefId, Arc<PrintedCharacteristics>)> {
        self.front.as_ref()
    }

    /// Carry a copied object's front face, keeping its mana value (CR 712.8).
    pub fn set_front_face(&mut self, front: Option<(CardDefId, Arc<PrintedCharacteristics>)>) {
        self.front = front;
    }

    /// Show the back face, a row of the engine catalog’s `ContentPack::back_face`.
    pub fn turn_back_face_up(
        &mut self,
        (id, definition): (CardDefId, Arc<PrintedCharacteristics>),
    ) {
        let front = (self.definition_id, Arc::clone(&self.definition));
        self.show(id, definition);
        self.front = Some(front);
    }

    /// Show the front face, if the back face is up.
    pub fn turn_front_face_up(&mut self) {
        if let Some((id, definition)) = self.front.take() {
            self.show(id, definition);
        }
    }

    /// Show one face. Its kind stays (a double-faced token is a token on
    /// either face), and a copy's exceptions are in the copiable values of
    /// each (CR 707.8a, 707.9b).
    fn show(&mut self, id: CardDefId, definition: Arc<PrintedCharacteristics>) {
        self.definition_id = id;
        self.registry_key = PublicId(id.0);
        self.definition = definition;
        if let Some(mut copy) = self.copy.take() {
            copy.of = id;
            for exception in &copy.exceptions {
                exception.apply_copiable(self);
            }
            self.copy = Some(copy);
        }
    }

    /// Snapshot the current copiable face without copying its ability programs.
    pub fn copiable_definition(&self) -> Arc<PrintedCharacteristics> {
        Arc::clone(&self.definition)
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
