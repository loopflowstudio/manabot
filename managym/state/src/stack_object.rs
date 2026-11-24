use super::ability::Effect;
use super::card::{AbilityGrant, CardDefId};
use super::game_object::{Exact, ObjectId, ObjectLki, PlayerId, PublicId, Target};
use crate::binding::ObjectBinding;
use crate::payment::PaymentReceipt;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellOnStack {
    pub cast_from: super::zone::ZoneType,
    pub flashback: bool,
    pub id: PublicId,
    #[serde(rename = "card")]
    pub object: ObjectId,
    pub controller: PlayerId,
    /// Immutable meaning of the source card. The serialized field name stays
    /// legacy-compatible for benchmark snapshots and observation adapters.
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    pub targets: Vec<Target>,
    /// Which targeting requirement (index into the card's
    /// `target_requirements()`) each entry of `targets` was chosen for.
    pub target_req_indices: Vec<usize>,
    /// Whether the kicker cost was paid while casting (CR 702.33).
    pub kicked: bool,
    /// The value of X announced while casting (CR 107.3); its mana value
    /// on the stack includes it (CR 202.3e).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: u8,
    pub mode: Option<usize>,
    /// The mana actually spent to cast it (CR 601.2h), for "mana spent to
    /// cast" and converge; empty for a spell cast without paying mana.
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub spent: PaymentReceipt,
}

/// A spell as it is on the stack or last existed there (CR 608.2h): what
/// "its controller", "the mana spent to cast that spell" and "its mana value"
/// read after the spell is countered, resolves or otherwise leaves. Copies of
/// spells are spells too (CR 707.10); one cast without paying mana spent none.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellLki {
    #[serde(serialize_with = "super::game_object::exact")]
    pub object: ObjectId,
    pub definition_id: CardDefId,
    pub controller: PlayerId,
    /// Including its announced X (CR 202.3e).
    pub mana_value: u8,
    /// Total mana actually spent to cast it (CR 601.2h).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub spent: u16,
    #[serde(skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    pub colors: crate::mana::Colors,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ActivatedAbilityOnStack {
    /// Chosen effects, captured before costs; resolution never re-reads a source.
    pub effects: Vec<Effect>,
    /// No catalog identity describes this dynamically granted program.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub program: bool,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub source_transforms: u32,
    /// The definition lending the activated ability, fixed at activation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<CardDefId>,
    pub id: PublicId,
    pub controller: PlayerId,
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    /// The exact source; recorded as its slot under `source_card` and whole
    /// under `source_ref`.
    #[serde(flatten, serialize_with = "serialize_activation_source")]
    pub source_object: ObjectId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    /// The clause of each target when the ability has several; empty means
    /// every target is clause 0.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub target_req_indices: Vec<usize>,
    /// The announced value of X (CR 602.2b, 601.2b).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: u8,
    /// For an ability granted by an attached permanent: that exact object and
    /// its grant, fixed at activation (`ability_index` names the grantee's list).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilityGrant>,
    /// What activating it paid (CR 602.2b): the mana spent and the objects
    /// tapped, sacrificed, discarded or exiled for its cost.
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub paid: PaymentReceipt,
}

fn serialize_activation_source<S: serde::Serializer>(
    source: &ObjectId,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(2))?;
    map.serialize_entry("source_card", source)?;
    map.serialize_entry("source_ref", &Exact(source))?;
    map.end()
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TriggeredAbilityOnStack {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub program: bool,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub triggering_amount: i32,
    /// Exact subjects captured at delayed-trigger creation, independent of event context.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub remembered: Vec<ObjectBinding>,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub source_transforms: u32,
    pub id: PublicId,
    pub controller: PlayerId,
    #[serde(flatten)]
    pub source_identity: crate::game_object::AbilitySource,
    pub source_lki: Option<ObjectLki>,
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub target_req_indices: Vec<usize>,
    pub context: Option<Target>,
    /// The trigger's player ("they": the caster of the triggering spell).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerId>,
    /// The ability's X (`PendingTrigger::x`).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: u8,
    /// The mana spent to cast the permanent whose entering triggered this,
    /// for its own "mana spent to cast" (`PendingTrigger::spent`).
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub spent: PaymentReceipt,
    /// A printed modal trigger's chosen modes (CR 700.2), expanded from
    /// its printed `Effect::TriggerModes` at resolution.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chosen_modes: Vec<usize>,
    /// Delayed, granted and reflexive triggers resolve these effects
    /// instead of looking up `Object::abilities[ability_index]`.
    pub inline_effects: Option<Vec<Effect>>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StackObject {
    Spell(SpellOnStack),
    ActivatedAbility(ActivatedAbilityOnStack),
    TriggeredAbility(TriggeredAbilityOnStack),
}

impl StackObject {
    pub fn id(&self) -> PublicId {
        match self {
            StackObject::Spell(spell) => spell.id,
            StackObject::ActivatedAbility(ability) => ability.id,
            StackObject::TriggeredAbility(triggered) => triggered.id,
        }
    }

    pub fn controller(&self) -> PlayerId {
        match self {
            StackObject::Spell(spell) => spell.controller,
            StackObject::ActivatedAbility(ability) => ability.controller,
            StackObject::TriggeredAbility(triggered) => triggered.controller,
        }
    }

    /// The object as a target: a spell by its object, an ability by its id.
    pub fn as_target(&self) -> Target {
        match self {
            StackObject::Spell(spell) => Target::StackSpell(spell.object),
            object => Target::StackAbility(object.id()),
        }
    }

    /// The spell's object, or the object whose ability this is.
    pub fn source_object(&self) -> ObjectId {
        match self {
            StackObject::Spell(spell) => spell.object,
            StackObject::ActivatedAbility(ability) => ability.source_object(),
            StackObject::TriggeredAbility(triggered) => triggered.source_object(),
        }
    }

    /// The exact source object of an ability; spells have none.
    pub fn source_ref(&self) -> Option<ObjectId> {
        match self {
            StackObject::Spell(_) => None,
            StackObject::ActivatedAbility(ability) => Some(ability.source_object()),
            StackObject::TriggeredAbility(triggered) => triggered.source_ref(),
        }
    }

    pub fn source_definition_id(&self) -> CardDefId {
        match self {
            StackObject::Spell(spell) => spell.source_definition_id,
            StackObject::ActivatedAbility(ability) => ability.source_definition_id,
            StackObject::TriggeredAbility(triggered) => triggered.source_definition_id,
        }
    }

    /// Legacy numeric observation projection. Internal state uses
    /// [`CardDefId`]; the current Python/observation ABI still exposes an
    /// integer registry key with the same value.
    pub fn source_card_registry_key(&self) -> PublicId {
        PublicId(self.source_definition_id().0)
    }

    pub fn ability_index(&self) -> Option<usize> {
        match self {
            StackObject::Spell(_) => None,
            StackObject::ActivatedAbility(ability) => Some(ability.ability_index),
            StackObject::TriggeredAbility(triggered) => Some(triggered.ability_index),
        }
    }

    pub fn targets(&self) -> &[Target] {
        match self {
            StackObject::Spell(spell) => &spell.targets,
            StackObject::ActivatedAbility(ability) => &ability.targets,
            StackObject::TriggeredAbility(triggered) => &triggered.targets,
        }
    }
}

impl TriggeredAbilityOnStack {
    pub fn source_object(&self) -> ObjectId {
        self.source_identity.object()
    }
    pub fn source_ref(&self) -> Option<ObjectId> {
        self.source_identity.reference()
    }
}

impl ActivatedAbilityOnStack {
    pub fn source_object(&self) -> ObjectId {
        self.source_object
    }
}
