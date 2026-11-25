use super::ability::Effect;
use super::card::{AbilityGrant, CardDefId};
use super::game_object::{AbilitySource, Exact, ObjectId, ObjectLki, PlayerId, PublicId, Target};
use crate::binding::ObjectBinding;
use crate::card::SpellProfile;
use crate::payment::PaymentReceipt;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellOnStack {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub exile_instead_of_graveyard: bool,
    pub cast_from: super::zone::ZoneType,
    pub flashback: bool,
    /// Cast from a prepared copy (a "prepared spell"). Copies of it are not.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub from_prepared: bool,
    #[serde(rename = "adventure", skip_serializing_if = "SpellProfile::is_front")]
    pub profile: SpellProfile,
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
    #[serde(default, skip_serializing_if = "SpellProfile::is_front")]
    pub profile: SpellProfile,
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
    /// Exact object attached to the source when the ability was activated.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "crate::game_object::exact")]
    pub attached: Option<ObjectId>,
    /// The announced value of X (CR 602.2b, 601.2b).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: i32,
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
    /// Captured characteristics if the catalog row alone cannot identify the
    /// triggering face. Forks and copied abilities share this immutable value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_face: Option<std::sync::Arc<super::card::PrintedCharacteristics>>,
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
    pub source_identity: AbilitySource,
    pub source_lki: Option<ObjectLki>,
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    pub target_req_indices: Vec<usize>,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub context: Option<Target>,
    /// The trigger's player ("they": the caster of the triggering spell).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerId>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub kicked: bool,
    /// The ability's X (`PendingTrigger::x`).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: i32,
    /// The mana spent to cast the permanent whose entering triggered this,
    /// for its own "mana spent to cast" (`PendingTrigger::spent`).
    #[serde(skip_serializing_if = "PaymentReceipt::is_empty")]
    pub spent: PaymentReceipt,
    /// The modes chosen for a printed modal trigger (CR 700.2b); empty for
    /// any other trigger, or when "up to" chose none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chosen_modes: Vec<usize>,
    /// Non-printed triggers (delayed, reflexive, granted, firebending)
    /// resolve these effects instead of `Object::abilities[ability_index]`.
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

    /// An effect of a spell uses the spell itself as its exact source.
    /// Abilities retain the source captured when they were created.
    pub fn effect_source(&self) -> AbilitySource {
        match self {
            Self::Spell(spell) => AbilitySource::new(spell.object, Some(spell.object)),
            Self::ActivatedAbility(ability) => {
                AbilitySource::new(ability.source_object, Some(ability.source_object))
            }
            Self::TriggeredAbility(trigger) => trigger.source_identity,
        }
    }

    pub fn source_object(&self) -> ObjectId {
        self.effect_source().object()
    }

    /// The exact source object of an ability; spells have none.
    pub fn source_ref(&self) -> Option<ObjectId> {
        match self {
            Self::Spell(_) => None,
            ability => ability.effect_source().reference(),
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

    /// The spell's kicker choice, including the choice carried by its ETB trigger.
    pub fn kicked(&self) -> bool {
        match self {
            Self::Spell(spell) => spell.kicked,
            Self::TriggeredAbility(trigger) => trigger.kicked,
            Self::ActivatedAbility(_) => false,
        }
    }

    pub fn ability_index(&self) -> Option<usize> {
        match self {
            StackObject::Spell(_) => None,
            StackObject::ActivatedAbility(ability) => Some(ability.ability_index),
            StackObject::TriggeredAbility(triggered) => Some(triggered.ability_index),
        }
    }

    pub fn x(&self) -> i32 {
        match self {
            Self::Spell(spell) => i32::from(spell.x),
            Self::ActivatedAbility(ability) => ability.x,
            Self::TriggeredAbility(trigger) => trigger.x,
        }
    }

    pub fn target_req_indices(&self) -> &[usize] {
        match self {
            Self::Spell(spell) => &spell.target_req_indices,
            Self::ActivatedAbility(ability) => &ability.target_req_indices,
            Self::TriggeredAbility(trigger) => &trigger.target_req_indices,
        }
    }

    pub fn targets(&self) -> &[Target] {
        match self {
            StackObject::Spell(spell) => &spell.targets,
            StackObject::ActivatedAbility(ability) => &ability.targets,
            StackObject::TriggeredAbility(triggered) => &triggered.targets,
        }
    }

    pub fn targets_mut(&mut self) -> &mut [Target] {
        match self {
            StackObject::Spell(spell) => &mut spell.targets,
            StackObject::ActivatedAbility(ability) => &mut ability.targets,
            StackObject::TriggeredAbility(triggered) => &mut triggered.targets,
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
