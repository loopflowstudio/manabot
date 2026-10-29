use super::ability::Effect;
use super::card::{AbilityGrant, CardDefId};
use super::game_object::{CardId, ObjectId, ObjectLki, ObjectRef, PlayerId, Target};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellOnStack {
    pub cast_from: super::zone::ZoneType,
    pub flashback: bool,
    pub id: ObjectId,
    pub card: CardId,
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
    /// The announced value of X (CR 601.2b); 0 for spells without {X}.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    pub mode: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ActivatedAbilityOnStack {
    pub id: ObjectId,
    pub controller: PlayerId,
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    pub source_card: CardId,
    pub source_ref: ObjectRef,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    /// The exact object the source Aura or Equipment was attached to when
    /// the ability was activated, before costs such as sacrificing it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attached: Option<ObjectRef>,
    /// The announced value of X (CR 602.2b, 601.2b).
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// For an ability granted by an attached permanent: that exact object and
    /// its grant, fixed at activation (`ability_index` names the grantee's list).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilityGrant>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TriggeredAbilityOnStack {
    pub id: ObjectId,
    pub controller: PlayerId,
    pub source_card: CardId,
    pub source_ref: Option<ObjectRef>,
    pub source_lki: Option<ObjectLki>,
    #[serde(rename = "source_card_registry_key")]
    pub source_definition_id: CardDefId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub target_req_indices: Vec<usize>,
    pub context: Option<Target>,
    /// Delayed triggers resolve these effects instead of looking up
    /// `Card::abilities[ability_index]`.
    pub inline_effects: Option<Vec<Effect>>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StackObject {
    Spell(SpellOnStack),
    ActivatedAbility(ActivatedAbilityOnStack),
    TriggeredAbility(TriggeredAbilityOnStack),
}

impl StackObject {
    pub fn id(&self) -> ObjectId {
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
    pub fn source_card_registry_key(&self) -> ObjectId {
        ObjectId(self.source_definition_id().0)
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
