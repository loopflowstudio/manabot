use super::ability::Effect;
use super::game_object::{CardId, ObjectId, ObjectLki, ObjectRef, PlayerId, Target};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpellOnStack {
    pub id: ObjectId,
    pub card: CardId,
    pub controller: PlayerId,
    pub source_card_registry_key: ObjectId,
    pub targets: Vec<Target>,
    /// Which targeting requirement (index into the card's
    /// `target_requirements()`) each entry of `targets` was chosen for.
    pub target_req_indices: Vec<usize>,
    /// Whether the kicker cost was paid while casting (CR 702.33).
    pub kicked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivatedAbilityOnStack {
    pub id: ObjectId,
    pub controller: PlayerId,
    pub source_card_registry_key: ObjectId,
    pub source_card: CardId,
    pub source_permanent_object_id: ObjectId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriggeredAbilityOnStack {
    pub id: ObjectId,
    pub controller: PlayerId,
    pub source_card: CardId,
    pub source_ref: Option<ObjectRef>,
    pub source_lki: Option<ObjectLki>,
    pub source_card_registry_key: ObjectId,
    pub ability_index: usize,
    pub targets: Vec<Target>,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub context: Option<Target>,
    /// Delayed triggers resolve these effects instead of looking up
    /// `Card::abilities[ability_index]`.
    pub inline_effects: Option<Vec<Effect>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

    pub fn source_card_registry_key(&self) -> ObjectId {
        match self {
            StackObject::Spell(spell) => spell.source_card_registry_key,
            StackObject::ActivatedAbility(ability) => ability.source_card_registry_key,
            StackObject::TriggeredAbility(triggered) => triggered.source_card_registry_key,
        }
    }

    pub fn source_permanent_object_id(&self) -> Option<ObjectId> {
        match self {
            StackObject::Spell(_) | StackObject::TriggeredAbility(_) => None,
            StackObject::ActivatedAbility(ability) => Some(ability.source_permanent_object_id),
        }
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
