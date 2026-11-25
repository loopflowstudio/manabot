//! Waiting trigger occurrences and the resumable, two-pass placement round.
//! Collection owns arrivals; placement owns a detached batch until every
//! controller has placed its abilities (CR 603.3b, 2026-09-25 rules snapshot).

use std::sync::Arc;

use crate::agent::continuation::ProgramProjection;

use crate::state::binding::ObjectBinding;

use crate::state::{
    ability::{DelayedExpiry, DelayedFrequency, Effect},
    game_object::{AbilitySource, ObjectId, ObjectLki, PlayerId, Target},
    predicate::CardPredicate,
};

/// Stable occurrence identity; never a strategic ordering preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TriggerId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerOrderingClass {
    Ordinary,
    AbilityTriggered,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TriggerPlacement {
    pub remaining: Vec<PendingTrigger>,
    pub active: PlayerId,
    /// Ordinary AP/NAP, then ability-triggered AP/NAP. Four means complete.
    pub group: u8,
    pub placing: Option<PendingTrigger>,
}

impl TriggerPlacement {
    pub fn controller(&self) -> PlayerId {
        if self.group.is_multiple_of(2) {
            self.active
        } else {
            PlayerId(1 - self.active.0)
        }
    }

    pub fn in_current_group(&self, trigger: &PendingTrigger) -> bool {
        let class = if self.group < 2 {
            TriggerOrderingClass::Ordinary
        } else {
            TriggerOrderingClass::AbilityTriggered
        };
        self.group < 4 && trigger.controller == self.controller() && trigger.ordering_class == class
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingTrigger {
    /// An inline program granted by a continuous effect, without catalog identity.
    pub program: bool,
    pub triggering_amount: i32,
    /// Exact subjects captured at delayed-trigger creation, independent of event context.
    pub remembered: Vec<ObjectBinding>,
    pub source_identity: AbilitySource,
    pub targets: Vec<Target>,
    pub target_req_indices: Vec<usize>,
    pub requirement_index: usize,
    /// Definition/controller/presentation facts for that exact source.
    pub source_lki: Option<ObjectLki>,
    pub ability_index: usize,
    pub controller: PlayerId,
    pub id: TriggerId,
    pub ordering_class: TriggerOrderingClass,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub context: Option<Target>,
    /// The player the triggering event names ("they": the caster of the
    /// triggering spell), separate from its context object.
    pub player: Option<PlayerId>,
    /// Kicker choice carried by the triggering source's spell.
    pub kicked: bool,
    /// The ability's X: for an entering permanent's own trigger, the X of
    /// the spell that became it (CR 107.3m); for a counters or per-creature
    /// combat damage trigger, its event's amount ("that many"); otherwise 0.
    pub x: u8,
    /// For an entering permanent's own trigger, the mana spent to cast the
    /// spell that became it; otherwise empty.
    pub spent: crate::state::payment::PaymentReceipt,
    /// The attacking creature whose attack triggered its own ability.
    pub attacker: Option<crate::state::game_object::PermanentId>,
    /// The modes of a printed modal ability chosen so far (CR 700.2b).
    pub chosen_modes: Vec<usize>,
    /// Whether the modal choice is complete, so `chosen_modes` is final;
    /// distinguishes an "up to" ability that chose none from one not yet
    /// choosing.
    pub modes_chosen: bool,
    /// Non-printed triggers (delayed, reflexive, granted, firebending) carry
    /// their effects inline instead of referencing `Object::abilities`
    /// (`ability_index` is unused when set).
    pub inline_effects: Option<Vec<Effect>>,
}

/// A delayed triggered ability (CR 603.7) waiting for its event. It fires
/// with `effects` inline, from `source_card` (the exact `source_ref`, when
/// that is a battlefield object, supplies last known information).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct DelayedTrigger {
    /// Exact subjects captured at delayed-trigger creation, independent of event context.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub remembered: Vec<ObjectBinding>,
    pub event: DelayedTriggerEvent,
    pub controller: PlayerId,
    #[serde(flatten)]
    pub source_identity: AbilitySource,
    pub effects: DelayedEffects,
    pub frequency: DelayedFrequency,
    pub expiry: DelayedExpiry,
}

/// A delayed trigger's instructions, fixed when it is created, with their
/// learning tokens and text: every observation of the waiting trigger shows
/// that program, so it is projected once rather than per observation. Forks
/// share it. Serializes as the instruction list alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayedEffects(Arc<DelayedProgram>);

#[derive(Debug, PartialEq, Eq)]
struct DelayedProgram {
    effects: Vec<Effect>,
    projection: ProgramProjection,
}

impl DelayedEffects {
    pub fn new(effects: Vec<Effect>) -> Self {
        let projection = ProgramProjection::of(&effects);
        Self(Arc::new(DelayedProgram {
            effects,
            projection,
        }))
    }

    pub fn effects(&self) -> &[Effect] {
        &self.0.effects
    }

    pub(crate) fn projection(&self) -> &ProgramProjection {
        &self.0.projection
    }
}

impl serde::Serialize for DelayedEffects {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.effects.serialize(serializer)
    }
}

/// Event matching only; frequency and expiry belong to the waiting watcher.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedTriggerEvent {
    /// The exact object dies (or, `or_exiled`, is exiled: earthbend's "when
    /// it dies or is exiled"). Removed once it leaves the battlefield any
    /// way; the departed permanent is the trigger's context.
    Leaves {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        watched: ObjectId,
        or_exiled: bool,
    },
    /// "Whenever a creature dies".
    CreatureDies,
    /// "At the beginning of the end step", optionally only the controller's.
    EndStep {
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        your_turn: bool,
    },
    /// "When you cast a [predicate] spell"; that spell is the context.
    YouCastSpell(Box<CardPredicate>),
    /// "At the beginning of upkeep".
    Upkeep,
}

/// A card exiled with an exact `source` object (CR 607.2a). "Until [source]
/// leaves the battlefield" (Earth Kingdom Jailer): when it leaves, the
/// exiled card returns immediately — no trigger, no stack (CR 603.6e).
/// Otherwise ("exiled with this creature", Boiling Rock Rioter) the card
/// stays linked for abilities of the departed source. A later incarnation of the same physical
/// card is not the source.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ExileLink {
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub source: ObjectId,
    /// The appearance found by the exiling effect (CR 400.7j), not any
    /// later appearance of the same physical card.
    /// V16 recorded a row here, so its bytes keep the slot alone.
    #[serde(rename = "exiled_card")]
    pub exiled_object: ObjectId,
    /// The card returns as the source leaves the battlefield.
    #[serde(skip_serializing_if = "is_true")]
    pub returns: bool,
}

fn is_true(value: &bool) -> bool {
    *value
}

impl PendingTrigger {
    /// A newly fired occurrence, before its controller chooses modes or targets.
    pub fn new(
        id: TriggerId,
        controller: PlayerId,
        source_identity: AbilitySource,
        ability_index: usize,
    ) -> Self {
        Self {
            id,
            controller,
            source_identity,
            ability_index,
            source_lki: None,
            ordering_class: TriggerOrderingClass::Ordinary,
            inline_effects: None,
            program: false,
            remembered: Vec::new(),
            targets: Vec::new(),
            target_req_indices: Vec::new(),
            requirement_index: 0,
            chosen_modes: Vec::new(),
            modes_chosen: false,
            context: None,
            player: None,
            attacker: None,
            triggering_amount: 0,
            kicked: false,
            x: 0,
            spent: Default::default(),
        }
    }

    /// Targets already committed to the requirement being filled.
    pub(super) fn clause_targets(&self) -> impl Iterator<Item = Target> + Clone + '_ {
        self.targets
            .iter()
            .zip(&self.target_req_indices)
            .filter(|(_, index)| **index == self.requirement_index)
            .map(|(target, _)| *target)
    }

    pub fn source_object(&self) -> ObjectId {
        self.source_identity.object()
    }
    pub fn source_ref(&self) -> Option<ObjectId> {
        self.source_identity.reference()
    }
    pub fn current_clause_targets(&self) -> impl Iterator<Item = Target> + Clone + '_ {
        self.targets
            .iter()
            .zip(&self.target_req_indices)
            .filter(|(_, index)| **index == self.requirement_index)
            .map(|(target, _)| *target)
    }
}

impl DelayedTrigger {
    pub fn source_object(&self) -> ObjectId {
        self.source_identity.object()
    }
    pub fn source_ref(&self) -> Option<ObjectId> {
        self.source_identity.reference()
    }
}

// The pending record historically separated its two source projections with
// target progress. Preserve that field order in v16's streamed state bytes.
impl serde::Serialize for PendingTrigger {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        if self.program {
            map.serialize_entry("program", &self.program)?;
        }
        if !self.remembered.is_empty() {
            map.serialize_entry("remembered", &self.remembered)?;
        }
        map.serialize_entry("source_card", &self.source_identity.object())?;
        map.serialize_entry("targets", &self.targets)?;
        map.serialize_entry("target_req_indices", &self.target_req_indices)?;
        map.serialize_entry("requirement_index", &self.requirement_index)?;
        let source_ref = self.source_identity.reference();
        map.serialize_entry("source_ref", &crate::state::game_object::Exact(&source_ref))?;
        map.serialize_entry("source_lki", &self.source_lki)?;
        map.serialize_entry("ability_index", &self.ability_index)?;
        map.serialize_entry("controller", &self.controller)?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("ordering_class", &self.ordering_class)?;
        map.serialize_entry("context", &self.context)?;
        if self.player.is_some() {
            map.serialize_entry("player", &self.player)?;
        }
        if self.kicked {
            map.serialize_entry("kicked", &true)?;
        }
        if self.triggering_amount != 0 {
            map.serialize_entry("triggering_amount", &self.triggering_amount)?;
        }
        if self.x != 0 {
            map.serialize_entry("x", &self.x)?;
        }
        if !self.spent.is_empty() {
            map.serialize_entry("spent", &self.spent)?;
        }
        if self.attacker.is_some() {
            map.serialize_entry("attacker", &self.attacker)?;
        }
        if !self.chosen_modes.is_empty() {
            map.serialize_entry("chosen_modes", &self.chosen_modes)?;
        }
        if self.modes_chosen {
            map.serialize_entry("modes_chosen", &true)?;
        }
        map.serialize_entry("inline_effects", &self.inline_effects)?;
        map.end()
    }
}
