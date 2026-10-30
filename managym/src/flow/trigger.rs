//! Waiting trigger occurrences and the resumable, two-pass placement round.
//! Collection owns arrivals; placement owns a detached batch until every
//! controller has placed its abilities (CR 603.3b, 2026-09-25 rules snapshot).

use crate::state::{
    ability::Effect,
    game_object::{CardId, ObjectLki, ObjectRef, PlayerId, Target},
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PendingTrigger {
    pub source_card: CardId,
    pub targets: Vec<Target>,
    pub target_req_indices: Vec<usize>,
    pub requirement_index: usize,
    /// Exact source object captured when the ability triggered. Synthetic
    /// test/delayed triggers may have no battlefield source.
    pub source_ref: Option<ObjectRef>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerId>,
    /// The ability's X: for an entering permanent's own trigger, the X of
    /// the spell that became it (CR 107.3m); otherwise 0.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// The attacking creature whose attack triggered its own ability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attacker: Option<crate::state::game_object::PermanentId>,
    /// The modes of a "choose up to" ability chosen so far.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub chosen_modes: Vec<usize>,
    /// Delayed triggers carry their effects inline instead of referencing
    /// `Card::abilities` (`ability_index` is unused when set).
    pub inline_effects: Option<Vec<Effect>>,
}

/// A delayed triggered ability (CR 603.7) waiting for its event. It fires
/// with `effects` inline, from `source_card` (the exact `source_ref`, when
/// that is a battlefield object, supplies last known information).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct DelayedTrigger {
    pub event: DelayedTriggerEvent,
    pub controller: PlayerId,
    pub source_card: CardId,
    pub source_ref: Option<ObjectRef>,
    pub effects: Vec<Effect>,
    /// "this turn": removed in the cleanup step if it has not fired.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub this_turn: bool,
}

/// What a waiting delayed trigger watches for. All but `CreatureDies` fire
/// once (CR 603.7c).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedTriggerEvent {
    /// The exact object dies (or, `or_exiled`, is exiled: earthbend's "when
    /// it dies or is exiled"). Removed once it leaves the battlefield any
    /// way; the departed permanent is the trigger's context.
    Leaves { watched: ObjectRef, or_exiled: bool },
    /// "Whenever a creature dies".
    CreatureDies,
    /// "At the beginning of the next end step"; `context` is the trigger's
    /// context ("sacrifice that token").
    NextEndStep { context: Option<Target> },
    /// "When you next cast a [predicate] spell"; that spell is the context.
    YouCastSpell(CardPredicate),
}

/// "Exile [card] until [source] leaves the battlefield" linkage (Earth
/// Kingdom Jailer). When the exact `source` object leaves the battlefield the
/// exiled card returns immediately — no trigger, no stack (CR 603.6e). A later
/// incarnation of the same physical card is not the duration source.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ExileLink {
    pub source: ObjectRef,
    pub exiled_card: CardId,
}
