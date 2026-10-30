//! Waiting trigger occurrences and the resumable, two-pass placement round.
//! Collection owns arrivals; placement owns a detached batch until every
//! controller has placed its abilities (CR 603.3b, 2026-09-25 rules snapshot).

use crate::state::{
    ability::Effect,
    game_object::{CardId, ObjectLki, ObjectRef, PlayerId, Target},
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
    /// Delayed triggers carry their effects inline instead of referencing
    /// `Card::abilities` (`ability_index` is unused when set).
    pub inline_effects: Option<Vec<Effect>>,
}

/// A one-shot delayed triggered ability (CR 603.7) watching a specific
/// card's next departure from the battlefield. Removed once it fires or
/// once the watched permanent leaves in a non-matching way.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct DelayedTrigger {
    pub watched: ObjectRef,
    pub controller: PlayerId,
    pub kind: DelayedTriggerKind,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedTriggerKind {
    /// Earthbend: "When it dies or is exiled, return it to the battlefield
    /// tapped."
    ReturnToBattlefieldTapped,
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
