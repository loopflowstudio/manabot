use crate::state::{
    ability::Effect,
    game_object::{CardId, PlayerId, Target},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingTrigger {
    pub source_card: CardId,
    pub ability_index: usize,
    pub controller: PlayerId,
    pub enqueue_order: u64,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub context: Option<Target>,
    /// Delayed triggers carry their effects inline instead of referencing
    /// `Card::abilities` (`ability_index` is unused when set).
    pub inline_effects: Option<Vec<Effect>>,
}

/// A one-shot delayed triggered ability (CR 603.7) watching a specific
/// card's next departure from the battlefield. Removed once it fires or
/// once the watched permanent leaves in a non-matching way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayedTrigger {
    pub watched_card: CardId,
    pub controller: PlayerId,
    pub kind: DelayedTriggerKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DelayedTriggerKind {
    /// Earthbend: "When it dies or is exiled, return it to the battlefield
    /// tapped."
    ReturnToBattlefieldTapped,
}

/// "Exile [card] until [source] leaves the battlefield" linkage (Earth
/// Kingdom Jailer). When `source_card` leaves the battlefield the exiled
/// card returns immediately — no trigger, no stack (CR 603.6e).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExileLink {
    pub source_card: CardId,
    pub exiled_card: CardId,
}
