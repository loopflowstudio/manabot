use crate::state::game_object::{CardId, PlayerId, Target};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingTrigger {
    pub source_card: CardId,
    pub ability_index: usize,
    pub controller: PlayerId,
    pub enqueue_order: u64,
    /// Object from the triggering event that the ability's effects
    /// reference (e.g. the spell that targeted a warded permanent).
    pub context: Option<Target>,
}
