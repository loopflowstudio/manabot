use crate::state::game_object::{CardId, PlayerId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingTrigger {
    pub source_card: CardId,
    pub ability_index: usize,
    pub controller: PlayerId,
    pub enqueue_order: u64,
}
