use super::{
    game_object::{CardId, PlayerId},
    target::Target,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackObject {
    Spell {
        card: CardId,
    },
    TriggeredAbility {
        source_card: CardId,
        ability_index: usize,
        controller: PlayerId,
        target: Option<Target>,
    },
}
