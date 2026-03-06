use super::game_object::{CardId, PermanentId, PlayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Target {
    Player(PlayerId),
    Permanent(PermanentId),
    StackSpell(CardId),
}
