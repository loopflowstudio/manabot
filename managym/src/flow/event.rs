use crate::state::{
    game_object::{CardId, PlayerId},
    zone::ZoneType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameEvent {
    CardMoved {
        card: CardId,
        from: Option<ZoneType>,
        to: ZoneType,
        controller: PlayerId,
    },
}
