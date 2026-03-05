use crate::{
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameEvent {
    CardMoved {
        card: CardId,
        from: ZoneType,
        to: ZoneType,
    },
    DamageDealt {
        source: Option<CardId>,
        target: DamageTarget,
        amount: u32,
    },
    LifeChanged {
        player: PlayerId,
        old: i32,
        new: i32,
    },
    SpellCast {
        card: CardId,
        target: Option<Target>,
    },
    SpellResolved {
        card: CardId,
    },
    SpellCountered {
        card: CardId,
        by: Option<CardId>,
    },
    TurnStarted {
        player: PlayerId,
    },
    StepStarted {
        step: StepKind,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DamageTarget {
    Player(PlayerId),
    Permanent(PermanentId),
}
