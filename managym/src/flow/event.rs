use crate::{
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventEntity {
    Card(CardId),
    Permanent(PermanentId),
    Player(PlayerId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameEvent {
    CardMoved {
        card: CardId,
        from: Option<ZoneType>,
        to: ZoneType,
        controller: PlayerId,
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
    AbilityTriggered {
        source_card: CardId,
        controller: PlayerId,
    },
    /// A player drew a card; `nth_this_turn` counts that player's draws
    /// this turn (resets each turn).
    CardDrawn {
        player: PlayerId,
        nth_this_turn: u32,
    },
    /// A permanent transitioned from untapped to tapped.
    PermanentTapped {
        permanent: PermanentId,
        for_mana: bool,
    },
    /// The declare-attackers turn-based action completed with one or more
    /// attackers (CR 508.1 — attack triggers fire on the whole batch).
    AttackersDeclared {
        player: PlayerId,
        attackers: Vec<PermanentId>,
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
