use crate::{
    flow::turn::StepKind,
    state::{
        game_object::{CardId, Incarnation, ObjectId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

/// Viewer-safe exact identity captured while a rules object still exists.
///
/// This is committed domain-event data, not a presentation queue: the
/// experience adapter decides which typed `PresentationEvent` to project.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ObjectEventRef {
    pub entity: ObjectId,
    pub incarnation: Incarnation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EventSubject {
    Object(ObjectEventRef),
    Player(PlayerId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EventEntity {
    Card(CardId),
    Permanent(PermanentId),
    Player(PlayerId),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
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
        targets: Vec<Target>,
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
    /// A permanent became the target of a spell (ward, CR 702.21).
    PermanentTargeted {
        permanent: PermanentId,
        spell: CardId,
        spell_controller: PlayerId,
    },
    /// The declare-attackers turn-based action completed with one or more
    /// attackers (CR 508.1 — attack triggers fire on the whole batch).
    AttackersDeclared {
        player: PlayerId,
        attackers: Vec<PermanentId>,
    },
    /// Viewer-safe attacker identities captured at declaration completion.
    CombatAttackersDeclared {
        player: PlayerId,
        defender: PlayerId,
        attackers: Vec<ObjectEventRef>,
    },
    /// Final legal assignments after the complete blocker declaration loop.
    BlockersDeclared {
        assignments: Vec<(ObjectEventRef, Vec<ObjectEventRef>)>,
    },
    /// Combat damage in native assignment order.
    CombatDamageDealt {
        source: ObjectEventRef,
        target: EventSubject,
        amount: u32,
    },
    /// One simultaneous state-based death batch in deterministic object order.
    PermanentsDied {
        objects: Vec<ObjectEventRef>,
    },
    TurnStarted {
        player: PlayerId,
        turn_number: u32,
    },
    StepStarted {
        step: StepKind,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DamageTarget {
    Player(PlayerId),
    Permanent(PermanentId),
}
