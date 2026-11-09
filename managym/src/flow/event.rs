use std::sync::Arc;

use crate::{
    flow::turn::StepKind,
    state::{
        card::CardDefId,
        game_object::{CardId, Incarnation, ObjectId, PermanentId, PlayerId, Target},
        permanent::CounterKind,
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
    PermanentSacrificed {
        card: CardId,
        controller: PlayerId,
    },
    /// Public definition reveal; outside/hidden copies have no public object identity.
    CardRevealed {
        owner: PlayerId,
        definition: CardDefId,
    },
    CardMoved {
        card: CardId,
        from: Option<ZoneType>,
        to: ZoneType,
        controller: PlayerId,
        /// The card arrived face down: its identity is not public.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        face_down: bool,
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
    /// An activated or triggered ability was countered (CR 701.6b): its
    /// stack id, the card it came from, its controller and the countering
    /// source.
    AbilityCountered {
        ability: ObjectId,
        source_card: CardId,
        controller: PlayerId,
        by: Option<CardId>,
    },
    /// A triggered ability was put on the stack as `ability`; `attacker` is
    /// the attacking creature whose attack triggered its own ability.
    AbilityTriggered {
        source_card: CardId,
        controller: PlayerId,
        ability: ObjectId,
        #[serde(skip_serializing_if = "Option::is_none")]
        attacker: Option<PermanentId>,
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
    /// A permanent became the target of a spell or ability (ward, CR
    /// 702.21): `by` is a `Target::StackSpell` or `Target::StackAbility`.
    PermanentTargeted {
        permanent: PermanentId,
        by: Target,
        by_controller: PlayerId,
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
    /// `by` put `count` (> 0) `kind` counters on a permanent (CR 122.6),
    /// leaving `total` of that kind on it.
    CountersPut {
        permanent: PermanentId,
        kind: CounterKind,
        count: i32,
        total: i32,
        by: PlayerId,
    },
    /// A player bent (waterbend, earthbend, firebend or airbend).
    Bent {
        player: PlayerId,
        kind: crate::flow::turn::Bending,
    },
    /// A player scried (CR 701.22).
    Scried {
        player: PlayerId,
    },
    /// A player finished surveilling, even with an empty library (CR 701.25).
    Surveilled {
        player: PlayerId,
    },
    /// A player searched their library (CR 701.23), whatever it found.
    LibrarySearched {
        player: PlayerId,
    },
    /// The creatures that dealt combat damage to a player in one combat
    /// damage step (CR 510.2).
    CombatDamageToPlayers {
        creatures: Vec<PermanentId>,
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

/// The variant of a `GameEvent`, which trigger conditions are indexed by
/// (`flow/triggers.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    PermanentSacrificed,
    CardRevealed,
    CardMoved,
    DamageDealt,
    LifeChanged,
    SpellCast,
    SpellResolved,
    SpellCountered,
    AbilityCountered,
    AbilityTriggered,
    CardDrawn,
    PermanentTapped,
    PermanentTargeted,
    AttackersDeclared,
    CombatAttackersDeclared,
    BlockersDeclared,
    CombatDamageDealt,
    CountersPut,
    Bent,
    Scried,
    LibrarySearched,
    CombatDamageToPlayers,
    PermanentsDied,
    TurnStarted,
    StepStarted,
    Surveilled,
}

impl GameEvent {
    pub fn kind(&self) -> EventKind {
        match self {
            GameEvent::PermanentSacrificed { .. } => EventKind::PermanentSacrificed,
            GameEvent::CardRevealed { .. } => EventKind::CardRevealed,
            GameEvent::CardMoved { .. } => EventKind::CardMoved,
            GameEvent::DamageDealt { .. } => EventKind::DamageDealt,
            GameEvent::LifeChanged { .. } => EventKind::LifeChanged,
            GameEvent::SpellCast { .. } => EventKind::SpellCast,
            GameEvent::SpellResolved { .. } => EventKind::SpellResolved,
            GameEvent::SpellCountered { .. } => EventKind::SpellCountered,
            GameEvent::AbilityCountered { .. } => EventKind::AbilityCountered,
            GameEvent::AbilityTriggered { .. } => EventKind::AbilityTriggered,
            GameEvent::CardDrawn { .. } => EventKind::CardDrawn,
            GameEvent::PermanentTapped { .. } => EventKind::PermanentTapped,
            GameEvent::PermanentTargeted { .. } => EventKind::PermanentTargeted,
            GameEvent::AttackersDeclared { .. } => EventKind::AttackersDeclared,
            GameEvent::CombatAttackersDeclared { .. } => EventKind::CombatAttackersDeclared,
            GameEvent::BlockersDeclared { .. } => EventKind::BlockersDeclared,
            GameEvent::CombatDamageDealt { .. } => EventKind::CombatDamageDealt,
            GameEvent::CountersPut { .. } => EventKind::CountersPut,
            GameEvent::Bent { .. } => EventKind::Bent,
            GameEvent::Scried { .. } => EventKind::Scried,
            GameEvent::Surveilled { .. } => EventKind::Surveilled,
            GameEvent::LibrarySearched { .. } => EventKind::LibrarySearched,
            GameEvent::CombatDamageToPlayers { .. } => EventKind::CombatDamageToPlayers,
            GameEvent::PermanentsDied { .. } => EventKind::PermanentsDied,
            GameEvent::TurnStarted { .. } => EventKind::TurnStarted,
            GameEvent::StepStarted { .. } => EventKind::StepStarted,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DamageTarget {
    Player(PlayerId),
    Permanent(PermanentId),
}

/// The committed rules-event ledger, in emission order.
///
/// Every `SEGMENT` events the unsealed tail is sealed into an immutable,
/// reference-counted segment, so a fork shares the game's history and copies
/// at most one partial segment: cloning a long game no longer costs its past.
/// Readers see one ordered sequence; `Index`, `iter` and `len` cover the
/// whole ledger, and `take` drains it like the `Vec` it replaces.
#[derive(Clone, Debug, Default)]
pub struct EventLedger {
    sealed: Option<Arc<Segment>>,
    tail: Vec<GameEvent>,
}

#[derive(Debug)]
struct Segment {
    earlier: Option<Arc<Segment>>,
    events: Vec<GameEvent>,
    /// Events in this segment and every earlier one.
    len: usize,
}

const SEGMENT: usize = 32;

impl EventLedger {
    pub fn push(&mut self, event: GameEvent) {
        if self.tail.len() == SEGMENT {
            let events = std::mem::replace(&mut self.tail, Vec::with_capacity(SEGMENT));
            let len = self.sealed_len() + events.len();
            self.sealed = Some(Arc::new(Segment {
                earlier: self.sealed.take(),
                events,
                len,
            }));
        }
        self.tail.push(event);
    }

    fn sealed_len(&self) -> usize {
        self.sealed.as_ref().map_or(0, |segment| segment.len)
    }

    pub fn len(&self) -> usize {
        self.sealed_len() + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sealed segments, oldest first.
    fn segments(&self) -> Vec<&[GameEvent]> {
        let mut segments = Vec::new();
        let mut cursor = self.sealed.as_deref();
        while let Some(segment) = cursor {
            segments.push(segment.events.as_slice());
            cursor = segment.earlier.as_deref();
        }
        segments.reverse();
        segments
    }

    pub fn iter(&self) -> impl Iterator<Item = &GameEvent> {
        self.segments()
            .into_iter()
            .flatten()
            .chain(self.tail.iter())
    }

    pub fn get(&self, index: usize) -> Option<&GameEvent> {
        let sealed_len = self.sealed_len();
        if index >= sealed_len {
            return self.tail.get(index - sealed_len);
        }
        let mut cursor = self.sealed.as_deref();
        while let Some(segment) = cursor {
            let start = segment.len - segment.events.len();
            if index >= start {
                return segment.events.get(index - start);
            }
            cursor = segment.earlier.as_deref();
        }
        None
    }

    /// Remove and return every event, oldest first.
    pub fn take(&mut self) -> Vec<GameEvent> {
        let events = self.iter().cloned().collect();
        self.clear();
        events
    }

    pub fn clear(&mut self) {
        self.sealed = None;
        self.tail.clear();
    }
}

impl std::ops::Index<usize> for EventLedger {
    type Output = GameEvent;

    fn index(&self, index: usize) -> &GameEvent {
        self.get(index).expect("event index within the ledger")
    }
}

/// Equal ledgers hold the same events, however they are segmented.
impl PartialEq for EventLedger {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for EventLedger {}

impl PartialEq<Vec<GameEvent>> for EventLedger {
    fn eq(&self, other: &Vec<GameEvent>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

/// Serializes as the plain event sequence, so witnesses and hashes read
/// exactly what the `Vec` they replace produced.
impl serde::Serialize for EventLedger {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn life(new: i32) -> GameEvent {
        GameEvent::LifeChanged {
            player: PlayerId(0),
            old: new + 1,
            new,
        }
    }

    #[test]
    fn ledger_reads_as_one_sequence_across_sealed_segments_and_forks() {
        let mut ledger = EventLedger::default();
        let count = 3 * SEGMENT + 5;
        for index in 0..count {
            ledger.push(life(index as i32));
        }
        let expected: Vec<_> = (0..count as i32).map(life).collect();
        assert_eq!(ledger.len(), count);
        assert_eq!(ledger, expected);
        assert!(ledger.iter().eq(expected.iter()));
        for (index, event) in expected.iter().enumerate() {
            assert_eq!(&ledger[index], event);
        }
        assert_eq!(ledger.get(count), None);
        assert_eq!(
            serde_json::to_value(&ledger).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );

        // A fork shares the sealed history and diverges independently.
        let mut fork = ledger.clone();
        fork.push(life(-1));
        ledger.push(life(-2));
        assert_eq!(ledger.len(), count + 1);
        assert_eq!(fork[count], life(-1));
        assert_eq!(ledger[count], life(-2));
        assert_ne!(fork, ledger);
        assert!(fork.iter().take(count).eq(expected.iter()));

        let drained = ledger.take();
        assert_eq!(drained.len(), count + 1);
        assert_eq!(drained[..count], expected[..]);
        assert!(ledger.is_empty());
        assert_eq!(fork.len(), count + 1, "draining one fork leaves the other");
    }
}
