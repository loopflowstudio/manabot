//! Typed proposed-event, replacement/prevention, and commit pipeline.
//!
//! `GameEvent` remains the factual committed-event stream consumed by
//! triggers, observations, and presentation. A batch retains proposals until
//! every replacement has been chosen, then commits them under one pre-event
//! source snapshot and returns its results with the consequences its caller
//! runs next.

use crate::state::{
    ability::Actor,
    card::{DamageTransform, Keyword},
};
use crate::{
    flow::{event::GameEvent, game::Game},
    state::card::StaticAbility,
    state::{
        card::{CardDefId, ReplacementEffect},
        effect::{ContinuousEffect, Modification},
        game_object::{ObjectId, ObjectLki, PermanentId, PlayerId, PublicId},
        permanent::{CounterKind, Counters},
        zone::ZoneType,
    },
};

/// How a card enters its new zone.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ZoneEntry {
    /// The resolving Adventure proposes a graveyard move. Only its own
    /// replacement transfers the right to cast to the exile successor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolving_adventure: Option<PlayerId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adventure_player: Option<PlayerId>,
    #[serde(
        rename = "cast_adventure",
        skip_serializing_if = "crate::state::card::SpellProfile::is_front"
    )]
    pub profile: crate::state::card::SpellProfile,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub prepared: bool,
    /// Battlefield: enters tapped.
    pub tapped: bool,
    /// Battlefield or exile: enters with these counters ("exile it with a
    /// stash counter on it").
    pub counters: Counters,
    /// Battlefield: the announced X of the resolving spell becoming this
    /// permanent.
    pub x: u8,
    /// Battlefield: enters under this player's control; `None` is its owner.
    /// A resolving permanent spell enters under its controller (CR 110.2).
    pub controller: Option<PlayerId>,
    /// Exile: enters face down (foretell, CR 702.143a), so the move is
    /// secret from everyone but its owner (CR 406.3).
    pub face_down: bool,
    /// Battlefield: enters with its back face up (CR 712.14).
    pub transformed: bool,
    /// CR 614.1c, 614.12: this entry owns the prospective copy already on
    /// the row. It is an instruction for the new object, not old copy state
    /// carried across a zone change. The row already records its values in
    /// v16, and `PendingEvent::copy_chosen` records the answered choice.
    #[serde(skip)]
    pub entry_copy: bool,
    /// The instruction removes the publicly identified top position.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub library_top_departure: bool,
    /// The owner selected this departure from a privately inspected prefix.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub inspected_library_departure: bool,
}

impl ZoneEntry {
    /// The terms `zone` takes: battlefield terms only for the battlefield;
    /// face down and counters for exile.
    pub(crate) fn for_zone(&self, zone: ZoneType) -> Self {
        let mut entry = match zone {
            ZoneType::Battlefield => Self {
                face_down: false,
                ..self.clone()
            },
            ZoneType::Stack => Self {
                profile: self.profile,
                ..Default::default()
            },
            ZoneType::Graveyard => Self {
                resolving_adventure: self.resolving_adventure,
                ..Default::default()
            },
            ZoneType::Exile => Self {
                adventure_player: self.adventure_player,
                face_down: self.face_down,
                counters: self.counters.clone(),
                ..Self::default()
            },
            _ => Self::default(),
        };
        entry.inspected_library_departure = self.inspected_library_departure;
        entry.library_top_departure = self.library_top_departure;
        entry
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum ProposedDamageTarget {
    Player(PlayerId),
    Object(#[serde(serialize_with = "crate::state::game_object::exact")] ObjectId),
}

/// Initial cause is retained even after a replacement makes the placement an
/// effect for CR 614.16. Payment/turn completion still belongs to its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum EventCause {
    Effect,
    Cost,
    TurnAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct CounterProvenance {
    pub cause: EventCause,
    pub replaced: bool,
}

impl CounterProvenance {
    pub(crate) const fn new(cause: EventCause) -> Self {
        Self {
            cause,
            replaced: false,
        }
    }

    fn is_effect(self) -> bool {
        self.cause == EventCause::Effect || self.replaced
    }
}

/// Copiable values are captured before a creation can suspend. No physical
/// token or identity exists until creation replacements have finished.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum TokenSource {
    /// The token definition with this `token.*` key.
    Registered(String),
    Copy {
        definition_id: crate::state::card::CardDefId,
        /// The copied object as it stood; the token takes its `printed`.
        definition: crate::state::card::ObjectDefinition,
        front: Option<(
            crate::state::card::CardDefId,
            std::sync::Arc<crate::state::card::PrintedCharacteristics>,
        )>,
        copy: Option<crate::state::effect::CopyEffect>,
        exceptions: Vec<crate::state::effect::Modification>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProposedEvent {
    Damage {
        source: Option<ObjectId>,
        target: ProposedDamageTarget,
        amount: i32,
        combat: bool,
    },
    LifeChange {
        player: PlayerId,
        delta: i32,
    },
    Destroy {
        target: ObjectId,
    },
    Sacrifice {
        target: ObjectId,
    },
    CreateTokens {
        definition: Box<TokenSource>,
        controller: PlayerId,
        count: i32,
        entry: ZoneEntry,
    },
    CounterChange {
        provenance: CounterProvenance,
        target: ObjectId,
        kind: CounterKind,
        delta: i32,
        by: Option<PlayerId>,
    },
    ZoneMove {
        object_id: ObjectId,
        from: Option<ZoneType>,
        to: ZoneType,
        entry: ZoneEntry,
    },
}

// V16 records the former bare row and optional exact identity. Project both
// from the proposal's one identity and source zone; they cannot disagree.
impl serde::Serialize for ProposedEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStructVariant;
        match self {
            Self::Damage {
                source,
                target,
                amount,
                combat,
            } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 0, "Damage", 4)?;
                variant.serialize_field("source", &RecordedExact(source))?;
                variant.serialize_field("target", target)?;
                variant.serialize_field("amount", amount)?;
                variant.serialize_field("combat", combat)?;
                variant.end()
            }
            Self::LifeChange { player, delta } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 1, "LifeChange", 2)?;
                variant.serialize_field("player", player)?;
                variant.serialize_field("delta", delta)?;
                variant.end()
            }
            Self::Destroy { target } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 2, "Destroy", 1)?;
                variant.serialize_field("target", &RecordedExact(target))?;
                variant.end()
            }
            Self::Sacrifice { target } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 3, "Sacrifice", 1)?;
                variant.serialize_field("target", &RecordedExact(target))?;
                variant.end()
            }
            Self::CreateTokens {
                definition,
                controller,
                count,
                entry,
            } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 4, "CreateTokens", 4)?;
                variant.serialize_field("definition", definition)?;
                variant.serialize_field("controller", controller)?;
                variant.serialize_field("count", count)?;
                variant.serialize_field("entry", entry)?;
                variant.end()
            }
            Self::CounterChange {
                provenance,
                target,
                kind,
                delta,
                by,
            } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 5, "CounterChange", 5)?;
                variant.serialize_field("provenance", provenance)?;
                variant.serialize_field("target", &RecordedExact(target))?;
                variant.serialize_field("kind", kind)?;
                variant.serialize_field("delta", delta)?;
                variant.serialize_field("by", by)?;
                variant.end()
            }
            Self::ZoneMove {
                object_id,
                from,
                to,
                entry,
            } => {
                let mut variant =
                    serializer.serialize_struct_variant("ProposedEvent", 6, "ZoneMove", 5)?;
                variant.serialize_field("card", object_id)?;
                variant.serialize_field("object", &RecordedExact(&from.map(|_| *object_id)))?;
                variant.serialize_field("from", from)?;
                variant.serialize_field("to", to)?;
                variant.serialize_field("entry", entry)?;
                variant.end()
            }
        }
    }
}

struct RecordedExact<'a, T>(&'a T);
impl<T: crate::state::game_object::ExactIds> serde::Serialize for RecordedExact<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        crate::state::game_object::exact(self.0, serializer)
    }
}

/// The definition namespace is part of identity: a static and a replacement
/// at index zero on the same object are different effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ReplacementId {
    #[serde(serialize_with = "crate::state::game_object::exact")]
    pub(crate) source: ObjectId,
    pub(crate) definition_index: usize,
    pub(crate) static_ability: bool,
    /// The timestamp of the resolved continuous effect that offers it
    /// (`Fleeting Flight`'s shield); absent for a printed replacement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effect: Option<PublicId>,
}

/// One replacement the affected player may apply. A printed replacement's
/// identity is its source object and definition index; a resolved
/// continuous effect's (`Fleeting Flight`'s shield) is the object it
/// protects and the effect's timestamp, so two shields, or a shield and the
/// creature's own printed replacement, stay distinct.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ReplacementCandidate {
    identity: ReplacementId,
    pub(crate) effect: ReplacementEffect,
    /// The spell or ability that created a resolved effect, kept after it
    /// left (last-known information); absent for a printed replacement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_definition: Option<CardDefId>,
}

impl ReplacementCandidate {
    /// A replacement printed at `definition_index` of the source's
    /// `replacement_effects`.
    fn printed(source: ObjectId, definition_index: usize, effect: ReplacementEffect) -> Self {
        Self {
            identity: ReplacementId {
                source,
                definition_index,
                static_ability: false,
                effect: None,
            },
            effect,
            source_definition: None,
        }
    }

    /// A replacement derived from the source's static ability at
    /// `definition_index`.
    fn printed_static(
        source: ObjectId,
        definition_index: usize,
        effect: ReplacementEffect,
    ) -> Self {
        let mut candidate = Self::printed(source, definition_index, effect);
        candidate.identity.static_ability = true;
        candidate
    }

    /// The prevention `record` places on `object`, named for its source.
    fn shield(object: ObjectId, record: &ContinuousEffect) -> Self {
        Self {
            identity: ReplacementId {
                source: object,
                definition_index: 0,
                static_ability: false,
                effect: Some(record.timestamp),
            },
            effect: ReplacementEffect::PreventCombatDamage,
            source_definition: record.source,
        }
    }

    pub(crate) fn source(&self) -> ObjectId {
        self.identity.source
    }

    pub(crate) fn description(&self) -> String {
        match &self.effect {
            ReplacementEffect::ExileInsteadOfGraveyard => {
                "Exile without Adventure permission".into()
            }
            ReplacementEffect::AdventureExile => "Exile on Adventure".into(),
            ReplacementEffect::EntersPrepared => "Enter prepared".into(),
            ReplacementEffect::PreventCombatDamageToOrFromSelf
            | ReplacementEffect::PreventCombatDamage => "Prevent all combat damage".into(),
            ReplacementEffect::ModifyDamage {
                transform: DamageTransform::Prevent(amount),
                ..
            } => {
                format!("Prevent {amount} damage")
            }
            ReplacementEffect::ModifyDamage {
                transform: DamageTransform::Double,
                ..
            } => "Double the damage".into(),
            ReplacementEffect::EntersTapped
            | ReplacementEffect::EntersTappedUnlessControls { .. } => "Enter tapped".into(),
            ReplacementEffect::EntersWithPlusOneCounters { count, .. } => {
                format!("Enter with {count} +1/+1 counters")
            }
            ReplacementEffect::EntersWithXCounters { kind } => {
                format!("Enter with X {kind:?} counters")
            }
            ReplacementEffect::EntersWithCounters { kind, count } => {
                format!("Enter with {count} {kind:?} counters")
            }
            ReplacementEffect::OpponentDamagePlusCounters { kind } => {
                format!("Increase damage by this source's {kind:?} counters")
            }
            ReplacementEffect::UnspentManaBecomesRed => "Unspent mana becomes red".into(),
            ReplacementEffect::MultiplyTokens { multiplier } => {
                format!("Create {multiplier} times as many tokens")
            }
            ReplacementEffect::ModifyCounters {
                multiplier,
                additional,
                ..
            } => format!("Place {multiplier} times the counters, plus {additional}"),
            ReplacementEffect::EntersAsCopy { .. } => "Enter as a copy".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct PendingEvent {
    pub event: ProposedEvent,
    /// Index of the original proposal; a creation's token entries keep its ordinal.
    pub(crate) ordinal: usize,
    pub(crate) sacrifice: Option<(ObjectId, PlayerId)>,
    pub(crate) applied: Vec<ReplacementId>,
    /// Its "enter as a copy" choice has been made (or had no candidate).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) copy_chosen: bool,
}

/// A logical simultaneous event: its current proposals in replacement order,
/// the cursor of the next proposal still to be checked, and how the owning
/// instruction consumes the committed results. No proposal is committed while
/// another member still needs a replacement choice. Nothing on the board
/// changes between proposal and commit, so pre-event trigger sources and last
/// known information are read as the batch commits rather than retained.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct EventBatch {
    pub events: Vec<PendingEvent>,
    pub cursor: usize,
    pub binding: EventBinding,
}

impl EventBatch {
    /// Choosing replacements for an entry exposes the entering objects. Their
    /// current hidden-zone membership remains fixed until the batch commits.
    pub(crate) fn public_entries(&self, out: &mut Vec<ObjectId>) {
        for pending in &self.events {
            if let ProposedEvent::ZoneMove {
                object_id: object,
                from: Some(_),
                to: ZoneType::Battlefield,
                ..
            } = pending.event
            {
                out.push(object);
            }
        }
    }
}

/// Which result of the interrupted instruction its frame consumes. Consequent
/// child batches never replace these results with their own moves.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum EventBinding {
    None,
    Moved {
        #[serde(serialize_with = "crate::state::game_object::exact")]
        linked: Option<(ObjectId, bool)>,
        shuffle: bool,
        library_end: Option<crate::state::ability::LibraryEnd>,
        reveal: bool,
    },
    ThisWay,
    /// A Destroy instruction, including when its zone moves end in exile.
    Destroyed,
    LifeLost,
    Damage {
        result: Option<usize>,
    },
    Deaths,
    Combat,
    PlaceInLibrary {
        second_from_top: bool,
    },
    Shuffle {
        player: PlayerId,
    },
    Bending {
        player: PlayerId,
        kind: crate::flow::turn::Bending,
    },
    Created {
        attacking: bool,
    },
}

/// One member's outcome, in the order of the batch's events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EventResult {
    pub committed: bool,
    /// Signed difference of the committed life totals for a LifeChange member.
    pub life_delta: i32,
    pub moved: Option<ObjectId>,
    pub excess_damage: Option<i32>,
    pub combat_damage: Option<(ObjectId, u32)>,
    pub combat_hit: Option<crate::flow::event::CombatPlayerHit>,
    /// Life this damage member gains through lifelink; its batch commits it.
    pub lifelink: Option<crate::flow::damage::LifelinkGain>,
}

pub(crate) enum BatchProgress {
    Choice {
        player: PlayerId,
        candidates: Vec<ReplacementCandidate>,
    },
    /// The member at the cursor may enter as a copy of one of `candidates`
    /// (CR 707.5); its entering controller chooses before any other entry
    /// replacement applies (CR 616.1c).
    EntryCopy {
        player: PlayerId,
        object: ObjectId,
        candidates: Vec<PermanentId>,
    },
    /// Every member committed. Cards returning as a linked exile ends are
    /// consequences of the committed event: a new simultaneous event the
    /// caller runs next, which never replaces these results.
    Complete {
        results: Vec<EventResult>,
        followups: Vec<ProposedEvent>,
    },
}

impl Game {
    pub(crate) fn event_batch(
        &self,
        binding: EventBinding,
        events: impl IntoIterator<Item = ProposedEvent>,
    ) -> EventBatch {
        EventBatch {
            events: events
                .into_iter()
                .enumerate()
                .filter_map(|(ordinal, event)| self.pending_event(ordinal, event))
                .collect(),
            cursor: 0,
            binding,
        }
    }

    /// A current proposal ready for replacement. Destruction and sacrifice
    /// become the graveyard move they cause (an indestructible permanent is
    /// not destroyed), and a death the rules redirect is proposed as exile.
    fn pending_event(&self, ordinal: usize, mut event: ProposedEvent) -> Option<PendingEvent> {
        if !self.proposal_is_current(&event) {
            return None;
        }
        let mut sacrifice = None;
        if let ProposedEvent::Destroy { target } | ProposedEvent::Sacrifice { target } = &event {
            let id = self.lookup_current_permanent(*target).ok()?;
            let permanent = self.state.permanents[id].as_ref()?;
            if matches!(event, ProposedEvent::Destroy { .. })
                && self
                    .characteristics(id)
                    .keywords
                    .has(Keyword::Indestructible)
            {
                return None;
            }
            if matches!(event, ProposedEvent::Sacrifice { .. }) {
                sacrifice = Some((permanent.object, permanent.controller));
            }
            event = ProposedEvent::ZoneMove {
                object_id: permanent.object,
                from: Some(ZoneType::Battlefield),
                to: ZoneType::Graveyard,
                entry: ZoneEntry::default(),
            };
        }
        self.normalize_proposal(&mut event);
        Some(PendingEvent {
            event,
            ordinal,
            sacrifice,
            applied: Vec::new(),
            copy_chosen: false,
        })
    }

    fn normalize_proposal(&self, event: &mut ProposedEvent) {
        // CR 614.1a: replace any death of the exact affected incarnation,
        // including sacrifice and zero toughness, not only lethal damage.
        if let ProposedEvent::ZoneMove {
            object_id: object,
            from: Some(ZoneType::Battlefield),
            to,
            ..
        } = event
        {
            // CR 702.84a: an unearthed permanent leaving for anywhere but
            // exile is exiled instead.
            if *to != ZoneType::Exile
                && *to != ZoneType::Battlefield
                && self
                    .lookup_current_permanent(*object)
                    .is_ok_and(|id| self.exiled_if_it_would_leave(id))
            {
                *to = ZoneType::Exile;
                return;
            }
            if *to == ZoneType::Graveyard
                && self
                    .lookup_current_permanent(*object)
                    .ok()
                    .is_some_and(|id| {
                        // A finality counter (CR 122.1h) or an effect.
                        self.exiled_if_it_would_die(id)
                            || self.state.permanents[id].as_ref().is_some_and(|permanent| {
                                permanent.counter_count(CounterKind::Finality) > 0
                            })
                    })
            {
                *to = ZoneType::Exile;
            }
        }
    }

    pub(crate) fn advance_event_batch(&mut self, batch: &mut EventBatch) -> BatchProgress {
        // Earlier instructions and paid tap/life costs happened before this
        // proposal. Match them while their sources still exist; committing a
        // sacrifice must not erase the source's already-earned tap trigger.
        self.process_game_events();
        while batch.cursor < batch.events.len() {
            if let Some(choice) = self.entry_copy_choice(&mut batch.events[batch.cursor]) {
                return choice;
            }
            let pending = &batch.events[batch.cursor];
            let candidates: Vec<_> = self
                .collect_replacements(&pending.event)
                .into_iter()
                .filter(|candidate| !pending.applied.contains(&candidate.identity))
                .collect();
            match candidates.as_slice() {
                // Token entry is nested within creation (CR 616.1g): once the
                // creation's replacements finish, the final group is allocated
                // and takes the creation's place as ordinary simultaneous
                // entries, each checked for its own replacements before any
                // member commits.
                [] if matches!(pending.event, ProposedEvent::CreateTokens { .. }) => {
                    let PendingEvent {
                        event:
                            ProposedEvent::CreateTokens {
                                definition,
                                controller,
                                count,
                                entry,
                            },
                        ordinal,
                        ..
                    } = batch.events.remove(batch.cursor)
                    else {
                        unreachable!("matched creation")
                    };
                    let entries = self.allocate_tokens(*definition, controller, count, entry);
                    batch.events.splice(
                        batch.cursor..batch.cursor,
                        entries
                            .into_iter()
                            .filter_map(|event| self.pending_event(ordinal, event)),
                    );
                }
                [] => batch.cursor += 1,
                [candidate] => self.apply_replacement(&mut batch.events[batch.cursor], candidate),
                _ => {
                    return BatchProgress::Choice {
                        player: self.affected_player(&pending.event),
                        candidates,
                    }
                }
            }
        }
        // Every member is ready. The board has not changed since the
        // proposals were made, so leave-the-battlefield abilities and last
        // known information are read now, before any member commits, as the
        // permanents all were before the event (CR 603.10a).
        let departing: Vec<Option<ObjectLki>> = batch
            .events
            .iter()
            .map(|pending| match &pending.event {
                ProposedEvent::ZoneMove {
                    object_id,
                    from: Some(ZoneType::Battlefield),
                    ..
                } => self.snapshot_current_permanent(*object_id),
                _ => None,
            })
            .collect();
        let sources = if departing.iter().any(Option::is_some) {
            self.snapshot_trigger_sources()
        } else {
            Vec::new()
        };
        let mut results = Vec::with_capacity(batch.events.len());
        let mut followups = Vec::new();
        let mut died = Vec::new();
        let mut hits = Vec::new();
        let mut totals: Vec<(ObjectId, u32)> = Vec::new();
        // Sacrifice is committed with this batch, after all replacement
        // choices. Match its subjects while every pre-batch permanent still
        // exists: moving a Food first would erase the predicate it satisfied.
        // Trigger placement still waits for the entire resolution to finish.
        if batch
            .events
            .iter()
            .any(|pending| pending.sacrifice.is_some())
        {
            let mut departed = std::collections::BTreeSet::new();
            let mut sacrifices = false;
            for pending in &batch.events {
                if !self.proposal_is_current(&pending.event) {
                    continue;
                }
                if let ProposedEvent::ZoneMove { object_id, .. } = &pending.event {
                    if !departed.insert(*object_id) {
                        continue;
                    }
                }
                if let Some((object_id, controller)) = pending.sacrifice {
                    self.state.turn.history[controller.0].permanents_sacrificed += 1;
                    self.emit(GameEvent::PermanentSacrificed {
                        object_id,
                        controller,
                    });
                    sacrifices = true;
                }
            }
            if sacrifices {
                self.process_event_batch(&sources);
            }
        }
        for (pending, departing_lki) in batch.events.iter().zip(departing) {
            let event = &pending.event;
            if !self.proposal_is_current(event) {
                results.push(EventResult::default());
                continue;
            }
            let death = match (&batch.binding, event) {
                (
                    EventBinding::Deaths,
                    ProposedEvent::ZoneMove {
                        object_id,
                        to: ZoneType::Graveyard,
                        ..
                    },
                ) => self.object_event_ref(*object_id),
                _ => None,
            };
            let known_hand_entry = match (&batch.binding, event) {
                (
                    EventBinding::Moved { reveal: true, .. },
                    ProposedEvent::ZoneMove {
                        object_id,
                        from: Some(from @ (ZoneType::Library | ZoneType::Graveyard)),
                        to: ZoneType::Hand,
                        ..
                    },
                ) => {
                    let card = self.state.entry_object(*object_id);
                    let (owner, definition) = (card.owner, card.definition_id);
                    // The graveyard card is already public. A library card
                    // must be revealed before its move into the hand.
                    if *from == ZoneType::Library {
                        self.emit(GameEvent::CardRevealed { owner, definition });
                    }
                    // A known library top can also update this count during the move.
                    let count = self.state.players[owner.0]
                        .known_hand
                        .get(&definition)
                        .copied()
                        .unwrap_or(0)
                        + 1;
                    Some((owner, definition, count))
                }
                _ => None,
            };
            let result = self.commit_proposed_event(event, departing_lki, &mut followups);
            if result.committed {
                if matches!(
                    batch.binding,
                    EventBinding::Moved {
                        library_end: Some(crate::state::ability::LibraryEnd::Bottom),
                        ..
                    }
                ) {
                    if let Some(object_id) = result.moved.filter(|object_id| {
                        self.state.zones.zone_of(*object_id) == Some(ZoneType::Library)
                    }) {
                        self.place_entrant_on_bottom(object_id);
                    }
                }
                if let Some((owner, definition, count)) = known_hand_entry {
                    self.set_known_hand_count(owner, definition, count);
                }
                died.extend(death);
                if matches!(batch.binding, EventBinding::Combat) {
                    hits.extend(result.combat_hit);
                    if let Some((source, amount)) = result.combat_damage {
                        match totals.iter_mut().find(|(dealer, _)| *dealer == source) {
                            Some((_, total)) => *total += amount,
                            None => totals.push((source, amount)),
                        }
                    }
                }
            }
            results.push(result);
        }
        self.commit_lifelink_gains(results.iter().filter_map(|result| result.lifelink));
        if let EventBinding::Moved {
            linked: Some((source, returns)),
            ..
        } = batch.binding
        {
            for object in results.iter().filter_map(|result| result.moved) {
                if self.state.zones.zone_of(object) != Some(ZoneType::Exile) {
                    continue;
                }
                if returns && self.lookup_current_permanent(source).is_err() {
                    followups.push(self.zone_move_proposal(
                        object,
                        ZoneType::Battlefield,
                        ZoneEntry::default(),
                    ));
                } else {
                    self.state
                        .exile_links
                        .push(crate::flow::trigger::ExileLink {
                            source,
                            exiled_object: object,
                            returns,
                        });
                }
            }
        }
        if !died.is_empty() {
            self.emit_observation_event(GameEvent::PermanentsDied { objects: died });
        }
        if !hits.is_empty() {
            self.emit(GameEvent::CombatDamageToPlayers { hits });
        }
        if !totals.is_empty() {
            self.emit_trigger_event(GameEvent::CombatDamageTotals { dealt: totals });
        }
        self.process_event_batch(&sources);
        BatchProgress::Complete { results, followups }
    }

    /// The physical tokens of a creation whose replacements have finished,
    /// allocated outside every zone, as entry proposals under its terms.
    fn allocate_tokens(
        &mut self,
        definition: TokenSource,
        controller: PlayerId,
        count: i32,
        entry: ZoneEntry,
    ) -> Vec<ProposedEvent> {
        let mut entries = Vec::new();
        for _ in 0..count {
            let object = match &definition {
                TokenSource::Registered(key) => self.prepare_token(key, controller),
                TokenSource::Copy {
                    definition_id,
                    definition,
                    front,
                    copy,
                    exceptions,
                } => {
                    let mut template = crate::state::game_object::Object::from_definition(
                        crate::state::game_object::PublicId(0),
                        controller,
                        *definition_id,
                        std::sync::Arc::clone(&definition.printed),
                        true,
                    );
                    template.set_front_face(front.clone());
                    template.copy = copy.clone();
                    Some(self.prepare_token_copy(template, controller, exceptions.clone()))
                }
            };
            if let Some(object) = object {
                entries.push(self.zone_move_proposal(object, ZoneType::Battlefield, entry.clone()));
            }
        }
        entries
    }

    /// The pending entry's "enter as a copy" question, once: `None` when it
    /// has none, was answered, or has nothing to copy.
    fn entry_copy_choice(&self, pending: &mut PendingEvent) -> Option<BatchProgress> {
        let ProposedEvent::ZoneMove {
            object_id: object,
            to: ZoneType::Battlefield,
            entry,
            ..
        } = &pending.event
        else {
            return None;
        };
        if pending.copy_chosen {
            return None;
        }
        let (scope, predicate, _) = self.entry_copy_terms(*object, entry)?;
        let player = entry
            .controller
            .unwrap_or(self.state.entry_object(*object).owner);
        let controllers = self.scope_players(&scope, player, Some(*object), None);
        let candidates: Vec<PermanentId> = controllers
            .into_iter()
            .flat_map(|controller| self.iter_battlefield_permanents(controller))
            .filter(|id| self.permanent_matches_predicate(*id, predicate))
            .collect();
        let object = *object;
        if candidates.is_empty() {
            pending.copy_chosen = true;
            return None;
        }
        Some(BatchProgress::EntryCopy {
            player,
            object,
            candidates,
        })
    }

    /// Answer the cursor member's entry-copy question: it becomes a copy of
    /// `of` (or stays itself) before its other entry replacements are found.
    pub(crate) fn choose_entry_copy(&mut self, batch: &mut EventBatch, of: Option<PermanentId>) {
        let pending = &mut batch.events[batch.cursor];
        pending.copy_chosen = true;
        let ProposedEvent::ZoneMove {
            object_id: object,
            ref entry,
            ..
        } = pending.event
        else {
            return;
        };
        let Some(source) = of.and_then(|id| self.state.permanents[id].as_ref()) else {
            return;
        };
        let source = self.state[source.object].clone();
        let Some((_, _, exceptions)) = self.entry_copy_terms(object, entry) else {
            return;
        };
        let exceptions = exceptions.to_vec();
        self.state
            .entry_object_mut(object)
            .become_copy_of(&source, &exceptions);
        if let ProposedEvent::ZoneMove { entry, .. } = &mut pending.event {
            entry.entry_copy = true;
        }
    }

    /// Own replacements of the face that would enter (CR 614.12). Setting a
    /// land to a basic land type removes its printed abilities (CR 305.7).
    /// Other permanents can still replace the same entry.
    fn entry_replacements(&self, object: ObjectId, entry: &ZoneEntry) -> &[ReplacementEffect] {
        let face = self.entry_printing(object, entry);
        if !face.replacement_effects.is_empty() && face.types.is_land() {
            if self
                .layer_board()
                .land_type_overwrites
                .iter()
                .any(|predicate| self.entry_matches_predicate(object, entry, predicate))
            {
                return &[];
            }
        }
        &face.replacement_effects
    }

    /// The scope, predicate and copy exceptions of the card's "enter as a
    /// copy" replacement, if it has one.
    fn entry_copy_terms(
        &self,
        object: ObjectId,
        entry: &ZoneEntry,
    ) -> Option<(
        crate::state::ability::Scope,
        &crate::state::predicate::CardPredicate,
        &[crate::state::effect::Modification],
    )> {
        self.entry_replacements(object, entry)
            .iter()
            .find_map(|effect| match effect {
                ReplacementEffect::EntersAsCopy {
                    scope,
                    predicate,
                    exceptions,
                } => Some((*scope, predicate, exceptions.as_slice())),
                _ => None,
            })
    }

    pub(crate) fn choose_replacement(
        &self,
        batch: &mut EventBatch,
        candidate: &ReplacementCandidate,
    ) {
        self.apply_replacement(&mut batch.events[batch.cursor], candidate);
    }

    fn affected_player(&self, event: &ProposedEvent) -> PlayerId {
        match event {
            ProposedEvent::Damage { target, .. } => self
                .damage_target_controller(*target)
                .expect("current target"),
            ProposedEvent::CreateTokens { controller, .. } => *controller,
            ProposedEvent::LifeChange { player, .. } => *player,
            ProposedEvent::Destroy { target }
            | ProposedEvent::Sacrifice { target }
            | ProposedEvent::CounterChange { target, .. } => {
                let id = self
                    .lookup_current_permanent(*target)
                    .expect("current object");
                self.state.permanents[id]
                    .as_ref()
                    .expect("permanent")
                    .controller
            }
            ProposedEvent::ZoneMove {
                object_id: object,
                entry,
                ..
            } => entry
                .controller
                .unwrap_or(self.state.entry_object(*object).owner),
        }
    }

    /// Execute an event whose caller requires an immediate result. Game
    /// operations that can ask a replacement question retain an EventBatch on
    /// their EffectFrame; this entry point must never turn Pending into success.
    pub(crate) fn apply_immediate_event(&mut self, event: ProposedEvent) -> bool {
        let batch = self.event_batch(EventBinding::None, [event]);
        self.complete_immediate_batch(batch)
            .first()
            .is_some_and(|result| result.committed)
    }

    /// Commit `batch` and every consequence of it without a retained caller,
    /// returning the root's results.
    /// An optional entry copy is declined: no caller can ask for it.
    pub(crate) fn complete_immediate_batch(&mut self, mut batch: EventBatch) -> Vec<EventResult> {
        let (results, mut followups) = self.advance_immediate_batch(&mut batch);
        while !followups.is_empty() {
            let mut next = self.event_batch(EventBinding::None, followups);
            followups = self.advance_immediate_batch(&mut next).1;
        }
        results
    }

    fn advance_immediate_batch(
        &mut self,
        batch: &mut EventBatch,
    ) -> (Vec<EventResult>, Vec<ProposedEvent>) {
        loop {
            match self.advance_event_batch(batch) {
                BatchProgress::Complete { results, followups } => return (results, followups),
                BatchProgress::EntryCopy { .. } => self.choose_entry_copy(batch, None),
                BatchProgress::Choice { .. } => {
                    panic!("event requiring a choice needs a retained caller")
                }
            }
        }
    }

    fn proposal_is_current(&self, event: &ProposedEvent) -> bool {
        match event {
            ProposedEvent::Damage {
                source,
                target,
                amount,
                ..
            } => {
                *amount > 0
                    && source.is_none_or(|source| self.damage_source_is_known(source))
                    && match target {
                        ProposedDamageTarget::Player(player) => {
                            self.state.players.get(player.0).is_some()
                        }
                        ProposedDamageTarget::Object(object) => {
                            self.lookup_current_permanent(*object).is_ok()
                        }
                    }
            }
            ProposedEvent::CreateTokens {
                controller, count, ..
            } => *count > 0 && self.state.players.get(controller.0).is_some(),
            ProposedEvent::LifeChange { player, delta } => {
                *delta != 0 && self.state.players.get(player.0).is_some()
            }
            ProposedEvent::Destroy { target }
            | ProposedEvent::Sacrifice { target }
            | ProposedEvent::CounterChange { target, .. } => {
                self.lookup_current_permanent(*target).is_ok()
            }
            ProposedEvent::ZoneMove {
                object_id,
                from,
                to,
                entry,
            } => {
                if self.state.objects.get(object_id.slot()).is_none()
                    || self.state.zones.zone_of(*object_id) != *from
                    || *from == Some(*to)
                    || *entry != entry.for_zone(*to)
                {
                    return false;
                }
                self.state.zones.is_latest(*object_id)
            }
        }
    }

    /// A damage source is the current object, a departed one with last known
    /// information, or an earlier incarnation of a real card: a resolved
    /// spell is the source of the delayed triggers it created (CR 603.7d)
    /// though no battlefield record of it exists.
    fn damage_source_is_known(&self, source: ObjectId) -> bool {
        self.state.live(source).is_some()
            || self.object_lki(source).is_some()
            || self
                .state
                .zones
                .generation(source.slot())
                .is_some_and(|current| source.generation() < current)
    }

    fn collect_replacements(&self, event: &ProposedEvent) -> Vec<ReplacementCandidate> {
        let mut candidates = match event {
            ProposedEvent::ZoneMove {
                object_id,
                from: Some(ZoneType::Stack),
                to: ZoneType::Graveyard,
                entry,
            } => {
                let mut choices = Vec::new();
                for (index, replacement) in [
                    entry
                        .resolving_adventure
                        .map(|_| ReplacementEffect::AdventureExile),
                    self.stack_spell(*object_id)
                        .filter(|spell| spell.exile_instead_of_graveyard)
                        .map(|_| ReplacementEffect::ExileInsteadOfGraveyard),
                ]
                .into_iter()
                .enumerate()
                {
                    if let Some(effect) = replacement {
                        choices.push(ReplacementCandidate::printed(*object_id, index, effect));
                    }
                }
                choices
            }
            ProposedEvent::CreateTokens {
                controller, count, ..
            } if *count > 0 => {
                let mut out = Vec::new();
                for (id, view) in self.layer_views(self.iter_battlefield_permanents(*controller)) {
                    if view.abilities_lost {
                        continue;
                    }
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    for (definition_index, effect) in self.state[permanent.object]
                        .replacement_effects
                        .iter()
                        .enumerate()
                    {
                        if matches!(effect, ReplacementEffect::MultiplyTokens { .. }) {
                            out.push(ReplacementCandidate::printed(
                                self.permanent_object(id).expect("battlefield"),
                                definition_index,
                                effect.clone(),
                            ));
                        }
                    }
                }
                out
            }

            ProposedEvent::Damage {
                source: damage_source,
                target,
                amount,
                combat,
            } if *amount > 0 => {
                let Some(affected_controller) = self.damage_target_controller(*target) else {
                    return Vec::new();
                };
                let source_facts = damage_source.and_then(|id| self.damage_source_facts(id));
                let source_controller = source_facts.map(|(controller, _)| controller);
                let mut out = Vec::new();
                // Preserve permanent-slot order while sharing one immutable
                // board read across the replacement scan.
                let ids =
                    self.state
                        .permanents
                        .iter()
                        .enumerate()
                        .filter_map(|(index, permanent)| {
                            permanent.as_ref().map(|_| PermanentId(index))
                        });
                for (id, view) in self.layer_views(ids) {
                    if view.abilities_lost {
                        continue;
                    }
                    let Some(source) = self.permanent_object(id) else {
                        continue;
                    };
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");

                    let source_side = source_controller == Some(permanent.controller)
                        && permanent.controller != affected_controller;
                    for (definition_index, effect) in self.state[permanent.object]
                        .replacement_effects
                        .iter()
                        .enumerate()
                    {
                        let applies = match effect {
                            ReplacementEffect::PreventCombatDamageToOrFromSelf => {
                                *combat
                                    && (*damage_source == Some(source)
                                        || *target == ProposedDamageTarget::Object(source))
                            }
                            ReplacementEffect::ModifyDamage {
                                creature_source,
                                source_by,
                                recipient_by,
                                ..
                            } => {
                                (!*creature_source
                                    || source_facts.is_some_and(|(_, creature)| creature))
                                    && recipient_by
                                        .includes(permanent.controller, affected_controller)
                                    && (*source_by == Actor::AnyPlayer
                                        || source_controller.is_some_and(|p| {
                                            source_by.includes(permanent.controller, p)
                                        }))
                            }
                            ReplacementEffect::OpponentDamagePlusCounters { .. } => source_side,
                            _ => false,
                        };
                        if applies {
                            out.push(ReplacementCandidate::printed(
                                source,
                                definition_index,
                                effect.clone(),
                            ));
                        }
                    }
                }
                // A resolved shield on the damaged permanent ("prevent all
                // combat damage that would be dealt to it this turn") is an
                // effect record, not one of its abilities, so losing all
                // abilities leaves it in place (CR 615.1).
                if let (true, ProposedDamageTarget::Object(object)) = (*combat, *target) {
                    if let Ok(id) = self.lookup_current_permanent(object) {
                        out.extend(
                            self.effects_on(id)
                                .filter(|effect| {
                                    effect.modification == Modification::PreventCombatDamage
                                })
                                .map(|effect| ReplacementCandidate::shield(object, effect)),
                        );
                    }
                }
                out
            }
            ProposedEvent::ZoneMove {
                object_id,
                to: ZoneType::Battlefield,
                entry,
                ..
            } => {
                let source = *object_id;
                // Entry conditions read the prospective controller, including
                // a spell or reanimation controlled by someone other than its owner.
                let entering_controller = entry
                    .controller
                    .unwrap_or(self.state.entry_object(*object_id).owner);
                let mut entries: Vec<_> = self
                    .entry_replacements(*object_id, entry)
                    .iter()
                    .enumerate()
                    .filter_map(|(definition_index, effect)| {
                        let effect = match effect {
                            ReplacementEffect::EntersPrepared
                            | ReplacementEffect::EntersTapped
                            | ReplacementEffect::EntersWithXCounters { .. }
                            | ReplacementEffect::EntersWithCounters { .. } => effect.clone(),
                            ReplacementEffect::EntersWithPlusOneCounters { condition, .. }
                                if condition.as_ref().is_none_or(|condition| {
                                    self.check_static_condition(condition, entering_controller)
                                }) =>
                            {
                                effect.clone()
                            }
                            ReplacementEffect::EntersTappedUnlessControls { predicate }
                                if !self
                                    .battlefield_permanents(entering_controller)
                                    .into_iter()
                                    .any(|id| self.permanent_matches_predicate(id, predicate)) =>
                            {
                                ReplacementEffect::EntersTapped
                            }
                            _ => return None,
                        };
                        Some(ReplacementCandidate::printed(
                            source,
                            definition_index,
                            effect,
                        ))
                    })
                    .collect();
                // Keep own replacements before battlefield grants. Borrow the
                // shared characteristics; discovering one entry rule must not
                // clone every static ability's program on each recheck.
                let ids = [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.iter_battlefield_permanents(player));
                for (id, view) in self.layer_views(ids) {
                    let Some(source) = self.permanent_object(id) else {
                        continue;
                    };
                    for (definition_index, ability) in view.static_abilities.iter().enumerate() {
                        let predicate = match ability {
                            StaticAbility::EntersTapped(predicate) => predicate,
                            StaticAbility::OpponentsEnterTapped(predicate)
                                if self.state.permanents[id].as_ref().unwrap().controller
                                    != entering_controller =>
                            {
                                predicate
                            }
                            _ => continue,
                        };
                        if self.entry_matches_predicate(*object_id, entry, predicate) {
                            entries.push(ReplacementCandidate::printed_static(
                                source,
                                definition_index,
                                ReplacementEffect::EntersTapped,
                            ));
                        }
                    }
                }
                entries
            }
            _ => Vec::new(),
        };
        // Positive placements, including prospective entry counters. This is
        // deliberately recomputed after each transformation: an own-entry
        // replacement may create the first counters another effect can modify.
        let recipient = match event {
            ProposedEvent::CounterChange { target, delta, .. } if *delta > 0 => {
                self.lookup_current_permanent(*target).ok().map(|id| {
                    let permanent = self.state.permanents[id].as_ref().expect("current target");
                    (permanent.object, permanent.controller)
                })
            }
            ProposedEvent::ZoneMove {
                object_id,
                to: ZoneType::Battlefield,
                entry,
                ..
            } if !entry.counters.is_empty() => Some((
                *object_id,
                entry
                    .controller
                    .unwrap_or(self.state.entry_object(*object_id).owner),
            )),
            _ => None,
        };
        if let Some((object_id, controller)) = recipient {
            for (id, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
                if view.abilities_lost {
                    continue;
                }
                let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                for (definition_index, effect) in self.state[permanent.object]
                    .replacement_effects
                    .iter()
                    .enumerate()
                {
                    let ReplacementEffect::ModifyCounters {
                        kind,
                        predicate,
                        effects_only,
                        ..
                    } = effect
                    else {
                        continue;
                    };
                    let matches_object = match event {
                        ProposedEvent::CounterChange { target, .. } => self
                            .lookup_current_permanent(*target)
                            .is_ok_and(|id| self.permanent_matches_predicate(id, predicate)),
                        ProposedEvent::ZoneMove { entry, .. } => {
                            self.entry_matches_predicate(object_id, entry, predicate)
                        }
                        _ => unreachable!("counter recipient is an entry or a permanent"),
                    };
                    let matches_kind = match event {
                        ProposedEvent::CounterChange { kind: placed, .. } => {
                            kind.is_none_or(|kind| kind == *placed)
                        }
                        ProposedEvent::ZoneMove { entry, .. } => {
                            entry.counters.iter().any(|(placed, count)| {
                                count > 0 && kind.is_none_or(|kind| kind == placed)
                            })
                        }
                        _ => unreachable!("counter placement"),
                    };
                    let matches_cause = !effects_only
                        || match event {
                            ProposedEvent::CounterChange { provenance, .. } => {
                                provenance.is_effect()
                            }
                            // Enters-with counters are replacement effects, including
                            // intrinsic Saga/loyalty/defense entry rules (CR 614.16).
                            ProposedEvent::ZoneMove { .. } => true,
                            _ => false,
                        };
                    if matches_object && matches_kind && matches_cause {
                        candidates.push(ReplacementCandidate::printed(
                            self.permanent_object(id).expect("battlefield"),
                            definition_index,
                            effect.clone(),
                        ));
                    }
                }
            }
        }
        candidates
    }

    fn damage_target_controller(&self, target: ProposedDamageTarget) -> Option<PlayerId> {
        match target {
            ProposedDamageTarget::Player(player) => Some(player),
            ProposedDamageTarget::Object(object_ref) => {
                let permanent_id = self.lookup_current_permanent(object_ref).ok()?;
                self.state.permanents[permanent_id]
                    .as_ref()
                    .map(|permanent| permanent.controller)
            }
        }
    }

    /// Automatic and player-chosen replacements consume their identity together
    /// with the transformation, so neither path can offer it a second time.
    fn apply_replacement(&self, pending: &mut PendingEvent, candidate: &ReplacementCandidate) {
        pending.applied.push(candidate.identity);
        match (&mut pending.event, &candidate.effect) {
            (
                ProposedEvent::ZoneMove { to, entry, .. },
                ReplacementEffect::ExileInsteadOfGraveyard,
            ) => {
                *to = ZoneType::Exile;
                *entry = ZoneEntry::default();
            }
            (ProposedEvent::ZoneMove { to, entry, .. }, ReplacementEffect::AdventureExile) => {
                *to = ZoneType::Exile;
                entry.adventure_player = entry.resolving_adventure.take();
            }
            (
                ProposedEvent::Damage { amount, .. },
                ReplacementEffect::PreventCombatDamageToOrFromSelf
                | ReplacementEffect::PreventCombatDamage,
            ) => *amount = 0,
            (
                ProposedEvent::CreateTokens { count, .. },
                ReplacementEffect::MultiplyTokens { multiplier },
            ) => {
                *count = count.saturating_mul(*multiplier).max(0);
            }
            (
                ProposedEvent::CounterChange {
                    delta, provenance, ..
                },
                ReplacementEffect::ModifyCounters {
                    multiplier,
                    additional,
                    ..
                },
            ) => {
                provenance.replaced = true;
                *delta = delta
                    .saturating_mul(*multiplier)
                    .saturating_add(*additional)
                    .max(0);
            }
            (
                ProposedEvent::ZoneMove { entry, .. },
                ReplacementEffect::ModifyCounters {
                    kind,
                    multiplier,
                    additional,
                    ..
                },
            ) => {
                let kinds: Vec<_> = entry
                    .counters
                    .iter()
                    .filter(|(placed, count)| *count > 0 && kind.is_none_or(|kind| kind == *placed))
                    .collect();
                for (kind, count) in kinds {
                    let replaced = count
                        .saturating_mul(*multiplier)
                        .saturating_add(*additional)
                        .max(0);
                    entry.counters.change(kind, replaced - count);
                }
            }

            (
                ProposedEvent::Damage { amount, .. },
                ReplacementEffect::ModifyDamage {
                    transform: DamageTransform::Prevent(prevented),
                    ..
                },
            ) => {
                *amount = amount.saturating_sub_unsigned(*prevented).max(0);
            }
            (
                ProposedEvent::Damage { amount, .. },
                ReplacementEffect::ModifyDamage {
                    transform: DamageTransform::Double,
                    ..
                },
            ) => {
                *amount = amount.saturating_mul(2);
            }
            (ProposedEvent::ZoneMove { entry, .. }, ReplacementEffect::EntersPrepared) => {
                entry.prepared = true;
            }
            (ProposedEvent::ZoneMove { entry, .. }, ReplacementEffect::EntersTapped) => {
                entry.tapped = true;
            }
            (
                ProposedEvent::ZoneMove { entry, .. },
                ReplacementEffect::EntersWithPlusOneCounters { count, .. },
            ) => {
                debug_assert!(*count >= 0, "entry counter count must be nonnegative");
                entry
                    .counters
                    .change(CounterKind::PlusOnePlusOne, (*count).max(0));
            }
            (
                ProposedEvent::ZoneMove { entry, .. },
                ReplacementEffect::EntersWithXCounters { kind },
            ) => {
                entry.counters.change(*kind, i32::from(entry.x));
            }
            (
                ProposedEvent::ZoneMove { entry, .. },
                ReplacementEffect::EntersWithCounters { kind, count },
            ) => {
                entry.counters.change(*kind, (*count).max(0));
            }
            // Damage reduced to zero is no longer dealt (release notes).
            (
                ProposedEvent::Damage { amount, .. },
                ReplacementEffect::OpponentDamagePlusCounters { kind },
            ) if *amount > 0 => {
                let counters = self
                    .lookup_current_permanent(candidate.identity.source)
                    .ok()
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(0, |permanent| permanent.counter_count(*kind));
                *amount = amount.saturating_add(counters);
            }
            _ => {}
        }
    }

    fn commit_proposed_event(
        &mut self,
        event: &ProposedEvent,
        lki: Option<ObjectLki>,
        followups: &mut Vec<ProposedEvent>,
    ) -> EventResult {
        let mut result = EventResult::default();
        result.committed = match *event {
            ProposedEvent::Damage {
                source,
                target,
                amount,
                combat,
            } => {
                return self.commit_damage(source, target, amount, combat);
            }
            ProposedEvent::LifeChange { player, delta } => {
                result.life_delta = self.commit_life_change(player, delta);
                result.life_delta != 0
            }
            ProposedEvent::CreateTokens { .. } => unreachable!("creation owns an entry batch"),
            ProposedEvent::Destroy { .. } | ProposedEvent::Sacrifice { .. } => {
                unreachable!("normalized before replacement")
            }
            ProposedEvent::CounterChange {
                target,
                kind,
                delta,
                by,
                ..
            } => self.commit_counter_change(target, kind, delta, by),
            ProposedEvent::ZoneMove {
                object_id,
                from,
                to,
                ref entry,
            } => {
                result.moved = Some(self.commit_zone_move(
                    object_id,
                    from,
                    to,
                    entry.for_zone(to),
                    lki,
                    followups,
                ));
                true
            }
        };
        result
    }

    /// Counter results of a committed event (including damage) stay inside
    /// that event's batch; they do not start another replacement/trigger pass.
    pub(crate) fn commit_counter_change(
        &mut self,
        target: ObjectId,
        kind: CounterKind,
        delta: i32,
        by: Option<PlayerId>,
    ) -> bool {
        let Ok(permanent_id) = self.lookup_current_permanent(target) else {
            return false;
        };
        // A land-type overwrite's counters change other players' mana.
        if self.state[target]
            .statics
            .iter()
            .any(|ability| matches!(ability, StaticAbility::SetsLandType(_)))
        {
            self.invalidate_mana_cache(PlayerId(0));
            self.invalidate_mana_cache(PlayerId(1));
        }
        let timestamp =
            (delta > 0 && kind.keyword().is_some()).then(|| self.state.id_gen_mut().next_id());
        let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
            return false;
        };
        if !permanent.change_counters(kind, delta) {
            return false;
        }
        if let Some(timestamp) = timestamp {
            permanent.keyword_counter_timestamps.insert(kind, timestamp);
        }
        let total = permanent.counter_count(kind);
        if total == 0 {
            permanent.keyword_counter_timestamps.remove(&kind);
        }
        if let Some(by) = by.filter(|_| delta > 0) {
            self.emit(GameEvent::CountersPut {
                permanent: permanent_id,
                kind,
                count: delta,
                total,
                by,
            });
        }
        true
    }

    pub(crate) fn change_counters(
        &mut self,
        permanent_id: PermanentId,
        kind: CounterKind,
        delta: i32,
    ) -> bool {
        let Some(target) = self.permanent_object(permanent_id) else {
            return false;
        };
        self.apply_immediate_event(ProposedEvent::CounterChange {
            provenance: CounterProvenance::new(EventCause::Effect),
            target,
            kind,
            delta,
            by: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        flow::{event::GameEvent, game::Game},
        state::{game_object::PlayerId, player::PlayerConfig, zone::ZoneType},
    };

    use super::{
        CounterKind, CounterProvenance, EventBinding, EventCause, ProposedEvent, ZoneEntry,
    };

    fn game_with_ogre() -> (Game, crate::state::game_object::PermanentId) {
        let deck = BTreeMap::from([("Gray Ogre".to_string(), 16), ("Mountain".to_string(), 24)]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", deck),
                PlayerConfig::new("p1", BTreeMap::from([("Mountain".to_string(), 40)])),
            ],
            614_900,
            false,
        );
        let permanent = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .expect("scenario ogre");
        (game, permanent)
    }

    #[test]
    fn zone_move_records_both_v16_identity_projections_from_one_object() {
        let (mut game, permanent) = game_with_ogre();
        let old = game.permanent_object(permanent).unwrap();
        let object = game.move_object(old, ZoneType::Hand);
        let proposal = game.zone_move_proposal(object, ZoneType::Exile, ZoneEntry::default());
        let value = serde_json::to_value(&proposal).unwrap();
        assert_eq!(value["ZoneMove"]["card"], serde_json::json!(object.slot()));
        assert_eq!(
            value["ZoneMove"]["object"],
            serde_json::json!({"entity": object.slot(), "incarnation": object.generation()})
        );
        let mut zones = crate::state::zone::ZoneManager::default();
        let entering = zones.allocate();
        let proposal = ProposedEvent::ZoneMove {
            object_id: entering,
            from: None,
            to: ZoneType::Battlefield,
            entry: ZoneEntry::default(),
        };
        assert!(serde_json::to_value(&proposal).unwrap()["ZoneMove"]["object"].is_null());
    }

    #[test]
    fn counter_cause_survives_replacement_and_controls_effect_only_applicability() {
        use crate::state::card::ReplacementEffect;
        use crate::state::predicate::CardPredicate;
        for cause in [EventCause::Effect, EventCause::Cost, EventCause::TurnAction] {
            for modified in [false, true] {
                let (mut game, target) = game_with_ogre();
                for (effects_only, multiplier, additional) in [(true, 2, 0), (false, 1, 1)] {
                    if !effects_only && !modified {
                        continue;
                    }
                    let source = game
                        .scenario_force_battlefield(PlayerId(0), "Mountain", true)
                        .unwrap();
                    let object = game.state.permanents[source].as_ref().unwrap().object;
                    game.state[object].replacement_effects =
                        vec![ReplacementEffect::ModifyCounters {
                            effects_only,
                            kind: None,
                            predicate: CardPredicate::default(),
                            multiplier,
                            additional,
                        }];
                }
                let mut batch = game.event_batch(
                    EventBinding::None,
                    [ProposedEvent::CounterChange {
                        target: game.permanent_object(target).unwrap(),
                        kind: CounterKind::Loyalty,
                        delta: 1,
                        by: Some(PlayerId(0)),
                        provenance: CounterProvenance::new(cause),
                    }],
                );
                while let super::BatchProgress::Choice { candidates, .. } =
                    game.advance_event_batch(&mut batch)
                {
                    // Apply the unconditional +1 first when both are initially
                    // eligible. For costs/turn actions it is initially the only
                    // replacement; its result then admits the effect-only one.
                    let candidate = candidates
                        .iter()
                        .find(|candidate| {
                            matches!(
                                candidate.effect,
                                ReplacementEffect::ModifyCounters {
                                    effects_only: false,
                                    ..
                                }
                            )
                        })
                        .unwrap();
                    game.choose_replacement(&mut batch, candidate);
                }
                let expected = if modified {
                    4
                } else if cause == EventCause::Effect {
                    2
                } else {
                    1
                };
                assert_eq!(
                    game.state.permanents[target]
                        .as_ref()
                        .unwrap()
                        .counter_count(CounterKind::Loyalty),
                    expected
                );
                let ProposedEvent::CounterChange { provenance, .. } = batch.events[0].event else {
                    panic!("counter event")
                };
                assert_eq!(
                    provenance.cause, cause,
                    "replacement never rewrites cost/turn ownership"
                );
                assert_eq!(provenance.replaced, modified || cause == EventCause::Effect);
            }
        }
    }

    #[test]
    fn exact_object_refs_gate_counter_destruction_and_zone_commits() {
        let (mut game, permanent) = game_with_ogre();
        let object_id = game.state.permanents[permanent]
            .as_ref()
            .expect("ogre")
            .object;
        let stale = game.permanent_object(permanent).expect("old object");

        let object_id = game.move_object(object_id, ZoneType::Hand);
        let object_id = game.move_object(object_id, ZoneType::Battlefield);
        let current = game.state.live(object_id).expect("new object");
        let current_permanent = game.state.permanent_of(object_id).expect("new permanent");

        assert!(!game.apply_immediate_event(ProposedEvent::CounterChange {
            provenance: CounterProvenance::new(EventCause::Effect),
            target: stale,
            kind: CounterKind::PlusOnePlusOne,
            delta: 2,
            by: None,
        }));
        assert!(!game.apply_immediate_event(ProposedEvent::Destroy { target: stale }));
        assert!(!game.apply_immediate_event(ProposedEvent::ZoneMove {
            object_id: stale,
            from: Some(ZoneType::Battlefield),
            to: ZoneType::Exile,
            entry: ZoneEntry::default(),
        }));
        assert_eq!(game.state.live(object_id), Some(current));
        assert_eq!(
            game.state.permanents[current_permanent]
                .as_ref()
                .expect("current permanent")
                .counter_count(CounterKind::PlusOnePlusOne),
            0
        );

        assert!(game.apply_immediate_event(ProposedEvent::CounterChange {
            provenance: CounterProvenance::new(EventCause::Effect),
            target: current,
            kind: CounterKind::PlusOnePlusOne,
            delta: 2,
            by: None,
        }));
        assert_eq!(
            game.state.permanents[current_permanent]
                .as_ref()
                .expect("current permanent")
                .counter_count(CounterKind::PlusOnePlusOne),
            2
        );
        assert!(game.apply_immediate_event(ProposedEvent::Destroy { target: current }));
        assert_eq!(
            game.state
                .zones
                .successor(current)
                .and_then(|died| game.state.zones.zone_of(died)),
            Some(ZoneType::Graveyard)
        );
        assert!(game.object_lki(current).is_some());
    }

    #[test]
    fn life_change_result_uses_the_applied_difference_at_the_storage_limit() {
        let (mut game, _) = game_with_ogre();
        let player = PlayerId(0);
        *game.state.life_mut(player).unwrap() = i32::MAX - 1;
        let batch = game.event_batch(
            EventBinding::None,
            [ProposedEvent::LifeChange { player, delta: 3 }],
        );
        let results = game.complete_immediate_batch(batch);
        assert_eq!(game.state.players[0].life, i32::MAX);
        assert_eq!(results[0].life_delta, 1);
    }

    #[test]
    fn fog_bank_prevents_only_combat_damage_at_its_own_endpoints() {
        use super::{BatchProgress, ProposedDamageTarget};
        let deck = BTreeMap::from([("Fog Bank".into(), 20), ("Twinflame Tyrant".into(), 20)]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", deck.clone()),
                PlayerConfig::new("p1", deck),
            ],
            614901,
            false,
        );
        let fog = game
            .scenario_force_battlefield(PlayerId(1), "Fog Bank", true)
            .unwrap();
        game.state.permanents[fog]
            .as_mut()
            .unwrap()
            .change_counters(CounterKind::PlusOnePlusOne, 1);
        let power = game.characteristics(fog).pt().0;
        assert_eq!(power, 1);
        let cat = game
            .create_token("token.cat_1_1_w_lifelink", PlayerId(0), false)
            .unwrap();
        let other = game
            .create_token("token.cat_1_1_w_lifelink", PlayerId(1), false)
            .unwrap();
        let damage = |source, target, combat, amount| ProposedEvent::Damage {
            source: Some(source),
            target: ProposedDamageTarget::Object(target),
            combat,
            amount,
        };
        for (source, target, combat, amount, expected) in [
            (fog, cat, true, power, 0),
            (cat, fog, true, 1, 0),
            (fog, cat, false, power, 1),
            (cat, fog, false, 1, 1),
            (cat, other, true, 1, 1),
        ] {
            let mut branch = game.clone();
            branch.apply_immediate_event(damage(
                game.permanent_object(source).unwrap(),
                game.permanent_object(target).unwrap(),
                combat,
                amount,
            ));
            assert_eq!(
                branch.state.permanents[target].as_ref().unwrap().damage,
                expected
            );
            assert_eq!(
                branch.state.players[0].life,
                20 + if source == cat { expected } else { 0 }
            );
        }

        let tyrant = game
            .scenario_force_battlefield(PlayerId(0), "Twinflame Tyrant", true)
            .unwrap();
        let fog_object = game.permanent_object(fog).unwrap();
        let tyrant_object = game.permanent_object(tyrant).unwrap();
        let proposal = damage(game.permanent_object(cat).unwrap(), fog_object, true, 1);
        let mut batch = game.event_batch(EventBinding::None, [proposal.clone()]);
        let BatchProgress::Choice { player, candidates } = game.advance_event_batch(&mut batch)
        else {
            panic!("competing prevention and doubling")
        };
        assert_eq!(player, PlayerId(1));
        assert_eq!(candidates.len(), 2);
        assert_eq!(batch.events[0].event, proposal);
        assert!(batch.events[0].applied.is_empty());
        let fog_effect = candidates
            .iter()
            .find(|c| c.source() == fog_object)
            .unwrap();
        let double = candidates
            .iter()
            .find(|c| c.source() == tyrant_object)
            .unwrap();
        for (choice, intermediate, identities) in [
            (fog_effect, 0, vec![fog_effect.identity]),
            (double, 2, vec![double.identity, fog_effect.identity]),
        ] {
            let mut branch = game.clone();
            let mut pending = batch.clone();
            branch.choose_replacement(&mut pending, choice);
            assert!(
                matches!(pending.events[0].event, ProposedEvent::Damage { amount, .. } if amount == intermediate)
            );
            assert_eq!(pending.events[0].applied, [choice.identity]);
            let BatchProgress::Complete { followups, .. } =
                branch.advance_event_batch(&mut pending)
            else {
                panic!("replacement order must complete")
            };
            assert!(followups.is_empty());
            assert_eq!(pending.events[0].applied, identities);
            assert!(matches!(
                pending.events[0].event,
                ProposedEvent::Damage { amount: 0, .. }
            ));
            assert_eq!(branch.state.permanents[fog].as_ref().unwrap().damage, 0);
            assert_eq!(branch.state.players[0].life, 20);
        }
    }

    #[test]
    fn pridemate_triggers_once_per_lifelink_source_in_a_damage_batch() {
        use super::ProposedDamageTarget;
        let player = PlayerId(0);
        let deck = BTreeMap::from([("Ajani's Pridemate".into(), 20), ("Plains".into(), 20)]);
        for distinct in [false, true] {
            let mut game = Game::new(
                vec![
                    PlayerConfig::new("p0", deck.clone()),
                    PlayerConfig::new("p1", deck.clone()),
                ],
                614900,
                false,
            );
            let mate = game
                .scenario_force_battlefield(player, "Ajani's Pridemate", true)
                .unwrap();
            let mate_object = game.permanent_object(mate).unwrap();
            let mut cat = |owner| {
                let id = game
                    .create_token("token.cat_1_1_w_lifelink", owner, false)
                    .unwrap();
                game.permanent_object(id).unwrap()
            };
            let first = cat(player);
            let second = if distinct { cat(player) } else { first };
            let blocker = cat(PlayerId(1));
            game.state.events.clear();
            let damage = |source, target, amount| ProposedEvent::Damage {
                source: Some(source),
                target,
                amount,
                combat: true,
            };
            let batch = game.event_batch(
                EventBinding::None,
                [
                    damage(first, ProposedDamageTarget::Object(blocker), 2),
                    damage(second, ProposedDamageTarget::Player(PlayerId(1)), 3),
                ],
            );
            game.complete_immediate_batch(batch);
            let gains: Vec<_> = game
                .state
                .events
                .iter()
                .filter_map(|event| match event {
                    GameEvent::LifeChanged {
                        player: PlayerId(0),
                        old,
                        new,
                        ..
                    } if new > old => Some(new - old),
                    _ => None,
                })
                .collect();
            assert_eq!(gains, if distinct { vec![2, 3] } else { vec![5] });
            assert_eq!(game.state.players[0].life, 25);
            let triggers = &game.state.pending_triggers;
            assert_eq!(triggers.len(), gains.len());
            assert!(triggers
                .iter()
                .all(|trigger| trigger.source_object() == mate_object));
            if distinct {
                let first = triggers[0].id;
                assert!(game.flush_triggers().is_some());
                game.choose_trigger(player, first).unwrap();
            }
            assert!(game.flush_triggers().is_none());
            for _ in &gains {
                game.resolve_top_of_stack();
            }
            let size = 2 + gains.len() as i32;
            assert_eq!(game.characteristics(mate).pt(), (size, size));
        }
    }

    #[test]
    fn direct_life_change_commits_once_and_emits_the_legacy_fact() {
        let (mut game, _) = game_with_ogre();
        game.state.events.clear();

        assert!(game.apply_immediate_event(ProposedEvent::LifeChange {
            player: PlayerId(0),
            delta: -3,
        }));

        assert_eq!(game.state.players[0].life, 17);
        assert_eq!(
            game.state.events,
            vec![GameEvent::LifeChanged {
                player: PlayerId(0),
                old: 20,
                new: 17,
                gain_ordinal: 0,
            }]
        );
    }
}
