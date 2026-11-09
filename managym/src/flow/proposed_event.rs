//! Typed proposed-event, replacement/prevention, and commit pipeline.
//!
//! `GameEvent` remains the factual committed-event stream consumed by
//! triggers, observations, and presentation. A batch retains proposals until
//! every replacement has been chosen, then commits them under one pre-event
//! source snapshot and returns its results with the consequences its caller
//! runs next.

use crate::state::card::Keyword;
use crate::{
    flow::{event::GameEvent, game::Game},
    state::card::StaticAbility,
    state::{
        card::ReplacementEffect,
        game_object::{CardId, EntityId, ObjectLki, ObjectRef, PermanentId, PlayerId},
        permanent::{CounterKind, Counters},
        zone::ZoneType,
    },
};

/// How a card enters its new zone.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ZoneEntry {
    /// Battlefield: enters tapped.
    pub tapped: bool,
    /// Battlefield: enters with these counters.
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
}

impl ZoneEntry {
    /// The terms `zone` takes: battlefield terms only for the battlefield,
    /// face down only for exile.
    pub(crate) fn for_zone(&self, zone: ZoneType) -> Self {
        match zone {
            ZoneType::Battlefield => Self {
                face_down: false,
                ..self.clone()
            },
            ZoneType::Exile => Self {
                face_down: self.face_down,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum ProposedDamageTarget {
    Player(PlayerId),
    Object(ObjectRef),
}

/// Initial cause is retained even after a replacement makes the placement an
/// effect for CR 614.16. Payment/turn completion still belongs to its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum EventCause {
    Effect,
    // The causal domain includes costs even before a registered ability pays
    // positive counters. Directed CR 614.16 witnesses exercise this boundary.
    #[allow(dead_code)]
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
pub(crate) enum TokenDefinition {
    Registered(String),
    Copy {
        definition_id: crate::state::card::CardDefId,
        definition: std::sync::Arc<crate::state::card::CardDefinition>,
        front: Option<(
            crate::state::card::CardDefId,
            std::sync::Arc<crate::state::card::CardDefinition>,
        )>,
        copy: Option<crate::state::effect::CopyEffect>,
        exceptions: Vec<crate::state::effect::Modification>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum ProposedEvent {
    Damage {
        source: Option<ObjectRef>,
        target: ProposedDamageTarget,
        amount: i32,
        combat: bool,
    },
    LifeChange {
        player: PlayerId,
        delta: i32,
    },
    Destroy {
        target: ObjectRef,
    },
    Sacrifice {
        target: ObjectRef,
    },
    CreateTokens {
        definition: Box<TokenDefinition>,
        controller: PlayerId,
        count: i32,
        entry: ZoneEntry,
    },
    CounterChange {
        provenance: CounterProvenance,
        target: ObjectRef,
        kind: CounterKind,
        delta: i32,
        by: Option<PlayerId>,
    },
    ZoneMove {
        card: CardId,
        object: Option<ObjectRef>,
        from: Option<ZoneType>,
        to: ZoneType,
        entry: ZoneEntry,
    },
}

/// The definition namespace is part of identity: a static and a replacement
/// at index zero on the same object are different effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ReplacementId {
    pub(crate) source: ObjectRef,
    pub(crate) definition_index: usize,
    pub(crate) static_ability: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ReplacementCandidate {
    identity: ReplacementId,
    pub(crate) effect: ReplacementEffect,
}

impl ReplacementCandidate {
    pub(crate) fn source(&self) -> ObjectRef {
        self.identity.source
    }

    pub(crate) fn description(&self) -> String {
        match &self.effect {
            ReplacementEffect::PreventDamageToController { amount } => {
                format!("Prevent {amount} damage")
            }
            ReplacementEffect::DoubleDamageToController => "Double the damage".into(),
            ReplacementEffect::EntersTapped
            | ReplacementEffect::EntersTappedUnlessControls { .. } => "Enter tapped".into(),
            ReplacementEffect::EntersWithPlusOneCounters { count, .. } => {
                format!("Enter with {count} +1/+1 counters")
            }
            ReplacementEffect::EntersWithXCounters { kind } => {
                format!("Enter with X {kind:?} counters")
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct PendingEvent {
    pub event: ProposedEvent,
    /// Index of the original proposal; a creation's token entries keep its ordinal.
    pub(crate) ordinal: usize,
    pub(crate) sacrifice: Option<(CardId, PlayerId)>,
    pub(crate) applied: Vec<ReplacementId>,
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
    pub(crate) fn public_entries(&self, out: &mut Vec<ObjectRef>) {
        for pending in &self.events {
            if let ProposedEvent::ZoneMove {
                object: Some(object),
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
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) enum EventBinding {
    #[default]
    None,
    Moved {
        linked: Option<(ObjectRef, bool)>,
        shuffle: bool,
        library_end: Option<crate::state::ability::LibraryEnd>,
        reveal: bool,
    },
    ThisWay,
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
    pub moved: Option<ObjectRef>,
    pub excess_damage: Option<i32>,
}

pub(crate) enum BatchProgress {
    Choice {
        player: PlayerId,
        candidates: Vec<ReplacementCandidate>,
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
    pub(crate) fn event_batch(&self, events: Vec<ProposedEvent>) -> EventBatch {
        EventBatch {
            events: events
                .into_iter()
                .enumerate()
                .filter_map(|(ordinal, event)| self.pending_event(ordinal, event))
                .collect(),
            cursor: 0,
            binding: EventBinding::None,
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
                sacrifice = Some((permanent.card, permanent.controller));
            }
            event = ProposedEvent::ZoneMove {
                card: permanent.card,
                object: Some(*target),
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
        })
    }

    fn normalize_proposal(&self, event: &mut ProposedEvent) {
        // CR 614.1a: replace any death of the exact affected incarnation,
        // including sacrifice and zero toughness, not only lethal damage.
        if let ProposedEvent::ZoneMove {
            object: Some(object),
            from: Some(ZoneType::Battlefield),
            to,
            ..
        } = event
        {
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
                    card,
                    from: Some(ZoneType::Battlefield),
                    ..
                } => self.snapshot_current_permanent(*card),
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
                if let ProposedEvent::ZoneMove { card, .. } = &pending.event {
                    if !departed.insert(*card) {
                        continue;
                    }
                }
                if let Some((card, controller)) = pending.sacrifice {
                    self.state.turn.history[controller.0].permanents_sacrificed += 1;
                    self.emit(GameEvent::PermanentSacrificed { card, controller });
                    sacrifices = true;
                }
            }
            if sacrifices {
                self.process_event_batch(&sources);
            }
        }
        for (pending, departing_lki) in batch.events.iter().zip(departing) {
            if !self.proposal_is_current(&pending.event) {
                results.push(EventResult::default());
                continue;
            }
            let mut event = pending.event.clone();
            if let ProposedEvent::ZoneMove { to, entry, .. } = &mut event {
                *entry = entry.for_zone(*to);
            }
            let moved_card = match &event {
                ProposedEvent::ZoneMove { card, .. } => Some(*card),
                _ => None,
            };
            let excess_damage = match &event {
                ProposedEvent::Damage {
                    source,
                    target: ProposedDamageTarget::Object(target),
                    amount,
                    ..
                } => Some(
                    self.lookup_current_permanent(*target)
                        .ok()
                        .map_or(0, |id| self.excess_for_damage(*source, id, *amount)),
                ),
                _ => None,
            };
            let death = if matches!(batch.binding, EventBinding::Deaths) {
                match &event {
                    ProposedEvent::ZoneMove {
                        object: Some(object),
                        to: ZoneType::Graveyard,
                        ..
                    } => self.object_event_ref(*object),
                    _ => None,
                }
            } else {
                None
            };
            let hit = if matches!(batch.binding, EventBinding::Combat) {
                match &event {
                    ProposedEvent::Damage {
                        source: Some(source),
                        target: ProposedDamageTarget::Player(_),
                        amount,
                        ..
                    } if *amount > 0 => self.lookup_current_permanent(*source).ok(),
                    _ => None,
                }
            } else {
                None
            };
            let reveal = if matches!(batch.binding, EventBinding::Moved { reveal: true, .. }) {
                match &event {
                    ProposedEvent::ZoneMove {
                        card,
                        from: Some(ZoneType::Graveyard | ZoneType::Library),
                        to: ZoneType::Hand,
                        ..
                    } => Some((
                        self.state.cards[*card].owner,
                        self.state.cards[*card].definition_id,
                    )),
                    _ => None,
                }
            } else {
                None
            };
            if let Some((owner, definition)) = reveal {
                self.emit(GameEvent::CardRevealed { owner, definition });
            }
            let committed = self.commit_proposed_event(event, departing_lki, &mut followups);
            if committed {
                if matches!(
                    batch.binding,
                    EventBinding::Moved {
                        library_end: Some(crate::state::ability::LibraryEnd::Bottom),
                        ..
                    }
                ) {
                    if let Some(card) = moved_card
                        .filter(|card| self.state.zones.zone_of(*card) == Some(ZoneType::Library))
                    {
                        self.put_on_bottom_of_library(card);
                    }
                }
                if let Some((owner, definition)) = reveal {
                    let count = self.state.players[owner.0]
                        .known_hand
                        .get(&definition)
                        .copied()
                        .unwrap_or(0);
                    self.set_known_hand_count(owner, definition, count + 1);
                }
                died.extend(death);
                hits.extend(hit);
            }
            results.push(EventResult {
                committed,
                excess_damage,
                moved: moved_card
                    .filter(|_| committed)
                    .and_then(|card| self.current_object_ref(card)),
            });
        }
        if let EventBinding::Moved {
            linked: Some((source, returns)),
            ..
        } = batch.binding
        {
            for object in results.iter().filter_map(|result| result.moved) {
                let card = CardId::from(object.entity);
                if self.state.zones.zone_of(card) != Some(ZoneType::Exile) {
                    continue;
                }
                if returns && self.lookup_current_permanent(source).is_err() {
                    followups.push(self.zone_move_proposal(
                        card,
                        ZoneType::Battlefield,
                        ZoneEntry::default(),
                    ));
                } else {
                    self.state
                        .exile_links
                        .push(crate::flow::trigger::ExileLink {
                            source,
                            exiled_card: card,
                            returns,
                        });
                }
            }
        }
        if !died.is_empty() {
            self.emit_observation_event(GameEvent::PermanentsDied { objects: died });
        }
        if !hits.is_empty() {
            self.emit(GameEvent::CombatDamageToPlayers { creatures: hits });
        }
        self.process_event_batch(&sources);
        BatchProgress::Complete { results, followups }
    }

    /// The physical tokens of a creation whose replacements have finished,
    /// allocated outside every zone, as entry proposals under its terms.
    fn allocate_tokens(
        &mut self,
        definition: TokenDefinition,
        controller: PlayerId,
        count: i32,
        entry: ZoneEntry,
    ) -> Vec<ProposedEvent> {
        let mut entries = Vec::new();
        for _ in 0..count {
            let card = match &definition {
                TokenDefinition::Registered(name) => self.prepare_token(name, controller),
                TokenDefinition::Copy {
                    definition_id,
                    definition,
                    front,
                    copy,
                    exceptions,
                } => {
                    let mut template = crate::state::card::Card::from_definition(
                        crate::state::game_object::ObjectId(0),
                        controller,
                        *definition_id,
                        std::sync::Arc::clone(definition),
                    );
                    template.set_front_face(front.clone());
                    template.copy = copy.clone();
                    Some(self.prepare_token_copy(template, controller, exceptions.clone()))
                }
            };
            if let Some(card) = card {
                entries.push(self.zone_move_proposal(card, ZoneType::Battlefield, entry.clone()));
            }
        }
        entries
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
            ProposedEvent::ZoneMove { card, entry, .. } => {
                entry.controller.unwrap_or(self.state.cards[*card].owner)
            }
        }
    }

    /// Execute an event whose caller requires an immediate result. Game
    /// operations that can ask a replacement question retain an EventBatch on
    /// their EffectFrame; this entry point must never turn Pending into success.
    pub(crate) fn apply_immediate_event(&mut self, event: ProposedEvent) -> bool {
        let batch = self.event_batch(vec![event]);
        self.complete_immediate_batch(batch)
            .first()
            .is_some_and(|result| result.committed)
    }

    /// Commit `batch` and every consequence of it without a retained caller,
    /// returning the root's results.
    pub(crate) fn complete_immediate_batch(&mut self, mut batch: EventBatch) -> Vec<EventResult> {
        let BatchProgress::Complete {
            results,
            mut followups,
        } = self.advance_event_batch(&mut batch)
        else {
            panic!("event requiring a choice needs a retained caller")
        };
        while !followups.is_empty() {
            let mut next = self.event_batch(followups);
            let BatchProgress::Complete {
                followups: more, ..
            } = self.advance_event_batch(&mut next)
            else {
                panic!("event requiring a choice needs a retained caller")
            };
            followups = more;
        }
        results
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
                    && source.is_none_or(|object_ref| self.object_ref_is_known(object_ref))
                    && match target {
                        ProposedDamageTarget::Player(player) => {
                            self.state.players.get(player.0).is_some()
                        }
                        ProposedDamageTarget::Object(object_ref) => {
                            self.lookup_current_permanent(*object_ref).is_ok()
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
                card,
                object,
                from,
                to,
                entry,
            } => {
                if self.state.cards.get(card.0).is_none()
                    || self.state.zones.zone_of(*card) != *from
                    || *from == Some(*to)
                    || *entry != entry.for_zone(*to)
                {
                    return false;
                }
                match (from, object) {
                    (Some(_), Some(object_ref)) => {
                        object_ref.entity == EntityId::from(*card)
                            && self.current_object_ref(*card) == Some(*object_ref)
                    }
                    (None, None) => self.current_object_ref(*card).is_none(),
                    _ => false,
                }
            }
        }
    }

    fn object_ref_is_known(&self, object_ref: ObjectRef) -> bool {
        let card = CardId::from(object_ref.entity);
        self.state.cards.get(card.0).is_some()
            && (self.current_object_ref(card) == Some(object_ref)
                || self.object_lki(object_ref).is_some())
    }

    fn collect_replacements(&self, event: &ProposedEvent) -> Vec<ReplacementCandidate> {
        let mut candidates = match event {
            ProposedEvent::CreateTokens {
                controller, count, ..
            } if *count > 0 => {
                let mut out = Vec::new();
                for (id, view) in self.layer_views(self.iter_battlefield_permanents(*controller)) {
                    if view.abilities_lost {
                        continue;
                    }
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    for (definition_index, effect) in self.state.cards[permanent.card]
                        .replacement_effects
                        .iter()
                        .enumerate()
                    {
                        if matches!(effect, ReplacementEffect::MultiplyTokens { .. }) {
                            out.push(ReplacementCandidate {
                                identity: ReplacementId {
                                    source: self.permanent_object_ref(id).expect("battlefield"),
                                    definition_index,
                                    static_ability: false,
                                },
                                effect: effect.clone(),
                            });
                        }
                    }
                }
                out
            }

            ProposedEvent::Damage {
                source,
                target,
                amount,
                ..
            } if *amount > 0 => {
                let Some(affected_controller) = self.damage_target_controller(*target) else {
                    return Vec::new();
                };
                let source_controller =
                    source.and_then(|source| self.damage_source_controller(source));
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
                    let Some(source) = self.permanent_object_ref(id) else {
                        continue;
                    };
                    let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                    let affected_side = permanent.controller == affected_controller;
                    let source_side = source_controller == Some(permanent.controller)
                        && permanent.controller != affected_controller;
                    for (definition_index, effect) in self.state.cards[permanent.card]
                        .replacement_effects
                        .iter()
                        .enumerate()
                    {
                        let applies = match effect {
                            ReplacementEffect::PreventDamageToController { .. }
                            | ReplacementEffect::DoubleDamageToController => affected_side,
                            ReplacementEffect::OpponentDamagePlusCounters { .. } => source_side,
                            _ => false,
                        };
                        if applies {
                            out.push(ReplacementCandidate {
                                identity: ReplacementId {
                                    source,
                                    definition_index,
                                    static_ability: false,
                                },
                                effect: effect.clone(),
                            });
                        }
                    }
                }
                out
            }
            ProposedEvent::ZoneMove {
                card,
                object,
                to: ZoneType::Battlefield,
                entry,
                ..
            } => {
                let source = object.unwrap_or(ObjectRef {
                    entity: EntityId::from(*card),
                    incarnation: self.state.object_incarnations[*card],
                });
                // Entry conditions read the prospective controller, including
                // a spell or reanimation controlled by someone other than its owner.
                let entering_controller = entry.controller.unwrap_or(self.state.cards[*card].owner);
                let mut entries: Vec<_> = self.state.cards[*card]
                    .replacement_effects
                    .iter()
                    .enumerate()
                    .filter_map(|(definition_index, effect)| {
                        let effect = match effect {
                            ReplacementEffect::EntersTapped
                            | ReplacementEffect::EntersWithXCounters { .. } => effect.clone(),
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
                        Some(ReplacementCandidate {
                            identity: ReplacementId {
                                source,
                                definition_index,
                                static_ability: false,
                            },
                            effect,
                        })
                    })
                    .collect();
                // Keep own replacements before battlefield grants. Borrow the
                // shared characteristics; discovering one entry rule must not
                // clone every static ability's program on each recheck.
                let ids = [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.iter_battlefield_permanents(player));
                for (id, view) in self.layer_views(ids) {
                    let Some(source) = self.permanent_object_ref(id) else {
                        continue;
                    };
                    for (definition_index, ability) in view.static_abilities.iter().enumerate() {
                        if matches!(ability, StaticAbility::EntersTapped(predicate)
                            if self.card_matches_predicate(*card, predicate))
                        {
                            entries.push(ReplacementCandidate {
                                identity: ReplacementId {
                                    source,
                                    definition_index,
                                    static_ability: true,
                                },
                                effect: ReplacementEffect::EntersTapped,
                            });
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
                    (permanent.card, permanent.controller)
                })
            }
            ProposedEvent::ZoneMove {
                card,
                to: ZoneType::Battlefield,
                entry,
                ..
            } if !entry.counters.is_empty() => Some((
                *card,
                entry.controller.unwrap_or(self.state.cards[*card].owner),
            )),
            _ => None,
        };
        if let Some((card, controller)) = recipient {
            for (id, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
                if view.abilities_lost {
                    continue;
                }
                let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                for (definition_index, effect) in self.state.cards[permanent.card]
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
                    let matches_card = match event {
                        ProposedEvent::CounterChange { target, .. } => self
                            .lookup_current_permanent(*target)
                            .is_ok_and(|id| self.permanent_matches_predicate(id, predicate)),
                        _ => self.card_matches_predicate(card, predicate),
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
                    if matches_card && matches_kind && matches_cause {
                        candidates.push(ReplacementCandidate {
                            identity: ReplacementId {
                                source: self.permanent_object_ref(id).expect("battlefield"),
                                definition_index,
                                static_ability: false,
                            },
                            effect: effect.clone(),
                        });
                    }
                }
            }
        }
        candidates
    }

    /// The controller of a damage source: a permanent, a spell on the
    /// stack, or the last known information of an object that left.
    fn damage_source_controller(&self, source: ObjectRef) -> Option<PlayerId> {
        if let Ok(id) = self.lookup_current_permanent(source) {
            return self.state.permanents[id].as_ref().map(|p| p.controller);
        }
        let card = CardId::from(source.entity);
        if self.current_object_ref(card) == Some(source) {
            if let Some(controller) = self.stack_spell_controller(card) {
                return Some(controller);
            }
        }
        self.object_lki(source).map(|lki| lki.controller)
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
                ReplacementEffect::PreventDamageToController { amount: prevented },
            ) => {
                debug_assert!(*prevented >= 0, "prevention amount must be nonnegative");
                *amount = amount.saturating_sub((*prevented).max(0)).max(0);
            }
            (ProposedEvent::Damage { amount, .. }, ReplacementEffect::DoubleDamageToController) => {
                *amount = amount.saturating_mul(2);
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
        event: ProposedEvent,
        lki: Option<ObjectLki>,
        followups: &mut Vec<ProposedEvent>,
    ) -> bool {
        match event {
            ProposedEvent::Damage {
                source,
                target,
                amount,
                combat,
            } => self.commit_damage(source, target, amount, combat),
            ProposedEvent::LifeChange { player, delta } => self.commit_life_change(player, delta),
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
                card,
                object,
                from,
                to,
                entry,
            } => self.commit_zone_move(card, object, from, to, entry, lki, followups),
        }
    }

    /// Counter results of a committed event (including damage) stay inside
    /// that event's batch; they do not start another replacement/trigger pass.
    pub(crate) fn commit_counter_change(
        &mut self,
        target: ObjectRef,
        kind: CounterKind,
        delta: i32,
        by: Option<PlayerId>,
    ) -> bool {
        let Ok(permanent_id) = self.lookup_current_permanent(target) else {
            return false;
        };
        // A land-type overwrite's counters change other players' mana.
        if self.state.cards[CardId::from(target.entity)]
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
        let Some(target) = self.permanent_object_ref(permanent_id) else {
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

    use super::{CounterKind, CounterProvenance, EventCause, ProposedEvent, ZoneEntry};

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
                    let card = game.state.permanents[source].as_ref().unwrap().card;
                    game.state.cards[card].replacement_effects =
                        vec![ReplacementEffect::ModifyCounters {
                            effects_only,
                            kind: None,
                            predicate: CardPredicate::default(),
                            multiplier,
                            additional,
                        }];
                }
                let mut batch = game.event_batch(vec![ProposedEvent::CounterChange {
                    target: game.permanent_object_ref(target).unwrap(),
                    kind: CounterKind::Loyalty,
                    delta: 1,
                    by: Some(PlayerId(0)),
                    provenance: CounterProvenance::new(cause),
                }]);
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
        let card = game.state.permanents[permanent]
            .as_ref()
            .expect("ogre")
            .card;
        let stale = game.permanent_object_ref(permanent).expect("old object");

        game.move_card(card, ZoneType::Hand);
        game.move_card(card, ZoneType::Battlefield);
        let current = game.current_object_ref(card).expect("new object");
        let current_permanent = game.state.card_to_permanent[card].expect("new permanent");

        assert!(!game.apply_immediate_event(ProposedEvent::CounterChange {
            provenance: CounterProvenance::new(EventCause::Effect),
            target: stale,
            kind: CounterKind::PlusOnePlusOne,
            delta: 2,
            by: None,
        }));
        assert!(!game.apply_immediate_event(ProposedEvent::Destroy { target: stale }));
        assert!(!game.apply_immediate_event(ProposedEvent::ZoneMove {
            card,
            object: Some(stale),
            from: Some(ZoneType::Battlefield),
            to: ZoneType::Exile,
            entry: ZoneEntry::default(),
        }));
        assert_eq!(game.current_object_ref(card), Some(current));
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
        assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Graveyard));
        assert!(game.object_lki(current).is_some());
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
            }]
        );
    }
}
