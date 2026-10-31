//! Typed proposed-event, replacement/prevention, and commit pipeline.
//!
//! `GameEvent` remains the factual committed-event stream consumed by
//! triggers, observations, and presentation. These proposals are ephemeral
//! mutation requests: validate exact identity, collect declarative
//! replacements, transform deterministically, then commit.

use crate::{
    flow::game::Game,
    state::{
        card::ReplacementEffect,
        game_object::{CardId, EntityId, ObjectRef, PermanentId, PlayerId},
        permanent::{CounterKind, Counters},
        zone::ZoneType,
    },
};

/// How a card enters its new zone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
}

impl ZoneEntry {
    /// The terms `zone` takes: battlefield terms only for the battlefield,
    /// face down only for exile.
    fn for_zone(self, zone: ZoneType) -> Self {
        match zone {
            ZoneType::Battlefield => Self {
                face_down: false,
                ..self
            },
            ZoneType::Exile => Self {
                face_down: self.face_down,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProposedDamageTarget {
    Player(PlayerId),
    Object(ObjectRef),
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    CounterChange {
        target: ObjectRef,
        kind: CounterKind,
        delta: i32,
    },
    ZoneMove {
        card: CardId,
        object: Option<ObjectRef>,
        from: Option<ZoneType>,
        to: ZoneType,
        entry: ZoneEntry,
    },
}

#[derive(Clone, Debug)]
struct ReplacementCandidate {
    source: ObjectRef,
    definition_index: usize,
    effect: ReplacementEffect,
}

impl Game {
    pub(crate) fn apply_proposed_event(&mut self, mut event: ProposedEvent) -> bool {
        if !self.proposal_is_current(&event) {
            return false;
        }

        // CR 614.1a: replace any death of the exact affected incarnation,
        // including sacrifice and zero toughness, not only lethal damage.
        if let ProposedEvent::ZoneMove {
            object: Some(object),
            from: Some(ZoneType::Battlefield),
            to,
            ..
        } = &mut event
        {
            if *to == ZoneType::Graveyard
                && self
                    .lookup_current_permanent(*object)
                    .ok()
                    .is_some_and(|id| self.exiled_if_it_would_die(id))
            {
                *to = ZoneType::Exile;
            }
        }
        let replacements = self.collect_replacements(&event);
        for replacement in replacements {
            self.apply_replacement(&mut event, &replacement);
        }
        // A redirected move keeps only the terms its new destination takes.
        if let ProposedEvent::ZoneMove { to, entry, .. } = &mut event {
            *entry = entry.for_zone(*to);
        }

        self.commit_proposed_event(event)
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
            ProposedEvent::LifeChange { player, delta } => {
                *delta != 0 && self.state.players.get(player.0).is_some()
            }
            ProposedEvent::Destroy { target } | ProposedEvent::CounterChange { target, .. } => {
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
            ProposedEvent::Damage { source, target, .. } => {
                let Some(affected_controller) = self.damage_target_controller(*target) else {
                    return Vec::new();
                };
                let source_controller =
                    source.and_then(|source| self.damage_source_controller(source));
                let mut out = Vec::new();
                for (index, permanent) in
                    self.state
                        .permanents
                        .iter()
                        .enumerate()
                        .filter_map(|(index, permanent)| {
                            permanent.as_ref().map(|permanent| (index, permanent))
                        })
                {
                    if self.characteristics(PermanentId(index)).abilities_lost {
                        continue;
                    }
                    let Some(source) = self.permanent_object_ref(PermanentId(index)) else {
                        continue;
                    };
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
                                source,
                                definition_index,
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
                ..
            } => {
                let source = object.unwrap_or(ObjectRef {
                    entity: EntityId::from(*card),
                    incarnation: self.state.object_incarnations[*card],
                });
                // Entering objects come under their owner's control; the
                // condition reads that player's battlefield before the move.
                let entering_controller = self.state.cards[*card].owner;
                self.state.cards[*card]
                    .replacement_effects
                    .iter()
                    .enumerate()
                    .filter_map(|(definition_index, effect)| {
                        let effect = match effect {
                            ReplacementEffect::EntersTapped
                            | ReplacementEffect::EntersWithPlusOneCounters { .. }
                            | ReplacementEffect::EntersWithXCounters { .. } => effect.clone(),
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
                            source,
                            definition_index,
                            effect,
                        })
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        // The affected player orders damage replacements (CR 616.1); the
        // opponent's increase applies last, after prevention and doubling,
        // which is never worse for that player.
        candidates.sort_by_key(|candidate| {
            (
                matches!(
                    candidate.effect,
                    ReplacementEffect::OpponentDamagePlusCounters { .. }
                ),
                candidate.source,
                candidate.definition_index,
            )
        });
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

    fn apply_replacement(&self, event: &mut ProposedEvent, candidate: &ReplacementCandidate) {
        match (event, &candidate.effect) {
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
                ReplacementEffect::EntersWithPlusOneCounters { count },
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
                    .lookup_current_permanent(candidate.source)
                    .ok()
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(0, |permanent| permanent.counter_count(*kind));
                *amount = amount.saturating_add(counters);
            }
            _ => {}
        }
    }

    fn commit_proposed_event(&mut self, event: ProposedEvent) -> bool {
        match event {
            ProposedEvent::Damage {
                source,
                target,
                amount,
                combat,
            } => self.commit_damage(source, target, amount, combat),
            ProposedEvent::LifeChange { player, delta } => self.commit_life_change(player, delta),
            ProposedEvent::Destroy { target } => {
                let Ok(permanent_id) = self.lookup_current_permanent(target) else {
                    return false;
                };
                if self.characteristics(permanent_id).keywords.indestructible {
                    return false;
                }
                let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                    return false;
                };
                let card = permanent.card;
                self.apply_proposed_event(ProposedEvent::ZoneMove {
                    card,
                    object: Some(target),
                    from: Some(ZoneType::Battlefield),
                    to: ZoneType::Graveyard,
                    entry: ZoneEntry::default(),
                })
            }
            ProposedEvent::CounterChange {
                target,
                kind,
                delta,
            } => {
                let Ok(permanent_id) = self.lookup_current_permanent(target) else {
                    return false;
                };
                let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
                    return false;
                };
                permanent.change_counters(kind, delta)
            }
            ProposedEvent::ZoneMove {
                card,
                object,
                from,
                to,
                entry,
            } => self.commit_zone_move(card, object, from, to, entry),
        }
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
        self.apply_proposed_event(ProposedEvent::CounterChange {
            target,
            kind,
            delta,
        })
    }

    /// Exact-object counter mutation for rules continuations.
    pub(crate) fn change_plus_one_counters_on_object(
        &mut self,
        target: ObjectRef,
        delta: i32,
    ) -> bool {
        self.apply_proposed_event(ProposedEvent::CounterChange {
            target,
            kind: CounterKind::PlusOnePlusOne,
            delta,
        })
    }

    /// Exact-object destruction for rules continuations and SBA batches.
    pub(crate) fn destroy_object(&mut self, target: ObjectRef) -> bool {
        self.apply_proposed_event(ProposedEvent::Destroy { target })
    }

    /// Move one exact current object without following its physical card into
    /// a later incarnation. Trigger collection is owned by the caller so an
    /// SBA batch can expose one simultaneous pre-batch source snapshot.
    pub(crate) fn move_object_to_zone(&mut self, object: ObjectRef, to: ZoneType) -> bool {
        let card = CardId::from(object.entity);
        let from = self.state.zones.zone_of(card);
        self.apply_proposed_event(ProposedEvent::ZoneMove {
            card,
            object: Some(object),
            from,
            to,
            entry: ZoneEntry::default(),
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

    use super::{CounterKind, ProposedEvent, ZoneEntry};

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

        assert!(!game.apply_proposed_event(ProposedEvent::CounterChange {
            target: stale,
            kind: CounterKind::PlusOnePlusOne,
            delta: 2,
        }));
        assert!(!game.apply_proposed_event(ProposedEvent::Destroy { target: stale }));
        assert!(!game.apply_proposed_event(ProposedEvent::ZoneMove {
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

        assert!(game.apply_proposed_event(ProposedEvent::CounterChange {
            target: current,
            kind: CounterKind::PlusOnePlusOne,
            delta: 2,
        }));
        assert_eq!(
            game.state.permanents[current_permanent]
                .as_ref()
                .expect("current permanent")
                .counter_count(CounterKind::PlusOnePlusOne),
            2
        );
        assert!(game.apply_proposed_event(ProposedEvent::Destroy { target: current }));
        assert_eq!(game.state.zones.zone_of(card), Some(ZoneType::Graveyard));
        assert!(game.object_lki(current).is_some());
    }

    #[test]
    fn direct_life_change_commits_once_and_emits_the_legacy_fact() {
        let (mut game, _) = game_with_ogre();
        game.state.events.clear();

        assert!(game.apply_proposed_event(ProposedEvent::LifeChange {
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
