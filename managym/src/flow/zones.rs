// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    flow::{
        decision::Decision,
        event::GameEvent,
        game::{CastChoices, Game},
        proposed_event::{ProposedEvent, ZoneEntry},
    },
    state::{
        card::CardDefId,
        card::Restriction,
        game_object::{CardId, Incarnation, ObjectRef, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        permanent::Permanent,
        predicate::CardPredicate,
        stack_object::{SpellOnStack, StackObject},
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn set_known_hand_count(
        &mut self,
        player: PlayerId,
        definition: CardDefId,
        count: u32,
    ) {
        let known = &mut self.state.players[player.0].known_hand;
        if count == 0 {
            known.remove(&definition);
        } else {
            known.insert(definition, count);
        }
    }

    /// A sacrifice is a committed event distinct from destruction (CR 701.21
    /// and 701.8 in the September 25, 2026 rules). Indestructible does not
    /// prevent it; the departing object's controller pays this cost.
    /// Match its triggers against the pre-departure battlefield, including the
    /// sacrificed source, before replacement effects can change its destination.
    pub(crate) fn sacrifice_permanent(&mut self, permanent: PermanentId) {
        let Some(source) = self.state.permanents[permanent].as_ref() else {
            return;
        };
        let card = source.card;
        let controller = source.controller;
        let sources = self.snapshot_trigger_sources();
        self.emit(GameEvent::PermanentSacrificed { card, controller });
        self.process_game_events_with_sources(&sources);
        self.move_card(card, ZoneType::Graveyard);
    }

    /// Tapped creatures the opponent of `player` controls.
    pub(crate) fn tapped_opponent_creatures(&self, player: PlayerId) -> Vec<PermanentId> {
        self.battlefield_permanents(PlayerId(1 - player.0))
            .into_iter()
            .filter(|id| {
                self.permanent_is_battlefield_creature(*id)
                    && self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| permanent.tapped)
            })
            .collect()
    }

    /// The next placement of `remaining` counters, if any remain and a
    /// tapped opposing creature can receive one.
    pub(crate) fn distribute_counters(
        &self,
        player: PlayerId,
        kind: CounterKind,
        remaining: usize,
    ) -> Option<Decision> {
        (remaining > 0 && !self.tapped_opponent_creatures(player).is_empty()).then_some(
            Decision::DistributeCounters {
                player,
                kind,
                remaining,
            },
        )
    }

    pub(crate) fn sacrifice_candidates(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| self.permanent_matches_predicate(*id, predicate))
            .collect()
    }

    /// Stable private search candidates, independent of library order.
    pub(crate) fn basic_land_search_candidates(
        &self,
        player: PlayerId,
        subtype: &Option<String>,
        to_hand: bool,
    ) -> Vec<CardId> {
        let mut predicate = CardPredicate {
            subtype: subtype.clone(),
            ..Default::default()
        };
        // Landcycling finds any card of the subtype ("a Plains card"); a
        // search to hand without a subtype is for "a basic land card".
        if !to_hand || subtype.is_none() {
            predicate.card_type = Some(crate::state::card::CardType::Land);
            predicate.supertype = Some("basic".into());
        }
        self.library_search_candidates(player, &predicate)
    }

    /// Library cards matching `predicate`, in a stable order independent of
    /// library order.
    pub(crate) fn library_search_candidates(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> Vec<CardId> {
        let mut candidates: Vec<CardId> = self
            .state
            .zones
            .zone_cards(ZoneType::Library, player)
            .iter()
            .copied()
            .filter(|card| predicate.matches_card(&self.state.cards[*card]))
            .collect();
        candidates.sort_unstable_by_key(|card| (self.state.cards[*card].definition_id, *card));
        candidates
    }

    pub(crate) fn shuffle_library(&mut self, player: PlayerId) {
        let state = &mut *self.state;
        state
            .zones
            .shuffle(ZoneType::Library, player, &mut state.rng);
    }

    /// Reveal before moving; public hand knowledge retains definition counts,
    /// never the identity of the physical copy.
    pub(crate) fn reveal_card_to_hand(&mut self, card: CardId, player: PlayerId) {
        let definition = self.state.cards[card].definition_id;
        self.emit(GameEvent::CardRevealed {
            owner: player,
            definition,
        });
        self.move_card(card, ZoneType::Hand);
        if self.state.zones.contains(card, ZoneType::Hand, player) {
            let count = self.state.players[player.0]
                .known_hand
                .get(&definition)
                .copied()
                .unwrap_or(0);
            self.set_known_hand_count(player, definition, count + 1);
        }
    }

    /// `player` reveals their hand: every card in it becomes public
    /// definition knowledge, as with a card revealed into hand.
    pub(crate) fn reveal_hand(&mut self, player: PlayerId) {
        let mut counts = std::collections::BTreeMap::new();
        for card in self.state.zones.zone_cards(ZoneType::Hand, player).to_vec() {
            let definition = self.state.cards[card].definition_id;
            self.emit(GameEvent::CardRevealed {
                owner: player,
                definition,
            });
            *counts.entry(definition).or_insert(0_u32) += 1;
        }
        for (definition, count) in counts {
            self.set_known_hand_count(player, definition, count);
        }
    }

    /// The controller of `card`'s spell on the stack (CR 110.2), or its
    /// owner once it has left the stack.
    pub(crate) fn spell_caster(&self, card: CardId) -> PlayerId {
        self.find_spell_on_stack_index(card)
            .and_then(|index| match &self.state.stack_objects[index] {
                StackObject::Spell(spell) => Some(spell.controller),
                _ => None,
            })
            .unwrap_or(self.state.cards[card].owner)
    }

    /// The permanents `player` controls, in battlefield order.
    pub(crate) fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card])
            .collect()
    }

    /// The untap step (CR 502.3), less permanents a static effect keeps
    /// from untapping during it.
    pub(crate) fn untap_all_permanents(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if self.has_restriction(permanent_id, Restriction::DoesntUntap) {
                continue;
            }
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                permanent.untap();
            }
        }
        self.invalidate_mana_cache(player);
    }

    pub(crate) fn mark_permanents_not_summoning_sick(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                permanent.summoning_sick = false;
            }
        }
    }

    pub fn draw_cards(&mut self, player: PlayerId, amount: usize) {
        for _ in 0..amount {
            if self.state.zones.size(ZoneType::Library, player) == 0 {
                self.state.players[player.0].drew_when_empty = true;
                break;
            }

            if let Some(card) = self.state.zones.top(ZoneType::Library, player) {
                if let Some(capture) = &mut self.card_capture {
                    capture.drawing = Some(card);
                }
                self.move_card(card, ZoneType::Hand);
                if let Some(capture) = &mut self.card_capture {
                    capture.drawing = None;
                }
                self.state.turn.cards_drawn_this_turn[player.0] += 1;
                self.emit(GameEvent::CardDrawn {
                    player,
                    nth_this_turn: self.state.turn.cards_drawn_this_turn[player.0],
                });
            }
        }
    }

    /// Tap a permanent, emitting a `PermanentTapped` event on the
    /// untapped-to-tapped transition (CR 603.10-style "becomes tapped").
    /// Returns whether the permanent transitioned.
    pub(crate) fn tap_permanent(&mut self, permanent_id: PermanentId, for_mana: bool) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
            return false;
        };
        if permanent.tapped {
            return false;
        }
        permanent.tap();
        let controller = permanent.controller;
        self.invalidate_mana_cache(controller);
        self.emit(GameEvent::PermanentTapped {
            permanent: permanent_id,
            for_mana,
        });
        if for_mana {
            self.fire_triggered_mana_abilities(permanent_id, controller);
        }
        true
    }

    /// Immediate bonus from tapping this source for mana (CR 605.1b).
    /// Waterbend taps are cost payment and never enter this authority.
    pub(crate) fn tap_mana_bonus(
        &self,
        tapped: PermanentId,
        controller: PlayerId,
    ) -> crate::state::mana::Mana {
        let mut produced = crate::state::mana::Mana::default();
        for permanent_id in self.battlefield_permanents(controller) {
            let permanent = self.state.permanents[permanent_id].as_ref().unwrap();
            for ability in &self.state.cards[permanent.card].triggered_mana_abilities {
                if self.permanent_matches_predicate(tapped, &ability.predicate) {
                    produced.add(&ability.mana);
                }
            }
        }
        produced
    }

    fn fire_triggered_mana_abilities(&mut self, tapped: PermanentId, controller: PlayerId) {
        let produced = self.tap_mana_bonus(tapped, controller);
        self.state.players[controller.0].mana_pool.add(&produced);
    }

    /// Create a token from a registered token definition under `controller`'s
    /// control. Tokens are full cards in `GameState::cards`; state-based
    /// actions remove them once they leave the battlefield (CR 704.5d).
    /// Returns the token's permanent, absent if it did not stay on the battlefield.
    pub fn create_token(
        &mut self,
        name: &str,
        controller: PlayerId,
        tapped_and_attacking: bool,
    ) -> Option<PermanentId> {
        let object_id = self.state.id_gen.next_id();
        let Some(card) = self.state.content.instantiate(name, controller, object_id) else {
            debug_assert!(false, "unknown token definition: {name}");
            return None;
        };
        debug_assert!(card.is_token, "created token must be a token definition");

        let card_id = CardId(self.state.cards.len());
        self.state.cards.push(card);
        self.state.card_to_permanent.push(None);
        self.state.object_incarnations.push(Incarnation::INITIAL);
        self.move_card_with_entry(
            card_id,
            ZoneType::Battlefield,
            ZoneEntry {
                tapped: tapped_and_attacking,
                ..ZoneEntry::default()
            },
        );

        if tapped_and_attacking {
            if let Some(permanent_id) = self.state.card_to_permanent[card_id] {
                if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                    permanent.attacking = true;
                }
                // A token put onto the battlefield attacking joins combat but
                // was never declared, so no attack triggers fire (CR 508.4a).
                if let Some(combat) = self.state.combat.as_mut() {
                    combat.attackers.push(permanent_id);
                    combat.attacker_to_blockers.entry(permanent_id).or_default();
                }
            }
        }
        self.invalidate_mana_cache(controller);
        self.state.card_to_permanent[card_id]
    }

    /// Count permanents a player controls matching a predicate (e.g.
    /// "for each Island you control").
    pub(crate) fn count_battlefield_matching(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> usize {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| self.permanent_matches_predicate(*id, predicate))
            .count()
    }

    /// Count cards in a player's graveyard matching a predicate (e.g.
    /// "Lesson cards in your graveyard").
    pub(crate) fn count_graveyard_matching(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> usize {
        self.state
            .zones
            .zone_cards(ZoneType::Graveyard, player)
            .iter()
            .filter(|card_id| predicate.matches_card(&self.state.cards[*card_id]))
            .count()
    }

    pub(crate) fn emit(&mut self, event: GameEvent) {
        crate::agent::policy_history::record(self, &event);
        self.state.pending_events.push(event.clone());
        self.state.observation_events.push(event.clone());
        self.state.events.push(event);
    }

    /// Publish an additive viewer fact in the committed observation window
    /// without changing trigger matching or the rules-event ledger.
    /// Presentation projection consumes these facts; policy encoders filter
    /// their event types.
    pub(crate) fn emit_observation_event(&mut self, event: GameEvent) {
        self.state.observation_events.push(event);
    }

    pub(crate) fn push_spell_to_stack(
        &mut self,
        card: CardId,
        controller: PlayerId,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
        cast: CastChoices,
    ) {
        let cast_from = self.state.zones.zone_of(card).expect("cast source zone");
        let flashback = cast_from == ZoneType::Graveyard;
        self.move_card(card, ZoneType::Stack);
        let id = self.state.id_gen.next_id();
        let source_definition_id = self.state.cards[card].definition_id;
        self.state
            .stack_objects
            .push(StackObject::Spell(SpellOnStack {
                id,
                card,
                controller,
                cast_from,
                flashback,
                source_definition_id,
                targets,
                target_req_indices,
                kicked: cast.kicked,
                x: cast.x,
                mode: None,
            }));
    }

    pub(crate) fn push_to_stack(&mut self, stack_object: StackObject) {
        self.state.stack_objects.push(stack_object);
    }

    pub(crate) fn pop_stack(&mut self) -> Option<StackObject> {
        self.state.stack_objects.pop()
    }

    pub(crate) fn stack_is_empty(&self) -> bool {
        self.state.stack_objects.is_empty()
    }

    pub fn move_card(&mut self, card: CardId, to_zone: ZoneType) {
        self.move_card_with_entry(card, to_zone, ZoneEntry::default());
    }

    pub(crate) fn move_card_with_entry(
        &mut self,
        card: CardId,
        to_zone: ZoneType,
        entry: ZoneEntry,
    ) {
        let from = self.state.zones.zone_of(card);
        let to_zone = if from == Some(ZoneType::Stack) && to_zone != ZoneType::Exile
            && self.find_spell_on_stack_index(card).is_some_and(|index| {
                matches!(&self.state.stack_objects[index], StackObject::Spell(spell) if spell.flashback)
            }) { ZoneType::Exile } else { to_zone };
        let object = from.and_then(|_| self.current_object_ref(card));
        let committed = self.apply_proposed_event(ProposedEvent::ZoneMove {
            card,
            object,
            from,
            to: to_zone,
            entry,
        });
        if committed {
            self.process_game_events();
        }
    }

    pub(crate) fn commit_zone_move(
        &mut self,
        card: CardId,
        departing_ref: Option<ObjectRef>,
        old_zone: Option<ZoneType>,
        to_zone: ZoneType,
        entry: ZoneEntry,
    ) -> bool {
        let owner = self.state.cards[card].owner;

        if old_zone == Some(ZoneType::Hand) && to_zone != ZoneType::Hand {
            let definition = self.state.cards[card].definition_id;
            let count = self.state.players[owner.0]
                .known_hand
                .get(&definition)
                .copied()
                .unwrap_or(0);
            // Public departures remove one known minimum, regardless of which
            // identical copy left. A secret departure invalidates all facts:
            // retaining a definition-specific decrement would leak that card.
            if to_zone == ZoneType::Library || entry.face_down {
                let definitions: Vec<_> = self.state.players[owner.0]
                    .known_hand
                    .keys()
                    .copied()
                    .collect();
                for definition in definitions {
                    self.set_known_hand_count(owner, definition, 0);
                }
            } else if count > 0 {
                self.set_known_hand_count(owner, definition, count - 1);
            }
        }

        let mut event_controller = owner;
        let departing_lki = if old_zone == Some(ZoneType::Battlefield) {
            self.snapshot_current_permanent(card)
        } else {
            None
        };

        if old_zone.is_some() {
            self.advance_object_incarnation(card);
        }

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.card_to_permanent[card].take() {
                if let Some(permanent) = self.state.permanents[permanent_id].as_ref() {
                    event_controller = permanent.controller;
                }
                self.state.permanents[permanent_id] = None;
            }
        }

        if let Some(lki) = departing_lki {
            self.record_object_lki(lki);
        }
        let was_cast =
            old_zone == Some(ZoneType::Stack) && self.find_spell_on_stack_index(card).is_some();
        if old_zone == Some(ZoneType::Stack) {
            if let Some(index) = self.find_spell_on_stack_index(card) {
                self.state.stack_objects.remove(index);
            }
        }

        let holder = match (to_zone, entry.controller) {
            (ZoneType::Battlefield, Some(controller)) => controller,
            _ => owner,
        };
        self.state.zones.move_card(card, holder, to_zone);

        if to_zone == ZoneType::Battlefield {
            let permanent_id = PermanentId(self.state.permanents.len());
            let mut permanent =
                Permanent::new(self.state.id_gen.next_id(), card, &self.state.cards[card]);
            permanent.controller = holder;
            permanent.was_cast = was_cast;
            permanent.tapped = entry.tapped;
            permanent.counters = entry.counters;
            event_controller = permanent.controller;
            self.state.permanents.push(Some(permanent));
            if self.state.card_to_permanent.len() <= card.0 {
                self.state.card_to_permanent.resize(card.0 + 1, None);
            }
            self.state.card_to_permanent[card] = Some(permanent_id);
        }

        if old_zone == Some(ZoneType::Battlefield) || to_zone == ZoneType::Battlefield {
            // The permanent's controller gains or loses its mana abilities,
            // whoever owns the card.
            self.invalidate_mana_cache(event_controller);
        }
        self.capture_card_move(card, old_zone, Some(to_zone), event_controller);
        let event = GameEvent::CardMoved {
            card,
            from: old_zone,
            to: to_zone,
            controller: event_controller,
            face_down: entry.face_down,
        };
        self.emit(event);
        if old_zone == Some(ZoneType::Battlefield) {
            self.handle_leaves_battlefield(departing_ref, to_zone);
        }
        // An attachment that grants or removes abilities changes the mana
        // its creature's controller, possibly the opponent, can produce.
        let card_def = &self.state.cards[card];
        if (old_zone == Some(ZoneType::Battlefield) || to_zone == ZoneType::Battlefield)
            && (card_def.attached_characteristics().is_some()
                || card_def.attached_grants().next().is_some())
        {
            self.invalidate_mana_cache(PlayerId(0));
            self.invalidate_mana_cache(PlayerId(1));
        }
        true
    }

    /// Leave-the-battlefield bookkeeping: fire one-shot delayed triggers
    /// (earthbend returns) and end "exiled until this leaves" durations
    /// (Jailer links — the exiled card returns immediately, no stack).
    fn handle_leaves_battlefield(&mut self, departing_ref: Option<ObjectRef>, to_zone: ZoneType) {
        // Delayed triggers watching this card fire on dies/exiled and are
        // dropped otherwise (they watched this battlefield stint only).
        let watching: Vec<_> = {
            let mut kept = Vec::new();
            let mut fired = Vec::new();
            for trigger in std::mem::take(&mut self.state.delayed_triggers) {
                if departing_ref.is_some_and(|object_ref| trigger.watched == object_ref) {
                    fired.push(trigger);
                } else if departing_ref
                    .is_some_and(|object_ref| trigger.watched.entity == object_ref.entity)
                {
                    // The watched incarnation is already gone. Never retarget
                    // a delayed trigger to a later object of the same entity.
                } else {
                    kept.push(trigger);
                }
            }
            self.state.delayed_triggers = kept;
            fired
        };
        if matches!(to_zone, ZoneType::Graveyard | ZoneType::Exile) {
            for trigger in watching {
                match trigger.kind {
                    crate::flow::trigger::DelayedTriggerKind::ReturnToBattlefieldTapped => {
                        if let Some(source_lki) = self.object_lki(trigger.watched).cloned() {
                            self.enqueue_delayed_trigger(
                                trigger.watched,
                                source_lki,
                                trigger.controller,
                                vec![
                                    crate::state::ability::Effect::ReturnSourceToBattlefieldTapped,
                                ],
                            );
                        }
                    }
                }
            }
        }

        // CR 400.7: continuous effects on the departing object end.
        if let Some(object) = departing_ref {
            self.end_effects_on(object);
        }

        // "Until this creature leaves the battlefield" durations end now:
        // linked exiled cards return under their owners' control.
        let ended: Vec<_> = {
            let mut kept = Vec::new();
            let mut ended = Vec::new();
            for link in std::mem::take(&mut self.state.exile_links) {
                if departing_ref.is_some_and(|object_ref| link.source == object_ref) {
                    ended.push(link);
                } else {
                    kept.push(link);
                }
            }
            self.state.exile_links = kept;
            ended
        };
        for link in ended {
            let owner = self.state.cards[link.exiled_card].owner;
            if self
                .state
                .zones
                .contains(link.exiled_card, ZoneType::Exile, owner)
            {
                self.move_card(link.exiled_card, ZoneType::Battlefield);
            }
        }
    }

    pub(crate) fn assert_stack_consistent(&self) {
        #[cfg(debug_assertions)]
        {
            use std::collections::{BTreeMap, HashSet};

            fn card_counts(cards: &[CardId]) -> BTreeMap<CardId, usize> {
                let mut counts = BTreeMap::new();
                for card in cards {
                    *counts.entry(*card).or_insert(0_usize) += 1;
                }
                counts
            }

            let zone_spell_cards: Vec<CardId> = [PlayerId(0), PlayerId(1)]
                .into_iter()
                .flat_map(|player| {
                    self.state
                        .zones
                        .zone_cards(ZoneType::Stack, player)
                        .iter()
                        .copied()
                })
                .collect();

            let stack_spell_cards: Vec<CardId> = self
                .state
                .stack_objects
                .iter()
                .filter_map(|stack_object| match stack_object {
                    StackObject::Spell(spell) => Some(spell.card),
                    _ => None,
                })
                .collect();

            let mut seen = HashSet::new();
            for card in &stack_spell_cards {
                assert!(
                    seen.insert(*card),
                    "duplicate spell card on stack: {card:?} in {stack_spell_cards:?}"
                );
            }

            assert_eq!(
                card_counts(&zone_spell_cards),
                card_counts(&stack_spell_cards),
                "stack/zone spell mismatch: zone={zone_spell_cards:?}, stack={stack_spell_cards:?}"
            );
        }
    }

    pub(crate) fn find_spell_on_stack_index(&self, card: CardId) -> Option<usize> {
        self.state
            .stack_objects
            .iter()
            .position(|object| matches!(object, StackObject::Spell(spell) if spell.card == card))
    }

    /// The controller of `card` while it is a spell on the stack.
    pub(crate) fn stack_spell_controller(&self, card: CardId) -> Option<PlayerId> {
        self.state
            .stack_objects
            .iter()
            .find_map(|object| match object {
                StackObject::Spell(spell) if spell.card == card => Some(spell.controller),
                _ => None,
            })
    }
}
