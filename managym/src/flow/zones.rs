// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        game_object::{CardId, Incarnation, ObjectRef, PermanentId, PlayerId, Target},
        permanent::Permanent,
        predicate::CardPredicate,
        stack_object::{SpellOnStack, StackObject},
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card])
            .collect()
    }

    pub(crate) fn untap_all_permanents(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
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
                self.move_card(card, ZoneType::Hand);
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

    /// Triggered mana abilities (CR 605.1b): "Whenever you tap a [predicate]
    /// for mana, add [mana]." They don't use the stack — mana is added to
    /// the controller's pool immediately, so it can help pay the very cost
    /// that caused the tap (waterbend composition).
    fn fire_triggered_mana_abilities(&mut self, tapped: PermanentId, controller: PlayerId) {
        let mut produced = Vec::new();
        for permanent_id in self.battlefield_permanents(controller) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            for ability in &self.state.cards[permanent.card].triggered_mana_abilities {
                if self.permanent_matches_predicate(tapped, &ability.predicate) {
                    produced.push(ability.mana.clone());
                }
            }
        }
        for mana in produced {
            self.state.players[controller.0].mana_pool.add(&mana);
        }
    }

    /// Create a token from a registered token definition under `controller`'s
    /// control. Tokens are full cards in `GameState::cards`; state-based
    /// actions remove them once they leave the battlefield (CR 704.5d).
    pub fn create_token(
        &mut self,
        name: &str,
        controller: PlayerId,
        tapped_and_attacking: bool,
    ) {
        let object_id = self.state.id_gen.next_id();
        let Some(card) = self
            .state
            .content
            .instantiate(name, controller, object_id)
        else {
            debug_assert!(false, "unknown token definition: {name}");
            return;
        };
        debug_assert!(card.is_token, "created token must be a token definition");

        let card_id = CardId(self.state.cards.len());
        self.state.cards.push(card);
        self.state.card_to_permanent.push(None);
        self.state.object_incarnations.push(Incarnation::INITIAL);
        self.move_card(card_id, ZoneType::Battlefield);

        if tapped_and_attacking {
            if let Some(permanent_id) = self.state.card_to_permanent[card_id] {
                if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                    permanent.tapped = true;
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
        self.state.pending_events.push(event.clone());
        self.state.observation_events.push(event.clone());
        self.state.events.push(event);
    }

    pub(crate) fn push_spell_to_stack(
        &mut self,
        card: CardId,
        controller: PlayerId,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
        kicked: bool,
    ) {
        self.move_card(card, ZoneType::Stack);
        self.state
            .stack_objects
            .push(StackObject::Spell(SpellOnStack {
                id: self.state.id_gen.next_id(),
                card,
                controller,
                source_definition_id: self.state.cards[card].definition_id,
                targets,
                target_req_indices,
                kicked,
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
        let owner = self.state.cards[card].owner;
        let old_zone = self.state.zones.zone_of(card);
        if old_zone == Some(to_zone) {
            // Reordering within one zone is not a zone change and does not
            // create a new rules object (CR 400.7).
            return;
        }

        let mut event_controller = owner;
        let departing_ref = old_zone.and_then(|_| self.current_object_ref(card));
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
        if old_zone == Some(ZoneType::Stack) {
            if let Some(index) = self.find_spell_on_stack_index(card) {
                self.state.stack_objects.remove(index);
            }
        }

        self.state.zones.move_card(card, owner, to_zone);

        if to_zone == ZoneType::Battlefield {
            let permanent_id = PermanentId(self.state.permanents.len());
            let permanent =
                Permanent::new(self.state.id_gen.next_id(), card, &self.state.cards[card]);
            event_controller = permanent.controller;
            self.state.permanents.push(Some(permanent));
            if self.state.card_to_permanent.len() <= card.0 {
                self.state.card_to_permanent.resize(card.0 + 1, None);
            }
            self.state.card_to_permanent[card] = Some(permanent_id);
        }

        let event = GameEvent::CardMoved {
            card,
            from: old_zone,
            to: to_zone,
            controller: event_controller,
        };
        self.emit(event);
        if old_zone == Some(ZoneType::Battlefield) {
            self.handle_leaves_battlefield(card, departing_ref, to_zone);
        }
        self.process_game_events();
    }

    /// Leave-the-battlefield bookkeeping: fire one-shot delayed triggers
    /// (earthbend returns) and end "exiled until this leaves" durations
    /// (Jailer links — the exiled card returns immediately, no stack).
    fn handle_leaves_battlefield(
        &mut self,
        card: CardId,
        departing_ref: Option<ObjectRef>,
        to_zone: ZoneType,
    ) {
        // Delayed triggers watching this card fire on dies/exiled and are
        // dropped otherwise (they watched this battlefield stint only).
        let watching: Vec<_> = {
            let mut kept = Vec::new();
            let mut fired = Vec::new();
            for trigger in std::mem::take(&mut self.state.delayed_triggers) {
                if trigger.watched_card == card {
                    fired.push(trigger);
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
                        self.enqueue_delayed_trigger(
                            card,
                            trigger.controller,
                            vec![crate::state::ability::Effect::ReturnSourceToBattlefieldTapped],
                        );
                    }
                }
            }
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
                self.invalidate_mana_cache(owner);
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
}
