// zones.rs
// Zone management, card movement, and utility methods on Game.

use std::sync::Arc;

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        decision::Decision,
        event::GameEvent,
        game::{CastChoices, Game},
        proposed_event::{ProposedEvent, ZoneEntry},
        statics::Characteristics,
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        card::{Card, CardDefId, Restriction, StaticAbility},
        effect::{CopyEffect, Modification},
        game_object::{
            CardId, Incarnation, ObjectId, ObjectLki, ObjectRef, PermanentId, PlayerId, Target,
        },
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

    /// Tapped creatures the opponent of `player` controls.
    pub(crate) fn tapped_opponent_creatures(&self, player: PlayerId) -> Vec<PermanentId> {
        self.iter_battlefield_permanents(PlayerId(1 - player.0))
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
        self.iter_battlefield_permanents(player)
            .filter(|id| self.permanent_matches_predicate(*id, predicate))
            .collect()
    }

    /// Zone inspection never reveals a library's order. All consumers use
    /// this stable order for private selection candidates and visible cards.
    pub(crate) fn selection_zone_cards(&self, owner: PlayerId, zone: ZoneType) -> Vec<CardId> {
        let mut cards = self.state.zones.zone_cards(zone, owner).to_vec();
        cards.sort_unstable_by_key(|card| (self.state.cards[*card].definition_id, *card));
        cards
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
        for card in self.state.zones.zone_cards(ZoneType::Hand, player).to_vec() {
            self.emit(GameEvent::CardRevealed {
                owner: player,
                definition: self.state.cards[card].definition_id,
            });
        }
        self.remember_hand(player);
    }

    /// In a two-player game, inspecting the opponent's hand establishes the
    /// same definition minima as revealing it, without a reveal event.
    pub(crate) fn remember_hand(&mut self, player: PlayerId) {
        let mut counts = std::collections::BTreeMap::new();
        for card in self.state.zones.zone_cards(ZoneType::Hand, player) {
            *counts
                .entry(self.state.cards[*card].definition_id)
                .or_insert(0_u32) += 1;
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
        self.iter_battlefield_permanents(player).collect()
    }

    /// Borrowed traversal for readers; mutation paths use the owned snapshot.
    pub(crate) fn iter_battlefield_permanents(
        &self,
        player: PlayerId,
    ) -> impl Iterator<Item = PermanentId> + '_ {
        self.state
            .zones
            .zone_cards(ZoneType::Battlefield, player)
            .iter()
            .filter_map(|card| self.state.card_to_permanent[card])
    }

    /// The untap step (CR 502.3), less permanents a static effect keeps
    /// from untapping during it.
    pub(crate) fn untap_all_permanents(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            if self.has_restriction(permanent_id, Restriction::DoesntUntap)
                || self.doesnt_untap(permanent_id)
            {
                continue;
            }
            self.state.untap(permanent_id);
        }
        self.invalidate_mana_cache(player);
    }

    /// Untap the other players' permanents that untap during each other
    /// player's untap step (Bender's Waterskin), unless they've lost that
    /// ability.
    pub(crate) fn untap_during_other_untap_steps(&mut self, active: PlayerId) {
        let other = PlayerId(1 - active.0);
        for permanent_id in self.battlefield_permanents(other) {
            let untaps = self
                .characteristics(permanent_id)
                .static_abilities
                .iter()
                .any(|ability| matches!(ability, StaticAbility::UntapsDuringOtherUntapSteps));
            if untaps {
                if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                    permanent.untap();
                }
                self.invalidate_mana_cache(other);
            }
        }
    }

    pub(crate) fn mark_permanents_not_summoning_sick(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            self.state.clear_summoning_sickness(permanent_id);
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
                self.state.turn.history[player.0].cards_drawn += 1;
                self.emit(GameEvent::CardDrawn {
                    player,
                    nth_this_turn: self.state.turn.history[player.0].cards_drawn,
                });
            }
        }
    }

    /// Tap a permanent, emitting a `PermanentTapped` event on the
    /// untapped-to-tapped transition (CR 603.10-style "becomes tapped").
    /// Returns whether the permanent transitioned.
    pub(crate) fn tap_permanent(&mut self, permanent_id: PermanentId, for_mana: bool) -> bool {
        let Some(controller) = self.permanent_controller(permanent_id) else {
            return false;
        };
        if !self.state.tap(permanent_id) {
            return false;
        }
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
        self.tap_mana_bonus_from(&self.tap_bonus_sources(controller), tapped)
    }

    /// `controller`'s permanents with triggered mana abilities, read from one
    /// board. A scan pricing several sources at one immutable state reads this
    /// once, so debug validates the board per scan rather than per source.
    pub(crate) fn tap_bonus_sources(&self, controller: PlayerId) -> Vec<Arc<Characteristics>> {
        self.layer_views(self.iter_battlefield_permanents(controller))
            .map(|(_, view)| view)
            .filter(|view| !view.triggered_mana_abilities.is_empty())
            .collect()
    }

    pub(crate) fn tap_mana_bonus_from(
        &self,
        sources: &[Arc<Characteristics>],
        tapped: PermanentId,
    ) -> crate::state::mana::Mana {
        let mut produced = crate::state::mana::Mana::default();
        for view in sources {
            for ability in &view.triggered_mana_abilities {
                if self.permanent_matches_predicate(tapped, &ability.predicate) {
                    produced.add(&ability.mana);
                }
            }
        }
        produced
    }

    /// Untap a permanent; its controller's producible mana changes.
    pub(crate) fn untap_permanent(&mut self, permanent_id: PermanentId) {
        if let Some(controller) = self.permanent_controller(permanent_id) {
            self.state.untap(permanent_id);
            self.invalidate_mana_cache(controller);
        }
    }

    fn permanent_controller(&self, permanent_id: PermanentId) -> Option<PlayerId> {
        self.state.permanents[permanent_id]
            .as_ref()
            .map(|permanent| permanent.controller)
    }

    fn fire_triggered_mana_abilities(&mut self, tapped: PermanentId, controller: PlayerId) {
        let produced = self.tap_mana_bonus(tapped, controller);
        self.state.mana_pool_mut(controller).add(&produced);
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
        let batch = self.event_batch(vec![ProposedEvent::CreateTokens {
            definition: Box::new(crate::flow::proposed_event::TokenDefinition::Registered(
                name.into(),
            )),
            controller,
            count: 1,
            entry: ZoneEntry {
                tapped: tapped_and_attacking,
                ..Default::default()
            },
        }]);
        let results = self.complete_immediate_batch(batch);
        for object in results.iter().filter_map(|result| result.moved) {
            self.finish_created_token(CardId::from(object.entity), tapped_and_attacking);
        }
        results
            .iter()
            .filter_map(|result| result.moved)
            .find_map(|object| self.lookup_current_permanent(object).ok())
    }

    pub(crate) fn prepare_token(&mut self, name: &str, controller: PlayerId) -> Option<CardId> {
        let object_id = self.state.id_gen_mut().next_id();
        let Some(card) = default_content_pack().instantiate(name, controller, object_id) else {
            debug_assert!(false, "unknown token definition: {name}");
            return None;
        };
        debug_assert!(card.is_token, "created token must be a token definition");

        let card_id = self.add_card(card);
        Some(card_id)
    }

    pub(crate) fn finish_created_token(&mut self, card: CardId, attacking: bool) {
        if let Some(id) = self.state.card_to_permanent[card] {
            if attacking {
                self.state.permanents[id].as_mut().expect("token").attacking = true;
                if let Some(combat) = self.state.combat.as_mut() {
                    combat.attackers.push(id);
                    combat.attacker_to_blockers.entry(id).or_default();
                }
            }
        }
    }

    /// Create a token that's a copy of `source` under `controller` (CR 707.2,
    /// 111.1). A card's definition is its copiable values: play never
    /// mutates it, and counters, damage, tapped state and non-copy effects
    /// live on the `Permanent` and in effect records. The source card keeps
    /// its definition after leaving the battlefield, so a departed source is
    /// copied as it last existed there. `exceptions` become part of the
    /// token's copiable values after the source's own (CR 707.9b); the
    /// token's `CopyEffect` records both and the definition at the root.
    pub(crate) fn prepare_token_copy(
        &mut self,
        mut copy: Card,
        controller: PlayerId,
        exceptions: Vec<Modification>,
    ) -> CardId {
        copy.id = self.state.id_gen.next_id();
        copy.owner = controller;
        // Any write detaches the shared definition (`DerefMut`), so a copy
        // of a token without exceptions keeps sharing its source's.
        if !copy.is_token {
            copy.is_token = true;
        }
        for exception in &exceptions {
            let copiable = exception.apply_copiable(&mut copy);
            debug_assert!(copiable, "copy exception {exception:?} is a rules effect");
        }
        let mut effect = copy.copy.take().unwrap_or(CopyEffect {
            of: copy.definition_id,
            exceptions: Vec::new(),
        });
        effect.exceptions.extend(exceptions);
        copy.copy = Some(effect);
        self.add_card(copy)
    }

    /// Add a card created during play to the game, outside every zone.
    fn add_card(&mut self, card: Card) -> CardId {
        let card_id = CardId(self.state.cards.len());
        self.state.cards.push(card);
        self.state.card_to_permanent.push(None);
        self.state.object_incarnations.push(Incarnation::INITIAL);
        card_id
    }

    /// Count permanents a player controls matching a predicate (e.g.
    /// "for each Island you control").
    pub(crate) fn count_battlefield_matching(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> usize {
        self.iter_battlefield_permanents(player)
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
            .filter(|card_id| self.card_matches_predicate(**card_id, predicate))
            .count()
    }

    pub(crate) fn emit(&mut self, event: GameEvent) {
        crate::agent::policy_history::record(self, &event);
        self.state.pending_events_mut().push(event.clone());
        self.state.observation_events_mut().push(event.clone());
        self.state.events_mut().push(event);
    }

    /// Publish an additive viewer fact in the committed observation window
    /// without changing trigger matching or the rules-event ledger.
    /// Presentation projection consumes these facts; policy encoders filter
    /// their event types.
    pub(crate) fn emit_observation_event(&mut self, event: GameEvent) {
        self.state.observation_events_mut().push(event);
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
        let id = self.state.id_gen_mut().next_id();
        let source_definition_id = self.state.cards[card].definition_id;
        self.state
            .stack_objects_mut()
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
        self.state.stack_objects_mut().push(stack_object);
    }

    pub(crate) fn pop_stack(&mut self) -> Option<StackObject> {
        self.state.stack_objects_mut().pop()
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
        let event = self.zone_move_proposal(card, to_zone, entry);
        self.apply_immediate_event(event);
    }

    pub(crate) fn zone_move_proposal(
        &self,
        card: CardId,
        to_zone: ZoneType,
        mut entry: ZoneEntry,
    ) -> ProposedEvent {
        let from = self.state.zones.zone_of(card);
        let to_zone = if from == Some(ZoneType::Stack) && to_zone != ZoneType::Exile
            && self.find_spell_on_stack_index(card).is_some_and(|index| {
                matches!(&self.state.stack_objects[index], StackObject::Spell(spell) if spell.flashback)
            }) { ZoneType::Exile } else { to_zone };
        let object = from.and_then(|_| self.current_object_ref(card));
        entry = entry.for_zone(to_zone);
        if to_zone == ZoneType::Battlefield && self.enters_as_saga(card, &entry) {
            entry.counters.change(CounterKind::Lore, 1);
        }
        ProposedEvent::ZoneMove {
            card,
            object,
            from,
            to: to_zone,
            entry,
        }
    }

    /// Whether the face that enters is a Saga: the back face when it enters
    /// transformed (CR 712.14), so a Saga returning as a creature gets no lore.
    fn enters_as_saga(&self, card: CardId, entry: &ZoneEntry) -> bool {
        let printed = &self.state.cards[card];
        if entry.transformed {
            if let Some((_, back)) = default_content_pack().back_face(printed.definition_id) {
                return back.is_saga();
            }
        }
        printed.is_saga()
    }

    pub(crate) fn commit_zone_move(
        &mut self,
        card: CardId,
        departing_ref: Option<ObjectRef>,
        old_zone: Option<ZoneType>,
        to_zone: ZoneType,
        entry: ZoneEntry,
        departing_lki: Option<ObjectLki>,
        followups: &mut Vec<ProposedEvent>,
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
        if let Some(old_zone) = old_zone {
            self.state.advance_incarnation(card, old_zone);
        }

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.card_to_permanent[card].take() {
                if let Some(permanent) = self.state.permanents[permanent_id].as_ref() {
                    event_controller = permanent.controller;
                }
                self.state.permanents[permanent_id] = None;
            }
            // CR 712.8a: a double-faced card is front face up outside the
            // battlefield.
            self.state.cards[card].turn_front_face_up();
        }

        if let Some(lki) = departing_lki {
            self.record_object_lki(lki);
        }
        let cast_spell = (old_zone == Some(ZoneType::Stack))
            .then(|| self.find_spell_on_stack_index(card))
            .flatten()
            .map(|index| self.state.stack_objects_mut().remove(index));
        let was_cast = cast_spell.is_some();
        let cast_x = match cast_spell {
            Some(StackObject::Spell(spell)) => spell.x,
            _ => 0,
        };

        let holder = match (to_zone, entry.controller) {
            (ZoneType::Battlefield, Some(controller)) => controller,
            _ => owner,
        };
        self.state.move_card(card, holder, to_zone);

        if to_zone == ZoneType::Battlefield {
            if entry.transformed {
                self.show_back_face(card);
            }
            let permanent_id = PermanentId(self.state.permanents.len());
            let mut permanent = Permanent::new(
                self.state.id_gen_mut().next_id(),
                card,
                &self.state.cards[card],
            );
            permanent.controller = holder;
            permanent.was_cast = was_cast;
            permanent.cast_x = cast_x;
            permanent.tapped = entry.tapped;
            permanent.counters = entry.counters;
            event_controller = permanent.controller;
            self.state.permanents.push(Some(permanent));
            if self.state.cards[card].types.is_land() {
                self.state.turn.history[holder.0].land_entered = true;
            }
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
        if to_zone == ZoneType::Battlefield {
            if let Some(permanent) = self.state.card_to_permanent[card] {
                let counters = self.state.permanents[permanent]
                    .as_ref()
                    .expect("entered")
                    .counters
                    .clone();
                for (kind, count) in counters.iter() {
                    self.emit(GameEvent::CountersPut {
                        permanent,
                        kind,
                        count,
                        total: count,
                        by: holder,
                    });
                }
            }
        }
        if old_zone == Some(ZoneType::Battlefield) {
            self.handle_leaves_battlefield(departing_ref, to_zone, followups);
        }
        // An attachment that grants or removes abilities changes the mana
        // its creature's controller, possibly the opponent, can produce.
        let card_def = &self.state.cards[card];
        if (old_zone == Some(ZoneType::Battlefield) || to_zone == ZoneType::Battlefield)
            && (card_def.attached_characteristics().is_some()
                || card_def.attached_grants().next().is_some()
                || card_def
                    .statics
                    .iter()
                    .any(|ability| matches!(ability, StaticAbility::SetsLandType(_))))
        {
            self.invalidate_mana_cache(PlayerId(0));
            self.invalidate_mana_cache(PlayerId(1));
        }
        true
    }

    /// `player` bent: record the kind in their turn history and emit the
    /// event "whenever you waterbend, earthbend, firebend, or airbend" reads.
    pub(crate) fn record_bending(&mut self, player: PlayerId, kind: crate::flow::turn::Bending) {
        self.state.turn.history[player.0].bent |= kind as u8;
        self.emit(GameEvent::Bent { player, kind });
    }

    /// Transform a double-faced permanent (CR 701.27a): it turns over and
    /// stays the same object, keeping its counters, damage, attachments and
    /// combat status (CR 701.27a). Anything else can't transform and stays as
    /// it is (CR 701.27c).
    pub fn transform_permanent(&mut self, permanent: PermanentId) {
        let Some(card) = self.state.permanents[permanent].as_ref().map(|p| p.card) else {
            return;
        };
        if self.state.cards[card].is_transformed() {
            self.state.cards[card].turn_front_face_up();
        } else if !self.show_back_face(card) {
            return;
        }
        self.state.permanents[permanent]
            .as_mut()
            .unwrap()
            .transforms += 1;
        // Its mana abilities and static grants may change for either player.
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
    }

    /// Show a card's back face; false when it has none.
    fn show_back_face(&mut self, card: CardId) -> bool {
        let definition = self.state.cards[card].definition_id;
        let Some(back) = default_content_pack().back_face(definition) else {
            return false;
        };
        self.state.cards[card].turn_back_face_up(back);
        true
    }

    /// Give `controller` control of a battlefield permanent (CR 108.4): it
    /// moves to their battlefield without changing zones, and has not been
    /// under their control continuously since their turn began (CR 302.6).
    /// "For as long as you control" effects of its previous controller end.
    pub(crate) fn change_control(&mut self, permanent_id: PermanentId, controller: PlayerId) {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return;
        };
        if permanent.controller == controller {
            return;
        }
        let (card, previous) = (permanent.card, permanent.controller);
        let permanent = self.state.permanents[permanent_id].as_mut().unwrap();
        permanent.controller = controller;
        permanent.summoning_sick = true;
        permanent.attacking = false;
        self.state
            .zones
            .move_card(card, controller, ZoneType::Battlefield);
        self.invalidate_mana_cache(previous);
        self.invalidate_mana_cache(controller);
        if let Some(object) = self.permanent_object_ref(permanent_id) {
            self.end_effects_for_control_change(object, controller);
        }
    }

    /// Leave-the-battlefield bookkeeping: fire delayed triggers watching the
    /// object (earthbend returns) and end "exiled until this leaves" durations
    /// (Jailer links — the exiled card returns immediately, no stack).
    fn handle_leaves_battlefield(
        &mut self,
        departing_ref: Option<ObjectRef>,
        to_zone: ZoneType,
        followups: &mut Vec<ProposedEvent>,
    ) {
        // Delayed triggers watching this object fire as it dies (or is
        // exiled, for earthbend) and are dropped otherwise: they watched this
        // battlefield stint only.
        let watches = |trigger: &DelayedTrigger| match trigger.event {
            DelayedTriggerEvent::Leaves { watched, .. } => {
                departing_ref.is_some_and(|object_ref| watched.entity == object_ref.entity)
            }
            _ => false,
        };
        if self.state.delayed_triggers.iter().any(watches) {
            let (watching, kept): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.state.delayed_triggers)
                    .into_iter()
                    .partition(watches);
            *self.state.delayed_triggers_mut() = kept;
            let departed = departing_ref
                .and_then(|object_ref| self.object_lki(object_ref))
                .map(|lki| Target::Permanent(lki.permanent));
            for trigger in watching {
                let DelayedTriggerEvent::Leaves { watched, or_exiled } = trigger.event else {
                    continue;
                };
                // Never retarget a delayed trigger to a later object of the
                // same entity.
                if Some(watched) == departing_ref
                    && (to_zone == ZoneType::Graveyard || (or_exiled && to_zone == ZoneType::Exile))
                {
                    self.fire_delayed_trigger(&trigger, departed);
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
        for link in ended.into_iter().filter(|link| link.returns) {
            let owner = self.state.cards[link.exiled_card].owner;
            if self
                .state
                .zones
                .contains(link.exiled_card, ZoneType::Exile, owner)
            {
                followups.push(self.zone_move_proposal(
                    link.exiled_card,
                    ZoneType::Battlefield,
                    ZoneEntry::default(),
                ));
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

    /// The stack position of the activated or triggered ability `id`.
    pub(crate) fn find_ability_on_stack_index(&self, id: ObjectId) -> Option<usize> {
        self.state
            .stack_objects
            .iter()
            .position(|object| !matches!(object, StackObject::Spell(_)) && object.id() == id)
    }

    /// The card whose ability `id` is on the stack.
    pub(crate) fn stack_ability_source(&self, id: ObjectId) -> Option<CardId> {
        let index = self.find_ability_on_stack_index(id)?;
        match &self.state.stack_objects[index] {
            StackObject::ActivatedAbility(ability) => Some(ability.source_card),
            StackObject::TriggeredAbility(triggered) => Some(triggered.source_card),
            StackObject::Spell(_) => None,
        }
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
