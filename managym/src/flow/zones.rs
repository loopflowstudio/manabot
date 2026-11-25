// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        decision::Decision,
        event::GameEvent,
        game::{CastChoices, Game},
        proposed_event::{EventBinding, ProposedEvent, ZoneEntry},
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        card::{CardDefId, Restriction, StaticAbility},
        effect::{CopyEffect, Modification},
        game_object::{
            CardId, Object, ObjectId, ObjectLki, PermanentId, PlayerId, PublicId, Target,
        },
        payment::PaymentReceipt,
        permanent::CounterKind,
        permanent::Permanent,
        predicate::CardPredicate,
        stack_object::{SpellLki, SpellOnStack, StackObject},
        zone::ZoneType,
    },
};

/// CR 400.7d: a permanent can read how its spell was cast. Entry replacements
/// belong to `ZoneEntry`, separately from facts about the departing object.
#[derive(Clone, Debug, Default)]
struct CastFacts {
    x: u8,
    spent: PaymentReceipt,
}

impl Game {
    pub(crate) fn set_known_hand_count(
        &mut self,
        player: PlayerId,
        definition: CardDefId,
        count: u32,
    ) {
        let known = self.state.known_hand_mut(player);
        if count == 0 {
            known.remove(&definition);
        } else {
            known.insert(definition, count);
        }
    }

    /// Retain only independent minima after a hand departure. A definition
    /// is supplied only when the opponent can identify the departing card.
    pub(crate) fn forget_hand_card(&mut self, player: PlayerId, definition: Option<CardDefId>) {
        let known = self.state.known_hand_mut(player);
        if let Some(definition) = definition {
            if let Some(count) = known.get_mut(&definition) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    known.remove(&definition);
                }
            }
        } else {
            known.retain(|_, count| {
                *count = count.saturating_sub(1);
                *count > 0
            });
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
    pub(crate) fn selection_zone_objects(&self, owner: PlayerId, zone: ZoneType) -> Vec<ObjectId> {
        let mut objects = self.state.zones.zone_objects(zone, owner).to_vec();
        objects.sort_unstable_by_key(|object| (self.state[*object].definition_id, *object));
        objects
    }

    pub(crate) fn selection_zone_cards(&self, owner: PlayerId, zone: ZoneType) -> Vec<CardId> {
        self.selection_zone_objects(owner, zone)
            .into_iter()
            .filter_map(|object| self.state.card_of(object))
            .collect()
    }

    /// Whether `object` is the top card of its owner's library, so moving it
    /// shifts every remembered library position up by one.
    pub(super) fn is_library_top(&self, object: ObjectId) -> bool {
        self.state
            .zones
            .top(ZoneType::Library, self.state[object].owner)
            == Some(object)
    }

    pub(crate) fn shuffle_library(&mut self, player: PlayerId) {
        self.state.library_order_pins_mut(player).clear();
        self.state.known_library_mut(player).forget_order();
        let state = &mut *self.state;
        state
            .zones
            .shuffle(ZoneType::Library, player, &mut state.rng);
        self.assert_library_knowledge();
    }

    /// Reveal before moving; public hand knowledge retains definition counts,
    /// never the identity of the physical copy. `entry` says whether the
    /// card's library position was public as it left.
    pub(crate) fn reveal_object_to_hand(
        &mut self,
        object: ObjectId,
        player: PlayerId,
        entry: ZoneEntry,
    ) {
        let definition = self.state.entry_object(object).definition_id;
        self.emit(GameEvent::CardRevealed {
            owner: player,
            definition,
        });
        let before = self.state.players[player.0]
            .known_hand
            .get(&definition)
            .copied()
            .unwrap_or(0);
        let moved = self.move_object_with_entry(object, ZoneType::Hand, entry);
        if self.state.object(moved).is_ok()
            && self.state.zones.contains(moved, ZoneType::Hand, player)
        {
            self.set_known_hand_count(player, definition, before + 1);
        }
    }

    /// `player` reveals their hand: every card in it becomes public
    /// definition knowledge, as with a card revealed into hand.
    pub(crate) fn reveal_hand(&mut self, player: PlayerId) {
        for object in self
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .to_vec()
        {
            self.emit(GameEvent::CardRevealed {
                owner: player,
                definition: self.state[object].definition_id,
            });
        }
        self.remember_hand(player);
    }

    /// In a two-player game, inspecting the opponent's hand establishes the
    /// same definition minima as revealing it, without a reveal event.
    pub(crate) fn remember_hand(&mut self, player: PlayerId) {
        let mut counts = std::collections::BTreeMap::new();
        for object in self.state.zones.zone_objects(ZoneType::Hand, player).iter() {
            *counts
                .entry(self.state[*object].definition_id)
                .or_insert(0_u32) += 1;
        }
        for (definition, count) in counts {
            self.set_known_hand_count(player, definition, count);
        }
    }

    /// The permanents `player` controls, in battlefield order.
    pub(crate) fn battlefield_permanents(&self, player: PlayerId) -> Vec<PermanentId> {
        self.iter_battlefield_permanents(player).collect()
    }

    /// Borrowed traversal for readers; mutation paths use the owned snapshot.
    /// Phased-out permanents are treated as though they don't exist
    /// (CR 702.26b), so every rules reader skips them here.
    pub(crate) fn iter_battlefield_permanents(
        &self,
        player: PlayerId,
    ) -> impl Iterator<Item = PermanentId> + '_ {
        self.iter_battlefield_including_phased_out(player)
            .filter(|&id| !self.is_phased_out(id))
    }

    /// Every permanent in `player`'s battlefield zone, phased out or not:
    /// the observation's rows, which show phased-out permanents with their
    /// record (they are public, CR 702.26b).
    pub(crate) fn iter_battlefield_including_phased_out(
        &self,
        player: PlayerId,
    ) -> impl Iterator<Item = PermanentId> + '_ {
        self.state
            .zones
            .battlefield_objects(player)
            .iter()
            .filter_map(|object| self.state.permanent_of(*object))
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
            self.untap_permanent(permanent_id);
        }
    }

    /// Untap the other players' permanents that untap during each other
    /// player's untap step (Bender's Waterskin), unless they've lost that
    /// ability.
    pub(crate) fn untap_during_other_untap_steps(&mut self, active: PlayerId) {
        let other = PlayerId(1 - active.0);
        for permanent_id in self.battlefield_permanents(other) {
            if self
                .layer_view(permanent_id)
                .static_abilities
                .contains(&StaticAbility::UntapsDuringOtherUntapSteps)
            {
                self.untap_permanent(permanent_id);
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

            if let Some(object) = self.state.zones.top(ZoneType::Library, player) {
                if self.card_capture.is_some() || self.statistics.is_some() {
                    self.capture_drawing = Some(object);
                }
                self.move_object_with_entry(
                    object,
                    ZoneType::Hand,
                    ZoneEntry {
                        library_top_departure: true,
                        ..Default::default()
                    },
                );
                self.capture_drawing = None;
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
    /// Returns whether the permanent transitioned. Tapping for mana triggers
    /// nothing here: triggered mana abilities wait for the mana ability to
    /// produce mana (CR 106.12a; `flow/mana.rs`).
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
        true
    }

    /// Untap a permanent, emitting `PermanentUntapped` on the tapped-to-
    /// untapped transition ("becomes untapped"); a stun counter removed
    /// instead (CR 122.1d) or an untapped permanent emits nothing. Every
    /// untap, the untap step's included, comes through here. Returns whether
    /// the permanent transitioned.
    pub(crate) fn untap_permanent(&mut self, permanent_id: PermanentId) -> bool {
        let Some(controller) = self.permanent_controller(permanent_id) else {
            return false;
        };
        let untapped = self.state.untap(permanent_id);
        self.invalidate_mana_cache(controller);
        if untapped {
            self.emit(GameEvent::PermanentUntapped {
                permanent: permanent_id,
            });
        }
        untapped
    }

    fn permanent_controller(&self, permanent_id: PermanentId) -> Option<PlayerId> {
        self.state.permanents[permanent_id]
            .as_ref()
            .map(|permanent| permanent.controller)
    }

    /// Create a token from the token definition keyed `key` (`token.<name>`)
    /// under `controller`'s control. A token is an `ObjectKind::Token` object in
    /// `GameState::objects`, never a card; state-based actions remove it once it
    /// leaves the battlefield (CR 704.5d).
    /// Returns the token's permanent, absent if it did not stay on the battlefield.
    pub fn create_token(
        &mut self,
        key: &str,
        controller: PlayerId,
        tapped_and_attacking: bool,
    ) -> Option<PermanentId> {
        let batch = self.event_batch(
            EventBinding::None,
            [ProposedEvent::CreateTokens {
                definition: Box::new(crate::flow::proposed_event::TokenSource::Registered(
                    key.into(),
                )),
                controller,
                count: 1,
                entry: ZoneEntry {
                    tapped: tapped_and_attacking,
                    ..Default::default()
                },
            }],
        );
        let results = self.complete_immediate_batch(batch);
        for object in results.iter().filter_map(|result| result.moved) {
            self.finish_created_token(object, tapped_and_attacking);
        }
        results
            .iter()
            .filter_map(|result| result.moved)
            .find_map(|object| self.lookup_current_permanent(object).ok())
    }

    pub(crate) fn prepare_token(&mut self, key: &str, controller: PlayerId) -> Option<ObjectId> {
        let public_id = self.state.id_gen_mut().next_id();
        let Some(object) = default_content_pack().instantiate_token(key, controller, public_id)
        else {
            debug_assert!(false, "{key:?} is not a key in the token catalog");
            return None;
        };

        Some(self.add_object(object))
    }

    pub(crate) fn finish_created_token(&mut self, object: ObjectId, attacking: bool) {
        if let Some(id) = self.state.permanent_of(object) {
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
    /// live on the `Permanent` and in effect records. The caller supplies
    /// the copiable source values; this allocator never looks up a departed
    /// appearance through its storage slot. `exceptions` become part of the
    /// token's copiable values after the source's own (CR 707.9b); the
    /// token's `CopyEffect` records both and the definition at the root.
    pub(crate) fn prepare_token_copy(
        &mut self,
        copy: Object,
        controller: PlayerId,
        exceptions: Vec<Modification>,
    ) -> ObjectId {
        let id = self.state.id_gen.next_id();
        // Token status is the object's kind, not a copiable value; only an
        // exception's write detaches the shared definition (`DerefMut`).
        let mut copy = Object::token_copy(copy, id, controller);
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
        self.add_object(copy)
    }

    /// Allocate a newly created token or spell copy outside every zone.
    /// All companion stores grow here, in the same creation order.
    fn add_object(&mut self, object: Object) -> ObjectId {
        self.state.objects.push(object);
        self.state.object_to_permanent.push(None);
        self.state.zones.allocate()
    }

    /// A spell copy starts on the stack at incarnation zero. Capture its
    /// creation without emitting a rules zone-change event.
    pub(crate) fn create_spell_copy(
        &mut self,
        source: Object,
        id: PublicId,
        controller: PlayerId,
    ) -> ObjectId {
        let copy = Object::spell_copy(source, id, controller);
        let object = self.add_object(copy);
        self.state.place_object(object, controller, ZoneType::Stack);
        self.capture_object_move(object, None, Some(ZoneType::Stack), controller);
        object
    }

    pub(super) fn prepared_copy(&self, source: ObjectId) -> Option<ObjectId> {
        self.state
            .cast_permissions
            .grants
            .iter()
            .find(|grant| {
                grant.prepared
                    && grant.source == Some(source)
                    && self.state.zones.zone_of(grant.object) == Some(ZoneType::Exile)
            })
            .map(|grant| grant.object)
    }

    pub(super) fn unprepare_inset(&mut self, source: ObjectId) {
        let Some(copy) = self.prepared_copy(source) else {
            return;
        };
        let owner = self.state[copy].owner;
        self.state.zones.remove_object(copy);
        self.capture_object_move(copy, Some(ZoneType::Exile), None, owner);
        self.state
            .cast_permissions
            .grants
            .retain(|grant| !(grant.prepared && grant.source == Some(source)));
    }

    pub(super) fn prepare_inset(&mut self, source: ObjectId) {
        use crate::{flow::cast_permission::PlayGrant, state::card::InsetSpell};
        let Some(InsetSpell::Prepared(program)) = &self.state[source].inset else {
            return;
        };
        if self.prepared_copy(source).is_some() {
            return;
        }
        let Ok(id) = self.lookup_current_permanent(source) else {
            return;
        };
        let controller = self.state.permanents[id].as_ref().unwrap().controller;
        let program = program.clone();
        let id = self.state.id_gen.next_id();
        let copy = Object::from_definition(
            id,
            controller,
            self.state[source].definition_id,
            program,
            false,
        );
        let copy = Object::spell_copy(copy, id, controller);
        let object = self.add_object(copy);
        self.state.place_object(object, controller, ZoneType::Exile);
        self.capture_object_move(object, None, Some(ZoneType::Exile), controller);
        self.state.cast_permissions.grant(PlayGrant {
            source: Some(source),
            prepared: true,
            follows_source_controller: true,
            ..PlayGrant::new(object, controller)
        });
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
            .zone_cards(ZoneType::Graveyard, player)
            .filter(|card| self.object_matches_predicate(self.state.object_of(*card), predicate))
            .count()
    }

    pub(crate) fn emit(&mut self, mut event: GameEvent) {
        crate::agent::policy_history::record(self, &event);
        // A pending gain keeps its ordinal even if more gains commit before
        // trigger matching. Life loss counts the applied amount (CR 119.3).
        if let GameEvent::LifeChanged {
            player,
            old,
            new,
            gain_ordinal,
        } = &mut event
        {
            if new > old {
                let history = &mut self.state.turn.history[player.0];
                history.life_gained += old.abs_diff(*new);
                history.life_gains += 1;
                *gain_ordinal = history.life_gains;
            } else if new < old {
                self.state.turn.history[player.0].life_lost += old.abs_diff(*new);
            }
        }
        self.state.pending_events_mut().push(event.clone());
        self.state.observation_events_mut().push(event.clone());
        self.state.events_mut().push(event);
    }

    /// Offer `event` to triggered abilities only: it joins no ledger,
    /// observation or policy history, because it restates what those
    /// already record in another shape.
    pub(crate) fn emit_trigger_event(&mut self, event: GameEvent) {
        self.state.pending_events_mut().push(event);
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
        object: ObjectId,
        controller: PlayerId,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
        cast: CastChoices,
    ) -> ObjectId {
        self.spend_play_route(controller, cast.route);
        let cast_from = self.state.zones.zone_of(object).expect("cast source zone");
        // CR 400.7g, h: the spell is the object the card became on the stack.
        let object = self.move_object_with_entry(
            object,
            ZoneType::Stack,
            ZoneEntry {
                profile: cast.profile,
                library_top_departure: self.is_library_top(object),
                ..Default::default()
            },
        );
        let id = self.state.id_gen_mut().next_id();
        let source_definition_id = self.state[object].definition_id;
        self.state
            .stack_objects_mut()
            .push(StackObject::Spell(SpellOnStack {
                id,
                object,
                controller,
                cast_from,
                flashback: cast.flashback,
                from_prepared: cast.from_prepared,
                profile: cast.profile,
                exile_instead_of_graveyard: cast.exile_instead_of_graveyard,
                source_definition_id,
                targets,
                target_req_indices,
                kicked: cast.kicked,
                x: cast.x,
                mode: None,
                spent: Default::default(),
            }));
        object
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

    /// Return the object this move made (CR 400.7j), even if a consequent
    /// event moves it again. A move that does not happen (the object is
    /// already there, or already gone) returns `object` itself.
    pub fn move_object(&mut self, object: ObjectId, to_zone: ZoneType) -> ObjectId {
        self.move_object_with_entry(object, to_zone, ZoneEntry::default())
    }

    pub(crate) fn move_object_with_entry(
        &mut self,
        object: ObjectId,
        to_zone: ZoneType,
        entry: ZoneEntry,
    ) -> ObjectId {
        if !self.state.zones.is_latest(object) {
            return object;
        }
        let event = self.zone_move_proposal(object, to_zone, entry);
        let batch = self.event_batch(EventBinding::None, [event]);
        self.complete_immediate_batch(batch)
            .into_iter()
            .next()
            .and_then(|result| result.moved)
            .unwrap_or(object)
    }

    pub(crate) fn zone_move_proposal(
        &self,
        object_id: ObjectId,
        to_zone: ZoneType,
        mut entry: ZoneEntry,
    ) -> ProposedEvent {
        let from = self.state.zones.zone_of(object_id);
        let to_zone = if from == Some(ZoneType::Stack)
            && to_zone != ZoneType::Exile
            && self
                .stack_spell(object_id)
                .is_some_and(|spell| spell.flashback)
        {
            ZoneType::Exile
        } else {
            to_zone
        };
        entry = entry.for_zone(to_zone);
        if to_zone == ZoneType::Battlefield && self.enters_as_saga(object_id, &entry) {
            entry.counters.change(CounterKind::Lore, 1);
        }
        ProposedEvent::ZoneMove {
            object_id,
            from,
            to: to_zone,
            entry,
        }
    }

    /// Whether the face that enters is a Saga: the back face when it enters
    /// transformed (CR 712.14), so a Saga returning as a creature gets no lore.
    fn enters_as_saga(&self, object: ObjectId, entry: &ZoneEntry) -> bool {
        let printed = self.state.entry_object(object);
        if entry.transformed {
            if let Some((_, back)) = default_content_pack().back_face(printed.definition_id) {
                return back.is_saga();
            }
        }
        printed.is_saga()
    }

    /// Return the appearance minted by this move (CR 400.7), including
    /// the initial appearance of an object entering its first zone.
    pub(crate) fn commit_zone_move(
        &mut self,
        prior_object: ObjectId,
        old_zone: Option<ZoneType>,
        to_zone: ZoneType,
        entry: ZoneEntry,
        departing_lki: Option<ObjectLki>,
        followups: &mut Vec<ProposedEvent>,
    ) -> ObjectId {
        let prior = self.state.entry_object(prior_object);
        let (owner, definition) = (prior.owner, prior.definition_id);
        let public_source = old_zone.is_some_and(|zone| {
            !matches!(zone, ZoneType::Hand | ZoneType::Library)
                && !(zone == ZoneType::Exile && self.is_face_down_in_exile(prior_object))
        });
        let known_top = if old_zone == Some(ZoneType::Library) {
            self.state
                .library_order_pins_mut(owner)
                .remove(&prior_object);
            if entry.library_top_departure {
                self.state.known_library_mut(owner).remove_top()
            } else if entry.inspected_library_departure {
                let position = self
                    .library_from_top(owner)
                    .position(|object| object == prior_object)
                    .expect("departing inspected card in library");
                self.state
                    .known_library_mut(owner)
                    .remove_position(position)
            } else {
                self.state.known_library_mut(owner).forget_order();
                None
            }
        } else {
            None
        };
        if old_zone == Some(ZoneType::Hand) && to_zone != ZoneType::Hand {
            let seen = (to_zone != ZoneType::Library && !entry.face_down).then_some(definition);
            self.forget_hand_card(owner, seen);
        }

        let holder = match (to_zone, entry.controller) {
            (ZoneType::Battlefield, Some(controller)) => controller,
            _ => owner,
        };
        let event_controller = departing_lki.as_ref().map_or(holder, |lki| lki.controller);
        if let Some(lki) = departing_lki {
            let history = &mut self.state.turn.history[lki.controller.0];
            history.permanents_left += 1;
            history.nonland_permanents_left += u32::from(!lki.was_land());
            history.creatures_died +=
                u32::from(to_zone == ZoneType::Graveyard && lki.was_creature());
            if to_zone == ZoneType::Graveyard && self.state.combat.is_some() {
                if let Some(exposure) = self.exposure_mut() {
                    exposure.combat.deaths.insert(lki.permanent);
                }
            }
            self.state.object_to_permanent[prior_object.slot()] = None;
            self.state.permanents[lki.permanent] = None;
            self.record_object_lki(lki);
        }
        let cast = match (old_zone == Some(ZoneType::Stack))
            .then(|| self.stack_index_of(Target::StackSpell(prior_object)))
            .flatten()
            .map(|index| self.state.stack_objects_mut().remove(index))
        {
            Some(StackObject::Spell(spell)) => {
                let mut lki = self.spell_lki(&spell);
                // Only triggers still referring to this stack appearance can copy it.
                // Capture after any retargets, before the zone change restores its face.
                let context = Some(Target::StackSpell(spell.object));
                if self
                    .unplaced_triggers()
                    .any(|trigger| trigger.context == context)
                    || self.state.stack_objects.iter().any(|item| {
                        matches!(item,
                        StackObject::TriggeredAbility(trigger) if trigger.context == context)
                    })
                {
                    lki.copiable = Some(std::sync::Arc::new(
                        crate::state::stack_object::CopiableSpell {
                            spell: spell.clone(),
                            values: self.state[spell.object].copiable_values(),
                        },
                    ));
                }
                self.state.spell_lki_mut().insert(spell.object, lki);
                (to_zone == ZoneType::Battlefield).then_some(CastFacts {
                    x: spell.x,
                    spent: spell.spent,
                })
            }
            _ => None,
        };

        let object_id = self
            .state
            .mint_appearance(prior_object, holder, to_zone, entry.entry_copy);
        match entry.profile {
            crate::state::card::SpellProfile::BackFace => {
                assert!(self.show_back_face(object_id), "offered back face")
            }
            profile => self.state[object_id].show_inset(profile),
        }
        if let Some(player) = entry.adventure_player {
            if self.state[object_id].is_card() {
                self.state
                    .cast_permissions
                    .grant(crate::flow::cast_permission::PlayGrant {
                        on_adventure: true,
                        ..crate::flow::cast_permission::PlayGrant::new(object_id, player)
                    });
            }
        }
        let entered_definition = self.state[object_id].definition_id;
        if to_zone == ZoneType::Library {
            self.state.library_order_pins_mut(owner).insert(object_id);
            // A copied or transformed source can restore a different printing.
            // Never carry its old visible definition into the hidden zone.
            let same_definition = entered_definition == definition;
            let public = (public_source && same_definition).then_some(entered_definition);
            let own = ((public_source || old_zone == Some(ZoneType::Hand)) && same_definition)
                .then_some(entered_definition);
            self.state.known_library_mut(owner).insert_top(public, own);
        }
        if to_zone == ZoneType::Hand && known_top == Some(entered_definition) {
            *self
                .state
                .known_hand_mut(owner)
                .entry(entered_definition)
                .or_default() += 1;
        }
        if old_zone == Some(ZoneType::Battlefield) && to_zone == ZoneType::Graveyard {
            self.state.turn.graveyard_from_battlefield.push(object_id);
        }
        if to_zone == ZoneType::Exile && !entry.counters.is_empty() {
            self.state
                .object_counters
                .insert(object_id, entry.counters.clone());
        }

        let entered_permanent = (to_zone == ZoneType::Battlefield).then(|| {
            if entry.transformed {
                self.show_back_face(object_id);
            }
            let permanent_id = PermanentId(self.state.permanents.len());
            let mut permanent = Permanent::new(
                self.state.id_gen_mut().next_id(),
                object_id,
                &self.state[object_id],
            );
            permanent.controller = holder;
            permanent.entry_turn = self.state.turn.turn_number;
            permanent.was_cast = cast.is_some();
            if let Some(cast) = cast {
                permanent.cast_x = cast.x;
                permanent.cast_spent = cast.spent;
            }
            permanent.tapped = entry.tapped;
            permanent.counters = entry.counters;
            self.state.permanents.push(Some(permanent));
            if self.state[object_id].types.is_land() {
                self.state.turn.history[holder.0].land_entered = true;
            }
            self.state.object_to_permanent[object_id.slot()] = Some(permanent_id);
            if entry.prepared {
                self.prepare_inset(object_id);
            }
            permanent_id
        });

        if old_zone == Some(ZoneType::Battlefield) || to_zone == ZoneType::Battlefield {
            // The permanent's controller gains or loses its mana abilities,
            // whoever owns the card.
            self.invalidate_mana_cache(event_controller);
        }
        self.capture_object_move(object_id, old_zone, Some(to_zone), event_controller);
        let event = GameEvent::CardMoved {
            object_id,
            departed: old_zone.map(|_| prior_object),
            from: old_zone,
            to: to_zone,
            controller: event_controller,
            face_down: entry.face_down,
        };
        self.emit(event);
        if let Some(permanent) = entered_permanent {
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
        if old_zone == Some(ZoneType::Battlefield) {
            self.handle_leaves_battlefield(prior_object, to_zone, followups);
        }
        // An attachment that grants or removes abilities changes the mana
        // its creature's controller, possibly the opponent, can produce.
        let card_def = &self.state[object_id];
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
        self.assert_library_knowledge();
        object_id
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
        let Some(object) = self.state.permanents[permanent].as_ref().map(|p| p.object) else {
            return;
        };
        if self.state[object].is_transformed() {
            self.state[object].turn_front_face_up();
        } else if !self.show_back_face(object) {
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
    fn show_back_face(&mut self, object: ObjectId) -> bool {
        let definition = self.state[object].definition_id;
        let Some(back) = default_content_pack().back_face(definition) else {
            return false;
        };
        self.state[object].turn_back_face_up(back);
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
        let (object_id, previous) = (permanent.object, permanent.controller);
        let permanent = self.state.permanents[permanent_id].as_mut().unwrap();
        permanent.controller = controller;
        permanent.summoning_sick = true;
        permanent.attacking = false;
        self.state
            .place_object(object_id, controller, ZoneType::Battlefield);
        self.invalidate_mana_cache(previous);
        self.invalidate_mana_cache(controller);
        if let Some(object) = self.permanent_object(permanent_id) {
            self.end_effects_for_control_change(object, controller);
        }
    }

    /// Leave-the-battlefield bookkeeping: fire delayed triggers watching the
    /// object (earthbend returns) and end "exiled until this leaves" durations
    /// (Jailer links — the exiled card returns immediately, no stack).
    fn handle_leaves_battlefield(
        &mut self,
        departed_object: ObjectId,
        to_zone: ZoneType,
        followups: &mut Vec<ProposedEvent>,
    ) {
        // Delayed triggers watching this object fire as it dies (or is
        // exiled, for earthbend) and are dropped otherwise: they watched this
        // battlefield stint only.
        let watches = |trigger: &DelayedTrigger| matches!(trigger.event, DelayedTriggerEvent::Leaves { watched, .. } if watched == departed_object);
        if self.state.delayed_triggers.iter().any(watches) {
            let (watching, kept): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.state.delayed_triggers)
                    .into_iter()
                    .partition(watches);
            *self.state.delayed_triggers_mut() = kept;
            let departed = self
                .object_lki(departed_object)
                .map(|lki| Target::Permanent(lki.permanent));
            for trigger in watching {
                let DelayedTriggerEvent::Leaves { or_exiled, .. } = trigger.event else {
                    continue;
                };
                if to_zone == ZoneType::Graveyard || (or_exiled && to_zone == ZoneType::Exile) {
                    self.fire_delayed_trigger(&trigger, departed);
                }
            }
        }

        // CR 400.7: continuous effects on the departing object end.
        self.end_effects_on(departed_object);

        // "Until this creature leaves the battlefield" durations end now:
        // linked exiled cards return under their owners' control.
        let (ended, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.state.exile_links)
            .into_iter()
            .partition(|link| link.source == departed_object);
        self.state.exile_links = kept;
        for link in ended.into_iter().filter(|link| link.returns) {
            let Ok(exiled_object) = self.state.object(link.exiled_object) else {
                continue;
            };
            let owner = exiled_object.owner;
            let exiled = link.exiled_object;
            if self.state.zones.contains(exiled, ZoneType::Exile, owner) {
                followups.push(self.zone_move_proposal(
                    exiled,
                    ZoneType::Battlefield,
                    ZoneEntry::default(),
                ));
            }
        }
    }

    pub(crate) fn assert_stack_consistent(&self) {
        #[cfg(debug_assertions)]
        {
            use std::collections::HashSet;

            let mut zone_spell_objects: Vec<ObjectId> = [PlayerId(0), PlayerId(1)]
                .into_iter()
                .flat_map(|player| {
                    self.state
                        .zones
                        .zone_objects(ZoneType::Stack, player)
                        .iter()
                        .copied()
                })
                .collect();

            let mut stack_spell_objects: Vec<ObjectId> = self
                .state
                .stack_objects
                .iter()
                .filter_map(|stack_object| match stack_object {
                    StackObject::Spell(spell) => Some(spell.object),
                    _ => None,
                })
                .collect();

            let mut seen = HashSet::new();
            for object in &stack_spell_objects {
                assert!(
                    seen.insert(*object),
                    "duplicate spell object on stack: {object:?} in {stack_spell_objects:?}"
                );
            }

            // Same objects in the stack zone and on the stack, as multisets.
            zone_spell_objects.sort_unstable();
            stack_spell_objects.sort_unstable();
            assert_eq!(
                zone_spell_objects,
                stack_spell_objects,
                "stack/zone spell mismatch: zone={zone_spell_objects:?}, stack={stack_spell_objects:?}"
            );
        }
    }

    /// A spell as it is on the stack now, or as it last existed there
    /// (CR 608.2h): countered, resolved, copied or stolen spells alike.
    pub(crate) fn spell_info(&self, object: ObjectId) -> Option<SpellLki> {
        self.stack_spell(object)
            .map(|spell| self.spell_lki(spell))
            .or_else(|| self.state.spell_lki.get(&object).cloned())
    }

    fn spell_lki(&self, spell: &SpellOnStack) -> SpellLki {
        let object = &self.state[spell.object];
        SpellLki {
            copiable: None,
            object: spell.object,
            definition_id: object.definition_id,
            controller: spell.controller,
            profile: spell.profile,
            mana_value: object.mana_value_with_x(spell.x),
            spent: spell.spent.total(),
            colors: object.colors(),
        }
    }

    /// Stack positions use the same exact spell or ability target as commands.
    pub(crate) fn stack_index_of(&self, target: Target) -> Option<usize> {
        self.state
            .stack_objects
            .iter()
            .position(|object| object.as_target() == target)
    }

    /// The card whose ability `id` is on the stack.
    pub(crate) fn stack_ability_source(&self, id: PublicId) -> Option<ObjectId> {
        let index = self.stack_index_of(Target::StackAbility(id))?;
        Some(self.state.stack_objects[index].source_object())
    }

    /// The spell `object` is on the stack.
    pub(crate) fn stack_spell(&self, object: ObjectId) -> Option<&SpellOnStack> {
        self.state
            .stack_objects
            .iter()
            .find_map(|stacked| match stacked {
                StackObject::Spell(spell) if spell.object == object => Some(spell),
                _ => None,
            })
    }
}
