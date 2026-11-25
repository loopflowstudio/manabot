// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        decision::Decision,
        event::GameEvent,
        game::{CastChoices, Game},
        proposed_event::{
            EntryFace, EventBinding, EventResult, PendingEvent, ProposedEvent, ZoneEntry,
        },
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        card::{CardDefId, PrintedCharacteristics, Restriction, StaticAbility},
        effect::CopyException,
        game_object::{
            CardId, Object, ObjectId, ObjectLki, PermanentId, PlayerId, PublicId, Target,
        },
        permanent::CounterKind,
        permanent::{Permanent, SpellEntry},
        predicate::CardPredicate,
        stack_object::{SpellLki, SpellOnStack, StackObject},
        zone::ZoneType,
    },
};

/// The prospective entry and the permanent share these copiable values.
static MANIFESTED: std::sync::LazyLock<std::sync::Arc<PrintedCharacteristics>> =
    std::sync::LazyLock::new(|| {
        std::sync::Arc::new(PrintedCharacteristics {
            types: crate::state::card::CardTypes::new([crate::state::card::CardType::Creature]),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        })
    });

/// A change of the face a battlefield object shows that keeps the object
/// (CR 701.27a, 708.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FormTransition {
    /// Turn a transforming object over (CR 701.27a).
    Transform,
    /// Turn a face-down object face up (CR 708.8).
    TurnFaceUp,
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

    /// A melded permanent sums the two front faces (CR 202.3c). A token
    /// copying the result has no constituent cards and keeps mana value zero.
    pub(super) fn meld_mana_value(&self, first: CardId, second: CardId) -> u8 {
        [first, second]
            .into_iter()
            .filter_map(|card| default_content_pack().definition(self.state.cards[card].definition))
            .filter_map(|definition| definition.mana_cost.as_ref())
            .map(|cost| cost.mana_value)
            .sum()
    }

    pub(super) fn meld_partners<'a>(
        &'a self,
        player: PlayerId,
        source: ObjectId,
        name: &'a str,
    ) -> impl Iterator<Item = ObjectId> + 'a {
        let owns_source = self
            .lookup_current_permanent(source)
            .ok()
            .is_some_and(|id| {
                self.state[source].owner == player
                    && self.state.permanents[id].as_ref().unwrap().controller == player
            });
        self.iter_battlefield_permanents(player)
            .filter(move |id| {
                owns_source
                    && self.permanent_is_battlefield_creature(*id)
                    && self.characteristics(*id).name == name
            })
            .filter_map(|id| self.permanent_object(id))
            .filter(move |object| *object != source && self.state[*object].owner == player)
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

    pub(super) fn put_library_top_in_hand(&mut self, object: ObjectId) -> ObjectId {
        self.move_object_with_entry(
            object,
            ZoneType::Hand,
            ZoneEntry {
                library_top_departure: true,
                ..Default::default()
            },
        )
    }

    pub fn draw_cards(&mut self, player: PlayerId, amount: usize) {
        self.draw(player, amount, false);
    }

    pub(super) fn draw(&mut self, player: PlayerId, amount: usize, draw_step: bool) {
        for _ in 0..amount {
            let Some(object) = self.state.zones.top(ZoneType::Library, player) else {
                self.state.players[player.0].drew_when_empty = true;
                break;
            };
            if self.card_capture.is_some() || self.statistics.is_some() {
                self.capture_drawing = Some(object);
            }
            let drawn = self.put_library_top_in_hand(object);
            self.capture_drawing = None;
            self.state.turn.history[player.0].cards_drawn += 1;
            self.emit(GameEvent::CardDrawn {
                player,
                nth_this_turn: self.state.turn.history[player.0].cards_drawn,
            });
            self.presentation_drawn(player, drawn, draw_step);
        }
    }

    /// Tap a permanent, emitting a `PermanentTapped` event on the
    /// untapped-to-tapped transition (CR 603.10-style "becomes tapped").
    /// Returns whether the permanent transitioned. Tapping for mana triggers
    /// nothing here: triggered mana abilities wait for the mana ability to
    /// produce mana (CR 106.12a; `flow/mana.rs`).
    pub(crate) fn tap_permanent(&mut self, permanent_id: PermanentId, for_mana: bool) -> bool {
        if !self.state.tap(permanent_id) {
            return false;
        }
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
        let untapped = self.state.untap(permanent_id);
        if untapped {
            self.emit(GameEvent::PermanentUntapped {
                permanent: permanent_id,
            });
        }
        untapped
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
        for object in results.iter().flat_map(|result| result.arrivals()) {
            self.finish_created_token(object, tapped_and_attacking);
        }
        results
            .iter()
            .flat_map(|result| result.arrivals())
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
        values: &crate::state::game_object::CopiableValues,
        controller: PlayerId,
        exceptions: &[CopyException],
    ) -> ObjectId {
        let id = self.state.id_gen.next_id();
        let copy = Object::token_with_copiable_values(
            id,
            controller,
            (
                values.definition_id.unwrap_or(CardDefId::ANONYMOUS),
                std::sync::Arc::clone(&values.definition.printed),
            ),
            values.faces.clone(),
            values.copy.clone(),
            exceptions,
        );
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
        source: ObjectId,
        id: PublicId,
        controller: PlayerId,
    ) -> ObjectId {
        let copy = Object::spell_copy(self.state[source].clone(), id, controller);
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
        self.presentation_event(&event);
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
        self.presentation_event(&event);
        self.state.observation_events_mut().push(event);
    }

    pub(crate) fn was_cast_from_graveyard(&self, spell: &SpellOnStack) -> bool {
        spell.cast_from == ZoneType::Graveyard && self.state[spell.object].is_card()
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

    /// Ask after replacements have fixed the destination and before any
    /// member of the simultaneous batch has moved (CR 713.8).
    pub(super) fn meld_order_choice(
        &self,
        pending: &PendingEvent,
        effect_controller: Option<PlayerId>,
    ) -> Option<Decision> {
        let ProposedEvent::ZoneMove { object_id, to, .. } = pending.event else {
            return None;
        };
        if pending.second_card_first.is_some() {
            return None;
        }
        let second = self.state.zones.melded_card(object_id)?;
        let first = self.state.card_of(object_id).expect("melded primary card");
        Some(Decision::OrderMeldedCards {
            player: match to {
                ZoneType::Exile => effect_controller?,
                ZoneType::Library | ZoneType::Graveyard => self.state.cards[first].owner,
                _ => return None,
            },
            cards: [first, second],
        })
    }

    pub(super) fn order_meld_arrivals(
        &mut self,
        result: &mut EventResult,
        second_first: bool,
        binding: &EventBinding,
    ) {
        if second_first {
            result.moved.swap(0, 1);
        }
        let [Some(first), Some(second)] = result.moved else {
            return;
        };
        let owner = self.state[first].owner;
        match self.state.zones.zone_of(first) {
            // The instruction places both arrivals together at its stated depth.
            Some(ZoneType::Library)
                if matches!(
                    binding,
                    EventBinding::PlaceInLibrary { .. }
                        | EventBinding::Moved {
                            library_end: Some(crate::state::ability::LibraryEnd::Bottom),
                            ..
                        }
                ) => {}
            Some(ZoneType::Library) => {
                self.place_library_arrival(owner, first, usize::MAX);
                self.arrange_library_objects(owner, &[first, second]);
            }
            // Exile appends appearances in timestamp order (earliest first).
            Some(ZoneType::Exile) if second_first => {
                self.state
                    .zones
                    .reposition(ZoneType::Exile, owner, second, usize::MAX)
            }
            Some(ZoneType::Graveyard) => {
                self.state
                    .zones
                    .reposition(ZoneType::Graveyard, owner, first, usize::MAX)
            }
            _ => {}
        }
    }

    /// Place the surviving arrivals of one event in draw order. The owner
    /// knows the relative order they chose for two constituent cards.
    pub(super) fn place_library_arrivals(&mut self, result: &EventResult, second_from_top: bool) {
        let mut arrivals = result
            .arrivals()
            .filter(|object| self.state.zones.zone_of(*object) == Some(ZoneType::Library));
        let Some(first) = arrivals.next() else {
            return;
        };
        let second = arrivals.next();
        let owner = self.state[first].owner;
        // The arrivals entered above the previous top. Insert each
        // at the same index so the first chosen is drawn first.
        let index = if second_from_top {
            self.state
                .zones
                .size(ZoneType::Library, owner)
                .saturating_sub(2 + usize::from(second.is_some()))
        } else {
            0
        };
        self.place_library_arrival(owner, first, index);
        if let Some(second) = second {
            self.place_library_arrival(owner, second, index);
            self.arrange_library_objects(owner, &[first, second]);
        }
    }

    /// Reposition an arrival both players could follow from its zone move.
    fn place_library_arrival(&mut self, owner: PlayerId, object: ObjectId, index: usize) {
        let from = self
            .library_from_top(owner)
            .position(|card| card == object)
            .expect("arrival in library");
        let last = self.state.zones.size(ZoneType::Library, owner) - 1;
        let to = last.saturating_sub(index);
        self.state.known_library_mut(owner).move_position(from, to);
        self.state
            .zones
            .reposition(ZoneType::Library, owner, object, index);
        self.assert_library_knowledge();
    }

    /// Return the object this move made (CR 400.7j), even if a consequent
    /// event moves it again. A move that does not happen (the object is
    /// already there, or already gone) returns `object` itself.
    pub fn move_object(&mut self, object: ObjectId, to_zone: ZoneType) -> ObjectId {
        self.move_object_with_entry(object, to_zone, ZoneEntry::default())
    }

    /// Put the selected card onto the battlefield face down under the chooser's control.
    pub fn manifest(&mut self, object: ObjectId, controller: PlayerId) -> ObjectId {
        self.move_object_with_entry(
            object,
            ZoneType::Battlefield,
            ZoneEntry {
                face: EntryFace::Manifested,
                controller: Some(controller),
                ..Default::default()
            },
        )
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
            .and_then(|result| result.arrivals().next())
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
                .spell_on_stack(object_id)
                .is_some_and(|spell| spell.flashback)
        {
            ZoneType::Exile
        } else {
            to_zone
        };
        entry = entry.for_zone(to_zone);
        if to_zone == ZoneType::Battlefield
            && self
                .battlefield_entry_face(object_id, &entry)
                .is_some_and(PrintedCharacteristics::is_saga)
        {
            entry.counters.change(CounterKind::Lore, 1);
        }
        ProposedEvent::ZoneMove {
            object_id,
            from,
            to: to_zone,
            entry,
        }
    }

    /// The face that would enter (CR 712.14). A missing face or an instant
    /// or sorcery face cannot enter the battlefield (CR 304.4, 307.4).
    pub(super) fn battlefield_entry_face(
        &self,
        object: ObjectId,
        entry: &ZoneEntry,
    ) -> Option<&PrintedCharacteristics> {
        let object = self.state.entry_object(object);
        let face: &PrintedCharacteristics = match entry.face {
            EntryFace::Front => object,
            EntryFace::ModalBack => {
                &default_content_pack()
                    .back_face_row(object.retained_base().0)?
                    .1
                    .printed
            }
            EntryFace::Transformed => {
                let pack = default_content_pack();
                let back = pack.transforming_faces(object)?.back.0;
                &pack.definition(back)?.printed
            }
            EntryFace::Manifested => &MANIFESTED,
            EntryFace::Melded { result, .. } => {
                default_content_pack().definition(result)?.printed.as_ref()
            }
        };
        (!face.types.is_instant_or_sorcery()).then_some(face)
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
    ) -> EventResult {
        // A write through `zones` advances every revision; most moves have
        // no melded card to release, so read before writing.
        let second = self
            .state
            .zones
            .melded_card(prior_object)
            .and_then(|_| self.state.zones.separate_melded_card(prior_object));
        let owner = self.state.entry_object(prior_object).owner;

        let definition = self.state.entry_object(prior_object).definition_id;
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
            let seen =
                (to_zone != ZoneType::Library && !entry.is_face_down()).then_some(definition);
            self.forget_hand_card(owner, seen);
        }

        let holder = match (to_zone, entry.controller) {
            (ZoneType::Battlefield, Some(controller)) => controller,
            _ => owner,
        };
        let entry_face_down = entry.is_face_down();
        let event_controller = departing_lki.as_ref().map_or(holder, |lki| lki.controller);
        if let Some(mut lki) = departing_lki {
            lki.meld_partner = second;
            lki.copiable = default_content_pack()
                .retain_copiable_values(self.state.entry_object(prior_object));
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
                let lki = self.spell_lki(&spell);
                self.state.spell_lki_mut().insert(spell.object, lki);
                (to_zone == ZoneType::Battlefield).then_some(SpellEntry {
                    x: spell.x,
                    kicked: spell.kicked,
                    payment: self.state[spell.object].is_card().then_some(spell.spent),
                })
            }
            _ => None,
        };

        let object_id = self
            .state
            .mint_appearance(prior_object, holder, to_zone, entry.entry_copy);
        match entry.profile {
            crate::state::card::SpellProfile::BackFace => self.show_back_face(object_id),
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
            match entry.face {
                EntryFace::Front => {}
                EntryFace::ModalBack => {
                    // A resolving spell copy already holds its selected face.
                    if !self.state[object_id].is_transformed() {
                        self.show_back_face(object_id);
                    }
                }
                EntryFace::Transformed => {
                    self.turn_over(object_id, FormTransition::Transform);
                }
                EntryFace::Manifested => {
                    self.state[object_id].turn_face_down(MANIFESTED.clone());
                }
                EntryFace::Melded { partner, result } => {
                    let partner_slot = self.state.zones.join_melded_card(object_id, partner);
                    self.capture_object_move(partner_slot, old_zone, Some(to_zone), holder);
                    let row = default_content_pack()
                        .definition(result)
                        .expect("meld result");
                    self.state[object_id].show_melded(result, row.printed.clone());
                }
            }
            let permanent_id = PermanentId(self.state.permanents.len());
            let mut permanent = Permanent::new(
                self.state.id_gen_mut().next_id(),
                object_id,
                &self.state[object_id],
            );
            permanent.manifested = entry.face == EntryFace::Manifested;
            permanent.controller = holder;
            permanent.entry_turn = self.state.turn.turn_number;
            permanent.spell_entry = entry.spell.or_else(|| cast.map(std::sync::Arc::new));
            permanent.tapped = entry.tapped;
            permanent.counters = entry.counters;
            permanent.attached_to = entry.attached_to;
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

        self.capture_object_move(object_id, old_zone, Some(to_zone), event_controller);
        let event = GameEvent::CardMoved {
            object_id,
            departed: old_zone.map(|_| prior_object),
            from: old_zone,
            to: to_zone,
            controller: event_controller,
            face_down: entry_face_down,
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
        let second = second.map(|second| {
            let arrival = self.state.mint_appearance(second, owner, to_zone, false);
            if to_zone == ZoneType::Library {
                self.state.library_order_pins_mut(owner).insert(arrival);
                let definition = self.state[arrival].definition_id;
                self.state
                    .known_library_mut(owner)
                    .insert_top(None, Some(definition));
            }
            if to_zone == ZoneType::Graveyard {
                self.state.turn.graveyard_from_battlefield.push(arrival);
            }
            self.capture_object_move(arrival, old_zone, Some(to_zone), event_controller);
            arrival
        });
        self.assert_library_knowledge();
        EventResult {
            committed: true,
            moved: [Some(object_id), second],
            ..Default::default()
        }
    }

    /// `player` bent: record the kind in their turn history and emit the
    /// event "whenever you waterbend, earthbend, firebend, or airbend" reads.
    pub(crate) fn record_bending(&mut self, player: PlayerId, kind: crate::flow::turn::Bending) {
        self.state.turn.history[player.0].bent |= kind as u8;
        self.emit(GameEvent::Bent { player, kind });
    }

    /// Show the back face chosen for cast or play. Its eligibility was
    /// checked by the casting authority.
    fn show_back_face(&mut self, object: ObjectId) {
        let faces = default_content_pack()
            .transforming_faces(&self.state[object])
            .expect("offered back face");
        debug_assert!(!faces.back_up);
        self.state[object].show_transforming_face(faces.turned_over());
    }

    /// Transform a double-faced permanent (CR 701.27a): it turns over and
    /// stays the same object, keeping its counters, damage, attachments and
    /// combat status (CR 701.27a). Anything else can't transform and stays as
    /// it is (CR 701.27c).
    pub fn transform_permanent(&mut self, permanent: PermanentId) {
        let Some(object) = self.state.permanents[permanent].as_ref().map(|p| p.object) else {
            return;
        };
        if self.turn_over(object, FormTransition::Transform) {
            self.state.permanents[permanent]
                .as_mut()
                .unwrap()
                .transforms += 1;
        }
    }

    /// Change the face a battlefield object shows without making a new
    /// object; false when it cannot. Transformability comes from its physical
    /// backing (CR 712.9, 712.10), and an instant or sorcery face is never
    /// turned up on the battlefield. Copy exceptions apply once to the raw
    /// face shown (CR 707.9b).
    pub(crate) fn turn_over(&mut self, object: ObjectId, transition: FormTransition) -> bool {
        match transition {
            FormTransition::Transform => {
                let Some(faces) = default_content_pack()
                    .transforming_faces(&self.state[object])
                    .map(|faces| faces.turned_over())
                else {
                    return false;
                };
                if faces.shown().1.types.is_instant_or_sorcery() {
                    return false;
                }
                self.state[object].show_transforming_face(faces);
                true
            }
            FormTransition::TurnFaceUp => self.state[object].turn_face_up(),
        }
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
        let object_id = permanent.object;
        let permanent = self.state.permanents[permanent_id].as_mut().unwrap();
        permanent.controller = controller;
        permanent.summoning_sick = true;
        permanent.attacking = false;
        self.state
            .place_object(object_id, controller, ZoneType::Battlefield);
        if let Some(object) = self.permanent_object(permanent_id) {
            self.end_effects_for_control_change(object, controller);
        }
    }

    /// Leave-the-battlefield bookkeeping: fire delayed triggers watching the
    /// object (earthbend returns) and end "exiled until this leaves" durations
    /// (Jailer links — the exiled card returns immediately, no stack).
    fn handle_leaves_battlefield(
        &mut self,
        departing_object: ObjectId,
        to_zone: ZoneType,
        followups: &mut Vec<ProposedEvent>,
    ) {
        // Delayed triggers watching this object fire as it dies (or is
        // exiled, for earthbend) and are dropped otherwise: they watched this
        // battlefield stint only.
        let watches = |trigger: &DelayedTrigger| match trigger.event {
            DelayedTriggerEvent::Leaves { watched, .. } => watched == departing_object,
            _ => false,
        };
        if self.state.delayed_triggers.iter().any(watches) {
            let (watching, kept): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.state.delayed_triggers)
                    .into_iter()
                    .partition(watches);
            *self.state.delayed_triggers_mut() = kept;
            let departed = self
                .object_lki(departing_object)
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
        self.end_effects_on(departing_object);

        // "Until this creature leaves the battlefield" durations end now:
        // linked exiled cards return under their owners' control.
        let (ended, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.state.exile_links)
            .into_iter()
            .partition(|link| link.source == departing_object && link.returns);
        self.state.exile_links = kept;
        for link in ended {
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
        self.spell_on_stack(object)
            .map(|spell| self.spell_lki(spell))
            .or_else(|| self.state.spell_lki.get(&object).cloned())
    }

    fn spell_lki(&self, spell: &SpellOnStack) -> SpellLki {
        let object = &self.state[spell.object];
        SpellLki {
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

    pub(crate) fn spell_on_stack(&self, object: ObjectId) -> Option<&SpellOnStack> {
        self.state.stack_objects.iter().find_map(|item| match item {
            StackObject::Spell(spell) if spell.object == object => Some(spell),
            _ => None,
        })
    }
}
