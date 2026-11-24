// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        decision::Decision,
        event::GameEvent,
        game::{CastChoices, Game},
        proposed_event::{EventBinding, EventResult, PendingEvent, ProposedEvent, ZoneEntry},
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        card::{CardDefId, PrintedCharacteristics, Restriction, StaticAbility},
        effect::CopyException,
        game_object::{
            CardId, Object, ObjectId, ObjectLki, PermanentId, PlayerId, PublicId, Target,
        },
        permanent::CounterKind,
        permanent::Permanent,
        predicate::CardPredicate,
        stack_object::{SpellOnStack, StackObject},
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
    /// never the identity of the physical copy.
    pub(crate) fn reveal_object_to_hand(&mut self, object: ObjectId, player: PlayerId) {
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
        let moved = self.move_object(object, ZoneType::Hand);
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

    /// The controller of `card`'s spell on the stack (CR 110.2), or its
    /// owner once it has left the stack.
    pub(crate) fn spell_caster(&self, object: ObjectId) -> PlayerId {
        self.stack_spell_controller(object).unwrap_or_else(|| {
            self.state
                .objects
                .provenance(object)
                .expect("source provenance")
                .1
        })
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
            self.state.untap(permanent_id);
        }
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
            }
        }
    }

    pub(crate) fn mark_permanents_not_summoning_sick(&mut self, player: PlayerId) {
        for permanent_id in self.battlefield_permanents(player) {
            self.state.clear_summoning_sickness(permanent_id);
        }
    }

    pub(super) fn put_library_top_in_hand(&mut self, object: ObjectId) {
        self.move_object_with_entry(
            object,
            ZoneType::Hand,
            ZoneEntry {
                library_top_departure: true,
                ..Default::default()
            },
        );
    }

    pub fn draw_cards(&mut self, player: PlayerId, amount: usize) {
        for _ in 0..amount {
            let Some(object) = self.state.zones.top(ZoneType::Library, player) else {
                self.state.players[player.0].drew_when_empty = true;
                break;
            };
            if self.card_capture.is_some() || self.statistics.is_some() {
                self.capture_drawing = Some(object);
            }
            self.put_library_top_in_hand(object);
            self.capture_drawing = None;
            self.state.turn.history[player.0].cards_drawn += 1;
            self.emit(GameEvent::CardDrawn {
                player,
                nth_this_turn: self.state.turn.history[player.0].cards_drawn,
            });
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
        let batch = self.event_batch(vec![ProposedEvent::CreateTokens {
            definition: Box::new(crate::flow::proposed_event::TokenSource::Registered(
                key.into(),
            )),
            controller,
            count: 1,
            entry: ZoneEntry {
                tapped: tapped_and_attacking,
                ..Default::default()
            },
        }]);
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

    pub(crate) fn emit(&mut self, event: GameEvent) {
        crate::agent::policy_history::record(self, &event);
        // Every life total change is a LifeChanged event, so this is the one
        // place loss this turn is counted (CR 119.3).
        if let GameEvent::LifeChanged { player, old, new } = &event {
            if new < old {
                self.state.turn.history[player.0].life_lost += old.abs_diff(*new);
            }
        }
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
        let cast_from = self.state.zones.zone_of(object).expect("cast source zone");
        let flashback = cast_from == ZoneType::Graveyard;
        // CR 400.7g, h: the spell is the object the card became on the stack.
        let object = self.move_object(object, ZoneType::Stack);
        let id = self.state.id_gen_mut().next_id();
        let source_definition_id = self.state[object].definition_id;
        self.state
            .stack_objects_mut()
            .push(StackObject::Spell(SpellOnStack {
                id,
                object,
                controller,
                cast_from,
                flashback,
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
                self.reposition_library_object(owner, first, usize::MAX);
                self.remember_library_objects(owner, &[first, second]);
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
        self.reposition_library_object(owner, first, index);
        if let Some(second) = second {
            self.reposition_library_object(owner, second, index);
            self.remember_library_objects(owner, &[first, second]);
        }
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
                manifested: true,
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
        let batch = self.event_batch(vec![event]);
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
        let face = if entry.manifested {
            &*MANIFESTED
        } else if let Some((_, result)) = entry.meld {
            default_content_pack().definition(result)?.printed.as_ref()
        } else if entry.transformed {
            let back = default_content_pack().transforming_faces(object)?.back.0;
            default_content_pack().definition(back)?.printed.as_ref()
        } else {
            object
        };
        (!face.types.is_spell()).then_some(face)
    }

    // Keep the evaluated proposal fields together with its pre-move LKI and
    // consequent-event sink at this single commit boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_zone_move(
        &mut self,
        object_id: ObjectId,
        old_zone: Option<ZoneType>,
        to_zone: ZoneType,
        entry: ZoneEntry,
        departing_lki: Option<ObjectLki>,
        followups: &mut Vec<ProposedEvent>,
    ) -> EventResult {
        let departing_object = object_id;
        let second = self.state.zones.separate_melded_card(object_id);
        let owner = self.state.entry_object(object_id).owner;

        let definition = self.state.entry_object(object_id).definition_id;
        let public_source = old_zone.is_some_and(|zone| {
            !matches!(zone, ZoneType::Hand | ZoneType::Library)
                && !(zone == ZoneType::Exile && self.is_foretold(object_id))
        });
        let known_top = if old_zone == Some(ZoneType::Library) {
            self.state.library_order_pins_mut(owner).remove(&object_id);
            if entry.library_top_departure {
                self.state.known_library_mut(owner).remove_top()
            } else {
                self.state.known_library_mut(owner).forget_order();
                None
            }
        } else {
            None
        };
        if old_zone == Some(ZoneType::Hand) && to_zone != ZoneType::Hand {
            let seen = (to_zone != ZoneType::Library && !entry.face_down && !entry.manifested)
                .then_some(definition);
            self.forget_hand_card(owner, seen);
        }

        let mut event_controller = owner;

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.object_to_permanent[(object_id).slot()].take() {
                if let Some(permanent) = self.state.permanents[permanent_id].as_ref() {
                    event_controller = permanent.controller;
                }
                if to_zone == ZoneType::Graveyard && self.state.combat.is_some() {
                    if let Some(exposure) = self.exposure_mut() {
                        exposure.combat.deaths.insert(permanent_id);
                    }
                }
                self.state.permanents[permanent_id] = None;
            }
        }

        if let Some(mut lki) = departing_lki {
            let object = self.state.entry_object(object_id);
            lki.meld_partner = second;
            lki.copiable = default_content_pack().retain_copiable_values(object);
            self.record_object_lki(lki);
        }
        if old_zone == Some(ZoneType::Stack) {
            if let Some(index) = self.stack_index_of(Target::StackSpell(object_id)) {
                self.state.stack_objects_mut().remove(index);
            }
        }
        let holder = match (to_zone, entry.controller) {
            (ZoneType::Battlefield, Some(controller)) => controller,
            _ => owner,
        };
        // From here `object_id` is the appearance this move makes.
        let object_id = self
            .state
            .mint_appearance(object_id, holder, to_zone, entry.entry_copy);
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

        if to_zone == ZoneType::Battlefield {
            if let Some((second, result)) = entry.meld {
                let second_slot = self.state.zones.join_melded_card(object_id, second);
                self.capture_object_move(second_slot, old_zone, Some(to_zone), holder);
                let row = default_content_pack()
                    .definition(result)
                    .expect("meld result");
                self.state[object_id].show_melded(result, row.printed.clone());
            }
            if entry.manifested {
                self.state[object_id].turn_face_down(MANIFESTED.clone());
            }
            if entry.transformed {
                self.turn_over(object_id, FormTransition::Transform);
            }
            let permanent_id = PermanentId(self.state.permanents.len());
            let mut permanent = Permanent::new(
                self.state.id_gen_mut().next_id(),
                object_id,
                &self.state[object_id],
            );
            permanent.manifested = entry.manifested;
            permanent.controller = holder;
            permanent.entry_turn = self.state.turn.turn_number;
            permanent.spell_entry = entry.spell;
            permanent.tapped = entry.tapped;
            permanent.counters = entry.counters;
            permanent.attached_to = entry.attached_to;
            event_controller = permanent.controller;
            self.state.permanents.push(Some(permanent));
            if self.state[object_id].types.is_land() {
                self.state.turn.history[holder.0].land_entered = true;
            }
            self.state.object_to_permanent[(object_id).slot()] = Some(permanent_id);
        }

        self.capture_object_move(object_id, old_zone, Some(to_zone), event_controller);
        let event = GameEvent::CardMoved {
            object_id,
            departed: old_zone.map(|_| departing_object),
            from: old_zone,
            to: to_zone,
            controller: event_controller,
            face_down: entry.face_down || entry.manifested,
        };
        self.emit(event);
        if to_zone == ZoneType::Battlefield {
            if let Some(permanent) = self.state.permanent_of(object_id) {
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
            self.handle_leaves_battlefield(departing_object, to_zone, followups);
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
            excess_damage: None,
        }
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
                if faces.shown().1.types.is_spell() {
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

    /// Stack targets identify a spell by its exact object and an ability by its stack id.
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

    /// The controller of `card` while it is a spell on the stack.
    pub(crate) fn stack_spell_controller(&self, object_id: ObjectId) -> Option<PlayerId> {
        self.spell_on_stack(object_id).map(|spell| spell.controller)
    }
}
