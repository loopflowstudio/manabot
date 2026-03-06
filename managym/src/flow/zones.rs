// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        permanent::Permanent,
        stack::StackObject,
        target::Target,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn push_spell_to_stack(&mut self, card: CardId) {
        self.move_card(card, ZoneType::Stack);
        self.state.stack.push(StackObject::Spell { card });
    }

    pub(crate) fn push_triggered_to_stack(
        &mut self,
        source_card: CardId,
        ability_index: usize,
        controller: PlayerId,
        target: Option<Target>,
    ) {
        self.state.stack.push(StackObject::TriggeredAbility {
            source_card,
            ability_index,
            controller,
            target,
        });
    }

    pub(crate) fn pop_stack(&mut self) -> Option<StackObject> {
        self.state.stack.pop()
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
                .stack
                .iter()
                .filter_map(|stack_object| match stack_object {
                    StackObject::Spell { card } => Some(*card),
                    StackObject::TriggeredAbility { .. } => None,
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

    pub(crate) fn draw_cards(&mut self, player: PlayerId, amount: usize) {
        for _ in 0..amount {
            if self.state.zones.size(ZoneType::Library, player) == 0 {
                self.state.players[player.0].drew_when_empty = true;
                break;
            }

            if let Some(card) = self.state.zones.top(ZoneType::Library, player) {
                self.move_card(card, ZoneType::Hand);
            }
        }
    }

    pub fn move_card(&mut self, card: CardId, to_zone: ZoneType) {
        let owner = self.state.cards[card].owner;
        let old_zone = self.state.zones.zone_of(card);
        let mut event_controller = owner;

        if old_zone == Some(ZoneType::Battlefield) {
            if let Some(permanent_id) = self.state.card_to_permanent[card].take() {
                if let Some(permanent) = self.state.permanents[permanent_id].as_ref() {
                    event_controller = permanent.controller;
                }
                self.state.permanents[permanent_id] = None;
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

        self.state.pending_events.push(GameEvent::CardMoved {
            card,
            from: old_zone,
            to: to_zone,
            controller: event_controller,
        });
    }
}
