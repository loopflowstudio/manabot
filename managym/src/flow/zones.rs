// zones.rs
// Zone management, card movement, and utility methods on Game.

use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        permanent::Permanent,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn clear_mana_pools(&mut self) {
        for player in &mut self.state.players {
            player.mana_pool.clear();
        }
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
