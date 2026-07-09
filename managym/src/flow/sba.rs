// sba.rs
// State-based actions.

use crate::{
    flow::game::Game,
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn perform_state_based_actions(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            // CR 704.5a, 704.5b — A player loses at 0 or less life or for drawing from empty library.
            if self.state.players[player.0].life <= 0
                || self.state.players[player.0].drew_when_empty
            {
                self.lose_game(player);
            }
        }

        if self.is_game_over() {
            return;
        }

        let mut to_destroy = Vec::new();
        for permanent_id in self
            .state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(idx, perm)| perm.as_ref().map(|_| PermanentId(idx)))
        {
            // CR 704.5f — A creature with toughness 0 or less is put into
            // its owner's graveyard (an earthbent land losing its counters
            // is a 0/0 and dies).
            let zero_toughness = self.permanent_is_creature(permanent_id)
                && self.effective_toughness(permanent_id) <= 0;
            // CR 704.5g — Creatures with lethal damage are destroyed.
            if zero_toughness || self.has_lethal_damage(permanent_id) {
                to_destroy.push(permanent_id);
            }
        }

        for permanent_id in to_destroy {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = permanent.card;
            let controller = permanent.controller;
            self.move_card(card, ZoneType::Graveyard);
            self.invalidate_mana_cache(controller);
        }

        // CR 704.5d — A token in a zone other than the battlefield ceases to
        // exist. This runs after the move to the graveyard, so death triggers
        // from tokens have already been enqueued.
        let mut tokens_to_remove = Vec::new();
        for (index, card) in self.state.cards.iter().enumerate() {
            if !card.is_token {
                continue;
            }
            let card_id = CardId(index);
            match self.state.zones.zone_of(card_id) {
                Some(ZoneType::Battlefield) | None => {}
                Some(_) => tokens_to_remove.push((card_id, card.owner)),
            }
        }
        for (card_id, owner) in tokens_to_remove {
            self.state.zones.remove_card(card_id, owner);
        }
    }

    pub(crate) fn lose_game(&mut self, player: PlayerId) {
        self.state.players[player.0].alive = false;
    }
}
