// sba.rs
// State-based actions.

use crate::{
    flow::game::Game,
    state::{
        game_object::{PermanentId, PlayerId},
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
            let permanent = self.state.permanents[permanent_id].as_ref().unwrap();
            let card = &self.state.cards[permanent.card];
            // CR 704.5g — Creatures with lethal damage are destroyed.
            if permanent.has_lethal_damage(card) {
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
    }

    pub(crate) fn lose_game(&mut self, player: PlayerId) {
        self.state.players[player.0].alive = false;
    }
}
