// sba.rs
// State-based actions.

use crate::{
    flow::game::Game,
    state::{
        game_object::{CardId, ObjectRef, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

impl Game {
    /// Perform one simultaneous batch of the supported state-based actions.
    /// Returns whether the batch changed state; the stabilization authority
    /// repeats this method until it returns false (CR 704.3).
    pub(crate) fn perform_state_based_actions(&mut self) -> bool {
        let mut performed = false;
        for player in [PlayerId(0), PlayerId(1)] {
            // CR 704.5a, 704.5b — A player loses at 0 or less life or for drawing from empty library.
            if self.state.players[player.0].life <= 0
                || self.state.players[player.0].drew_when_empty
            {
                performed |= self.lose_game(player);
            }
        }

        if self.is_game_over() {
            return performed;
        }

        // Candidate discovery uses exact object identities. The mutable
        // storage slot is resolved only while inspecting the current object;
        // no later commit can accidentally follow a re-entered card.
        let mut candidates: Vec<(ObjectRef, PlayerId, bool)> = Vec::new();
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
            let destroy = !zero_toughness && self.has_lethal_damage(permanent_id);
            if !zero_toughness && !destroy {
                continue;
            }
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let Some(object_ref) = self.permanent_object_ref(permanent_id) else {
                continue;
            };
            candidates.push((object_ref, permanent.controller, destroy));
        }
        candidates.sort_by_key(|(object_ref, _, _)| *object_ref);

        // CR 704.3 — all applicable SBAs are one simultaneous event. Commits
        // have a deterministic ObjectRef order for replay, while every death
        // event sees the same pre-batch trigger-source snapshot.
        let trigger_sources = self.snapshot_trigger_sources();
        for (object_ref, controller, destroy) in candidates {
            let committed = if destroy {
                // CR 704.5g destruction has its own proposal before the
                // consequent battlefield-to-graveyard move.
                self.destroy_object(object_ref)
            } else {
                // CR 704.5f is a zone move, not destruction.
                self.move_object_to_zone(object_ref, ZoneType::Graveyard)
            };
            if committed {
                performed = true;
                self.invalidate_mana_cache(controller);
            }
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
            performed = true;
        }

        if performed {
            self.process_game_events_with_sources(&trigger_sources);
        }

        performed
    }

    pub(crate) fn lose_game(&mut self, player: PlayerId) -> bool {
        if !self.state.players[player.0].alive {
            return false;
        }
        self.state.players[player.0].alive = false;
        true
    }
}
