// sba.rs
// State-based actions.

use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

impl Game {
    /// Duplicate legendary names, grouped by controller in APNAP order.
    fn legend_groups(&self) -> Vec<Vec<CardId>> {
        let mut groups = Vec::new();
        for player in [self.active_player(), self.non_active_player()] {
            let mut names = std::collections::BTreeMap::<&str, Vec<CardId>>::new();
            for id in self.battlefield_permanents(player) {
                let permanent = self.state.permanents[id].as_ref().expect("battlefield");
                let card = &self.state.cards[permanent.card];
                if card
                    .supertypes
                    .iter()
                    .any(|kind| kind.eq_ignore_ascii_case("legendary"))
                {
                    names.entry(&card.name).or_default().push(permanent.card);
                }
            }
            groups.extend(names.into_values().filter(|cards| cards.len() > 1));
        }
        groups
    }

    /// Perform one simultaneous batch, or suspend for required legend choices.
    /// Stabilization repeats changed batches before granting priority (CR 704.3).
    pub(crate) fn perform_state_based_actions(&mut self) -> bool {
        // Choose all legends before applying any of this simultaneous batch.
        // Terminal SBAs take precedence over asking a player who already lost.
        if self
            .state
            .players
            .iter()
            .all(|p| p.life > 0 && !p.drew_when_empty)
        {
            let groups = self.legend_groups();
            if let Some(first) = groups.first() {
                let id = self.state.card_to_permanent[first[0]].expect("legend");
                let player = self.state.permanents[id]
                    .as_ref()
                    .expect("legend")
                    .controller;
                self.suspend_rule_decision(crate::flow::decision::Decision::LegendRule {
                    player,
                    groups,
                    keep: Vec::new(),
                });
                return false;
            }
        }
        self.perform_state_based_actions_with_legends(&[])
    }

    pub(crate) fn perform_state_based_actions_with_legends(&mut self, keep: &[CardId]) -> bool {
        let legend_losers: Vec<CardId> = self
            .legend_groups()
            .into_iter()
            .flatten()
            .filter(|card| !keep.contains(card))
            .collect();
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
        let mut candidates = Vec::new();
        for (permanent_id, permanent) in self
            .state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(idx, perm)| perm.as_ref().map(|p| (PermanentId(idx), p)))
        {
            // CR 704.5f — A creature with toughness 0 or less is put into
            // its owner's graveyard (an earthbent land losing its counters
            // is a 0/0 and dies).
            let zero_toughness = self.permanent_is_creature(permanent_id)
                && self.effective_toughness(permanent_id) <= 0;
            // CR 704.5g — Creatures with lethal damage are destroyed.
            let destroy = !zero_toughness && self.has_lethal_damage(permanent_id);
            let legend_loser = legend_losers.contains(&permanent.card);
            if !zero_toughness && !destroy && !legend_loser {
                continue;
            }
            let Some(object_ref) = self.permanent_object_ref(permanent_id) else {
                continue;
            };
            let Some(event_ref) = self.object_event_ref(object_ref) else {
                continue;
            };
            candidates.push((
                object_ref,
                event_ref,
                permanent.controller,
                destroy && !legend_loser,
            ));
        }
        candidates.sort_by_key(|(object_ref, _, _, _)| *object_ref);

        // CR 704.3 — all applicable SBAs are one simultaneous event. Commits
        // have a deterministic ObjectRef order for replay, while every death
        // event sees the same pre-batch trigger-source snapshot.
        let trigger_sources = self.snapshot_trigger_sources();
        let mut died = Vec::new();
        for (object_ref, event_ref, controller, destroy) in candidates {
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
                if self.state.zones.zone_of(CardId::from(object_ref.entity))
                    == Some(ZoneType::Graveyard)
                {
                    died.push(event_ref);
                }
                self.invalidate_mana_cache(controller);
            }
        }
        if !died.is_empty() {
            self.emit_observation_event(GameEvent::PermanentsDied { objects: died });
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
            self.journal_zone_move(card_id, owner);
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
