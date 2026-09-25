// search.rs
// Determinized-search primitives on Game: hidden-information resampling and
// uniformly-random playouts to terminal. Used by flat Monte Carlo search.

use std::collections::BTreeSet;

use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::action::{ActionSpace, ActionSpaceKind, AgentError},
    flow::game::Game,
    state::{
        game_object::{CardId, PlayerId},
        zone::ZoneType,
    },
};

/// SplitMix64-style mix of two u64s into a stream-independent sub-seed.
pub fn mix_seed(a: u64, b: u64) -> u64 {
    let mut z = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Game {
    /// Replace the game RNG with a fresh seeded stream (for reproducible
    /// playouts from cloned states).
    pub fn reseed(&mut self, seed: u64) {
        self.state.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Sample one world consistent with `perspective`'s observation.
    ///
    /// Resample the opponent's residual unknown hand and both library orders.
    /// Decklists are public; outside sideboard copies never enter this pool.
    /// Accordingly:
    /// - public known-hand definition counts are reserved; unknown slots are
    ///   sampled uniformly from the residual hand ∪ library pool, with the
    ///   remainder becoming their shuffled library;
    /// - `perspective`'s own library is reshuffled (its order is unknown to
    ///   them, but its contents are determined by the public zones + hand).
    ///
    /// Public facts and own choices are preserved; opponent Learn choices refresh.
    pub fn determinize(&mut self, perspective: PlayerId, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let opponent = PlayerId((perspective.0 + 1) % 2);
        self.journal_zones();
        let (mut known, mut residual) = self.hidden_hand_partition(opponent);
        let unknown_slots = self.state.zones.size(ZoneType::Hand, opponent) - known.len();
        residual.shuffle(&mut rng);
        let library = residual.split_off(unknown_slots);
        known.extend(residual);
        self.state.zones.reassign_hidden(opponent, known, library);
        self.state
            .zones
            .shuffle_canonical(ZoneType::Library, perspective, &mut rng);

        self.repin_revealed_library_cards();
        if self.current_action_space.as_ref().is_some_and(|space| {
            space.player == Some(opponent) && space.kind == ActionSpaceKind::Learn
        }) {
            if let Some(space) = self.suspended_decision_action_space() {
                self.publish_action_space(space);
            }
        }
    }

    /// Canonical physical witnesses for public definition counts, plus the
    /// residual unseen pool. Never pins the physical copy that was revealed.
    pub(crate) fn hidden_hand_partition(&self, player: PlayerId) -> (Vec<CardId>, Vec<CardId>) {
        let mut pool: Vec<_> = [ZoneType::Hand, ZoneType::Library]
            .into_iter()
            .flat_map(|zone| self.state.zones.zone_cards(zone, player).iter().copied())
            .collect();
        pool.sort_unstable_by_key(|card| card.0);
        let mut counts = self.state.players[player.0].known_hand.clone();
        let mut known = Vec::new();
        let mut residual = Vec::new();
        for card in pool {
            if let Some(count) = counts.get_mut(&self.state.cards[card].definition_id) {
                if *count > 0 {
                    known.push(card);
                    *count -= 1;
                    continue;
                }
            }
            residual.push(card);
        }
        assert!(
            counts.values().all(|count| *count == 0),
            "known hand must be available in unseen pool"
        );
        assert!(
            known.len() <= self.state.zones.size(ZoneType::Hand, player),
            "known hand exceeds hand size"
        );
        (known, residual)
    }

    pub(crate) fn refresh_priority_actions(&mut self, player: PlayerId) {
        self.invalidate_mana_cache(PlayerId(0));
        self.invalidate_mana_cache(PlayerId(1));
        let actions = self.compute_player_actions(player);
        self.publish_action_space(ActionSpace {
            player: Some(player),
            kind: ActionSpaceKind::Priority,
            actions,
            focus: Vec::new(),
        });
    }

    pub(crate) fn repin_revealed_library_cards(&mut self) {
        let Some(suspended) = &self.state.suspended_decision else {
            return;
        };
        let player = suspended.decision.player();
        let revealed = suspended.decision.revealed_cards().to_vec();
        if revealed.is_empty() {
            return;
        }
        for card in &revealed {
            self.state.zones.move_card(*card, player, ZoneType::Library);
        }
        let revealed_set = revealed.iter().copied().collect::<BTreeSet<_>>();
        let library = self.state.zones.zone_cards_mut(ZoneType::Library, player);
        library.retain(|card| !revealed_set.contains(card));
        // revealed[0] is the top of the library = last element.
        for card in revealed.iter().rev() {
            library.push(*card);
        }
    }

    /// Play both sides uniformly-random-legal to terminal.
    ///
    /// Returns `Ok(Some(winner_index))` on a decided game, `Ok(None)` if the
    /// game is a draw or `max_steps` was reached without termination.
    /// `hit_cap` (when provided) is set to true only in the cap case.
    pub fn random_playout(
        &mut self,
        max_steps: usize,
        hit_cap: Option<&mut bool>,
    ) -> Result<Option<usize>, AgentError> {
        let mut steps = 0usize;
        while !self.is_game_over() {
            if steps >= max_steps {
                if let Some(flag) = hit_cap {
                    *flag = true;
                }
                return Ok(None);
            }
            let action_count = self
                .current_action_space
                .as_ref()
                .map_or(0, |space| space.actions.len());
            if action_count == 0 {
                return Err(AgentError("random_playout: empty action space".to_string()));
            }
            let index = self.state.rng.gen_range(0..action_count);
            self.step(index)?;
            steps += 1;
        }
        Ok(self.winner_index())
    }
}
