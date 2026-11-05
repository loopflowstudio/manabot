// search.rs
// Determinized-search primitives on Game: hidden-information resampling and
// uniformly-random playouts to terminal. Used by flat Monte Carlo search.

use std::collections::BTreeSet;

use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::action::{ActionSpace, ActionSpaceKind, AgentError},
    flow::{decision::Decision, game::Game},
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
        *self.state.rng_mut() = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Sample one world consistent with `perspective`'s observation.
    ///
    /// Resample the opponent's residual unknown hand and both library orders.
    /// The opponent's decklist must be open to `perspective` (callers check
    /// [`Self::require_open_decklist`]); outside sideboard copies never enter
    /// this pool. Accordingly:
    /// - public known-hand definition counts are reserved; unknown slots are
    ///   sampled uniformly from the residual hand ∪ library pool, with the
    ///   remainder becoming their shuffled library;
    /// - `perspective`'s own library is reshuffled (its order is unknown to
    ///   them, but its contents are determined by the public zones + hand).
    ///
    /// - the opponent's face-down foretold slots are filled from the same
    ///   unseen pool's first foretell-bearing cards, as possible-world
    ///   materialization fills them.
    ///
    /// Public facts and own choices are preserved; opponent private choices refresh.
    pub fn determinize(&mut self, perspective: PlayerId, seed: u64) -> Result<(), AgentError> {
        // Rejection is needed only when a retained unknown choice constrains
        // the deal. Ordinary priority/search roots pay no extra full clone.
        let requirements = self.hidden_selection_requirements(perspective);
        let original = self.has_hidden_selection(perspective).then(|| self.clone());
        for attempt in 0..64 {
            if attempt > 0 {
                *self = original.as_ref().expect("constrained selection").clone();
            }
            let sample_seed = if attempt == 0 {
                seed
            } else {
                mix_seed(seed, attempt)
            };
            self.determinize_zones(perspective, sample_seed);
            let mut rng = ChaCha8Rng::seed_from_u64(mix_seed(sample_seed, 241));
            if self.refresh_hidden_selections(perspective, &mut rng, &requirements) {
                self.repin_revealed_library_cards();
                self.refresh_sampled_decision(perspective);
                return Ok(());
            }
        }
        *self = original.expect("constrained selection");
        Err(AgentError(
            "no sampled deal satisfies retained selection commitments".into(),
        ))
    }

    fn determinize_zones(&mut self, perspective: PlayerId, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let opponent = PlayerId((perspective.0 + 1) % 2);
        assert!(
            self.decklist_known(opponent, perspective),
            "determinize requires the opponent's open decklist"
        );
        // Searching an opponent's hand and library reveals their exact
        // membership to the searcher until this instruction finishes.
        let searching_opponent = self.searches_private_zones(perspective, opponent);
        if !searching_opponent {
            if self.shares_information(perspective, opponent) {
                self.shuffle_private_library(opponent, &mut rng);
            } else {
                let (mut known, mut residual) = self.hidden_hand_partition(opponent, perspective);
                residual.extend(self.foretold_cards(opponent));
                residual.sort_unstable_by_key(|card| card.0);
                residual.shuffle(&mut rng);
                assert!(
                    self.fill_foretold_slots(opponent, &mut residual),
                    "foretold cards are in their own pool"
                );
                let unknown_slots = self.state.zones.size(ZoneType::Hand, opponent) - known.len();
                let mut library = residual.split_off(unknown_slots);
                let pending = self.sampling_fixed_library_cards(opponent, perspective);
                if !pending.is_empty() {
                    library.extend(pending);
                    library.shuffle(&mut rng);
                }
                known.extend(residual);
                self.state.zones.reassign_hidden(opponent, known, library);
            }
        }
        self.shuffle_private_library(perspective, &mut rng);
    }

    fn refresh_sampled_decision(&mut self, perspective: PlayerId) {
        let opponent = PlayerId(1 - perspective.0);
        // An opponent's hidden hand can change under resampling. Rebuild
        // private choices from that sampled authority, never keep old hand IDs.
        let refresh = self
            .state
            .suspended_decision
            .as_ref()
            .is_some_and(|suspended| {
                matches!(suspended.decision, Decision::DiscardFromRevealedHand { .. })
                    || suspended.decision.player() == opponent
                        && matches!(
                            suspended.decision,
                            Decision::SelectCards { .. }
                                | Decision::Scry { .. }
                                | Decision::Surveil { .. }
                                | Decision::LookAndSelect { .. }
                                | Decision::OrderLibrary { .. }
                                | Decision::Learn { .. }
                                | Decision::Discard { .. }
                                | Decision::SearchDistinctLands { .. }
                                | Decision::SearchNamed { .. }
                                | Decision::CastDuringResolution { .. }
                        )
            });
        if refresh {
            if let Some(space) = self.suspended_decision_action_space() {
                self.publish_action_space(space);
            }
        }
    }

    /// Shuffle an unknown library order while preserving its permitted top.
    pub(crate) fn shuffle_private_library(&mut self, player: PlayerId, rng: &mut ChaCha8Rng) {
        let known_top = self
            .library_top_card(player)
            .filter(|_| self.looks_at_library_top(player));
        self.state
            .zones
            .shuffle_canonical(ZoneType::Library, player, rng);
        if let Some(card) = known_top {
            let library = self.state.zones.zone_cards_mut(ZoneType::Library, player);
            library.retain(|other| *other != card);
            library.push(card);
        }
    }

    /// Whether `viewer` knows `owner`'s main deck and sideboard: always its
    /// own, and the opponent's when that decklist is open.
    pub fn decklist_known(&self, owner: PlayerId, viewer: PlayerId) -> bool {
        owner == viewer || self.state.players[owner.0].disclosure.is_open()
    }

    /// Refuse hidden-information sampling for `viewer` when the opponent's
    /// decklist is closed: its unseen pool would be read from the true deck.
    pub fn require_open_decklist(&self, viewer: PlayerId) -> Result<(), AgentError> {
        let opponent = PlayerId((viewer.0 + 1) % 2);
        if self.decklist_known(opponent, viewer) {
            Ok(())
        } else {
            Err(AgentError(format!(
                "decklist closed: player {} cannot sample the opponent's hidden cards",
                viewer.0
            )))
        }
    }

    /// `owner`'s cards that `viewer` cannot see: the library, and for an
    /// opponent also the hand and face-down foretold exile. For an opponent
    /// this is exactly the possible-world pool, including any publicly known
    /// hand cards.
    pub fn unseen_cards(&self, owner: PlayerId, viewer: PlayerId) -> Vec<CardId> {
        let pending_library = self.sampling_fixed_library_cards(owner, viewer);
        let zones: &[ZoneType] = if owner == viewer {
            &[ZoneType::Library]
        } else {
            &[ZoneType::Hand, ZoneType::Library]
        };
        let mut cards: Vec<CardId> = zones
            .iter()
            .flat_map(|zone| self.state.zones.zone_cards(*zone, owner).iter().copied())
            .filter(|card| !pending_library.contains(card))
            .collect();
        if !self.shares_information(viewer, owner) {
            cards.extend(self.foretold_cards(owner));
        }
        cards
    }

    /// Canonical physical witnesses for public definition counts, plus the
    /// residual unseen pool. Never pins the physical copy that was revealed.
    pub(crate) fn hidden_hand_partition(
        &self,
        player: PlayerId,
        viewer: PlayerId,
    ) -> (Vec<CardId>, Vec<CardId>) {
        let fixed = self.sampling_fixed_objects(viewer);
        let is_fixed = |card: CardId| fixed.iter().any(|object| object.entity == card.into());
        let mut pool: Vec<_> = [ZoneType::Hand, ZoneType::Library]
            .into_iter()
            .flat_map(|zone| self.state.zones.zone_cards(zone, player).iter().copied())
            .filter(|card| {
                !is_fixed(*card) || !self.state.zones.contains(*card, ZoneType::Library, player)
            })
            .collect();
        pool.sort_unstable_by_key(|card| card.0);
        let mut counts = self.state.players[player.0].known_hand.clone();
        let mut known = Vec::new();
        let mut residual = Vec::new();
        // Public entrants in a hand pin the physical witness; ordinary reveals
        // still reserve only a definition minimum, as before.
        pool.retain(|card| {
            if is_fixed(*card) && self.state.zones.contains(*card, ZoneType::Hand, player) {
                known.push(*card);
                if let Some(count) = counts.get_mut(&self.state.cards[*card].definition_id) {
                    *count = count.saturating_sub(1);
                }
                false
            } else {
                true
            }
        });
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
        let inspections: Vec<_> = suspended
            .decisions()
            .map(|decision| decision.revealed_cards().to_vec())
            .filter(|cards| !cards.is_empty())
            .collect();
        for revealed in inspections {
            let player = self.state.cards[revealed[0]].owner;
            for card in &revealed {
                self.state.zones.move_card(*card, player, ZoneType::Library);
            }
            let revealed_set = revealed.iter().copied().collect::<BTreeSet<_>>();
            let library = self.state.zones.zone_cards_mut(ZoneType::Library, player);
            library.retain(|card| !revealed_set.contains(card));
            // revealed[0] is the top of the library = last element.
            library.extend(revealed.iter().rev());
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
            let index = self.state.rng_mut().gen_range(0..action_count);
            self.step(index)?;
            steps += 1;
        }
        Ok(self.winner_index())
    }
}
