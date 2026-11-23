// search.rs
// Determinized-search primitives on Game: hidden-information resampling and
// uniformly-random playouts to terminal. Used by flat Monte Carlo search.

use std::collections::BTreeSet;

use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::action::{ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        decision::{Decision, ResolutionRef},
        game::{Game, GameState},
        selection_sampling::HiddenCommitments,
    },
    state::{
        game_object::{ObjectId, PlayerId},
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

    /// Shuffle only unobserved, uncommitted positions, in native seat order.
    /// Zone swaps retain exact appearances and leave the future rules RNG alone.
    pub fn shuffle_unseen_libraries(&mut self, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let fixed = self.pending_library_objects();
        let visible_tops = self.visible_library_tops();
        for owner in [PlayerId(0), PlayerId(1)] {
            let knowledge = &self.state.players[owner.0].known_library;
            let library = self.state.zones.zone_objects(ZoneType::Library, owner);
            let slots: Vec<_> = library
                .iter()
                .enumerate()
                .filter_map(|(index, object)| {
                    let rank = library.len() - 1 - index;
                    let pinned = (rank == 0 && visible_tops[owner.0])
                        || knowledge.public.contains_key(&rank)
                        || knowledge.own.contains_key(&rank)
                        || self.state.library_order_pins[owner.0].contains(object)
                        || fixed.contains(object);
                    (!pinned).then_some(index)
                })
                .collect();
            for i in (1..slots.len()).rev() {
                let j = rng.gen_range(0..=i);
                self.state
                    .zones
                    .swap(ZoneType::Library, owner, slots[i], slots[j]);
            }
        }
    }

    /// Pending instructions retain exact objects. A search over a library
    /// also retains the entire order presented to its chooser.
    fn pending_library_objects(&self) -> BTreeSet<ObjectId> {
        let mut fixed: BTreeSet<_> = self.pending_public_entries().into_iter().collect();
        fixed.extend(self.captured_bindings().map(|binding| binding.object));
        if let Some(suspended) = &self.state.suspended_decision {
            for part in suspended.continuation() {
                match part {
                    ResolutionRef::Frame(frame) => fixed.extend(&frame.selected),
                    ResolutionRef::Decision(decision) => {
                        fixed.extend(decision.revealed_objects());
                        if let Some((_, inspected)) = decision.inspected_library_objects() {
                            fixed.extend(inspected);
                        }
                        let searched = match decision {
                            Decision::SearchNamed { owner, .. }
                            | Decision::SearchDistinctNames { player: owner, .. } => Some(*owner),
                            Decision::SelectCards {
                                from,
                                candidates,
                                selected,
                                ..
                            } => {
                                fixed.extend(candidates);
                                fixed.extend(selected);
                                from.as_ref()
                                    .filter(|s| s.zone == ZoneType::Library)
                                    .map(|s| s.owner)
                            }
                            _ => None,
                        };
                        if let Some(owner) = searched {
                            fixed.extend(self.library_from_top(owner));
                        }
                    }
                }
            }
        }
        fixed
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
    /// - viewer-known library positions reserve distinct definition witnesses;
    ///   only the remaining order is shuffled, for both players.
    ///
    /// - the opponent's face-down foretold slots are filled from the same
    ///   unseen pool's first foretell-bearing cards, as possible-world
    ///   materialization fills them.
    ///
    /// Public facts and own choices are preserved; opponent private choices refresh.
    pub fn determinize(&mut self, perspective: PlayerId, seed: u64) -> Result<(), AgentError> {
        // Rejection is needed only when a retained unknown choice constrains
        // the deal. Ordinary priority/search roots pay no extra full clone.
        let commitments = self.hidden_commitments(perspective);
        let original = commitments.constrains().then(|| self.rollback_snapshot());
        for attempt in 0..64 {
            if attempt > 0 {
                *self = original
                    .as_ref()
                    .expect("constrained selection")
                    .rollback_snapshot();
            }
            let sample_seed = if attempt == 0 {
                seed
            } else {
                mix_seed(seed, attempt)
            };
            self.determinize_zones(perspective, &commitments, sample_seed);
            let mut rng = ChaCha8Rng::seed_from_u64(mix_seed(sample_seed, 241));
            if self.rebind_commitments(&commitments, &mut rng) {
                self.refresh_sampled_library_knowledge(perspective);
                self.assert_library_knowledge();
                self.refresh_sampled_decision(perspective);
                return Ok(());
            }
        }
        *self = original.expect("constrained selection");
        Err(AgentError(
            "no sampled deal satisfies retained selection commitments".into(),
        ))
    }

    fn determinize_zones(
        &mut self,
        perspective: PlayerId,
        commitments: &HiddenCommitments,
        seed: u64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let opponent = PlayerId((perspective.0 + 1) % 2);
        assert!(
            self.decklist_known(opponent, perspective),
            "determinize requires the opponent's open decklist"
        );
        // Searching an opponent's library (with or without their hand) fixes
        // their exact membership for the searcher until the instruction
        // finishes.
        let searching_opponent = self.inspects_library(perspective, opponent);
        if !searching_opponent {
            if self.shares_information(perspective, opponent) {
                self.shuffle_known_library(opponent, perspective, commitments, &mut rng);
            } else {
                let (mut known, mut residual) = self.hidden_hand_partition(opponent, commitments);
                let reserved = self
                    .reserve_library_facts(opponent, perspective, &mut residual, commitments)
                    .expect("known hand and library have distinct witnesses");
                residual.extend(
                    self.foretold_cards(opponent)
                        .map(|card| self.state.object_of(card)),
                );
                residual.sort_unstable_by_key(|object| object.slot());
                residual.shuffle(&mut rng);
                assert!(
                    self.fill_foretold_slots(opponent, &mut residual),
                    "foretold cards are in their own pool"
                );
                let unknown_slots = self.state.zones.size(ZoneType::Hand, opponent) - known.len();
                let mut library = residual.split_off(unknown_slots);
                let pending: Vec<_> = commitments
                    .pinned_library_objects(self, opponent)
                    .into_iter()
                    .filter(|object| !reserved.iter().any(|(_, fixed)| fixed == object))
                    .collect();
                if !pending.is_empty() {
                    library.extend(pending);
                    library.shuffle(&mut rng);
                }
                known.extend(residual);
                let library = Self::place_library_facts(library, reserved)
                    .expect("valid known library positions");
                self.state.reassign_hidden(opponent, known, library);
            }
        }
        self.shuffle_known_library(perspective, perspective, commitments, &mut rng);
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
                                | Decision::ManifestDread { .. }
                                | Decision::Surveil { .. }
                                | Decision::LookAndSelect { .. }
                                | Decision::OrderLibrary { .. }
                                | Decision::Learn { .. }
                                | Decision::Discard { .. }
                                | Decision::SearchDistinctNames { .. }
                                | Decision::SearchNamed { .. }
                                | Decision::CastDuringResolution { .. }
                        )
            });
        let priority = self.current_action_space.as_ref().is_some_and(|space| {
            space.kind == ActionSpaceKind::Priority
                && space
                    .actions
                    .first()
                    .is_some_and(|a| a.player() == opponent)
                && !self.shares_information(perspective, opponent)
        });
        let space = if refresh {
            self.suspended_decision_action_space()
        } else if priority {
            Some(self.priority_action_space(opponent))
        } else {
            None
        };
        if let Some(space) = space {
            // Republishing advances the epoch. Keep an unchanged decision's
            // Commands valid. Publication resolves the chooser from its actions.
            let unchanged = self.current_action_space.as_ref().is_some_and(|current| {
                current.kind == space.kind
                    && current.actions == space.actions
                    && current.focus == space.focus
            });
            if !unchanged {
                self.publish_action_space(space);
            }
        }
    }

    /// Whether `viewer` knows `owner`'s main deck and sideboard: always its
    /// own, and the opponent's when that decklist is open.
    pub fn decklist_known(&self, owner: PlayerId, viewer: PlayerId) -> bool {
        owner == viewer || self.state.disclosure.is_open()
    }

    /// Refuse hidden-information sampling for `viewer` when the opponent's
    /// decklist is closed: its unseen pool would be read from the true deck.
    pub fn require_open_decklist(&self, viewer: PlayerId) -> Result<(), AgentError> {
        self.state.sampling_viewer(viewer.0).map(|_| ())
    }

    /// Canonical physical witnesses for public definition counts, plus the
    /// residual unseen pool. Never pins the physical copy that was revealed.
    pub(crate) fn hidden_hand_partition(
        &self,
        player: PlayerId,
        commitments: &HiddenCommitments,
    ) -> (Vec<ObjectId>, Vec<ObjectId>) {
        let is_fixed = |object: ObjectId| commitments.pins(object);
        let mut pool: Vec<_> = [ZoneType::Hand, ZoneType::Library]
            .into_iter()
            .flat_map(|zone| self.state.zones.zone_objects(zone, player).iter().copied())
            .filter(|object| {
                !is_fixed(*object)
                    || !self
                        .state
                        .zones
                        .contains(*object, ZoneType::Library, player)
            })
            .collect();
        pool.sort_unstable_by_key(|object| object.slot());
        let mut counts = self.state.players[player.0].known_hand.clone();
        let mut known = Vec::new();
        let mut residual = Vec::new();
        // Public entrants in a hand pin the physical witness; ordinary reveals
        // still reserve only a definition minimum, as before.
        pool.retain(|object| {
            if is_fixed(*object) && self.state.zones.contains(*object, ZoneType::Hand, player) {
                known.push(*object);
                if let Some(count) = counts.get_mut(&self.state[*object].definition_id) {
                    *count = count.saturating_sub(1);
                }
                false
            } else {
                true
            }
        });
        for object in pool {
            if let Some(count) = counts.get_mut(&self.state[object].definition_id) {
                if *count > 0 {
                    known.push(object);
                    *count -= 1;
                    continue;
                }
            }
            residual.push(object);
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
        self.publish_action_space(self.priority_action_space(player));
    }

    pub(crate) fn priority_action_space(&self, player: PlayerId) -> ActionSpace {
        ActionSpace {
            player: Some(player),
            kind: ActionSpaceKind::Priority,
            actions: self.compute_player_actions(player),
            focus: Vec::new(),
        }
    }

    /// Inspection lists describe an ordered prefix. Revealed search selections
    /// describe membership only and must never be repositioned as a prefix.
    pub(crate) fn repin_library_inspections(&mut self, viewer: PlayerId) {
        let Some(suspended) = &self.state.suspended_decision else {
            return;
        };
        let inspections: Vec<_> = suspended
            .decisions()
            .filter_map(Decision::library_inspection)
            .filter(|(owner, _)| !self.shares_information(viewer, *owner))
            .map(|(owner, objects)| (owner, objects.to_vec()))
            .collect();
        for (owner, objects) in inspections {
            // This is constrained sampling placement, preserving the facts
            // already checked while binding each inspection slot.
            for object in objects.iter().rev() {
                self.state
                    .zones
                    .reposition(ZoneType::Library, owner, *object, usize::MAX);
            }
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

impl GameState {
    pub(crate) fn sampling_viewer(&self, viewer: usize) -> Result<PlayerId, AgentError> {
        if viewer >= self.players.len() {
            return Err(AgentError(format!(
                "determinize: perspective {viewer} out of range"
            )));
        }
        if !self.disclosure.is_open() {
            return Err(AgentError(format!(
                "decklist closed: player {viewer} cannot sample the opponent's hidden cards"
            )));
        }
        Ok(PlayerId(viewer))
    }
}
