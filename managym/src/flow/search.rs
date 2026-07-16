// search.rs
// Determinized-search primitives on Game: hidden-information resampling and
// uniformly-random playouts to terminal. Used by flat Monte Carlo search.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{game_object::PlayerId, zone::ZoneType},
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
    /// Hidden information in this engine is exactly: the opponent's hand, and
    /// the order of both libraries (decklists are known to both players; all
    /// other zones — battlefield, graveyard, stack, exile, command — are
    /// public). Accordingly:
    /// - the opponent's hand is replaced by a uniform sample of |hand| cards
    ///   from their unseen pool (hand ∪ library), with the remainder becoming
    ///   their shuffled library;
    /// - `perspective`'s own library is reshuffled (its order is unknown to
    ///   them, but its contents are determined by the public zones + hand).
    ///
    /// All public state — including the current action space — is preserved.
    pub fn determinize(&mut self, perspective: PlayerId, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let opponent = PlayerId((perspective.0 + 1) % 2);
        self.journal_zones();
        self.state.zones.resample_hidden(opponent, &mut rng);
        self.state
            .zones
            .shuffle(ZoneType::Library, perspective, &mut rng);

        // A pending mid-resolution decision may have revealed library cards
        // to the deciding player (scry / look-at-top-N). Those cards are
        // known information — pin them back on top in their revealed order.
        if let Some(suspended) = &self.state.suspended_decision {
            let player = suspended.decision.player();
            let revealed = suspended.decision.revealed_cards().to_vec();
            if !revealed.is_empty() {
                let library = self.state.zones.zone_cards_mut(ZoneType::Library, player);
                library.retain(|card| !revealed.contains(card));
                // revealed[0] is the top of the library = last element.
                for card in revealed.iter().rev() {
                    library.push(*card);
                }
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
            let index = self.state.rng.gen_range(0..action_count);
            self.step(index)?;
            steps += 1;
        }
        Ok(self.winner_index())
    }
}
