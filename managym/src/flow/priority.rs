//! Who holds priority, how many players have passed in a row, and the
//! CR 104.4b watch over the resolutions those passes allow.

use std::collections::BTreeMap;

use crate::state::game_object::PlayerId;

#[derive(Clone, Debug, serde::Serialize)]
pub struct PriorityState {
    pub holder: PlayerId,
    pub consecutive_passes: usize,
    pub pass_count: usize,
    pub sba_done: bool,
    /// Resolutions since a player last did anything but pass, while the
    /// stack has not emptied. Boxed so the common empty case stays one word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_watch: Option<Box<MandatoryLoopWatch>>,
    /// Count this resolution once its consequences have stabilized. Only a
    /// forced decision during resolution or stabilization can be published
    /// with a count pending. Those continuations are hashed. Priority spends
    /// the count and a decision with multiple options clears it.
    #[serde(skip)]
    pub resolution_owed: bool,
}

impl Default for PriorityState {
    fn default() -> Self {
        Self {
            holder: PlayerId(0),
            consecutive_passes: 0,
            pass_count: 0,
            sba_done: false,
            loop_watch: None,
            resolution_owed: false,
        }
    }
}

impl PriorityState {
    pub fn start_round(&mut self, active: PlayerId) {
        self.holder = active;
        self.consecutive_passes = 0;
        self.sba_done = false;
    }

    pub fn clear_loop_watch(&mut self) {
        self.loop_watch = None;
        self.resolution_owed = false;
    }

    pub fn on_pass(&mut self, next: PlayerId) {
        self.consecutive_passes += 1;
        self.holder = next;
    }

    pub fn on_non_pass_action(&mut self, active: PlayerId) {
        self.start_round(active);
        self.clear_loop_watch();
    }
}

/// What one watched resolution left behind. `fingerprint` is the
/// rules-relevant state with life and poison totals masked
/// ([`crate::flow::game::GameState::loop_fingerprint`]).
#[derive(Clone, Copy, Debug)]
pub struct LoopSample {
    pub fingerprint: u64,
    pub life: [i32; 2],
    pub poison: [i32; 2],
    /// Whether each seat can lose the game (no Herald of Eternal Dawn).
    pub can_lose: [bool; 2],
}

/// The lowest life and highest poison each seat had reached when a
/// fingerprint was last seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
struct LoopMark {
    life_low: [i32; 2],
    poison_high: [i32; 2],
}

/// CR 104.4b: a game in a loop of mandatory actions, "repeating a sequence
/// of events with no way to stop", is a draw. Watches the resolutions that
/// follow from passes alone; any other action, a new step or an emptied
/// stack ends the watch.
///
/// A loop is a return to an earlier rules-relevant state while no player who
/// can lose has moved toward losing since: no new low of life, no new high
/// of poison. A finite chain over counters, permanents or cards changes that
/// state at every step, so it never repeats and is never drawn; a drain
/// against a player who can lose sets a new low each cycle and runs to the
/// loss. Only after [`Self::UNWATCHED`] resolutions does the watch
/// fingerprint at all, so ordinary trigger chains cost a counter.
/// [`Self::CAP`] ends a loop that grows something unbounded, which never
/// repeats a state.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MandatoryLoopWatch {
    resolutions: u32,
    life_low: [i32; 2],
    poison_high: [i32; 2],
    seen: BTreeMap<u64, LoopMark>,
}

impl MandatoryLoopWatch {
    /// Resolutions a chain may run before its states are fingerprinted.
    pub const UNWATCHED: u32 = 64;
    /// Resolutions without any other action that make a loop regardless.
    pub const CAP: u32 = 10_000;

    /// Resolutions counted since the watch began.
    pub fn resolutions(&self) -> u32 {
        self.resolutions
    }

    /// Record one resolution; true once the loop is established. `sample`
    /// runs only once the chain has outlasted [`Self::UNWATCHED`].
    pub fn repeats(&mut self, sample: impl FnOnce() -> LoopSample) -> bool {
        self.resolutions += 1;
        if self.resolutions >= Self::CAP {
            return true;
        }
        if self.resolutions <= Self::UNWATCHED {
            return false;
        }
        let now = sample();
        let fingerprint = now.fingerprint;
        if self.seen.is_empty() {
            self.life_low = now.life;
            self.poison_high = now.poison;
        }
        for seat in 0..2 {
            self.life_low[seat] = self.life_low[seat].min(now.life[seat]);
            self.poison_high[seat] = self.poison_high[seat].max(now.poison[seat]);
        }
        let mark = LoopMark {
            life_low: self.life_low,
            poison_high: self.poison_high,
        };
        self.seen.insert(fingerprint, mark).is_some_and(|before| {
            (0..2).all(|seat| {
                !now.can_lose[seat]
                    || (before.life_low[seat] == mark.life_low[seat]
                        && before.poison_high[seat] == mark.poison_high[seat])
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{LoopSample, MandatoryLoopWatch};

    fn sample(fingerprint: u64, life: i32, poison: i32) -> LoopSample {
        LoopSample {
            fingerprint,
            life: [20, life],
            poison: [0, poison],
            can_lose: [true, true],
        }
    }

    /// A watch whose chain has outlasted the unwatched prefix.
    fn watched() -> MandatoryLoopWatch {
        MandatoryLoopWatch {
            resolutions: MandatoryLoopWatch::UNWATCHED,
            ..Default::default()
        }
    }

    #[test]
    fn a_short_chain_is_never_fingerprinted() {
        let mut watch = MandatoryLoopWatch::default();
        assert!((0..MandatoryLoopWatch::UNWATCHED)
            .all(|_| !watch.repeats(|| unreachable!("sampled an unwatched resolution"))));
    }

    #[test]
    fn states_that_never_repeat_are_no_loop() {
        let mut watch = watched();
        assert!((0..1000).all(|n| !watch.repeats(|| sample(n, 20, 0))));
    }

    #[test]
    fn a_repeated_state_is_a_loop_unless_a_player_who_can_lose_is_losing() {
        let mut steady = watched();
        // The first cycle reaches a new low of 19; the next one, at no new
        // low, repeats.
        assert!(!steady.repeats(|| sample(1, 20, 0)));
        assert!(!steady.repeats(|| sample(2, 19, 0)));
        assert!(!steady.repeats(|| sample(1, 20, 0)));
        assert!(steady.repeats(|| sample(2, 19, 0)));

        let mut draining = watched();
        assert!((0..1000).all(|n| !draining.repeats(|| sample(n % 2, 2000 - n as i32, 0))));

        let mut poisoning = watched();
        assert!((0..1000).all(|n| !poisoning.repeats(|| sample(n % 2, 20, n as i32))));

        let mut protected = watched();
        let protected_sample = |n: u64| LoopSample {
            can_lose: [true, false],
            ..sample(n % 2, 2000 - n as i32, n as i32)
        };
        assert!(!protected.repeats(|| protected_sample(0)));
        assert!(!protected.repeats(|| protected_sample(1)));
        assert!(protected.repeats(|| protected_sample(2)));
    }

    #[test]
    fn an_unbounded_chain_ends_at_the_cap() {
        let mut watch = MandatoryLoopWatch::default();
        let ended = (0..u64::from(MandatoryLoopWatch::CAP))
            .position(|n| watch.repeats(|| sample(n, 20, 0)));
        assert_eq!(ended, Some(MandatoryLoopWatch::CAP as usize - 1));
    }
}
