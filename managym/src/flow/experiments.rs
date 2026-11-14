// experiments.rs
// Temporary decision-shape switches for learning experiments.
//
// `ExperimentSwitches` is read from the process environment once, when a game
// is set up, and cloned with the game, so every fork of a game keeps the shape
// it started with. An experiment's job command records the variables; replay
// and evaluation must run with the same values.

/// Decision shapes an experiment turns on for a whole process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExperimentSwitches {
    /// `MANAGYM_EXPERIMENT_COMBAT_PICK=1`: a combat declaration is one
    /// repeated decision over every creature still undeclared, finished by
    /// `Action::Decline`, in place of one yes/no decision per creature in
    /// engine order (`flow/tick.rs`).
    pub combat_pick: bool,
}

impl ExperimentSwitches {
    pub(crate) fn from_env() -> Self {
        Self {
            combat_pick: std::env::var("MANAGYM_EXPERIMENT_COMBAT_PICK").as_deref() == Ok("1"),
        }
    }
}
