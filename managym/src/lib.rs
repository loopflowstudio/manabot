pub mod agent;
pub mod benchmark;
<<<<<<< HEAD
=======
#[cfg(feature = "benchmark")]
pub mod benchmark_allocations;
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
pub mod canonical_replay;
pub mod cardsets;
pub mod conformance;
pub mod decision;
<<<<<<< HEAD
=======
pub mod decision_capture;
mod decision_context;
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
pub mod experience;
pub mod flow;
pub mod infra;
pub mod possible_worlds;
pub mod python;
pub mod search_state;
pub mod semantic;
pub mod state;
pub mod study;

pub use agent::env::Env;
pub use agent::vector_env::VectorEnv;
pub use flow::game::Game;
pub use state::hash::{MatchStateHash, MATCH_STATE_HASH_VERSION};
pub use state::player::PlayerConfig;

/// Rules and observation/action meaning; see WORLDS.md for compatibility.
<<<<<<< HEAD
pub const WORLD_VERSION: &str = "w9";
=======
pub const WORLD_VERSION: &str = "w13";
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
