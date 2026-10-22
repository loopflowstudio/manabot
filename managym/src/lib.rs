pub mod agent;
pub mod benchmark;
pub mod canonical_replay;
pub mod cardsets;
pub mod conformance;
pub mod decision;
<<<<<<< HEAD
=======
mod decision_context;
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
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
pub const WORLD_VERSION: &str = "w10";
=======
pub const WORLD_VERSION: &str = "w11";
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
