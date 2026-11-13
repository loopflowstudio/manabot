use crate::agent::{action::AgentError, env::Env, observation::Observation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentPolicy {
    None,
    Random,
    Heuristic,
}

impl OpponentPolicy {
    pub fn select_action(self, env: &mut Env, obs: &Observation) -> Result<i64, AgentError> {
        match self {
            Self::None => Err(AgentError("opponent policy disabled".to_string())),
            Self::Heuristic => env
                .heuristic
                .get_or_insert_with(Default::default)
                .select_action(obs, None)
                .map(|i| i as i64),
            Self::Random => Ok(env.random_action_index()? as i64),
        }
    }
}
