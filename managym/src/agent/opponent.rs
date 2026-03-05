use crate::agent::{action::AgentError, env::Env};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentPolicy {
    None,
    Passive,
    Random,
}

impl OpponentPolicy {
    pub fn select_action(self, env: &mut Env) -> Result<i64, AgentError> {
        match self {
            Self::None => Err(AgentError("opponent policy disabled".to_string())),
            Self::Passive => Ok(env.pass_priority_action_index()? as i64),
            Self::Random => Ok(env.random_action_index()? as i64),
        }
    }
}
