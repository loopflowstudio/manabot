use crate::agent::{action::AgentError, env::Env};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentPolicy {
    None,
    Passive,
    Random,
}

impl OpponentPolicy {
    pub fn select_action(self, env: &mut Env) -> Result<Option<i64>, AgentError> {
        let action = match self {
            Self::None => return Ok(None),
            Self::Passive => env.pass_priority_action_index()? as i64,
            Self::Random => env.random_action_index()? as i64,
        };
        Ok(Some(action))
    }
}
