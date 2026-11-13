use crate::agent::{action::AgentError, env::Env, observation::Observation};

/// Who supplies decisions for one native chooser seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerPolicy {
    Agent,
    Random,
    Heuristic,
}

impl PlayerPolicy {
    pub fn select_action(self, env: &mut Env, obs: &Observation) -> Result<i64, AgentError> {
        match self {
            Self::Agent => Err(AgentError("agent requires an external action".into())),
            Self::Random => Ok(env.random_action_index()? as i64),
            Self::Heuristic => env.heuristics[obs.agent.player_index as usize]
                .get_or_insert_with(Default::default)
                .select_action(obs, None)
                .map(|index| index as i64),
        }
    }
}

#[deprecated(note = "use per-seat PlayerPolicy; removal in 0.3.0")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentPolicy {
    None,
    Random,
    Heuristic,
}

#[allow(deprecated)]
impl OpponentPolicy {
    pub fn select_action(self, env: &mut Env, obs: &Observation) -> Result<i64, AgentError> {
        self.player_policies()[1].select_action(env, obs)
    }

    pub fn player_policies(self) -> [PlayerPolicy; 2] {
        [
            PlayerPolicy::Agent,
            match self {
                Self::None => PlayerPolicy::Agent,
                Self::Random => PlayerPolicy::Random,
                Self::Heuristic => PlayerPolicy::Heuristic,
            },
        ]
    }
}
