use crate::agent::{action::AgentError, env::Env, observation::Observation};

/// Who supplies decisions for one native chooser seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlayerPolicy {
    Agent,
    Random,
    Heuristic,
}

impl PlayerPolicy {
    pub fn select_action(self, env: &mut Env, obs: &Observation) -> Result<i64, AgentError> {
        let index = self.select_for_step(env, obs)?;
        env.forget_provenance();
        Ok(index)
    }

    pub(super) fn select_for_step(
        self,
        env: &mut Env,
        obs: &Observation,
    ) -> Result<i64, AgentError> {
        match self {
            Self::Agent => Err(AgentError("agent requires an external action".into())),
            Self::Random => Ok(env.sample_action_index()? as i64),
            Self::Heuristic => env.heuristics[obs.agent.player_index as usize]
                .get_or_insert_with(Default::default)
                .select_action(obs, None)
                .map(|index| index as i64),
        }
    }
}
