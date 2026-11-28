use crate::agent::{action::AgentError, env::Env, observation::Observation};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr};

/// Who supplies decisions for one native chooser seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerPolicy {
    Agent,
    Random,
    Heuristic { max_actions: Option<u32> },
}

impl fmt::Display for PlayerPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agent => f.write_str("agent"),
            Self::Random => f.write_str("random"),
            Self::Heuristic { max_actions: None } => f.write_str("heuristic"),
            Self::Heuristic {
                max_actions: Some(max),
            } => write!(f, "heuristic:{max}"),
        }
    }
}

impl FromStr for PlayerPolicy {
    type Err = AgentError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "agent" => Ok(Self::Agent),
            "random" => Ok(Self::Random),
            "heuristic" => Ok(Self::Heuristic { max_actions: None }),
            _ => {
                let max_actions = name
                    .strip_prefix("heuristic:")
                    .and_then(|value| value.parse::<u32>().ok())
                    .filter(|&value| value > 0)
                    .ok_or_else(|| AgentError(format!("unsupported player policy: {name}")))?;
                Ok(Self::Heuristic {
                    max_actions: Some(max_actions),
                })
            }
        }
    }
}

impl Serialize for PlayerPolicy {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for PlayerPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl PlayerPolicy {
    pub fn is_heuristic(&self) -> bool {
        matches!(self, Self::Heuristic { .. })
    }

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
            Self::Heuristic { max_actions } => env.heuristics[obs.agent.player_index as usize]
                .get_or_insert_with(Default::default)
                .select_action(obs, max_actions.map(|max| max as usize))
                .map(|index| index as i64),
        }
    }
}
