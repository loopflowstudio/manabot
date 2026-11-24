//! Detached slot reads and transactional loading of fresh games and branches.
use super::{ObservationWriteError, SlotState, VectorEnv};
use crate::{
    agent::{
        action::AgentError,
        env::{Env, GameRecording, GameSnapshot, RestoreOptions, SnapshotMetadata},
        observation::Observation,
    },
    infra::profiler::{InfoDict, InfoValue},
};
use std::{collections::HashSet, sync::Arc};

/// Loading a branch never spends a fresh seed. A fresh load uses the slot's
/// next scheduled seed and the vector's ordinary roster sampler.
#[derive(Clone)]
pub enum SlotLoad {
    Fresh(usize),
    Snapshot(usize, Arc<GameSnapshot>, RestoreOptions),
}

impl SlotLoad {
    pub fn slot(&self) -> usize {
        match self {
            Self::Fresh(slot) | Self::Snapshot(slot, ..) => *slot,
        }
    }
}

#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
pub struct SlotLoadResult {
    pub slot: usize,
    pub generation: u64,
    pub seed: u64,
    pub acting_player: Option<usize>,
    pub terminal: bool,
    pub source_digest: Option<String>,
}

struct SlotReplacement {
    result: SlotLoadResult,
    env: Env,
    obs: Observation,
    info: InfoDict,
    state: SlotState,
    next_seed: u64,
    recording: GameRecording,
}

impl VectorEnv {
    fn read_slots<T>(
        &self,
        mask: &[bool],
        read: impl Fn(&Env) -> Result<T, AgentError>,
    ) -> Result<Vec<Option<T>>, AgentError> {
        self.validate_actions_len(mask.len())?;
        mask.iter()
            .enumerate()
            .map(|(slot, &selected)| {
                if !selected {
                    return Ok(None);
                }
                self.slots[slot].readable()?;
                read(&self.envs[slot]).map(Some)
            })
            .collect()
    }

    pub fn snapshot_slots(
        &self,
        mask: &[bool],
    ) -> Result<Vec<Option<Arc<GameSnapshot>>>, AgentError> {
        self.read_slots(mask, |env| env.snapshot().map(Arc::new))
    }

    pub fn slot_metadata(
        &self,
        mask: &[bool],
    ) -> Result<Vec<Option<SnapshotMetadata>>, AgentError> {
        self.read_slots(mask, Env::snapshot_metadata)
    }

    /// Prepare and admit every selected position before publishing the first row.
    /// The registered encoder validates against the same layout used for writing.
    pub fn load_slots_into(
        &mut self,
        requests: &[SlotLoad],
        mut validate: impl FnMut(usize, &Observation) -> Result<(), ObservationWriteError>,
        mut write: impl FnMut(usize, &Observation, bool) -> Result<(), ObservationWriteError>,
    ) -> Result<Vec<(SlotLoadResult, InfoDict)>, ObservationWriteError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_ready_for_step()?;
        let mut seen = HashSet::with_capacity(requests.len());
        for request in requests {
            let slot = request.slot();
            if slot >= self.len() || !seen.insert(slot) {
                return Err(AgentError(format!("invalid or duplicate slot {slot}")).into());
            }
        }
        let mut prepared = Vec::with_capacity(requests.len());
        for request in requests {
            let next = self.prepare_slot(request)?;
            validate(request.slot(), &next.obs)?;
            prepared.push(next);
        }
        let mut results = Vec::with_capacity(prepared.len());
        for next in prepared {
            let slot = next.result.slot;
            write(slot, &next.obs, next.result.terminal)?;
            self.envs[slot].replace_slot(next.env, next.recording);
            self.slots[slot] = next.state;
            self.next_seeds[slot] = next.next_seed;
            results.push((next.result, next.info));
        }
        Ok(results)
    }

    fn prepare_slot(&self, request: &SlotLoad) -> Result<SlotReplacement, AgentError> {
        let slot = request.slot();
        let previous = &self.envs[slot];
        let rewind = matches!(request, SlotLoad::Snapshot(..))
            && self.slots[slot] == SlotState::FreshOpening;
        let mut next_seed = self.next_seeds[slot];
        let (env, obs, mut info, state, source_digest) = match request {
            SlotLoad::Fresh(_) => {
                let mut env = previous.replacement_root(next_seed);
                let (obs, info) = Self::reset_env_on_env(
                    &self.uniform_roster,
                    &mut env,
                    &mut next_seed,
                    self.seed_stride,
                    &self.player_configs[slot],
                )?;
                (env, obs, info, SlotState::FreshOpening, None)
            }
            SlotLoad::Snapshot(_, source, options) => {
                if !source.compatible(
                    *options,
                    self.player_policies,
                    self.skip_trivial,
                    self.decision_memory_enabled,
                    self.decision_memory_include_own,
                ) {
                    return Err(AgentError(
                        "snapshot policies, skip_trivial or decision memory differ from vector"
                            .into(),
                    ));
                }
                let env = source.restore(*options)?;
                let obs = env.observation_for_player(env.current_agent_index().unwrap_or(0))?;
                if rewind {
                    next_seed = previous.reset_seed();
                }
                (
                    env,
                    obs,
                    InfoDict::new(),
                    SlotState::Playing,
                    Some(source.state_digest().to_owned()),
                )
            }
        };
        let terminal = env.is_game_over();
        if let Some(winner) = env.winner_index() {
            info.insert("winner_index".into(), InfoValue::Int(winner as i64));
        }
        let result = SlotLoadResult {
            slot,
            generation: env.capture_generation,
            seed: env.reset_seed(),
            acting_player: if terminal {
                None
            } else {
                env.current_agent_index()
            },
            terminal,
            source_digest,
        };
        let recording = if rewind || self.slots[slot] == SlotState::Poisoned {
            Default::default()
        } else {
            previous.current_recording()?
        };
        Ok(SlotReplacement {
            result,
            env,
            obs,
            info,
            state: if terminal { SlotState::Terminal } else { state },
            next_seed,
            recording,
        })
    }
}
