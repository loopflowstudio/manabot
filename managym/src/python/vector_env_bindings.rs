#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::{cell::Cell, slice, time::Instant};

#[cfg(feature = "python")]
use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyModule},
};

#[cfg(feature = "python")]
use crate::{
    agent::{
        action::AgentError,
        env::HandSnapshot,
        observation::Observation,
        observation_encoder::{
            encode, encode_into, EncodedObservationMut, ObservationEncoderConfig, ACTION_DIM,
        },
        rollout_pool::RolloutPool,
        vector_env::{ObservationWriteError, VectorEnv},
    },
    infra::profiler::InfoDict,
    python::{
        bindings::{map_agent_err, PyEnv, PyObservation, PyPlayerConfig},
        convert::{info_dict_to_pydict, require_numpy_array},
    },
    state::player::PlayerConfig,
};

#[cfg(feature = "python")]
type PyResetResult = (PyObservation, PyObject);
#[cfg(feature = "python")]
type PyStepResult = (PyObservation, f64, bool, bool, PyObject);

#[cfg(feature = "python")]
struct ObservationBuffers {
    relations: Option<Py<PyAny>>,
    effects: Option<Py<PyAny>>,
    effect_details: Option<Py<PyAny>>,
    effect_members: Option<Py<PyAny>>,
    name_candidates: Option<Py<PyAny>>,
    facts: Option<Py<PyAny>>,
    semantic_cards: Py<PyAny>,
    known_hand: Py<PyAny>,
    decklists: Option<Py<PyAny>>,
    player_turn_state: Option<Py<PyAny>>,
    card_turn_state: Option<Py<PyAny>>,
    agent_player: Py<PyAny>,
    opponent_player: Py<PyAny>,
    agent_cards: Py<PyAny>,
    opponent_cards: Py<PyAny>,
    agent_permanents: Py<PyAny>,
    opponent_permanents: Py<PyAny>,
    actions: Py<PyAny>,
    events: Py<PyAny>,
    action_focus: Py<PyAny>,
    agent_player_valid: Py<PyAny>,
    opponent_player_valid: Py<PyAny>,
    agent_cards_valid: Py<PyAny>,
    opponent_cards_valid: Py<PyAny>,
    agent_permanents_valid: Py<PyAny>,
    opponent_permanents_valid: Py<PyAny>,
    actions_valid: Py<PyAny>,
    events_valid: Py<PyAny>,
    // Step outcomes; absent for observation-only writes.
    step_state: Option<StepStateBuffers>,
}

#[cfg(feature = "python")]
struct StepStateBuffers {
    rewards: Py<PyAny>,
    terminated: Py<PyAny>,
    truncated: Py<PyAny>,
}

#[cfg(feature = "python")]
#[pyclass(name = "VectorEnv")]
pub struct PyVectorEnv {
    inner: VectorEnv,
    config: ObservationEncoderConfig,
    num_envs: usize,
    buffers: Option<ObservationBuffers>,
    last_info: Vec<InfoDict>,
    timings: BufferedCallTimings,
}

/// Cumulative wall time inside buffered calls: every step or reset, split into
/// stepping (all work in the call except encoding) and observation encoding.
#[cfg(feature = "python")]
#[derive(Default)]
struct BufferedCallTimings {
    stepping_nanos: u64,
    encoding_nanos: u64,
}

// A buffered call stays on its calling thread while the GIL is released, and each
// stream's row is encoded inside it, so a thread-local counter needs no locks.
#[cfg(feature = "python")]
thread_local! {
    static ENCODING_NANOS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVectorEnv {
    #[cfg(feature = "benchmark")]
    fn _benchmark_load_root(&mut self, root: &crate::python::bindings::PyEnv) -> PyResult<()> {
        let env = root
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
        self.inner.benchmark_load_root(&env).map_err(map_agent_err)
    }

    #[cfg(feature = "benchmark")]
    fn _benchmark_allocations_start(&self) {
        crate::benchmark_allocations::start();
    }

    #[cfg(feature = "benchmark")]
    fn _benchmark_allocations_stop(&self) -> (u64, u64) {
        crate::benchmark_allocations::stop()
    }

    #[new]
    #[pyo3(signature = (num_envs, seed=0, skip_trivial=true, opponent_policy=None, seed_stride=None, *, player_policies=None))]
    fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        opponent_policy: Option<&str>,
        seed_stride: Option<u64>,
        player_policies: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let policies = parse_player_policies(player_policies, opponent_policy)?;
        let mut inner = VectorEnv::new_with_player_policies(num_envs, seed, skip_trivial, policies);
        if let Some(stride) = seed_stride {
            inner = inner.with_seed_stride(stride).map_err(map_agent_err)?;
        }

        Ok(Self {
            inner,
            config: ObservationEncoderConfig::default(),
            num_envs,
            buffers: None,
            last_info: Vec::new(),
            timings: BufferedCallTimings::default(),
        })
    }

    fn reset_all(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<Vec<PyResetResult>> {
        let configs = player_configs.into_iter().map(PlayerConfig::from).collect();
        let results = self.inner.reset_all(configs).map_err(map_agent_err)?;
        self.last_info = results.iter().map(|(_, info)| info.clone()).collect();

        Ok(results
            .into_iter()
            .map(|(obs, info)| {
                let py_dict = info_dict_to_pydict(py, &info);
                (PyObservation::from(obs), py_dict.into_any().unbind())
            })
            .collect())
    }

    fn step(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<Vec<PyStepResult>> {
        let results = self.inner.step(&actions).map_err(map_agent_err)?;
        self.last_info = results.iter().map(|result| result.info.clone()).collect();

        Ok(results
            .into_iter()
            .map(|result| {
                let py_dict = info_dict_to_pydict(py, &result.info);
                (
                    PyObservation::from(result.obs),
                    result.reward,
                    result.terminated,
                    result.truncated,
                    py_dict.into_any().unbind(),
                )
            })
            .collect())
    }

    #[pyo3(signature = (buffers, action_overflow="error"))]
    fn set_buffers(&mut self, buffers: Bound<'_, PyDict>, action_overflow: &str) -> PyResult<()> {
        let mut config = input_buffer_config(&buffers)?;
        config.action_overflow = action_overflow.parse().map_err(PyValueError::new_err)?;
        let buffers = read_observation_buffers(&buffers, self.num_envs, &config, true)?;
        self.config = config;
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Scenario harness only; does not change deck admission or future resets.
    fn scenario_naming_vocabulary_into_buffers(&mut self, py: Python<'_>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.scenario_naming_vocabulary_into(|index, obs| {
                write_buffers
                    .write_encoded_row(index, obs, 0.0, false, false, &config)
                    .map_err(AgentError::from)
            })?;
            Ok(vec![InfoDict::new(); inner.len()])
        })
    }

    fn reset_all_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<()> {
        let configs: Vec<PlayerConfig> =
            player_configs.into_iter().map(PlayerConfig::from).collect();
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_all_into(configs, |env_index, obs, reward, terminated, truncated| {
                write_buffers
                    .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    fn reset_each_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<Vec<PyPlayerConfig>>,
    ) -> PyResult<()> {
        let configs = player_configs
            .into_iter()
            .map(|row| row.into_iter().map(PlayerConfig::from).collect())
            .collect();
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_each_into(configs, |env_index, obs, reward, terminated, truncated| {
                write_buffers
                    .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    fn reassign_roots_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<Option<Vec<PyPlayerConfig>>>,
    ) -> PyResult<()> {
        let configs = player_configs
            .into_iter()
            .map(|row| row.map(|players| players.into_iter().map(PlayerConfig::from).collect()))
            .collect();
        self.run_into_buffers(py, move |inner, buffers, config| {
            inner.reassign_roots_into(
                configs,
                |_, obs| {
                    encode(obs, &config)
                        .map(|_| ())
                        .map_err(|error| AgentError(error.to_string()))
                },
                |index, obs, reward, terminated, truncated| {
                    buffers
                        .write_encoded_row(index, obs, reward, terminated, truncated, &config)
                        .map_err(AgentError::from)
                },
            )
        })
    }

    #[pyo3(signature = (actions, active=None))]
    fn step_into_buffers(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
        active: Option<Vec<bool>>,
    ) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            if let Some(mask) = &active {
                if mask.len() != inner.len() {
                    return Err(AgentError(
                        "active mask length must match environments".into(),
                    ));
                }
                for (index, enabled) in mask.iter().enumerate() {
                    if !enabled {
                        write_buffers.write_step_state(index, 0.0, false, false)?;
                    }
                }
            }
            inner.step_into(
                &actions,
                active.as_deref(),
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                        .map_err(AgentError::from)
                },
            )
        })
    }

    /// Per-env `skip_trivial` collapse counters for the current games.
    /// Each counter resets when its env's game resets.
    fn skip_trivial_counts(&self) -> Vec<usize> {
        self.inner.skip_trivial_counts()
    }

    /// Per-env index of the player holding the current decision (-1 if no
    /// decision is pending).
    fn current_agent_indices(&self) -> Vec<i64> {
        self.inner.current_agent_indices()
    }

    fn set_uniform_roster(&mut self, roster: Vec<PyPlayerConfig>) -> PyResult<()> {
        self.inner
            .set_uniform_roster(roster.into_iter().map(PlayerConfig::from).collect())
            .map_err(map_agent_err)
    }

    #[pyo3(signature = (decks, distinct=true))]
    fn set_training_decks(&mut self, decks: Vec<PyPlayerConfig>, distinct: bool) -> PyResult<()> {
        self.inner
            .set_training_decks(
                decks.into_iter().map(PlayerConfig::from).collect(),
                distinct,
            )
            .map_err(map_agent_err)
    }

    fn current_player_configs(&self) -> Vec<Vec<PyPlayerConfig>> {
        self.inner
            .current_player_configs()
            .into_iter()
            .map(|players| players.into_iter().map(PyPlayerConfig::from).collect())
            .collect()
    }

    fn current_matchup_indices(&self) -> Vec<(usize, usize)> {
        self.inner.current_matchup_indices()
    }

    fn current_game_seeds(&self) -> Vec<u64> {
        self.inner.current_game_seeds()
    }

    fn current_game_generations(&self) -> Vec<u64> {
        self.inner.current_game_generations()
    }

    fn hand_snapshots(&self, players: Vec<usize>) -> PyResult<Vec<HandSnapshot>> {
        self.inner.hand_snapshots(&players).map_err(map_agent_err)
    }

    fn current_action_summaries(&self) -> Vec<(usize, String)> {
        self.inner.current_action_summaries()
    }

    #[pyo3(signature = (active=None))]
    fn semantic_decision_frames_json(
        &self,
        active: Option<Vec<bool>>,
    ) -> PyResult<Vec<Option<String>>> {
        self.inner
            .semantic_decision_frames_json(active.as_deref())
            .map_err(map_agent_err)
    }

    #[pyo3(signature = (actions, active, recover_capacity=false))]
    fn step_isolated_into_buffers(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
        active: Vec<bool>,
        recover_capacity: bool,
    ) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, buffers, config| {
            // Never leave a preceding terminal flag on a failed/inactive row.
            for index in 0..inner.len() {
                buffers.write_step_state(index, 0.0, false, false)?;
            }
            inner.step_isolated_into(
                &actions,
                &active,
                recover_capacity,
                |index, obs, reward, done, truncated| {
                    buffers.write_encoded_row(index, obs, reward, done, truncated, &config)
                },
            )
        })
    }

    fn replace_failed_into_buffers(&mut self, py: Python<'_>, failed: Vec<bool>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, buffers, config| {
            inner.replace_failed_into(&failed, |index, obs, reward, done, truncated| {
                buffers
                    .write_encoded_row(index, obs, reward, done, truncated, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    fn reset_seeds_into_buffers(&mut self, py: Python<'_>, seeds: Vec<u64>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_seeds_into(seeds, |index, obs, reward, terminated, truncated| {
                write_buffers
                    .write_encoded_row(index, obs, reward, terminated, truncated, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    fn enable_exposure_capture(&mut self) -> PyResult<()> {
        self.inner.enable_exposure_capture().map_err(map_agent_err)
    }

    fn take_exposure(&mut self) -> Vec<Vec<String>> {
        self.inner.take_exposure()
    }

    fn current_exposure(&self) -> PyResult<Vec<String>> {
        self.inner.current_exposure().map_err(map_agent_err)
    }

    fn get_last_info(&self, py: Python<'_>) -> Vec<PyObject> {
        self.last_info
            .iter()
            .map(|info| info_dict_to_pydict(py, info).into_any().unbind())
            .collect()
    }

    /// Cumulative `(stepping_seconds, encoding_seconds)` of buffered calls.
    fn buffered_call_timings(&self) -> (f64, f64) {
        (
            self.timings.stepping_nanos as f64 / 1e9,
            self.timings.encoding_nanos as f64 / 1e9,
        )
    }
}

/// Every configurable axis follows the supplied ObservationSpace on both paths.
/// Companion shapes, dtypes and storage are checked before buffers are written.
/// The buffer set must be policy input v9's: fixed relation, shared-effect and
/// name-candidate widths, with fact capacity read from the fact axis.
#[cfg(feature = "python")]
pub(super) fn input_buffer_config(
    buffers: &Bound<'_, PyDict>,
) -> PyResult<ObservationEncoderConfig> {
    // Strip only the optional batch axis. Full companion and batch shapes are
    // validated by the caller before any writes.
    let shape = |name: &str| -> PyResult<[usize; 2]> {
        let dimensions: Vec<usize> = buffers
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?
            .getattr("shape")?
            .extract()?;
        match dimensions.as_slice() {
            &[rows, width] | &[_, rows, width] if !dimensions.contains(&0) => Ok([rows, width]),
            _ => Err(PyValueError::new_err(format!(
                "invalid {name} buffer shape"
            ))),
        }
    };
    let [max_events, event_width] = shape("events")?;
    let policy_history = match event_width {
        crate::agent::observation_encoder::EVENT_DIM => false,
        crate::agent::policy_history::HISTORY_DIM => true,
        _ => return Err(PyValueError::new_err("unsupported event buffer width")),
    };
    if policy_history && max_events > crate::agent::policy_history::HISTORY_LIMIT {
        return Err(PyValueError::new_err(
            "public history v1 requires 1..32 rows",
        ));
    }
    let mut config = ObservationEncoderConfig {
        max_cards_per_player: shape("agent_cards")?[0],
        max_permanents_per_player: shape("agent_permanents")?[0],
        max_actions: shape("actions")?[0],
        max_focus_objects: shape("action_focus")?[1],
        max_events,
        policy_history,
        ..ObservationEncoderConfig::default()
    };
    // One layout: every v9 buffer is present at its fixed width.
    let fixed = [
        (
            "relations",
            Some(crate::agent::continuation::RELATION_ROWS),
            crate::agent::continuation::RELATION_DIM,
        ),
        ("facts", None, crate::agent::observation_encoder::FACT_DIM),
        (
            "effects",
            Some(crate::agent::shared_effects::EFFECT_ROWS),
            crate::agent::shared_effects::EFFECT_DIM,
        ),
        (
            "effect_details",
            Some(crate::agent::shared_effects::DETAIL_ROWS),
            crate::agent::shared_effects::DETAIL_DIM,
        ),
        (
            "effect_members",
            Some(crate::agent::shared_effects::MEMBER_ROWS),
            crate::agent::shared_effects::MEMBER_DIM,
        ),
        (
            "name_candidates",
            Some(config.max_actions),
            crate::agent::observation_encoder::NAME_DIM,
        ),
    ];
    for (name, rows, width) in fixed {
        if !buffers.contains(name)? {
            return Err(PyValueError::new_err(format!(
                "policy input v9 requires a {name} buffer"
            )));
        }
        let [actual_rows, actual_width] = shape(name)?;
        if actual_width != width || rows.is_some_and(|rows| rows != actual_rows) {
            return Err(PyValueError::new_err(format!(
                "policy input v9 requires {name} of shape [{}, {width}]",
                rows.map_or("capacity".to_string(), |rows| rows.to_string())
            )));
        }
    }
    if !buffers.contains("decklists")? {
        return Err(PyValueError::new_err(
            "policy input v9 requires a decklists buffer",
        ));
    }
    config.max_facts = shape("facts")?[0];
    // Input v10 is v9 plus both turn-state buffers.
    config.turn_state = match (
        buffers.contains("player_turn_state")?,
        buffers.contains("card_turn_state")?,
    ) {
        (false, false) => false,
        (true, true) => {
            let player = shape("player_turn_state")?;
            let card = shape("card_turn_state")?;
            if player != [2, crate::agent::observation_encoder::TURN_STATE_DIM]
                || card != [2, config.max_cards_per_player]
            {
                return Err(PyValueError::new_err(format!(
                    "policy input v10 requires player_turn_state of shape [2, {}] and \
                     card_turn_state of shape [2, {}]",
                    crate::agent::observation_encoder::TURN_STATE_DIM,
                    config.max_cards_per_player
                )));
            }
            true
        }
        _ => {
            return Err(PyValueError::new_err(
                "policy input v10 requires both player_turn_state and card_turn_state",
            ))
        }
    };
    Ok(config)
}

#[cfg(feature = "python")]
fn read_observation_buffers(
    buffers: &Bound<'_, PyDict>,
    n: usize,
    c: &ObservationEncoderConfig,
    step_state: bool,
) -> PyResult<ObservationBuffers> {
    let required = |name: &str, shape: &[usize]| -> PyResult<Option<Py<PyAny>>> {
        Ok(Some(
            require_numpy_array(buffers, name, shape, "float32")?.unbind(),
        ))
    };
    Ok(ObservationBuffers {
        name_candidates: required(
            "name_candidates",
            &[
                n,
                c.max_actions,
                crate::agent::observation_encoder::NAME_DIM,
            ],
        )?,
        relations: required(
            "relations",
            &[
                n,
                crate::agent::continuation::RELATION_ROWS,
                crate::agent::continuation::RELATION_DIM,
            ],
        )?,
        effects: required(
            "effects",
            &[
                n,
                crate::agent::shared_effects::EFFECT_ROWS,
                crate::agent::shared_effects::EFFECT_DIM,
            ],
        )?,
        effect_details: required(
            "effect_details",
            &[
                n,
                crate::agent::shared_effects::DETAIL_ROWS,
                crate::agent::shared_effects::DETAIL_DIM,
            ],
        )?,
        effect_members: required(
            "effect_members",
            &[
                n,
                crate::agent::shared_effects::MEMBER_ROWS,
                crate::agent::shared_effects::MEMBER_DIM,
            ],
        )?,
        facts: required(
            "facts",
            &[n, c.max_facts, crate::agent::observation_encoder::FACT_DIM],
        )?,
        known_hand: require_numpy_array(
            buffers,
            "known_hand",
            &[n, 2, c.max_cards_per_player, 2],
            "float32",
        )?
        .unbind(),
        player_turn_state: if c.turn_state {
            required(
                "player_turn_state",
                &[n, 2, crate::agent::observation_encoder::TURN_STATE_DIM],
            )?
        } else {
            None
        },
        card_turn_state: if c.turn_state {
            required("card_turn_state", &[n, 2, c.max_cards_per_player])?
        } else {
            None
        },
        decklists: required(
            "decklists",
            &[
                n,
                2,
                c.max_cards_per_player,
                crate::agent::observation_encoder::DECKLIST_DIM,
            ],
        )?,
        semantic_cards: require_numpy_array(
            buffers,
            "semantic_cards",
            &[n, 2, c.max_cards_per_player],
            "float32",
        )?
        .unbind(),
        agent_player: require_numpy_array(
            buffers,
            "agent_player",
            &[n, 1, crate::agent::observation_encoder::PLAYER_DIM],
            "float32",
        )?
        .unbind(),
        opponent_player: require_numpy_array(
            buffers,
            "opponent_player",
            &[n, 1, crate::agent::observation_encoder::PLAYER_DIM],
            "float32",
        )?
        .unbind(),
        agent_cards: require_numpy_array(
            buffers,
            "agent_cards",
            &[
                n,
                c.max_cards_per_player,
                crate::agent::observation_encoder::CARD_DIM,
            ],
            "float32",
        )?
        .unbind(),
        opponent_cards: require_numpy_array(
            buffers,
            "opponent_cards",
            &[
                n,
                c.max_cards_per_player,
                crate::agent::observation_encoder::CARD_DIM,
            ],
            "float32",
        )?
        .unbind(),
        agent_permanents: require_numpy_array(
            buffers,
            "agent_permanents",
            &[
                n,
                c.max_permanents_per_player,
                crate::agent::observation_encoder::PERMANENT_DIM,
            ],
            "float32",
        )?
        .unbind(),
        opponent_permanents: require_numpy_array(
            buffers,
            "opponent_permanents",
            &[
                n,
                c.max_permanents_per_player,
                crate::agent::observation_encoder::PERMANENT_DIM,
            ],
            "float32",
        )?
        .unbind(),
        actions: require_numpy_array(
            buffers,
            "actions",
            &[n, c.max_actions, ACTION_DIM],
            "float32",
        )?
        .unbind(),
        events: require_numpy_array(
            buffers,
            "events",
            &[n, c.max_events, c.event_dim()],
            "float32",
        )?
        .unbind(),
        action_focus: require_numpy_array(
            buffers,
            "action_focus",
            &[n, c.max_actions, c.max_focus_objects],
            "int32",
        )?
        .unbind(),
        agent_player_valid: require_numpy_array(buffers, "agent_player_valid", &[n, 1], "float32")?
            .unbind(),
        opponent_player_valid: require_numpy_array(
            buffers,
            "opponent_player_valid",
            &[n, 1],
            "float32",
        )?
        .unbind(),
        agent_cards_valid: require_numpy_array(
            buffers,
            "agent_cards_valid",
            &[n, c.max_cards_per_player],
            "float32",
        )?
        .unbind(),
        opponent_cards_valid: require_numpy_array(
            buffers,
            "opponent_cards_valid",
            &[n, c.max_cards_per_player],
            "float32",
        )?
        .unbind(),
        agent_permanents_valid: require_numpy_array(
            buffers,
            "agent_permanents_valid",
            &[n, c.max_permanents_per_player],
            "float32",
        )?
        .unbind(),
        opponent_permanents_valid: require_numpy_array(
            buffers,
            "opponent_permanents_valid",
            &[n, c.max_permanents_per_player],
            "float32",
        )?
        .unbind(),
        actions_valid: require_numpy_array(
            buffers,
            "actions_valid",
            &[n, c.max_actions],
            "float32",
        )?
        .unbind(),
        events_valid: require_numpy_array(buffers, "events_valid", &[n, c.max_events], "float32")?
            .unbind(),
        step_state: if step_state {
            Some(StepStateBuffers {
                rewards: require_numpy_array(buffers, "rewards", &[n], "float64")?.unbind(),
                terminated: require_numpy_array(buffers, "terminated", &[n], "uint8")?.unbind(),
                truncated: require_numpy_array(buffers, "truncated", &[n], "uint8")?.unbind(),
            })
        } else {
            None
        },
    })
}

#[cfg(feature = "python")]
impl PyVectorEnv {
    fn run_into_buffers(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut VectorEnv,
                SendWriteBuffers,
                ObservationEncoderConfig,
            ) -> Result<Vec<InfoDict>, AgentError>
            + Send,
    ) -> PyResult<()> {
        let buffers = self
            .buffers
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;
        let config = self.config;
        let inner = &mut self.inner;

        let started = Instant::now();
        ENCODING_NANOS.set(0);
        let result = with_send_write_buffers(py, &buffers, |write_buffers| {
            py.allow_threads(|| run(inner, write_buffers, config))
                .map_err(map_agent_err)
        });
        let encoding = ENCODING_NANOS.get();
        self.timings.encoding_nanos += encoding;
        self.timings.stepping_nanos +=
            (started.elapsed().as_nanos() as u64).saturating_sub(encoding);
        self.buffers = Some(buffers);
        self.last_info = result?;
        Ok(())
    }
}

/// Batched policy-rollout pool for one flat-MC search decision.
///
/// Created via `Env.rollout_pool(...)`. All simulations of the decision run
/// simultaneously: `encode_active()` / `step_active(actions)` write each
/// active simulation's encoded observation into caller-owned numpy buffers
/// (same layout as `VectorEnv.set_buffers`, leading dim = capacity) and
/// return the active slot indices, so a policy network can pick every
/// rollout action with one batched forward pass per step.
#[cfg(feature = "python")]
#[pyclass(name = "RolloutPool")]
pub struct PyRolloutPool {
    inner: RolloutPool,
    config: ObservationEncoderConfig,
    buffers: Option<ObservationBuffers>,
}

#[cfg(feature = "python")]
impl PyRolloutPool {
    pub fn from_inner(inner: RolloutPool) -> Self {
        Self {
            inner,
            config: ObservationEncoderConfig::default(),
            buffers: None,
        }
    }

    fn run_with_buffers<R: Send>(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut RolloutPool,
                SendWriteBuffers,
                ObservationEncoderConfig,
            ) -> Result<R, AgentError>
            + Send,
    ) -> PyResult<R> {
        let buffers = self
            .buffers
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;
        let config = self.config;
        let inner = &mut self.inner;

        let result = with_send_write_buffers(py, &buffers, |write_buffers| {
            py.allow_threads(|| run(inner, write_buffers, config))
                .map_err(map_agent_err)
        });
        self.buffers = Some(buffers);
        result
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRolloutPool {
    #[getter]
    fn num_slots(&self) -> usize {
        self.inner.num_slots()
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    #[getter]
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Index of the deciding player the pool scores for.
    #[getter]
    fn hero_index(&self) -> usize {
        self.inner.hero_index()
    }

    /// Acting player index per slot (-1 for slots no longer active). The
    /// encoded observation of an active slot is from this player's
    /// perspective — value-at-leaf scoring flips V for non-hero actors.
    fn acting_players(&self) -> Vec<i64> {
        self.inner.acting_players()
    }

    /// Root action index per slot (layout is (world, action, rollout)).
    fn root_actions(&self) -> Vec<usize> {
        self.inner.root_actions()
    }

    /// Attach caller-owned numpy buffers with leading dimension `capacity`
    /// (>= num_slots). The rewards/terminated/truncated buffers are required
    /// for layout parity with VectorEnv but are never written.
    fn set_buffers(&mut self, buffers: Bound<'_, PyDict>, capacity: usize) -> PyResult<()> {
        let config = input_buffer_config(&buffers)?;
        if capacity < self.inner.num_slots() {
            return Err(PyValueError::new_err(format!(
                "buffer capacity {capacity} < num_slots {}",
                self.inner.num_slots()
            )));
        }
        let buffers = read_observation_buffers(&buffers, capacity, &config, true)?;
        self.config = config;
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Encode observations for every active slot; returns their indices.
    fn encode_active(&mut self, py: Python<'_>) -> PyResult<Vec<usize>> {
        self.run_with_buffers(py, |inner, write_buffers, config| {
            inner.encode_active_into(|slot_index, obs| {
                write_buffers
                    .encode_observation(slot_index, obs, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    /// Step every active slot with `actions[slot_index]`; returns the slot
    /// indices still active afterwards (their observations re-encoded).
    fn step_active(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<Vec<usize>> {
        self.run_with_buffers(py, move |inner, write_buffers, config| {
            inner.step_active_into(&actions, |slot_index, obs| {
                write_buffers
                    .encode_observation(slot_index, obs, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    /// Finish all still-active slots with engine-side random play to
    /// terminal (hybrid rollouts). Returns the number of slots finished.
    fn finish_random(&mut self, py: Python<'_>) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.finish_random().map_err(map_agent_err))
    }

    /// (scores, simulations, cap_hits) — mean playout score per root action.
    fn scores(&self) -> (Vec<f64>, u64, u64) {
        self.inner.scores()
    }
}

#[cfg(feature = "python")]
pub fn register_vector_env_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_observations_into, m)?)?;
    m.add_class::<PyVectorEnv>()?;
    m.add_class::<PyRolloutPool>()
}

/// Write each env's acting-player observation into row `i` of buffers from
/// `ObservationEncoder.allocate(capacity)`; returns each row's legal-action count.
///
/// Every legal action must fit (`action_overflow="error"`). Rows at and past
/// `len(envs)` are untouched, and no step-state buffers are read. The GIL stays
/// held, so no Python code can touch the caller-owned buffers during writes.
#[cfg(feature = "python")]
#[pyfunction]
fn encode_observations_into(
    py: Python<'_>,
    envs: Vec<PyRef<'_, PyEnv>>,
    buffers: Bound<'_, PyDict>,
    capacity: usize,
) -> PyResult<Vec<usize>> {
    if envs.len() > capacity {
        return Err(PyValueError::new_err(format!(
            "{} envs exceed buffer capacity {capacity}",
            envs.len()
        )));
    }
    let config = input_buffer_config(&buffers)?;
    let buffers = read_observation_buffers(&buffers, capacity, &config, false)?;
    with_observation_field_slices(py, &buffers, |out| {
        envs.iter()
            .enumerate()
            .map(|(index, branch)| {
                let env = branch
                    .inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
                let actor = env
                    .current_agent_index()
                    .ok_or_else(|| PyValueError::new_err("env has no acting player"))?;
                let observation = env.observation_for_player(actor).map_err(map_agent_err)?;
                let row = out.encoded_row_mut(index, &config).map_err(map_agent_err)?;
                encode_into(&observation, &config, row)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                Ok(observation.action_space.actions.len())
            })
            .collect()
    })
}

#[cfg(feature = "python")]
pub(crate) fn parse_player_policies(
    values: Option<Vec<String>>,
    legacy: Option<&str>,
) -> PyResult<[crate::agent::opponent::PlayerPolicy; 2]> {
    use crate::agent::opponent::PlayerPolicy;
    if values.is_some() && legacy.is_some() {
        return Err(PyValueError::new_err(
            "use player_policies or opponent_policy, not both",
        ));
    }
    let values = if let Some(legacy) = legacy {
        if !["none", "random", "heuristic"].contains(&legacy) {
            return Err(PyValueError::new_err(format!(
                "unsupported opponent_policy: {legacy}"
            )));
        }
        Python::with_gil(|py| -> PyResult<()> {
            py.import_bound("warnings")?.call_method1("warn", ("opponent_policy is deprecated; use player_policies=('agent', '<policy>'); removal in 0.3.0", py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(), 2))?;
            Ok(())
        })?;
        vec![
            "agent".into(),
            if legacy == "none" { "agent" } else { legacy }.into(),
        ]
    } else {
        values.unwrap_or_else(|| vec!["agent".into(); 2])
    };
    if values.len() != 2 {
        return Err(PyValueError::new_err(
            "player_policies requires exactly two seats",
        ));
    }
    let parse = |value: &str| match value {
        "agent" => Ok(PlayerPolicy::Agent),
        "random" => Ok(PlayerPolicy::Random),
        "heuristic" => Ok(PlayerPolicy::Heuristic),
        _ => Err(PyValueError::new_err(format!(
            "unsupported player policy: {value}"
        ))),
    };
    Ok([parse(&values[0])?, parse(&values[1])?])
}

#[cfg(feature = "python")]
fn mutable_slice_from_buffer<'py, T: Element>(
    py: Python<'py>,
    buffer: &'py PyBuffer<T>,
    key: &'static str,
) -> PyResult<&'py mut [T]> {
    let cells = buffer.as_mut_slice(py).ok_or_else(|| {
        PyValueError::new_err(format!("buffer '{key}' must be writable and C-contiguous"))
    })?;
    let ptr = cells.as_ptr() as *mut T;
    let len = cells.len();
    // SAFETY: `as_mut_slice` guarantees writable C-contiguous storage for `T`.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

#[cfg(feature = "python")]
fn typed_numpy_buffer<'py, T: Element>(
    py: Python<'py>,
    array: &'py Py<PyAny>,
    key: &'static str,
) -> PyResult<PyBuffer<T>> {
    PyBuffer::<T>::get_bound(array.bind(py))
        .map_err(|err| PyTypeError::new_err(format!("invalid buffer '{key}': {err}")))
}

#[cfg(feature = "python")]
fn send_slice_from_buffer<'py, T: Element>(
    py: Python<'py>,
    buffer: &'py PyBuffer<T>,
    key: &'static str,
) -> PyResult<SendSlice<T>> {
    let data = mutable_slice_from_buffer(py, buffer, key)?;
    Ok(SendSlice::from_mut_slice(data, key))
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct SendSlice<T> {
    ptr: *mut T,
    len: usize,
    field: &'static str,
}

#[cfg(feature = "python")]
// SAFETY: SendSlice wraps a raw pointer into a Python-owned numpy buffer.
// The GIL is released before use (py.allow_threads), and each env index
// writes to disjoint rows, so no aliasing occurs.
unsafe impl<T> Send for SendSlice<T> {}
unsafe impl<T> Sync for SendSlice<T> {}

#[cfg(feature = "python")]
impl<T> SendSlice<T> {
    fn from_mut_slice(data: &mut [T], field: &'static str) -> Self {
        Self {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            field,
        }
    }

    fn row_ptr(&self, row_index: usize, row_len: usize) -> Result<*mut T, AgentError> {
        let start = row_index
            .checked_mul(row_len)
            .ok_or_else(|| AgentError(format!("overflow while indexing '{}'", self.field)))?;
        let end = start
            .checked_add(row_len)
            .ok_or_else(|| AgentError(format!("overflow while indexing '{}'", self.field)))?;

        if end > self.len {
            return Err(AgentError(format!(
                "row {row_index} for '{}' out of bounds (row_len={row_len}, total={})",
                self.field, self.len
            )));
        }

        // SAFETY: bounds are checked above.
        Ok(unsafe { self.ptr.add(start) })
    }

    fn element_ptr(&self, index: usize) -> Result<*mut T, AgentError> {
        if index >= self.len {
            return Err(AgentError(format!(
                "index {index} for '{}' out of bounds (total={})",
                self.field, self.len
            )));
        }
        // SAFETY: bounds are checked above.
        Ok(unsafe { self.ptr.add(index) })
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct SendObservationFieldSlices {
    relations: Option<SendSlice<f32>>,
    effects: Option<SendSlice<f32>>,
    effect_details: Option<SendSlice<f32>>,
    effect_members: Option<SendSlice<f32>>,
    name_candidates: Option<SendSlice<f32>>,
    facts: Option<SendSlice<f32>>,
    semantic_cards: SendSlice<f32>,
    known_hand: SendSlice<f32>,
    decklists: Option<SendSlice<f32>>,
    player_turn_state: Option<SendSlice<f32>>,
    card_turn_state: Option<SendSlice<f32>>,
    agent_player: SendSlice<f32>,
    opponent_player: SendSlice<f32>,
    agent_cards: SendSlice<f32>,
    opponent_cards: SendSlice<f32>,
    agent_permanents: SendSlice<f32>,
    opponent_permanents: SendSlice<f32>,
    actions: SendSlice<f32>,
    events: SendSlice<f32>,
    action_focus: SendSlice<i32>,
    agent_player_valid: SendSlice<f32>,
    opponent_player_valid: SendSlice<f32>,
    agent_cards_valid: SendSlice<f32>,
    opponent_cards_valid: SendSlice<f32>,
    agent_permanents_valid: SendSlice<f32>,
    opponent_permanents_valid: SendSlice<f32>,
    actions_valid: SendSlice<f32>,
    events_valid: SendSlice<f32>,
}

#[cfg(feature = "python")]
impl SendObservationFieldSlices {
    fn encoded_row_mut(
        &self,
        env_index: usize,
        config: &ObservationEncoderConfig,
    ) -> Result<EncodedObservationMut<'_>, AgentError> {
        let semantic_cards_len = 2 * config.max_cards_per_player;
        let semantic_cards_ptr = self.semantic_cards.row_ptr(env_index, semantic_cards_len)?;
        let known_hand_len = 4 * config.max_cards_per_player;
        let known_hand_ptr = self.known_hand.row_ptr(env_index, known_hand_len)?;
        let cards_len = config.cards_len();
        let permanents_len = config.permanents_len();
        let actions_len = config.actions_len();
        let action_focus_len = config.action_focus_len();

        let agent_player_ptr = self
            .agent_player
            .row_ptr(env_index, crate::agent::observation_encoder::PLAYER_DIM)?;
        let opponent_player_ptr = self
            .opponent_player
            .row_ptr(env_index, crate::agent::observation_encoder::PLAYER_DIM)?;
        let agent_cards_ptr = self.agent_cards.row_ptr(env_index, cards_len)?;
        let opponent_cards_ptr = self.opponent_cards.row_ptr(env_index, cards_len)?;
        let agent_permanents_ptr = self.agent_permanents.row_ptr(env_index, permanents_len)?;
        let opponent_permanents_ptr = self
            .opponent_permanents
            .row_ptr(env_index, permanents_len)?;
        let actions_ptr = self.actions.row_ptr(env_index, actions_len)?;
        let action_focus_ptr = self.action_focus.row_ptr(env_index, action_focus_len)?;
        let agent_player_valid_ptr = self.agent_player_valid.row_ptr(env_index, 1)?;
        let opponent_player_valid_ptr = self.opponent_player_valid.row_ptr(env_index, 1)?;
        let agent_cards_valid_ptr = self
            .agent_cards_valid
            .row_ptr(env_index, config.max_cards_per_player)?;
        let opponent_cards_valid_ptr = self
            .opponent_cards_valid
            .row_ptr(env_index, config.max_cards_per_player)?;
        let agent_permanents_valid_ptr = self
            .agent_permanents_valid
            .row_ptr(env_index, config.max_permanents_per_player)?;
        let opponent_permanents_valid_ptr = self
            .opponent_permanents_valid
            .row_ptr(env_index, config.max_permanents_per_player)?;
        let actions_valid_ptr = self.actions_valid.row_ptr(env_index, config.max_actions)?;
        let events_ptr = self.events.row_ptr(env_index, config.events_len())?;
        let events_valid_ptr = self.events_valid.row_ptr(env_index, config.max_events)?;

        Ok(EncodedObservationMut {
            relations: if let Some(buffer) = self.relations {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.relations_len())?,
                        config.relations_len(),
                    )
                }
            } else {
                &mut []
            },
            effects: if let Some(buffer) = self.effects {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effects_len())?,
                        config.effects_len(),
                    )
                }
            } else {
                &mut []
            },
            effect_details: if let Some(buffer) = self.effect_details {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effect_details_len())?,
                        config.effect_details_len(),
                    )
                }
            } else {
                &mut []
            },
            name_candidates: if let Some(buffer) = self.name_candidates {
                // SAFETY: registered buffers partition environments into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.name_candidates_len())?,
                        config.name_candidates_len(),
                    )
                }
            } else {
                &mut []
            },
            effect_members: if let Some(buffer) = self.effect_members {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effect_members_len())?,
                        config.effect_members_len(),
                    )
                }
            } else {
                &mut []
            },
            facts: if let Some(buffer) = self.facts {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.facts_len())?,
                        config.facts_len(),
                    )
                }
            } else {
                &mut []
            },
            // SAFETY: each call uses disjoint per-env rows.
            known_hand: unsafe { slice::from_raw_parts_mut(known_hand_ptr, known_hand_len) },
            decklists: if let Some(buffer) = self.decklists {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.decklists_len())?,
                        config.decklists_len(),
                    )
                }
            } else {
                &mut []
            },
            player_turn_state: if let Some(buffer) = self.player_turn_state {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.player_turn_state_len())?,
                        config.player_turn_state_len(),
                    )
                }
            } else {
                &mut []
            },
            card_turn_state: if let Some(buffer) = self.card_turn_state {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.card_turn_state_len())?,
                        config.card_turn_state_len(),
                    )
                }
            } else {
                &mut []
            },
            // SAFETY: each call uses disjoint per-env rows.
            semantic_cards: unsafe {
                slice::from_raw_parts_mut(semantic_cards_ptr, semantic_cards_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_player: unsafe {
                slice::from_raw_parts_mut(
                    agent_player_ptr,
                    crate::agent::observation_encoder::PLAYER_DIM,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_player: unsafe {
                slice::from_raw_parts_mut(
                    opponent_player_ptr,
                    crate::agent::observation_encoder::PLAYER_DIM,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_cards: unsafe { slice::from_raw_parts_mut(agent_cards_ptr, cards_len) },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_cards: unsafe { slice::from_raw_parts_mut(opponent_cards_ptr, cards_len) },
            // SAFETY: each call uses disjoint per-env rows.
            agent_permanents: unsafe {
                slice::from_raw_parts_mut(agent_permanents_ptr, permanents_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_permanents: unsafe {
                slice::from_raw_parts_mut(opponent_permanents_ptr, permanents_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            actions: unsafe { slice::from_raw_parts_mut(actions_ptr, actions_len) },
            // SAFETY: each call uses disjoint per-env rows.
            events: unsafe { slice::from_raw_parts_mut(events_ptr, config.events_len()) },
            // SAFETY: each call uses disjoint per-env rows.
            action_focus: unsafe { slice::from_raw_parts_mut(action_focus_ptr, action_focus_len) },
            // SAFETY: each call uses disjoint per-env rows.
            agent_player_valid: unsafe { slice::from_raw_parts_mut(agent_player_valid_ptr, 1) },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_player_valid: unsafe {
                slice::from_raw_parts_mut(opponent_player_valid_ptr, 1)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_cards_valid: unsafe {
                slice::from_raw_parts_mut(agent_cards_valid_ptr, config.max_cards_per_player)
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_cards_valid: unsafe {
                slice::from_raw_parts_mut(opponent_cards_valid_ptr, config.max_cards_per_player)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_permanents_valid: unsafe {
                slice::from_raw_parts_mut(
                    agent_permanents_valid_ptr,
                    config.max_permanents_per_player,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_permanents_valid: unsafe {
                slice::from_raw_parts_mut(
                    opponent_permanents_valid_ptr,
                    config.max_permanents_per_player,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            actions_valid: unsafe {
                slice::from_raw_parts_mut(actions_valid_ptr, config.max_actions)
            },
            // SAFETY: each call uses disjoint per-env rows.
            events_valid: unsafe { slice::from_raw_parts_mut(events_valid_ptr, config.max_events) },
        })
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct SendWriteBuffers {
    fields: SendObservationFieldSlices,
    rewards: SendSlice<f64>,
    terminated: SendSlice<u8>,
    truncated: SendSlice<u8>,
}

#[cfg(feature = "python")]
impl SendWriteBuffers {
    fn encode_observation(
        &self,
        env_index: usize,
        observation: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationWriteError> {
        let out = self.fields.encoded_row_mut(env_index, config)?;
        encode_into(observation, config, out).map_err(ObservationWriteError::from)
    }

    fn write_step_state(
        &self,
        env_index: usize,
        reward: f64,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), AgentError> {
        let reward_ptr = self.rewards.element_ptr(env_index)?;
        let terminated_ptr = self.terminated.element_ptr(env_index)?;
        let truncated_ptr = self.truncated.element_ptr(env_index)?;

        // SAFETY: each env index is written at most once per step/reset call.
        unsafe {
            *reward_ptr = reward;
            *terminated_ptr = terminated as u8;
            *truncated_ptr = truncated as u8;
        }
        Ok(())
    }

    fn write_encoded_row(
        &self,
        env_index: usize,
        observation: &Observation,
        reward: f64,
        terminated: bool,
        truncated: bool,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationWriteError> {
        let started = Instant::now();
        let encoded = self.encode_observation(env_index, observation, config);
        ENCODING_NANOS.set(ENCODING_NANOS.get() + started.elapsed().as_nanos() as u64);
        encoded?;
        self.write_step_state(env_index, reward, terminated, truncated)
            .map_err(ObservationWriteError::from)
    }
}

#[cfg(feature = "python")]
fn with_send_write_buffers<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(SendWriteBuffers) -> PyResult<R>,
) -> PyResult<R> {
    let step_state = buffers
        .step_state
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("step-state buffers are required"))?;
    let rewards_buffer = typed_numpy_buffer::<f64>(py, &step_state.rewards, "rewards")?;
    let terminated_buffer = typed_numpy_buffer::<u8>(py, &step_state.terminated, "terminated")?;
    let truncated_buffer = typed_numpy_buffer::<u8>(py, &step_state.truncated, "truncated")?;
    with_observation_field_slices(py, buffers, |fields| {
        f(SendWriteBuffers {
            fields,
            rewards: send_slice_from_buffer(py, &rewards_buffer, "rewards")?,
            terminated: send_slice_from_buffer(py, &terminated_buffer, "terminated")?,
            truncated: send_slice_from_buffer(py, &truncated_buffer, "truncated")?,
        })
    })
}

#[cfg(feature = "python")]
fn with_observation_field_slices<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(SendObservationFieldSlices) -> PyResult<R>,
) -> PyResult<R> {
    let relations_buffer = buffers
        .relations
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "relations"))
        .transpose()?;
    let effects_buffer = buffers
        .effects
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effects"))
        .transpose()?;
    let effect_details_buffer = buffers
        .effect_details
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effect_details"))
        .transpose()?;
    let name_candidates_buffer = buffers
        .name_candidates
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "name_candidates"))
        .transpose()?;
    let effect_members_buffer = buffers
        .effect_members
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effect_members"))
        .transpose()?;
    let facts_buffer = buffers
        .facts
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "facts"))
        .transpose()?;
    let semantic_cards_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.semantic_cards, "semantic_cards")?;
    let known_hand_buffer = typed_numpy_buffer::<f32>(py, &buffers.known_hand, "known_hand")?;
    let decklists_buffer = buffers
        .decklists
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "decklists"))
        .transpose()?;
    let player_turn_state_buffer = buffers
        .player_turn_state
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "player_turn_state"))
        .transpose()?;
    let card_turn_state_buffer = buffers
        .card_turn_state
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "card_turn_state"))
        .transpose()?;
    let agent_player_buffer = typed_numpy_buffer::<f32>(py, &buffers.agent_player, "agent_player")?;
    let opponent_player_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_player, "opponent_player")?;
    let agent_cards_buffer = typed_numpy_buffer::<f32>(py, &buffers.agent_cards, "agent_cards")?;
    let opponent_cards_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_cards, "opponent_cards")?;
    let agent_permanents_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_permanents, "agent_permanents")?;
    let opponent_permanents_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_permanents, "opponent_permanents")?;
    let actions_buffer = typed_numpy_buffer::<f32>(py, &buffers.actions, "actions")?;
    let events_buffer = typed_numpy_buffer::<f32>(py, &buffers.events, "events")?;
    let action_focus_buffer = typed_numpy_buffer::<i32>(py, &buffers.action_focus, "action_focus")?;
    let agent_player_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_player_valid, "agent_player_valid")?;
    let opponent_player_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_player_valid, "opponent_player_valid")?;
    let agent_cards_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_cards_valid, "agent_cards_valid")?;
    let opponent_cards_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_cards_valid, "opponent_cards_valid")?;
    let agent_permanents_valid_buffer = typed_numpy_buffer::<f32>(
        py,
        &buffers.agent_permanents_valid,
        "agent_permanents_valid",
    )?;
    let opponent_permanents_valid_buffer = typed_numpy_buffer::<f32>(
        py,
        &buffers.opponent_permanents_valid,
        "opponent_permanents_valid",
    )?;
    let actions_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.actions_valid, "actions_valid")?;
    let events_valid_buffer = typed_numpy_buffer::<f32>(py, &buffers.events_valid, "events_valid")?;
    let field_slices = SendObservationFieldSlices {
        relations: relations_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "relations"))
            .transpose()?,
        effects: effects_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effects"))
            .transpose()?,
        effect_details: effect_details_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effect_details"))
            .transpose()?,
        name_candidates: name_candidates_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "name_candidates"))
            .transpose()?,
        effect_members: effect_members_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effect_members"))
            .transpose()?,
        facts: facts_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "facts"))
            .transpose()?,
        known_hand: send_slice_from_buffer(py, &known_hand_buffer, "known_hand")?,
        decklists: decklists_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decklists"))
            .transpose()?,
        player_turn_state: player_turn_state_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "player_turn_state"))
            .transpose()?,
        card_turn_state: card_turn_state_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "card_turn_state"))
            .transpose()?,
        semantic_cards: send_slice_from_buffer(py, &semantic_cards_buffer, "semantic_cards")?,
        agent_player: send_slice_from_buffer(py, &agent_player_buffer, "agent_player")?,
        opponent_player: send_slice_from_buffer(py, &opponent_player_buffer, "opponent_player")?,
        agent_cards: send_slice_from_buffer(py, &agent_cards_buffer, "agent_cards")?,
        opponent_cards: send_slice_from_buffer(py, &opponent_cards_buffer, "opponent_cards")?,
        agent_permanents: send_slice_from_buffer(py, &agent_permanents_buffer, "agent_permanents")?,
        opponent_permanents: send_slice_from_buffer(
            py,
            &opponent_permanents_buffer,
            "opponent_permanents",
        )?,
        actions: send_slice_from_buffer(py, &actions_buffer, "actions")?,
        events: send_slice_from_buffer(py, &events_buffer, "events")?,
        action_focus: send_slice_from_buffer(py, &action_focus_buffer, "action_focus")?,
        agent_player_valid: send_slice_from_buffer(
            py,
            &agent_player_valid_buffer,
            "agent_player_valid",
        )?,
        opponent_player_valid: send_slice_from_buffer(
            py,
            &opponent_player_valid_buffer,
            "opponent_player_valid",
        )?,
        agent_cards_valid: send_slice_from_buffer(
            py,
            &agent_cards_valid_buffer,
            "agent_cards_valid",
        )?,
        opponent_cards_valid: send_slice_from_buffer(
            py,
            &opponent_cards_valid_buffer,
            "opponent_cards_valid",
        )?,
        agent_permanents_valid: send_slice_from_buffer(
            py,
            &agent_permanents_valid_buffer,
            "agent_permanents_valid",
        )?,
        opponent_permanents_valid: send_slice_from_buffer(
            py,
            &opponent_permanents_valid_buffer,
            "opponent_permanents_valid",
        )?,
        actions_valid: send_slice_from_buffer(py, &actions_valid_buffer, "actions_valid")?,
        events_valid: send_slice_from_buffer(py, &events_valid_buffer, "events_valid")?,
    };

    f(field_slices)
}
