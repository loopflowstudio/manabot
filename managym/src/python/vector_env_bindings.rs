#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::slice;

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
        observation::Observation,
        observation_encoder::{
            encode_into, EncodedObservationMut, ObservationEncoderConfig, ACTION_DIM, CARD_DIM,
            PERMANENT_DIM, PLAYER_DIM,
        },
        opponent::OpponentPolicy,
        vector_env::VectorEnv,
    },
    infra::profiler::InfoDict,
    python::{
        bindings::{map_agent_err, PyObservation, PyPlayerConfig},
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
    agent_player: Py<PyAny>,
    opponent_player: Py<PyAny>,
    agent_cards: Py<PyAny>,
    opponent_cards: Py<PyAny>,
    agent_permanents: Py<PyAny>,
    opponent_permanents: Py<PyAny>,
    actions: Py<PyAny>,
    action_focus: Py<PyAny>,
    agent_player_valid: Py<PyAny>,
    opponent_player_valid: Py<PyAny>,
    agent_cards_valid: Py<PyAny>,
    opponent_cards_valid: Py<PyAny>,
    agent_permanents_valid: Py<PyAny>,
    opponent_permanents_valid: Py<PyAny>,
    actions_valid: Py<PyAny>,
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
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVectorEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=0, skip_trivial=true, opponent_policy="none"))]
    fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        opponent_policy: &str,
    ) -> PyResult<Self> {
        let policy = parse_opponent_policy(opponent_policy)?;

        Ok(Self {
            inner: VectorEnv::new(num_envs, seed, skip_trivial, policy),
            config: ObservationEncoderConfig::default(),
            num_envs,
            buffers: None,
            last_info: Vec::new(),
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

    fn set_buffers(&mut self, buffers: Bound<'_, PyDict>) -> PyResult<()> {
        let n = self.num_envs;
        let c = self.config;

        self.buffers = Some(ObservationBuffers {
            agent_player: require_numpy_array(
                &buffers,
                "agent_player",
                &[n, 1, PLAYER_DIM],
                "float32",
            )?
            .unbind(),
            opponent_player: require_numpy_array(
                &buffers,
                "opponent_player",
                &[n, 1, PLAYER_DIM],
                "float32",
            )?
            .unbind(),
            agent_cards: require_numpy_array(
                &buffers,
                "agent_cards",
                &[n, c.max_cards_per_player, CARD_DIM],
                "float32",
            )?
            .unbind(),
            opponent_cards: require_numpy_array(
                &buffers,
                "opponent_cards",
                &[n, c.max_cards_per_player, CARD_DIM],
                "float32",
            )?
            .unbind(),
            agent_permanents: require_numpy_array(
                &buffers,
                "agent_permanents",
                &[n, c.max_permanents_per_player, PERMANENT_DIM],
                "float32",
            )?
            .unbind(),
            opponent_permanents: require_numpy_array(
                &buffers,
                "opponent_permanents",
                &[n, c.max_permanents_per_player, PERMANENT_DIM],
                "float32",
            )?
            .unbind(),
            actions: require_numpy_array(
                &buffers,
                "actions",
                &[n, c.max_actions, ACTION_DIM],
                "float32",
            )?
            .unbind(),
            action_focus: require_numpy_array(
                &buffers,
                "action_focus",
                &[n, c.max_actions, c.max_focus_objects],
                "int32",
            )?
            .unbind(),
            agent_player_valid: require_numpy_array(
                &buffers,
                "agent_player_valid",
                &[n, 1],
                "float32",
            )?
            .unbind(),
            opponent_player_valid: require_numpy_array(
                &buffers,
                "opponent_player_valid",
                &[n, 1],
                "float32",
            )?
            .unbind(),
            agent_cards_valid: require_numpy_array(
                &buffers,
                "agent_cards_valid",
                &[n, c.max_cards_per_player],
                "float32",
            )?
            .unbind(),
            opponent_cards_valid: require_numpy_array(
                &buffers,
                "opponent_cards_valid",
                &[n, c.max_cards_per_player],
                "float32",
            )?
            .unbind(),
            agent_permanents_valid: require_numpy_array(
                &buffers,
                "agent_permanents_valid",
                &[n, c.max_permanents_per_player],
                "float32",
            )?
            .unbind(),
            opponent_permanents_valid: require_numpy_array(
                &buffers,
                "opponent_permanents_valid",
                &[n, c.max_permanents_per_player],
                "float32",
            )?
            .unbind(),
            actions_valid: require_numpy_array(
                &buffers,
                "actions_valid",
                &[n, c.max_actions],
                "float32",
            )?
            .unbind(),
            rewards: require_numpy_array(&buffers, "rewards", &[n], "float64")?.unbind(),
            terminated: require_numpy_array(&buffers, "terminated", &[n], "uint8")?.unbind(),
            truncated: require_numpy_array(&buffers, "truncated", &[n], "uint8")?.unbind(),
        });
        Ok(())
    }

    fn reset_all_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<()> {
        let configs: Vec<PlayerConfig> =
            player_configs.into_iter().map(PlayerConfig::from).collect();
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.par_reset_all_into(
                configs,
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                },
            )
        })
    }

    fn step_into_buffers(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.par_step_into(
                &actions,
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                },
            )
        })
    }

    fn get_last_info(&self, py: Python<'_>) -> Vec<PyObject> {
        self.last_info
            .iter()
            .map(|info| info_dict_to_pydict(py, info).into_any().unbind())
            .collect()
    }
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

        let result = with_send_write_buffers(py, &buffers, |write_buffers| {
            py.allow_threads(|| run(inner, write_buffers, config))
                .map_err(map_agent_err)
        });
        self.buffers = Some(buffers);
        self.last_info = result?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub fn register_vector_env_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorEnv>()
}

#[cfg(feature = "python")]
fn parse_opponent_policy(value: &str) -> PyResult<OpponentPolicy> {
    match value {
        "none" => Ok(OpponentPolicy::None),
        "passive" => Ok(OpponentPolicy::Passive),
        "random" => Ok(OpponentPolicy::Random),
        _ => Err(PyValueError::new_err(format!(
            "unsupported opponent_policy: {value}"
        ))),
    }
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
unsafe impl<T> Send for SendSlice<T> {}
#[cfg(feature = "python")]
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
    agent_player: SendSlice<f32>,
    opponent_player: SendSlice<f32>,
    agent_cards: SendSlice<f32>,
    opponent_cards: SendSlice<f32>,
    agent_permanents: SendSlice<f32>,
    opponent_permanents: SendSlice<f32>,
    actions: SendSlice<f32>,
    action_focus: SendSlice<i32>,
    agent_player_valid: SendSlice<f32>,
    opponent_player_valid: SendSlice<f32>,
    agent_cards_valid: SendSlice<f32>,
    opponent_cards_valid: SendSlice<f32>,
    agent_permanents_valid: SendSlice<f32>,
    opponent_permanents_valid: SendSlice<f32>,
    actions_valid: SendSlice<f32>,
}

#[cfg(feature = "python")]
impl SendObservationFieldSlices {
    fn encoded_row_mut(
        &self,
        env_index: usize,
        config: &ObservationEncoderConfig,
    ) -> Result<EncodedObservationMut<'_>, AgentError> {
        let cards_len = config.cards_len();
        let permanents_len = config.permanents_len();
        let actions_len = config.actions_len();
        let action_focus_len = config.action_focus_len();

        let agent_player_ptr = self.agent_player.row_ptr(env_index, PLAYER_DIM)?;
        let opponent_player_ptr = self.opponent_player.row_ptr(env_index, PLAYER_DIM)?;
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

        Ok(EncodedObservationMut {
            // SAFETY: each call uses disjoint per-env rows.
            agent_player: unsafe { slice::from_raw_parts_mut(agent_player_ptr, PLAYER_DIM) },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_player: unsafe { slice::from_raw_parts_mut(opponent_player_ptr, PLAYER_DIM) },
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
    ) -> Result<(), AgentError> {
        let out = self.fields.encoded_row_mut(env_index, config)?;
        encode_into(observation, config, out).map_err(|err| AgentError(err.to_string()))
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
    ) -> Result<(), AgentError> {
        self.encode_observation(env_index, observation, config)?;
        self.write_step_state(env_index, reward, terminated, truncated)
    }
}

#[cfg(feature = "python")]
fn with_send_write_buffers<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(SendWriteBuffers) -> PyResult<R>,
) -> PyResult<R> {
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
    let rewards_buffer = typed_numpy_buffer::<f64>(py, &buffers.rewards, "rewards")?;
    let terminated_buffer = typed_numpy_buffer::<u8>(py, &buffers.terminated, "terminated")?;
    let truncated_buffer = typed_numpy_buffer::<u8>(py, &buffers.truncated, "truncated")?;

    let field_slices = SendObservationFieldSlices {
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
    };

    f(SendWriteBuffers {
        fields: field_slices,
        rewards: send_slice_from_buffer(py, &rewards_buffer, "rewards")?,
        terminated: send_slice_from_buffer(py, &terminated_buffer, "terminated")?,
        truncated: send_slice_from_buffer(py, &truncated_buffer, "truncated")?,
    })
}
