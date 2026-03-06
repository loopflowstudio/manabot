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
        observation::Observation,
        observation_encoder::{
            encode_into, EncodedObservationMut, ObservationEncoderConfig, ACTION_DIM, CARD_DIM,
            PERMANENT_DIM, PLAYER_DIM,
        },
        opponent::OpponentPolicy,
        vector_env::{StepResult, VectorEnv},
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
        let buffers = self
            .buffers
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;

        let configs: Vec<PlayerConfig> =
            player_configs.into_iter().map(PlayerConfig::from).collect();
        let results = self.inner.reset_all(configs).map_err(map_agent_err)?;

        self.last_info = results.iter().map(|(_, info)| info.clone()).collect();
        write_reset_results_into_buffers(py, buffers, &self.config, &results)
    }

    fn step_into_buffers(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<()> {
        let buffers = self
            .buffers
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;

        let results = self.inner.step(&actions).map_err(map_agent_err)?;

        self.last_info = results.iter().map(|result| result.info.clone()).collect();
        write_step_results_into_buffers(py, buffers, &self.config, &results)
    }

    fn get_last_info(&self, py: Python<'_>) -> Vec<PyObject> {
        self.last_info
            .iter()
            .map(|info| info_dict_to_pydict(py, info).into_any().unbind())
            .collect()
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
fn row_mut<'a, T>(
    data: &'a mut [T],
    row_index: usize,
    row_len: usize,
    field: &'static str,
) -> PyResult<&'a mut [T]> {
    let start = row_index
        .checked_mul(row_len)
        .ok_or_else(|| PyValueError::new_err(format!("overflow while indexing '{field}'")))?;
    let end = start
        .checked_add(row_len)
        .ok_or_else(|| PyValueError::new_err(format!("overflow while indexing '{field}'")))?;
    let total_len = data.len();
    data.get_mut(start..end).ok_or_else(|| {
        PyValueError::new_err(format!(
            "row {row_index} for '{field}' out of bounds (row_len={row_len}, total={total_len})"
        ))
    })
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
struct ObservationFieldSlices<'a> {
    agent_player: &'a mut [f32],
    opponent_player: &'a mut [f32],
    agent_cards: &'a mut [f32],
    opponent_cards: &'a mut [f32],
    agent_permanents: &'a mut [f32],
    opponent_permanents: &'a mut [f32],
    actions: &'a mut [f32],
    action_focus: &'a mut [i32],
    agent_player_valid: &'a mut [f32],
    opponent_player_valid: &'a mut [f32],
    agent_cards_valid: &'a mut [f32],
    opponent_cards_valid: &'a mut [f32],
    agent_permanents_valid: &'a mut [f32],
    opponent_permanents_valid: &'a mut [f32],
    actions_valid: &'a mut [f32],
}

#[cfg(feature = "python")]
impl ObservationFieldSlices<'_> {
    fn encoded_row_mut(
        &mut self,
        env_index: usize,
        config: &ObservationEncoderConfig,
    ) -> PyResult<EncodedObservationMut<'_>> {
        Ok(EncodedObservationMut {
            agent_player: row_mut(self.agent_player, env_index, PLAYER_DIM, "agent_player")?,
            opponent_player: row_mut(
                self.opponent_player,
                env_index,
                PLAYER_DIM,
                "opponent_player",
            )?,
            agent_cards: row_mut(
                self.agent_cards,
                env_index,
                config.cards_len(),
                "agent_cards",
            )?,
            opponent_cards: row_mut(
                self.opponent_cards,
                env_index,
                config.cards_len(),
                "opponent_cards",
            )?,
            agent_permanents: row_mut(
                self.agent_permanents,
                env_index,
                config.permanents_len(),
                "agent_permanents",
            )?,
            opponent_permanents: row_mut(
                self.opponent_permanents,
                env_index,
                config.permanents_len(),
                "opponent_permanents",
            )?,
            actions: row_mut(self.actions, env_index, config.actions_len(), "actions")?,
            action_focus: row_mut(
                self.action_focus,
                env_index,
                config.action_focus_len(),
                "action_focus",
            )?,
            agent_player_valid: row_mut(
                self.agent_player_valid,
                env_index,
                1,
                "agent_player_valid",
            )?,
            opponent_player_valid: row_mut(
                self.opponent_player_valid,
                env_index,
                1,
                "opponent_player_valid",
            )?,
            agent_cards_valid: row_mut(
                self.agent_cards_valid,
                env_index,
                config.max_cards_per_player,
                "agent_cards_valid",
            )?,
            opponent_cards_valid: row_mut(
                self.opponent_cards_valid,
                env_index,
                config.max_cards_per_player,
                "opponent_cards_valid",
            )?,
            agent_permanents_valid: row_mut(
                self.agent_permanents_valid,
                env_index,
                config.max_permanents_per_player,
                "agent_permanents_valid",
            )?,
            opponent_permanents_valid: row_mut(
                self.opponent_permanents_valid,
                env_index,
                config.max_permanents_per_player,
                "opponent_permanents_valid",
            )?,
            actions_valid: row_mut(
                self.actions_valid,
                env_index,
                config.max_actions,
                "actions_valid",
            )?,
        })
    }
}

#[cfg(feature = "python")]
struct WriteBuffers<'a> {
    fields: ObservationFieldSlices<'a>,
    rewards: &'a mut [f64],
    terminated: &'a mut [u8],
    truncated: &'a mut [u8],
}

#[cfg(feature = "python")]
impl WriteBuffers<'_> {
    fn encode_observation(
        &mut self,
        env_index: usize,
        observation: &Observation,
        config: &ObservationEncoderConfig,
    ) -> PyResult<()> {
        let out = self.fields.encoded_row_mut(env_index, config)?;
        encode_into(observation, config, out).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn write_step_state(
        &mut self,
        env_index: usize,
        reward: f64,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        *self
            .rewards
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("reward buffer out of bounds"))? = reward;
        *self
            .terminated
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("terminated buffer out of bounds"))? =
            terminated as u8;
        *self
            .truncated
            .get_mut(env_index)
            .ok_or_else(|| PyValueError::new_err("truncated buffer out of bounds"))? =
            truncated as u8;
        Ok(())
    }
}

#[cfg(feature = "python")]
fn with_write_buffers<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(WriteBuffers<'_>) -> PyResult<R>,
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

    let field_slices = ObservationFieldSlices {
        agent_player: mutable_slice_from_buffer(py, &agent_player_buffer, "agent_player")?,
        opponent_player: mutable_slice_from_buffer(py, &opponent_player_buffer, "opponent_player")?,
        agent_cards: mutable_slice_from_buffer(py, &agent_cards_buffer, "agent_cards")?,
        opponent_cards: mutable_slice_from_buffer(py, &opponent_cards_buffer, "opponent_cards")?,
        agent_permanents: mutable_slice_from_buffer(
            py,
            &agent_permanents_buffer,
            "agent_permanents",
        )?,
        opponent_permanents: mutable_slice_from_buffer(
            py,
            &opponent_permanents_buffer,
            "opponent_permanents",
        )?,
        actions: mutable_slice_from_buffer(py, &actions_buffer, "actions")?,
        action_focus: mutable_slice_from_buffer(py, &action_focus_buffer, "action_focus")?,
        agent_player_valid: mutable_slice_from_buffer(
            py,
            &agent_player_valid_buffer,
            "agent_player_valid",
        )?,
        opponent_player_valid: mutable_slice_from_buffer(
            py,
            &opponent_player_valid_buffer,
            "opponent_player_valid",
        )?,
        agent_cards_valid: mutable_slice_from_buffer(
            py,
            &agent_cards_valid_buffer,
            "agent_cards_valid",
        )?,
        opponent_cards_valid: mutable_slice_from_buffer(
            py,
            &opponent_cards_valid_buffer,
            "opponent_cards_valid",
        )?,
        agent_permanents_valid: mutable_slice_from_buffer(
            py,
            &agent_permanents_valid_buffer,
            "agent_permanents_valid",
        )?,
        opponent_permanents_valid: mutable_slice_from_buffer(
            py,
            &opponent_permanents_valid_buffer,
            "opponent_permanents_valid",
        )?,
        actions_valid: mutable_slice_from_buffer(py, &actions_valid_buffer, "actions_valid")?,
    };
    let rewards = mutable_slice_from_buffer(py, &rewards_buffer, "rewards")?;
    let terminated_buffer = typed_numpy_buffer::<u8>(py, &buffers.terminated, "terminated")?;
    let truncated_buffer = typed_numpy_buffer::<u8>(py, &buffers.truncated, "truncated")?;
    let terminated = mutable_slice_from_buffer(py, &terminated_buffer, "terminated")?;
    let truncated = mutable_slice_from_buffer(py, &truncated_buffer, "truncated")?;

    f(WriteBuffers {
        fields: field_slices,
        rewards,
        terminated,
        truncated,
    })
}

#[cfg(feature = "python")]
fn write_step_results_into_buffers(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    config: &ObservationEncoderConfig,
    results: &[StepResult],
) -> PyResult<()> {
    with_write_buffers(py, buffers, |mut write_buffers| {
        for (env_index, result) in results.iter().enumerate() {
            write_buffers.encode_observation(env_index, &result.obs, config)?;
            write_buffers.write_step_state(
                env_index,
                result.reward,
                result.terminated,
                result.truncated,
            )?;
        }
        Ok(())
    })
}

#[cfg(feature = "python")]
fn write_reset_results_into_buffers(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    config: &ObservationEncoderConfig,
    results: &[(Observation, InfoDict)],
) -> PyResult<()> {
    with_write_buffers(py, buffers, |mut write_buffers| {
        for (env_index, (obs, _)) in results.iter().enumerate() {
            write_buffers.encode_observation(env_index, obs, config)?;
            write_buffers.write_step_state(env_index, 0.0, false, false)?;
        }
        Ok(())
    })
}
