#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::sync::Mutex;

#[cfg(feature = "python")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

#[cfg(feature = "python")]
use crate::{
    agent::{opponent::OpponentPolicy, vector_env::VectorEnv},
    python::{
        bindings::{map_agent_err, PyObservation, PyPlayerConfig},
        convert::info_dict_to_pydict,
    },
    state::player::PlayerConfig,
};

#[cfg(feature = "python")]
type PyResetResult = (PyObservation, PyObject);
#[cfg(feature = "python")]
type PyStepResult = (PyObservation, f64, bool, bool, PyObject);

#[cfg(feature = "python")]
#[pyclass(name = "VectorEnv")]
pub struct PyVectorEnv {
    inner: Mutex<VectorEnv>,
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
            inner: Mutex::new(VectorEnv::new(num_envs, seed, skip_trivial, policy)),
        })
    }

    fn reset_all(
        &self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<Vec<PyResetResult>> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vector env lock poisoned"))?;
        let configs = player_configs.into_iter().map(PlayerConfig::from).collect();
        let results = env.reset_all(configs).map_err(map_agent_err)?;

        Ok(results
            .into_iter()
            .map(|(obs, info)| {
                let py_dict = info_dict_to_pydict(py, &info);
                (PyObservation::from(obs), py_dict.into_any().unbind())
            })
            .collect())
    }

    fn step(&self, py: Python<'_>, actions: Vec<i64>) -> PyResult<Vec<PyStepResult>> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vector env lock poisoned"))?;
        let results = env.step(&actions).map_err(map_agent_err)?;

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
