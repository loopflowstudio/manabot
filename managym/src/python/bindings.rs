// PyO3's #[pymethods] macro triggers false-positive `useless_conversion`
// warnings in generated wrappers under strict clippy settings.
#![allow(clippy::useless_conversion)]

#[cfg(feature = "python")]
use std::{collections::HashMap, sync::Mutex};

#[cfg(feature = "python")]
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[cfg(feature = "python")]
use crate::{agent::env::Env, python::convert::info_dict_to_pydict, state::player::PlayerConfig};

#[cfg(feature = "python")]
#[pyclass(name = "PlayerConfig")]
#[derive(Clone)]
pub struct PyPlayerConfig {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub decklist: HashMap<String, usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPlayerConfig {
    #[new]
    fn new(name: String, decklist: HashMap<String, usize>) -> Self {
        Self { name, decklist }
    }
}

#[cfg(feature = "python")]
impl From<PyPlayerConfig> for PlayerConfig {
    fn from(value: PyPlayerConfig) -> Self {
        PlayerConfig {
            name: value.name,
            decklist: value.decklist.into_iter().collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Env")]
pub struct PyEnv {
    inner: Mutex<Env>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (seed=0, skip_trivial=true, enable_profiler=false, enable_behavior_tracking=false))]
    fn new(
        seed: u64,
        skip_trivial: bool,
        enable_profiler: bool,
        enable_behavior_tracking: bool,
    ) -> Self {
        Self {
            inner: Mutex::new(Env::new(
                seed,
                skip_trivial,
                enable_profiler,
                enable_behavior_tracking,
            )),
        }
    }

    fn reset(
        &self,
        py: Python<'_>,
        player_configs: Vec<PyPlayerConfig>,
    ) -> PyResult<(String, PyObject)> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let configs = player_configs.into_iter().map(PlayerConfig::from).collect();
        let (obs, info) = env
            .reset(configs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let py_dict = info_dict_to_pydict(py, &info);
        Ok((obs.to_json(), py_dict.into_any().unbind()))
    }

    fn step(&self, py: Python<'_>, action: usize) -> PyResult<(String, f64, bool, bool, PyObject)> {
        let mut env = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;

        let (obs, reward, terminated, truncated, info) = env
            .step(action)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let py_dict = info_dict_to_pydict(py, &info);
        Ok((
            obs.to_json(),
            reward,
            terminated,
            truncated,
            py_dict.into_any().unbind(),
        ))
    }
}

#[cfg(feature = "python")]
#[pymodule]
pub fn _managym(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlayerConfig>()?;
    m.add_class::<PyEnv>()?;
    Ok(())
}
