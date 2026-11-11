//! Scalar adapter over the same viewer-observation selector used by VectorEnv.
use crate::{
    agent::{heuristic::Heuristic, observation::Observation},
    python::bindings::PyObservation,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "Heuristic")]
pub struct PyHeuristic {
    inner: Heuristic,
    max_actions: Option<usize>,
}
#[pymethods]
impl PyHeuristic {
    #[new]
    #[pyo3(signature = (max_actions=None))]
    fn new(max_actions: Option<usize>) -> PyResult<Self> {
        if max_actions == Some(0) {
            return Err(PyValueError::new_err(
                "heuristic max_actions must be positive",
            ));
        }
        Ok(Self {
            inner: Heuristic::default(),
            max_actions,
        })
    }
    fn select_action(&mut self, observation: &PyObservation) -> PyResult<usize> {
        let raw: Observation = observation.clone().into();
        self.inner
            .select_action(&raw, self.max_actions)
            .map_err(|e| PyValueError::new_err(e.0))
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}
