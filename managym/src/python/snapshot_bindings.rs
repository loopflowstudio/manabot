//! Python ownership boundary for immutable positions and independent live Envs.
#![allow(clippy::useless_conversion)]

use super::bindings::{map_agent_err, ActionSpaceEnum, PhaseEnum, PyEnv};
use crate::agent::{BranchMode, GameSnapshot, RestoreOptions, SnapshotMetadata};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "GameSnapshot", frozen)]
pub(super) struct PyGameSnapshot(pub(super) Arc<GameSnapshot>);

#[pymethods]
impl PyGameSnapshot {
    #[getter]
    fn state_digest(&self, py: Python<'_>) -> String {
        py.allow_threads(|| self.0.state_digest().to_owned())
    }

    #[getter]
    fn acting_player(&self) -> Option<usize> {
        self.0.acting_player()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed()
    }

    #[pyo3(signature = (options=None))]
    fn restore(&self, py: Python<'_>, options: Option<&PyRestoreOptions>) -> PyResult<PyEnv> {
        let options = options.map(|o| o.0).unwrap_or_default();
        py.allow_threads(|| self.0.restore(options))
            .map(PyEnv::from_live)
            .map_err(map_agent_err)
    }

    fn metadata(&self) -> PySnapshotMetadata {
        PySnapshotMetadata(self.0.metadata())
    }

    fn to_replay_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.0.to_replay_json())
            .map_err(map_agent_err)
    }

    #[staticmethod]
    fn _check_replay(py: Python<'_>, raw: &str) -> PyResult<()> {
        py.allow_threads(|| GameSnapshot::check_replay(raw))
            .map_err(map_agent_err)
    }

    #[staticmethod]
    fn from_replay_json(py: Python<'_>, raw: &str) -> PyResult<Self> {
        py.allow_threads(|| GameSnapshot::from_replay_json(raw))
            .map(|snapshot| Self(Arc::new(snapshot)))
            .map_err(map_agent_err)
    }
}

#[pyclass(name = "RestoreOptions", frozen)]
pub(super) struct PyRestoreOptions(pub(super) RestoreOptions);
#[pymethods]
impl PyRestoreOptions {
    #[getter]
    fn forced_first_action(&self) -> Option<usize> {
        self.0.forced_first_action
    }

    #[new]
    fn new() -> Self {
        Self(RestoreOptions::default())
    }
    #[staticmethod]
    #[pyo3(signature = (*, reseed=None, enable_decision_memory=None, forced_first_action=None))]
    fn exact(
        reseed: Option<u64>,
        enable_decision_memory: Option<bool>,
        forced_first_action: Option<usize>,
    ) -> Self {
        Self(RestoreOptions {
            mode: BranchMode::Exact { reseed },
            enable_decision_memory,
            forced_first_action,
        })
    }
    #[staticmethod]
    #[pyo3(signature = (seed, *, enable_decision_memory=None, forced_first_action=None))]
    fn shuffle_libraries(
        seed: u64,
        enable_decision_memory: Option<bool>,
        forced_first_action: Option<usize>,
    ) -> Self {
        Self(RestoreOptions {
            mode: BranchMode::ShuffleLibraries { seed },
            enable_decision_memory,
            forced_first_action,
        })
    }
    #[staticmethod]
    #[pyo3(signature = (viewer, seed, *, enable_decision_memory=None, forced_first_action=None))]
    fn determinize(
        viewer: isize,
        seed: u64,
        enable_decision_memory: Option<bool>,
        forced_first_action: Option<usize>,
    ) -> PyResult<Self> {
        if !(0..=1).contains(&viewer) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "viewer must be 0 or 1",
            ));
        }
        Ok(Self(RestoreOptions {
            mode: BranchMode::Determinize {
                viewer: viewer as usize,
                seed,
            },
            enable_decision_memory,
            forced_first_action,
        }))
    }
}

#[pyclass(name = "SnapshotMetadata", frozen)]
pub(super) struct PySnapshotMetadata(pub(super) SnapshotMetadata);
#[pymethods]
impl PySnapshotMetadata {
    #[getter]
    fn turn(&self) -> u32 {
        self.0.turn
    }
    #[getter]
    fn phase(&self) -> PhaseEnum {
        self.0.phase.into()
    }
    #[getter]
    fn life(&self) -> [i32; 2] {
        self.0.life
    }
    #[getter]
    fn board_counts(&self) -> [usize; 2] {
        self.0.board_counts
    }
    #[getter]
    fn stack_len(&self) -> usize {
        self.0.stack_len
    }
    #[getter]
    fn action_count(&self) -> usize {
        self.0.action_count
    }
    #[getter]
    fn action_kind(&self) -> Option<ActionSpaceEnum> {
        self.0.action_kind.map(Into::into)
    }
    #[getter]
    fn acting_player(&self) -> Option<usize> {
        self.0.acting_player
    }
    #[getter]
    fn terminal(&self) -> bool {
        self.0.terminal
    }
    #[getter]
    fn active_card_defs(&self) -> Vec<u32> {
        self.0.active_card_defs.iter().map(|id| id.0).collect()
    }
}
