//! Freeze the encoder's live prefixes before they cross the Python boundary.
// PyO3 wrappers generate identity conversions for PyResult.
#![allow(clippy::useless_conversion)]
use super::bindings::{map_agent_err, map_encode_err, PyEnv};
use super::layout_bindings::ObservationLayout;
use crate::agent::{
    observation::Observation,
    observation_encoder::{InputLayout, ObservationEncoderConfig},
    observation_rows::PackedRows,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::collections::BTreeMap;

struct FrozenTable {
    seat_axis: bool,
    offsets: Py<PyBytes>,
    values: Py<PyBytes>,
}

/// Live v11 rows in batch order. Only a native game projection can produce this type.
#[pyclass(frozen, module = "managym")]
pub struct ObservationBatch {
    layout: Py<ObservationLayout>,
    tables: BTreeMap<String, FrozenTable>,
    #[pyo3(get)]
    batch_size: usize,
}

#[pymethods]
impl ObservationBatch {
    #[getter]
    fn layout(&self, py: Python<'_>) -> Py<ObservationLayout> {
        self.layout.clone_ref(py)
    }

    /// Per table: a seat axis flag, native-endian int32 row offsets and packed
    /// records described by layout.csr_fields. Both byte exports are immutable. A two-seat
    /// table has two consecutive spans per observation.
    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        for (key, table) in &self.tables {
            out.set_item(
                key,
                (
                    table.seat_axis,
                    table.offsets.bind(py),
                    table.values.bind(py),
                ),
            )?;
        }
        Ok(out)
    }

    /// Retained action counts, including first-legal truncation.
    #[getter]
    fn action_counts(&self, py: Python<'_>) -> Vec<i32> {
        let offsets = self.tables["actions"].offsets.as_bytes(py);
        offsets
            .windows(8)
            .step_by(4)
            .map(|pair| {
                i32::from_ne_bytes(pair[4..].try_into().unwrap())
                    - i32::from_ne_bytes(pair[..4].try_into().unwrap())
            })
            .collect()
    }
}

pub(super) fn config(
    layout: &ObservationLayout,
    overflow: &str,
) -> PyResult<ObservationEncoderConfig> {
    if layout.config.layout != InputLayout::OneToken {
        return Err(PyValueError::new_err(
            "observation batches require policy input v11",
        ));
    }
    Ok(ObservationEncoderConfig {
        action_overflow: overflow.parse().map_err(PyValueError::new_err)?,
        ..layout.config
    })
}

pub(super) fn batch(
    py: Python<'_>,
    layout: Py<ObservationLayout>,
    config: ObservationEncoderConfig,
    observations: impl Iterator<Item = PyResult<Observation>>,
) -> PyResult<ObservationBatch> {
    let mut rows = PackedRows::new(&layout.borrow(py).arrays);
    let mut batch_size = 0;
    for observation in observations {
        rows.append(&observation?, &config)
            .map_err(map_encode_err)?;
        batch_size += 1;
    }
    let tables = rows
        .tables
        .into_iter()
        .map(|(key, table)| {
            let offsets: Vec<_> = table
                .offsets
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect();
            (
                key,
                FrozenTable {
                    seat_axis: table.seat_axis,
                    offsets: PyBytes::new_bound(py, &offsets).unbind(),
                    values: PyBytes::new_bound(py, &table.values).unbind(),
                },
            )
        })
        .collect();
    Ok(ObservationBatch {
        layout,
        tables,
        batch_size,
    })
}

/// Encode acting-view frontiers, preserving input order and repeated games.
#[pyfunction]
#[pyo3(signature = (envs, layout, action_overflow="error"))]
fn encode_observation_batch(
    py: Python<'_>,
    envs: Vec<PyRef<'_, PyEnv>>,
    layout: Py<ObservationLayout>,
    action_overflow: &str,
) -> PyResult<ObservationBatch> {
    let config = config(&layout.borrow(py), action_overflow)?;
    batch(
        py,
        layout,
        config,
        envs.iter().map(|branch| {
            let env = branch
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
            let actor = env
                .current_agent_index()
                .ok_or_else(|| PyValueError::new_err("env has no acting player"))?;
            env.observation_for_player(actor).map_err(map_agent_err)
        }),
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ObservationBatch>()?;
    m.add_function(wrap_pyfunction!(encode_observation_batch, m)?)
}
