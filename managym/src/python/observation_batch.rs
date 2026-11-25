//! Freeze the encoder's live prefixes before they cross the Python boundary.
// PyO3 wrappers generate identity conversions for PyResult.
#![allow(clippy::useless_conversion)]
mod archive;

use super::bindings::{map_agent_err, map_encode_err, PyEnv, PyObservation};
use super::layout_bindings::{ObservationLayout, PackedTableLayout};
use crate::agent::{
    observation::Observation, observation_encoder::ObservationEncoderConfig,
    observation_rows::PackedRows,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::collections::BTreeMap;

struct FrozenTable {
    stride: usize,
    seat_axis: bool,
    offsets: Py<PyBytes>,
    values: Py<PyBytes>,
}

/// Live v11 rows in batch order, frozen by native encoding. Pickle and archives
/// (`to_bytes`/`from_bytes`) preserve them; archives validate the layout, row
/// boundaries and joins on load.
#[pyclass(frozen, module = "managym")]
pub struct ObservationBatch {
    layout: Py<ObservationLayout>,
    tables: BTreeMap<String, FrozenTable>,
    #[pyo3(get)]
    batch_size: usize,
}

#[pymethods]
impl ObservationBatch {
    /// Store the configured layout and packed records, including empty seat spans.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        archive::encode(py, self)
    }

    /// Reload a complete v11 batch after checking its physical layout and native joins.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        archive::decode(py, data)
    }

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

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let restore = py
            .import_bound("managym._managym")?
            .getattr("_restore_observation_batch")?;
        Ok((
            restore,
            (self.layout.bind(py), self.batch_size, self.tables(py)?),
        )
            .into_py(py))
    }

    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __deepcopy__(slf: Py<Self>, _memo: &Bound<'_, PyAny>) -> Py<Self> {
        slf
    }

    /// Gather arbitrary coordinates across compatible native batches, including repeats.
    fn gather(&self, py: Python<'_>, rows: Vec<(PyRef<'_, Self>, usize)>) -> PyResult<Self> {
        let config = self.layout.borrow(py).config;
        for (batch, index) in &rows {
            if *index >= batch.batch_size || batch.layout.borrow(py).config != config {
                return Err(PyValueError::new_err(
                    "invalid observation batch coordinate or layout",
                ));
            }
        }
        let mut tables = BTreeMap::new();
        for (key, source) in &self.tables {
            let groups = if source.seat_axis { 2 } else { 1 };
            let mut offsets = Vec::with_capacity((1 + rows.len() * groups) * 4);
            let mut values = Vec::new();
            let mut end = 0i32;
            offsets.extend(end.to_ne_bytes());
            for (batch, index) in &rows {
                let table = &batch.tables[key];
                let bytes = table.offsets.as_bytes(py);
                let at = |group: usize| {
                    i32::from_ne_bytes(bytes[group * 4..group * 4 + 4].try_into().unwrap())
                };
                let group = index * groups;
                let start = at(group);
                let stop = at(group + groups);
                for i in 1..=groups {
                    offsets.extend(
                        end.checked_add(at(group + i) - start)
                            .ok_or_else(|| {
                                PyValueError::new_err("observation batch exceeds int32 row offsets")
                            })?
                            .to_ne_bytes(),
                    );
                }
                end += stop - start;
                values.extend_from_slice(
                    &table.values.as_bytes(py)
                        [start as usize * table.stride..stop as usize * table.stride],
                );
            }
            tables.insert(
                key.clone(),
                FrozenTable {
                    seat_axis: source.seat_axis,
                    stride: source.stride,
                    offsets: PyBytes::new_bound(py, &offsets).unbind(),
                    values: PyBytes::new_bound(py, &values).unbind(),
                },
            );
        }
        Ok(Self {
            layout: self.layout.clone_ref(py),
            tables,
            batch_size: rows.len(),
        })
    }

    /// Share ordinary spans and release memory bytes after interning the viewer prefixes.
    fn blank_decision_memory(&self, py: Python<'_>) -> Self {
        let empty_offsets = PyBytes::new_bound(py, &vec![0; 4 * (1 + self.batch_size)]).unbind();
        let empty_values = PyBytes::new_bound(py, &[]).unbind();
        let tables = self
            .tables
            .iter()
            .map(|(key, table)| {
                let memory = key.starts_with("decision_");
                (
                    key.clone(),
                    FrozenTable {
                        stride: table.stride,
                        seat_axis: table.seat_axis,
                        offsets: if memory {
                            &empty_offsets
                        } else {
                            &table.offsets
                        }
                        .clone_ref(py),
                        values: if memory { &empty_values } else { &table.values }.clone_ref(py),
                    },
                )
            })
            .collect();
        Self {
            layout: self.layout.clone_ref(py),
            tables,
            batch_size: self.batch_size,
        }
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
    Ok(freeze(py, layout, rows, batch_size))
}

pub(super) fn freeze(
    py: Python<'_>,
    layout: Py<ObservationLayout>,
    rows: PackedRows,
    batch_size: usize,
) -> ObservationBatch {
    let tables = {
        let schema = layout.borrow(py);
        rows.tables
            .into_iter()
            .map(|(key, table)| {
                let stride = schema.packed[&key].stride;
                let offsets: Vec<_> = table
                    .offsets
                    .into_iter()
                    .flat_map(i32::to_ne_bytes)
                    .collect();
                (
                    key,
                    FrozenTable {
                        stride,
                        seat_axis: table.seat_axis,
                        offsets: PyBytes::new_bound(py, &offsets).unbind(),
                        values: PyBytes::new_bound(py, &table.values).unbind(),
                    },
                )
            })
            .collect()
    };
    ObservationBatch {
        layout,
        tables,
        batch_size,
    }
}

impl FrozenTable {
    fn restore(
        spec: PackedTableLayout,
        batch_size: usize,
        raw: &Bound<'_, PyAny>,
        key: &str,
    ) -> PyResult<Self> {
        let invalid = || PyValueError::new_err(format!("invalid packed observation table {key}"));
        let (seat_axis, offsets, values): (bool, Bound<'_, PyBytes>, Bound<'_, PyBytes>) =
            raw.extract().map_err(|_| invalid())?;
        let offset_len = batch_size
            .checked_mul(if spec.seat_axis { 2 } else { 1 })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(4));
        if seat_axis != spec.seat_axis || offset_len != Some(offsets.as_bytes().len()) {
            return Err(invalid());
        }
        let mut ends = offsets
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(i32::from_ne_bytes);
        if ends.next() != Some(0) {
            return Err(invalid());
        }
        let mut previous = 0;
        for end in ends {
            if end < previous || (end - previous) as usize > spec.capacity {
                return Err(invalid());
            }
            previous = end;
        }
        if (previous as usize).checked_mul(spec.stride) != Some(values.as_bytes().len()) {
            return Err(invalid());
        }
        Ok(Self {
            seat_axis,
            stride: spec.stride,
            offsets: offsets.unbind(),
            values: values.unbind(),
        })
    }
}

#[pyfunction]
fn _restore_observation_batch(
    py: Python<'_>,
    layout: Py<ObservationLayout>,
    batch_size: usize,
    tables: &Bound<'_, PyDict>,
) -> PyResult<ObservationBatch> {
    let schema = layout.borrow(py);
    if tables.len() != schema.packed.len() {
        return Err(PyValueError::new_err(
            "expected the tables of a v11 observation layout",
        ));
    }
    let mut restored = BTreeMap::new();
    for (key, &spec) in &schema.packed {
        let raw = tables
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("missing observation table {key}")))?;
        restored.insert(
            key.clone(),
            FrozenTable::restore(spec, batch_size, &raw, key)?,
        );
    }
    drop(schema);
    Ok(ObservationBatch {
        layout,
        batch_size,
        tables: restored,
    })
}

/// Encode acting-view envs or standalone observations, preserving input order.
#[pyfunction]
#[pyo3(signature = (envs, layout, action_overflow="error"))]
fn encode_observation_batch(
    py: Python<'_>,
    envs: Vec<Bound<'_, PyAny>>,
    layout: Py<ObservationLayout>,
    action_overflow: &str,
) -> PyResult<ObservationBatch> {
    let config = config(&layout.borrow(py), action_overflow)?;
    batch(
        py,
        layout,
        config,
        envs.iter().map(|item| {
            if let Ok(branch) = item.extract::<PyRef<'_, PyEnv>>() {
                let env = branch
                    .inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
                let actor = env
                    .current_agent_index()
                    .ok_or_else(|| PyValueError::new_err("env has no acting player"))?;
                env.observation_for_player(actor).map_err(map_agent_err)
            } else if let Ok(obs) = item.extract::<PyObservation>() {
                Ok(Observation::from(obs))
            } else {
                Err(PyTypeError::new_err(
                    "encode_observation_batch expects Env or Observation items",
                ))
            }
        }),
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ObservationBatch>()?;
    m.add_function(wrap_pyfunction!(_restore_observation_batch, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation_batch, m)?)
}
