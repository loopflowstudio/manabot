//! Freeze the encoder's live prefixes before they cross the Python boundary.
use super::bindings::{map_agent_err, map_encode_err, PyEnv};
use super::layout_bindings::ObservationLayout;
use crate::agent::{
    observation::Observation,
    observation_encoder::{allocate, encode_counted, InputLayout, ObservationEncoderConfig},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use std::collections::BTreeMap;

struct Table {
    offsets: Vec<i32>,
    values: Vec<u8>,
    capacity: usize,
    width: usize,
    seat_axis: bool,
}

impl Table {
    fn append<T: Copy>(
        &mut self,
        source: &[T],
        counts: &[usize],
        bytes: fn(T) -> [u8; 4],
    ) -> PyResult<()> {
        for (seat, &count) in counts.iter().enumerate() {
            let end = i32::try_from(count)
                .ok()
                .and_then(|n| self.offsets.last()?.checked_add(n))
                .ok_or_else(|| {
                    PyValueError::new_err("observation batch exceeds int32 row offsets")
                })?;
            let start = seat * self.capacity * self.width;
            self.values.extend(
                source[start..start + count * self.width]
                    .iter()
                    .copied()
                    .flat_map(bytes),
            );
            self.offsets.push(end);
        }
        Ok(())
    }
}

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
    /// values in layout dtype. Both byte exports are immutable. A two-seat
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
    let mut tables: BTreeMap<_, _> = layout
        .borrow(py)
        .arrays
        .iter()
        .filter(|(key, _)| !key.ends_with("_valid"))
        .map(|(key, array)| {
            let seat_axis = array.axes[0] == "viewer_relative_seat" && key != "player_turn_state";
            let axis = usize::from(seat_axis);
            (
                key.clone(),
                Table {
                    capacity: array.shape[axis].unwrap(),
                    width: array.shape[axis + 1..].iter().map(|n| n.unwrap()).product(),
                    seat_axis,
                    offsets: vec![0],
                    values: Vec::new(),
                },
            )
        })
        .collect();
    let mut row = allocate(&config);
    let mut batch_size = 0;
    for observation in observations {
        let n =
            encode_counted(&observation?, &config, row.buffers_mut()).map_err(map_encode_err)?;
        macro_rules! append {
            ($field:ident, $counts:expr) => {
                tables.get_mut(stringify!($field)).unwrap().append(
                    &row.$field,
                    &$counts,
                    f32::to_ne_bytes,
                )?
            };
        }
        append!(agent_player, [1]);
        append!(opponent_player, [1]);
        append!(player_turn_state, [2]);
        append!(agent_cards, [n.cards[0]]);
        append!(opponent_cards, [n.cards[1]]);
        append!(semantic_cards, n.cards);
        append!(card_turn_state, n.cards);
        append!(known_hand, n.known_hand);
        append!(known_library, n.known_library);
        append!(decklists, n.decklists);
        append!(agent_permanents, [n.permanents[0]]);
        append!(opponent_permanents, [n.permanents[1]]);
        append!(stack, [n.stack]);
        append!(events, [n.events]);
        append!(actions, [n.actions]);
        append!(name_candidates, [n.actions]);
        tables.get_mut("action_focus").unwrap().append(
            &row.action_focus,
            &[n.actions],
            i32::to_ne_bytes,
        )?;
        append!(relations, [n.relations]);
        append!(facts, [n.facts]);
        append!(effects, [n.effects]);
        append!(effect_details, [n.effect_details]);
        append!(effect_members, [n.effect_members]);
        if config.max_decision_records > 0 {
            append!(decision_records, [n.decision_memory[0]]);
            append!(decision_reference_details, [n.decision_memory[1]]);
            append!(decision_references, [n.decision_memory[2]]);
            append!(decision_members, [n.decision_memory[3]]);
        }
        batch_size += 1;
    }
    let tables = tables
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
