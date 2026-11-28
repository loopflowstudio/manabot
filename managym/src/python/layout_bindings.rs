//! Dictionary discovery of native observation encodings, without constructing an Env.
// PyO3's #[pyfunction] wrappers trigger false-positive `useless_conversion`.
#![allow(clippy::useless_conversion)]
use crate::agent::{
    observation_encoder::ObservationEncoderConfig,
    observation_layout::{self as native_layout, ValueLayout},
};
use native_layout::ArrayLayout;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::collections::BTreeMap;

fn value_dict<'py>(py: Python<'py>, layout: &ValueLayout) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new_bound(py);
    value.set_item("kind", layout.kind.name())?;
    if let Some(scale) = layout.kind.scale() {
        value.set_item("scale", scale)?;
    }
    if let Some(size) = layout.kind.size() {
        value.set_item("size", size)?;
    }
    value.set_item("encoding", layout.encoding)?;
    value.set_item("range", layout.range)?;
    Ok(value)
}

/// Column names, value encodings and fixed extents of each encoded array.
///
/// Variable axes have a caller-selected capacity; seats are viewer-relative.
#[pyfunction]
#[pyo3(signature = (*, policy_history=false, decision_memory=false))]
fn observation_layout(
    py: Python<'_>,
    policy_history: bool,
    decision_memory: bool,
) -> PyResult<PyObject> {
    let mut arrays = native_layout::observation_layout(policy_history);
    if decision_memory {
        arrays.extend(native_layout::decision_memory_layout());
    }
    arrays_dict(py, &arrays)
}

fn arrays_dict(py: Python<'_>, arrays: &BTreeMap<String, ArrayLayout>) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    for (key, array) in arrays {
        let value = PyDict::new_bound(py);
        value.set_item("axes", &array.axes)?;
        value.set_item("shape", &array.shape)?;
        value.set_item("dtype", array.dtype)?;
        let columns = PyDict::new_bound(py);
        for column in &array.columns {
            let metadata = value_dict(py, &column.value)?;
            metadata.set_item("index", column.index)?;
            columns.set_item(column.name, metadata)?;
        }
        value.set_item("columns", columns)?;
        value.set_item("row_kinds", &array.row_kinds)?;
        value.set_item("row_kind_names", &array.row_kind_names)?;
        if key == "effect_details" {
            let tokens: BTreeMap<_, _> = crate::agent::shared_effects::PROGRAM_DETAIL_KINDS
                .into_iter()
                .map(|(name, token)| (array.row_kind_names[name], token as u16))
                .collect();
            value.set_item("program_token_kinds", tokens)?;
        }
        if let Some(element) = &array.element {
            value.set_item("element", value_dict(py, element)?)?;
        }
        out.set_item(key, value)?;
    }
    Ok(out.into())
}

/// Physical row boundaries for a configured CSR table.
#[derive(Clone, Copy)]
pub struct PackedTableLayout {
    pub seat_axis: bool,
    pub capacity: usize,
    pub stride: usize,
}

impl PackedTableLayout {
    fn new(key: &str, array: &ArrayLayout) -> PyResult<Self> {
        let seat_axis = array.packed_seat_axis(key);
        let axis = usize::from(seat_axis);
        let capacity = array.shape[axis].expect("resolved capacity");
        let fields = array.packed_fields(key);
        let stride = if array.element.is_some() {
            array.shape[axis + 1..]
                .iter()
                .try_fold(fields[0].storage.bytes(), |bytes, extent| {
                    bytes.checked_mul(extent.expect("resolved extent"))
                })
                .ok_or_else(|| PyValueError::new_err("packed row stride exceeds address space"))?
        } else {
            fields.iter().map(|field| field.storage.bytes()).sum()
        };
        Ok(Self {
            seat_axis,
            capacity,
            stride,
        })
    }
}

/// Resolved policy input, shared by batch encoding and archive validation.
#[pyclass(frozen, module = "managym")]
#[derive(Clone)]
pub struct ObservationLayout {
    pub config: ObservationEncoderConfig,
    pub arrays: BTreeMap<String, ArrayLayout>,
    pub packed: BTreeMap<String, PackedTableLayout>,
}

impl ObservationLayout {
    pub fn from_config(config: ObservationEncoderConfig) -> PyResult<Self> {
        let arrays = native_layout::configured_arrays(&config).map_err(PyValueError::new_err)?;
        let packed = arrays
            .iter()
            .filter(|(key, _)| !key.ends_with("_valid"))
            .map(|(key, array)| Ok((key.clone(), PackedTableLayout::new(key, array)?)))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            config,
            arrays,
            packed,
        })
    }
}

#[pymethods]
impl ObservationLayout {
    #[new]
    #[pyo3(signature = (*, policy_input_version=11, policy_history=false, max_cards_per_player=60, max_permanents_per_player=40, max_stack_objects=32, max_actions=512, max_focus_objects=2, max_events=32, max_facts=512, max_decision_records=0, max_decision_references=0, max_decision_reference_details=0, max_decision_members=0))]
    fn new(
        policy_input_version: u8,
        policy_history: bool,
        max_cards_per_player: usize,
        max_permanents_per_player: usize,
        max_stack_objects: usize,
        max_actions: usize,
        max_focus_objects: usize,
        max_events: usize,
        max_facts: usize,
        max_decision_records: usize,
        max_decision_references: usize,
        max_decision_reference_details: usize,
        max_decision_members: usize,
    ) -> PyResult<Self> {
        if policy_input_version != 11 {
            return Err(PyValueError::new_err(format!("unsupported policy input v{policy_input_version}; v0.2.6 is the last release supporting v9-v10, and ac601834d is the last revision supporting v1-v8")));
        }
        Self::from_config(ObservationEncoderConfig {
            policy_history,
            max_cards_per_player,
            max_permanents_per_player,
            max_stack_objects,
            max_actions,
            max_focus_objects,
            max_events,
            max_facts,
            max_decision_records,
            max_decision_references,
            max_decision_reference_details,
            max_decision_members,
            ..Default::default()
        })
    }

    // Spaces travel with policies through deepcopy, pickle (all protocols) and spawned workers.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let py = slf.py();
        let c = slf.borrow().config;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("policy_input_version", 11)?;
        kwargs.set_item("policy_history", c.policy_history)?;
        kwargs.set_item("max_cards_per_player", c.max_cards_per_player)?;
        kwargs.set_item("max_permanents_per_player", c.max_permanents_per_player)?;
        kwargs.set_item("max_actions", c.max_actions)?;
        kwargs.set_item("max_focus_objects", c.max_focus_objects)?;
        kwargs.set_item("max_stack_objects", c.max_stack_objects)?;
        kwargs.set_item("max_events", c.max_events)?;
        kwargs.set_item("max_facts", c.max_facts)?;
        kwargs.set_item("max_decision_records", c.max_decision_records)?;
        kwargs.set_item("max_decision_references", c.max_decision_references)?;
        kwargs.set_item(
            "max_decision_reference_details",
            c.max_decision_reference_details,
        )?;
        kwargs.set_item("max_decision_members", c.max_decision_members)?;
        let new = py.import_bound("copyreg")?.getattr("__newobj_ex__")?;
        Ok((new, (slf.get_type(), PyTuple::empty_bound(py), kwargs)).into_py(py))
    }

    /// Buffers on each variable axis and their row dimensions before batching.
    #[getter]
    fn axis_buffers(&self) -> BTreeMap<&'static str, Vec<(String, usize)>> {
        let mut buffers = BTreeMap::new();
        for (key, array) in &self.arrays {
            if let Some((axis, dimension)) = crate::agent::observation_extents::row_axis(key, array)
            {
                buffers
                    .entry(axis)
                    .or_insert_with(Vec::new)
                    .push((key.clone(), dimension - 1));
            }
        }
        buffers
    }

    #[getter]
    fn arrays(&self, py: Python<'_>) -> PyResult<PyObject> {
        arrays_dict(py, &self.arrays)
    }

    /// Divisor of each fact kind's raw CSR payload in its logical v11 columns.
    #[getter]
    fn csr_fact_divisors(&self) -> BTreeMap<u8, u8> {
        crate::agent::observation_encoder::FactKind::ALL
            .into_iter()
            .map(|kind| (kind as u8, kind.payload_divisor()))
            .collect()
    }

    /// Physical fields of each v11 CSR record: name, NumPy dtype and logical column.
    #[getter]
    fn csr_fields(&self) -> BTreeMap<&str, Vec<(&str, &str, usize)>> {
        self.arrays
            .iter()
            .filter(|(key, _)| !key.ends_with("_valid"))
            .map(|(key, array)| {
                (
                    key.as_str(),
                    array
                        .packed_fields(key)
                        .into_iter()
                        .map(|f| (f.name, f.storage.dtype(), f.index))
                        .collect(),
                )
            })
            .collect()
    }

    #[getter]
    fn shapes(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        for (key, array) in &self.arrays {
            out.set_item(
                key,
                PyTuple::new_bound(py, array.shape.iter().map(|extent| extent.unwrap())),
            )?;
        }
        Ok(out.into())
    }

    #[getter]
    fn dtypes(&self) -> BTreeMap<String, &str> {
        self.arrays
            .iter()
            .map(|(k, a)| (k.clone(), a.dtype))
            .collect()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ObservationLayout>()?;
    module.add_function(wrap_pyfunction!(observation_layout, module)?)?;
    Ok(())
}
