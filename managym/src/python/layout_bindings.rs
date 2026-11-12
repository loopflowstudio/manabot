//! Dictionary discovery of native policy layouts, without constructing an Env.
// PyO3's #[pyfunction] wrappers trigger false-positive `useless_conversion`.
#![allow(clippy::useless_conversion)]
use crate::agent::observation_layout::{self as native_layout, ValueLayout};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn value_dict<'py>(py: Python<'py>, layout: &ValueLayout) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new_bound(py);
    value.set_item("kind", layout.kind)?;
    value.set_item("encoding", layout.encoding)?;
    value.set_item("range", layout.range)?;
    Ok(value)
}

/// Native column names, encodings and fixed extents of each encoded array.
///
/// Keys are the encoder's array names for `policy_input_version` (1–9) and
/// `policy_history_version` (0 legacy events, 1 public history). A `None`
/// extent is a caller-selected capacity. Seats are viewer-relative. Unsupported
/// versions raise `ValueError`; no game or buffer is created.
#[pyfunction]
#[pyo3(signature = (policy_input_version, *, policy_history_version=0))]
fn observation_layout(
    py: Python<'_>,
    policy_input_version: i64,
    policy_history_version: i64,
) -> PyResult<PyObject> {
    let version = u8::try_from(policy_input_version)
        .map_err(|_| PyValueError::new_err("unsupported policy input version; expected 1..9"))?;
    let history = u8::try_from(policy_history_version).map_err(|_| {
        PyValueError::new_err("unsupported policy history version; expected 0 or 1")
    })?;
    let layout =
        native_layout::observation_layout(version, history).map_err(PyValueError::new_err)?;
    let out = PyDict::new_bound(py);
    for (key, array) in layout {
        let value = PyDict::new_bound(py);
        value.set_item("axes", array.axes)?;
        value.set_item("shape", array.shape)?;
        value.set_item("dtype", array.dtype)?;
        let columns = PyDict::new_bound(py);
        for column in array.columns {
            let metadata = value_dict(py, &column.value)?;
            metadata.set_item("index", column.index)?;
            columns.set_item(column.name, metadata)?;
        }
        value.set_item("columns", columns)?;
        value.set_item("row_kinds", array.row_kinds)?;
        if let Some(element) = array.element {
            value.set_item("element", value_dict(py, &element)?)?;
        }
        out.set_item(key, value)?;
    }
    Ok(out.into())
}

/// The input version to pin for a new consumer with no saved configuration.
///
/// Resolve it once and save it with the model; it does not change the implicit
/// v6 default of `encode_observation` or any saved version.
#[pyfunction]
fn recommended_policy_input_version() -> u8 {
    native_layout::RECOMMENDED_POLICY_INPUT_VERSION
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(observation_layout, module)?)?;
    module.add_function(wrap_pyfunction!(recommended_policy_input_version, module)?)?;
    Ok(())
}
