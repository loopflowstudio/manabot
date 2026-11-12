//! Dictionary discovery of native observation encodings, without constructing an Env.
// PyO3's #[pyfunction] wrappers trigger false-positive `useless_conversion`.
#![allow(clippy::useless_conversion)]
use crate::agent::observation_layout::{self as native_layout, ValueLayout};
use pyo3::{prelude::*, types::PyDict};

fn value_dict<'py>(py: Python<'py>, layout: &ValueLayout) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new_bound(py);
    value.set_item("kind", layout.kind)?;
    value.set_item("encoding", layout.encoding)?;
    value.set_item("range", layout.range)?;
    Ok(value)
}

/// Column names, value encodings and fixed extents of each encoded array.
///
/// Keys are the encoder's array names; `policy_history` selects the public
/// history rows of `events`; `turn_state` adds the input v10
/// `player_turn_state` and `card_turn_state` arrays. A `None` extent is a caller-selected capacity.
/// Seats are viewer-relative. No game or buffer is created.
#[pyfunction]
#[pyo3(signature = (*, policy_history=false, turn_state=false))]
fn observation_layout(
    py: Python<'_>,
    policy_history: bool,
    turn_state: bool,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    for (key, array) in native_layout::observation_layout(policy_history, turn_state) {
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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(observation_layout, module)?)?;
    Ok(())
}
