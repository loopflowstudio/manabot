#[cfg(feature = "python")]
use crate::infra::profiler::{InfoDict, InfoValue};
#[cfg(feature = "python")]
use pyo3::{
    exceptions::{PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyDictMethods},
    Bound, Python,
};

#[cfg(feature = "python")]
pub fn info_dict_to_pydict<'py>(py: Python<'py>, info: &InfoDict) -> Bound<'py, PyDict> {
    let out = PyDict::new_bound(py);
    for (k, v) in info {
        match v {
            InfoValue::String(s) => {
                let _ = out.set_item(k, s);
            }
            InfoValue::Map(m) => {
                let nested = info_dict_to_pydict(py, m);
                let _ = out.set_item(k, nested);
            }
            InfoValue::Int(i) => {
                let _ = out.set_item(k, i);
            }
            InfoValue::Float(f) => {
                let _ = out.set_item(k, f);
            }
        }
    }
    out
}

#[cfg(feature = "python")]
pub fn shape_to_vec(shape_obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    shape_obj.extract::<Vec<usize>>()
}

#[cfg(feature = "python")]
pub fn require_numpy_array<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    expected_shape: &[usize],
    expected_dtype_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyKeyError::new_err(format!(
            "buffers missing required key '{key}'"
        )));
    };

    let dtype_name = value
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    if dtype_name != expected_dtype_name {
        return Err(PyTypeError::new_err(format!(
            "buffer '{key}' must have dtype {expected_dtype_name}, got {dtype_name}"
        )));
    }

    let shape = shape_to_vec(&value.getattr("shape")?)?;
    if shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "buffer '{key}' must have shape {:?}, got {:?}",
            expected_shape, shape
        )));
    }

    let flags = value.getattr("flags")?;
    let c_contiguous = flags.getattr("c_contiguous")?.extract::<bool>()?;
    if !c_contiguous {
        return Err(PyValueError::new_err(format!(
            "buffer '{key}' must be C-contiguous"
        )));
    }

    let writable = flags.getattr("writeable")?.extract::<bool>()?;
    if !writable {
        return Err(PyValueError::new_err(format!(
            "buffer '{key}' must be writable"
        )));
    }

    Ok(value)
}
