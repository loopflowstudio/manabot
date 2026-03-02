use crate::infra::profiler::InfoDict;

#[cfg(feature = "python")]
use crate::infra::profiler::InfoValue;
#[cfg(feature = "python")]
use pyo3::{
    types::{PyDict, PyDictMethods},
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

#[cfg(not(feature = "python"))]
pub fn info_dict_to_placeholder(info: &InfoDict) -> usize {
    info.len()
}
