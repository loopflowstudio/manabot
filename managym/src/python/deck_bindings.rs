//! Python surface of decks and decklist imports (`state::deck`,
//! `cards::decklist`). Records convert to and from their serde JSON form and
//! validate as Pydantic fields, so manabot configurations and Academy traces
//! hold the native record itself rather than a Python copy.
#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

use std::{collections::BTreeMap, path::PathBuf};

use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyModule, PyTuple, PyType},
};
use serde::{de::DeserializeOwned, Serialize};

use super::bindings::PyPlayerConfig;
use crate::{
    cards::{
        deck_admission::{check_decks as native_check_decks, DeckIssue, DEVELOPMENT_RULESET},
        decklist::{
            admit_decklists as native_admit_decklists, parse_decklist, read_decklist, DeckEntry,
            DeckImport, Decklist, DecklistError,
        },
    },
    format::limited::{
        basic_lands, basics, card_count, CardPool, Limited, LimitedBuild, DEFAULT_BUILD_CAP,
        LIMITED_MIN_CARDS,
    },
    state::deck::Deck,
};

pyo3::create_exception!(_managym, DeckImportError, PyValueError);

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn to_python(py: Python<'_>, value: &impl Serialize) -> PyResult<PyObject> {
    let text = serde_json::to_string(value).map_err(value_error)?;
    Ok(py
        .import_bound("json")?
        .call_method1("loads", (text,))?
        .unbind())
}

fn from_python<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let text: String = value
        .py()
        .import_bound("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&text).map_err(value_error)
}

/// A Pydantic core schema that accepts the record or its JSON form and
/// serializes to the JSON form.
fn pydantic_schema(cls: &Bound<'_, PyType>) -> PyResult<PyObject> {
    let schema = cls.py().import_bound("pydantic_core.core_schema")?;
    let serialization = schema.call_method1(
        "plain_serializer_function_ser_schema",
        (cls.getattr("to_dict")?,),
    )?;
    let kwargs = PyDict::new_bound(cls.py());
    kwargs.set_item("serialization", serialization)?;
    Ok(schema
        .call_method(
            "no_info_plain_validator_function",
            (cls.getattr("_validate")?,),
            Some(&kwargs),
        )?
        .unbind())
}

/// One `#[pymethods]` block per record: the given methods plus JSON
/// conversion and Pydantic validation.
macro_rules! json_record {
    ($type:ty { $($methods:tt)* }) => {
        #[pymethods]
        impl $type {
            $($methods)*

            fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
                to_python(py, self)
            }

            #[staticmethod]
            fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                from_python(value)
            }

            #[staticmethod]
            fn _validate(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                match value.extract::<Self>() {
                    Ok(record) => Ok(record),
                    Err(_) => from_python(value),
                }
            }

            #[classmethod]
            fn __get_pydantic_core_schema__(
                cls: &Bound<'_, PyType>,
                _source: &Bound<'_, PyAny>,
                _handler: &Bound<'_, PyAny>,
            ) -> PyResult<PyObject> {
                pydantic_schema(cls)
            }

            // Records are immutable values: copies are clones, and pickling
            // round-trips through the JSON form.
            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(PyObject, (PyObject,))> {
                let py = slf.py();
                let from_dict = slf.get_type().getattr("from_dict")?.unbind();
                Ok((from_dict, (to_python(py, &*slf.borrow())?,)))
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}({})",
                    stringify!($type),
                    serde_json::to_string(self).unwrap_or_default()
                )
            }
        }
    };
}

json_record!(Deck {
    #[new]
    #[pyo3(signature = (main, sideboard=None))]
    fn py_new(main: BTreeMap<String, usize>, sideboard: Option<BTreeMap<String, usize>>) -> Self {
        Deck::new(main, sideboard.unwrap_or_default())
    }

    #[getter]
    fn main(&self) -> BTreeMap<String, usize> {
        self.main.clone()
    }

    #[getter]
    fn sideboard(&self) -> BTreeMap<String, usize> {
        self.sideboard.clone()
    }

    /// Content identity; see `Deck::identity`.
    #[getter(identity)]
    fn py_identity(&self) -> String {
        self.identity()
    }

    /// A seat setup named by the deck's identity, submitted to the world.
    fn player(&self) -> PyPlayerConfig {
        PyPlayerConfig {
            name: self.identity(),
            deck: self.clone(),
            submitted: true,
            disclosure: Default::default(),
        }
    }
});

json_record!(DeckIssue {
    #[getter]
    fn line(&self) -> Option<usize> {
        self.line
    }

    #[getter]
    fn section(&self) -> Option<&'static str> {
        self.section.map(|section| section.as_str())
    }

    #[getter]
    fn card_name(&self) -> Option<String> {
        self.card_name.clone()
    }

    #[getter]
    fn code(&self) -> String {
        self.code.clone()
    }

    #[getter]
    fn message(&self) -> String {
        self.message.clone()
    }
});

json_record!(DeckEntry {
    #[getter]
    fn line(&self) -> usize {
        self.line
    }

    #[getter]
    fn section(&self) -> &'static str {
        self.section.as_str()
    }

    #[getter]
    fn count(&self) -> usize {
        self.count
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn card_def_id(&self) -> Option<u32> {
        self.card_def_id
    }

    #[getter]
    fn key(&self) -> Option<String> {
        self.key.clone()
    }
});

json_record!(Decklist {
    /// Parse `source` (unresolved; names resolve on admission).
    #[new]
    #[pyo3(signature = (source, filename="deck.txt"))]
    fn py_new(source: &str, filename: &str) -> Self {
        parse_decklist(source, filename)
    }

    /// Read a bounded UTF-8 file, keeping BOM and newline bytes; the filename
    /// defaults to the path's final component.
    #[staticmethod]
    #[pyo3(signature = (path, filename=None))]
    fn read(path: PathBuf, filename: Option<&str>) -> PyResult<Self> {
        read_decklist(&path, filename).map_err(value_error)
    }

    #[getter]
    fn dialect(&self) -> String {
        self.dialect.clone()
    }

    #[getter]
    fn source(&self) -> String {
        self.source.clone()
    }

    #[getter]
    fn filename(&self) -> String {
        self.filename.clone()
    }

    #[getter]
    fn entries(&self) -> Vec<DeckEntry> {
        self.entries.clone()
    }

    #[getter]
    fn issues(&self) -> Vec<DeckIssue> {
        self.issues.clone()
    }

    /// Raise `ValueError` unless this retained source re-admits to exactly
    /// `deck` against the installed world.
    #[pyo3(name = "validate_deck")]
    fn py_validate_deck(&self, deck: Deck) -> PyResult<()> {
        self.validate_deck(&deck).map_err(value_error)
    }
});

json_record!(DeckImport {
    #[getter]
    fn decklist(&self) -> Decklist {
        self.decklist.clone()
    }

    #[getter]
    fn deck(&self) -> Deck {
        self.deck.clone()
    }
});

fn decklist_error(py: Python<'_>, error: DecklistError) -> PyErr {
    let DecklistError(rejected) = &error;
    let err = DeckImportError::new_err(error.to_string());
    let decklists = PyDict::new_bound(py);
    let attached = rejected
        .iter()
        .try_for_each(|(label, decklist)| decklists.set_item(label, decklist.clone().into_py(py)))
        .and_then(|()| err.value_bound(py).setattr("decklists", decklists));
    match attached {
        Ok(()) => err,
        Err(attach) => attach,
    }
}

/// Each deck's construction and eligibility issues in this world, in order,
/// without allocating a game.
#[pyfunction]
fn check_decks(decks: Vec<Deck>) -> Vec<Vec<DeckIssue>> {
    native_check_decks(&decks)
}

/// Admit labelled decklists, rejecting all of them with `DeckImportError`
/// (its `decklists` maps each failing label to its diagnosed record).
#[pyfunction]
fn admit_decklists(
    py: Python<'_>,
    decklists: Vec<(String, Decklist)>,
) -> PyResult<Vec<DeckImport>> {
    native_admit_decklists(&decklists).map_err(|e| decklist_error(py, e))
}

/// Names and positive integer counts, each malformed count named by its card.
fn card_counts(cards: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, usize>> {
    cards
        .iter()
        .map(|(name, count)| {
            let name: String = name.extract()?;
            let parsed = if count.is_instance_of::<PyBool>() {
                None
            } else {
                count.extract::<usize>().ok().filter(|&n| n > 0)
            };
            parsed
                .map(|n| (name.clone(), n))
                .ok_or_else(|| value_error(format!("{name}: count must be a positive integer")))
        })
        .collect()
}

// Count rows cross the boundary as NumPy arrays over the card axis
// (`CardDefId` order); managym never imports torch.

/// A C-contiguous NumPy array of `shape` holding `data`.
fn numpy_array<T: Element + Copy>(
    py: Python<'_>,
    data: &[T],
    shape: &[usize],
    dtype: &str,
) -> PyResult<PyObject> {
    let np = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr(dtype)?)?;
    let array = np.call_method("empty", (shape.to_vec(),), Some(&kwargs))?;
    PyBuffer::<T>::get_bound(&array)?.copy_from_slice(py, data)?;
    Ok(array.unbind())
}

/// A boolean NumPy array, written as one byte per value.
fn numpy_bools(py: Python<'_>, values: &[bool], shape: &[usize]) -> PyResult<PyObject> {
    let bytes = values.iter().map(|&b| u8::from(b)).collect::<Vec<_>>();
    let array = numpy_array(py, &bytes, shape, "uint8")?;
    let np = py.import_bound("numpy")?;
    Ok(array
        .bind(py)
        .call_method1("view", (np.getattr("bool_")?,))?
        .unbind())
}

/// Integer counts of shape `[n, cards]` (or one `[cards]` row unless
/// `matrix`); a negative, non-integer or misshapen input is a `ValueError`.
fn count_rows(value: &Bound<'_, PyAny>, what: &str, matrix: bool) -> PyResult<CountRows> {
    let py = value.py();
    let np = py.import_bound("numpy")?;
    let array = np.call_method1("asarray", (value,))?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind != "i" && kind != "u" {
        return Err(value_error(format!("{what} must be an integer array")));
    }
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    let cards = card_count();
    let rows = match (matrix, shape.as_slice()) {
        (true, &[n, c]) if c == cards => n,
        (false, &[c]) if c == cards => 1,
        _ => {
            let expected = if matrix { "[n, cards]" } else { "[cards]" };
            return Err(value_error(format!(
                "{what} must have shape {expected} with {cards} cards, got {shape:?}"
            )));
        }
    };
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("int64")?)?;
    let contiguous = np.call_method("ascontiguousarray", (array,), Some(&kwargs))?;
    let flat = PyBuffer::<i64>::get_bound(&contiguous)?.to_vec(py)?;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = flat[r * cards..(r + 1) * cards]
            .iter()
            .map(|&n| u32::try_from(n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                value_error(format!(
                    "{what} row {r}: counts must be non-negative and fit 32 bits"
                ))
            })?;
        out.push(row);
    }
    Ok(out)
}

type CountRows = Vec<Vec<u32>>;

/// Pool and deck rows of one shape, paired by row.
fn paired_rows(
    pools: &Bound<'_, PyAny>,
    decks: &Bound<'_, PyAny>,
) -> PyResult<(CountRows, CountRows)> {
    let pools = count_rows(pools, "pools", true)?;
    let decks = count_rows(decks, "decks", true)?;
    if pools.len() != decks.len() {
        return Err(value_error(format!(
            "pools has {} rows and decks has {}",
            pools.len(),
            decks.len()
        )));
    }
    Ok((pools, decks))
}

#[pymethods]
impl CardPool {
    #[new]
    fn py_new(cards: &Bound<'_, PyDict>) -> PyResult<Self> {
        Self::new(card_counts(cards)?).map_err(value_error)
    }

    /// A pool from int counts of shape `[cards]` over the card axis.
    #[staticmethod]
    #[pyo3(name = "from_counts")]
    fn py_from_counts(counts: &Bound<'_, PyAny>) -> PyResult<Self> {
        let row = count_rows(counts, "counts", false)?.remove(0);
        Self::from_counts(&row).map_err(value_error)
    }

    #[getter(cards)]
    fn py_cards(&self) -> BTreeMap<String, usize> {
        self.cards().clone()
    }

    /// Int32 counts of shape `[cards]` over the card axis.
    #[getter(counts)]
    fn py_counts(&self, py: Python<'_>) -> PyResult<PyObject> {
        let counts = self.counts().iter().map(|&n| n as i32).collect::<Vec<_>>();
        numpy_array(py, &counts, &[counts.len()], "int32")
    }

    fn __repr__(&self) -> String {
        format!("CardPool({:?})", self.cards())
    }
}

#[pymethods]
impl Limited {
    #[new]
    fn py_new() -> Self {
        Self
    }

    #[pyo3(name = "build", signature = (pool, max_cards=DEFAULT_BUILD_CAP))]
    fn py_build(&self, pool: &CardPool, max_cards: usize) -> PyResult<LimitedBuild> {
        self.build(pool.clone(), max_cards).map_err(value_error)
    }

    #[pyo3(name = "issues")]
    fn py_issues(&self, pool: &CardPool, deck: &Bound<'_, PyAny>) -> PyResult<Vec<DeckIssue>> {
        Ok(self.issues(pool, &from_python(deck)?))
    }

    /// Copies that can still be added (int32 `[n, cards]`) and whether each
    /// deck can stop (bool `[n]`); an unreachable deck raises naming its row.
    #[pyo3(name = "available", signature = (pools, decks, max_cards=DEFAULT_BUILD_CAP))]
    fn py_available<'py>(
        &self,
        py: Python<'py>,
        pools: &Bound<'py, PyAny>,
        decks: &Bound<'py, PyAny>,
        max_cards: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (pools, decks) = paired_rows(pools, decks)?;
        let n = pools.len();
        let mut copies = Vec::with_capacity(n * card_count());
        let mut can_stop = Vec::with_capacity(n);
        for (r, (pool, deck)) in pools.iter().zip(&decks).enumerate() {
            let next = self
                .available(pool, deck, max_cards)
                .map_err(|e| value_error(format!("row {r}: {e}")))?;
            copies.extend(next.copies.iter().map(|&c| c as i32));
            can_stop.push(next.can_stop);
        }
        Ok(PyTuple::new_bound(
            py,
            [
                numpy_array(py, &copies, &[n, card_count()], "int32")?,
                numpy_bools(py, &can_stop, &[n])?,
            ],
        ))
    }

    /// Whether each deck passes the Limited rules, with no cap (bool `[n]`).
    #[pyo3(name = "legal")]
    fn py_legal(
        &self,
        py: Python<'_>,
        pools: &Bound<'_, PyAny>,
        decks: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let (pools, decks) = paired_rows(pools, decks)?;
        let legal = pools
            .iter()
            .zip(&decks)
            .enumerate()
            .map(|(r, (pool, deck))| {
                self.legal(pool, deck)
                    .map_err(|e| value_error(format!("row {r}: {e}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        numpy_bools(py, &legal, &[legal.len()])
    }

    #[getter]
    fn min_cards(&self) -> usize {
        LIMITED_MIN_CARDS
    }

    /// Cards a pool does not limit, bool `[cards]`.
    #[getter(basics)]
    fn py_basics(&self, py: Python<'_>) -> PyResult<PyObject> {
        numpy_bools(py, basics(), &[card_count()])
    }

    #[staticmethod]
    fn basic_cards() -> Vec<String> {
        basic_lands()
    }

    fn __repr__(&self) -> &'static str {
        "Limited()"
    }
}

#[pymethods]
impl LimitedBuild {
    #[pyo3(name = "select")]
    fn py_select(&mut self, card: &str) -> PyResult<()> {
        self.select(card).map_err(value_error)
    }

    /// Each card that can still be selected and its remaining copies.
    #[getter(available)]
    fn py_available(&self) -> BTreeMap<String, usize> {
        self.available_cards()
    }

    #[getter]
    fn can_stop(&self) -> bool {
        self.available().can_stop
    }

    #[getter(deck)]
    fn py_deck(&self) -> BTreeMap<String, usize> {
        self.deck()
    }

    fn __repr__(&self) -> String {
        format!("LimitedBuild(deck={:?})", self.deck())
    }
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DeckImportError", py.get_type_bound::<DeckImportError>())?;
    m.add(
        "MAX_DECKLIST_BYTES",
        crate::cards::decklist::MAX_SOURCE_BYTES,
    )?;
    m.add_class::<CardPool>()?;
    // `managym.format` mirrors `src/format/`: rules, not the cards available.
    let format = PyModule::new_bound(py, "format")?;
    format.add_class::<Limited>()?;
    format.add_class::<LimitedBuild>()?;
    m.add_submodule(&format)?;
    m.add_class::<Deck>()?;
    m.add_class::<DeckIssue>()?;
    m.add_class::<DeckEntry>()?;
    m.add_class::<Decklist>()?;
    m.add_class::<DeckImport>()?;
    m.add("DEVELOPMENT_RULESET", DEVELOPMENT_RULESET)?;
    // `PlayerConfig.world` selection of decks submitted to the world.
    m.add("WORLD_SETUP", crate::world::WORLD_POOL)?;
    m.add_function(wrap_pyfunction!(check_decks, m)?)?;
    m.add_function(wrap_pyfunction!(admit_decklists, m)?)?;
    Ok(())
}
