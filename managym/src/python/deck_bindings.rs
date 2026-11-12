//! Python surface of decks and decklist imports (`state::deck`,
//! `cards::decklist`). Records convert to and from their serde JSON form and
//! validate as Pydantic fields, so manabot configurations and Academy traces
//! hold the native record itself rather than a Python copy.
#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

use std::{collections::BTreeMap, path::PathBuf};

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyModule, PyType},
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
    format::limited::{basic_lands, CardPool, Limited, LimitedBuild, DEFAULT_BUILD_CAP},
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

#[pymethods]
impl CardPool {
    #[new]
    fn py_new(cards: &Bound<'_, PyDict>) -> PyResult<Self> {
        Self::new(card_counts(cards)?).map_err(value_error)
    }

    #[getter(cards)]
    fn py_cards(&self) -> BTreeMap<String, usize> {
        self.cards().clone()
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

    #[staticmethod]
    fn basic_cards() -> Vec<String> {
        basic_lands().into_iter().collect()
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

    #[getter]
    fn selectable(&self) -> Vec<String> {
        self.selection().cards
    }

    #[getter]
    fn can_stop(&self) -> bool {
        self.selection().can_stop
    }

    #[getter(deck)]
    fn py_deck(&self) -> BTreeMap<String, usize> {
        self.deck().clone()
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
