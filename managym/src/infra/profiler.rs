use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfoValue {
    String(String),
    Map(BTreeMap<String, InfoValue>),
}

pub type InfoDict = BTreeMap<String, InfoValue>;

pub fn empty_info_dict() -> InfoDict {
    BTreeMap::new()
}

pub fn insert_info<S: Into<String>>(dict: &mut InfoDict, key: S, value: InfoValue) {
    dict.insert(key.into(), value);
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total_time: f64,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct Profiler {
    enabled: bool,
    _max_entries: usize,
}

impl Profiler {
    pub fn new(enabled: bool, max_entries: usize) -> Self {
        Self {
            enabled,
            _max_entries: max_entries,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn track(&self, _name: &str) -> Scope {
        Scope {}
    }

    pub fn get_stats(&self) -> BTreeMap<String, Stats> {
        BTreeMap::new()
    }
}

pub struct Scope {}

impl Drop for Scope {
    fn drop(&mut self) {}
}
