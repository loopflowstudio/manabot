use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
    time::Instant,
};

#[derive(Debug, Clone, PartialEq)]
pub enum InfoValue {
    String(String),
    Map(BTreeMap<String, InfoValue>),
    Int(i64),
    Float(f64),
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
struct TimingNode {
    stats: Stats,
    durations: Vec<f64>,
    max_samples: usize,
}

impl TimingNode {
    fn new(max_samples: usize) -> Self {
        Self {
            stats: Stats::default(),
            durations: Vec::new(),
            max_samples,
        }
    }

    fn push_duration(&mut self, duration: f64) {
        if self.max_samples == 0 {
            return;
        }
        self.durations.push(duration);
        if self.durations.len() > self.max_samples {
            let overflow = self.durations.len() - self.max_samples;
            self.durations.drain(0..overflow);
        }
    }
}

#[derive(Debug, Clone)]
struct ProfilerInner {
    max_samples: usize,
    stack: Vec<String>,
    nodes: BTreeMap<String, TimingNode>,
}

#[derive(Debug)]
pub struct Profiler {
    enabled: bool,
    inner: Mutex<ProfilerInner>,
}

impl Clone for Profiler {
    fn clone(&self) -> Self {
        let inner = self
            .inner
            .lock()
            .map(|inner| inner.clone())
            .unwrap_or_else(|_| ProfilerInner {
                max_samples: 0,
                stack: Vec::new(),
                nodes: BTreeMap::new(),
            });

        Self {
            enabled: self.enabled,
            inner: Mutex::new(inner),
        }
    }
}

impl Profiler {
    pub fn new(enabled: bool, max_entries: usize) -> Self {
        Self {
            enabled,
            inner: Mutex::new(ProfilerInner {
                max_samples: max_entries,
                stack: Vec::new(),
                nodes: BTreeMap::new(),
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn track(&self, name: &str) -> Scope<'_> {
        if !self.enabled || name.is_empty() {
            return Scope::disabled();
        }

        let Ok(mut inner) = self.inner.lock() else {
            return Scope::disabled();
        };

        let parent = inner.stack.last().cloned();
        let path = if let Some(parent_path) = &parent {
            format!("{parent_path}/{name}")
        } else {
            name.to_string()
        };

        let max_samples = inner.max_samples;
        inner
            .nodes
            .entry(path.clone())
            .or_insert_with(|| TimingNode::new(max_samples));
        inner.stack.push(path.clone());

        Scope {
            profiler: Some(self),
            path: Some(path),
            start: Instant::now(),
        }
    }

    pub fn get_stats(&self) -> BTreeMap<String, Stats> {
        let Ok(inner) = self.inner.lock() else {
            return BTreeMap::new();
        };

        inner
            .nodes
            .iter()
            .map(|(path, node)| (path.clone(), node.stats.clone()))
            .collect()
    }

    pub fn export_baseline(&self) -> String {
        let stats = self.get_stats();
        let mut out = String::new();
        for (path, stat) in stats {
            out.push_str(&format!("{path}\t{}\t{}\n", stat.total_time, stat.count));
        }
        out
    }

    pub fn parse_baseline(baseline: &str) -> BTreeMap<String, (f64, u64)> {
        let mut parsed = BTreeMap::new();
        for line in baseline.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut cols = line.split('\t');
            let Some(path) = cols.next() else {
                continue;
            };
            let Some(total_time) = cols.next().and_then(|v| v.parse::<f64>().ok()) else {
                continue;
            };
            let Some(count) = cols.next().and_then(|v| v.parse::<u64>().ok()) else {
                continue;
            };
            parsed.insert(path.to_string(), (total_time, count));
        }
        parsed
    }

    pub fn compare_to_baseline(&self, baseline: &str) -> String {
        let baseline_stats = Self::parse_baseline(baseline);
        let current_stats = self.get_stats();

        let mut keys = BTreeSet::new();
        keys.extend(baseline_stats.keys().cloned());
        keys.extend(current_stats.keys().cloned());

        let mut out = String::new();
        out.push_str("Profile Comparison (baseline vs current):\n");
        out.push_str(&format!(
            "{:<50} {:>12} {:>12} {:>10} {:>10}\n",
            "Path", "Baseline", "Current", "Change", "Count"
        ));
        out.push_str(&"-".repeat(94));
        out.push('\n');

        for key in keys {
            match (baseline_stats.get(&key), current_stats.get(&key)) {
                (Some((base_time, base_count)), Some(current)) => {
                    let pct = if *base_time > 0.0 {
                        ((current.total_time - *base_time) / *base_time) * 100.0
                    } else {
                        0.0
                    };
                    let change = if pct > 1.0 {
                        format!("+{pct:.1}%")
                    } else if pct < -1.0 {
                        format!("{pct:.1}%")
                    } else {
                        "~0%".to_string()
                    };
                    out.push_str(&format!(
                        "{:<50} {:>10.4}s {:>10.4}s {:>10} {:>10}\n",
                        key,
                        *base_time,
                        current.total_time,
                        change,
                        current.count as i64 - *base_count as i64
                    ));
                }
                (Some((base_time, _)), None) => {
                    out.push_str(&format!(
                        "{:<50} {:>10.4}s {:>12} {:>10} {:>10}\n",
                        key, *base_time, "(removed)", "-100%", "N/A"
                    ));
                }
                (None, Some(current)) => {
                    out.push_str(&format!(
                        "{:<50} {:>12} {:>10.4}s {:>10} {:>10}\n",
                        key, "(new)", current.total_time, "+NEW", current.count
                    ));
                }
                (None, None) => {}
            }
        }

        out
    }

    fn end_scope(&self, path: &str, elapsed: f64) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        if let Some(node) = inner.nodes.get_mut(path) {
            node.stats.total_time += elapsed;
            node.stats.count += 1;
            node.push_duration(elapsed);
        }

        if inner.stack.last().is_some_and(|current| current == path) {
            inner.stack.pop();
            return;
        }

        if let Some(i) = inner.stack.iter().rposition(|current| current == path) {
            inner.stack.remove(i);
        }
    }
}

pub struct Scope<'a> {
    profiler: Option<&'a Profiler>,
    path: Option<String>,
    start: Instant,
}

impl<'a> Scope<'a> {
    fn disabled() -> Self {
        Self {
            profiler: None,
            path: None,
            start: Instant::now(),
        }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        let (Some(profiler), Some(path)) = (self.profiler, self.path.as_deref()) else {
            return;
        };
        profiler.end_scope(path, self.start.elapsed().as_secs_f64());
    }
}
