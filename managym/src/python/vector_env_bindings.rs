#![allow(clippy::useless_conversion)]
#![allow(unexpected_cfgs)]

#[cfg(feature = "python")]
use std::{cell::Cell, slice, sync::Mutex, time::Instant};

#[cfg(feature = "python")]
use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyModule},
};

#[cfg(feature = "python")]
use super::observation_batch::{self, ObservationBatch};
use crate::agent::observation_rows::PackedRows;
use crate::{
    agent::{
        action::AgentError,
        env::HandSnapshot,
        observation::Observation,
        observation_encoder::{
            encode, encode_into, encode_into_incremental, EncodedObservation,
            EncodedObservationMut, ObservationEncoderConfig,
        },
        rollout_pool::RolloutPool,
        vector_env::{
            ObservationRead, ObservationWriteError, SlotLoad, SlotLoadResult, SlotReplacement,
            VectorEnv,
        },
    },
    infra::profiler::InfoDict,
    python::{
        bindings::{map_agent_err, map_encode_err, PyEnv, PyObservation, PyPlayerConfig},
        convert::{info_dict_to_pydict, require_ndarray, require_numpy_array},
        layout_bindings::ObservationLayout,
        snapshot_bindings::{PyGameSnapshot, PyRestoreOptions, PySnapshotMetadata},
    },
    state::player::PlayerConfig,
};

#[cfg(feature = "python")]
type PyResetResult = (PyObservation, PyObject);
#[cfg(feature = "python")]
type PyStepResult = (PyObservation, f64, bool, bool, PyObject);

/// Deprecated dense transport buffer handles; removed in 0.3.0.
#[cfg(feature = "python")]
struct ObservationBuffers {
    relations: Option<Py<PyAny>>,
    effects: Option<Py<PyAny>>,
    effect_details: Option<Py<PyAny>>,
    effect_members: Option<Py<PyAny>>,
    name_candidates: Option<Py<PyAny>>,
    facts: Option<Py<PyAny>>,
    semantic_cards: Py<PyAny>,
    known_hand: Py<PyAny>,
    known_library: Option<Py<PyAny>>,
    decklists: Option<Py<PyAny>>,
    player_turn_state: Option<Py<PyAny>>,
    card_turn_state: Option<Py<PyAny>>,
    stack: Option<Py<PyAny>>,
    stack_valid: Option<Py<PyAny>>,
    agent_player: Py<PyAny>,
    opponent_player: Py<PyAny>,
    agent_cards: Py<PyAny>,
    opponent_cards: Py<PyAny>,
    agent_permanents: Py<PyAny>,
    opponent_permanents: Py<PyAny>,
    actions: Py<PyAny>,
    events: Py<PyAny>,
    decision_records: Option<Py<PyAny>>,
    decision_reference_details: Option<Py<PyAny>>,
    decision_references: Option<Py<PyAny>>,
    decision_members: Option<Py<PyAny>>,
    action_focus: Py<PyAny>,
    agent_player_valid: Py<PyAny>,
    opponent_player_valid: Py<PyAny>,
    agent_cards_valid: Py<PyAny>,
    opponent_cards_valid: Py<PyAny>,
    agent_permanents_valid: Py<PyAny>,
    opponent_permanents_valid: Py<PyAny>,
    actions_valid: Py<PyAny>,
    events_valid: Py<PyAny>,
    // Step outcomes; absent for observation-only writes.
    step_state: Option<StepStateBuffers>,
}

#[cfg(feature = "python")]
struct StepStateBuffers {
    shape: Vec<usize>,
    rewards: Py<PyAny>,
    terminated: Py<PyAny>,
    truncated: Py<PyAny>,
}

#[cfg(feature = "python")]
#[pyclass(name = "VectorEnv")]
pub struct PyVectorEnv {
    inner: VectorEnv,
    config: ObservationEncoderConfig,
    num_envs: usize,
    buffers: Option<ObservationBuffers>,
    packed: Option<PackedVector>,
    last_info: Vec<InfoDict>,
    timings: BufferedCallTimings,
    memory_encoding: Vec<Mutex<crate::agent::decision_memory::MemoryEncoding>>,
}

/// Published slot coordinates survive masked calls. Each successful call freezes
/// its writes together; replaced slots drop their previous batch reference.
struct PackedVector {
    layout: Py<ObservationLayout>,
    state: StepStateBuffers,
    slots: Vec<Option<(Py<ObservationBatch>, usize)>>,
}

struct PackedWrites {
    rows: PackedRows,
    slots: Vec<usize>,
}

impl PackedWrites {
    fn new(layout: &ObservationLayout) -> Self {
        Self {
            rows: PackedRows::new(&layout.arrays),
            slots: Vec::new(),
        }
    }

    fn append(
        &mut self,
        slot: usize,
        obs: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationWriteError> {
        self.rows.append(obs, config)?;
        self.slots.push(slot);
        Ok(())
    }
}

#[pyclass(name = "SlotLoad", frozen)]
#[derive(Clone)]
struct PySlotLoad(SlotLoad);

#[pymethods]
impl PySlotLoad {
    #[getter]
    fn slot(&self) -> usize {
        self.0.slot()
    }

    fn with_slot(&self, slot: usize) -> Self {
        Self(self.0.with_slot(slot))
    }

    /// A fresh game using the setup already configured for this slot.
    #[getter]
    fn uses_slot_setup(&self) -> bool {
        matches!(self.0, SlotLoad::Fresh(_, None))
    }

    /// Detached replay authority for a recording wrapper, including the intervention.
    #[getter]
    fn branch_root(&self) -> Option<(PyGameSnapshot, PyRestoreOptions)> {
        match &self.0 {
            SlotLoad::Fresh(..) => None,
            SlotLoad::Snapshot(_, source, options) => {
                Some((PyGameSnapshot(source.clone()), PyRestoreOptions(*options)))
            }
        }
    }

    #[staticmethod]
    #[pyo3(signature = (slot, player_configs=None))]
    fn fresh(slot: usize, player_configs: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let setup = player_configs
            .as_ref()
            .map(super::bindings::game_setup)
            .transpose()?;
        Ok(Self(SlotLoad::Fresh(slot, setup)))
    }

    #[staticmethod]
    #[pyo3(signature = (slot, snapshot, options=None))]
    fn snapshot(
        slot: usize,
        snapshot: &PyGameSnapshot,
        options: Option<&PyRestoreOptions>,
    ) -> Self {
        Self(SlotLoad::Snapshot(
            slot,
            snapshot.0.clone(),
            options.map(|o| o.0).unwrap_or_default(),
        ))
    }
}

#[pymethods]
impl SlotLoadResult {
    fn with_slot(&self, slot: usize) -> Self {
        Self {
            slot,
            ..self.clone()
        }
    }
}

/// Cumulative wall time inside buffered calls: every step or reset, split into
/// stepping (all work in the call except encoding) and observation encoding.
#[cfg(feature = "python")]
#[derive(Default)]
struct BufferedCallTimings {
    stepping_nanos: u64,
    encoding_nanos: u64,
}

// A buffered call stays on its calling thread while the GIL is released, and each
// stream's row is encoded inside it, so a thread-local counter needs no locks.
#[cfg(feature = "python")]
thread_local! {
    static ENCODING_NANOS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVectorEnv {
    #[cfg(feature = "benchmark")]
    fn _benchmark_allocations_start(&self) {
        crate::allocator::start();
    }

    #[cfg(feature = "benchmark")]
    fn _benchmark_allocations_stop(&self) -> (u64, u64) {
        crate::allocator::stop()
    }

    #[new]
    #[pyo3(signature = (num_envs, seed=0, skip_trivial=true, seed_stride=None, *, player_policies=None, decision_memory_include_own=false, decision_memory=false))]
    fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        seed_stride: Option<u64>,
        player_policies: Option<Vec<String>>,
        decision_memory_include_own: bool,
        decision_memory: bool,
    ) -> PyResult<Self> {
        let policies = parse_player_policies(player_policies)?;
        let mut inner = VectorEnv::new(num_envs, seed, skip_trivial, policies)
            .with_decision_memory(decision_memory)
            .with_decision_memory_include_own(decision_memory_include_own);
        if let Some(stride) = seed_stride {
            inner = inner.with_seed_stride(stride).map_err(map_agent_err)?;
        }

        Ok(Self {
            inner,
            config: ObservationEncoderConfig::default(),
            num_envs,
            memory_encoding: (0..num_envs).map(|_| Mutex::default()).collect(),
            buffers: None,
            packed: None,
            last_info: Vec::new(),
            timings: BufferedCallTimings::default(),
        })
    }

    fn reset_all(
        &mut self,
        py: Python<'_>,
        player_configs: Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyResetResult>> {
        let configs = super::bindings::game_setup(&player_configs)?;
        let results = self.inner.reset_all(configs).map_err(map_agent_err)?;
        self.last_info = results.iter().map(|(_, info)| info.clone()).collect();

        Ok(results
            .into_iter()
            .map(|(obs, info)| {
                let py_dict = info_dict_to_pydict(py, &info);
                (PyObservation::from(obs), py_dict.into_any().unbind())
            })
            .collect())
    }

    fn step(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<Vec<PyStepResult>> {
        let results = self.inner.step(&actions).map_err(map_agent_err)?;
        self.last_info = results.iter().map(|result| result.info.clone()).collect();

        Ok(results
            .into_iter()
            .map(|result| {
                let py_dict = info_dict_to_pydict(py, &result.info);
                (
                    PyObservation::from(result.obs),
                    result.reward,
                    result.terminated,
                    result.truncated,
                    py_dict.into_any().unbind(),
                )
            })
            .collect())
    }

    /// Advance all streams and return the exact emitted observations (including
    /// step events), rewards, termination and truncation in matching row order.
    #[pyo3(signature = (actions, layout, action_overflow="error"))]
    fn step_observation_batch(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
        layout: Py<ObservationLayout>,
        action_overflow: &str,
    ) -> PyResult<(
        super::observation_batch::ObservationBatch,
        Vec<f64>,
        Vec<bool>,
        Vec<bool>,
    )> {
        let config = super::observation_batch::config(&layout.borrow(py), action_overflow)?;
        let mut rows = PackedRows::new(&layout.borrow(py).arrays);
        let (mut rewards, mut terminated, mut truncated) = (Vec::new(), Vec::new(), Vec::new());
        self.last_info = py
            .allow_threads(|| {
                self.inner
                    .step_with_writer(&actions, None, |_, obs, reward, done, cut| {
                        rows.append(obs, &config)?;
                        rewards.push(reward);
                        terminated.push(done);
                        truncated.push(cut);
                        Ok(())
                    })
            })
            .map_err(map_write_error)?;
        let batch = observation_batch::freeze(py, layout, rows, rewards.len());
        Ok((batch, rewards, terminated, truncated))
    }

    /// Register step outcomes and select owned v11 observations for buffered calls.
    #[pyo3(signature = (state, layout, action_overflow="error"))]
    fn set_observation_layout(
        &mut self,
        py: Python<'_>,
        state: Bound<'_, PyDict>,
        layout: Py<ObservationLayout>,
        action_overflow: &str,
    ) -> PyResult<()> {
        let config = observation_batch::config(&layout.borrow(py), action_overflow)?;
        if state.len() != 3 {
            return Err(PyValueError::new_err("expected three step outcome buffers"));
        }
        let state = read_step_state(&state, &[self.num_envs])?;
        self.packed = Some(PackedVector {
            layout,
            state,
            slots: (0..self.num_envs).map(|_| None).collect(),
        });
        self.config = config;
        self.buffers = None;
        Ok(())
    }

    fn observation_rows(&self, py: Python<'_>) -> PyResult<Vec<(Py<ObservationBatch>, usize)>> {
        self.packed
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no registered observation layout"))?
            .slots
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|(batch, i)| (batch.clone_ref(py), *i))
                    .ok_or_else(|| PyValueError::new_err("vector has no encoded observation"))
            })
            .collect()
    }

    fn observations_for_players(
        &mut self,
        py: Python<'_>,
        players: Vec<Option<usize>>,
    ) -> PyResult<(Vec<ObservationRead>, ObservationBatch)> {
        let packed = self
            .packed
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no registered observation layout"))?;
        let mut rows = PackedRows::new(&packed.layout.borrow(py).arrays);
        let config = self.config;
        let reads = self
            .inner
            .read_observations_for_players(&players, |_, obs| {
                rows.append(obs, &config)
                    .map_err(ObservationWriteError::from)
            })
            .map_err(map_write_error)?;
        let batch = observation_batch::freeze(py, packed.layout.clone_ref(py), rows, reads.len());
        Ok((reads, batch))
    }

    #[pyo3(signature = (buffers, action_overflow="error", *, layout=None))]
    fn set_buffers(
        &mut self,
        buffers: Bound<'_, PyDict>,
        action_overflow: &str,
        layout: Option<&ObservationLayout>,
    ) -> PyResult<()> {
        let inferred = layout.is_none();
        if !inferred {
            super::bindings::warn_deprecated(
                buffers.py(),
                "VectorEnv.set_buffers is deprecated; use VectorEnv.set_observation_layout for v11 observations; removal in 0.3.0",
            )?;
        }
        let layout = registered_layout(&buffers, layout)?;
        if inferred
            && self.inner.decision_memory_enabled() != (layout.config.max_decision_records > 0)
        {
            return Err(PyValueError::new_err(
                "decision memory buffers differ from vector configuration",
            ));
        }
        let mut config = layout.config;
        config.action_overflow = action_overflow.parse().map_err(PyValueError::new_err)?;
        let buffers = read_observation_buffers(&buffers, Some(self.num_envs), &layout, true)?;
        self.config = config;
        self.memory_encoding = (0..self.num_envs).map(|_| Mutex::default()).collect();
        self.buffers = Some(buffers);
        self.packed = None;
        Ok(())
    }

    /// Scenario harness only; does not change deck admission or future resets.
    fn scenario_naming_vocabulary_into_buffers(&mut self, py: Python<'_>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.scenario_naming_vocabulary_into(|index, obs| {
                write_buffers
                    .write_encoded_row(index, obs, 0.0, false, false, &config)
                    .map_err(AgentError::from)
            })?;
            Ok(vec![InfoDict::new(); inner.len()])
        })
    }

    fn reset_all_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let configs = super::bindings::game_setup(&player_configs)?;
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_with_writer(
                vec![configs; inner.len()],
                None,
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                },
            )
        })
    }

    #[pyo3(signature = (player_configs, *, seeds=None))]
    fn reset_each_into_buffers(
        &mut self,
        py: Python<'_>,
        player_configs: Vec<Bound<'_, PyAny>>,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<()> {
        let configs = player_configs
            .into_iter()
            .map(|row| super::bindings::game_setup(&row))
            .collect::<PyResult<Vec<_>>>()?;
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_with_writer(
                configs,
                seeds,
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                },
            )
        })
    }

    fn next_fresh_seeds(&self) -> Vec<u64> {
        self.inner.next_fresh_seeds().to_vec()
    }

    fn snapshot_slots(
        &mut self,
        py: Python<'_>,
        mask: Vec<bool>,
    ) -> PyResult<Vec<Option<PyGameSnapshot>>> {
        let inner = &mut self.inner;
        py.allow_threads(move || inner.snapshot_slots(&mask))
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.map(PyGameSnapshot))
                    .collect()
            })
            .map_err(map_agent_err)
    }

    fn slot_metadata(
        &mut self,
        py: Python<'_>,
        mask: Vec<bool>,
    ) -> PyResult<Vec<Option<PySnapshotMetadata>>> {
        let inner = &mut self.inner;
        py.allow_threads(move || inner.slot_metadata(&mask))
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.map(PySnapshotMetadata))
                    .collect()
            })
            .map_err(map_agent_err)
    }

    fn load_slots_into_buffers(
        &mut self,
        py: Python<'_>,
        requests: Vec<PySlotLoad>,
    ) -> PyResult<Vec<SlotLoadResult>> {
        let prepared = self.prepare_slot_buffers(py, requests)?;
        self.publish_slot_buffers(py, prepared)
    }

    /// Keep every vector exclusively borrowed across both phases of a sharded load.
    #[staticmethod]
    fn _load_shard_buffers(
        py: Python<'_>,
        batches: Vec<(PyRefMut<'_, PyVectorEnv>, Vec<PySlotLoad>)>,
    ) -> PyResult<Vec<Vec<SlotLoadResult>>> {
        let mut admitted = Vec::with_capacity(batches.len());
        for (mut vector, requests) in batches {
            let replacements = vector.prepare_slot_buffers(py, requests)?;
            admitted.push((vector, replacements));
        }
        admitted
            .into_iter()
            .map(|(mut vector, replacements)| vector.publish_slot_buffers(py, replacements))
            .collect()
    }

    #[pyo3(signature = (actions, active=None))]
    fn step_into_buffers(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
        active: Option<Vec<bool>>,
    ) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            if let Some(mask) = &active {
                if mask.len() != inner.len() {
                    return Err(
                        AgentError("active mask length must match environments".into()).into(),
                    );
                }
                for (index, enabled) in mask.iter().enumerate() {
                    if !enabled {
                        write_buffers.write_step_state(index, 0.0, false, false)?;
                    }
                }
            }
            let infos = inner.step_with_writer(
                &actions,
                active.as_deref(),
                |env_index, obs, reward, terminated, truncated| {
                    write_buffers
                        .write_encoded_row(env_index, obs, reward, terminated, truncated, &config)
                },
            )?;
            for slot in inner.parked_slots() {
                write_buffers.clear_memory_cursor(slot);
            }
            Ok(infos)
        })
    }

    /// Per-env `skip_trivial` collapse counters for the current games.
    /// Each counter resets when its env's game resets.
    fn skip_trivial_counts(&self) -> Vec<usize> {
        self.inner.skip_trivial_counts()
    }

    /// Per-env index of the player holding the current decision (-1 if no
    /// decision is pending).
    fn current_agent_indices(&self) -> Vec<i64> {
        self.inner.current_agent_indices()
    }

    fn set_uniform_roster(&mut self, roster: Vec<PyPlayerConfig>) -> PyResult<()> {
        self.inner
            .set_uniform_roster(roster.into_iter().map(PlayerConfig::from).collect())
            .map_err(map_agent_err)
    }

    #[pyo3(signature = (decks, distinct=true))]
    fn set_training_decks(&mut self, decks: Vec<PyPlayerConfig>, distinct: bool) -> PyResult<()> {
        self.inner
            .set_training_decks(
                decks.into_iter().map(PlayerConfig::from).collect(),
                distinct,
            )
            .map_err(map_agent_err)
    }

    fn current_player_configs(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .current_player_configs()
            .into_iter()
            .zip(self.inner.current_decklists())
            .zip(self.inner.current_mana_floats())
            .zip(self.inner.current_mana_payment())
            .map(|(((players, disclosure), taps), payment)| {
                super::bindings::python_game_setup(py, players, disclosure, taps, payment)
            })
            .collect()
    }

    fn current_matchup_indices(&self) -> Vec<(usize, usize)> {
        self.inner.current_matchup_indices()
    }

    fn enable_profiler(&mut self) {
        self.inner.enable_profiler();
    }

    fn record_games(&mut self, active: Vec<bool>) -> PyResult<()> {
        self.inner.record_games(&active).map_err(map_agent_err)
    }

    #[pyo3(signature = (active=None))]
    fn state_digests(&self, active: Option<Vec<bool>>) -> PyResult<Vec<Option<String>>> {
        self.inner
            .state_digests(active.as_deref())
            .map_err(map_agent_err)
    }

    fn current_game_seeds(&self) -> Vec<u64> {
        self.inner.current_game_seeds()
    }

    fn decision_memory_cursors(&self) -> PyResult<Vec<[[u64; 5]; 2]>> {
        self.inner.decision_memory_cursors().map_err(map_agent_err)
    }

    fn decision_memory_counts(&self) -> PyResult<Vec<[[usize; 4]; 2]>> {
        self.inner.decision_memory_counts().map_err(map_agent_err)
    }

    fn current_game_generations(&self) -> Vec<u64> {
        self.inner.current_game_generations()
    }

    /// Caller-owned observation-only buffers; registered execution buffers and
    /// diagnostics are untouched. Keep the GIL held throughout the read/write.
    #[pyo3(signature = (players, buffers, action_overflow="error", *, layout))]
    fn write_observations_for_players(
        &self,
        py: Python<'_>,
        players: Vec<Option<usize>>,
        buffers: Bound<'_, PyDict>,
        action_overflow: &str,
        layout: &ObservationLayout,
    ) -> PyResult<Vec<ObservationRead>> {
        let mut config = layout.config;
        config.action_overflow = action_overflow.parse().map_err(PyValueError::new_err)?;
        let buffers = read_observation_buffers(&buffers, Some(self.num_envs), layout, false)?;
        with_observation_field_slices(py, &buffers, |out| {
            self.inner
                .write_observations_for_players(&players, |slot, observation| {
                    let row = out.encoded_row_mut(slot, &config)?;
                    encode_into(observation, &config, row)
                        .map_err(|err| AgentError(err.to_string()))
                })
                .map_err(map_agent_err)
        })
    }

    fn hand_snapshots(&self, players: Vec<usize>) -> PyResult<Vec<HandSnapshot>> {
        self.inner.hand_snapshots(&players).map_err(map_agent_err)
    }

    fn current_action_summaries(&self) -> Vec<(usize, String)> {
        self.inner.current_action_summaries()
    }

    #[pyo3(signature = (active=None))]
    fn semantic_decision_frames_json(
        &self,
        active: Option<Vec<bool>>,
    ) -> PyResult<Vec<Option<String>>> {
        self.inner
            .semantic_decision_frames_json(active.as_deref())
            .map_err(map_agent_err)
    }

    #[pyo3(signature = (actions, active, recover_capacity=false))]
    fn step_isolated_into_buffers(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
        active: Vec<bool>,
        recover_capacity: bool,
    ) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, buffers, config| {
            // Never leave a preceding terminal flag on a failed/inactive row.
            for index in 0..inner.len() {
                buffers.write_step_state(index, 0.0, false, false)?;
            }
            let infos = inner.step_isolated_with_writer(
                &actions,
                &active,
                recover_capacity,
                |index, obs, reward, done, truncated| {
                    buffers.write_encoded_row(index, obs, reward, done, truncated, &config)
                },
            )?;
            for slot in inner.parked_slots() {
                buffers.clear_memory_cursor(slot);
            }
            Ok(infos)
        })
    }

    fn reset_seeds_into_buffers(&mut self, py: Python<'_>, seeds: Vec<u64>) -> PyResult<()> {
        self.run_into_buffers(py, move |inner, write_buffers, config| {
            inner.reset_seeds_with_writer(seeds, |index, obs, reward, terminated, truncated| {
                write_buffers.write_encoded_row(index, obs, reward, terminated, truncated, &config)
            })
        })
    }

    fn enable_statistics(&mut self) -> PyResult<()> {
        self.inner.enable_statistics().map_err(map_agent_err)
    }

    fn statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        super::bindings::statistics_object(py, &self.inner.statistics(), "summaries")
    }

    fn take_statistics(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        super::bindings::statistics_object(py, &self.inner.take_statistics(), "summaries")
    }

    fn enable_exposure_capture(&mut self) -> PyResult<()> {
        self.inner.enable_exposure_capture().map_err(map_agent_err)
    }

    fn take_exposure(&mut self) -> Vec<Vec<String>> {
        self.inner.take_exposure()
    }

    fn decision_observations(&self, active: Vec<bool>) -> PyResult<Vec<Option<PyObservation>>> {
        self.inner
            .decision_observations(&active)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.map(PyObservation::from))
                    .collect()
            })
            .map_err(map_agent_err)
    }

    #[pyo3(signature = (active=None))]
    fn current_exposure(&self, active: Option<Vec<bool>>) -> PyResult<Vec<String>> {
        self.inner
            .selected_exposure(active.as_deref())
            .map_err(map_agent_err)
    }

    fn get_last_info(&self, py: Python<'_>) -> Vec<PyObject> {
        self.last_info
            .iter()
            .map(|info| info_dict_to_pydict(py, info).into_any().unbind())
            .collect()
    }

    /// Cumulative `(stepping_seconds, encoding_seconds)` of buffered calls.
    fn buffered_call_timings(&self) -> (f64, f64) {
        (
            self.timings.stepping_nanos as f64 / 1e9,
            self.timings.encoding_nanos as f64 / 1e9,
        )
    }
}

#[cfg(feature = "python")]
fn registered_layout<'a>(
    buffers: &Bound<'_, PyDict>,
    layout: Option<&'a ObservationLayout>,
) -> PyResult<std::borrow::Cow<'a, ObservationLayout>> {
    if let Some(layout) = layout {
        return Ok(std::borrow::Cow::Borrowed(layout));
    }
    super::bindings::warn_deprecated(
        buffers.py(),
        "set_buffers without layout is deprecated; pass layout=...; removal in 0.3.0",
    )?;
    Ok(std::borrow::Cow::Owned(ObservationLayout::from_config(
        input_buffer_config(buffers)?,
    )?))
}

/// Recover the input configuration of pre-layout receipts and deprecated calls.
/// Removed in 0.3.0.
#[cfg(feature = "python")]
pub(super) fn input_buffer_config(
    buffers: &Bound<'_, PyDict>,
) -> PyResult<ObservationEncoderConfig> {
    // Strip only the optional batch axis. Full companion and batch shapes are
    // validated by the caller before any writes.
    let shape = |name: &str| -> PyResult<[usize; 2]> {
        let value = buffers
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
        require_ndarray(name, &value)?;
        let dimensions: Vec<usize> = value.getattr("shape")?.extract()?;
        match dimensions.as_slice() {
            &[rows, width] | &[_, rows, width] if !dimensions.contains(&0) => Ok([rows, width]),
            _ => Err(PyValueError::new_err(format!(
                "invalid {name} buffer shape"
            ))),
        }
    };
    let [max_events, event_width] = shape("events")?;
    let policy_history = match event_width {
        crate::agent::observation_encoder::EVENT_DIM => false,
        crate::agent::policy_history::HISTORY_DIM => true,
        _ => return Err(PyValueError::new_err("unsupported event buffer width")),
    };
    let optional_rows = |name: &str| -> PyResult<usize> {
        if buffers.contains(name)? {
            Ok(shape(name)?[0])
        } else {
            Ok(0)
        }
    };
    let stack = buffers.contains("stack")?;
    Ok(ObservationEncoderConfig {
        layout: if stack {
            crate::agent::observation_encoder::InputLayout::OneToken
        } else {
            crate::agent::observation_encoder::InputLayout::Legacy
        },
        turn_state: stack || buffers.contains("player_turn_state")?,
        max_stack_objects: optional_rows("stack")?,
        max_cards_per_player: shape("agent_cards")?[0],
        max_permanents_per_player: shape("agent_permanents")?[0],
        max_actions: shape("actions")?[0],
        max_focus_objects: shape("action_focus")?[1],
        max_facts: shape("facts")?[0],
        max_events,
        policy_history,
        max_decision_records: optional_rows("decision_records")?,
        max_decision_references: optional_rows("decision_references")?,
        max_decision_reference_details: optional_rows("decision_reference_details")?,
        max_decision_members: optional_rows("decision_members")?,
        ..Default::default()
    })
}

/// Check one observation's caller-owned buffers (no batch axis) against the
/// whole contract of `config`: every array present, with its shape, dtype,
/// C-contiguity and writability. Nothing is written; the scalar encoder runs
/// this before encoding, so a malformed buffer set is never reported as a
/// game that outgrew its capacity.
#[cfg(feature = "python")]
pub(crate) fn check_observation_buffers(
    buffers: &Bound<'_, PyDict>,
    layout: &ObservationLayout,
) -> PyResult<()> {
    read_observation_buffers(buffers, None, layout, false).map(|_| ())
}

#[cfg(feature = "python")]
fn read_observation_buffers(
    buffers: &Bound<'_, PyDict>,
    batch: Option<usize>,
    layout: &ObservationLayout,
    step_state: bool,
) -> PyResult<ObservationBuffers> {
    for key in buffers.keys().iter() {
        let name = key
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("unexpected non-string buffer key"))?;
        if !(layout.arrays.contains_key(&name)
            || step_state && matches!(name.as_str(), "rewards" | "terminated" | "truncated"))
        {
            return Err(PyValueError::new_err(format!("unexpected buffer '{name}'")));
        }
    }
    let array = |name: &str| {
        let a = &layout.arrays[name];
        let shape: Vec<_> = batch
            .iter()
            .chain(a.shape.iter().flatten())
            .copied()
            .collect();
        Ok::<_, PyErr>(require_numpy_array(buffers, name, &shape, a.dtype)?.unbind())
    };
    let optional = |name: &str| {
        layout
            .arrays
            .contains_key(name)
            .then(|| array(name))
            .transpose()
    };
    Ok(ObservationBuffers {
        relations: optional("relations")?,
        effects: optional("effects")?,
        effect_details: optional("effect_details")?,
        effect_members: optional("effect_members")?,
        name_candidates: optional("name_candidates")?,
        facts: optional("facts")?,
        semantic_cards: array("semantic_cards")?,
        known_hand: array("known_hand")?,
        known_library: optional("known_library")?,
        decklists: optional("decklists")?,
        player_turn_state: optional("player_turn_state")?,
        card_turn_state: optional("card_turn_state")?,
        stack: optional("stack")?,
        stack_valid: optional("stack_valid")?,
        agent_player: array("agent_player")?,
        opponent_player: array("opponent_player")?,
        agent_cards: array("agent_cards")?,
        opponent_cards: array("opponent_cards")?,
        agent_permanents: array("agent_permanents")?,
        opponent_permanents: array("opponent_permanents")?,
        actions: array("actions")?,
        events: array("events")?,
        decision_records: optional("decision_records")?,
        decision_reference_details: optional("decision_reference_details")?,
        decision_references: optional("decision_references")?,
        decision_members: optional("decision_members")?,
        action_focus: array("action_focus")?,
        agent_player_valid: array("agent_player_valid")?,
        opponent_player_valid: array("opponent_player_valid")?,
        agent_cards_valid: array("agent_cards_valid")?,
        opponent_cards_valid: array("opponent_cards_valid")?,
        agent_permanents_valid: array("agent_permanents_valid")?,
        opponent_permanents_valid: array("opponent_permanents_valid")?,
        actions_valid: array("actions_valid")?,
        events_valid: array("events_valid")?,
        step_state: if step_state {
            Some(read_step_state(buffers, batch.as_slice())?)
        } else {
            None
        },
    })
}

fn read_step_state(buffers: &Bound<'_, PyDict>, shape: &[usize]) -> PyResult<StepStateBuffers> {
    Ok(StepStateBuffers {
        shape: shape.to_vec(),
        rewards: require_numpy_array(buffers, "rewards", shape, "float64")?.unbind(),
        terminated: require_numpy_array(buffers, "terminated", shape, "uint8")?.unbind(),
        truncated: require_numpy_array(buffers, "truncated", shape, "uint8")?.unbind(),
    })
}

fn map_write_error(error: ObservationWriteError) -> PyErr {
    match error {
        ObservationWriteError::Encoding(error) => map_encode_err(error),
        ObservationWriteError::Buffer(error) => map_agent_err(error),
    }
}

#[cfg(feature = "python")]
impl PyVectorEnv {
    fn prepare_slot_buffers(
        &mut self,
        py: Python<'_>,
        requests: Vec<PySlotLoad>,
    ) -> PyResult<Vec<SlotReplacement>> {
        let requests: Vec<_> = requests.into_iter().map(|request| request.0).collect();
        self.with_buffers(py, move |inner, buffers, config| {
            let mut admission: Option<EncodedObservation> = None;
            inner
                .prepare_slots(&requests, |slot, obs| {
                    // Owners may have resized a registered array between calls.
                    // Check every selected destination before publishing any of them.
                    // Packed rows are appended only when published, so a rejected
                    // load (including a sibling shard's) leaves no frozen batch.
                    if let Destination::Dense(fields) = &buffers.fields {
                        fields.encoded_row_mut(slot, &config)?;
                    }
                    buffers.rewards.element_ptr(slot)?;
                    buffers.terminated.element_ptr(slot)?;
                    buffers.truncated.element_ptr(slot)?;
                    if let Some(row) = &mut admission {
                        encode_into(obs, &config, row.buffers_mut())?;
                    } else {
                        admission = Some(encode(obs, &config)?);
                    }
                    Ok(())
                })
                .map_err(map_write_error)
        })
    }

    fn publish_slot_buffers(
        &mut self,
        py: Python<'_>,
        replacements: Vec<SlotReplacement>,
    ) -> PyResult<Vec<SlotLoadResult>> {
        let loaded = self.with_buffers(py, move |inner, buffers, config| {
            inner
                .publish_slots(replacements, |slot, obs, terminal| {
                    buffers.clear_memory_cursor(slot);
                    buffers.write_encoded_row(slot, obs, 0.0, terminal, false, &config)?;
                    if terminal {
                        buffers.clear_memory_cursor(slot);
                    }
                    Ok(())
                })
                .map_err(map_write_error)
        })?;
        self.last_info.resize_with(self.num_envs, InfoDict::new);
        Ok(loaded
            .into_iter()
            .map(|(result, info)| {
                self.last_info[result.slot] = info;
                result
            })
            .collect())
    }

    fn run_into_buffers(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut VectorEnv,
                SendWriteBuffers,
                ObservationEncoderConfig,
            ) -> Result<Vec<InfoDict>, ObservationWriteError>
            + Send,
    ) -> PyResult<()> {
        let packed = self.packed.is_some();
        self.last_info = self.with_buffers(py, |inner, buffers, config| {
            run(inner, buffers, config).map_err(|error| {
                if packed {
                    map_write_error(error)
                } else {
                    map_agent_err(error.into())
                }
            })
        })?;
        Ok(())
    }

    fn with_buffers<R: Send>(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(&mut VectorEnv, SendWriteBuffers, ObservationEncoderConfig) -> PyResult<R>
            + Send,
    ) -> PyResult<R> {
        let config = self.config;
        let inner = &mut self.inner;
        let started = Instant::now();
        ENCODING_NANOS.set(0);
        let result = if let Some(packed) = &mut self.packed {
            let writer = Mutex::new(PackedWrites::new(&packed.layout.borrow(py)));
            let result =
                with_step_writer(py, &packed.state, Destination::Packed(&writer), |buffers| {
                    py.allow_threads(move || run(inner, buffers, config))
                });
            let writes = writer.into_inner().unwrap();
            if result.is_ok() {
                let batch = Py::new(
                    py,
                    observation_batch::freeze(
                        py,
                        packed.layout.clone_ref(py),
                        writes.rows,
                        writes.slots.len(),
                    ),
                )?;
                for (index, slot) in writes.slots.into_iter().enumerate() {
                    packed.slots[slot] = Some((batch.clone_ref(py), index));
                }
            }
            result
        } else {
            let buffers = self
                .buffers
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;
            let memory_encoding = &self.memory_encoding;
            with_send_write_buffers(py, buffers, |write_buffers| {
                py.allow_threads(|| {
                    run(
                        inner,
                        SendWriteBuffers {
                            memory_encoding: Some(memory_encoding),
                            ..write_buffers
                        },
                        config,
                    )
                })
            })
        };
        let encoding = ENCODING_NANOS.get();
        self.timings.encoding_nanos += encoding;
        self.timings.stepping_nanos +=
            (started.elapsed().as_nanos() as u64).saturating_sub(encoding);
        result
    }
}

/// Batched policy-rollout pool for one flat-MC search decision.
///
/// Created via `Env.rollout_pool(...)`. All simulations of the decision run
/// simultaneously: `encode_active()` / `step_active(actions)` write each
/// active simulation's encoded observation into caller-owned numpy buffers
/// (same layout as `VectorEnv.set_buffers`, leading dim = capacity) and
/// return the active slot indices, so a policy network can pick every
/// rollout action with one batched forward pass per step.
#[cfg(feature = "python")]
#[pyclass(name = "RolloutPool")]
pub struct PyRolloutPool {
    inner: RolloutPool,
    config: ObservationEncoderConfig,
    buffers: Option<ObservationBuffers>,
    packed_layout: Option<Py<ObservationLayout>>,
}

#[cfg(feature = "python")]
impl PyRolloutPool {
    pub fn from_inner(inner: RolloutPool) -> Self {
        Self {
            inner,
            config: ObservationEncoderConfig::default(),
            buffers: None,
            packed_layout: None,
        }
    }

    fn run_with_buffers<R: Send>(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut RolloutPool,
                SendWriteBuffers,
                ObservationEncoderConfig,
            ) -> Result<R, AgentError>
            + Send,
    ) -> PyResult<R> {
        let buffers = self
            .buffers
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("set_buffers() must be called first"))?;
        let config = self.config;
        let inner = &mut self.inner;

        let result = with_send_write_buffers(py, &buffers, |write_buffers| {
            py.allow_threads(|| run(inner, write_buffers, config))
                .map_err(map_agent_err)
        });
        self.buffers = Some(buffers);
        result
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRolloutPool {
    #[getter]
    fn num_slots(&self) -> usize {
        self.inner.num_slots()
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    #[getter]
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Index of the deciding player the pool scores for.
    #[getter]
    fn hero_index(&self) -> usize {
        self.inner.hero_index()
    }

    /// Acting player index per slot (-1 for slots no longer active). The
    /// encoded observation of an active slot is from this player's
    /// perspective — value-at-leaf scoring flips V for non-hero actors.
    fn acting_players(&self) -> Vec<i64> {
        self.inner.acting_players()
    }

    /// Root action index per slot (layout is (world, action, rollout)).
    fn root_actions(&self) -> Vec<usize> {
        self.inner.root_actions()
    }

    /// Register the v11 observation layout for typed CSR batch rollouts.
    #[pyo3(signature = (layout, action_overflow="error"))]
    fn set_observation_layout(
        &mut self,
        py: Python<'_>,
        layout: Py<ObservationLayout>,
        action_overflow: &str,
    ) -> PyResult<()> {
        let config = observation_batch::config(&layout.borrow(py), action_overflow)?;
        self.config = config;
        self.packed_layout = Some(layout);
        self.buffers = None;
        Ok(())
    }

    /// Encode observations for every active slot into one immutable v11 batch.
    fn encode_active_batch(&mut self, py: Python<'_>) -> PyResult<(Vec<usize>, ObservationBatch)> {
        let layout = self
            .packed_layout
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no registered observation layout"))?
            .clone_ref(py);
        let mut rows = PackedRows::new(&layout.borrow(py).arrays);
        let config = self.config;
        let inner = &mut self.inner;
        let mut encode_err = None;
        let active = py
            .allow_threads(|| {
                inner.encode_active_into(|_, obs| {
                    rows.append(obs, &config).map_err(|error| {
                        encode_err = Some(error);
                        AgentError(String::new())
                    })
                })
            })
            .map_err(|err| {
                if let Some(encode_err) = encode_err {
                    map_encode_err(encode_err)
                } else {
                    map_agent_err(err)
                }
            })?;
        let count = active.len();
        Ok((active, observation_batch::freeze(py, layout, rows, count)))
    }

    /// Step every active slot and return the still-active indices with one v11 batch.
    fn step_active_batch(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
    ) -> PyResult<(Vec<usize>, ObservationBatch)> {
        let layout = self
            .packed_layout
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("no registered observation layout"))?
            .clone_ref(py);
        let mut rows = PackedRows::new(&layout.borrow(py).arrays);
        let config = self.config;
        let inner = &mut self.inner;
        let mut encode_err = None;
        let active = py
            .allow_threads(|| {
                inner.step_active_into(&actions, |_, obs| {
                    rows.append(obs, &config).map_err(|error| {
                        encode_err = Some(error);
                        AgentError(String::new())
                    })
                })
            })
            .map_err(|err| {
                if let Some(encode_err) = encode_err {
                    map_encode_err(encode_err)
                } else {
                    map_agent_err(err)
                }
            })?;
        let count = active.len();
        Ok((active, observation_batch::freeze(py, layout, rows, count)))
    }

    /// Deprecated dense buffer registration; removed in 0.3.0.
    /// Attach caller-owned numpy buffers with leading dimension `capacity`
    /// (>= num_slots). The rewards/terminated/truncated buffers are required
    /// for layout parity with VectorEnv but are never written.
    #[pyo3(signature=(buffers, capacity, *, layout=None))]
    fn set_buffers(
        &mut self,
        buffers: Bound<'_, PyDict>,
        capacity: usize,
        layout: Option<&ObservationLayout>,
    ) -> PyResult<()> {
        if layout.is_some() {
            super::bindings::warn_deprecated(
                buffers.py(),
                "RolloutPool.set_buffers is deprecated; use RolloutPool.set_observation_layout with encode_active_batch/step_active_batch; removal in 0.3.0",
            )?;
        }
        let layout = registered_layout(&buffers, layout)?;
        let config = layout.config;
        if capacity < self.inner.num_slots() {
            return Err(PyValueError::new_err(format!(
                "buffer capacity {capacity} < num_slots {}",
                self.inner.num_slots()
            )));
        }
        let buffers = read_observation_buffers(&buffers, Some(capacity), &layout, true)?;
        self.config = config;
        self.buffers = Some(buffers);
        self.packed_layout = None;
        Ok(())
    }

    /// Encode observations for every active slot; returns their indices.
    fn encode_active(&mut self, py: Python<'_>) -> PyResult<Vec<usize>> {
        self.run_with_buffers(py, |inner, write_buffers, config| {
            inner.encode_active_into(|slot_index, obs| {
                write_buffers
                    .encode_observation(slot_index, obs, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    /// Step every active slot with `actions[slot_index]`; returns the slot
    /// indices still active afterwards (their observations re-encoded).
    fn step_active(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<Vec<usize>> {
        self.run_with_buffers(py, move |inner, write_buffers, config| {
            inner.step_active_into(&actions, |slot_index, obs| {
                write_buffers
                    .encode_observation(slot_index, obs, &config)
                    .map_err(AgentError::from)
            })
        })
    }

    /// Finish all still-active slots with engine-side random play to
    /// terminal (hybrid rollouts). Returns the number of slots finished.
    fn finish_random(&mut self, py: Python<'_>) -> PyResult<usize> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.finish_random().map_err(map_agent_err))
    }

    /// (scores, simulations, cap_hits) — mean playout score per root action.
    fn scores(&self) -> (Vec<f64>, u64, u64) {
        self.inner.scores()
    }
}

#[cfg(feature = "python")]
pub fn register_vector_env_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_observations_into, m)?)?;
    super::observation_batch::register(m)?;
    m.add_class::<PyVectorEnv>()?;
    m.add_class::<PySlotLoad>()?;
    m.add_class::<SlotLoadResult>()?;
    m.add_class::<PyRolloutPool>()
}

/// Write each env's acting-player observation into row `i` of buffers from
/// `ObservationSpace.allocate(capacity)`; returns each row's legal-action count.
///
/// Every legal action must fit (`action_overflow="error"`). Rows at and past
/// `len(envs)` are untouched, and no step-state buffers are read. The GIL stays
/// held, so no Python code can touch the caller-owned buffers during writes.
#[cfg(feature = "python")]
#[pyfunction]
fn encode_observations_into(
    py: Python<'_>,
    envs: Vec<PyRef<'_, PyEnv>>,
    layout: &ObservationLayout,
    buffers: Bound<'_, PyDict>,
    capacity: usize,
) -> PyResult<Vec<usize>> {
    super::bindings::warn_deprecated(
        py,
        "encode_observations_into is deprecated; use encode_observation_batch for v11 observations; removal in 0.3.0",
    )?;
    if envs.len() > capacity {
        return Err(PyValueError::new_err(format!(
            "{} envs exceed buffer capacity {capacity}",
            envs.len()
        )));
    }
    let config = layout.config;
    let buffers = read_observation_buffers(&buffers, Some(capacity), layout, false)?;
    with_observation_field_slices(py, &buffers, |out| {
        envs.iter()
            .enumerate()
            .map(|(index, branch)| {
                let env = branch
                    .inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("env lock poisoned"))?;
                let actor = env
                    .current_agent_index()
                    .ok_or_else(|| PyValueError::new_err("env has no acting player"))?;
                let observation = env.observation_for_player(actor).map_err(map_agent_err)?;
                let row = out.encoded_row_mut(index, &config).map_err(map_agent_err)?;
                encode_into(&observation, &config, row)
                    .map_err(crate::python::bindings::map_encode_err)?;
                Ok(observation.action_space.actions.len())
            })
            .collect()
    })
}

#[cfg(feature = "python")]
pub(crate) fn parse_player_policies(
    values: Option<Vec<String>>,
) -> PyResult<[crate::agent::player_policy::PlayerPolicy; 2]> {
    use crate::agent::player_policy::PlayerPolicy;
    let Some(values) = values else {
        return Ok([PlayerPolicy::Agent; 2]);
    };
    if values.len() != 2 {
        return Err(PyValueError::new_err(
            "player_policies requires exactly two seats",
        ));
    }
    let parse = |value: &str| match value {
        "agent" => Ok(PlayerPolicy::Agent),
        "random" => Ok(PlayerPolicy::Random),
        "heuristic" => Ok(PlayerPolicy::Heuristic),
        _ => Err(PyValueError::new_err(format!(
            "unsupported player policy: {value}"
        ))),
    };
    Ok([parse(&values[0])?, parse(&values[1])?])
}

#[cfg(feature = "python")]
fn mutable_slice_from_buffer<'py, T: Element>(
    py: Python<'py>,
    buffer: &'py PyBuffer<T>,
    key: &'static str,
) -> PyResult<&'py mut [T]> {
    let cells = buffer.as_mut_slice(py).ok_or_else(|| {
        PyValueError::new_err(format!("buffer '{key}' must be writable and C-contiguous"))
    })?;
    let ptr = cells.as_ptr() as *mut T;
    let len = cells.len();
    // SAFETY: `as_mut_slice` guarantees writable C-contiguous storage for `T`.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

#[cfg(feature = "python")]
fn typed_numpy_buffer<'py, T: Element>(
    py: Python<'py>,
    array: &'py Py<PyAny>,
    key: &'static str,
) -> PyResult<PyBuffer<T>> {
    PyBuffer::<T>::get_bound(array.bind(py))
        .map_err(|err| PyTypeError::new_err(format!("invalid buffer '{key}': {err}")))
}

#[cfg(feature = "python")]
fn send_slice_from_buffer<'py, T: Element>(
    py: Python<'py>,
    buffer: &'py PyBuffer<T>,
    key: &'static str,
) -> PyResult<SendSlice<T>> {
    let data = mutable_slice_from_buffer(py, buffer, key)?;
    Ok(SendSlice::from_mut_slice(data, key))
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct SendSlice<T> {
    ptr: *mut T,
    len: usize,
    field: &'static str,
}

#[cfg(feature = "python")]
// SAFETY: SendSlice wraps a raw pointer into a Python-owned numpy buffer.
// The GIL is released before use (py.allow_threads), and each env index
// writes to disjoint rows, so no aliasing occurs.
unsafe impl<T> Send for SendSlice<T> {}
unsafe impl<T> Sync for SendSlice<T> {}

#[cfg(feature = "python")]
impl<T> SendSlice<T> {
    fn from_mut_slice(data: &mut [T], field: &'static str) -> Self {
        Self {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            field,
        }
    }

    fn row_ptr(&self, row_index: usize, row_len: usize) -> Result<*mut T, AgentError> {
        let start = row_index
            .checked_mul(row_len)
            .ok_or_else(|| AgentError(format!("overflow while indexing '{}'", self.field)))?;
        let end = start
            .checked_add(row_len)
            .ok_or_else(|| AgentError(format!("overflow while indexing '{}'", self.field)))?;

        if end > self.len {
            return Err(AgentError(format!(
                "row {row_index} for '{}' out of bounds (row_len={row_len}, total={})",
                self.field, self.len
            )));
        }

        // SAFETY: bounds are checked above.
        Ok(unsafe { self.ptr.add(start) })
    }

    fn element_ptr(&self, index: usize) -> Result<*mut T, AgentError> {
        if index >= self.len {
            return Err(AgentError(format!(
                "index {index} for '{}' out of bounds (total={})",
                self.field, self.len
            )));
        }
        // SAFETY: bounds are checked above.
        Ok(unsafe { self.ptr.add(index) })
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
struct SendObservationFieldSlices {
    relations: Option<SendSlice<f32>>,
    effects: Option<SendSlice<f32>>,
    effect_details: Option<SendSlice<f32>>,
    effect_members: Option<SendSlice<f32>>,
    name_candidates: Option<SendSlice<f32>>,
    facts: Option<SendSlice<f32>>,
    semantic_cards: SendSlice<f32>,
    known_hand: SendSlice<f32>,
    known_library: Option<SendSlice<f32>>,
    decklists: Option<SendSlice<f32>>,
    player_turn_state: Option<SendSlice<f32>>,
    card_turn_state: Option<SendSlice<f32>>,
    stack: Option<SendSlice<f32>>,
    stack_valid: Option<SendSlice<f32>>,
    agent_player: SendSlice<f32>,
    opponent_player: SendSlice<f32>,
    agent_cards: SendSlice<f32>,
    opponent_cards: SendSlice<f32>,
    agent_permanents: SendSlice<f32>,
    opponent_permanents: SendSlice<f32>,
    actions: SendSlice<f32>,
    events: SendSlice<f32>,
    decision_records: Option<SendSlice<f32>>,
    decision_reference_details: Option<SendSlice<f32>>,
    decision_references: Option<SendSlice<f32>>,
    decision_members: Option<SendSlice<f32>>,
    action_focus: SendSlice<i32>,
    agent_player_valid: SendSlice<f32>,
    opponent_player_valid: SendSlice<f32>,
    agent_cards_valid: SendSlice<f32>,
    opponent_cards_valid: SendSlice<f32>,
    agent_permanents_valid: SendSlice<f32>,
    opponent_permanents_valid: SendSlice<f32>,
    actions_valid: SendSlice<f32>,
    events_valid: SendSlice<f32>,
}

#[cfg(feature = "python")]
impl SendObservationFieldSlices {
    fn encoded_row_mut(
        &self,
        env_index: usize,
        config: &ObservationEncoderConfig,
    ) -> Result<EncodedObservationMut<'_>, AgentError> {
        let semantic_cards_len = 2 * config.max_cards_per_player;
        let semantic_cards_ptr = self.semantic_cards.row_ptr(env_index, semantic_cards_len)?;
        let known_hand_len = 4 * config.max_cards_per_player;
        let known_hand_ptr = self.known_hand.row_ptr(env_index, known_hand_len)?;
        let cards_len = config.cards_len();
        let permanents_len = config.permanents_len();
        let actions_len = config.actions_len();
        let action_focus_len = config.action_focus_len();

        let agent_player_ptr = self.agent_player.row_ptr(env_index, config.player_dim())?;
        let opponent_player_ptr = self
            .opponent_player
            .row_ptr(env_index, config.player_dim())?;
        let agent_cards_ptr = self.agent_cards.row_ptr(env_index, cards_len)?;
        let opponent_cards_ptr = self.opponent_cards.row_ptr(env_index, cards_len)?;
        let agent_permanents_ptr = self.agent_permanents.row_ptr(env_index, permanents_len)?;
        let opponent_permanents_ptr = self
            .opponent_permanents
            .row_ptr(env_index, permanents_len)?;
        let actions_ptr = self.actions.row_ptr(env_index, actions_len)?;
        let action_focus_ptr = self.action_focus.row_ptr(env_index, action_focus_len)?;
        let agent_player_valid_ptr = self.agent_player_valid.row_ptr(env_index, 1)?;
        let opponent_player_valid_ptr = self.opponent_player_valid.row_ptr(env_index, 1)?;
        let agent_cards_valid_ptr = self
            .agent_cards_valid
            .row_ptr(env_index, config.max_cards_per_player)?;
        let opponent_cards_valid_ptr = self
            .opponent_cards_valid
            .row_ptr(env_index, config.max_cards_per_player)?;
        let agent_permanents_valid_ptr = self
            .agent_permanents_valid
            .row_ptr(env_index, config.max_permanents_per_player)?;
        let opponent_permanents_valid_ptr = self
            .opponent_permanents_valid
            .row_ptr(env_index, config.max_permanents_per_player)?;
        let actions_valid_ptr = self.actions_valid.row_ptr(env_index, config.max_actions)?;
        let events_ptr = self.events.row_ptr(env_index, config.events_len())?;
        let events_valid_ptr = self.events_valid.row_ptr(env_index, config.max_events)?;

        Ok(EncodedObservationMut {
            decision_records: if let Some(buffer) = self.decision_records {
                // SAFETY: registration validates storage and each env owns a disjoint row.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(
                            env_index,
                            config.max_decision_records * crate::agent::decision_memory::RECORD_DIM,
                        )?,
                        config.max_decision_records * crate::agent::decision_memory::RECORD_DIM,
                    )
                }
            } else {
                &mut []
            },
            decision_reference_details: if let Some(buffer) = self.decision_reference_details {
                // SAFETY: registration validates storage and each env owns a disjoint row.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(
                            env_index,
                            config.max_decision_reference_details
                                * crate::agent::decision_memory::DETAIL_DIM,
                        )?,
                        config.max_decision_reference_details
                            * crate::agent::decision_memory::DETAIL_DIM,
                    )
                }
            } else {
                &mut []
            },
            decision_references: if let Some(buffer) = self.decision_references {
                // SAFETY: registration validates storage and each env owns a disjoint row.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(
                            env_index,
                            config.max_decision_references
                                * crate::agent::decision_memory::REFERENCE_DIM,
                        )?,
                        config.max_decision_references
                            * crate::agent::decision_memory::REFERENCE_DIM,
                    )
                }
            } else {
                &mut []
            },
            decision_members: if let Some(buffer) = self.decision_members {
                // SAFETY: registration validates storage and each env owns a disjoint row.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(
                            env_index,
                            config.max_decision_members * crate::agent::decision_memory::MEMBER_DIM,
                        )?,
                        config.max_decision_members * crate::agent::decision_memory::MEMBER_DIM,
                    )
                }
            } else {
                &mut []
            },
            relations: if let Some(buffer) = self.relations {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.relations_len())?,
                        config.relations_len(),
                    )
                }
            } else {
                &mut []
            },
            effects: if let Some(buffer) = self.effects {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effects_len())?,
                        config.effects_len(),
                    )
                }
            } else {
                &mut []
            },
            effect_details: if let Some(buffer) = self.effect_details {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effect_details_len())?,
                        config.effect_details_len(),
                    )
                }
            } else {
                &mut []
            },
            name_candidates: if let Some(buffer) = self.name_candidates {
                // SAFETY: registered buffers partition environments into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.name_candidates_len())?,
                        config.name_candidates_len(),
                    )
                }
            } else {
                &mut []
            },
            effect_members: if let Some(buffer) = self.effect_members {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.effect_members_len())?,
                        config.effect_members_len(),
                    )
                }
            } else {
                &mut []
            },
            facts: if let Some(buffer) = self.facts {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.facts_len())?,
                        config.facts_len(),
                    )
                }
            } else {
                &mut []
            },
            // SAFETY: each call uses disjoint per-env rows.
            known_library: if let Some(buffer) = self.known_library {
                // SAFETY: registered buffers partition each environment into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.known_library_len())?,
                        config.known_library_len(),
                    )
                }
            } else {
                &mut []
            },
            known_hand: unsafe { slice::from_raw_parts_mut(known_hand_ptr, known_hand_len) },
            decklists: if let Some(buffer) = self.decklists {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.decklists_len())?,
                        config.decklists_len(),
                    )
                }
            } else {
                &mut []
            },
            player_turn_state: if let Some(buffer) = self.player_turn_state {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.player_turn_state_len())?,
                        config.player_turn_state_len(),
                    )
                }
            } else {
                &mut []
            },
            card_turn_state: if let Some(buffer) = self.card_turn_state {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.card_turn_state_len())?,
                        config.card_turn_state_len(),
                    )
                }
            } else {
                &mut []
            },
            stack: if let Some(buffer) = self.stack {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.stack_len())?,
                        config.stack_len(),
                    )
                }
            } else {
                &mut []
            },
            stack_valid: if let Some(buffer) = self.stack_valid {
                // SAFETY: the registered buffer is partitioned into disjoint rows.
                unsafe {
                    slice::from_raw_parts_mut(
                        buffer.row_ptr(env_index, config.stack_rows())?,
                        config.stack_rows(),
                    )
                }
            } else {
                &mut []
            },
            // SAFETY: each call uses disjoint per-env rows.
            semantic_cards: unsafe {
                slice::from_raw_parts_mut(semantic_cards_ptr, semantic_cards_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_player: unsafe {
                slice::from_raw_parts_mut(agent_player_ptr, config.player_dim())
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_player: unsafe {
                slice::from_raw_parts_mut(opponent_player_ptr, config.player_dim())
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_cards: unsafe { slice::from_raw_parts_mut(agent_cards_ptr, cards_len) },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_cards: unsafe { slice::from_raw_parts_mut(opponent_cards_ptr, cards_len) },
            // SAFETY: each call uses disjoint per-env rows.
            agent_permanents: unsafe {
                slice::from_raw_parts_mut(agent_permanents_ptr, permanents_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_permanents: unsafe {
                slice::from_raw_parts_mut(opponent_permanents_ptr, permanents_len)
            },
            // SAFETY: each call uses disjoint per-env rows.
            actions: unsafe { slice::from_raw_parts_mut(actions_ptr, actions_len) },
            // SAFETY: each call uses disjoint per-env rows.
            events: unsafe { slice::from_raw_parts_mut(events_ptr, config.events_len()) },
            // SAFETY: each call uses disjoint per-env rows.
            action_focus: unsafe { slice::from_raw_parts_mut(action_focus_ptr, action_focus_len) },
            // SAFETY: each call uses disjoint per-env rows.
            agent_player_valid: unsafe { slice::from_raw_parts_mut(agent_player_valid_ptr, 1) },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_player_valid: unsafe {
                slice::from_raw_parts_mut(opponent_player_valid_ptr, 1)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_cards_valid: unsafe {
                slice::from_raw_parts_mut(agent_cards_valid_ptr, config.max_cards_per_player)
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_cards_valid: unsafe {
                slice::from_raw_parts_mut(opponent_cards_valid_ptr, config.max_cards_per_player)
            },
            // SAFETY: each call uses disjoint per-env rows.
            agent_permanents_valid: unsafe {
                slice::from_raw_parts_mut(
                    agent_permanents_valid_ptr,
                    config.max_permanents_per_player,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            opponent_permanents_valid: unsafe {
                slice::from_raw_parts_mut(
                    opponent_permanents_valid_ptr,
                    config.max_permanents_per_player,
                )
            },
            // SAFETY: each call uses disjoint per-env rows.
            actions_valid: unsafe {
                slice::from_raw_parts_mut(actions_valid_ptr, config.max_actions)
            },
            // SAFETY: each call uses disjoint per-env rows.
            events_valid: unsafe { slice::from_raw_parts_mut(events_valid_ptr, config.max_events) },
        })
    }
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
enum Destination<'a> {
    Dense(SendObservationFieldSlices),
    Packed(&'a Mutex<PackedWrites>),
}

struct SendWriteBuffers<'a> {
    memory_encoding: Option<&'a [Mutex<crate::agent::decision_memory::MemoryEncoding>]>,
    fields: Destination<'a>,
    rewards: SendSlice<f64>,
    terminated: SendSlice<u8>,
    truncated: SendSlice<u8>,
}

#[cfg(feature = "python")]
impl SendWriteBuffers<'_> {
    fn clear_memory_cursor(&self, slot: usize) {
        if let Some(cursors) = self.memory_encoding {
            *cursors[slot].lock().expect("exclusive buffer row") = Default::default();
        }
    }

    fn encode_observation(
        &self,
        env_index: usize,
        observation: &Observation,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationWriteError> {
        let out = match &self.fields {
            Destination::Packed(writer) => {
                return writer
                    .lock()
                    .unwrap()
                    .append(env_index, observation, config)
            }
            Destination::Dense(fields) => fields.encoded_row_mut(env_index, config)?,
        };
        match self.memory_encoding {
            Some(cursors) => {
                let mut cursor = cursors[env_index].lock().expect("exclusive buffer row");
                encode_into_incremental(observation, config, out, &mut cursor)
            }
            None => encode_into(observation, config, out),
        }
        .map_err(ObservationWriteError::from)
    }

    fn write_step_state(
        &self,
        env_index: usize,
        reward: f64,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), AgentError> {
        let reward_ptr = self.rewards.element_ptr(env_index)?;
        let terminated_ptr = self.terminated.element_ptr(env_index)?;
        let truncated_ptr = self.truncated.element_ptr(env_index)?;

        // SAFETY: each env index is written at most once per step/reset call.
        unsafe {
            *reward_ptr = reward;
            *terminated_ptr = terminated as u8;
            *truncated_ptr = truncated as u8;
        }
        Ok(())
    }

    fn write_encoded_row(
        &self,
        env_index: usize,
        observation: &Observation,
        reward: f64,
        terminated: bool,
        truncated: bool,
        config: &ObservationEncoderConfig,
    ) -> Result<(), ObservationWriteError> {
        let started = Instant::now();
        let encoded = self.encode_observation(env_index, observation, config);
        ENCODING_NANOS.set(ENCODING_NANOS.get() + started.elapsed().as_nanos() as u64);
        encoded?;
        self.write_step_state(env_index, reward, terminated, truncated)
            .map_err(ObservationWriteError::from)
    }
}

#[cfg(feature = "python")]
fn with_send_write_buffers<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(SendWriteBuffers) -> PyResult<R>,
) -> PyResult<R> {
    let step_state = buffers
        .step_state
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("step-state buffers are required"))?;
    with_observation_field_slices(py, buffers, |fields| {
        with_step_writer(py, step_state, Destination::Dense(fields), f)
    })
}

fn with_step_writer<'a, R>(
    py: Python<'_>,
    step_state: &StepStateBuffers,
    fields: Destination<'a>,
    run: impl FnOnce(SendWriteBuffers<'a>) -> PyResult<R>,
) -> PyResult<R> {
    let rewards_buffer = typed_numpy_buffer::<f64>(py, &step_state.rewards, "rewards")?;
    let terminated_buffer = typed_numpy_buffer::<u8>(py, &step_state.terminated, "terminated")?;
    let truncated_buffer = typed_numpy_buffer::<u8>(py, &step_state.truncated, "truncated")?;
    for (key, shape) in [
        ("rewards", rewards_buffer.shape()),
        ("terminated", terminated_buffer.shape()),
        ("truncated", truncated_buffer.shape()),
    ] {
        if shape != step_state.shape {
            return Err(PyValueError::new_err(format!(
                "{key} shape changed after registration: expected {:?}, got {shape:?}",
                step_state.shape
            )));
        }
    }
    run(SendWriteBuffers {
        memory_encoding: None,
        fields,
        rewards: send_slice_from_buffer(py, &rewards_buffer, "rewards")?,
        terminated: send_slice_from_buffer(py, &terminated_buffer, "terminated")?,
        truncated: send_slice_from_buffer(py, &truncated_buffer, "truncated")?,
    })
}

#[cfg(feature = "python")]
fn with_observation_field_slices<R>(
    py: Python<'_>,
    buffers: &ObservationBuffers,
    f: impl FnOnce(SendObservationFieldSlices) -> PyResult<R>,
) -> PyResult<R> {
    let relations_buffer = buffers
        .relations
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "relations"))
        .transpose()?;
    let effects_buffer = buffers
        .effects
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effects"))
        .transpose()?;
    let effect_details_buffer = buffers
        .effect_details
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effect_details"))
        .transpose()?;
    let name_candidates_buffer = buffers
        .name_candidates
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "name_candidates"))
        .transpose()?;
    let effect_members_buffer = buffers
        .effect_members
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "effect_members"))
        .transpose()?;
    let facts_buffer = buffers
        .facts
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "facts"))
        .transpose()?;
    let semantic_cards_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.semantic_cards, "semantic_cards")?;
    let known_library_buffer = buffers
        .known_library
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "known_library"))
        .transpose()?;
    let known_hand_buffer = typed_numpy_buffer::<f32>(py, &buffers.known_hand, "known_hand")?;
    let decklists_buffer = buffers
        .decklists
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "decklists"))
        .transpose()?;
    let player_turn_state_buffer = buffers
        .player_turn_state
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "player_turn_state"))
        .transpose()?;
    let card_turn_state_buffer = buffers
        .card_turn_state
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "card_turn_state"))
        .transpose()?;
    let stack_buffer = buffers
        .stack
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "stack"))
        .transpose()?;
    let stack_valid_buffer = buffers
        .stack_valid
        .as_ref()
        .map(|array| typed_numpy_buffer::<f32>(py, array, "stack_valid"))
        .transpose()?;
    let agent_player_buffer = typed_numpy_buffer::<f32>(py, &buffers.agent_player, "agent_player")?;
    let opponent_player_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_player, "opponent_player")?;
    let agent_cards_buffer = typed_numpy_buffer::<f32>(py, &buffers.agent_cards, "agent_cards")?;
    let opponent_cards_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_cards, "opponent_cards")?;
    let agent_permanents_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_permanents, "agent_permanents")?;
    let opponent_permanents_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_permanents, "opponent_permanents")?;
    let actions_buffer = typed_numpy_buffer::<f32>(py, &buffers.actions, "actions")?;
    let decision_records_buffer = buffers
        .decision_records
        .as_ref()
        .map(|value| typed_numpy_buffer::<f32>(py, value, "decision_records"))
        .transpose()?;
    let decision_reference_details_buffer = buffers
        .decision_reference_details
        .as_ref()
        .map(|value| typed_numpy_buffer::<f32>(py, value, "decision_reference_details"))
        .transpose()?;
    let decision_references_buffer = buffers
        .decision_references
        .as_ref()
        .map(|value| typed_numpy_buffer::<f32>(py, value, "decision_references"))
        .transpose()?;
    let decision_members_buffer = buffers
        .decision_members
        .as_ref()
        .map(|value| typed_numpy_buffer::<f32>(py, value, "decision_members"))
        .transpose()?;
    let events_buffer = typed_numpy_buffer::<f32>(py, &buffers.events, "events")?;
    let action_focus_buffer = typed_numpy_buffer::<i32>(py, &buffers.action_focus, "action_focus")?;
    let agent_player_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_player_valid, "agent_player_valid")?;
    let opponent_player_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_player_valid, "opponent_player_valid")?;
    let agent_cards_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.agent_cards_valid, "agent_cards_valid")?;
    let opponent_cards_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.opponent_cards_valid, "opponent_cards_valid")?;
    let agent_permanents_valid_buffer = typed_numpy_buffer::<f32>(
        py,
        &buffers.agent_permanents_valid,
        "agent_permanents_valid",
    )?;
    let opponent_permanents_valid_buffer = typed_numpy_buffer::<f32>(
        py,
        &buffers.opponent_permanents_valid,
        "opponent_permanents_valid",
    )?;
    let actions_valid_buffer =
        typed_numpy_buffer::<f32>(py, &buffers.actions_valid, "actions_valid")?;
    let events_valid_buffer = typed_numpy_buffer::<f32>(py, &buffers.events_valid, "events_valid")?;
    let field_slices = SendObservationFieldSlices {
        relations: relations_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "relations"))
            .transpose()?,
        effects: effects_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effects"))
            .transpose()?,
        effect_details: effect_details_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effect_details"))
            .transpose()?,
        name_candidates: name_candidates_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "name_candidates"))
            .transpose()?,
        effect_members: effect_members_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "effect_members"))
            .transpose()?,
        facts: facts_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "facts"))
            .transpose()?,
        known_hand: send_slice_from_buffer(py, &known_hand_buffer, "known_hand")?,
        known_library: known_library_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "known_library"))
            .transpose()?,
        decklists: decklists_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decklists"))
            .transpose()?,
        player_turn_state: player_turn_state_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "player_turn_state"))
            .transpose()?,
        card_turn_state: card_turn_state_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "card_turn_state"))
            .transpose()?,
        stack: stack_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "stack"))
            .transpose()?,
        stack_valid: stack_valid_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "stack_valid"))
            .transpose()?,
        semantic_cards: send_slice_from_buffer(py, &semantic_cards_buffer, "semantic_cards")?,
        agent_player: send_slice_from_buffer(py, &agent_player_buffer, "agent_player")?,
        opponent_player: send_slice_from_buffer(py, &opponent_player_buffer, "opponent_player")?,
        agent_cards: send_slice_from_buffer(py, &agent_cards_buffer, "agent_cards")?,
        opponent_cards: send_slice_from_buffer(py, &opponent_cards_buffer, "opponent_cards")?,
        agent_permanents: send_slice_from_buffer(py, &agent_permanents_buffer, "agent_permanents")?,
        opponent_permanents: send_slice_from_buffer(
            py,
            &opponent_permanents_buffer,
            "opponent_permanents",
        )?,
        actions: send_slice_from_buffer(py, &actions_buffer, "actions")?,
        decision_records: decision_records_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decision_records"))
            .transpose()?,
        decision_reference_details: decision_reference_details_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decision_reference_details"))
            .transpose()?,
        decision_references: decision_references_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decision_references"))
            .transpose()?,
        decision_members: decision_members_buffer
            .as_ref()
            .map(|buffer| send_slice_from_buffer(py, buffer, "decision_members"))
            .transpose()?,
        events: send_slice_from_buffer(py, &events_buffer, "events")?,
        action_focus: send_slice_from_buffer(py, &action_focus_buffer, "action_focus")?,
        agent_player_valid: send_slice_from_buffer(
            py,
            &agent_player_valid_buffer,
            "agent_player_valid",
        )?,
        opponent_player_valid: send_slice_from_buffer(
            py,
            &opponent_player_valid_buffer,
            "opponent_player_valid",
        )?,
        agent_cards_valid: send_slice_from_buffer(
            py,
            &agent_cards_valid_buffer,
            "agent_cards_valid",
        )?,
        opponent_cards_valid: send_slice_from_buffer(
            py,
            &opponent_cards_valid_buffer,
            "opponent_cards_valid",
        )?,
        agent_permanents_valid: send_slice_from_buffer(
            py,
            &agent_permanents_valid_buffer,
            "agent_permanents_valid",
        )?,
        opponent_permanents_valid: send_slice_from_buffer(
            py,
            &opponent_permanents_valid_buffer,
            "opponent_permanents_valid",
        )?,
        actions_valid: send_slice_from_buffer(py, &actions_valid_buffer, "actions_valid")?,
        events_valid: send_slice_from_buffer(py, &events_valid_buffer, "events_valid")?,
    };

    f(field_slices)
}
