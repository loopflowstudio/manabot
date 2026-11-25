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
        observation_encoder::ObservationEncoderConfig,
        rollout_pool::RolloutPool,
        vector_env::{
            ObservationRead, ObservationWriteError, SlotLoad, SlotLoadResult, SlotReplacement,
            VectorEnv,
        },
    },
    infra::profiler::InfoDict,
    python::{
        bindings::{map_agent_err, map_encode_err, PyObservation, PyPlayerConfig},
        convert::{info_dict_to_pydict, require_numpy_array},
        layout_bindings::ObservationLayout,
        snapshot_bindings::{PyGameSnapshot, PyRestoreOptions, PySnapshotMetadata},
    },
    state::player::PlayerConfig,
};

#[cfg(feature = "python")]
type PyResetResult = (PyObservation, PyObject);
#[cfg(feature = "python")]
type PyStepResult = (PyObservation, f64, bool, bool, PyObject);

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
    packed: Option<PackedVector>,
    last_info: Vec<InfoDict>,
    timings: BufferedCallTimings,
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
    #[pyo3(signature = (num_envs, seed=0, skip_trivial=true, opponent_policy=None, seed_stride=None, *, player_policies=None, decision_memory_include_own=false, decision_memory=false))]
    fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        opponent_policy: Option<&str>,
        seed_stride: Option<u64>,
        player_policies: Option<Vec<String>>,
        decision_memory_include_own: bool,
        decision_memory: bool,
    ) -> PyResult<Self> {
        let policies = parse_player_policies(player_policies, opponent_policy)?;
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
        self.with_buffers(py, move |inner, buffers, config, layout| {
            // Admit every replacement into scratch rows before publishing any of
            // them, so a rejected load (including a sibling shard's) leaves no
            // frozen batch. Owners may have resized step outcome arrays between calls.
            let mut admission = PackedRows::new(&layout.arrays);
            inner
                .prepare_slots(&requests, |slot, obs| {
                    buffers.rewards.element_ptr(slot)?;
                    buffers.terminated.element_ptr(slot)?;
                    buffers.truncated.element_ptr(slot)?;
                    admission.append(obs, &config)?;
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
        let loaded = self.with_buffers(py, move |inner, buffers, config, _| {
            inner
                .publish_slots(replacements, |slot, obs, terminal| {
                    buffers.write_encoded_row(slot, obs, 0.0, terminal, false, &config)
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
        self.last_info = self.with_buffers(py, |inner, buffers, config, _| {
            run(inner, buffers, config).map_err(map_write_error)
        })?;
        Ok(())
    }

    /// Run with the registered step outcome arrays and one packed writer, then
    /// publish every written slot's row in a single frozen batch.
    fn with_buffers<R: Send>(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut VectorEnv,
                SendWriteBuffers,
                ObservationEncoderConfig,
                &ObservationLayout,
            ) -> PyResult<R>
            + Send,
    ) -> PyResult<R> {
        let config = self.config;
        let inner = &mut self.inner;
        let packed = self.packed.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("set_observation_layout() must be called first")
        })?;
        let started = Instant::now();
        ENCODING_NANOS.set(0);
        let borrowed = packed.layout.borrow(py);
        let layout: &ObservationLayout = &borrowed;
        let writer = Mutex::new(PackedWrites::new(layout));
        let result = with_step_writer(py, &packed.state, &writer, |buffers| {
            py.allow_threads(|| run(inner, buffers, config, layout))
        });
        drop(borrowed);
        let writes = writer.into_inner().unwrap();
        if result.is_ok() && !writes.slots.is_empty() {
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
/// simultaneously: `encode_active_batch()` / `step_active_batch(actions)`
/// return the active slot indices with one v11 batch of their observations,
/// so a policy network can pick every rollout action with one batched
/// forward pass per step.
#[cfg(feature = "python")]
#[pyclass(name = "RolloutPool")]
pub struct PyRolloutPool {
    inner: RolloutPool,
    config: ObservationEncoderConfig,
    packed_layout: Option<Py<ObservationLayout>>,
}

#[cfg(feature = "python")]
impl PyRolloutPool {
    pub fn from_inner(inner: RolloutPool) -> Self {
        Self {
            inner,
            config: ObservationEncoderConfig::default(),
            packed_layout: None,
        }
    }

    fn run_active_batch(
        &mut self,
        py: Python<'_>,
        run: impl FnOnce(
                &mut RolloutPool,
                &mut dyn FnMut(usize, &Observation) -> Result<(), AgentError>,
            ) -> Result<Vec<usize>, AgentError>
            + Send,
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
                run(inner, &mut |_, obs| {
                    rows.append(obs, &config).map_err(|error| {
                        encode_err = Some(error);
                        AgentError(String::new())
                    })
                })
            })
            .map_err(|err| encode_err.map_or_else(|| map_agent_err(err), map_encode_err))?;
        let count = active.len();
        Ok((active, observation_batch::freeze(py, layout, rows, count)))
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
        Ok(())
    }

    /// Encode observations for every active slot into one immutable v11 batch.
    fn encode_active_batch(&mut self, py: Python<'_>) -> PyResult<(Vec<usize>, ObservationBatch)> {
        self.run_active_batch(py, |inner, write| inner.encode_active_into(write))
    }

    /// Step every active slot and return the still-active indices with one v11 batch.
    fn step_active_batch(
        &mut self,
        py: Python<'_>,
        actions: Vec<i64>,
    ) -> PyResult<(Vec<usize>, ObservationBatch)> {
        self.run_active_batch(py, |inner, write| inner.step_active_into(&actions, write))
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
    super::observation_batch::register(m)?;
    m.add_class::<PyVectorEnv>()?;
    m.add_class::<PySlotLoad>()?;
    m.add_class::<SlotLoadResult>()?;
    m.add_class::<PyRolloutPool>()
}

#[cfg(feature = "python")]
pub(crate) fn parse_player_policies(
    values: Option<Vec<String>>,
    legacy: Option<&str>,
) -> PyResult<[crate::agent::player_policy::PlayerPolicy; 2]> {
    use crate::agent::player_policy::PlayerPolicy;
    if values.is_some() && legacy.is_some() {
        return Err(PyValueError::new_err(
            "use player_policies or opponent_policy, not both",
        ));
    }
    let values = if let Some(legacy) = legacy {
        if !["none", "random", "heuristic"].contains(&legacy) {
            return Err(PyValueError::new_err(format!(
                "unsupported opponent_policy: {legacy}"
            )));
        }
        Python::with_gil(|py| -> PyResult<()> {
            py.import_bound("warnings")?.call_method1("warn", ("opponent_policy is deprecated; use player_policies=('agent', '<policy>'); removal in 0.3.0", py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(), 2))?;
            Ok(())
        })?;
        vec![
            "agent".into(),
            if legacy == "none" { "agent" } else { legacy }.into(),
        ]
    } else {
        values.unwrap_or_else(|| vec!["agent".into(); 2])
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

struct SendWriteBuffers<'a> {
    rows: &'a Mutex<PackedWrites>,
    rewards: SendSlice<f64>,
    terminated: SendSlice<u8>,
    truncated: SendSlice<u8>,
}

#[cfg(feature = "python")]
impl SendWriteBuffers<'_> {
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
        let encoded = self
            .rows
            .lock()
            .unwrap()
            .append(env_index, observation, config);
        ENCODING_NANOS.set(ENCODING_NANOS.get() + started.elapsed().as_nanos() as u64);
        encoded?;
        self.write_step_state(env_index, reward, terminated, truncated)
            .map_err(ObservationWriteError::from)
    }
}

fn with_step_writer<'a, R>(
    py: Python<'_>,
    step_state: &StepStateBuffers,
    rows: &'a Mutex<PackedWrites>,
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
        rows,
        rewards: send_slice_from_buffer(py, &rewards_buffer, "rewards")?,
        terminated: send_slice_from_buffer(py, &terminated_buffer, "terminated")?,
        truncated: send_slice_from_buffer(py, &truncated_buffer, "truncated")?,
    })
}
