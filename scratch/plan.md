# MTG-437: 0.2.5 compatibility gaps for jace, ahead of 0.2.6

## Outcome

A 0.2.x library consumer that collects in worker processes and reads the
deprecated encoder shim runs on this branch unchanged. Concretely, for
`RyanSaxe/jace` at `9d212cb` (pinned to manabot `2ae692ac`):

- a `RolloutBatch` from `collect()` / `collect_pair()` crosses a `spawn`
  `multiprocessing.Queue` and learns to the same prepared inputs as the batch
  that never left its process, with and without decision memory;
- worker batches merge with `RolloutBatch.concatenate` without densifying, and
  jace's own dense `_merge_recorded` keeps working;
- `obs_space.encoder.max_actions` (and the other pre-#566 shortcut attributes,
  including via `ObservationEncoder(config).max_actions`) answer, with a
  `DeprecationWarning` naming `0.3.0`.

No version bump, no removal of a `0.3.0`-deprecated API. `v0.2.6` is cut after
this merges.

## What the probes showed (2026-11-24, this checkout at `f75ba48c72`)

Environment built here with `uv sync --locked --python 3.13 --extra dev
--extra artifacts`; the extension is `managym/_managym.cpython-313-darwin.so`.

| Object | `pickle.dumps` today |
| --- | --- |
| `ObservationSpace` | works, 627 KB (gym `Box` bounds); checkpoints store JSON config in safetensors, never pickled spaces |
| `managym.ObservationLayout` | works, 353 B (`__getnewargs_ex__`, `layout_bindings.rs`) |
| `managym.ObservationBatch` | `TypeError: cannot pickle 'managym.ObservationBatch' object` |
| `ObservationSelection`, `next_obs`, `PackedActions` (`native`), `RolloutBatch` | same error, all through the batch; additionally `ObservationSelection` has `@cached_property` caches (`batch`, `action_counts`, `spans`) in `__dict__` |
| `tuple[Prefix, ...]` (960 prefixes, 236 chunks) | works, 7.53 MB, carries `Tail.features` / `Prefix.serving` caches, skips `__post_init__` certificates, recursion grows with chain length |
| `archive_memory(...)` of the same prefixes | 3.05 MB |

`obs_space.encoder.max_actions` raises `AttributeError`. Missing from
`ObservationSpace` versus the pre-#566 `ObservationEncoder`
(`git show b5ae42ee03^:manabot/env/observation.py`): `max_actions`,
`cards_per_player`, `perms_per_player`, `max_events`, `max_stack_objects`,
`max_focus_objects`, `player_dim`, `card_dim`, `permanent_dim`, `action_dim`,
`event_dim`, `relation_dim`, `name_dim` (the brief's list), plus
`library_shape` and the six `num_*` enum counts.

What jace actually does (`/tmp/jace-audit-437/src`, the audited commit):

- `src/jace/collect/pool.py`: `mp.get_context("spawn")` queues;
  `collect_pair(..., batch_closure="complete-games")[0]` per worker; the result
  goes through `pack_batch_for_ipc`, which archives and strips only
  `recorded.memory`. The same prefixes stay reachable from
  `recorded.rows.memory` and `next_obs.memory`, so prefixes must pickle well
  wherever they are held, not only in one field.
- `src/jace/collect/agent.py::_merge_recorded` reads
  `batch.recorded.rows[key]` for every key and `shape[0]` of the first array:
  the `Mapping` face of `ObservationSelection`. Its merged batch has dense
  `recorded.rows`, `recorded.memory` restored from the archive, and
  `PackedActions.concatenate` output that still has `native` set.
- Complete-game batches are `[rows, 1]` with `trajectories=None`, one
  `next_obs` row and `next_done == [True]` (confirmed by probe: shape
  `(960, 1)`); jace keeps the last batch's `next_obs`.
- `src/jace/forks/collect.py:139,306` and `forks/batch.py:165,225` read
  `obs_space.encoder.max_actions`; everything else reads `.encoder.config`.

## Design

### 1–2. Transport — built in `8076ce311a`

`ObservationBatch.__reduce__` -> sealed `managym._restore_observation_batch`
(full table validation, shared stride helper in `freeze`/`gather`);
cache-free `__reduce__` on `ObservationSpace` (from `config`),
`ObservationSelection`, `Chunk`/`Tail`/`Prefix` (oldest-first `restore_prefix`),
`PrefixObservation`, `CompactObservation`. Default dataclass pickling of
`RolloutBatch` shares every batch and prefix once. Tests 1–3 below pass in
`tests/training/test_rollout_transport.py`.

### 3. Packed worker batch merge — built in `6c9358791d`

`RolloutBatch.concatenate` joins packed complete games or explicit trajectory spans, rebasing recorded rows, frozen bootstraps and worker-local branch groups independently. Both piece-level merges and the dense 0.2.x consumer path are covered in `tests/training/test_rollout_transport.py`.

### 4. Deprecated encoder shortcuts — built in `7e582bc943`

Both `space.encoder` and `ObservationEncoder(config)` expose all twenty historical shortcuts with the 0.3.0 warning across v9, v10 and v11. The legacy known-library and stack behavior is documented in `5834b2e087`.

## Acceptance check at gate

In-repo, headless:

```
uv run --extra dev pytest tests/training/test_rollout_transport.py \
  tests/env/test_named_observation_layout.py -q
```

`tests/training/test_rollout_transport.py` (new; name checked unique) covers
the full scenario end to end on real collected batches, once without and once
with decision memory:

1. A `spawn` child (`manabot.sim.child.ChildProcess`, never `multiprocessing`
   spawn of the test's `__main__`) collects complete games and puts the
   `RolloutBatch` on a queue; the parent, collecting with the same seed,
   compares: every array field equal, `policy_rows(indexes)` prepared by the
   model equal array for array, `next_obs` prepared equal, and with memory on
   the unique chunk count equal (compactness).
2. `pickle.loads(pickle.dumps(x))` and `copy.deepcopy(x)` for
   `ObservationSpace` (verifying < 4 KB), `ObservationBatch` (including
   `action_focus` with `max_focus_objects > 1`, `blank_decision_memory`, and
   `gather` with repeats), `ObservationSelection` (including after `.batch` and
   `.spans` have been read — verifying cached properties are NOT pickled and
   pickled size stays below dense `export()` bytes), `Prefix` (including
   `tail=None` and deep chains under `sys.setrecursionlimit(120)`),
   `PrefixObservation`, and `RolloutBatch`.
3. `_restore_observation_batch` rejects tampered tables (short offsets,
   non-zero start, decreasing offsets, span exceeding layout capacity, wrong
   value length, wrong `seat_axis`, unknown/missing key, legacy layout) with
   `ValueError`, and `restore_prefix` rejects truncated/broken chains with
   `ValueError`.
4. `RolloutBatch.concatenate` of two worker batches equals, row for row, the
   prepared inputs and behavior logs of the parts:
   - for complete-game batches (native and dense);
   - for explicit-trajectory batches (spans and bootstrap indexes shifted,
     including one worker with 0 bootstrap rows / `next_obs == {}`, and
     colliding `branch_group_id`s re-offset so `grouped_roots` succeeds);
   - incompatible parts (empty, multi-column `[T, E]` including 1-element list,
     mixed native/dense, mismatched `space.config` including
     `decision_memory_include_own`, mismatched optional fields, non-terminal
     complete-game batch) raise `ValueError`.
5. jace's 0.2.x path verbatim: `archive_memory` + strip `recorded.memory`,
   pickle, restore, then a dense merge through `recorded.rows[key]` with
   `PackedActions.concatenate`; `policy_rows` on the result prepares the same
   inputs as native concatenation.

`tests/env/test_named_observation_layout.py` gains the shim attributes on v11
and legacy v9/v10 configs: each equals the config, shape, or enum count it
names (including `library_shape` on v9/v10), warns with `0.3.0`, works via both
`space.encoder.<attr>` and `ObservationEncoder(config).<attr>`, and unknown /
dunder names still raise `AttributeError`.

Against the consumer itself (the demo): a throwaway environment outside the
checkout with jace at `9d212cb` and this branch's `manabot` and built
`managym` installed over its pin, then `uv run --no-sync pytest tests/test_forks.py
tests/test_fork_batch.py tests/test_rl.py tests/test_manabot_levers.py
tests/test_full_data.py -q`. Expected: the three fork tests that fail on
`v0.2.5` with `AttributeError` pass, and the pool tests finish instead of
hanging. Anything in jace that still fails for a reason outside this change is
reported, not patched here.

Rust: `cargo fmt --manifest-path managym/Cargo.toml --all`, clippy, and the
extension rebuilt and placed atomically before the Python tests. The change is
binding-only; engine digests and `managym/perf/baseline.json` do not move.
`tests/model/test_pinned_pilot.py` and `tests/test_package_boundaries.py` run
as existing consumers.

## Absent and error states

- Corrupt or tampered bytes passed to `_restore_observation_batch`: `ValueError`
  at restore time, never a Rust panic or NumPy `IndexError` at first read.
- Corrupt chunk, broken/truncated chain, or capacity overflow in a pickled
  prefix or `PrefixObservation`: `ValueError` from `Chunk`, `Tail`,
  `restore_prefix`, or `PrefixObservation.__init__`.
- `concatenate([])`, multi-column batches (even a 1-element list), mixed
  transports, mismatched `space.config` / capacities / optional fields,
  inconsistent `recorded.memory` vs `recorded.rows.memory`: `ValueError` naming
  the mismatch.
- A restored batch carries an `ObservationLayout` equal by `config`; `gather`
  compares `config` and `observation_spans` compares `shapes`, so batches from
  different workers combine cleanly.

## Exclusions

- No version bump, changelog release entry, or tag; no `0.3.0` removal.
- No public raw-bytes `ObservationBatch(...)` constructor.
- No change to jace. No multi-column `[time, trajectory]` concatenation.
- No change to `_build_complete_batch`'s shared `RecordedSpans` in
  `collect_pair` (outside 0.2.6 compatibility scope; tracked in
  `scratch/concept-review.html` as separate follow-up).
- `_merge_recorded`-style dense merging is kept working, not blessed: dense
  transport still goes in `0.3.0`.

## What goes

Nothing is removed; this is a patch release and every touched API stays.
`ObservationBatch`'s docstring is updated to note pickle preservation alongside
native encoding. `freeze`'s ad-hoc `values.len().checked_div(count)` and
`gather`'s `stride.max` workaround are replaced by one shared stride helper.
New shim attributes sit under the existing `REMOVAL_RELEASE = "0.3.0"` marker
in `manabot/env/observation.py` so rollover deletes them with `encoder`.

## Passes

1. **Transport — done** (`8076ce311a`, notes `35237fe704`). Spawn-queue
   round trip with and without memory prepares identically to local
   collection. One local run (seed 7, Torch seed 71, 960 rows): batch pickle
   8,346,219 B (6 ms dump / 40 ms load), memory prefixes 3,037,646 B,
   selection after reading caches 8,171,210 B, space 633 B.
2. **Merge and shim — done.** Seeds 7 and 11, four complete-game streams each, Torch seed 71: 960 and 972 independently pickled and restored learner rows joined in 1.05 ms in one local CPU run, with 1,932 retained prefixes. The timing excludes collection and IPC.
3. **Consumer proof — done.** All 35 tests in the five named files passed in
   31.46 s on macOS arm64, CPython 3.13.15 and Torch 2.15.0. This includes
   the three fork regressions, the GRU merge and learner update, and the
   worker startup error test. Pytest reported 7,421 deprecation warnings.
   The isolated environment used unchanged jace `9d212cb`, manabot built
   from `fb5e32451d`, and this branch’s cp313 managym wheel. Installing
   manabot’s required Etude SDK 0.4.7 over jace’s locked 0.4.3 also moved
   websockets from 17.4 to 16.1.1. `uv pip check` passed.

   A separate call to jace’s `MultiProcessCollector.collect` exercised a
   successful queue transfer and jace’s dense merge with two spawned workers,
   one environment each, seed 7 and four requested steps per environment.
   Stateless self-play returned 332 learner rows. A snapshot opponent with
   GRU decision memory returned 285 learner rows and 669 recorded prefixes.
   Both pools closed, both batches ended on a terminal row, and their first
   and last chosen-action log probabilities matched the gathered policy rows.
   The remaining checks belong to gate and CI.

## Pass 2 checks

All eleven transport tests, all three shim cases, the package-boundary checks
(including MLX), and the two committed pilots passed on macOS CPython 3.13.
The optional frontier Small u1952 pilot replays its game, but its maximum
probability difference at decision 8 is 0.00208 (chosen-action rank 1 in both).
The same digest-verified checkpoint and recording produced that exact failure
with the unpatched Python sources and a clean release build of the native
engine from `183f6daf59` (same engine and training sources as `f75ba48c72`).
The release reviewer can reproduce the comparison with the cached checkpoint:

```sh
MANABOT_U1952_CHECKPOINT="$PWD/.runs/artifact-cache/c95c61aa298528e5c2782430d5e3f90929d3e9b7ac8b1fc45f560ed6ab4bbaf8" \
  uv run --no-sync pytest -q 'tests/model/test_pinned_pilot.py::test_recorded_game[u1952]'
```

## Done when

jace's fork and pool tests pass against this branch with no jace change, the
in-repo transport test passes in a fresh `spawn` child (with and without
decision memory), `ObservationSpace` and `ObservationSelection` (even post-`.spans`)
pickle compactly below dense export size, and `RolloutBatch.concatenate`
handles complete-game and snapshot-trajectory worker merges cleanly.

## Docs

- `docs/training-loop.md` — worker-process pickling of `RolloutBatch` / native
  spans / decision-memory prefixes and `RolloutBatch.concatenate` (plus
  `RecordedObservations.concatenate` and `PackedActions.concatenate`).
- `managym/README.md` — `ObservationBatch` pickle/copy transport, restore
  validation, and sealed constructor contract.
- `docs/engine.md` — deprecated `ObservationSpace` / `space.encoder` shortcut
  attributes (removed in 0.3.0) and steady-state replacements.
