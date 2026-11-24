# Typed, ragged observations: measured baseline and slices

Task 238767. Base `4127aa973f`. Measured 2026-11-23 on an Apple M4 Max (16
cores, 128 GB), CPython 3.13.15, managym built by `uv sync` (release, mimalloc,
no allocation counting), Torch on CPU, `torch.set_num_threads(1)`.

**Load caveat.** The machine was shared during every reading (load average
140–185 on 16 cores: other checkouts' rustc and an `etude launch`). Wall
times are unusable, and `scripts/bench_training_update.py` (256 streams, 2
updates) hit its 240 s subprocess timeout at 13% CPU. Every timing below is
process CPU time (best of 5–7, median across snapshots). Counts, bytes and
exactness checks do not depend on load. No speed claim rests on these numbers
until the deferred same-machine commands below are run on an idle machine.

## Workload

`scratch/bench_tla.py:frontier_v11`: `experiments/tla_frontier.py` small
settings (17Lands TLA trophy corpus, 256 streams, 4 producers, 64 actions
`first-legal`, 512 facts, 128 cards / 96 permanents per player, 32 events)
with `policy_input_version=11`. The frontier itself runs v9, the deprecated
fork; v11 is the input this change targets. Probes play random legal actions
(`ShardedVectorEnv`, seed 24), not current-self, so board shapes are a proxy
for trained play.

## Baseline

### Buffer occupancy (`scratch/occupancy.py`, 400 batches, sampled every 8th)

- Dense v11 buffers: **235,224 B per row, 60.2 MB per 256-stream batch**;
  mean nonzero bytes **0.79 MB (1.3%)**.
- Largest buffers, with live rows (mean / p99 / max) against capacity:

| buffer | MB/256 | capacity | live mean/p99/max |
|---|---|---|---|
| effect_details | 12.58 | 4096 | 0.47 / 5 / 24 |
| agent_cards | 7.34 | 128 | 25.1 / 40 / 55 |
| opponent_cards | 7.34 | 128 | 5.1 / 19 / 26 |
| facts | 6.29 | 512 | 5.7 / 62 / 190 |
| agent/opponent_permanents | 5.11 each | 96 | 8.4 / 20 / 33 |
| effect_members | 4.19 | 2048 | 1.8 / 18 / 123 |
| relations | 3.21 | 64 | 2.4 / 16 / 42 |
| effects | 2.36 | 128 | 1.4 / 15 / 36 |
| actions | 1.64 | 64 | 3.7 / 11 / 29 |
| stack | 1.21 | 32 | 0.15 / 2 / 9 |

Raw: `scratch/occupancy-v11.json`. A row counts as live when any element is
nonzero, because facts, effects, details, members and relations have no
validity array or count in today's transport.

### Host stage CPU per 256-row batch (`scratch/stages.py`, 8 snapshots)

| stage | CPU ms |
|---|---|
| native step + encode (4 producers, all threads) | 38.5 (150.5 µs/decision) |
| retention: fancy-index copy of every key | 2.5 |
| admission (`check_observation`) | 16.6 |
| `prepare_admitted`, learning | 81.3 |
| `prepare_admitted`, CPU inference | 70.3 |
| forward, CPU inference, 1 thread (scale only) | 445 |

Preparation costs about twice what the engine spends stepping and encoding.
Raw: `scratch/stages-v11.json`, cProfile `scratch/prepare-profile.txt`.

### Where preparation goes (cProfile, 10× admission + learning prepare)

- `readings.read_numeric`: **54% cumulative** (47% own time). It loops over
  columns across every padded row.
- Then `admission.facts` 8.5%, `astype` 4.8%, `np.zeros` 3.2%, `isin` 3.4%,
  `_facts.padded` 3.8%, `_referenced_definitions` 3.2%, enum `__repr__` 2.3%
  (16k calls).

`read_numeric` by call site, per batch (`scratch/read-numeric-sites.json`):

| site | CPU ms | rows read | nonzero rows |
|---|---|---|---|
| cards (`prepare.py:715`) | 28.6 | 65,536 | 8,191 (12.5%) |
| permanents (`:650`) | 24.9 | 49,152 | 4,597 (9.4%) |
| relations (`:680`) | 6.2 | 16,384 | 654 (4.0%) |
| stack (`:769`) | 2.4 | 8,192 | 42 (0.5%) |
| everything else | 1.2 | already live-trimmed | |

Actions, facts and effects already arrive trimmed to live rows, by rescanning
validity. Object tables do not.

**Live-row probe** (`scratch/live_rows_probe.py`): gathering the live rows,
reading only those and scattering them into zeros is **bit-identical**
(`np.array_equal`) for cards, permanents and stack, and cuts those tables
from 42.8 to 6.7 CPU ms per batch (6.4×). Zero rows read as zero under every
reading (`one_hot`, `steps`, `log`, `rank`, `as_is`).

### Native padding writes

Zero-filling every buffer at 256 streams costs 1.4 CPU ms (numpy `fill`,
single thread), about 4% of native step+encode. The encoder zero-fills all
object, action and event buffers on every encode
(`observation_encoder.rs:872-953`). Removing padding alone would buy little
native time; the win is downstream, on the host.

### Rollout storage (analytic, from the store's own rule)

`ObservationRows` keeps `[capacity, *shape]` dense arrays: 227,032 B per row
for non-action keys, with capacity a quarter above the previous update's rows
and doubling on growth. `transitions=64` with complete-games closure is at
least 256 × 64 = 16,384 rows per update: **≥3.7 GB dense observations per
update (≥4.6 GB allocated with headroom)**. At the measured 1.3% nonzero, a
ragged store holding live rows plus int32 offsets would be about 3–5 KB per
row, roughly 50–80 MB. This is not measured yet (see deferred).

## Deferred: same-machine timings on an idle machine

Run each before and after a slice, on this M4 Max with no other builds:

```bash
uv run python scripts/bench_training_update.py --recipe scratch.bench_tla:frontier_v11 --streams 256 --updates 3 --json scratch/update-v11.json
uv run python scratch/stages.py
uv run python scratch/occupancy.py 400
/usr/bin/time -l uv run python scripts/bench_training_update.py --recipe scratch.bench_tla:frontier_v11 --streams 256 --updates 3   # peak RSS
uv run python scripts/engine_budget.py ...   # per docs/development.md#engine-performance-budget
cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- --layers observation
cargo run --release --features benchmark --example bench_observation --manifest-path managym/Cargo.toml
```

The update benchmark also records rows per update, which replaces the
analytic storage estimate above.

## Slices

Ordered by profile share. Each must keep complete choices, `first-legal`
overflow semantics, typed `ObservationCapacity` errors, native skipping,
replay digests, pinned pilots and the debug cache invariants.

### 1. Done (`fe46e6b375`) — skip zero padding in host numeric readings

Object readings skip all-zero rows (not invalid rows: masked external rows
may carry numeric features that `historical_mean` pools); relations read
only valid rows. Paired against base on the eight `scratch/stages.py`
snapshots: every `Prepared` array `np.array_equal`; median CPU ms/batch,
shared machine:

| Preparation | Base `4127aa973f` | Slice 1 |
| --- | ---: | ---: |
| Learning | 72.47 | 34.59 |
| Inference | 68.76 | 22.02 |

v9/v11 pinned games, layer-input checks and sparse-coordinate regression
passed; the optional u1952 checkpoint was not configured. Whole-update
throughput and RSS remain deferred (above).

### 2. Native emits exact live counts; Python stops rescanning

- **Next:** emit exact v11 counts with immutable, native-owned CSR values
  and offsets. Use the native representation to skip checks on binary
  masks, offset partitions and native index bounds. Keep model-specific
  vocabulary and saved-column checks. External dictionaries and writable
  copies keep their full admission and dense preparation semantics. A NumPy
  writeable flag alone cannot establish origin or exclude writable aliases.

- **Hypothesis:** admission (16.6 ms) and the trims in `prepare`/`_facts`
  (`padded`, `nonzero`, `isin`, `any(0)`) shrink in proportion to live rows
  once each table carries `count[B]` / `offsets[B+1]` written at encode time.
  The native cost of writing counts is under 1% of encode.
- **Deletes:** validity-rescan extents (and `axis_extents`, if #583 lands
  first), the `_valid` float arrays as a source of counts, and the `1e-12` /
  `(x - 1) + 1` float guards on references.
- **Check:** native directed test of counts on a constructed position;
  `tests/env/test_observation.py`, `tests/verify/test_native_buffer_capacities.py`,
  `tests/training/test_action_cutoff.py`; admission rejection contracts.

### 3. Typed columns: int32 references, ids, kinds; uint8 flags

- **Hypothesis:** removes the float→int `astype` calls (4.8%) and the 16-bit
  value halves, and shrinks the live bytes further. Correctness and
  simplicity come first; the speed gain is small.
- **Deletes:** float-encoded references, split i32 halves in transport,
  mixed `action_focus` dtypes (`collection.py:1140` float32 convention).
- **Check:** pinned pilots and saved `input_readings` (logical v11 columns
  unchanged); package boundaries; Academy/LLM text paths.

### 4. Ragged retention and rollout storage

- **Hypothesis:** per-update observation storage falls from ≥3.7 GB to under
  100 MB. Peak RSS falls by GBs at 256 streams, and minibatch gather copies
  live bytes only.
- **Deletes:** dense `[capacity, *shape]` `ObservationRows` arrays, and
  re-padding before the learner. A minibatch is a gather over CSR row spans.
- **Check:** gathering arbitrary indexes reproduces the dense rows exactly;
  `tests/training/test_batch_closure.py`, `test_self_play_loop.py`, resume
  tests; RSS from `/usr/bin/time -l` on the deferred workload.

No native speed slice is planned. Native step+encode is 38.5 ms, and its
padding writes are about 1.4 ms of that; nothing in this profile points at
rules work.

## Large boards (`bench_states --layers observe`, under load)

`duplicate-tokens` at sizes 8/32/64 (20/68/132 permanents) observes in
0.022/0.043/0.087 ms at priority, and `mixed` 4/8/16 in 0.026/0.040/0.052 ms:
linear in rows, digests `70e47630…` and `523886bd…`. In these states live rows
approach capacity, so they bound the ragged form's worst case. Offsets cost
one integer per row-group, so the form adds no growth term. Rerun them idle
before and after slices 2–3 and compare the digests. Raw:
`scratch/bench-states-observe*.jsonl`.

`bench_observation` panics at `examples/bench_observation.rs:43` (no
pass-priority action before main phase) on the base commit, so it gives no
encode-only number (`scratch/questions.md`).
