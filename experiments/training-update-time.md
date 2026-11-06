# Training seconds per update — b015d5

2026-11-05. Why self-play updates went from 28-36 s (MTG-181) to ~53 s (MTG-252),
what was fixed, and which machines to rent. No recipe, world or observation
version changed. No GPU was rented: GPU rows are retained receipts, and the
"after" GPU figure below is a projection, not a measurement.

## Finding

**The slowdown is mostly the rented host, not the world.** The same w13 recipe
and GPU took 29.8 s on one host and 53.5 s on another. On a fast host, w13 to w17
costs 10% per update. Training occupies exactly one CPU thread, so the machine's
single-thread speed sets the pace; extra vCPUs and a larger GPU buy nothing.

| Suspect | Verdict |
|---|---|
| Host CPU class | **Main cause**: 1.7-1.8x at fixed world and GPU |
| Current world and input | +28% per decision, +10% per update on a fast host |
| Longer games | Refuted: games per update flat, decisions per update down 14% |
| Few vCPUs | Refuted: every run used about one core; the recipe permits one producer |
| Inference at batch 256 | Host-bound: same on L4 and A40, changes with host |
| Learner targets (not suspected) | Largest removable cost: 15-27% of an update |

## Retained GPU receipts

256 streams, two seats, seconds per update, mean over runs with 30+ updates
(72 or more each). Hosts are told apart by speed and by the node's reported CPUs,
memory and kernel; receipts did not name the CPU model, so "fast" and "slow" are
inferred. `TrainingRun.host` now records it.

| World | GPU | Node CPUs | Host | Runs | s/update | Decisions | Collection | Learning | Targets |
|---|---|---|---|---|---|---|---|---|---|
| w13 | L4 | 48 | fast | 1 | 29.8 | 56,437 | 15.2 | 14.6 | 5.7 |
| w17 | L4 | 48 | fast | 3 | 32.7 | 48,557 | 17.6 | 15.1 | 5.0 |
| w17 | L4 | 128 | fast | 2 | 35.0 | 48,547 | 18.9 | 16.1 | 5.4 |
| w13 | L4 | 128 | slow | 1 | 53.5 | 57,369 | 23.5 | 29.9 | 14.2 |
| w17 | L4 | 128 | slow | 2 | 55.5 | 45,940 | 28.5 | 27.2 | 11.3 |
| w17 | A40 | 96 | slow | 9 | 55.7 | 49,343 | 30.1 | 25.5 | 9.4 |

Node CPU count does not identify the class: 128-CPU nodes appear in both. Kernel
does not either: A40 nodes on both kernels are slow.

Every row used 1.00-1.06 CPU cores at `actual_threads` 1. CUDA waits spin, so that alone
does not prove useful host work. The evidence for host-bound training is the
comparison: L4 and A40 match within a host class (policy inference 6.1-6.5 s vs
6.4-6.9 s), while the host class moves it (4.7 s fast, 6.3 s slow).

From w13 to w17 on a fast L4, native stepping and encoding grew 4.5 to 6.5 s, policy
inference 4.0 to 4.7 s and minibatches 6.8 to 8.2 s. MTG-213's ~150 s/update uses
different decks and was not attributed here.

## Fixed-hardware rows

One Apple M4 Max laptop, CPU device, real stage through `execute_regime`, 256
streams, 2 updates, µs per decision. Other work shared the machine (load 15-30):
identical code moved 90.8 to 106.3 s/update between `old` runs, so read the ranges,
not single values. Model phases on CPU are not GPU evidence.

| Phase | 183-era source | Current, before | Current, after |
|---|---|---|---|
| Policy inference | 709-783 | 1096-1156 | 1095-1178 |
| Environment step and encoding | 61-74 | 100-104 | 96-101 |
| Rollout recording | 7-9 | 41 | 12-13 |
| Bookkeeping (untimed before) | not measured | ~45 | 9-10 |
| Targets | 19-27 | 20-23 | 1.1-1.2 |
| Selection diagnostics | 6-8 | 8-9 | 0.9-1.0 |

The same recipe grew from 196,482 to 1,085,122 parameters and from 38,120 to 88,816
observation bytes per row between those sources. Of the current environment step,
native stepping is 69-73 µs and encoding about 10 µs, so encoding the larger
observation is not the cost. One sequential row at 128 facts instead of 512 took
101.6 s/update against 113-121 s, with policy inference near 998 µs. That is one
row under this noise, so it suggests a cost of about 10% on CPU and no more; it is
attribution only and names no recipe.

Six interleaved base/new pairs at 64 streams (ABBA order, so slow load drift cancels):

| Phase, µs per decision | Base min | New min | New/base per pair |
|---|---|---|---|
| Targets | 20.9 | 3.0 | 0.09-0.14 |
| Selection diagnostics | 7.2 | 0.9 | 0.09-0.12 |
| Rollout recording | 37.8 | 13.2 | 0.23-0.46 |
| Optimizer step | 55.5 | 49.4 | 0.52-0.90 |
| Minibatches (model unchanged) | 1203.9 | 1172.5 | 0.42-0.97 |

Sequential 256-stream rows showed minibatches at 841-961 µs after against 793-796
before. The pairs did not reproduce that, and all six favored the new source even
on unchanged model code, so neither direction is claimed for minibatches. A
256-stream interleaved run was not made.

Targets on Apple MPS for a ~48,000-row packed column: GAE 14.4 s to 6 ms, terminal
distances 41.6 s to 9 ms. That is the launch-latency mechanism, on a device other
than the one training rents.

## Changes

- **Attribution.** Receipts and the "Update time" Perf panel split collection into
  native stepping, observation encoding, preparation, inference, bookkeeping and
  assembly beside learning. `TrainingRun.host` names the CPU, schedulable CPUs
  and cgroup quota; the GPU stays in run identities. Older rows leave new series blank.
- **Targets.** A complete-game batch is one column of ~50,000 rows, so the recursion
  launched ~50,000 dependent device steps. `reverse_waves` advances every trajectory
  at once on the host; each row keeps its float operations and order.
- **Collection.** Pack actions once per step, refresh only restarted streams,
  assemble batches per key, vectorize `selected_types`.
- **Learner.** One finite check over all parameters; one transfer per dtype.

Same seed, old and new source, 3 updates at 24 streams: raw weights, EMA weights,
optimizer state and all counters are bit-identical (`7f1baaeda750ff2f`,
`1abb26d025850dbd`, `0de32d30eb8ee7ce`). This ran on CPU; CUDA is not checked.

Projection, unmeasured: scaling receipts by these reductions gives about 24 s on
a fast L4 (from 32.7) and 37 s on a slow host (from 55). The first run after merge
replaces this with a measurement.

## Machine recommendation

- **Choose single-thread speed, not vCPUs.** Do not pay for vCPUs above the current
  6-vCPU floor for one run. Lower floors were not tested.
- **Rent L4, not A40,** for this recipe: A40 was no faster on matching hosts.
- **Screen hosts.** Compare `time/collection/environment_step_and_encoding/native_stepping`
  and `host.cpu_model` in the first updates; a run at about 1.7x the expected pace
  is on a slow host. Once receipts name CPU models, a list of good ones can replace
  this check.

Next levers, not attempted: `elemental-uniform` requires one producer, so stepping
(about 20% of an update) cannot overlap inference; inference runs ~700 calls of ~76
rows per update, which rewards fewer launches. Packing several seeds on one host
was not tested.

Reproduce: `uv run python scripts/bench_training_update.py --streams 256 --updates 3`.
