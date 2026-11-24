# Typed, ragged observations

Task 238767. Jack Heart's task direction (comment `e1a01d1d-9cda-435e-80bf-16f66e372443`):

> Make sure the design and implementation explicitly address all three sources of friction at the managym -> manabot observation boundary:
> 1. Exact live counts emitted at encode time (ragged/CSR instead of fixed-capacity zero-padded boxes) so manabot stops moving ~60 MB of turn-1 zeros and stops rescanning extents (axis_extents).
> 2. Typed columns/buffers (int32 for IDs/refs/enums/offsets, uint8/bool for masks/flags, float32 only for real scalars) instead of packing ints/enums/booleans into float32 (eliminating float32 rounding hazards, split 16-bit halves, and repeated float-to-int casts).
> 3. Trusted native output vs untrusted dicts: eliminate redundant re-validation in admission of invariants (binary masks, CSR offsets, bounds, integer domains) that managym already guaranteed at encode time — e.g. via an immutable/read-only native-produced observation representation — while keeping full validation for untrusted/mutated external dict inputs.

This is a v11 transport change. Preserve logical v11 column names, meanings
and saved readings, checkpoint vocabularies, and the deprecated v9/v10
paths. Managym owns the layout and native joins. Manabot binds checkpoint
lookups and prepares the sole Torch and MLX input, `Prepared`. Preserve
complete choices, configured `first-legal` behavior, typed capacity errors,
forced-decision skipping, replay identity and debug cache invariants.

## Delivered

1. `fe46e6b375` skips all-zero dense object rows and reads only valid relations, matching every `Prepared` array on eight paired snapshots.
2. `f21f8fe4f5` carries immutable CSR spans from v11 live frontiers and unmasked vector steps through admission and preparation, including native search.
3. `cd7f023e25` emits typed v11 CSR directly from the shared encoder and prepares it for Torch and MLX. Eight paired batches matched the dense reference for every projected buffer, outcome and learning and inference `Prepared` field. Directed fixtures cover exact signed words and event IDs beyond float32 precision.

`b9b09bc6d9` added `axis_extents` to dense admission for gaps, nonzero
masked features and referenced empty slots. Native rows use encoder offsets.
Both paths still bind checkpoint layouts and vocabularies.

4. `f19935ecd7`, `19a52eb37c`, `583f8d25f7`, `3b641ff19e` retain typed v11
   spans from masked and isolated vector steps, root changes and fixed-view
   reads through sharded collection, bootstrap and learner gathers; native
   spans own v11 legal action fields and behavior logs keep original slot
   coordinates. Editable exports and legacy dense inputs keep full admission.
   The paired test in `tests/training/test_collection.py` (seed 24, three
   streams, two-row chunks, repeated out-of-order gather) compares
   observations, actions, behavior logs, values, outcomes, model outputs,
   replay prefixes and native state digests. `aa6cf95de3` records the
   measurements below.

## Remaining

No implementation slice remains. Open, for gate or a person on an idle host:
steady-state update throughput and peak learner RSS at 256 streams. Contrary
finding to keep visible: step and encoding is slower with typed CSR
(+1.25 CPU ms per 256-row batch; 8.36 → 10.99 wall s over three four-stream
updates), outweighed end to end by admission, preparation and memory savings.

## Performance review (2026-11-23)

`12dd3bd05b` (one validity mask for native and dense facts, actions and
events) is Python-only; it changes no engine source, so the engine
measurements below still describe the branch. Preparation, pinned pilot,
observation, bucketed shape, decision memory, collection parity and package
boundary tests pass on it (270 passed, 1 skipped). Supported gains: admission
12x cheaper, learning preparation 36% cheaper, retained rollout bytes about
45x smaller, peak RSS 45% lower, learner throughput 8% higher on four streams.
Negative result kept: typed step and encode costs +1.25 CPU ms per 256-row
batch. The bottleneck still worth attacking is that encode cost, after
repairing `managym/examples/bench_observation.rs` so it can be isolated; it
is a separate Task, not this transport change. Steady-state update throughput
and peak learner RSS at 256 streams remain unmeasured
(`uv run python -m scratch.compare_updates --streams 256 --updates 3` on an idle host).

## Measurements to carry forward

`scratch/compare_spans.py` ran eight paired batches through step 320 on an
Apple M4 Max with CPython 3.13.15, release managym with mimalloc and
single-thread CPU Torch. The workload uses `scratch/bench_tla.py:frontier_v11`
(small TLA frontier, CPU, v11, 256 streams, 64 `first-legal` slots) with
random legal actions from seed 24. Transport shrank from 60.2 MB dense
to 3,797,400 bytes with float CSR, then 1,858,818 bytes with typed CSR.
At 16,384 rows the median typed batch projects to 119 MB of transport.
Measured rollout storage and process RSS appear below.

| Stage (median CPU ms per batch) | Dense | Typed CSR |
| --- | ---: | ---: |
| Step and encode with outcomes | 30.43 | 31.68 |
| Admission | 10.64 | 0.87 |
| Admitted learning preparation | 30.82 | 19.66 |
| Admitted inference preparation | 22.67 | 19.71 |

Step timings include native advancement and freezing. The typed result is
1.25 CPU ms slower at that boundary. Slice 3 test results and earlier stage
probes are in `cd7f023e25:scratch/plan.md`. Optional u1952 was unavailable.
Strict all-feature Clippy reported four existing PyO3 binding lints.

### Complete frontier updates

`scratch/compare_updates.py` measures a complete two-seat update using the
same frontier recipe, action cap, 64-transition quota, four workers, seed 24,
one Torch CPU thread, and release extensions. Each reading starts a new process.
Retained bytes count distinct NumPy backing allocations, CSR records and offsets,
including bootstrap and behavior evidence. Process peak RSS also covers the
model, optimizer, allocator and Python objects. Collection and learner times can
overlap with prefetch. Memory capture sits inside observation retention.

On four streams, `4127aa973f` and `583f8d25f7` produced 878, 1,336 and
879 learner rows on successive updates, with identical SHA-256 digests of
the sampled actions on each update. `scratch/update-v11.json` retains each
reading. MB and GB use decimal units.

| Three paired updates | Dense | Typed CSR |
| --- | ---: | ---: |
| Retained rollout MB by update | 291.4 / 499.3 / 380.0 | 6.67 / 10.80 / 6.46 |
| Peak training process RSS (GB) | 1.147 | 0.630 |
| Total wall time (s) | 207.7 | 192.4 |
| Total process CPU (s) | 79.2 | 74.7 |
| Learner rows per wall second | 14.89 | 16.08 |
| Step and encoding (wall s) | 8.36 | 10.99 |
| Observation retention (wall s) | 0.199 | 0.228 |
| Collection preparation (wall s) | 11.42 | 10.68 |
| Learner admission including gather (thread CPU s) | 0.314 | 0.121 |
| Learner preparation including gather (thread CPU s) | 2.451 | 2.132 |

At 256 streams the typed collector closed a 59,252-row batch using
443,982,033 retained bytes. Process peak RSS at closure was 1,383,825,408
bytes. Neither revision completed its first full-size optimizer update
within 240 seconds. A separate dense collection-only run also reached
240 seconds. Concurrent host activity varied across the sequential runs.
The four-stream timings establish observed throughput for these runs.

Both constructed-board probes matched the base at all 15 checkpoints,
including the output counts and digests. The separate encode-only fixture
panics on the base and yielded no measurement.

```sh
cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- duplicate-tokens --sizes 8,32,64 --layers observe
cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- mixed --sizes 4,8,16 --layers observe
uv run python -m scratch.compare_updates --streams 4 --updates 3
```
