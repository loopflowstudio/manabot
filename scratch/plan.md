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

## Remaining

### 4. Use owned spans in training collection, retention and minibatches

- Feed the typed row writer into the native vector write callbacks in
  `managym/src/agent/vector_env.rs` and `managym/src/python/vector_env_bindings.rs`.
  `step_observation_batch` encodes after the unmasked step returns. That
  misses the write boundary where isolated stepping classifies capacity
  failures. Reuse the callbacks for masked and isolated steps, seeded resets,
  replacements and viewer reads. `PackedRows::append` currently assumes the
  caller discards the whole batch on error. Commit each observation atomically so a
  failed slot cannot affect a sibling's offsets. Keep stream order, outcomes,
  step events, cutoff diagnostics and typed errors through
  `managym/recording.py` and `manabot/training/rollout/producers.py`.
- Snapshot self-play (`82c5d1d1b7`, merged in `dbafc8738f`) now runs on
  native slots across shards. Preserve both terminal autoreset and parked
  slots, transactional snapshot loads, seeded root succession, and settlement
  of records, statistics and exposure for explicitly replaced roots. Freeze
  each horizon from its contributing viewer before releasing the slot. Keep
  decision-memory prefixes shared, with capture cursors tied to the native
  stream, viewer and epoch across snapshot branches and resume.
- Carry immutable v11 spans through `manabot/training/rollout/collection.py`,
  `actions.py` and `admission.py`. The collector currently copies dense
  non-action rows, then rejoins them with dense action projections from
  `PackedActions` on every learner gather. Keep one owner for the v11 legal
  action fields in the native spans, and retain behavior probabilities in
  original action coordinates. Gather arbitrary rows, including repeats and
  frozen bootstraps, with collection bucket extents and native provenance
  intact. Keep one sampling call per original row group and materialize
  prefixes within execution row limits. Mutable or external mappings still
  receive full admission.
- Check arbitrary gathers and cutoff/recovery, batch closure, self-play and
  resume, including sharded snapshot branches and fixed-view horizons. Compare
  logits, values, sampled actions and replay digests against the dense
  reference with paired seeds. Check pinned pilots and package boundaries.
  Measure rollout bytes, peak RSS and update throughput on the workload below.

## Measurements to carry forward

`scratch/compare_spans.py` ran eight paired batches through step 320 on an
Apple M4 Max with CPython 3.13.15, release managym with mimalloc and
single-thread CPU Torch. The workload uses `scratch/bench_tla.py:frontier_v11`
(small TLA frontier, CPU, v11, 256 streams, 64 `first-legal` slots) with
random legal actions from seed 24. Transport shrank from 60.2 MB dense
to 3,797,400 bytes with float CSR, then 1,858,818 bytes with typed CSR.
At 16,384 rows the median typed batch projects to 119 MB of transport.
Actual rollout storage and peak RSS remain open.

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

Compare base `4127aa973f` and final TLA self-play on the same idle machine.
The baseline update probe timed out at 240 seconds with 13% CPU on a busy
host, leaving throughput and RSS unmeasured. Record rows per update, timings
for encode, retention, admission and preparation, total rollout bytes and
training process peak RSS. Keep memory capture under retention.

```bash
/usr/bin/time -l uv run python scripts/bench_training_update.py --recipe scratch.bench_tla:frontier_v11 --streams 256 --updates 3 --json scratch/update-v11.json
cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- --layers observe
```

Follow the Engine Budget procedure in `docs/development.md` and compare the
`duplicate-tokens` and `mixed` observe digests (base `70e47630…`,
`523886bd…`, traces in `scratch/bench-states-observe*.jsonl`). These
constructed boards approach capacity. The separate encode-only fixture panics
on the base and yielded no measurement. See `scratch/questions.md`.
