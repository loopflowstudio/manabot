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

1. `fe46e6b375` skips all-zero object rows and reads only valid relations. The eight paired snapshots matched every `Prepared` array. Learning preparation fell from 72.47 to 34.59 CPU ms per batch and inference from 68.76 to 22.02. The v9/v11 pinned games and layer-input checks passed. Optional u1952 was unconfigured.
2. `f21f8fe4f5` carries immutable v11 CSR spans from live frontiers and unmasked vector steps through layout and vocabulary admission into Torch and MLX preparation. V11 native search uses it. The directed count, capacity, cutoff, external rejection, memory preparation and search checks passed. The paired transport result is below.

The task incorporated `b9b09bc6d9`, which added `axis_extents` for dense inputs
and normalized admitted references to integer slots. Dense mappings need that
scanner for gaps, nonzero masked features and referenced empty slots. Native
admission uses offsets and trusts encoded joins. Keep checkpoint-dependent
admission on both paths.

## Remaining slices

### 3. Emit typed v11 spans and consume them through Prepared

- Let `ObservationLayout` describe the physical types of the CSR fields.
  Write exact int32 IDs, references, enums, offsets and full record words,
  with uint8 or bool flags and float32 real scalars. Cover facts, shared
  effects, decision memory, actions and their focus, events and the
  remaining tables. Optional blocks and tagged payloads must retain their
  meaning and absence conventions.
- Slice 2's `batch` in `managym/src/python/observation_batch.rs` repacks a
  reusable padded `EncodedObservation` workspace. Marshal typed live rows
  from the shared encoder as it writes them. Keep one owner for row counts,
  column semantics and joins. A cast of the float workspace would still
  lose source integers before export. Eliminate padding writes and the
  separate float repack on the native typed path.
- Wire typed fields through `manabot/policy/native_rows.py`, `admission.py`,
  `prepare.py` and decision-memory preparation. Remove float ID/reference
  unpacking, split-word reconstruction and integer-domain rechecks from
  the native path. Follow `action_focus` through consumers, including the
  float tensor conversion in `manabot/env/vector_env.py`. Keep legacy and
  external dense formats readable with full strict checks. Preserve
  native type and checkpoint layout binding and retain one `Prepared`.
- Check exact signed words beyond float32 precision, flags and full
  dtype/layout coverage on a directed native fixture. Reuse
  `scratch/compare_spans.py` for parity of logical values, joins and
  `Prepared` under saved readings. Check pinned pilots, Torch and MLX,
  package boundaries, Academy, LLM text and retained replay/capture readers.

### 4. Use owned spans in training collection, retention and minibatches

- Migrate v11 collector destinations, row selection and rollout ownership
  together in `manabot/training/rollout/collection.py`. Reuse the counted
  export and preserve native reset, terminal autoreset, paused streams and
  isolated per-stream capacity recovery. The current
  `VectorEnv.step_observation_batch` uses serial `VectorEnv::step` and packs
  after advancement. It lacks the masked and isolated paths that the
  collector uses. Feed export into those existing paths with matching
  outcomes, events and error contracts.
- Replace v11 dense `ObservationRows` storage and learner repadding with
  arbitrary gathers of live spans. Preserve row order, repeated indexes,
  bootstrap rows, collection bucket extents and one sampling call per
  original group. Keep decision-memory prefixes shared and stream cursors
  tied to their native stream. Preserve native provenance through trusted
  selections, and use full admission for mutable or external inputs.
- Check arbitrary gathers and cutoff/recovery, batch closure, self-play and
  resume. Compare logits, values, sampled actions and replay digests with
  the dense reference using paired seeds. Measure observed rollout bytes,
  peak RSS and update throughput on the workload below.

## Measurements to carry forward

Apple M4 Max (16 cores, 128 GB), CPython 3.13.15, release managym with
mimalloc, CPU Torch with one thread. Baseline `4127aa973f` and slice 2
`f21f8fe4f5` ran on a shared machine (load roughly 140–185). Times below
are process CPU time. The baseline whole-update probe hit its 240 s
subprocess timeout at 13% CPU. Wall throughput and peak RSS are still open.

The workload uses `scratch/bench_tla.py:frontier_v11`, the small TLA frontier
settings on CPU with `policy_input_version=11`, 256 streams and 64
`first-legal` slots. The frontier recipe itself uses v9. The stage probes
use random legal actions (seed 24), a proxy for self-play board shapes.
The baseline moved 60.2 MB of dense buffers per step, with 0.79 MB nonzero
on average (`scratch/occupancy-v11.json`). Numeric readings accounted for
54% cumulative preparation CPU and casts for 4.8%
(`scratch/prepare-profile.txt`). Baseline stage CPU was 38.5 ms for native
step plus encode and 2.5 ms for retention copies (`scratch/stages-v11.json`).
A NumPy zero fill of the full batch cost 1.4 CPU ms.

`uv run python -m scratch.compare_spans` sampled eight paired 256-stream
batches through step 320. All dense projections, outcomes and learning and
inference `Prepared` fields matched, using slice 1's preparation as reference.

| Stage (median CPU ms per batch) | Dense | Native CSR |
| --- | ---: | ---: |
| Step and encode with outcomes | 28.72 | 33.61 |
| Admission | 13.89 | 1.01 |
| Admitted learning preparation | 30.09 | 24.88 |
| Admitted inference preparation | 22.95 | 18.71 |

The 4.89 ms producer difference includes stepping, packing and freezing.
The probe did not isolate encoding. Median live values and offsets used
3,797,400 bytes, including all-zero live rows and zero columns within live
rows. This projects to about 243 MB for 16,384 rows before headroom or
retention-specific compaction, against at least 3.7 GB dense non-action
storage. This is a volume estimate. Recompute it from typed values in
slice 4. The full historical profiles and occupancy table are in
`f21f8fe4f5:scratch/plan.md`.

### Deferred same-machine checks

On an idle machine, compare base and final TLA self-play by stage (encode,
retention, admission, preparation), rollout bytes, peak RSS and update
throughput. Keep the recorded rows per update with the results.

```bash
/usr/bin/time -l uv run python scripts/bench_training_update.py --recipe scratch.bench_tla:frontier_v11 --streams 256 --updates 3 --json scratch/update-v11.json
cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- --layers observe
```

Follow the Engine Budget procedure in `docs/development.md` and compare the
`duplicate-tokens` and `mixed` observe digests (base `70e47630…`,
`523886bd…`, traces in `scratch/bench-states-observe*.jsonl`). These
constructed boards approach capacity. The separate encode-only
`bench_observation` panics on the base at
`managym/examples/bench_observation.rs:43`, where the fixture unwraps an
absent PassPriority action. That invocation produced no baseline.
