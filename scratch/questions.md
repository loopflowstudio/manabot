# MTG-268 — batched local-update rollouts

The Task directive is the design; no separate accepted design was present.
It says: "Batching should bring 8-16 worlds near today's one-world cost."
That outcome remains unmet. No numeric definition of "near" or relaxation has
been agreed. Before the latest upstream sync, the direct semantic-hash path cost
3.69x/4.16x the old
singleton, versus 4.55x/4.92x after fact compression, 5.46x/5.72x after native
optimization and compression, and 8.0–8.1x before native optimization. These are single samples, not a variance
estimate. Implementation progress is not acceptance.

Prior direction: "Investigate the 0.259 s semantic JSON/hash cost" and
"Preserve exact viewer/event semantics and hash bytes; do not substitute a
witness hash without proving equivalence." That investigation is complete.
The current encoding investigation is also complete: the merged ordinary wheel
is installed, pending-entrant hash parity passes, and the teacher uses the
existing native encoder with bulk output copies and one fresh batch allocation.
Sixteen worlds take 4.55x/4.57x the retained merged-runtime singleton cost;
these are a new world/checkpoint, not comparable to the original ratios above.

Reversible choice: compute semantic SHA-256 under the Env lock on every read,
without reuse. Witness BLAKE3 is not equivalent. The bounded investigation is
complete; focused merged-runtime checks now pass, broader validation remains
for gate/CI, and the latency objective is open. Pre-sync parity below remains
historical; current evidence is in the encoding investigation section.

## Approach and Delete — do not maintain

Keep one live frontier, independent policy RNGs and stable world/action receipt
order. Preserve all RolloutReceipt fields, private audits, native Commands,
canonical hashes, legal support, terminal/sign semantics, sampling identities,
atomic precondition rejection and deadlines. No input/checkpoint/schema change
or batching flag. Compound execution is removed; historical readers survive.
The directive's compound fallback is obsolete under the existing removal in
`docs/training.md#compound-decisions-removed`; restoring it is outside this work.
The 2026-11-08 refresh found `origin/main` at `0aa7dabbd` (#433), already
included through sync target `00c7f1bba` in merge `0f5380bc9`. That sync adds
retained resolution/selection and replacement state, observation projections
and version-aware continuation encoding. The Task status supplied no new
direction. The earlier `841b1c50c` remains the historical benchmark reference,
not the current upstream revision.
The Task has no Wave, so there is no selected Wave memory to curate.

Delete — do not maintain (completed):

- Sequential production rollouts and separate singleton inference; the singleton
  entry point delegates to the batch implementation, and git supplies the oracle.
- Full Python semantic parsing at each ply and repeated root/config parsing.
- Globally empty CPU attention columns; restore coordinates and historical pooling.
- Pre-Command viewer projections and duplicate equivalent acting-view serialization.
- Rechecking fields of a Command just built from validated inputs under both locks.
- Embedding CPU fact columns unused by the entire inference batch; no cached state.
- Benchmark receipt-field allowlist; compare every field with explicit exceptions
  for run identity, timing and separately checked numerical values.
- Duplicate fact-word decoding and repeated boolean validity conversions; one
  `_word` helper and one validated mask serve compaction, encoding and joins.

- `_viewer_state_hash`'s Python JSON extraction and the scripted fixture's
  unused composite envelope. Keep the composite API for its real consumers.
- Duplicate prepared-constraint parsing: one decoded value serves validation
  and the world identity. Evidence tests share their match configuration.

Encoding cut completed after measurement: scalar Python-list conversion in
native array returns and per-row Python encoding/stack-copy assembly in
`predict_batch` are removed. The Python reference encoder remains for its real
consumers and parity oracle. Exact arrays, selected capacities, full action
support and independent writable outputs survive. Duplicate native profiling lifecycles and fact-program guards are also removed;
those responsibilities each have one owner. No next deletion is selected.

## Historical evidence and open work

Complete pre-sync measurement tables, commands, contrary findings and sync
resolution are retained at `0f5380bc9:scratch/questions.md`; earlier native
attribution is at `6b4a511:scratch/questions.md`, and fact compaction at
`f9e3801:scratch/questions.md`. Before this curation, the full uncommitted notes
were copied to `.cache/mtg-268/questions-before-encoding-compress.md`.
Raw timing JSON under `scratch/` and original receipts under
`.cache/mtg-268/reference/` remain unchanged. The reference execution owners are
loaded from `841b1c50c`, not installed as a production fallback.

Historical limits: all timings use two initialized opening positions, depth 20,
one CPU/inter-op thread and one sample per cell on Apple M4 Max/macOS 26.0.1,
Torch 2.10.0. They establish no variance estimate or production guarantee.
The direct-hash cut passed complete discrete parity with maximum base/value/target
errors 1.49e-10 / 5.96e-8 / 6.92e-9. Composite constructions fell 1,348 → 52;
16-world/original-singleton ratios were 3.69/4.16. Contrary evidence: position
43's singleton rose from 0.1261 to 0.17281 s. These are pre-sync measurements.
The teacher-source digest was repaired to include `facts.py`; retained receipts
predate that identity repair and were never rewritten.

The installed pre-sync binary was identified as
`21865ee8f4b9bbde2ef6463af4d583a8edd6b305c9b5d4a98e95fe813df61476`.
Rebuilding after merge `0f5380bc9` exposed the cross-world mismatch described
below. The completed rebuild and same-world checks do not waive original
receipt identities or rebind MTG-183/w13.

Native attribution requires `--features python,benchmark` (explicit features
replace maturin defaults). Those builds also instrument layers/allocations;
never compare their times with ordinary-wheel latency. Categories overlap.
Use `--retain-reference DIR` before replacing a runtime; `--native-reference DIR`
checks saved receipts. The original binary remains at
`.cache/mtg-268/reference-native.cpython-312-darwin.so`.

Gate owns complete local-update suites, affected model suites
(`test_value_aggregation`, `test_inference`, `test_agent`, `test_layer_input`,
`test_record_vocabulary`), and `tests/training/test_active_training.py`,
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`,
`tests/verify/test_native_buffer_capacities.py`. Native changes require the full
debug Rust suite, all-feature Clippy and applicable engine-budget CI checks.
Focused checks and release builds do not substitute for them.

Open for demo: what maximum 8–16-world/original-singleton latency ratio, and
which checkpoint and representative positions, define "near"? No answer or
relaxation is recorded. MTG-183/w13 remains unmeasured; no trained-strength
claim. Memory still scales with worlds × supported root actions. The README's
30-second example is a budget, not a latency promise. No additional optimization
or reuse is selected; canonical serialization and SHA-256 remain on every read.

## Encoding investigation (2026-11-08)

Direction: "Rebuild and atomically install the merged ordinary wheel; add
direct/composite hash parity coverage for the pending-public-entrant state."
Then: "Separate allocation, conversion and native encoding costs, and implement
a bounded reduction only where equivalence is demonstrated."

Merged ordinary comparisons pass all six cells against retained Python owners
on the same runtime (`scratch/mtg-268-merged-before.json`). The original-runtime
comparison fails at 42/1, retained in `scratch/mtg-268-merged-before.err`; original
receipts are unchanged. Merged references live separately in
`.cache/mtg-268/merged-reference`. World identity, policy identity, root hash,
history/constraints and rollout hashes/audits differ. Upstream changed catalog
definitions and observation/continuation semantics; freshly initialized model
bytes are now `57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`,
not the prior checkpoint. No cross-world speedup is claimed.

The standalone encoding probe checks every array on the 1,281 reference-frontier
observations exactly. Both encoders agree, but the existing native binding is
slower: 1.026 s versus 0.312 s profiled Python encoding, with 0.017 s NumPy
allocation. The Python probe includes cProfile overhead, so this is attribution,
not a paired performance estimate. Native output currently travels through
Python scalar lists; instrument that conversion before selecting a reduction.

`cargo test --manifest-path managym/Cargo.toml -p managym --lib hidden_selection_public_entry_is_pinned_through_both_samplers -j 2 -- --test-threads 2`
— passed in debug, now checking direct/composite/independent SHA-256 parity for
both viewers while a hand/library entrant waits for replacement, without
changing the authority witness.

Instrumented probe (`scratch/mtg-268-encoding-profile-before.json`) confirms
1.012 s of 1.028 s native encoding is output conversion/copy; native allocation
0.00416 s, native encoding 0.00462 s, buffer config 0.00267 s. Python-to-Rust
conversion is 0.00587 s across 2,562 calls (Python validation plus native encoding).
Reversible implementation choice: use PyO3's checked bulk buffer copy into
NumPy-owned arrays, then the existing validated copy into supplied buffers.
No new dependencies, borrowed pointer lifetime or buffer cache. The teacher
allocates its selected space once per frontier and encodes into contiguous rows,
removing the final stack copy. Full support rejects before native encoding even
when a checkpoint selected first-legal for training.

Final instrumented parity passes for all 1,281 reference-frontier observations
(`scratch/mtg-268-encoding-profile-after.json`). Native output conversion/copy
falls from 1.012 to 0.102 s; the full standalone native call from 1.028 to
0.117 s. Native allocation/encode remain 0.00403/0.00434 s. Python-to-Rust
conversion is 0.00561 s across both paths' 2,562 calls. These feature-gated probe
times and the cProfile Python timings are attribution only, not decision latency.

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders`
— six-cell full discrete parity against retained Python owners on the final
runtime AND saved merged-runtime receipts; exact arrays for 1,281 views.
Ordinary-wheel evidence: `scratch/mtg-268-encoding-after.json`, instrumented=false,
runtime SHA256 `0b286acd043f21ecfcf8ca7e9c3ad449ee38b3dffaa0aa8914c311d733d15a46`.
Checkpoint SHA256 remains the new merged-world `57fd8f99…42bddeb` above. Same
bounded positions 42/43, depth 20, one CPU/inter-op thread, seed 17, initialized
64-wide/two-layer scalar policy; one sample per cell, no variance estimate.

| Worlds | Position 42 before encoding cut → final (s) | Position 43 before encoding cut → final (s) |
| --- | ---: | ---: |
| 1 | 0.09170 → 0.08737 | 0.11086 → 0.10685 |
| 4 | 0.29665 → 0.26562 | 0.35659 → 0.36279 |
| 16 | 1.07097 → 0.99607 | 1.33296 → 1.25789 |

Contrary sample: position 43 / four worlds slowed slightly. The final 16-world
costs are 4.5483/4.5722 times the saved merged-runtime sequential singleton
(0.21900/0.27512 s), not the original-world singleton. Maximum base/value/target
errors are 1.0243e-10 / 4.0979e-8 / 3.1307e-9 (atol 1e-6, rtol 1e-5).
The comparator's discrete checks are unchanged. The separate final profile
reports encoding 0.1626 s (1,281 calls), batch allocation 0.00231 s (21 calls),
tensor assembly 0.000654 s, network 0.2558 s, step/audit 0.4745 s, direct hashes
0.1670 s, snapshots 0.00741 s. Categories can overlap; do not sum them.

`uv run --extra dev pytest tests/sim/test_search_command_evidence.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py tests/verify/test_native_buffer_capacities.py -q -k 'semantic_direct_hash or selected_command or real_root_replay or batched or frontier or batch_ or native_encoded_arrays or companion_shapes'`
— 36 passed; nine new history-on fixtures initially failed model construction
because they lacked `recent_events`/world settings. The fixture was corrected,
without changing production admission.
`uv run --extra dev pytest tests/sim/test_local_update_sampling.py -q -k batch_inputs_match_python_at_selected_capacities`
— all 18 cases passed: inputs v1–v9, history off/on, nondefault card/permanent/
action/focus/event/fact capacities, complete tensor and prediction parity.
Thus all 45 distinct selected Python cases pass across the two runs; no unresolved
failure. Direct-array tests also check writable independent f32/i32 storage;
shape rejection preserves caller buffers. Focused Ruff check/format, Cargo fmt
and `git diff --check` pass. Both feature-gated and ordinary release builds pass;
the ordinary wheel was atomically installed last. Broader checks remain for gate.

No new latency, trained-strength or acceptance claim. Pending work: the existing
broad gate/CI suites, the definition of "near", and MTG-183/w13 admission/evidence.

## Encoding compression (2026-11-08)

The benchmark shares one optional native-profiler lifecycle, including cleanup
on failure, and binds timing wrappers once per encoder probe. Native cost JSON
is narrowed at that boundary. Fact-program admission and kind selection live
only in `_programs`; older inputs still skip program work. No Rust changed in
this pass, so the ordinary wheel from the encoding investigation stays installed.
The teacher-source digest changes with the fact source; historical receipts
remain unchanged. Headless choice: retain both reference paths and the Python
encoder oracle, since they still serve independent parity checks.

`uv run --extra dev pytest tests/model/test_layer_input.py tests/model/test_record_vocabulary.py -q -k 'fact or local_program'`
— 18 passed, 20 deselected.
`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders`
— six-cell complete discrete parity against both reference owners and saved
merged-runtime receipts; exact arrays for 1,281 views. Ordinary-wheel output:
`scratch/mtg-268-encoding-compress.json`. Maximum base/value/target errors:
1.0243e-10 / 4.0979e-8 / 3.1307e-9. This is a parity recheck, not a speedup claim;
the focused tests briefly overlapped this run's startup. A bounded injected
lifecycle probe passed absent-timer, successful-timer and exception cleanup
cases. Focused Ruff check/format and `git diff --check` passed.

Historical notes were curated using the backup and git references above;
`lf context` remains below both scratch targets. Broad gate/CI validation and
the unresolved acceptance questions remain as listed above.
