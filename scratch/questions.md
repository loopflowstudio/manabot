# MTG-268 — batched local-update rollouts

## Current direction (2026-11-09)

Jack Heart asked, "isnt this a performance optimization", then invoked
`$optimize`. Follow the optimize checklist in this conversation, measuring
equal search workloads before and after and preserving search semantics.
This invocation does not supply a numeric acceptance ratio or approve shipping.

Optimize step 5 decision: **advance**. Read every measurement table row and
the current `_run_rollouts`, `apply_policy_choice`, and semantic-hash paths.
The dtype trial removed its profiled mechanism but did not demonstrate a
latency gain across the six cells; retain its evidence and the restored baseline.
Command/audit remains the largest bucket, but no newly measured removable
substep is identified. Prior redundant projections/checks were already removed;
dropping retained witnesses or substituting hashes changes the contract.
Snapshots/allocation are too small, and this trial does not justify repeating
dtype work. The best remaining investigation is finer native Command/audit
attribution on a controlled host; it is not a selected implementation hypothesis.
Finish compress, sync, realign and final measurement. This decision concludes
the bounded optimization loop, without asserting the original latency target
or moving the associated Task's graph.

Optimize step 4 is complete: the dtype-validation cut has no measurable
end-to-end latency improvement on the five alternating samples per wheel/cell.
All six ranges overlap; sixteen-world medians fall 4.29%/5.75%, while all four
smaller-cell medians rise. Receipt parity holds and the encoding profile improves,
but the predicted latency gain is inconclusive. Following the invoked measure
skill, the dtype cut and its tests were restored to last-kept HEAD `44895a104`,
and the preserved ordinary baseline extension was atomically reinstalled.
All earlier batching work and all candidate evidence survive. See the final
measurement section and `scratch/measure.html`. Equal-work latency remains this
investigation's decision metric; the original singleton objective is still open.

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
The ordinary-wheel encoding measurement puts sixteen worlds at 4.55x/4.57x
the retained merged-runtime singleton cost; these are a new world/checkpoint,
not comparable to the original ratios above. The subsequent compression pass
rechecked parity; its timings are not a separate speedup claim.

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
The 2026-11-08 realign refresh confirmed remote `main` at `0aa7dabbd` (#433), already
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
those responsibilities each have one owner. The later dtype-name formatting
experiment was restored after an inconclusive latency measurement; it is not
part of the kept implementation.

## Remaining work and decisions

The requested merged-wheel rebuild, pending-entrant hash coverage, six-cell
comparison and bounded encoding reduction are complete. There is no remaining
implementation step from that investigation. The full latency outcome remains
unmet. The bounded dtype-validation experiment below is complete and restored;
it does not change the original Task acceptance claim. No further optimization
is selected by this measurement step.

Gate owns complete local-update suites, affected model suites
(`test_value_aggregation`, `test_inference`, `test_agent`, `test_layer_input`,
`test_record_vocabulary`), and `tests/training/test_active_training.py`,
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`,
`tests/verify/test_native_buffer_capacities.py`. Native changes require the full
debug Rust suite, all-feature Clippy and applicable engine-budget CI checks.
Focused checks and release builds do not substitute for them.

Open for demo: what maximum 8–16-world/original-singleton latency ratio, and
which checkpoint and representative positions, define "near"? No answer or
relaxation is recorded. The measured grid is 1/4/16 worlds; eight worlds have
not been measured. The merged-world singleton is a diagnostic comparison,
not an agreed replacement for the original acceptance baseline.
MTG-183/w13 admission and evidence remain unresolved; no trained-strength
claim. A changed world cannot silently rebind that checkpoint or its evidence.
Memory still scales with worlds × supported root actions. The README's
30-second example is a budget, not a latency promise. Canonical serialization
and SHA-256 remain on every read.

## Historical evidence

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

The initial standalone encoding probe checked every array on the 1,281
reference-frontier observations exactly. Both encoders agreed, but the native
binding before the cut was slower: 1.026 s versus 0.312 s profiled Python encoding, with 0.017 s NumPy
allocation. The Python probe includes cProfile overhead, so this is attribution,
not a paired performance estimate. Native output then traveled through Python
scalar lists; the instrumentation below isolated that cost before its removal.

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

## Optimize baseline protocol (2026-11-09)

Claim under test: "At depth 20 on the two fixed initialized opening positions,
16 batched worlds take fewer seconds per decision than 16 sequential worlds,
with identical discrete receipts and numerical predictions within the existing
tolerance."

This run is optimize step 1 only. Reversible headless choice: reuse the committed
`scripts/bench_local_update.py` unchanged, invoke its 1/4/16-world, 42/43-position
grid five times serially, and retain all outputs separately. Each cell runs
sequential then batched, so each side receives five alternating measurements.
Each process warms both policies; separate instrumented profile searches are
excluded from timing samples. No games, evaluations or contention workloads.
The existing receipt comparator and saved merged-runtime receipts gate every
cell. Earlier single samples are context, not members of this new baseline.

The comparison deliberately shares the already-built ordinary native wheel and
current dependencies while loading all three retained Python execution owners
from `841b1c50c`; it measures the Python search/model changes on this world,
not the whole historical branch against a different-world native build. HEAD
`44895a104` is the last kept implementation and is not being replaced in this
baseline step. No unkept production change needs a keep/revert decision.


## Optimize baseline result (2026-11-09)

**Better for equal work.** Five alternating sequential/batched samples per cell,
30 pairs, zero benchmark errors, all 30 full discrete comparisons and all 30
saved-receipt comparisons passed. Each side executed 945 rollouts / 18,900
Commands inside the timed searches. No retries or deadline rejections. No
production files changed; keep HEAD `44895a104` as the next experiment's baseline.

Seconds per decision: median [minimum, maximum]; ratio is sequential median /
batched median, not the median of paired ratios.

| Position | Worlds | Sequential seconds | Batched seconds | Speedup | Rollouts / Commands per search |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.25963 [0.20944, 0.29722] | 0.10078 [0.08264, 0.11796] | 2.58× | 4 / 80 |
| 42 | 4 | 1.05051 [0.80673, 1.19994] | 0.34665 [0.29120, 0.74524] | 3.03× | 16 / 320 |
| 42 | 16 | 4.24566 [3.69680, 4.32050] | 1.26913 [1.22429, 1.27389] | 3.35× | 64 / 1280 |
| 43 | 1 | 0.32751 [0.29087, 0.36039] | 0.12348 [0.10428, 0.13945] | 2.65× | 5 / 100 |
| 43 | 4 | 1.30797 [1.25916, 1.72466] | 0.41476 [0.38302, 0.48030] | 3.15× | 20 / 400 |
| 43 | 16 | 5.57084 [4.48144, 6.10790] | 1.53841 [1.25904, 1.67621] | 3.62× | 80 / 1600 |

All six observed ranges are disjoint. The 16-world paired speedup ranges are
2.910–3.487× (42) and 3.460–4.092× (43). Keep the noisy 42/four-world sample
(0.74524 s batched); its cause was not measured. These are five samples, not a
confidence interval. Both positions show substantially more time and work at 16
worlds than at one, confirming that the harness distinguishes workload size.

Largest absolute base/value/target errors across live and retained comparisons:
1.0242171e-10 / 4.09781933e-08 / 3.13064846e-09
(atol 1e-6, rtol 1e-5). Receipt support, hashes, Commands, audits, terminal flags,
sampling identities, world identity and checkpoint identity pass unchanged.
The two openings exercise full-depth continuations, not terminal-heavy search.

Separate ordinary-wheel cProfile, position 42 / 16 worlds: total accounted
batched time 1.22840 s, excluded from the five latency samples. The unchanged
harness already runs separate profiles; `scratch/mtg268_profile.py` additionally
retains full pstats and self-time totals from one extra diagnostic invocation.
Its unprofiled timing cell is also excluded from the reported baseline.

| Batched category | Calls | Profile seconds | Share of profile | Impossible full-removal ceiling |
| --- | --- | --- | --- | --- |
| native_step_and_audit | 1280 | 0.4950 | 40.3% | 1.675× |
| network | 21 | 0.2917 | 23.7% | 1.311× |
| semantic_direct_hash | 1296 | 0.1917 | 15.6% | 1.185× |
| observation_encoding | 1281 | 0.1899 | 15.5% | 1.183× |
| session_snapshot | 64 | 0.0090 | 0.7% | 1.007× |

Source mapping: `LocalUpdateTeacher._run_rollouts` calls
`SelectedBranchRuntime.apply_policy_choice` (`managym/src/python/bindings.rs:3064`),
including offer/Command validation, stepping, audit and returned observation
conversion. The Python profile cannot divide that native bucket further.
Hash reads reach `Game::semantic_viewer_state_hash` (`managym/src/decision.rs:354`)
and preserve canonical viewer projection / SHA-256. `predict_batch`
(`manabot/sim/search/local_update.py:538`) reaches native encoding at
`bindings.rs:4263`. Its buffer copy validates source and destination dtype names
at `bindings.rs:340` and `managym/src/python/convert.rs:53`.
The profiler attributes 56,364 NumPy `_name_get` calls, 0.09521 s inclusive, to
`encode_observation_into`: 7.75% of the full profile, at most 1.084× if eliminated
entirely. This is a bounded candidate for optimize-plan, not a selected change
or an unprofiled saving; cProfile amplifies Python call costs. Preserve all shape,
dtype, independent-output and atomic pre-write validation contracts.

The measured gain is explained by network forwards falling 1,281 → 21
(2.54701 → 0.29167 profile seconds), composite semantic reads 1,298 → 2 plus
1,296 direct hashes, and less encoding work (0.40101 → 0.18988 s). The native
Command count remains 1,280 (0.53381 → 0.49500 s). Snapshot work is under 1%;
removing it cannot deliver the requested scaling. Profile categories are
inclusive; do not add nested model or dtype costs to their parents. The full
network-removal ceiling is only 1.311× on the current batched profile.

Reproduction: five serial invocations of
`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference`,
then `uv run --extra dev python scratch/mtg268_profile.py --worlds 16 --position-seeds 42 --native-reference .cache/mtg-268/merged-reference`.
Raw results: `scratch/mtg-268-baseline-{1..5}.json`; aggregation:
`scratch/mtg-268-baseline-summary.json`; complete profile:
`scratch/mtg-268-baseline-{before,after}.pstats` and corresponding profile JSON.
The harness SHA256 stayed `3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a`.
Native SHA256 stayed `0b286acd043f21ecfcf8ca7e9c3ad449ee38b3dffaa0aa8914c311d733d15a46`;
checkpoint stayed `57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`.
Shared Apple M4 Max, macOS 26.0.1, Python 3.12.12, Torch 2.10.0, one intra/inter-op
CPU thread, warmed policies, seed 17, depth 20, initialized scalar 64×2 model.
uv created the checkout-local environment from locked dependencies on first run;
setup was outside all search timings. Both sides used that environment.

Limits for optimize-plan: these same-runtime Python-owner comparisons include
all retained search/model cuts, not batching alone or a separately built
historical native engine. No cross-world or trained-strength claim. Original
MTG-183/w13 admission and the definition of “near” remain unresolved. The new
16-world/current sequential-singleton median ratios are 4.89× / 4.70×, diagnostic
only. Eight worlds remain unmeasured. No optimization was implemented, reverted,
published or shipped in this baseline step; broad gate checks remain pending.

## Optimize plan — primitive dtype validation (2026-11-09)

**Iterate.** Replace successful-path NumPy dtype-name construction in native
observation buffer validation with comparison of primitive dtype `kind` and
`itemsize`, formatting names only for a mismatch. This removes repeated Python
formatting rather than removing validation. Hypothesis: this one change cuts
16-world batched decision medians by about **3–5%** (roughly 1.21–1.23 s at
position 42 and 1.46–1.49 s at 43 against the retained baseline), with identical
arrays and receipts. These are predictions, not measured gains or acceptance
thresholds. cProfile magnifies Python call costs; the 7.75% profile share is an
optimistic full-removal ceiling, not an expected latency saving.

Evidence and mechanism:

- Read every row of `scratch/measure.html`, including all six workload cells
  and all five profile categories, and the full baseline summary. The 30 live
  and 30 retained comparisons pass. The noisy four-world sample at 42 and wide
  sixteen-world range at 43 stay in the record.
- `predict_batch` encodes every frontier row through `encode_observation_into`.
  `fill_encoded_into_existing_buffers` (`bindings.rs`) reads each generated
  source array's `dtype.name`; `require_numpy_array` (`convert.rs`) reads the
  destination's name again. The retained pstats confirm 56,364 `_name_get`
  calls, all under `encode_observation_into`, taking 0.09521 s inclusive.
  This is 1,281 observations × 22 arrays × two names. Both dtypes already
  expose the primitive type and byte width needed for the comparison.
- A read-only local NumPy probe found that native and swapped-endian f32/i32
  share their name, kind and itemsize. Full dtype equality would tighten the
  current scalar contract: keep its endian conversion through `np.copyto`.
  The probe also found `int64` and `longlong` both named `int64` but with
  different `dtype.num` (7 and 9); avoid numeric type-code equality as a
  general replacement for the shared validator. Structured/subarray examples
  have kind `V`, so must not match a primitive float or integer of equal size.

Implementation boundary:

Use the source dtype's primitive metadata as the expectation in the scalar
copy path; keep one owner for shape, dtype, C-contiguity and writability checks
in `convert.rs`. Adapt its vector callers only as needed to retain their
existing primitive expectations, without duplicating validators or introducing
a per-key dtype table. Preserve error classes and useful dtype-name diagnostics
on failure. Limit the fast comparison to the supported primitive types; do not
generalize kind/size equality to structured or parameterized dtypes.

Validate **all** buffers before the first copy, including a bad final buffer.
Keep selected axes and version-specific keys, overflow/admission checks,
independent writable outputs, fresh frontier allocation and `np.copyto`.
No persistent buffer/config cache, direct pointer writes, encoder rewrite,
dependency, policy/checkpoint/schema change or change to Commands, hashes,
viewer projections, audits, RNGs, deadlines or terminal/sign handling.

Why this candidate: Command/audit is still largest (40.3%), but its current
profile does not identify another removable substep. Earlier duplicate witness
projection/validation cuts already landed; dropping witnesses or substituting
BLAKE3 for semantic SHA-256 violates retained evidence. Hash caching has no
new invalidation/equivalence proof. Native scalar-list conversion, stack copies,
empty attention columns and unused fact columns were already removed; this
profile exposes a different remaining cost. Snapshot removal is capped below
1%; allocation is about 0.2%. A new model or audit redesign is a larger,
less-supported experiment than avoiding dtype-name formatting. No rejected
idea is being retried.

Implementation and measurement handoff:

1. Extend `tests/verify/test_native_buffer_capacities.py` with scalar rejection
   before any writes for wrong kind at equal width, wrong width at equal kind,
   structured dtype, noncontiguous and read-only buffers; put the invalid
   buffer late in iteration. Cover native/swapped-endian f32/i32 scalar copies
   and retain vector validation behavior if the shared helper changes.
2. Run existing exact encoder parity for v1–v9, history off/on and selected
   nondefault capacities, independent-output checks and full receipt comparison
   with the Python encoder oracle. Keep existing numerical tolerances
   (atol 1e-6, rtol 1e-5) and all discrete checks. Required debug Rust, Clippy
   and affected Python suites remain as listed above and in development docs.
3. Preserve the baseline ordinary binary and its SHA before the required root
   rebuild/atomic installation. Compare the **current batched Python owners**
   on the before and candidate ordinary wheels, with five serial alternating
   before/after samples per cell on the unchanged 1/4/16 × 42/43 grid. Use fresh
   processes after wheel swaps; build/setup/profile time stays outside timings.
   Keep the sequential-owner and saved merged-runtime receipt comparisons as
   parity controls. Both wheels must retain the same world/checkpoint/workload;
   record the deliberate runtime SHA change, never rewrite historical evidence.
   The old sequential/new batched speedup alone cannot establish this cut's gain.
4. Profile separately to confirm successful-path `_name_get` calls disappear
   from encoding and that encoding time falls. Judge the cut by unprofiled
   batched latency and parity, retaining every sample and reporting ranges and
   contrary cells. If improvement is indistinguishable from variability, report
   inconclusive and do not claim the predicted gain; if slower or incorrect,
   revert this cut. Do not bundle another optimization to conceal that result.

Headless reversible choice: take this narrow validation change as the next
experiment. This planning step changed only working notes; implementation,
measurement, original singleton acceptance and shipping remain separate.


## Dtype validation implementation (2026-11-09)

Step 3 changes only primitive dtype validation in the scalar/vector bindings.
The shared validator recognizes f32, f64, i32 and u8 from kind/itemsize; the
scalar path derives its expectation from the generated source dtype. Names are
read only for mismatch diagnostics. Complete validation still precedes scalar
copies. The source encoder, selected capacities, allocation, np.copyto and all
search/model/world identity owners are unchanged.

Baseline ordinary extension preserved **before any build or installation**:
`.cache/mtg-268/dtype-validation/baseline.cpython-312-darwin.so`, SHA256
`0b286acd043f21ecfcf8ca7e9c3ad449ee38b3dffaa0aa8914c311d733d15a46`.
Existing baseline JSON/pstats, merged-reference receipts and original-runtime
artifacts remain at their recorded paths. Do not compare this cut only against
the retained sequential Python owners: step 4 compares current batched owners
on both ordinary binaries in fresh processes after atomic swaps.

Baseline compatibility check:
`uv run --extra dev pytest tests/verify/test_native_buffer_capacities.py -q -k 'scalar_buffer_ or vector_buffer_validation or vector_buffer_binding'`
— 30 passed (`.cache/mtg-268/dtype-validation/baseline-tests-final.log`).
The initial fixtures incorrectly called native VectorEnv.close(), requested
unavailable f16 storage and expected vector swapped-endian rejection. Corrected
fixtures preserve the measured baseline contract; production behavior was not
changed to satisfy those assumptions. Scalar f32/i32 endian conversion passes.
Vector binding accepts swapped byte order; its direct-write behavior is outside
this scalar-copy optimization, and no conversion/rejection rule was added.
An exhaustive local NumPy sctypeDict probe found no primitive aliases whose
kind/size matches these four types but whose dtype.name differs.

One v6 encoding of an initialized observation makes 44 dtype-name calls across
22 arrays on the baseline (`.cache/mtg-268/dtype-validation/baseline-name-calls.json`).
The same candidate probe makes zero dtype-name calls
(`.cache/mtg-268/dtype-validation/candidate-name-calls.json`). These are mechanism
call counts, not timing evidence.
No latency claim, games, evaluations, publication or shipping in this step.


Both builds passed, serially with `CARGO_BUILD_JOBS=2`:
`uv run --extra play maturin build -i .venv/bin/python -m managym/Cargo.toml -o .cache/mtg-268/dtype-validation/debug-wheels`, then
`uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`.
Each wheel's cp312 extension was extracted to a temporary sibling and renamed
atomically over `manabot/managym/_managym.cpython-312-darwin.so`; fresh processes
performed every subsequent check. The final installed extension is ordinary,
not benchmark-instrumented. Build logs: `.cache/mtg-268/dtype-validation/{debug,release}-build.log`.

Candidate artifacts for step 4:

- `.cache/mtg-268/dtype-validation/candidate.cpython-312-darwin.so` and the
  installed extension both SHA256
  `236c5a7c7683cff0f71478f2cca14ccf35f758aafbcfd5122695f98a68bf8e66`.
- `.cache/mtg-268/dtype-validation/candidate.whl` retains the ordinary wheel.
  The debug wheel remains in `debug-wheels/`; its extension SHA256 is
  `ff7d5a0923ba630f34726ba5ebfb20998416b6c0a6f20f190556a277a7ef99e3`.
  Do not use the debug wheel in latency measurements.
- `.cache/mtg-268/dtype-validation/artifacts.json` records SHA256 for both
  ordinary binaries, wheels, verification logs, five baseline JSON samples,
  summary, pstats, merged-reference receipts and unchanged harness.
  The baseline binary was reverified after the final installation.

`uv run --extra dev pytest tests/verify/test_native_buffer_capacities.py tests/sim/test_local_update_sampling.py -q -k 'scalar_buffer_ or vector_buffer_validation or vector_buffer_binding or native_encoded_arrays or companion_shapes or batch_inputs_match_python_at_selected_capacities or nondefault_axes_match_scalar_vector_and_python'`
— 80 passed against the debug extension (`debug-tests.log`): all 30 new cases,
independent/writable outputs, companion-shape atomicity, scalar/vector selected
axes and exact batched Python-oracle inputs/predictions for v1–v9 with history
off/on. The 30 new cases also passed against the ordinary candidate
(`release-tests.log`), with the same selector as the baseline check above.

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders`
— six-cell complete discrete parity against retained sequential Python owners
and saved merged-runtime receipts; exact arrays for all 1,281 observed views.
Output: `.cache/mtg-268/dtype-validation/candidate-parity.json` (stderr beside it).
Maximum base/value/target errors remain
`1.0242171e-10 / 4.09781933e-8 / 3.13064846e-9` within unchanged tolerances.
Checkpoint SHA256 remains `57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`;
world identity, hashes, Commands and discrete receipts pass unchanged.
The harness SHA256 remains `3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a`.
Incidental timings/profile output from this parity run are **excluded** from
step 4's alternating before/after sample set and establish no latency gain.

Focused Ruff check/format, Cargo fmt and `git diff --check` pass. Affected user
docs still match the behavior; no documentation contract was weakened.
At the implementation handoff, step 4's five alternating ordinary-wheel samples
per cell, separate full profile and keep/revert decision were still pending;
the final section below records their completion and restoration. The existing broad
Python suites, complete debug Rust suite, all-feature Clippy and engine-budget
checks remain for gate/CI; the debug-extension checks above do not replace them.
The original singleton-latency acceptance and w13 evidence questions remain open.


## Dtype measurement protocol (2026-11-09)

Claim under test: “Primitive dtype metadata validation reduces median batched
16-world decision latency by 3–5% on fixed openings 42/43 at depth 20, with
unchanged receipts and numerical predictions within atol 1e-6 / rtol 1e-5.”
This is a hypothesis, not a result. Compare the preserved ordinary baseline
build from HEAD 44895a104 with the built ordinary candidate, keeping current
batched Python owners, environment and the unchanged harness on both sides.
All step-3 artifact hashes verified before starting. Five serial baseline then
candidate grid pairs use fresh processes after atomic extension replacement;
each process warms the policies identically. The harness’s separate profiles
are excluded from latency samples. Retain all stdout/stderr and exit codes,
including failed runs; no retries or sample removal. Separate full cProfiles
will retain dtype call counts and source attribution without replacing earlier
evidence. No games, evaluations, induced load or other optimization.

Headless decision rule: if latency ranges overlap, report no measurable
difference/inconclusive for the predicted gain. Follow the invoked measure
skill’s return-to-last-kept rule for a candidate that does not clear noise;
only the dtype cut and its added tests may be restored, preserving all earlier
batching work and measurement evidence.


## Dtype measurement result — restored (2026-11-09)

Inconclusive: no measurable end-to-end latency improvement. Restore the dtype
cut to the last kept commit, 44895a104. The sixteen-world median reductions of
4.29% (52.6 ms, position 42) and 5.75% (96.6 ms, position 43) lie inside broad
overlapping ranges. All six cells have overlapping ranges. The four smaller
cells have worse medians (1.65–4.72% higher), also within variability. These are
five samples per side, not confidence intervals. The 0.76723 s candidate sample
at 42/four worlds is retained; its cause was not measured. No latency sample
was dropped, replaced or retried. A quieter, controlled host with a newly
declared repeated protocol would be needed to resolve a 3–5% gain; this run
does not establish one and does not start that further investigation.

Comparison: current batched Python owners on the preserved ordinary baseline
and candidate release binaries, five serial baseline/candidate grid pairs.
Each swap copied to a sibling file and atomically renamed it; every benchmark
was a fresh process. Both builds were already completed before this step.
Same locked environment, warmed policies, Apple M4 Max/macOS 26.0.1,
CPython 3.12.12, Torch 2.10.0, one CPU/inter-op thread, initialized scalar
64-wide/two-layer model, seeds 42/43, search seed 17, depth 20, 1/4/16 worlds.
No games, evaluations, artificial contention or concurrent benchmark workers.
The new candidate comparison is not the historical sequential-to-batched gain.

The unchanged harness times teacher.search itself, including the rollout work,
but excludes setup, warming, profile searches, receipt comparisons and swaps.
All ten latency processes exited zero: 60 live sequential-owner comparisons
and 60 saved-receipt comparisons passed. Each wheel's timed batched searches
executed 945 rollouts / 18,900 Commands; its timed sequential controls executed
the same work. Both sides distinguish the larger workload: 16 worlds execute
16 times the singleton rollouts/Commands and take much longer. No terminal-heavy
positions are represented. Maximum absolute base/value/target errors remain
1.0242171e-10 / 4.09781933e-8 / 3.13064846e-9, within atol 1e-6 / rtol 1e-5.
Full discrete receipts, legal support, Commands, hashes, audits, sampling,
world and checkpoint identities pass. Runtime SHA is deliberately different.

Profiles are separate searches, excluded from latency measurements. Across
five profiles per wheel, encoding falls from median 0.19549 s
[0.17280, 0.23433] to 0.07677 s [0.06826, 0.10713]: 60.7% less profiled
encoding time, disjoint ranges. Every other listed profile bucket overlaps.
This is a micro result under cProfile, not a 60.7% search improvement.

An additional full profile on each wheel confirms 56,364 dtype _name_get calls
become zero (1,281 encodings × 22 arrays × two names). All baseline calls are
attributed by pstats to Env.encode_observation_into. Dtype names take 0.09052 s
of the baseline full profile's 1.18372 s: 7.65%, an optimistic complete-removal
ceiling of 1/(1 - 0.0765) = 1.083×. cProfile amplifies Python-call cost; this
is not an unprofiled bound or saving. Encoding as a whole takes 0.18232 s
(15.4%) there, versus 0.06452 s (6.45%) of the candidate's 1.00090 s profile.
Those additional full-profile timings have n=1 per side and no variability
estimate; their incidental unprofiled cells are excluded as well.

The limiter remains native Command application/audit: 0.48073 s (40.6%) of the
baseline full profile, 0.45022 s (45.0%) of the candidate profile. The candidate
also spends 26.1% in the network and 17.3% hashing. These native/profile buckets
do not explain unmeasured host variability and cannot support a stronger verdict.
Source mapping: LocalUpdateTeacher._run_rollouts and predict_batch in
manabot/sim/search/local_update.py call SelectedBranchRuntime.apply_policy_choice
and Env.encode_observation_into in managym/src/python/bindings.rs. The dtype
cut changes fill_encoded_into_existing_buffers and require_numpy_array in
managym/src/python/convert.rs; the latter's candidate is retained in candidate.patch.
Hashing still calls Game::semantic_viewer_state_hash in managym/src/decision.rs.
No hash, Command or model changes were made in this measurement step.

Errors: zero latency/process/parity failures in the ten planned samples. One
additional diagnostic setup failed before any search because the new wrapper
was named profile.py and shadowed the Python standard library. Its empty JSON
and traceback are retained as profile-baseline-setup-failed.*. Renaming only
the diagnostic wrapper to capture_profile.py fixed it; both full profiles then
exited zero. The frozen benchmark was unchanged. Missing failed-profile
measurements are absent, never zero-valued timing samples.

Disposition: the three Rust binding files and the dtype-added tests were
restored exactly to HEAD; the preserved baseline extension was atomically
reinstalled. The complete rejected diff, both ordinary binaries, candidate
wheel, all verification evidence and all samples remain in
.cache/mtg-268/dtype-validation. Earlier batching/model/hash/encoding work is
unchanged. No other optimization, Graph, publication or shipping.

Identity and reproduction:
- Harness SHA256: 3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a.
- Baseline/restored runtime: 0b286acd043f21ecfcf8ca7e9c3ad449ee38b3dffaa0aa8914c311d733d15a46.
- Candidate runtime: 236c5a7c7683cff0f71478f2cca14ccf35f758aafbcfd5122695f98a68bf8e66.
- Checkpoint: 57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb.
- Ten exact commands/order/exit codes: measurement-runs.json; runner: measure.py.
- Samples: measure-{1..5}-{baseline,candidate}.{json,err}; aggregation:
  measurement-summary.json. Each sample retains the sequential-control timings
  and all parity maxima as well as the batched timings used here.
- Full profiles: profile-{baseline,candidate}-{batched,sequential}.{pstats,json};
  diagnostic wrapper: capture_profile.py. Historic pstats/receipts are unchanged.

Remaining Task limits are unchanged: no trained-strength claim, unresolved
MTG-183/w13 admission, no measured eight-world point, and no agreed definition
of “near” the original singleton. Broader gate/CI validation belongs to the
remaining Task, not this rejected candidate's focused checks.


Seconds per batched decision, median [minimum, maximum], n=5 per wheel.
Positive reduction means a lower candidate median.

| Position | Worlds | Baseline s | Candidate s | Median reduction | Verdict | Rollouts / Commands |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.10346 [0.09089, 0.11682] | 0.10834 [0.08425, 0.19892] | -4.72% | No measurable difference | 4 / 80 |
| 42 | 4 | 0.34375 [0.29185, 0.37101] | 0.35470 [0.26137, 0.76723] | -3.18% | No measurable difference | 16 / 320 |
| 42 | 16 | 1.22505 [1.01662, 1.38053] | 1.17244 [1.01458, 1.57390] | +4.29% | No measurable difference | 64 / 1280 |
| 43 | 1 | 0.12370 [0.11262, 0.13320] | 0.12940 [0.11679, 0.14463] | -4.61% | No measurable difference | 5 / 100 |
| 43 | 4 | 0.39782 [0.37955, 0.44692] | 0.40437 [0.38055, 0.45163] | -1.65% | No measurable difference | 20 / 400 |
| 43 | 16 | 1.68050 [1.32344, 1.80968] | 1.58390 [1.49736, 1.70994] | +5.75% | No measurable difference | 80 / 1600 |

Separate profile searches at position 42 / 16 worlds, n=5 per wheel:

| Category | Calls | Baseline profile s | Candidate profile s | Verdict |
| --- | --- | --- | --- | --- |
| native_step_and_audit | 1280 | 0.51045 [0.45361, 0.55229] | 0.58284 [0.46486, 0.78931] | No measurable difference (profile only) |
| network | 21 | 0.30453 [0.27113, 0.35454] | 0.34090 [0.27540, 0.38127] | No measurable difference (profile only) |
| semantic_direct_hash | 1296 | 0.19909 [0.17477, 0.21178] | 0.25656 [0.17967, 0.32013] | No measurable difference (profile only) |
| observation_encoding | 1281 | 0.19549 [0.17280, 0.23433] | 0.07677 [0.06826, 0.10713] | Better (profile only) |
| session_snapshot | 64 | 0.00903 [0.00750, 0.01128] | 0.00987 [0.00794, 0.01189] | No measurable difference (profile only) |
| observation_allocation | 21 | 0.00253 [0.00228, 0.00338] | 0.00263 [0.00239, 0.00279] | No measurable difference (profile only) |
| tensor_assembly | 462 | 0.00078 [0.00071, 0.00083] | 0.00081 [0.00073, 0.00086] | No measurable difference (profile only) |


Final restoration verification: `uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders` exited zero on the restored ordinary wheel: all six live/saved receipt pairs and 1,281 exact encodings passed. Its incidental timings are excluded. All 40 retained step-3 artifacts other than the intentionally replaced installed extension reverified unchanged. The four restored source/test files match HEAD byte for byte; `git diff --check` passes. `measurement-artifacts.json` records the new artifact hashes and final installation. No process started by this step remains running.
