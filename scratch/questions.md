# MTG-268 — batched local-update rollouts

The Task directive is the design; no separate accepted design was present.
It says: "Batching should bring 8-16 worlds near today's one-world cost."
That outcome remains unmet. No numeric definition of "near" or relaxation has
been agreed. The direct semantic-hash path costs 3.69x/4.16x the old
singleton, versus 4.55x/4.92x after fact compression, 5.46x/5.72x after native
optimization and compression, and 8.0–8.1x before native optimization. These are single samples, not a variance
estimate. Implementation progress is not acceptance.

Latest iteration direction: "Investigate the 0.259 s semantic JSON/hash cost."
It asks to "implement a bounded direct-hash path if equivalence is established"
and "Preserve exact viewer/event semantics and hash bytes; do not substitute a
witness hash without proving equivalence."

Reversible choice: compute semantic SHA-256 under the Env lock on every read,
without reuse. Witness BLAKE3 is not equivalent. The bounded investigation is
complete; broad validation stays with gate and the latency objective is open.

## Approach and Delete — do not maintain

Keep one live frontier, independent policy RNGs and stable world/action receipt
order. Preserve all RolloutReceipt fields, private audits, native Commands,
canonical hashes, legal support, terminal/sign semantics, sampling identities,
atomic precondition rejection and deadlines. No input/checkpoint/schema change
or batching flag. Compound execution is removed; historical readers survive.
The directive's compound fallback is obsolete under the existing removal in
`docs/training.md#compound-decisions-removed`; restoring it is outside this work.
The 2026-11-08 realign refresh still found `origin/main` at `841b1c50c`,
already contained in the checkout base `b06478f43`; no upstream changes or new
Task direction needed reconciliation.
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

No remaining deletion target is selected; further latency cuts need measurement.

## Retained comparison evidence (2026-11-08)

Apple M4 Max, macOS 26.0.1, Torch 2.10.0; one CPU/inter-op thread; initialized
64-wide/two-layer scalar policy, current observation defaults; checkpoint SHA256
`5e3e03fb91c9fba2ea75bb7736d797505ea5385cf9dd538a9d0bab6b582899a6`.
Two seeded openings (42/43), search seed 17, depth 20, four/five supported root
actions. One unprofiled timed decision per cell, warm singleton inference;
separate profiling search at position 42 / 16 worlds. Bounded fixed positions
only; no games/evaluations, stress or artificial-load tests. These single samples
establish no variance estimate or production guarantee.

The benchmark loads reference execution owners from `841b1c50c` on the current
runtime and separately compares complete original-runtime receipts/times in
`.cache/mtg-268/reference/`. Original binary:
`.cache/mtg-268/reference-native.cpython-312-darwin.so`; aggregate original run:
`scratch/mtg-268-native-before.json`. Use `--retain-reference DIR` before replacing
a native library; `--native-reference DIR` checks saved receipts and identities.
Original runtime SHA256:
`fbc055ffae76c751a7f496d85388ee64adf101ffebc598a5e92d5ca12e46fda3`.
Prior fact-compression ordinary runtime SHA256:
`94dbd622e1449d28633e9843bbf0099a19fad22cc37f363295c4eb16dfdf9885`.

Native attribution uses thread-local timers under the existing Rust `benchmark`
feature, enabled only during the separate profiling search. Build with
`--features python,benchmark` (explicit features replace maturin defaults).
This also includes layer/allocation instrumentation: never compare these times
with ordinary-wheel latency. Ordinary wheels contain no timers.

The native profile selected witnesses: 1.990 s of 2.085 s native step/audit,
including 1.782 s projections/serialization/hashes and 0.145 s authority; rules
0.048 s. After removing redundant projections, witnesses took 0.758 s of
0.806 s step/audit (projections 0.519 s, authority 0.237 s, rules 0.025 s).
Full projection passes fell 2,560 → 1,280; authority/legality still run before
and after all 1,280 Commands. Raw runs:
`scratch/mtg-268-native-profile-before.json` and
`scratch/mtg-268-native-profile-after.json`. Categories overlap parents; conversion
measures struct/info conversion, not outer PyO3 boxing. Separate samples are not
paired microbenchmarks; increased authority time also shows variability.

The ordinary-wheel native compression run passed six-cell complete discrete
parity and numeric tolerance: `scratch/mtg-268-native-compress.json`. At 1/4/16
worlds, seconds were 0.131/0.419/1.559 (42) and 0.146/0.498/1.877 (43):
2.18–2.82x same-allocation speedup; 16-world/original-singleton 5.46x/5.72x.
Its profile was model 0.489 s (objects 0.386 s, attention 0.083 s), native
step/audit 0.427 s, encoding 0.289 s, semantic JSON/hash 0.268 s, tensor assembly
0.0154 s and snapshots 0.00729 s. Comparator mutation probes rejected altered
schema, aggregate values, missing-value support and audit JSON.

Full earlier native tables and validation: `6b4a511:scratch/questions.md`;
pre-native history: `ec39166:scratch/questions.md` and
`63f7d6a:scratch/questions.md`. Uncommitted notes were copied before this
curation to `.cache/mtg-268/questions-before-fact-compress.md`. Raw timing files
remain under `scratch/`. Network-call count alone did not predict latency;
evidence construction and remaining per-branch work still matter.

## Fact-column optimization (2026-11-08)

The choice was per-call CPU/no-autograd compaction of all-batch-empty fact
columns, with no retained state. Full initial measurements and focused checks
are in `f9e3801:scratch/questions.md` (18 fact/model cases passed). Raw files:
`scratch/mtg-268-object-before.json` and `scratch/mtg-268-object-after.json`.
Fact encoding fell from 0.247 s to 0.00974 s; initial final decision times at
1/4/16 worlds were 0.109/0.343/1.277 s (42) and 0.131/0.428/1.576 s (43).
Six-cell complete discrete parity passed; max base/value/target errors were
1.49e-10 / 5.96e-8 / 6.93e-9. No trained or dense-position claim follows.

Compression recheck:
`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/reference`
— all six cells pass complete discrete parity against both reference execution
owners and retained original-runtime receipts; output
`scratch/mtg-268-object-compress.json`, ordinary wheel. At 1/4/16 worlds, seconds
are 0.1087/0.3398/1.3008 (42) and 0.1261/0.4121/1.6140 (43).
Same-allocation speedups are 2.60–3.39x; 16-world/original-singleton ratios
4.5508/4.9226. Maximum base/value/target errors:
1.49e-10/5.96e-8/6.93e-9. Same fixture and single-sample limits apply; this is
parity evidence, not a claim that the cleanup improved latency. The separate
profile reports facts 0.00890 s, objects 0.152 s and network 0.248 s.

The realign check found that the teacher-source digest omitted `facts.py`.
The digest now includes that inference owner, so changing fact encoding changes
the implementation identity used by receipts and arena admission. Retained
timing receipts above predate this identity repair; their measured behavior and
latency remain evidence for the same inference computation. Historical evidence
was not rewritten. The digest remains an explicit owner list, not a claim of
transitive source coverage.

`uv run --extra dev pytest tests/sim/test_local_update.py -q -k 'teacher_identity_tracks_fact_encoder or real_root_replay'`
— 2 passed, 5 deselected during realign. The identity probe changes only the
fact source hash; current-root replay still agrees. Focused Ruff check/format
passes. No timing rerun was needed for this identity-only repair.

## Direct semantic hash (2026-11-08)

The direct and composite APIs share the checked fixed-viewer projection and
exact serialization. No event/hash algorithm, Command or audit changed. Rust
checks independently hash both viewers with/without events, a chooser and
terminal state; Python checks pre-reset rejection, forks and three Commands.

`uv run --extra dev python scripts/bench_local_update.py --worlds 16 --position-seeds 42 --native-reference .cache/mtg-268/reference`
— instrumented parity passes against both references; attribution retained in
`scratch/mtg-268-hash-profile.json`. Native semantic calls include 50 internal
composite reads in addition to the Python calls. Before → after, seconds:

| Native category | Before | After |
| --- | ---: | ---: |
| Fixed-viewer projection | 0.00854 | 0.00636 |
| Canonical viewer serialization | 0.12520 | 0.11654 |
| SHA-256 | 0.03591 | 0.03490 |
| Composite construction | 0.07404 | 0.00147 |
| Composite serialization | 0.01940 | 0.000017 |

Projection/serialization/hash counts stay 1,348. Composite constructions fall
1,348 → 52; Python composite JSON calls fall 1,298 → 2, plus 1,296 direct hash
calls. Thus the removable native cost is about 0.093 s; canonical serialization
and hashing still dominate direct reads. These feature-gated timings belong to
an instrumented build and are not ordinary-wheel latency comparisons.

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/reference`
— all six cells pass complete discrete parity and numeric tolerance against
reference teacher/model/FactEncoder and retained original-runtime receipts.
Ordinary-wheel results: `scratch/mtg-268-hash-after.json`, with
`native_instrumented_build=false`. Final installed runtime SHA256:
`21865ee8f4b9bbde2ef6463af4d583a8edd6b305c9b5d4a98e95fe813df61476`.
Same checkpoint, positions, depth 20, one thread, search seed and single-sample
limitations as above; no games/evaluations or artificial load.

| Worlds | Position 42 original → final (s) | Position 43 original → final (s) |
| --- | --- | --- |
| 1 | 0.28584 → 0.09045 | 0.32787 → 0.17281 |
| 4 | 1.05959 → 0.28020 | 1.31690 → 0.39134 |
| 16 | 4.40426 → 1.05505 | 5.24962 → 1.36387 |

Sixteen-world/original-singleton ratios: 3.6911 / 4.1597. Same-allocation
speedups: 1.90–4.17x. Maximum base/value/target errors against either reference:
1.49e-10 / 5.96e-8 / 6.92e-9 (atol 1e-6, rtol 1e-5). Complete hashes, audits,
Commands, support, allocation and sampling identities agree. The 16-world
position-42 profile retains 21 forwards, 1,280 Commands and 64 snapshots.

Contrary timing evidence: position 43's singleton is slower than the prior
fact-compression sample (0.17281 vs 0.1261 s). The separate ordinary-wheel
profile also varies substantially: direct hash 0.222 s, step/audit 0.705 s,
encoding 0.470 s and network 0.431 s. Do not infer paired speedups for this cut
or sum overlapping profile categories; no variance estimate or production claim.

`cargo test --manifest-path managym/Cargo.toml -p managym --test search_state_contract semantic_direct_hash -j 2 -- --test-threads 2`
— 1 passed in debug, including terminal fixtures.
`uv run --extra dev pytest tests/sim/test_search_command_evidence.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py -q -k 'semantic_direct_hash or selected_command or real_root_replay or batched or frontier or batch_'`
— 11 passed, 12 deselected on the ordinary wheel. Focused Ruff check/format,
Cargo fmt and `git diff --check` pass. Both release builds passed; the ordinary
wheel was atomically installed last. Final Python formatting/import cleanup
changes no measured computation. Full debug suites, Clippy and engine-budget
checks remain for gate/CI.

## Compression and remaining validation

The compression pass reuses decoded prepared constraints for validation and
identity, and one match fixture across the Python evidence tests. Native
production code needed no further simplification. Prior uncommitted notes were
copied to `.cache/mtg-268/questions-before-hash-compress.md` before curation;
measurements above remain historical, not retimed cleanup claims.

`uv run --extra dev pytest tests/sim/test_search_command_evidence.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py -q -k 'semantic_direct_hash or selected_command or real_root_replay or batched or frontier or batch_ or injected_sampler or direct_search'`
— 14 passed, 9 deselected after compression. Focused Ruff check/format and
`git diff --check` pass. No Rust source changed in this pass, so the ordinary
wheel remains installed. The teacher source digest changes with this cleanup;
retained timing receipts were not rewritten or remeasured.

Earlier witness-event debug validation passed one case; attention validation
passed 14 focused cases. Commands/results are at `f9e3801:scratch/questions.md`.
The current ordinary release wheel was atomically installed after the
instrumented build. No input shapes, world identity, checkpoint parameters or
schemas changed.

Gate owns complete local-update suites and affected model suites
(`tests/model/test_value_aggregation.py`, `tests/model/test_inference.py`,
`tests/model/test_agent.py`, `tests/model/test_layer_input.py`,
`tests/model/test_record_vocabulary.py`), plus
`tests/training/test_active_training.py`, `tests/env/test_observation.py`,
`tests/training/test_action_cutoff.py`, `tests/verify/test_native_buffer_capacities.py`.
The branch's native changes require the complete debug Rust suite (including
search-state contracts), all-feature Clippy and applicable engine-budget CI
checks. Focused tests and release builds do not substitute for them.

Canonical serialization and SHA-256 remain on every direct read; no additional
optimization or reuse is selected. The latency outcome is incomplete.

Open for demo: what maximum 8–16-world/original-singleton latency ratio, and
which checkpoint and representative positions, define "near"? No answer or
relaxation is recorded; the two initialized opening fixtures do not settle it.
MTG-183/w13 remains unmeasured and was not rebound to this world; no trained
strength claim. Memory still scales with worlds × supported root actions.
The README's 30-second example is an illustrative budget, not a latency promise.
