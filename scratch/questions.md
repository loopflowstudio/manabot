# MTG-268 — batched local-update rollouts

The Task directive is the design; no separate accepted design was present.
It says: "Batching should bring 8-16 worlds near today's one-world cost."
That outcome remains unmet. No numeric definition of "near" or relaxation has
been agreed. The compressed fact-column path costs 4.55x/4.92x the old
singleton, versus 5.46x/5.72x after native optimization and compression, and
8.0–8.1x before native optimization. These are single samples, not a variance
estimate. Implementation progress is not acceptance.

Latest iteration direction: "Extend scripts/bench_local_update.py to attribute
object-embedding work within manabot/model/actor_critic.py, currently 0.386 seconds
of 0.489 seconds model time." It also says: "Use that evidence to implement a
bounded optimization" and "Any state-derived reuse needs explicit ownership and
invalidation guarantees." This iteration is implemented and measured below;
broader validation remains for gate.

## Approach and Delete — do not maintain

Keep one live frontier, independent policy RNGs and stable world/action receipt
order. Preserve all RolloutReceipt fields, private audits, native Commands,
canonical hashes, legal support, terminal/sign semantics, sampling identities,
atomic precondition rejection and deadlines. No input/checkpoint/schema change
or batching flag. Compound execution is removed; historical readers survive.
The directive's compound fallback is obsolete under the existing removal in
`docs/training.md#compound-decisions-removed`; restoring it is outside this work.
On 2026-11-08, refreshed `origin/main` was `841b1c50c`, already contained in
the checkout base `b06478f43`; no additional upstream changes needed reconciliation.
The Task has no Wave, so there is no selected Wave memory to curate.

Delete — do not maintain (completed):

- Sequential production rollouts and singleton predictor; git supplies the oracle.
- Full Python semantic parsing at each ply and repeated root/config parsing.
- Globally empty CPU attention columns; restore coordinates and historical pooling.
- Pre-Command viewer projections and duplicate equivalent acting-view serialization.
- Rechecking fields of a Command just built from validated inputs under both locks.
- Embedding CPU fact columns unused by the entire inference batch; no cached state.
- Benchmark receipt-field allowlist; compare every field with explicit exceptions
  for run identity, timing and separately checked numerical values.
- Duplicate fact-word decoding and repeated boolean validity conversions; one
  `_word` helper and one validated mask serve compaction, encoding and joins.

No remaining deletion target is identified. The fact cleanup also separates
selected row indices (`used`) from numeric feature columns (`columns`). Further
latency work needs measurement, not another execution path.

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
Current ordinary runtime SHA256:
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

Reversible implementation choice: remove only all-batch-empty fact columns on
CPU with autograd disabled. Recompute the ordered selection from this call's
validated validity mask, then use the existing fact encoder. No retained state,
weight cache, invalidation protocol, schema or checkpoint changes. Sparse rows,
program ordinals, duplicate occurrences, batch/object ownership and all input
validation survive; training keeps its existing fixed-shape computation.

The separate ordinary-wheel profile at position 42 / 16 worlds selected facts:
0.247 s of 0.403 s object embeddings (0.503 s network). Links cost 0.063 s,
continuation rows 0.054 s; player/card/permanent projections together 0.011 s.
`scratch/mtg-268-object-before.json` retains that profile and its unprofiled
1.506 s decision. After the cut, facts cost 0.00974 s, objects 0.157 s and
network 0.256 s. Timing wrappers exist only during the separate profile and
are restored before later unprofiled searches. They attribute encoder components
without changing production timing. The benchmark now loads the reference
FactEncoder from git too, so its old model cannot inherit this optimization.

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/reference`
— all six cells pass against both reference teacher/model/FactEncoder on the
current runtime and the complete retained original-runtime receipts.
Output: `scratch/mtg-268-object-after.json`. Same checkpoint, runtime SHA256,
positions, depth, thread count and single-sample limitations above;
`native_instrumented_build=false`. No native rebuild needed.

| Worlds | Position 42 original → final (s) | Position 43 original → final (s) |
| --- | --- | --- |
| 1 | 0.286 → 0.109 | 0.328 → 0.131 |
| 4 | 1.060 → 0.343 | 1.317 → 0.428 |
| 16 | 4.404 → 1.277 | 5.250 → 1.576 |

Same-allocation speedups: 2.49–3.45x. Sixteen-world/original-singleton ratios:
4.4664 / 4.8058. Every discrete field, complete audits/hashes, root/sampling
identity, legal support and allocation matches. Maximum base/value/target
errors: 1.49e-10 / 5.96e-8 / 6.93e-9 (atol 1e-6, rtol 1e-5).
Forwards remain 1,281 → 21, Commands 1,280, semantic witnesses 1,298,
snapshots 64 and full Python parses 1,298 → 2 at the profiled position.
These initialized opening fixtures do not establish performance on dense facts,
a trained checkpoint or representative gameplay; no strength claim.

`uv run --extra dev pytest tests/model/test_layer_input.py tests/model/test_record_vocabulary.py -q -k 'fact or local_program'`
— 18 passed, 20 deselected after compression. Covers hashed/catalog ids,
sparse/empty batches, interior holes, duplicate programs, fresh batch/weight
visibility, autograd after inference and invalid-input rejection. Nonzero output
weights exercise fact effects. Focused Ruff check/format and `git diff --check`
also pass. No new Rust change; the ordinary wheel remains installed.

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

## Validation and remaining work

- Ordinary release wheel rebuilt and atomically installed after the instrumented
  builds. Both native profiles and six-cell final comparison pass parity.
- `uv run --extra dev pytest tests/sim/test_search_command_evidence.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py -q -k 'selected_command or real_root_replay or batched or frontier or batch_'`
  — 10 passed, 11 deselected after final compression. Covers identity/revision/offer rejection with and
  without audits, RNG changes at the same epoch, full post-apply witnesses,
  replay, depth-20 frontier parity, terminal signs, support and deadlines.
- `cargo test --manifest-path managym/Cargo.toml -p managym --test search_state_contract witness_preserves_acting_events_and_fixed_viewer_bytes -j 2 -- --test-threads 2`
  — 1 passed in debug after final compression; independent observation bytes/hashes with/without events
  and a chooser; final source rebuild passes without warnings.
- Cargo fmt, focused Ruff check/format and `git diff --check` pass; `lf context`
  stays below both token and byte limits.
- Prior attention validation: 14 focused tests passed, including six attention
  cases covering value aggregations, padding, joins and autograd after inference.

Gate still owns complete local-update suites and affected model suites
(`tests/model/test_value_aggregation.py`, `tests/model/test_inference.py`,
`tests/model/test_agent.py`), plus `tests/training/test_active_training.py`,
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`,
`tests/verify/test_native_buffer_capacities.py`. Native changes now require the
complete debug Rust suite, all-feature Clippy and applicable engine-budget CI
checks; focused debug and release-wheel checks do not substitute for them.
No input shapes, rules/world identity, checkpoint parameters or schemas changed.

Remaining implementation and evidence:

- The concrete object-attribution iteration is complete. Further latency work
  needs a new measured cut; no additional optimization is selected here.
- The pre-compression fact-column profile is distributed: native step/audit 0.424 s,
  encoding 0.278 s, semantic JSON/hash 0.264 s, network 0.256 s. Within the
  network, object embeddings take 0.157 s (links 0.060 s, continuation rows
  0.052 s) and attention 0.080 s. These are inclusive categories, not additive
  with their parents. The remaining evidence work still scales per branch.
- Gate adds `tests/model/test_layer_input.py` and
  `tests/model/test_record_vocabulary.py` to its affected model suites. No new
  Rust change in this iteration; earlier native changes still need gate/CI.

The latency outcome is incomplete; the achieved speedup does not replace it.

Open for demo: what maximum 8–16-world/original-singleton latency ratio, and
which checkpoint and representative positions, define "near"? No answer or
relaxation is recorded; the two initialized opening fixtures do not settle it.
MTG-183/w13 remains unmeasured and was not rebound to this world; no trained
strength claim. Memory still scales with worlds × supported root actions.
The README's 30-second example is an illustrative budget, not a latency promise.
