# MTG-268 — batched local-update rollouts

The Task directive is the design; no separate accepted design was present.
It says: "Batching should bring 8-16 worlds near today's one-world cost."
That outcome remains unmet. No numeric definition of "near" or relaxation has
been agreed. The final compression check costs 5.5–5.7x the old singleton,
versus 5.2–5.6x before these clarity cuts and 8.0–8.1x before the native
optimization. Single samples do not establish a change from the clarity cuts.
Implementation progress is not acceptance.

Latest iteration direction: "extend scripts/bench_local_update.py to distinguish
native rules execution, Command validation, witness construction and
returned-observation conversion, then optimize the dominant measured operation."
This iteration is implemented and measured; broader validation remains for gate.

## Approach and Delete — do not maintain

Keep one live frontier, independent policy RNGs and stable world/action receipt
order. Preserve all RolloutReceipt fields, private audits, native Commands,
canonical hashes, legal support, terminal/sign semantics, sampling identities,
atomic precondition rejection and deadlines. No input/checkpoint/schema change
or batching flag. Compound execution is removed; historical readers survive.

Delete — do not maintain (completed):
- Sequential production rollouts and singleton predictor; git supplies the oracle.
- Full Python semantic parsing at each ply and repeated root/config parsing.
- Globally empty CPU attention columns; restore coordinates and historical pooling.
- Pre-Command viewer projections and duplicate equivalent acting-view serialization.
- Rechecking fields of a Command just built from validated inputs under both locks.
- Benchmark receipt-field allowlist; compare every field with explicit exceptions
  for run identity, timing and separately checked numerical values.

The latest compression also gives viewer serialization/hashing one constructor,
checks aggregate action values and missing-value support, and closes native
profiling in `finally`. Earlier uncommitted notes were retained before curation
at `.cache/mtg-268/questions-before-compress.md`. No remaining deletion target
is identified; further latency work needs measurement, not another execution path.

## Native attribution (2026-11-08)

Reversible measurement choice: thread-local timers under the existing Rust
`benchmark` feature, enabled only during the separate profiling search. Build
with `--features python,benchmark` (explicit features replace maturin defaults).
Ordinary wheels contain no timers. This feature also includes existing layer
and allocation instrumentation; its elapsed times are not production latency.

At position 42 / 16 worlds, the initial profile selected witness work:
1.990 s witness construction out of 2.085 s native step/audit, with 1.782 s in
projections/serialization/hashes and 0.145 s in authority. Rules took 0.048 s;
validation 0.0020 s; returned native observation construction 0.0091 s.
`scratch/mtg-268-native-profile-before.json` retains that run. Its instrumented
unprofiled timing was much slower than the ordinary wheel; do not mix them.

After removing redundant projections, the same bounded instrumented fixture
reported 0.758 s witness construction (0.237 s authority, 0.519 s projections),
0.025 s rules, 0.00125 s Command validation, 0.00811 s returned observation build,
0.00211 s Python conversion and 0.00569 s audit record/serialization.
Native step/audit totaled 0.806 s. Full projection passes fell 2,560 → 1,280;
authority/legality still run before and after all 1,280 Commands.
`scratch/mtg-268-native-profile-after.json` retains the output. Projection and
authority categories overlap the witness parent; do not sum them twice.
Python conversion measures struct/info conversion, not outer PyO3 boxing.
The initial conversion category included receipt serialization; final code
attributes it to audit. Separate profile samples are not paired microbenchmarks
or a variance estimate; the increased authority time also shows run variability.

## Ordinary-wheel comparison before final compression

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/reference`
passed all six cells; output: `scratch/mtg-268-native-final.json`.
The benchmark loads both teacher and model from `841b1c50c`, compares them on the
current runtime, and separately compares the final teacher to full original
native-runtime receipts/times retained before rebuilding. The original evidence
is in `.cache/mtg-268/reference/`; the original binary is retained at
`.cache/mtg-268/reference-native.cpython-312-darwin.so`. Its aggregate run is
`scratch/mtg-268-native-before.json`. Use `--retain-reference DIR` before replacing
a native library; `--native-reference DIR` validates saved receipts and identities.
The output distinguishes shared-runtime and original-runtime comparisons.

Apple M4 Max, macOS 26.0.1, Torch 2.10.0; one CPU/inter-op thread; initialized
64-wide/two-layer scalar policy, current observation defaults; checkpoint SHA256
`5e3e03fb91c9fba2ea75bb7736d797505ea5385cf9dd538a9d0bab6b582899a6`.
Two seeded openings (42/43), search seed 17, depth 20, four/five supported root
actions. One unprofiled timed decision per cell, warm singleton inference;
separate profiling search at position 42 / 16 worlds. Bounded fixed positions
only; no games/evaluations, stress or artificial-load tests.

| Worlds | Position 42 original → final (s) | Position 43 original → final (s) |
| --- | --- | --- |
| 1 | 0.286 → 0.119 | 0.328 → 0.147 |
| 4 | 1.060 → 0.426 | 1.317 → 0.483 |
| 16 | 4.404 → 1.490 | 5.250 → 1.842 |

Same-allocation speedup is 2.23–2.96x. Sixteen worlds cost 5.21x / 5.62x the
original singleton. Original runtime SHA256:
`fbc055ffae76c751a7f496d85388ee64adf101ffebc598a5e92d5ca12e46fda3`;
final ordinary runtime:
`ce1803de08ce0a6fbeb804373e907541a95c99b44367887630602bc970e974ea`.
These single samples, taken before/after rebuilding, establish no variance or
production guarantee. Prior compression evidence remains in
`scratch/mtg-268-timing-compress.json` (1.62–1.99x; 16 worlds 8.0–8.1x singleton).
Earlier interpretation/history remains at `ec39166:scratch/questions.md` and
`63f7d6a:scratch/questions.md`, with earlier raw timing files under `scratch/`.

Every discrete rollout field (including complete audit JSON and observation
hashes), root/sampling identity, support and allocation matched the retained
original runtime. Maximum base/value/target errors:
1.49e-10 / 5.96e-8 / 6.93e-9 (atol 1e-6, rtol 1e-5). Float32 batching can round;
a sampling-threshold crossing may change other trajectories. Replay requires
retained source, runtime, checkpoint and allocation identities.

The final ordinary-wheel profile is now distributed: model 0.482 s (object
embeddings 0.383 s, attention 0.080 s), native step/audit 0.428 s, encoding
0.288 s, semantic JSON/hash 0.266 s, tensor assembly 0.0144 s and snapshots
0.00693 s. Forwards remain 1,281 → 21; native steps 1,280; semantic witnesses
1,298; snapshots 64. Full Python parsing remains 1,298 → 2. Model children
are included in the model total. No single remaining category explains the
entire distance from singleton cost.

## Final compression check (2026-11-08)

`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/reference`
— all six cells pass full discrete receipt parity and numeric tolerance after
rebuilding and atomically installing the ordinary release wheel. Output:
`scratch/mtg-268-native-compress.json`; runtime SHA256
`94dbd622e1449d28633e9843bbf0099a19fad22cc37f363295c4eb16dfdf9885`.
At 1/4/16 worlds, final seconds are 0.131/0.419/1.559 (42) and
0.146/0.498/1.877 (43). Same-allocation speedup is 2.18–2.82x; 16-world
ratios are 5.46x/5.72x the original singleton. Maximum base/value/target
errors remain 1.49e-10/5.96e-8/6.93e-9, including aggregate action values.
Same fixture and single-sample limits as above. Comparator mutation probes
reject altered schema, aggregate values, missing-value support and audit JSON.

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

Remaining implementation: further reduce the measured model, witness, encoding
and semantic serialization costs without discarding evidence or legal support,
then repeat the bounded comparison. The latency outcome is incomplete; do not
mark the Task complete or relax it to the achieved speedup. A new optimization
must have an ownership/invalidation argument if it reuses state-derived evidence.

Open for demo: the numeric ratio and representative positions defining "near".
MTG-183/w13 remains unmeasured and was not rebound to this world; no trained
strength claim. Memory still scales with worlds × supported root actions.
The README's 30-second example is an illustrative budget, not a latency promise.
