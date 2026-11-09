# MTG-268 — batched local-update rollouts

## Direction and next steps (2026-11-09)

Jack Heart asked, "isnt this a performance optimization", then invoked
`$optimize`. Step 5 chose **advance**: the equal-work baseline improves, but the
dtype trial did not demonstrate an end-to-end gain and was fully reverted.
Next: sync, realign and final measurement; no further optimization
is selected. Finer native Command/audit attribution on a controlled host is a
possible future investigation, not an implementation hypothesis for this run.
Do not launch a Graph, publish or ship.

The Task directive is the design; no separate accepted design was present:
"Batching should bring 8-16 worlds near today's one-world cost."
That outcome remains unmet. Equal-work speedup does not replace this objective,
and no numeric definition of "near" or relaxation has been agreed.
The directive also says: "Never run games/evals on the laptop; small fixed-position timing benches are fine, or use the Mac mini (mini-heart.local)."

Completed investigation directions, retained verbatim:

- "Investigate the 0.259 s semantic JSON/hash cost"
- "Preserve exact viewer/event semantics and hash bytes; do not substitute a witness hash without proving equivalence."
- "Rebuild and atomically install the merged ordinary wheel; add direct/composite hash parity coverage for the pending-public-entrant state."
- "Separate allocation, conversion and native encoding costs, and implement a bounded reduction only where equivalence is demonstrated."

Step 6 reviewed the full branch diff against checkout base `00c7f1bba`, its
callers, tests, configuration and docs. No further code reduction was justified:
the live frontier, batch prediction, semantic projection and fact-program paths
already have single owners. This pass only curates notes. The full pre-curation
notes, including every historical table, command, failure and trial plan, are
committed at `c0c3c2b4476755676115c602a97f57bacac6a7ea:scratch/questions.md`.
`scratch/measure.html` and all raw evidence stay unchanged. Source, harness and
installed runtime remain the last-kept implementation `44895a104`.

## Approach and Delete — do not maintain

Keep independent policy RNGs, stable world/action receipt order, every receipt
field and private audit, native Commands, canonical hashes, full legal support,
terminal/sign semantics, sampling identities, atomic rejection and deadlines.
No input/checkpoint/schema change or batching flag. Semantic SHA-256 is rebuilt
under the Env lock on every read; witness BLAKE3 is not equivalent. Memory still
scales with worlds × supported root actions.

Completed deletions; no remaining deletion target selected:

- Sequential production rollouts and separate singleton inference: one batch
  implementation; git supplies the timing oracle.
- Per-ply Python semantic parsing, unused composite envelopes and repeated
  root/config/prepared-constraint parsing. Real composite consumers remain.
- Pre-Command viewer projections, duplicate equivalent acting-view serialization
  and revalidation of a Command just constructed under both locks.
- Globally empty CPU attention/fact columns, preserving coordinates, historical
  pooling and program order. No state-derived cache.
- Receipt-field allowlists, duplicate fact-word decoding, validity conversions,
  fact-program guards and native profiling lifecycles.
- Scalar Python-list conversion in native array returns and per-row Python
  encoding/stack copies in `predict_batch`. Keep the Python encoder oracle,
  selected capacities and independent writable arrays.

Compound execution was already removed (`docs/training.md#compound-decisions-removed`);
retain historical readers, not the directive's obsolete sequential fallback.
The rejected dtype-name experiment and its tests stay reverted.

## Measurement identities and protocol

These are initialized-policy fixed-position measurements, not trained-strength
evidence: Apple M4 Max/macOS 26.0.1, CPython 3.12.12, Torch 2.10.0, locked checkout
venv, one CPU/inter-op thread, warmed scalar 64-wide/two-layer model, opening
seeds 42/43, search seed 17, depth 20, 1/4/16 worlds. No games, evaluations,
induced contention or concurrent benchmark workers. The harness times
`teacher.search`; setup, warming, profiles, comparisons and binary swaps are
outside the latency region. Categories are inclusive and can overlap;
cProfile magnifies Python call overhead. Five samples are not confidence intervals.

- Harness SHA256: `3e1c2bf414c4a7654fd6741eede6f979f704a14f9d4cbc2cd29cb76c4496d67a`.
- Ordinary baseline/restored runtime: `0b286acd043f21ecfcf8ca7e9c3ad449ee38b3dffaa0aa8914c311d733d15a46`.
- Rejected ordinary candidate runtime: `236c5a7c7683cff0f71478f2cca14ccf35f758aafbcfd5122695f98a68bf8e66`.
- Merged-world checkpoint: `57fd8f999d15426e9cebb73ce0f917fe987757f992bba2d5ab0b5c41542bddeb`.

Full discrete parity includes support, hashes, Commands, audits, terminal flags,
sampling, world and checkpoint identities. Maximum base/value/target errors in
both repeated studies: `1.0242171e-10 / 4.09781933e-8 / 3.13064846e-9`, within
`atol 1e-6 / rtol 1e-5`. Runtime/source identity differences are explicit comparator
exceptions, never a rewrite of saved evidence.

## Equal-work baseline (optimize step 1)

Five serial invocations of
`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference`
— 30 sequential/batched pairs; all 30 live and 30 saved-receipt comparisons pass,
zero errors, retries or deadline rejections. Each side executes 945 rollouts /
18,900 Commands. Each cell runs sequential then batched in the same process.
All three reference Python execution owners come from `841b1c50c`, sharing the
ordinary native runtime and current dependencies with `44895a104`. This measures
all retained Python search/model cuts, not batching alone or a historical engine.

Seconds per decision: median [minimum, maximum], n=5. Speedup divides medians.

| Position | Worlds | Sequential seconds | Batched seconds | Speedup | Rollouts / Commands per search |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.25963 [0.20944, 0.29722] | 0.10078 [0.08264, 0.11796] | 2.58× | 4 / 80 |
| 42 | 4 | 1.05051 [0.80673, 1.19994] | 0.34665 [0.29120, 0.74524] | 3.03× | 16 / 320 |
| 42 | 16 | 4.24566 [3.69680, 4.32050] | 1.26913 [1.22429, 1.27389] | 3.35× | 64 / 1280 |
| 43 | 1 | 0.32751 [0.29087, 0.36039] | 0.12348 [0.10428, 0.13945] | 2.65× | 5 / 100 |
| 43 | 4 | 1.30797 [1.25916, 1.72466] | 0.41476 [0.38302, 0.48030] | 3.15× | 20 / 400 |
| 43 | 16 | 5.57084 [4.48144, 6.10790] | 1.53841 [1.25904, 1.67621] | 3.62× | 80 / 1600 |

All six ranges are disjoint. Paired sixteen-world speedups range 2.910–3.487×
(42) / 3.460–4.092× (43). The noisy 42/four-world batched sample (0.74524 s)
is retained; its cause was not measured. Sixteen-world/current sequential-singleton
median ratios are 4.89× / 4.70×, diagnostic only.

Separate ordinary-wheel profile at 42/16: total 1.22840 s; its search and
incidental unprofiled cell are excluded from the five-sample baseline.

| Batched category | Calls | Profile seconds | Share of profile | Impossible full-removal ceiling |
| --- | --- | --- | --- | --- |
| native_step_and_audit | 1280 | 0.4950 | 40.3% | 1.675× |
| network | 21 | 0.2917 | 23.7% | 1.311× |
| semantic_direct_hash | 1296 | 0.1917 | 15.6% | 1.185× |
| observation_encoding | 1281 | 0.1899 | 15.5% | 1.183× |
| session_snapshot | 64 | 0.0090 | 0.7% | 1.007× |

Network forwards fall 1,281 → 21 (2.54701 → 0.29167 profile seconds), composite
reads 1,298 → 2 plus 1,296 direct hashes, encoding 0.40101 → 0.18988 s.
Command count stays 1,280 (0.53381 → 0.49500 s). The profile cannot split the
native Command/audit bucket further. NumPy `_name_get` accounts for 56,364 calls,
0.09521 s (7.75%; optimistic complete-removal ceiling 1.084×), motivating the
later rejected trial. These are profile mechanisms, not unprofiled savings.

Artifacts: `scratch/mtg-268-baseline-{1..5}.{json,err}`,
`scratch/mtg-268-baseline-summary.json`, and
`scratch/mtg-268-baseline-{before,after}.pstats`; full diagnostic invocation:
`uv run --extra dev python scratch/mtg268_profile.py --worlds 16 --position-seeds 42 --native-reference .cache/mtg-268/merged-reference`.
Its JSON/stderr are `scratch/mtg-268-baseline-profile.{json,err}`.

## Dtype trial — rejected and restored (optimize steps 2–4)

Hypothesis: “Primitive dtype metadata validation reduces median batched
16-world decision latency by 3–5% on fixed openings 42/43 at depth 20, with
unchanged receipts and numerical predictions within atol 1e-6 / rtol 1e-5.”
The trial compared primitive kind/itemsize instead of constructing dtype names
on successful scalar/vector validation, retaining names for errors, complete
pre-write checks and scalar `np.copyto` endian conversion. It changed no encoder,
allocation, model, hash or Command semantics.

Five serial baseline/candidate grid pairs used current batched Python owners
on both already-built ordinary binaries. Each atomic swap was followed by a
fresh process. All ten latency processes exited zero: 60 live and 60 saved
comparisons passed. Each wheel's timed batched searches and its sequential
controls each executed 945 rollouts / 18,900 Commands. No samples were removed,
retried or replaced; standalone parity/profile timings were excluded.

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

**Inconclusive under the declared overlap decision rule.** All six ranges
overlap; sixteen-world medians fall 52.6/96.6 ms, while all four smaller-cell
medians rise. The 0.76723 s 42/four-world candidate sample remains unexplained.
Resolving a 3–5% gain would require a newly declared protocol on a quieter host;
this result does not start that investigation.

Separate profile searches at 42/16, n=5 per wheel:

| Category | Calls | Baseline profile s | Candidate profile s | Verdict |
| --- | --- | --- | --- | --- |
| native_step_and_audit | 1280 | 0.51045 [0.45361, 0.55229] | 0.58284 [0.46486, 0.78931] | No measurable difference (profile only) |
| network | 21 | 0.30453 [0.27113, 0.35454] | 0.34090 [0.27540, 0.38127] | No measurable difference (profile only) |
| semantic_direct_hash | 1296 | 0.19909 [0.17477, 0.21178] | 0.25656 [0.17967, 0.32013] | No measurable difference (profile only) |
| observation_encoding | 1281 | 0.19549 [0.17280, 0.23433] | 0.07677 [0.06826, 0.10713] | Better (profile only) |
| session_snapshot | 64 | 0.00903 [0.00750, 0.01128] | 0.00987 [0.00794, 0.01189] | No measurable difference (profile only) |
| observation_allocation | 21 | 0.00253 [0.00228, 0.00338] | 0.00263 [0.00239, 0.00279] | No measurable difference (profile only) |
| tensor_assembly | 462 | 0.00078 [0.00071, 0.00083] | 0.00081 [0.00073, 0.00086] | No measurable difference (profile only) |

The 60.7% profiled encoding reduction has disjoint ranges; every other listed
bucket overlaps. An additional full profile (n=1/side) attributes all 56,364
baseline `_name_get` calls to `Env.encode_observation_into`; the candidate has
zero. Dtype names take 0.09052 of 1.18372 s (7.65%, full-removal ceiling 1.083×).
Encoding takes 0.18232 s versus 0.06452 of the candidate's 1.00090 s. Native
Command/audit remains largest: 0.48073 s (40.6%) versus 0.45022 s (45.0%);
candidate network/hash shares are 26.1%/17.3%. None establishes a latency gain.

Verification retained from the trial: 30 compatibility cases passed on the
baseline and ordinary candidate; 80 focused cases passed on the debug candidate,
including v1–v9 history-off/on selected-capacity and independent-output checks.
Both debug/release builds passed serially with `CARGO_BUILD_JOBS=2`; focused
Ruff, Cargo fmt and diff checks passed. Initial fixtures wrongly called
`VectorEnv.close()`, requested unavailable f16 storage and expected vector
swapped-endian rejection; corrections preserved measured baseline behavior.
Scalar swapped-endian copies pass; vector binding accepts swapped byte order.
No new endian rule was introduced. The NumPy alias probe found no supported
primitive kind/size alias with a differing dtype name; numeric dtype codes
would have rejected equivalent int64/longlong names (7 versus 9).

One extra diagnostic failed before search: wrapper `profile.py` shadowed the
stdlib. Its empty JSON and traceback remain as `profile-baseline-setup-failed.*`;
renaming it to `capture_profile.py` fixed setup. Both full profiles then passed.
The failed setup is absent measurement, never a zero timing.

The three Rust binding files and added dtype tests were restored byte-for-byte
to `44895a104`; the preserved baseline binary was atomically reinstalled.
`uv run --extra dev python scripts/bench_local_update.py --native-reference .cache/mtg-268/merged-reference --compare-encoders`
— restored wheel passes all six live/saved pairs and 1,281 exact encodings;
incidental timings excluded. All 40 step-3 artifacts except the deliberately
replaced installation reverified unchanged. Earlier batching work survives.

All trial paths below are under `.cache/mtg-268/dtype-validation/`:

- `candidate.patch`, `{baseline,candidate}.cpython-312-darwin.so`,
  `candidate.whl`, `debug-wheels/`: rejected diff and preserved binaries.
- `artifacts.json`, `measurement-artifacts.json`: hashes, verification logs,
  source/harness and restored installation identities. Debug extension SHA256:
  `ff7d5a0923ba630f34726ba5ebfb20998416b6c0a6f20f190556a277a7ef99e3`.
- `measurement-runs.json`, `measure.py`: exact ten commands, order and exit codes.
- `measure-{1..5}-{baseline,candidate}.{json,err}`, `measurement-summary.json`:
  every latency sample, sequential control and parity maximum.
- `profile-{baseline,candidate}-{batched,sequential}.{pstats,json}`,
  `capture_profile.py`: full profiles; earlier profiles remain untouched.
- `{baseline,candidate}-name-calls.json`: v6 mechanism probe (44 → 0 calls),
  `candidate-parity.json`, `restored-parity.json`, build/test logs: implementation,
  restoration checks and failures.

## Earlier evidence and world boundary

The 2026-11-08 realign refresh found remote main at `0aa7dabbd` (#433), already
included through `00c7f1bba` in merge `0f5380bc9`; no new Task direction or Wave.
`841b1c50c` remains the historical benchmark reference, not current upstream.
The merge changed catalog/continuation semantics: original-runtime receipt
comparison fails at 42/1 (`scratch/mtg-268-merged-before.err`). Original receipts
in `.cache/mtg-268/reference/` stay frozen; merged references live separately in
`.cache/mtg-268/merged-reference/`. No cross-world speedup or w13 admission follows.

Pre-sync tables, commands and contrary findings: `0f5380bc9:scratch/questions.md`;
native attribution: `6b4a511:scratch/questions.md`; fact compaction:
`f9e3801:scratch/questions.md`. Uncommitted pre-encoding-compress notes were saved
at `.cache/mtg-268/questions-before-encoding-compress.md`. The original binary
is `.cache/mtg-268/reference-native.cpython-312-darwin.so` (SHA256
`21865ee8f4b9bbde2ef6463af4d583a8edd6b305c9b5d4a98e95fe813df61476`).
Pre-sync 16-world/original-singleton ratios progressed 8.0–8.1 → 5.46/5.72 →
4.55/4.92 → 3.69/4.16, each a single sample. Contrary direct-hash result:
43/singleton rose 0.1261 → 0.17281 s. Parity maxima were
1.49e-10 / 5.96e-8 / 6.92e-9; composite constructions fell 1,348 → 52.
The source-digest repair added `facts.py`; older receipts were never rewritten.

Merged encoding evidence: `scratch/mtg-268-merged-before.json`,
`scratch/mtg-268-encoding-profile-{before,after}.json`,
`scratch/mtg-268-encoding-after.json`. Instrumented native output conversion/copy
fell 1.012 → 0.102 s; whole standalone native call 1.028 → 0.117 s, with all
arrays equal across 1,281 views. Original Python probe was 0.312 s with 0.017 s allocation;
final native allocation/encode were 0.00403/0.00434 s. Instrumentation is
attribution only (`--features python,benchmark` includes other timers too).

Ordinary-wheel decision latency, n=1 per cell:

| Worlds | Position 42 before encoding cut → final (s) | Position 43 before encoding cut → final (s) |
| --- | ---: | ---: |
| 1 | 0.09170 → 0.08737 | 0.11086 → 0.10685 |
| 4 | 0.29665 → 0.26562 | 0.35659 → 0.36279 |
| 16 | 1.07097 → 0.99607 | 1.33296 → 1.25789 |

43/four-worlds slowed. Final sixteen-world costs were 4.5483/4.5722 times the
merged sequential singleton (0.21900/0.27512 s), not the original-world singleton.
Pending-public-entrant direct/composite/independent SHA-256 parity passed in
debug (`hidden_selection_public_entry_is_pinned_through_both_samplers`). All 45
distinct selected Python cases passed after history fixtures supplied missing
`recent_events`/world settings. Later encoding compression passed 18 fact/program
cases and the six-cell/1,281-view check (`scratch/mtg-268-encoding-compress.json`);
its timing is not a speedup claim because test startup briefly overlapped.
Full commands/results remain in the pre-curation commit's encoding sections.

## Remaining gate and final-measurement requirements

Gate owns complete local-update suites; model suites `test_value_aggregation`,
`test_inference`, `test_agent`, `test_layer_input`, `test_record_vocabulary`;
`tests/training/test_active_training.py`, `tests/env/test_observation.py`,
`tests/training/test_action_cutoff.py`, `tests/verify/test_native_buffer_capacities.py`;
full debug Rust suite, all-feature Clippy and applicable engine-budget CI checks.
Focused checks and release builds do not replace these requirements.

Final measurement must use the restored ordinary wheel, retain all samples,
separate profile/setup time, and account explicitly for any sync-driven identity
change. Never compare across worlds or rebind frozen receipts. Preserve both
reference paths; `--retain-reference DIR` saves receipts before runtime replacement,
`--native-reference DIR` checks saved ones. This notes-only compression requires
no new build, behavior test or latency claim.

Open for demo: what maximum 8–16-world/original-singleton latency ratio, checkpoint
and representative positions define "near"? Eight worlds remain unmeasured;
these openings are not terminal-heavy. MTG-183/w13 admission/evidence remains
unresolved; there is no trained-strength claim. The README's 30 seconds is a
budget, not a promise. Command/audit remains the largest bucket, without a newly
identified removable substep; snapshots/allocation cannot explain the target gap.

Compression validation: `uv run python -` (inline table/manifest checks) — all
five tables preserved verbatim; 103 manifest entries verified, including the
restored runtime, harness and unchanged HTML. The mutable notes and superseded
candidate installation hash are excluded; manifests remain unchanged. All
non-scratch tracked files match `44895a104`. `git diff --check` passes;
`lf context` is below both scratch targets. No behavior tests or benchmarks
rerun for this notes-only diff; no background process started.

## Main sync — 2026-11-09

Assumption (reversible): keep MTG-268's branch-specific notes here. The
unrelated representation study's complete notes remain at
`4e0143b7f0b4c84ca48abf4d3d3d274fa23fd909:scratch/questions.md`;
its experiment direction and landing approval do not select work for MTG-268.
Main's code merges without conflicts. The measurements above predate this sync;
the installed native runtime is unchanged, so they establish no post-sync
performance or runtime/source parity. Rebuild and identity accounting remain
for gate and final measurement.
