# 855bff — standing pipeline speed

## Outcome and constraints

One bounded local command measures training and shared-inference evaluation across
`w64`, `w128l4`, `w160l8`, `a384l8`, retains a committed reference and compares real
sources on the same host. It reports stage attribution, batching utilization and
supplied-price rates. The default ladder's actual runtime must be measured;
the 3,600-second deadline is not evidence that it is practical.

No learning, collector, native, frozen recipe or evidence changes; no paid runs,
rentals, induced contention or cross-runner timing gate. CPU measurements do not
establish GPU speed. Batching optimization and long-run slowdown acceptance remain
separate outcomes.

## Implemented

`scripts/bench_training_update.py --suite` keeps `_execute` as the training executor.
`scripts/bench_pipeline.py` owns `WORKLOADS`, `run_suite` and `_measure_attempt`:
one bounded worker tree, retained attempts and cleanup, fixed v6/512-fact input,
float32, 256 streams and three updates. Evaluation uses initialized checkpoints
and four replay-verified BG/GR legs. Docs link the command and CI includes its tests.

`training_reading` excludes update one from collection/learning timing, using
`update_metrics.read_updates` / `stage_rows`. Batching counters are cumulative:
calls/update is a whole-stage average, rows/call and idle fraction include warmup,
not individual update measurements. In `SeatRoutedCollector._collect`, complete-game
streams idle only after meeting their quota and finishing a game; active rows then
shrink toward the tail. The existing counters expose this without a second store.

`compare` alternates clean reference and candidate sources for three repetitions,
retains all samples and selects whole minimum-duration samples after workload,
host, device, precision and behavior admission. Python and Torch versions are part
of host identity. Historical durations alone cannot produce a speedup. A reference
checkout is explicit and must contain the suite interface. MPS supports training
only; CPU/CUDA support both workloads. No device substitution.

## Reconciled with main

Merge `0a561319c` includes the Python 3.13/dependency changes (`b7c102f2c`) and
MTG-273 outcome-table repair (`1659b0e29`). The latter preserves timing keys while
`read_updates` expands selected outcome-table references. Keep that reader as the
owner; no benchmark-specific compatibility layer is needed. Python version was
missing from host admission and is now recorded and compared.

The earlier narrow CPU receipts were produced on dirty `402dfd5b2` source before
this sync; they demonstrate integration, not the merged source or a standing
reference. MTG-273's bounded index fixture does not prove long-run flat RSS or
collection time; neither will this three-update ladder. See
`experiments/self-play-collection-273.md` for that separate acceptance.

## Delete — do not maintain

Completed: the recording-only inline ladder check is replaced by
`_require_full_ladder`, shared with comparison admission. Coordinator comparison
coverage now uses the full ladder; subset diagnostics and retained errors remain.
No deletion targets remain. Test attempts derive their row identity from their
sample, avoiding nested unvalidated copies just to construct a full ladder.

## Remaining acceptance

1. Reference-host decision for demo: designate an owned CPU host for the complete
   suite, optionally a separate MPS training reference, or explicitly provision
   another host. Local CPU is a reversible candidate, not a settled designation.
2. On clean source with its matching release extension and locked environment,
   retain a complete default ladder, actual invocation time and all eight rows.
   Resolve capacity/runtime failures without silently changing workload identity.
   Review and commit `manabot/perf/baseline.json`; no baseline exists yet.
3. Run a real same-host interleaved reference/candidate comparison with matching
   suite options. Establish repeatable behavior admission; retain failures and
   incomparable results. Controlled fixtures alone do not satisfy this outcome.
4. Gate checks the merged metric-reader integration and configured evaluation CI
   collection. CUDA remains a named provisioned-host check; neither CUDA nor MPS
   has been exercised here. No speedup or learning-equivalence claim is established.

Prior validation: `uv run --locked --extra dev --extra evaluation pytest
tests/training/test_pipeline_speed.py tests/infra/test_ci_selection.py
tests/infra/test_release.py -q` — 78 passed before main sync. The earlier evaluation
CI collection had 579 cases before the checkpoint-identity test was added. Reuse
those results within that scope; focused realign validation and retained CPU
observations are in `scratch/questions.md`.

## Concept review — 2026-11-11

**Outcome for loop-or-next: open product choice (Blocked).** Which owned machine
and device should anchor the standing reference? The Task brief recommends the
mini, but does not designate it. Local CPU remains a reversible candidate, not
an answer. This determines where a future optimization can make a same-host
claim; MPS can anchor training only, and a different or rented host cannot use
the CPU reference as its control. No measurements or reference adoption occur
in this review.

### Experience and model

The interaction stays **measure → retain a reference → compare**. A reading is
one report containing attempts; a reference is a complete, clean reading
explicitly retained for a device's supported workloads. It needs no separate
service or lifecycle. A comparison reruns that source and the candidate; the
stored timing is context, not the denominator of a speedup. Keep `Sample`,
`Attempt` and `Report`: a failed attempt has evidence even when no sample exists.

The largest mismatch is a standing-reference workflow whose host and complete
reading do not yet exist. The implemented narrow proofs cannot fill that role.
Keep the default eight-row CPU/CUDA suite and the explicitly selected four-row
MPS training suite distinct; no device fallback or collector change follows
from low inference utilization.

**Reference admission repair:** recording and comparison now share the full-size,
device-supported-workload rule. A successful subset report supplied directly to
`--compare` rejects before workers start and retains its admission error. Subset
measurements remain available. Reference selection still requires the named host
and a reviewed complete reading.

### Usage and skill guidance

The reference section of `docs/development.md` now explains:
“A successful subset report is a diagnostic reading, not a reference. Both
recording and comparison require all four sizes and every workload supported by
the selected device. A rejected reference leaves an incomplete report; correct
the reference and retry with a new output directory.” Existing commands, rate
definitions and recovery guidance remain clear. No benchmark-specific skill
needs a new workflow; a future measure step consumes the same report and may
claim a speedup only from its admitted comparisons.

### Walkthrough and counterexamples

- Normal: on the designated CPU host, measure clean source, review and retain
  eight rows, then compare a candidate against a clean checkout of that source.
  `run_suite` owns execution; `compare` admits matching samples before choosing
  whole minima. Host, source, workload, behavior and evidence remain in the report.
- Failure/recovery: a game timeout retains prior rows, the failed attempt and
  its worker receipts, leaves the existing reference untouched and exits nonzero.
  Repair the cause and use a fresh directory; never combine old successes with
  a new attempt into an apparently continuous measurement.
- Counterexample: a successful one-size report is complete for its request but
  cannot become the standing ladder. The shared reference rule must reject it.
- Counterexample: a complete MPS training report is valid for that device, but
  cannot imply evaluation coverage. A changed behavior digest is incomparable,
  even if faster; it does not justify changing collection semantics here.

Next show the designated host, full-ladder elapsed time and retained rows, then
the real interleaved comparison. This review establishes neither runtime
practicality nor repeatable behavior admission.
