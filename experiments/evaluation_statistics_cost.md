# Evaluation statistics cost

## Question and method

Jack Heart, 2026-11-17: "fix the unified path; run some performance numbers to establish impact of stats code".

The intended claim is the end-to-end cost of collecting all-game card and
custom statistics on the real two-deal INT-6 code-anchor evaluation, and the
cost saved by sharing the verification traversal. This is exploratory, not a
claim about playing strength or CUDA inference.

`experiments/evaluation_statistics_cost.py` runs six arms, sequentially in
fresh processes, with five rotating repeats on one host. Each cell plays deals
3 and 4, both seat legs (four games), random versus scripted-greedy; every game must terminate
without error and pass replay. The exact fixture key and registration records
are shared by all sources. Cells use the same CPython/dependencies and thread
budgets. No artificial contention is induced. Timed work is `play_cell`, card
count reduction/retention and, for the split arm, the additional offline
reduction. It includes child startup, game execution, durable records, replay,
statistics and file writes. Parent import/setup and post-run hashing are outside
the timed region. Replay performs no policy inference.

- **main-cards:** last merged main `5af8490a1c`, its release-built engine and
  ordinary audited card statistics; a pre-redesign reference.
- **before-off:** kept pre-fix `de87823407`, verification without card evidence
  or custom statistics.
- **before-custom:** kept pre-fix, audited card statistics plus the separate
  offline pass needed for custom statistics.
- **edited-off:** new implementation, verification without statistics.
- **edited-cards:** new implementation, audited card statistics only.
- **edited-custom:** new implementation, card statistics plus Behaviour,
  conditional land/spell sequencing and draw counts in the verification pass.

Before and edited use byte-identical native extensions: no Rust source changed
in this slice. Main uses the separately built matching release extension.
All sources are isolated source archives, not extra managed worktrees.
The harness refuses its comparison if game actions/state digests/outcomes,
audited card rows, or custom tables differ. Tables are private and complete
verified games only. Five repetitions report medians and full ranges; intervals
that overlap do not establish a difference. There is no network/Etude storage,
bootstrap analysis or neural/search policy inference in the timed workload.

Replay RSS is the child's resident set sampled at replay completion, not peak
memory. Output bytes include aggregate/per-deal records, retained card evidence,
statistics, row/performance reports and any failure evidence actually left by
the cell. They exclude imported sources, build files and the measurement JSON.

## Admission

Current focused checks: 12 reducer/record/export checks passed; 52 evaluation,
package-boundary and child-isolation checks passed, including a real checkpoint
card evaluation. A representative script-local reducer then exposed that PyO3
ActionEnum types need public-name restoration across cloudpickle. The fix is
local to reducer serialization; the expanded nine reducer/record checks passed,
including script-local dataclass rows, native enum references, rejected replay,
error retention and a single walker shared with card evidence. No reducer or
policy timing from those checks is a reported measurement.

## Results

Pending the frozen, repeated comparison. No performance claim yet.

## CPU-work panel

The completed wall-time panel is retained unchanged: its ranges overlap too
widely to establish a throughput effect. A second, separately frozen panel
runs the same four already-recorded games through the real verification body,
measuring `time.process_time()` and elapsed seconds. Five rotating repetitions
compare edited verification only, edited verification plus audited card counts,
kept split-pass card/custom analysis, and edited unified card/custom analysis.
It excludes worker startup, durable output writes and reducer transport, which
remain included in the first panel. Imports, source binding and record parsing
are outside this timing region; validation/replay/capture, count reduction,
custom callbacks/final aggregation and table conversion are inside. It checks
equal verified decisions, audited card rows and custom tables. CPU seconds
measure scheduled work, not games/second or GPU cost. This panel cannot replace
the noisy end-to-end result.
