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

**Inconclusive throughput effect; keep the unified lifecycle for removing the
second analysis traversal.** All 30 cells passed (120 games, 8,280 decisions,
zero execution/replay failures). Gameplay hashes, card counts and custom tables
matched. Apple M4 Max, CPython 3.13.15, shared host; no GPU inference.
The end-to-end harness was frozen at `bfff4b975486`; the CPU panel was frozen
separately at `611c23b68e9d` without changing the first panel.

Seconds per four-game cell, five repetitions each:

| Arm | Median seconds | Full range | Output bytes/game |
| --- | ---: | ---: | ---: |
| Main card statistics | 4.093 | 2.291–6.648 | 87,487 |
| Before, statistics off | 4.421 | 3.383–6.069 | 480,254 |
| Before, separate custom analysis | 4.047 | 3.021–11.155 | 525,230 |
| Edited, statistics off | 3.510 | 3.360–9.042 | 480,308 |
| Edited, card statistics | 3.568 | 2.800–5.383 | 525,234 |
| Edited, unified custom statistics | 5.425 | 3.983–6.043 | 525,342 |

The overlapping ranges do not establish a slowdown or speedup. A quiet,
controlled host with longer representative evaluation cells would settle
end-to-end overhead; these numbers do not quantify GPU evaluation cost.

**Storage is a separate material cost.** Main retained gzip Command traces;
the redesign currently retains uncompressed canonical JSONL, including aggregate
and deal shards. That accounts for most of the roughly sixfold byte increase
versus main, and predates this reducer fix. Within edited evaluation, card
evidence adds about 45 KB/game, while these three aggregated custom tables add
about 108 bytes/game. Neither includes remote transfer/storage charges.

An unreported profile of verified replay attributes most Python work to viewer
projection, state digests, deepcopy and runtime identity construction. Reducer
callbacks were a small part of that profile. Profiling identifies source work;
its elapsed time is not used as a benchmark result. Large reducers can still
cost arbitrarily more than these examples.

Local raw evidence: `.runs/eval-stats-measured/{measurements,summary}.json`.
Gameplay SHA256: `3f6227e28e5c3b54d0f7ea6680220b630b72e83204afcb7635e9423607fcdd0c`.
Card SHA256: `b0758035c71c0bc8df3dcc6a6f08a7f562514b093f5e68b76eae2f0f376ff9b0`.
Custom SHA256: `b068c0d59bd6dcc9b26c3a9676037652ca27ddcf77cb6c982ae78ff8a3a1aa91`.


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

All 20 CPU-panel cells verified the same four games / 276 decisions, with
identical card/custom hashes and zero replay failures.

| Arm | Median CPU seconds / four games | Full range |
| --- | ---: | ---: |
| Verification only | 0.334 | 0.279–0.605 |
| Verification + card counts | 0.358 | 0.332–0.658 |
| Separate custom analysis | 0.509 | 0.435–0.683 |
| Unified custom analysis | 0.357 | 0.331–0.515 |

The observed median difference is 0.152 CPU seconds per four games between
split and unified analysis. The ranges overlap: **no measurable speedup under
the stated range criterion**, and no supported percentage overhead claim.
Unified custom work is about a tenth of the median end-to-end cell duration;
this ratio describes scale only, since the panels measure different quantities.
The functional reason to retain the fix is one traversal and one reducer
lifecycle across offline, live and evaluation use, with verified equal outputs.

CPU seconds by rotating repetition (0–4), retained so spread is reviewable:

| Arm | Samples |
| --- | --- |
| Verification | 0.604572, 0.334117, 0.347627, 0.283939, 0.279193 |
| Card counts | 0.658160, 0.418038, 0.358094, 0.332183, 0.356820 |
| Split | 0.683130, 0.538052, 0.509000, 0.434641, 0.475307 |
| Unified | 0.514556, 0.357313, 0.420935, 0.330797, 0.343355 |

Local raw CPU evidence: `.runs/eval-stats-cpu-measured/measurements.json`.
