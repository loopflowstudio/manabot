# MTG-237 engine budget demo — 2026-11-02

Experience: a maintainer reads the Engine Budget check, distinguishes timing
from correctness failures, and adopts an intentional change from its downloaded
measurement with a reason. [Design](mtg-237-engine-performance-budget.md),
[PR #413](https://github.com/loopflowstudio/manabot/pull/413),
[procedure](../docs/development.md#engine-performance-budget).

All judgments here are delegated headless reviews. No human observed these
runs or supplied new feedback. Jack Heart's steer permits publication and demo,
explicitly excluding shipping. No merge, Task completion, browser observation
or HTML review is claimed.

## Current result: retained speedup

[Run 41555845634, job 125654705413](https://github.com/loopflowstudio/manabot/actions/runs/41555845634/job/125654705413)
passed at `4b5e9c0e4555627664721e7a4e6c909120cd98ac` on 2026-11-03 UTC.
Native source equals merged main `51fbacbbf`, including derived-state revisions,
indexed records and cache-carrying clones absent from reference `88dc8c1ad`.
Both builds, measurements, comparison and upload passed; the log shows explicit
Bash with pipefail, both summaries and `PASS: within the recorded budget`.
Replaying the downloaded comparison exited 0.

| Metric | Same-runner reference | Candidate | Improvement |
|---|---:|---:|---:|
| Worst search reply | 10.226 s | 7.132 s | 30.26% less time |
| Total search | 290.711 s | 208.251 s | 28.36% less time |
| Step throughput | 8,833.0/s | 13,943.8/s | 36.65% less time per step |

Throughput also means 57.86% more steps/s. Both engines ran on one Intel Xeon
6973P-C (4 vCPU, release). All three full-score digests, step digest, 512 games
and 99,923 steps match; no game was capped. No cross-run timing comparison.

`gh run download 41555845634 --name engine-budget` retrieved
[artifact 13195347162](https://github.com/loopflowstudio/manabot/actions/runs/41555845634/artifacts/13195347162).
Measurement identity matches the exact published head/run. Temporary adoption
through the real `record --reason` CLI preserved every field and tolerance 0.1.
Delegated judgment then applied the existing retained-speedup rule: otherwise
a loss of this optimization could pass against the slower reference.

**Implemented:** production baseline adopts this candidate using
`.cache/engine-budget-demo/pass-3/pr-head/`, with the exact head, run, merged
main, deltas and unchanged digests in its reason. Current-reference docs match;
the earlier accepted-cost and T4/T5/T6 evidence remain. Direct reconciliation
checked identity, every measurement, digests/counts, tolerance and deltas.
No runtime, workflow, benchmark or threshold changed.

**Remaining:** PR #413 publishes adoption/docs at `7cfd0f17e`, but its
description still names the previous reference and says this speedup is pending.
Update that description and publish final reconciliation edits. Run
[41558082651](https://github.com/loopflowstudio/manabot/actions/runs/41558082651)
is already measuring that head against the adopted reference; Engine Budget
and broader suites were in progress at reconciliation on 2026-11-02. Demo owns
result and artifact verification. No separate measurement dispatch. Main-push
verification remains post-merge; there is no shipping approval.

Evidence in `.cache/engine-budget-demo/pass-3/`: `pr-head/` (artifact),
`pr-head.log`, `comparison.txt`, `record.txt`, `adopted-baseline.json`,
`verified-result.json`, `adoption-check.json`, `pr.json`, `checks.json`.
The retained `adoption-check.json` describes pass 2's old reference, not pass 3
adoption. Reconciliation checked the production baseline directly against
`pr-head/measurement.json` and both candidate JSONL files: exact identity and
measurement equality, with tolerance 0.1. The stale receipt is retained as-is.
At the retained snapshot Engine Budget, debug Rust, Python, conformance,
Academy, clean-machine Play, long integration and package/install checks passed;
main integration was running and Tests Result pending. Do not treat that
snapshot as proof that every CI check finished.

## Retained failure and recovery evidence

### False-green pipeline and repair

[Run 41548559113, job 125630133797](https://github.com/loopflowstudio/manabot/actions/runs/41548559113/job/125630133797)
at `24cfc0a8b052a1e92933e826c0eb05ad2779ae87` reported success despite
`FAIL: 1 check(s) outside the budget`: candidate 6,197.6/s versus reference
6,821.4/s, below the 6,201.3/s limit. Search worst (14.585/14.071 s), total
(429.012/411.497 s) and all digests passed. The artifact names that head/run.

Implicit `/usr/bin/bash -e {0}` let `tee` hide exit 1. Replaying the same
artifact under `bash -e` exited 0; explicit Bash with pipefail exited 1 with
identical output. Fixed in `79c623428`, protecting both benchmarks and compare.
Eight regressions execute actual workflow run bodies: pass/slow/changed-score
comparison, each of four failed benchmark pipelines, and reference-build recovery.
Implicit-shell copies reproduce the slowdown and benchmark failures.

Evidence: `.cache/engine-budget-demo/pr-head/`, `pr-head.log`,
`default-shell-compare.txt`, `pipefail-compare.txt`; repaired replay at
`pass-2/retained-failure.txt` and `pass-2/retained-failure-summary.md` exits 1.
The earlier successful [run 41543638120](https://github.com/loopflowstudio/manabot/actions/runs/41543638120)
at `fc6cdcb702628c1ee2acf4770d36ff5437bd7aad` proved selection, durable-head
identity, summaries and upload, but not failure propagation. Its search
worst was 14.097/14.359 s, total 411.678/413.414 s and throughput
6,610.7/6,757.4 per second (candidate/reference). Artifact and log:
`.cache/engine-budget-pr-413/`, `.cache/engine-budget-pr-413.log`.

### Decision pass 2 — 2026-11-02

[Run 41552162812, job 125642805847](https://github.com/loopflowstudio/manabot/actions/runs/41552162812/job/125642805847)
at `88dc8c1ad25cd99cca6026581a569f5c2f63ee1b` **failed correctly** under
explicit Bash: comparison exit 1, successful builds/measurement and final
always-run upload. Both summaries were present.

| Metric | Same-runner reference | Candidate | Result |
|---|---:|---:|---|
| Worst reply | 13.913 s | 14.857 s | Pass |
| Total search | 411.956 s | 433.796 s | Pass |
| Throughput | 6,774.5/s | 6,104.0/s | Fail: minimum 6,158.6/s |
| Full-score/step digests and counts | Committed | Identical | Pass |

Candidate used **10.98% more time per step** (9.90% fewer steps/s). The
10% duration budget divides reference throughput by 1.1. This and the earlier
near-threshold failure were not dismissed as established runner noise.

[Artifact 13193795768](https://github.com/loopflowstudio/manabot/actions/runs/41552162812/artifacts/13193795768)
downloaded successfully; exact head/run and AMD EPYC 7763 identity matched.
The real recording CLI adopted it into a temporary baseline with tolerance 0.1.
This proves hosted transport/adoption after comparison failure. The reference
predated merged decklist disclosure `230b2f563` and T5 `a8abf01b1`; candidate
native source equaled that main. The comparison does not isolate which upstream
change caused the cost. Delegated judgment accepted that merged engine as the
initial reference with its explicit cost, and production adoption followed.
The sync onto main `51fbacbbf` subsequently introduced the optimization measured
above; syncing alone established neither speedup nor full-score equality.

Evidence: `.cache/engine-budget-demo/pass-2/` contains `pr-head/`,
`pr-head.log`, `record.txt`, `adopted-baseline.json`. Publication corrected the
PR to say measurements are recorded before reference build and uploaded at the
end, and documented the repaired enforcement and recovery limits.

## Controlled proofs and limits

`uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q`:
**65 passed in 2.48 s** at pass 3 (65 in 2.20 s at pass 2). Reuse for unchanged
implementation; no rerun for note curation. Earlier 57-test success preceded
the workflow regressions and did not establish failure propagation.

Controlled fixtures prove a 20% duration increase fails all three timing rows;
a faster candidate with a changed seed-2 score fails that digest; a 5% slowdown
on a slower runner passes. Original CLI transcripts remain under
`.cache/engine-budget-demo/{twenty-percent-slowdown,changed-score-digest-speedup,slower-runner-within-budget,record}.txt`.

Reference-build recovery injects benchmark binaries, machine metadata and
Cargo exit 42, executing actual workflow recording and real `record --reason`.
It copies the declared upload path locally and adopts the retained candidate.
This proves local recovery, **not hosted reference-build-failure transport**.
Hosted transport is proven only after the comparison failure above. No artificial
load was induced. Prior gate evidence (before the shell repair): 855 debug Rust
tests, 90 affected Python tests, actionlint, Rust formatting and Clippy, focused
Ruff passed. Gate owns broader suites; no new local performance run is needed.

The first full-score reference was `e5615edcc` from dispatch 41540798745
(`.cache/engine-budget-score-digest`): 120 decisions, 409.358 s total,
14.258 s worst, 99,923 steps in 512 games at 6,829.1/s on EPYC 7763.
Earlier T4/T5/T6 timings retain the winner-only digest limitation; development
docs preserve their values, without claiming full-score equality.

Complete earlier transcripts and judgments:
`4b5e9c0e4:scratch/engine-budget-demo.md`. Uncommitted pass-3 notes before
curation: `.cache/engine-budget-demo/pre-compress-4b5e9c0e4/`.
