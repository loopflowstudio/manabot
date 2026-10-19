# Protocol017 integration preparation for MTG-122

At 2026-10-18 19:32 UTC, the Mini evaluation was live (PID 68448) and its
one-shot audit/report owner was waiting (PID 69462). Retained panel metadata
contained 1,712 replay-passed games, 15 complete panels, four running panels,
and no recorded game or panel errors. No score-based choice was made.

The preceding goal turn was a verified wait: both exact launchd jobs were
observed live. This turn also delivered new integration evidence, not another
training/evaluation attempt.

All eleven exact candidate recipes, their original index, the frozen protocol,
and three relevant complete source files were copied to MTG-122's existing
checkout at `scratch/mtg-135-longer-learning-017-preparation/`. Its working design
now links that package. Eighteen files totaling 113,637 bytes passed SHA256/length
verification; manifest SHA256 is
`a6a62c2e22e2595a2d315da17675f19e83b120fca5391497678c6a172173ee8d`.

Static AST/JSON inspection established that the receiving checkout lacks
`AtaraxosMoveLearning.iteration_schedule`, and `ComplexityPlan` requires at least
1,000 iterations in multiples of 100. The tested 64-stream recipe has 128
iterations, 524,288 transitions and a schedule mapped to 2,048 reference
iterations. Dropping the schedule field changes endpoint tau from
0.005076577477226473 to approximately 0.011663. Keeping 1,000 iterations instead
changes the experience horizon to 4,096,000 transitions. These are concrete
integration issues to resolve in MTG-122's final source, not launch blockers
invented by this handoff or permission to replace its controls/cohort.

Selection remains pending the full 22-fit/18,304-game comparison and its audit.
The final recommendation must explicitly address 64 streams; a 256-stream
winner would be an alternative. Preserve all 36,000 MTG-122 assessment games,
its common research catalog/fresh initialization, separate $35 cap, accepted
GPU-training/Mini-evaluation split and prohibition on laptop science. No model,
source, running process, paid allocation or scientific schedule was changed.

## Operational receipt handoff

A separate `scratch/mtg-135-mini-operations-20261018/` package now resides in
MTG-122's checkout. Its 98 files (2,596,902 bytes) retain both Mini calibration
attempts, the cancelled laptop attempt's metadata, parity/transfer receipts,
logging repair and exact evaluation/audit owner admissions and source. Manifest
SHA256 is `6343d04e92d2bd16bde9402240c4fe5ed9668e61812d8d992aa04f4ed7c3ab73`.
Every copied byte was checked. The original failed-control manifest remains
unchanged; its post-exit inventory explicitly covers the final traceback-bearing
log. A completed control or accepted admission is not full scientific completion.
The package also qualifies the renderer's historical budget wording: ancillary
reserve is not measured S3 cost attribution or an all-inclusive spending proof.

The full tested source pack was also copied to MTG-122's
`.runs/mtg135-handoff/protocol017-source/` and documented in
`scratch/mtg-135-protocol017-source.md`. It contains 2,208 tracked files/2,378 Git
objects, is 105,942,147 bytes, and has SHA256
`5c41c3af06513e8c3bfc3569e38bd6298295d29ca3811282fdbe77a19153468b`.
An isolated import and `git fsck --strict --no-reflogs` passed, with exact source
commit and tree identity verified. No source branch, code checkout, runtime,
model or active process was changed. This preserves the prepared scientific
commit even though no current branch contains it.

## Completion retention check — 2026-10-18 21:46 UTC

Completing SDK Task 88f96c removed its clean checkout. MTG-135 must retain its
research evidence and unrelated contributions before eventual completion.
Fresh `lf wt list --json` reports this laptop checkout dirty, and its current
head is later than the recorded merged PR 309 head. The inspected cleanup path
refuses uncommitted work and work beyond the merged head; do not discard either
to make cleanup succeed.

The Mini copy is an independent repository: `.git` is a directory, Git's common
directory resolves to `/Users/jack/src/managym-cpu-research/.git` on that host,
and HEAD is the frozen `3c3681d2b26a7fc8167928fa63fe24a6cd3d058a`. It contains
the verified training cohort and its own ongoing evaluation; local Task cleanup
does not supply authority to remove that separate copy. No files were moved or
changed on the Mini during this check.

The final MTG-122 delivery must include the complete audited results, figures,
recipe and evidence manifest as actual verified files, with retained raw evidence
and cancelled/failed attempts still accessible. Keep the laptop's cancelled
attempt separate from the Mini result. Do not use a successful merge or a Task
cleanup action as evidence that this scientific handoff is complete.

At 2026-10-18 22:00 UTC, the receiving design's interim-handoff paragraph was
reconciled with its existing scope clarification: MTG-122's current accepted
direction governs execution. The obsolete instruction to keep an execution hold
was removed from that paragraph. No Task was resumed, no scientific recipe was
selected, and no process or allocation was changed. The current accepted split,
full controls and pending complete-comparison requirement remain in place.

## Observed Mini evaluation time — 2026-10-18 22:22 UTC

Read-only inspection of the live Mini controller's completed-command events and
their start receipts provides a separate planning constraint for MTG-122. These
seed13591 commands each completed all four raw-checkpoint panels and 416 games
against one anchor. All rows below use policies trained with 64 streams.

| Model | Greedy command, minutes | Random command, minutes |
|---|---:|---:|
| 64×2 | 20.9 | 22.7 |
| 64×4 | 21.3 | 24.1 |
| 192×4 | 31.3 | 39.2 |
| 384×2 | 54.9 | 66.5 |
| 384×8 | 67.7 | Still running at this observation |

The authoritative receipts are the Mini's
`.runs/grid-007/learning-017/evaluation/controller.log` command-finished events,
the corresponding `commands/<name>/worker-status.json` start times, and each
command's four completed `panels/<name>/transitions-*/monitor.json` files. The
complete final handoff must retain these receipts with the full cohort.

These are whole-command elapsed times under the admitted four-worker CPU route,
including loading, games, replay and publication. They are not isolated model
inference measurements: policy-dependent game lengths and the concurrent model
mix also contribute. They do not change the frozen training-recipe selection
rule or supply a final playing-strength ranking. MTG-122 needs its own calibrated
Mini allocation for the shared research catalog and three pools; 36,000 games
must not be priced from GPU training speed or a small-model evaluation rate alone.

Planning arithmetic checked again at 2026-10-18 23:36 UTC: the two completed
384×2/64-stream commands took 3,292.129593 and 3,987.447525 seconds for 416 games
each. With the same equal anchor mix, checkpoint mix and measured command
durations, 36,000 games would consume about 87.5 worker-hours, or 21.9 elapsed
hours with ideal overlap across four slots. The calculation is
`(3292.129593 + 3987.447525) / 832 * 36000 / 3600 / 4`.

This is an extrapolation from one training seed on the current workload, not a
calibrated MTG-122 allocation or a lower bound. Its shared research catalog,
three pools, policies, game lengths and concurrent model mix can change the
runtime. Additional startup, final audit and report time also need allowance.
Use the complete timing receipts and a calibration on MTG-122's actual workload
to set its Mini deadlines while preserving the accepted 36,000-game scope.

At 2026-10-19 01:27 UTC, a fresh receiver read found that `scratch/questions.md`
still ended with the interim handoff's request to reconfirm an execution hold.
An appended clarification now links the already delivered scope clarification
and recipe preparation, preserving the historical entry while stating that the
interim handoff added no user instruction or approval requirement. MTG-122's own
accepted direction remains authoritative. No process, recipe, cohort, budget or
existing user hold was changed; the complete comparison and final handoff are
still pending.

## Final delivery — 2026-10-19

The audited protocol017 package was delivered to MTG-122 and verified there:
the report at `scratch/mtg-135-learning-017-final/` (53/53 manifest hashes pass)
and the full 9,553-file package at `.runs/mtg135-handoff/learning-017-final/`
(byte-identical to the laptop copy at `.runs/grid-007/learning-017-mini-final-20261019/`).
The operator review is in that package's README.md. Recommendation: 64×2 for
MTG-122's 64-stream plan; the frozen rule's overall pick is 64×2 at 256 streams.
MTG-122's design links the package. No MTG-122 authorization, cohort or budget changed.
