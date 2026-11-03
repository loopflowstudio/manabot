# MTG-237 engine budget demo — 2026-11-02

Experience: a maintainer reads the Engine Budget PR check, distinguishes a
slowdown from changed search results, and adopts an intentional change from
its downloaded measurement with a reason.

Design: [engine performance budget](mtg-237-engine-performance-budget.md).
Published change: [PR #413](https://github.com/loopflowstudio/manabot/pull/413).
Configured path: `.github/workflows/ci.yml`,
`.github/workflows/engine-budget.yml`, `scripts/engine_budget.py` and
`managym/perf/baseline.json`. Procedure:
[development guide](../docs/development.md#engine-performance-budget).

## Review authority and progress

The original review below is decision pass 1, run headlessly under the instruction to stand in for
Jack Heart at demo. No human observed this run or supplied new feedback.
The conclusions below are the agent's delegated judgment, not statements or
shipping approval from Jack Heart. His retained steer authorizes publication
and demo, explicitly excluding shipping approval.

The preceding work intended to publish and prove PR selection, durable-head
recording, candidate/reference summaries, same-runner comparison and artifact
identity. Publication and the earlier PR execution are now observed, materially
advancing that plan. There is no retained preceding demo result; that absence
is an evidence gap, not evidence that implementation made no progress.

## Observed behavior

- **Current-head failure:**
  [run 41548559113, Engine Budget job 125630133797](https://github.com/loopflowstudio/manabot/actions/runs/41548559113/job/125630133797)
  reports success at `24cfc0a8b052a1e92933e826c0eb05ad2779ae87`, while its
  comparison log says **FAIL: 1 check(s) outside the budget**. Candidate
  throughput is 6,197.6/s against 6,821.4/s, below the 6,201.3/s limit.
  Search worst reply (14.585 s versus 14.071 s), total (429.012 s versus
  411.497 s) and all digests pass. The downloaded artifact names the exact
  PR head and this run. Both summary blocks and the comparison execute.
  The artifact is retained at `.cache/engine-budget-demo/pr-head/`; the
  actual job log is `.cache/engine-budget-demo/pr-head.log`.
- **Cause reproduced:** the comparison step's logged shell is
  `/usr/bin/bash -e {0}`. The command pipes the comparator through `tee`,
  so `tee`'s success hides the comparator's exit 1. Replaying this artifact
  under `bash -e -c` exits **0**; replaying the identical pipeline under
  `bash --noprofile --norc -eo pipefail -c` exits **1**. Output is identical.
  Transcripts: `.cache/engine-budget-demo/default-shell-compare.txt` and
  `.cache/engine-budget-demo/pipefail-compare.txt`. The candidate/reference
  benchmark pipelines also use the implicit shell and need the same protection.
- [PR run 41543638120](https://github.com/loopflowstudio/manabot/actions/runs/41543638120)
  succeeded at `fc6cdcb702628c1ee2acf4770d36ff5437bd7aad`. Its Engine Budget job
  selected the actual PR head, emitted candidate and reference summaries,
  ran the flagless comparison and uploaded `engine-budget`. Downloaded
  `measurement.json` identifies that exact head and run. Search worst reply
  was 14.097 s against 14.359 s; total was 411.678 s against 413.414 s;
  throughput was 6,610.7/s against 6,757.4/s. All digest and budget rows passed.
  Evidence: `.cache/engine-budget-pr-413/` and
  `.cache/engine-budget-pr-413.log` (job 125613749944).
- Controlled CLI demonstration on this checkout: a 20% duration increase
  exited 1 with all three timing rows failing; unchanged digests still passed.
  A faster candidate with a changed seed-2 score digest exited 1 with exactly
  that digest row failing. A 5% slower candidate on a runner slower than the
  recorded machine exited 0. These are injected measurements, not induced
  machine load or new performance measurements.
- The actual `record --reason` CLI adopted the downloaded PR artifact into
  a temporary baseline. All measurement identity fields matched and the 10%
  tolerance survived. The committed baseline was not changed. This establishes
  ordinary artifact adoption, not recovery after a reference build failure.
- `uv run --extra dev pytest tests/infra/test_engine_budget.py
  tests/infra/test_ci_selection.py -q`: **57 passed**. CLI transcripts are in
  `.cache/engine-budget-demo/{twenty-percent-slowdown,changed-score-digest-speedup,slower-runner-within-budget,record}.txt`.

## Judgment and remaining implementation direction

The explicit failure messages fit the accepted purpose, but the green CI check
contradicts its own failing comparison. The demonstration **fails** the Task's
requirement that an over-budget candidate fail the PR. The earlier successful
run and 57 passing tests did not exercise propagation through the workflow's
pipeline; they do not rebut this observed defect.

Same-runner comparison is the useful reading of a committed budget given the
recorded CPU variation. Native random-legal throughput supplies a bounded
engine workload, with its difference from Python training throughput clearly
disclosed. Neither interpretation needs a redesign for this demo.

In addition, the design and published PR
promise candidate-artifact recovery when the reference no longer builds, but
no controlled reference-build failure has exercised that path. Source order
and `if: always()` are supporting evidence, not an observed failed-job recovery.
The passing comparison and the temporary baseline adoption above do not close
this gap.

Recommended next action: **iterate**, preserving the completed implementation.
First set an explicit Bash shell with pipefail for this workflow's run steps
(for example, job-level `defaults.run.shell: bash`) and add a controlled
workflow-level regression that executes the comparator-through-tee command
using the configured shell. The existing 20% slowdown must fail that command,
not merely a direct Python invocation. Also assert a failed benchmark cannot
be hidden by its tee pipeline. Re-publish and inspect the resulting PR check
and artifact. Do not widen the 10% budget or adopt this marginal measurement
to hide the shell defect; assess any remaining timing failure after enforcement
works. This direction is the agent's delegated review judgment.

Then add or run a bounded failure-path proof that records a candidate, fails only
the reference build, retains an adoptable candidate artifact, and successfully
runs the real `record --reason` command from it. Keep the original failure
visible; do not add load, relax comparison checks or move the production
baseline solely to manufacture this proof. Record exactly which part ran on
GitHub and which part used controlled fixtures.

Also correct the PR's wording that measurements are uploaded before building
the reference: they are **recorded** first and uploaded by the final always-run
step. This is a publication-description correction, not a workflow redesign.

The first main-push execution remains a post-merge check. It is not permission
to merge now. This demo does not settle the Task or restart a Graph. No HTML
review page or browser observation is claimed on this headless surface.

## Implementation response — 2026-11-02

The iterate direction is implemented locally: explicit Bash protects every
pipeline, and workflow edits select the Python regressions. Eight new cases
exercise the actual workflow run bodies, including a controlled reference
build exit 42 followed by adoption of the candidate through `record --reason`.
The upload path is copied locally; GitHub transport was not exercised in that
fixture. An implicit-shell copy reproduces failure in the slowdown and all
four benchmark regression cases. The working design records the focused
check result and remaining gate/publication/demo work. This response does not
replace the original run evidence or claim the repaired PR has run yet.

## Decision pass 2 — 2026-11-02

This review uses delegated headless judgment. Jack Heart supplied no new
observation or approval. His retained publication/demo boundary still excludes
shipping. The pass-1 failure remains valid historical evidence; this pass
evaluates the repair at published head `88dc8c1ad25cd99cca6026581a569f5c2f63ee1b`.

The intended progress was explicit Bash failure propagation, regressions through
the actual workflow commands, candidate recovery after a reference-build failure,
and corrected publication wording. The PR now says measurements are recorded
before the reference build and uploaded in the final always-run step.

Observed locally in pass 2:

- `uv run --extra dev pytest tests/infra/test_engine_budget.py
  tests/infra/test_ci_selection.py -q`: **65 passed in 2.20 s**. These include
  the 20% slowdown and faster-but-changed-score comparisons through `tee`,
  failures in each of the four benchmark pipelines, and candidate adoption
  after injected reference Cargo exit 42. The recovery proof executes the
  workflow's recording body and the real `record --reason` command. It copies
  the upload path locally; it does not prove hosted transport after failure.
- Replaying the original false-green run's downloaded artifact through the
  repaired comparison command and configured pipefail shell exits **1**,
  retaining exactly the throughput failure. No benchmark was rerun or timing
  changed. Transcript:
  `.cache/engine-budget-demo/pass-2/retained-failure.txt`; summary:
  `.cache/engine-budget-demo/pass-2/retained-failure-summary.md`.
- The repaired PR's Python Unit Tests job passed. Engine Budget run
  [41552162812](https://github.com/loopflowstudio/manabot/actions/runs/41552162812/job/125642805847)
  **failed correctly**, with comparison exit 1 under the logged shell
  `/usr/bin/bash --noprofile --norc -e -o pipefail {0}`. Candidate and reference
  builds, measurements, candidate recording and final artifact upload succeeded.
  Both measurement summaries are present. This directly closes the hosted
  false-green defect; it does not establish an in-budget candidate.

| Metric | Same-runner reference | Candidate | Result |
|---|---:|---:|---|
| Worst search reply | 13.913 s | 14.857 s | Pass |
| Total search | 411.956 s | 433.796 s | Pass |
| Step throughput | 6,774.5/s | 6,104.0/s | Fail: minimum 6,158.6/s |
| Search digests, step digest and counts | Committed values | Identical | Pass |

The candidate takes **10.98% more time per step** than this reference
(`6774.5 / 6104 - 1`), equivalent to 9.90% fewer steps/s. The comparator's
10% duration budget correctly uses reference throughput divided by 1.1.

The actual always-run upload produced
[artifact 13193795768](https://github.com/loopflowstudio/manabot/actions/runs/41552162812/artifacts/13193795768)
despite the comparison failure. `gh run download` retrieved it into
`.cache/engine-budget-demo/pass-2/pr-head/`. Its measurement names the exact
published head `88dc8c1ad25cd99cca6026581a569f5c2f63ee1b`, this run, and an AMD
EPYC 7763 runner. Replaying the downloaded comparison also exits 1. The real
`record --reason` command successfully adopted it into temporary
`.cache/engine-budget-demo/pass-2/adopted-baseline.json`, preserving all
measurement identity fields and tolerance 0.1. Transcript:
`.cache/engine-budget-demo/pass-2/record.txt`; hosted log:
`.cache/engine-budget-demo/pass-2/pr-head.log`.

This proves actual failed-job transport and adoption after a comparison
failure. Reference-build failure still has the bounded local proof described
above; no hosted reference-build failure was induced or claimed.

## Pass-2 judgment and next implementation

**Iterate, for a baseline reconciliation only.** The enforcement repair holds;
the published PR still fails its unchanged throughput budget. Pass 1's
false-green defect and absent recovery fixture are repaired. This is material
observed progress, not another pass with the same unresolved defect.

The committed reference `e5615edcc` predates merged main changes `230b2f563`
(decklist disclosure) and `a8abf01b1` (T5 token copies and effective battlefield
characteristics). `git diff a8abf01b1 HEAD -- managym/src` is empty. The diff
from the reference includes those upstream engine changes; the throughput
workload is unchanged, and CI uses the same benchmark sources on both sides.
This is an older engine reference, not evidence of runtime code introduced by
the budget PR. The source comparison does not isolate which upstream change
caused the measured cost. Two same-runner results now fail throughput near the
threshold; do not dismiss the current failure as established runner noise.

Under the instruction to stand in for Jack Heart, the agent accepts anchoring
this new budget to the already-merged engine and explicitly retaining this
measured cost. This is delegated demo judgment, **not a human statement or
shipping approval**. Reverting accepted upstream functionality or broadening
this Task into engine optimization would exceed the intended initial-budget
outcome. The 10% tolerance remains unchanged, and the earlier failed runs remain
evidence. No new design or workflow concept is required.

The next implementation should run the documented `record --reason` command
against this downloaded artifact to update the production baseline, with a
reason naming the synchronized main changes and the observed 10.98% per-step
cost (plus the run link). Update the current-baseline documentation and PR
description without rewriting historical T4/T5/T6 or full-score receipts.
Republish and inspect the resulting candidate/reference comparison. The exact
artifact above is sufficient; no separate measurement dispatch, engine change,
threshold relaxation or artificial load is needed. Production baseline adoption
was demonstrated only in a temporary file during this demo; the implementation
response below records the subsequent production adoption.

At review, Rust, Python unit, both conformance jobs, Academy integration/UI,
Clean-machine Play and long integration passed; main integration was still
running and Tests Result had not completed. The first main-push check remains
post-merge. No HTML page or human browser observation is claimed on this
headless surface. This demo closes no Session, restarts no Graph and authorizes
no merge.

## Baseline implementation response — 2026-11-02

The production baseline now adopts this run's exact candidate through
`record --reason`, explicitly accepting the merged engine's measured 10.98%
per-step cost and retaining tolerance 0.1. Artifact identity, every measurement,
and unchanged digests/counts were checked directly. Current-reference guidance
in `docs/development.md` now cites this run; historical receipts remain intact.
No runtime or workflow code changed. The next publication/demo owns updating
the PR copy and inspecting the new comparison; this response claims no new
hosted run or shipping approval.
