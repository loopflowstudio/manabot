# MTG-153 — evaluation anchors

Status (2026-10-19): implementation exists; scientific acceptance remains open.
Jack Heart approved the six-checkpoint MTG-137 cohort, heuristic and frozen Mini
10k anchors, three anchor pairings, original held-out deal families, targeting
records and an anchor recommendation on 2026-10-19. Mini only; $0; eight hours;
one scientific attempt; stop at first failure; no relaunch without Jack Heart's
approval. Historical producer artifacts and both code anchors remain unchanged.

Implementation choice (2026-10-19): Mini seed 12261 at 10k is the frozen anchor,
selected by seed before comparison. The numeric predictions and provisional
recommendation rule in `experiments/evaluation-anchors.md` remain draft until
committed before the full attempt. The rule is implemented, but adoption of a
daily regression anchor remains a review decision informed by measured headroom,
uncertainty and ordering; it is not an already accepted strength conclusion.

## What exists

Digest-bound checkpoint opponents preserve default MonitorProtocol serialization.
Admission binds the six retained checkpoint identities, historical greedy rows,
original deal families, registrations, source and native binary. The 15-pair
runner retains per-game traces and Study targeting, stops on invalid games,
claims an exclusive scientific-attempt marker and enforces a process-group
deadline. The marker prevents reuse of a prepared cohort; a fresh preparation
does not authorize a second scientific attempt.

Analysis requires complete layouts and targeting artifacts. It reports LvA,
AvL, LvL, AvA, whole-deal bootstrap intervals, paired greedy differences,
init/10k/25k group-mean ordering, seed ranges/spread, headroom, anchor strengths
and targeting fractions. Each checkpoint uses its original MTG-137 held-out
family; anchor-only pairs use Mini 12261's family. Self-play is a noisy sanity
check, not an exact 0.5 gate. Three Mini and two GPU seeds do not establish
method-level ordering; opponent-specific nontransitivity remains a confound.

The simplification is complete: first-failure policy belongs to the cohort,
which calls the arena with one selected leg; analysis has no partial-layout
mode. Frozen members/pairings resolve by name, admitted heuristic manifests are
reused, and targeting denominators are computed once per label/seat/cell.

## Evidence and limits

The prior focused suite recorded 26 passes; the subsequent runner failure and
targeting suite recorded five passes. These results are reused, not rerun for
reconciliation. The retained smoke completed 15/15 replay-valid games with no
failures, 15 Study records and 15 targeting artifacts in 28.31 seconds. Its
cohort digest is `e73c685ae47c076737ae02aacae264e01c802ec6dd8bfd94ec0fd42817635117`.
Reconciliation confirmed that binding and the native digest, but the prepared
source digest differs from current code after simplification. This smoke is
historical engineering evidence, not a current-source smoke or scientific result.
No scientific-attempt marker exists under `.runs/evaluation-anchors`.

The checkout and cached origin/main both point to MTG-152's heuristic anchor
commit `12fc7c6f`; no later upstream change is visible in those refs. The live
Intelligence Project retains full-game improvement and repeated-training evidence
requirements. Current `.lf/wave/intelligence/GOAL.md` and `MEMORY.md` are absent,
so current shared lessons and child memory coverage are unavailable. September's
archived Wave documents are historical context, not substitutes; no Wave files
were created or curated.

## Remaining work

Gate's affected suites remain pending. Fresh admission and a current-source
one-game-per-pairing smoke remain before scientific execution, using new output
paths because admission and smoke directories are never overwritten. The draft
protocol/predictions must be committed before the authorized single 3,000-game
attempt. The complete measured report and review of its provisional anchor
recommendation remain outstanding. Missing behavior has not reduced acceptance.

Check: `git diff --check` — passed (2026-10-19).

## Publication check (2026-10-19)

Publication withheld: MTG-153 acceptance requires the measured 3,000-game cohort
and an anchor recommendation. Neither exists; the only retained attempt status
is the historical 15-game smoke, and no scientific-attempt marker exists. Both
retained cohort admissions differ from current source. No PR exists for this
branch; HEAD equals the recorded base and cached origin/main at `12fc7c6f`.
The implementation remains uncommitted and preserved. The pr-publish step did
not start the scientific attempt, publish, arm auto-merge or complete the Task.

Current-source checks: `uv run pytest tests/experiments/test_evaluation_anchors.py
tests/training/test_monitor_evaluation.py tests/arena/test_match.py -q` — 37
passed in 63.85s. Ruff on the seven affected Python files and
`git diff --check` passed. These checks establish engineering behavior, not
the scientific acceptance outcome. Fresh admission, current-source smoke,
committed protocol/predictions and the authorized single scientific run with
its complete report remain necessary before publication.

## Corroborating evidence (MTG-135 protocol017 heuristic panel, 2026-10-19)
- 9,152 games, 22 recipes x 4 checkpoints, 13 deals x 8 legs, main 12fc7c6f, RunPod CPU.
- 524k endpoint vs heuristic 0.28-0.32 (384x8@64 streams: 0.12) vs 0.49-0.56 vs greedy; init 0.04 vs 0.27.
- Per cell at 524k (heuristic vs greedy): LvA 0.15/0.23, AvL 0.53/0.82, LvL 0.23/0.61, AvA 0.29/0.44.
- Paired heuristic-greedy gap nearly constant (-0.18 to -0.29): heuristic preserves greedy's ordering; it adds headroom, not new separation.
- Data: /Users/jack/src/managym-cpu-research/.runs/grid-007/learning-017-heuristic-cpu/analysis/
