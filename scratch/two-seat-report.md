# MTG-180 — graphical experiment report

2026-10-23 · **Revised draft implementation design for review.** Jack Heart's requested
outcomes in MTG-180 and `report-history-and-requirements.md` are binding. The
Manabot/Etude ownership split below is a recommendation, not a recorded approval
from Jack Heart. Upstream edits, deployment and merge remain outside this design.
Base inspected: `95ca2059ea443351cabd4e8c322991c36c0d8212`; preserve the existing
uncommitted numeric table-sort fix in `manabot/remote/live_evaluation.py`.

**Accepted API direction — Jack Heart, 2026-10-23:** the top-level report API is
through the experiment, with the report defined in Python. No author-written
manifest or configuration file. The proposed module/function names below remain
draft; this correction does not approve the entire ownership or implementation
proposal.

## Problem and demo

The two-seat observer computes Etude curves but draws only tables. Digest names
have no chronological meaning. A reader cannot compare heuristic strength,
training effort and direct contests through one consistent selection. The
retained prototype demonstrates the desired interactions, but duplicates chart
code and timing arithmetic and is not the implementation boundary.

**Authoring and demo:** the experiment's Python module defines its report beside
its recipes and analysis. The proposed public use is:

```python
from experiments import two_seat_ladder

page = two_seat_ladder.report()
```

This is the proposed shape, not an implemented API. `report()` selects the
experiment's recorded runs, control and evaluations and composes the sections in
Python. Authors edit that code to choose comparisons, panels, clocks and defaults.
Ordinary Python arguments may override a source or selection when useful; the
normal call requires no inventory of files. A new `Experiment` class is not
required merely to provide this entry point.

Etude's existing page rendering/export functions remain lower-level tools for
opening, regenerating and sharing the result as self-contained HTML. Rendering
never rewrites the experiment's Python definition. The CLI invokes that same
definition. Dependency extras are installation details, not report choices.

**Accepted CLI direction — Jack Heart, 2026-10-23:** `etude report EXPERIMENT`,
with the experiment as a required positional argument. No `reports write`
subcommand. Generation is the command's default action:

```sh
uv run etude report EXPERIMENT --open
uv run etude report EXPERIMENT --watch 5 --open
uv run etude report EXPERIMENT --out reports/2026-10-23
```

The first writes once, the second updates while running, and the third writes
once to a chosen historical destination. Each invokes the experiment's Python
report definition. Omitting the experiment produces usage help/error rather than
generating reports for every experiment. This accepted command shape is not yet
implemented; current Etude uses `reports write` with optional `--experiment`.
The optional flags above retain their current meanings as the proposed migration.
Resolving an
experiment name to its Python definition remains part of the API design.

Opening the page
shows two-seat and sparse one-seat heuristic points overlaid, aggregate plus
LvA/AvL/LvL/AvA panels, then direct contests and training performance. Selecting
Lessons/on-draw changes the curves, intervals, coverage and evidence rows together.
Changing clocks preserves the selection; regeneration/reload preserves it in the
URL. An unavailable baseline split is visibly absent. No game, checkpoint loader,
remote request or training process runs to produce this report.

The scaling cohort uses the same renderer with its own protocol selection:
additional-decision milestones beside collection time, elapsed time and recorded
cost. Its frozen initialization, seeds, actual milestone overshoot and missing
results stay visible. It is never silently combined with the older ladder.

## Inspected architecture and consequential findings

0. **Prior Python-report direction exists.** Etude Studio's decision 137
   (2026-10-13, `docs/design/decisions.md`) accepts “The report as an editable
   Python file, as manabot's notebook is.” Its historical
   `eef111d:docs/design/metrics-store.md`, “Reading and the page” and “Remaining
   work”, distinguishes the built generated page from the still-unimplemented
   editable Python definition. Current SDK source has experiment-grouped
   `pages(source, experiment)`, `page_data`, `render` and `write`, not a completed
   authorable experiment/report object API. Current Manabot `experiments/README.md`
   already puts experiment-specific recipes, execution and reporting in one
   Python module, with reusable instruments in `manabot/`. These are the starting
   points; the historical proposal is not evidence that an API already ships.
1. `live_evaluation.report` reads `track/` through `etude.eval.query.connect`,
   writes `curves.json`/`status.json`, and embeds Etude's table report. Its ledger
   owns evaluation attempts and observations, not training checkpoint declarations.
   `head_to_head.report` separately reads `DuelResult` and renders tables. Neither
   needs a new observer or evaluator to present the retained data.
2. Pinned and installed **etude-sdk 0.2.0** exports
   `etude.train.report.page_data/pages/render/write`, the vendored offline uPlot
   renderer, and the evaluation query/curve APIs. It **does not** export
   `etude.reports`. This corrects the inherited mana note's claim that the
   installed version also lacks `pages/write`; that note records an older probe.
   `reports` installs the S3 extra, while `evaluation` installs DuckDB; report
   commands using evaluation records need both. No dependency was changed here.
3. Inspected Etude SDK source at `5c18aca9e189cdcf2077269e6066585d1431dcba`
   (`src/etude/reports.py`, `train/report.py`, `train/report.html`) contains
   `Evaluation`, `record`, `read` and evaluation plots. `read` checks the exact
   training SourceKey, recorded checkpoint and evaluation subject. `record`
   describes a **prospective** request; retrospective report assembly must not
   claim that an old evaluation was requested before execution.
4. That source renderer still groups evaluation plots by training run, does not
   provide the requested domain facets/shared filtered evidence table, and
   persists clock/mode/hidden runs/live refresh but not the full requested view.
   Merely bumping the dependency will not deliver the report. Generic overlay,
   facet, linked zoom, range and table-selection support is proposed upstream
   work, not an available API.
5. `etude.eval.tournament.curves` already uses recorded checkpoint ordinals and
   natural names. Its `training_checkpoints` join chooses the first run/ordinal
   for a duplicated digest. This is inadequate for shared warmstarts and
   continuations. Exact run/checkpoint/evaluation bindings must select the
   coordinate; digest-only matching cannot identify which arm owns a point.
   Preserve the ordinary sorter; fix input binding, not hash-name sorting.
6. `etude_evaluation.arena_tables` validates all eight legs and actual seat decks,
   then writes both participants' rows. Candidate is Etude player 0; native
   on-play is separately `on_play`. Candidate-relative deck comes from actual
   seat assignment, not leg pairs. Aggregates must select the candidate once.
   Scores are win + half draw; the older dashboard's win-rate plots have a
   different estimand and cannot be reused unchanged.
7. `time_profile.training_profiles` and `time_cohort.native_cohort` already
   partition attempt-local time, reject overlapping totals, exclude inherited
   completed stages and avoid summing recovery progress endpoints. Reuse them.
   Prototype `build.py` instead manually subtracts phase counters, defaults
   missing RSS to zero and assigns some baseline clock endpoints by convention.
   Those shortcuts are not an admissible production data path.
8. Retained prototype snapshot has nine checkpoints, raw candidate rows with
   deal/leg/seat/matchup/score plus CPU game duration and decisions, 229 source
   references, native training data, baseline run and direct results. Raw rows
   permit whole-deal filtered estimates. Summary-only baseline/direct records
   cannot invent seat splits or recompute arbitrary subgroup intervals.
9. Scaling protocol/source remains at
   `/Users/jack/src/manabot.both-seats/.runs/two-seat-scaling-25/` (read-only).
   It declares seeds 165251–165253, nominal 2M/4M/8M additional-decision milestones,
   common one-seat update-1000 initialization and fresh Adam in both arms. The
   follower sets `training_seconds` to collection + learning; that field is
   active time here, not total elapsed. Preserve nominal and actual coordinates.
   No completed scaling result set was supplied in this checkout; protocol and
   launch files establish intended inputs, not current results or spend.

## Chosen approach and ownership

**Proposed split:** Etude owns the generic offline plotting/selection engine and
explicit checkpoint-to-evaluation joins. Manabot owns candidate-relative matchup
semantics, protocol comparability, whole-deal score projections, native timing,
cost meaning and reusable projections. The experiment's own Python module owns
the report composition and choice of comparisons. No Academy session, trace, table
protocol, selected-world loader or identity store changes are needed: this is a
retained-evidence report, not play/checkpoint admission. It makes no claims about
human completion, trace crash durability or policy-free game reconstruction.

The experiment's `report()` function reads its recorded experiment/run identities
through Etude and the existing Manabot readers. Baselines, arm labels, panel
selection and clock defaults are Python values in that definition. Readers may
accept typed Python values for alternate sources and test fixtures; do not add
a serialized `ReportInputs` schema or ask the author to maintain a second list
of files. Existing artifact receipts and identity records remain internal
evidence, not report-authoring configuration. Verify available source bytes
before projection. Do not
require checkpoint weights just to graph retained scores; distinguish retained
checkpoint references from verified local checkpoint-byte possession. Reuse
`report_snapshot` for native VerifyStore export when that is the input, preserving
its capture-window semantics. Never treat its separately copied files as one
transaction. Snapshot directories and projections are reproducible build inputs
and outputs, not another live authority.

For retained Etude declarations, query a local snapshot containing the original
training and evaluation records with their names/SourceKeys intact. Reject
conflicting source keys or checkpoint coordinates. Do not call `track()` to
manufacture historical training runs, copy evaluation scores into training
metrics or write links into a running source tree. Extend the Etude join to
accept explicit **retrospective bindings** for read-only report composition,
including provenance and binding time distinct from request/result times. Reuse
the same validation as `Evaluation/read`. This addition and the generic renderer
extensions require a reviewed, reproducibly installable SDK revision/release.

Where the original checkpoint declaration is absent, show the ledger/admission
coordinates as retained Manabot evidence with an explicit unlinked status; do
not fabricate a SourceKey, ordinal or prospective request. Such points may be
presented on their recorded numeric axis, but cannot claim the Etude join passed.
Reject ambiguous same-digest ownership instead of accepting Etude's first-run
choice. Missing time coordinates suppress points on that axis, without removing
their evidence rows. Axis labels distinguish total versus additional learner
decisions and recorded training, active, collection and elapsed seconds.

Produce typed point records containing cohort, arm, training run/seed, checkpoint
digest, binding provenance, protocol/opponent identities, candidate matchup and
seat, coordinate values/bases, W/D/L, score/interval method, completed/scheduled
deals/games and attempt status. Renderer-facing facets are generic named values;
Etude contains no Allies/Lessons constants. Native timing stays a typed projection
of `TimeCohort`; the scaling experiment's custom endpoint/metric records need an
explicit typed reader, not a counterfeit `TrainingRun` with invented fields.

## Presentation and evidence rules

- First section: heuristic score by arm, aggregate and four small multiples.
  An explicit comparison group checks opponent/version, world, inference and
  protocol/deal compatibility from retained records. Different benchmark hashes
  may reflect declared historical differences; expose them rather than rewriting
  them or asserting a causal comparison. Incompatible evidence gets separate
  panels. Never regenerate a frozen fingerprint with the current Interactive
  mirror helper to make records match.
- Shared run/control, matchup, candidate deck, play/draw, checkpoint range and
  metric-search selection feeds both charts and evidence rows. Matchup/deck
  choices intersect; an empty intersection says no retained evidence. Use stable
  identity-based colors, not array positions that shift when runs arrive. Store
  selections and linked zoom in URL state, with accessible labels and keyboard
  controls. An unrecognized stale selection is reported/reset visibly.
- Recompute filtered raw-row estimates server-side over whole deal blocks using
  the declared estimator. Precompute the finite domain selections so the browser
  does no resampling. Keep original summary-only intervals and method metadata;
  do not average subgroup bounds. Coverage uses the retained schedule, never a
  hard-coded 25 denominator. Partial/failed attempts remain visible, separate
  from completed points, including failed retries and missing requested points.
- Sparse baseline is independent of the candidate checkpoint grid. Show observed
  baseline points, never interpolate onto candidate milestones, extrapolate or
  substitute a 50% heuristic line. Missing scheduled points break lines;
  summary-only baseline filters with no split say unavailable.
- Direct head-to-head is separate, with a 50% reference, W/D/L, uncertainty,
  both checkpoint identities and both arms' real data/time/cost coordinates.
  Read retained raw rows when available for filtering; otherwise show supported
  groups and explain absent seat evidence. Do not rerun comparisons for controls.
- Training performance shows collection, learning, export, diagnostics and
  unattributed time, per-update trends and recorded throughput/resource measures.
  Keep parent and child phase measurements from being summed twice. Missing GPU
  utilization/RSS/timing is unavailable, not zero. Per-matchup CPU evaluation
  time/length/decisions is labeled separately; never allocate aggregate GPU time
  across matchups. Scaling collection-probe rates remain separate from end-to-end
  throughput and strength. Preserve individual seeds and paired milestones;
  within-deal intervals are not training-seed uncertainty.
- Freshness displays report generation, source observation/heartbeat and last
  completed evaluation separately. Regeneration cannot refresh source evidence.
  Paused/failed/uncertain continuation remains prominent. Reserved budget,
  incurred estimate and settled charges retain distinct labels and provenance;
  neither a ceiling nor a historical launch quote is an incurred cost. Missing
  arm cost allocation stays unknown; never split a shared rental evenly.

## Alternatives and failure boundaries

Keeping the prototype wholesale would deliver interaction quickly but create a
second renderer/accounting owner. Reject it. Reusing only the existing static
learning dashboard cannot meet shared filters or the score estimand. Reject it.
Etude 0.2.0 alone supplies useful curves but cannot meet exact-link and interaction
requirements; do not disguise that gap with an import fallback or copied template.
The recommended upstream extension plus experiment-owned Python composition retains one owner
per concern. Review of this boundary and availability of an installable SDK are
explicit prerequisites for the complete implementation, not reasons to start
another Flow or edit another checkout during kickoff.

Success means a reader can explain a strength difference and the actual effort
behind it, then inspect the exact contributing points after changing a filter.
Failure would be a beautiful graph that joins the wrong warmstart, turns missing
baseline/time into invented observations, or equates active time with wall cost.
Identity, missing-data and accounting tests therefore precede visual polish.

## Integration and deletion

**This slice — experiment-defined report:** one complete path from the
experiment's Python `report()` function through retained run/evaluation reads,
validated coordinate/score/time projections and the shared renderer. Add the
definition beside the experiment (proposed `experiments/two_seat_ladder.py`),
using reusable domain helpers in `manabot/training/`. The scaling comparison
defines its own report with the same helpers. Exact helper names and the smallest
Etude composition API remain implementation design work; do not create a generic
report framework or a `two_seat_report --manifest` CLI. Test the Python authoring
path itself, with fixture and headless browser checks.

Sequence: confirm the ownership proposal; supply/test the SDK join and generic
selection/composition contract starting from `pages/page_data/render/write`;
admit the locked dependency in isolation; implement the domain projection and
experiment's Python definition; cut both existing report functions over to that
composition without changing follower execution; prove the full offline path.
Existing observers call the same experiment-defined composition; they do not
carry a separate layout or invoke a manifest-based subprocess. For existing
report signatures, absent optional baseline/performance inputs yield
labeled unavailable sections. Keep status/curves downloads and old report URLs
usable. Rendering failures must preserve the previous complete page and report
the failure through the existing status path; they cannot stop the evaluator.

Delete — do not maintain: replace hand-built HTML/iframe composition inside
`live_evaluation.report` and `head_to_head.report` once the common output covers
their evidence. Remove the temporary ledger-sort implementation when exact
coordinate projection owns ordering; retain its chronological behavior and
regression test. Do not promote prototype `build.py`, `template.html` or copied
vendor assets into a maintained Manabot renderer. Preserve the prototype as
dated review evidence. Existing `learning_dashboard`, `time_profile`,
`time_cohort`, frozen results and other report consumers remain supported.

Forbidden: parallel chart engines, new controller/follower, monkey-patched SDK
internals, fabricated declarations/clocks, dual writes to scientific stores,
current-world fingerprint substitutions, or a green test run that skips the
required report capabilities. No new experiment, evaluation, rental, Mini service
change or report redeployment is in this design step. Rendering is local and
network-free, with atomic output replacement. Inputs are read-only; a failed
identity/hash check must stop that cohort rather than fall back to best effort.

## Done when / headless gate

Add a small deterministic fixture with exact training declarations, reversed
digest names, shared initialization in two arms, a missing baseline milestone,
failed/partial attempts, all eight legs, continuation timing and stale heartbeat.
The fixture is supplied through Python to the experiment's report definition,
without an author-written manifest/config file. The same fixture crosses source reads, joins, domain projections, rendered HTML
and browser controls. Include a separate scaling fixture with nominal/actual
additional-decision coordinates and active-versus-elapsed time.

Proposed acceptance commands (new tests do not exist yet):

```sh
uv run --isolated --extra dev --extra artifacts --extra reports --extra evaluation pytest tests/training/test_two_seat_report.py tests/training/test_etude_sdk.py tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py tests/training/test_time_profile.py tests/training/test_time_cohort.py -q
(cd academy/ui && npx playwright test --config e2e/report.playwright.config.ts)
```

The Python fixture calls the experiment's `report()` and Etude's export function
to generate the browser artifact. Add the dedicated Playwright config/test to
the existing toolchain, serving only that static report on an isolated loopback
port. Also verify that editing the Python definition changes the chosen panels
and that regeneration preserves that definition. It requires no Vite,
Svelte regeneration, display session, production host or credentials. Gate runs
once after generation. Expected: exact chart/table values under every domain
filter; correct 250→500→… order despite reversed hashes; sparse baseline/no false
50% line; identical URL state after reload; stable colors; linked zoom; keyboard
and narrow-viewport operation; no console errors or external network requests.
Python checks assert original input hashes unchanged, ambiguous joins rejected,
no hidden games/launch calls, whole-deal uncertainty, truthful missing states,
non-overlapping continuation accounting and three distinct freshness timestamps.

Retained real-data regeneration validates source compatibility separately from
synthetic fixture plumbing. The prototype and scaling protocol are available;
full scaling result capture remains an input gap, not permission to fetch via or
restart the running follower. No real checkpoint admission or human completion
claim follows from these report tests. Upstream SDK contract tests and its
released/pinned revision must be attached to delivery evidence.

Check: `uv run --extra dev --extra artifacts --extra reports --extra evaluation pytest tests/training/test_etude_sdk.py tests/training/test_etude_evaluation.py tests/training/test_time_cohort.py tests/training/test_time_profile.py -q` — **39 passed in 7.60s**; the initial invocation without `evaluation` had two missing-DuckDB failures. These validate existing APIs/accounting, not the proposed report or upstream extensions.
