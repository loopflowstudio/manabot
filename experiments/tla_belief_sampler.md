# TLA frozen-player belief sampler (MTG-357)

Exploratory. Collection and fitting have not started. There is no trained sampler,
held-out comparison or public prediction example yet. The experiment CLI described
in [the guide](../docs/tla-belief-sampler.md) supports saved-artifact `demo`;
remote admission and main dispatch remain to be implemented.

## Producer verification, 2026-11-14

The ordinary producer records establish that the staged Small checkpoint is the
raw endpoint of Etude run `59718fe34945`, not the experimental adoption used in
MTG-349. The retained R2 keys are relative to `r2://manabot/`:

| Record | Key | SHA-256 |
| --- | --- | --- |
| Endpoint checkpoint record | `jobs/59718fe34945/checkpoints/learner_transitions-82370334.json` | `7f8659083542eb36ea9bc181aef309466395300fb7669f2958011c642dc16f18` |
| Ordinary producer export | `jobs/59718fe34945/attempts/10d3740cb3a7/training/run.json` | `57910103734db654370fcbd36f766b4351b52fdadcc1464d77f82fa3fa34dc70` |

The checkpoint record names ordinal 80, 1,952 updates, 82,370,334 learner
transitions and tag `regime:tla-frontier-c5fba4-w64`. The producer export's
`self-play` stage names the same updates/transitions and raw artifact; its
`selected_artifact` agrees. All three point to 3,945,456 checkpoint bytes with
SHA-256 `c95c61aa298528e5c2782430d5e3f90929d3e9b7ac8b1fc45f560ed6ab4bbaf8`.
The staged `probe/data/final.pt` matches that identity.

The producer reports source `6cd54d349ccbbaf1f5f2aa10831ccfb3ecf4653b`, whose
tree `f2a131012ea8319b9cd00ac11d43f9d30b62ce04` equals the reported training
revision `ab2b757d9`. Preserve the original absence of `checkpoint_source`.
The producer export reports `paused` at its wall bound; the historical launcher's
`succeeded` outcome does not mean that the original update horizon was reached.

An isolated archive of `ab2b757d9` built with its locked CPython 3.12/Torch 2.10
runtime on the mini. The original `manabot.model.checkpoint.load_checkpoint`
loaded those unchanged bytes. The retained observation configuration is input
v9, card capacity 128 per player, permanent capacity 96, 512 facts, 32 events,
64 actions, `first-legal`, and two focus objects. Staged maintained capture
modules imported in that runtime. This is a loader/import proof; it supplies no
TLA game, sampling-validity or throughput evidence. The complete staged bundle
and native manifest still need durable admission receipts before collection.

## Current implementation

The public frozen-policy collector delegates to one scheduled capture core.
It preserves strict public-call behavior and supports the configured native
action prefix without remapping offer IDs. Scheduled capture supports a fixed
public schema, whole-game splits, bounded label-independent decision selection,
explicit unsupported-row coverage, per-game private shards and attempt receipts.
Continuation does not retry started slots; saved completed evidence is verified
before reuse. The sampler optimizer and frozen policy loader are reused;
`fit_saved_sampler` adds the attempt/reload boundary described below.

`sampling_evidence.EvidenceRoot` retains pre-dispatch Etude allocations and
requires fresh settlement for every declared run, including no remaining writer
and a known launcher amount. It applies the USD25 cumulative ceiling and USD2
admission subcap without counting a receipt twice. Setup belongs to the phase
it prepares. One root lock covers reservation through submission and closure
mutations.

Collection closure retains every scheduled status, private source hashes and
launcher receipt references. Orphan shards remain interrupted; completed games
with no representable rows remain completed. Only closed sources produce the
single immutable main dataset. Source additions, changes and lost bytes reject
later fits/reports. Admission stays separate. These guarantees currently have
injected-fixture validation, not a connected launcher adapter. Allocations bind
exact schedule slots or the frozen dataset/configuration/fit seed. A lost
submission response leaves that work consumed. Continuation uses fresh launcher
observations to distinguish live/unknown work from ended unavailable delivery;
local output absence cannot authorize another attempt.

The runtime classifies native panics and the known `no legal targets` failure
at active steps; unknown native, contract, reset and capacity failures remain
fatal. Private stops retain their cause and delivered Command prefix. Continuing
untouched slots requires explicit restored-prerequisite evidence after job
settlement. Coverage retains per-viewer saved rows and distinct retained-selection
and foretell reasons (overlapping reasons are not disjoint row totals).

`fit_saved_sampler` retains started/completed/stopped attempts and refuses an
interrupted seed, including an orphan candidate. Verified complete candidates
reload without optimization. Cohort reports resample declared game groups,
retain available legs/viewers together, and expose missing-leg counts separately
from fit-seed uncertainty. TLA same-field/reserved panels derive membership and
deal groups from the closed schedule, including zero-row and unattempted games.
The offline `demo` verifies seed357 and frozen sources, retains digest-bound
16/64-sample reports and a public JSON/Markdown prediction example. It performs
no collection or optimization. Its history display is the saved normalized
commitment summary; full ordered public traces are not retained in sampler rows.

Validation on the mini uses an isolated copy of this checkout and the existing
CPython 3.13 native runtime for current-world collector tests. These controlled
and tiny-game checks are software validation, separate from experiment admission.
The historical-runtime import proof above does not establish an end-to-end
historical-runtime capture round trip. The changed test file passed 12 checks;
a subsequent explicit repeated-deal-seed fix passed its new reset check and all
three affected real-collector cases. Ruff and whitespace checks passed.

## Execution dependency

The available Etude 0.3.2 and original 0.2.15 clients both refused the mini's
shared launcher database while reading machine inventory:

> `/Users/jack/.etude/launcher.db was written by another version of etude; move or delete it to start again`

The shared database was preserved. No experiment job, rental, admission game or
sampler fit was launched. No launcher cost receipt exists for this experiment.
A compatible read of the existing launcher/controller inventory is required to
establish available shared capacity and launch under the cumulative USD25 cap.
The shell process snapshot alone is not a launcher inventory.

Next: provide that compatible launcher path, finish the frozen worker bundle and
experiment CLI, then run the separately accounted eight-slot admission under
USD2. Freeze cohort size and fit allocation from its ordinary pace before main
collection. The proposed 1,024-slot schedule and three fit seeds remain planning
choices, not an executed freeze or permission to ignore missing attempts.

The next implementation connects the existing exact-work owner and worker
adapter to compatible Etude inventory, pre-submit lookup and artifact delivery.
`EvidenceRoot` coordinates invocations on one authoritative filesystem directory;
its lock cannot coordinate copied remote roots. Worker staging must execute only
its retained intent, reject controller operations and return verified private
outputs without merging its ledger or closure as authority. The local authority,
snapshot and output-admission checks are implemented and validated below;
connected transport remains unimplemented. A connected lookup/delivery reconciliation
proof is required before original-runtime admission; no additional experiment was
launched during realignment.

## Lifecycle implementation checks, 2026-11-14

On the mini, injected and ordinary tiny current-world fixtures in
`tests/belief/test_sampling_data.py` and `test_sampling_report.py` passed
30 checks in 2.22 seconds (final public-history fixture), including fatal-prefix/restoration, unknown/live job
rejection, cumulative costs, explicit closure, orphan exclusion, missing splits,
grouped uncertainty, both-viewer capture and a label-free saved demo. The focused
saved-fit interruption and completed-candidate reuse/corruption checks each
passed (one check each). These establish software behavior, not experimental
admission or a fitted TLA result.

A read-only inspection found the mini's shared launcher database has
`PRAGMA user_version=0`, a `workers` table and attempt references to `worker`;
its controller heartbeat is dated 2026-10-13. Available 0.2.15/0.3.2 clients
use a newer `sessions` schema. None of the 27 cached SDK database modules found
on the mini declares the old `workers` table. This narrows the compatibility
problem but supplies neither a compatible controller inventory nor proof that
shared capacity is idle. The database and controller were left unchanged.

The final saved-demo text/seed guard passed its focused check (1 passed, 1.32s).
The experiment-discovery test module could not collect in the retained worker
environment: its Etude SDK lacks `etude.eval.records.Seating`, imported by
`experiments/elemental.py`. Running that test's exact inventory assertion directly,
without importing the unrelated experiment modules, passed. This does not count
as a passing run of the whole discovery module; gate still needs the matched SDK.
Ruff check/format and `git diff --check` passed for the changed files.

The public example selects the first retained test row with nonempty public
history, without selecting on scores or predictions. Absence of such a row makes
the history example unavailable. The final 30-check owner run includes nonempty
public commitment features and validates the label/seed exclusion.

## Controller continuation implementation, 2026-11-14

The evidence owner now binds each allocation to exact schedule ordinals or a
frozen dataset/configuration/fit seed. Reservation and submission share one root
lock. A lost submission response leaves its work consumed; different launcher
lookup keys cannot authorize overlapping slots or repeat a fit seed. Fresh
launcher observations retain execution state, queued/retry writer closure,
cost and delivered byte inventories. Continuation distinguishes unknown/running
work from ended unavailable delivery and derives untouched work without inferring
interruption from controller exit. The frozen worker adapter executes the exact
reserved subset using the maintained collector/fitter. No live launcher adapter
or new remote command is claimed.

TLA reporting joins complete saved reports to the closed schedule's test panels
and groups by declared deal seed. Complete zero-row and unattempted games remain
in coverage even when an entire panel has no measurable predictions. Optional
`demo --cohort` retains this analysis alongside the public example; the cohort
contains saved report references, never hand-authored launcher settlements.

Mini validation: the two affected capture/report owner files passed 35 checks
in 2.36 seconds, including lost acknowledgement, competing controllers, queued
retry closure rejection, unavailable fit delivery, exact worker subsets and
schedule-derived panels. The subsequent optional panel-demo serialization check passed (1 check,
1.42s). After making launcher execution state mandatory and bounding directory
inventory reads to one scan, the six affected dispatch/worker checks passed
(0.83s). Ruff, Python compilation and `git diff --check` passed. These controlled checks establish software behavior, not TLA
admission, optimization or a scored comparison.

A fresh read-only `etude machines` using the retained original environment still
returned the same shared-database version refusal on 2026-11-14. Its public
`queue`/`launch` functions delegate to a selected controller and return a string;
that alone establishes no stable pre-submit lookup contract with the existing
controller. The database was preserved, no alternate controller or job was
created, and capacity/cost remain unknown. Connected admission still requires
a compatible inventory and verified pre-submit lookup path. No new experiment
cost receipt or result exists.

### Controller snapshots and returned evidence (2026-11-14)

The local authority boundary is implemented. The authoritative directory binds
host, path and filesystem identity; copied roots cannot mutate controller state.
Private worker snapshots export one retained job and immutable inputs, without
a ledger or closure authority. Returned output admission uses fresh launcher
observations, excludes live/queued/retry writers, verifies allowed paths and
bytes plus schedule/dataset/configuration receipts, and retains partial/fatal
evidence without manufacturing completion. Controller restoration, closure and
new dispatch require admission of known worker output.

Controlled validation: the nine selected authority, worker, delivery, dispatch
and fit-input checks in `tests/belief/test_sampling_data.py` passed in 0.56s
(18 deselected). The subsequent closure/restoration regression selection passed
4 checks (23 deselected, 0.49s), including freeze after admitted worker delivery.
Ruff, compilation and whitespace checks passed. The capture
runtime is an injected fixture and the fit checkpoint is synthetic delivery
bytes; no game collection, optimization or candidate loader admission occurred.

A fresh read-only `etude machines` on the mini again returned the same database
version refusal. Inspecting its installed SDK found `Scheduler.prepare_launch`
retaining `Launch` requests before source publication, `submit_launch` claiming
that intent, and store `launch_evidence`/`launches(run)` lookup. The remote
`controller.prepare` calls this path. This is a candidate integration point,
not a verified lost-response contract against the existing controller; a lost
prepare acknowledgement also needs reconciliation. Compatible inventory and a
connected lookup/delivery transcript remain missing. The shared database and
controller were preserved; no experiment job or new cost receipt was created.

The remaining handoff also includes preparation: `JobAllocation.run` must exist
before `dispatch`, whereas `prepare_launch` returns that reference. Existing
lost-submit fixtures start after preparation and do not cover a lost prepare
reply. Before connecting dispatch, verify a controller-supported request identity
that can recover the same preparation after a lost reply, and establish whether
preparation can enqueue work, acquire resources or start a writer. Retain that
request's exact work and whole-window cap before the first launcher mutation;
unresolved preparation blocks replacement and further dispatch. This is an
unimplemented adapter requirement, not evidence that the inspected SDK provides
recoverable preparation. No tests, games or fits were rerun during realignment.

### Preparation API inspection (2026-11-14, recorded 14:05 UTC)

The current MTG-349 environment on the mini identifies its installed distribution
as `etude-sdk 0.2.16`; this is separate from the earlier 0.2.15/0.3.2 checks.
The read-only inventory command was:

```sh
ssh -o BatchMode=yes mini-heart.local '/Users/jack/.local/bin/uv run --no-sync --project /Users/jack/src/manabot.mtg-349-score etude machines'
```

It exited 1 with the same database-version refusal quoted above. Capacity,
existing work and launcher costs remain unknown. No preparation or submission
was attempted, and the database/controller were preserved.

Inspection of that environment's `etude/launch/` sources narrows the handoff:

- `Scheduler.prepare_launch` preserves an explicit `Run.id`, otherwise assigning
  one, then generates a separate launch ID. It validates the jobs, writes
  `Database.begin_launch` and retains a diagnostic before returning. The inspected
  body does not freeze/publish source, admit runs or acquire a machine.
- `Database.begin_launch` rejects an admitted run or a previous preparing/admitting
  launch for the same run, including unresolved acquired sessions. Its
  `launches(run)` lookup can recover launch records by the supplied run identity.
  These source paths suggest a recovery key; they do not prove a connected
  recovery contract or authorize generating one for the preserved controller.
- `controller.prepare` first requests `prepare_launch`, then freezes/attaches
  source using the returned launch and run IDs. The inspected controller request
  operations expose no direct `launches`/`launch_evidence` lookup. Its
  `diagnostics(run=...)` selects pre-admission launches through `launches(run)`;
  whether that supplies sufficient exact binding and resumable preparation still
  needs proof against the compatible controller.
- `submit_launch` claims the preparing launch before source freezing and admission;
  a repeated claim fails. A lost reply requires reconciliation, not a blind retry.

Source SHA-256 identities (relative to that environment's site-packages):

| File | SHA-256 |
| --- | --- |
| `etude/launch/scheduler.py` | `9e70e8c738118bc3108e3d986d1a53ea1b3e6d8fff8253dac658c40146dca2d6` |
| `etude/launch/database.py` | `cefe65329d2d2d6f09ee6518e0a1f32e5508a828bade308c2b938cf0edb7d5a8` |
| `etude/launch/controller.py` | `58e22bbf8245251849e70419ab14b2e2736e621d99239ce73cf4e99e383c6c88` |

Stop at the compatible-controller inventory dependency. Once available, prove
lost-prepare recovery using an explicitly retained request identity and verify
writer closure, cost and delivery through that same controller before connecting
the existing evidence owner. No speculative adapter or additional ledger was
added. Original-runtime admission, training and comparison remain outstanding.
This pass changed evidence notes only; no tests, games or fits ran.
