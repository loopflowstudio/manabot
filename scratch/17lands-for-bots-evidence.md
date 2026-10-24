# Card evaluation source evidence — 2026-10-23

Observations, not accepted implementation decisions. The source investigation
below predates implementation; the final section records the preserved proof
baseline and reconciliation findings separately.

## Etude and Manabot

Etude source inspected at local revision `5c18aca`:
`/Users/jack/src/etude/docs/evaluation.md`,
`src/etude/eval/{match,query,records}.py`.
Its author-owned scalar tables are stored as per-match Parquet, compacted on
completion, and materialized in DuckDB by `connect`. Match records own resume;
failed callbacks retain no author rows and stop the run. `compare` resamples
matches. Tournament round robins and benchmarks already supply scheduling.
At investigation time, the adjacent checkout alone did not establish
published-package equivalence. Implementation subsequently verified installed
`etude-sdk[eval]==0.2.0`; its ordinary sync omits arbitrary evidence artifacts.

Manabot integration already exists in:

- `manabot/arena/benchmark.py`: `play` calls `play_cell`; `arena_tables`
  requires eight valid legs and maps physical seat back to participant identity.
- `manabot/arena/duel.py`: direct checkpoint comparisons use that projection.
- `manabot/arena/match.py`: bounded processes, pre-command retention and replay.
- `manabot/arena/replay.py`: verifies frames, Commands, state digests and outcomes.
- `manabot/verify/learnability/elemental_eval.py`: another specialized Etude
  callback exists. It is not the initial integration target or scheduled deletion.

Native evidence gaps identified before implementation (now addressed by the
separate passive recorder, not by changing these event contracts):

- `managym/src/flow/setup.rs` deals seven cards by moving directly through zones,
  without creating `GameEvent` entries. It explicitly does not implement mulligans.
- `managym/src/flow/event.rs`: `CardMoved` has card identity and zones;
  `CardDrawn` has player and ordinal, but no card identity.
- `managym/src/flow/zones.rs::draw_cards` performs a movement before emitting
  `CardDrawn`. Preserve that causal association in the new capture contract.
- `managym/src/agent/env.rs::finish_game_step` consumes observation events.
  Polling the acting player's observations is not a complete capture mechanism.
- Existing event queues participate in authority fingerprints. A passive
  recorder must not append metric-only events to those queues or alter hashes.

## 17Lands

Primary sources:

- [Metric definitions](https://www.17lands.com/metrics_definitions).
  The page is client rendered; inspected its linked
  `Routes.df73627b876c0e3787de.bundle.js` on 2026-10-23.
- [Public client, pinned source](https://github.com/rconroy293/mtga-log-client/blob/b87160b65cfe5be1ff9f526cecacf6405147b198/src/python/seventeenlands/mtga_follower.py).

Published metrics weight individual card copies. GIH combines opening copies
and later draws. Draw metrics exclude tutors and returns from other zones;
library redraws can count again. Opening-hand statistics include the kept
seven before London-mulligan bottoming. This makes deduplicating each card name
per game incorrect. These are descriptive associations, not causal card values.

The client consumes GRE game objects, hand zones and result messages. It keeps
hand sightings keyed by instance ID, plus opening/mulligan lists and history.
Consequently its `drawn_cards` list alone cannot establish the published GIH
count. The public client is a concrete parser-conformance target, not evidence
that the server's downstream processing has been reproduced.

## Alternatives considered

1. Increment GIH counters in Python's policy loop: smallest-looking change,
   but misses intermediate movement and cannot support faithful reprocessing.
2. Emit Arena-shaped JSON and treat the public client's output as metrics:
   useful export, but conflates hand sightings with draws and leaves native
   identities/cause information dependent on an external wire representation.
3. Retain native facts during verified replay, derive metrics once, and test
   an Arena export against the real parser: subsequently implemented. Preserves the existing
   simulator and evaluation ownership while keeping fidelity independently testable.

Unknowns: reference logs covering redraw/object-ID changes, verified Arena IDs
for every exported printing, and inaccessible downstream server transformations.
Do not synthesize evidence for these gaps. The keystone can promise tested
compatibility for a documented subset, not universal Arena log interchange.

The original investigation performed source reads only, without games, parser
execution or paid work. That historical boundary does not describe later proofs.

## Preserved implementation evidence and reconciliation — 2026-10-23

The supplied successful implement/compress/sync outcomes record 16 focused Python
tests, two focused debug Rust tests, native build, lint/format and CI collection.
Those proofs include bounded native and checkpoint games, interrupted resume,
nested evidence publication/rebuild and both-viewer offline pinned-parser
conformance. The test sources remain in `tests/arena/test_card_evaluation.py`,
`tests/arena/test_arena_log_conformance.py` and
`managym/src/flow/card_capture.rs`. The native parser fixture uses Mountain/Forest;
the synthetic fixture separately tests tutor/bounce/redraw hand sightings. These
are bounded fixtures, not all-printing or downstream-server parity evidence.
No preserved `.runs/` artifact directory is present in this checkout; earlier
successes are supplied outcomes, not newly inspected scale receipts.

Reconciliation inspected the current producer, reducer, query, exporter, verifier
and test sources. It found that scale verification covered only the first
participant's ranking and corrected it to check both. The new focused test
rejects corruption confined to the second participant. Source inspection also
found missing artifact-to-scheduled-match validation. The continuation below
closed that gap; the implemented contract and regressions are retained in the
implementation plan.

The synced base (`8c7f55813`, `003644bda`) already centralizes Etude integration
and retires historical study tooling. No selected Task/Wave supplies additional
accepted scope. Full affected/native suites, Clippy and the 1024-game proof
remain unverified here. No paid runs or external synthetic submissions occurred.

## Pursuit continuation — 2026-10-23

The caller reported original implement/compress success and successful sync after
repairing intent-to-add index state. Those steps were preserved, not restarted.
Resumed step 4 realign completed successfully: both-participant verification was
repaired (one focused regression passed), and scheduled-evidence admission was
identified as remaining implementation. The subsequent step 5 selected the full
implement → compress → sync → realign iteration for that concrete gap. Deferred
full suites and 1024-game proof remain gate/CI obligations.

The implement continuation closes that read-time admission gap: shared scheduled
conditions bind complete leg matrices to deals, seats, decks and retained
environment/runtime identities. Rehashed artifacts with matching rebuilt rows
are rejected locally and after delivery for wrong deals, repeated legs and
incompatible content/runtime; both subjects are checked. The native eight-game
evaluation still passes resume and delivered rebuild. Focused results are recorded
once in the plan. Initial check failures exposed the documented SDK requirement
to admit the new source with `git add -N`, then a tuple-versus-JSON-list bug in the
synthetic fixture; both were corrected. No external publication occurred.

The required iteration completed implement and compress successfully. Compress
reported 19 focused cases plus Ruff, formatting, compilation and whitespace
checks passing. The new card_schedule.py intent-to-add entry was staged before
sync; `lf sync` then exited successfully with local edits preserved. The following
realign inspected `card_schedule` and its producer/query consumers, the evidence
loader, both-participant verifier and corresponding regression sources. The plan
now consolidates those completed mechanisms and separates deferred gates from
implementation gaps. No further concrete implementation or concept gap was
identified in that scope. This pass changed only the existing plan/evidence notes
and reused the supplied proof; it did not rerun games or select subsequent work.

Iteration realign exited successfully after documentation-only reconciliation;
`git diff --check` passed and prior proofs were preserved. Step 5 assessment:
the scheduled-admission and both-participant gaps are closed, no concrete
implementation gap or missing decision remains, and deferred checks alone do
not block this pursuit boundary. Continue to concept-review (step 6).

Concept-review exited successfully with no design change or open product choice;
normal/failure/recovery semantics were recorded in the existing design, code was
unchanged and whitespace passed. Step 7 assessment accepts that result within
the GIH keystone and proceeds to authorized step 8 publication. Required full
regression/native/Clippy/1024-game checks remain explicit gate/CI obligations,
not passing claims. Publication does not authorize merge or paid execution.

Step 8 `lf -b pr-publish` exited successfully. Published PR #360:
https://github.com/loopflowstudio/manabot/pull/360
Parent inspection confirmed OPEN, ready for review, and no auto-merge request.
The PR accurately records scope and outstanding full Python/native suites,
Clippy and 1024-game proof. All child steps were awaited and inspected; pursuit
is complete at publication, with no merge or next work selected. This final
publication receipt is retained locally after the published commit.

## Package migration evidence — 2026-10-23

Jack Heart requested completing the migration in this checkout. The mixed source
package and exclusive old test directory are gone. Split models/execution/capture
into simulation, schedules/admission/metrics into evaluation, and checkpoint
staging/followers into experiment. Canonical hashes now have one infra owner.
Live imports, subprocess entry points, current guides, CI selection and release
checks use the new paths. Frozen source-path allowlists, serialized keys, seed
identifiers, artifact directories and fixtures retain their meanings.

Focused checks exposed stale monkeypatch module targets, corrected to the actual
simulation/checkpoint owners. The Academy search-score fixture began on a forced
action, where search correctly produces no scores; it now advances to a real
choice before asserting score coverage. Etude requires moved source paths to be
tracked before hashing them; intent-to-add admission supplied that without
changing digest logic. Missing local notebook/artifact extras were installed
from the lockfile before integration collection. Passing proofs are summarized
once in the design. No native source changed during this migration, no external
scientific runs or paid jobs were launched, and no log-import API was added.
