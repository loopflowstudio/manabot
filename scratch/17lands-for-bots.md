# Card rankings from evaluation games

Accepted outcome and implementation plan — reconciled 2026-10-23. Jack Heart
selected GIH card ranking as the first useful result and requested implementation
after review, then pursuit on 2026-10-23. The implemented keystone remains
CLI/SQL rankings, retained facts, Etude infrastructure and bounded offline Arena
logs. Later metrics, HTTP API and Academy screens remain draft follow-ups.
Engineering defaults dated 2026-10-23 are retained in `scratch/questions.md`;
they do not authorize paid runs or external synthetic submissions.

## What to build

Rank cards by Games in Hand win rate for a selected Manabot evaluation, using
Etude's queryable evaluation tables and retained game evidence that can also
produce a tested MTG Arena log projection.

Intent, verbatim: “maintain fidelity not just at the metric level but at the
data processing level”; “for now i basically just want to be able to rank cards
but GIH, but it should be designed aware that we will expand to the 17lands
style API”.

This is an **additive series**. The keystone is a local query/CLI result plus
processing fidelity; a website, HTTP API, and additional metrics are follow-ups.
No managed-work placement was requested.

## Intended demonstration

Choose byte-identified models, fixed deck assignments and deal seeds; run an
Etude evaluation containing thousands of games. Open its retained tables and
rank cards for one model against a selected opponent cohort. Display card,
GIH win rate, qualifying copies, distinct games, and uncertainty. Drill into a
card's contributing games through stable evidence references. Rebuild the
same ranking from retained evidence without model inference. Export one game's
two viewer-specific Arena logs and parse them offline with a pinned 17Lands
client, comparing its extracted fields against the expected game facts.

Implemented entry points (the existing `evaluation` extra supplies DuckDB):

```sh
uv run --extra evaluation python -m manabot.eval.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.eval.cards rank --source .runs/cards --run RUN --player AGENT --json
uv run --extra evaluation python -m manabot.eval.cards export-log --source .runs/cards --run RUN --game GAME --viewer 0 --mapping mapping.json --out game.log
```

## Current system

`eval.match.play_cell` owns bounded games and retained Commands;
`sim.replay` checks their execution. `eval.benchmark.evaluation_tables` emits
outcome-only `plays` rows; `eval.duel.play` uses it for model comparisons.
Etude already owns workers, identities, resume, Parquet delivery, and DuckDB
queries. Its `player` means tournament participant, not physical engine seat.

Native setup deals seven cards without mulligans. Opening cards still bypass
`GameEvent`, and `CardDrawn` omits card identity. The private native recorder now
captures setup before automatic progression and committed zone movements during
verification replay. It closes these capture gaps without inferring draws from
policy snapshots or adding metric events to the rules ledger.

The synced base already consolidates evaluation/tracking into Etude (`8c7f55813`)
and retires historical study tooling (`003644bda`). The implementation uses that
existing simulation/evaluation/Etude path; retired experiment runners are not integration targets.

## Data structures

- `CardEvaluationSpec`: admitted agents, world/content identity, authored matchup,
  seeds, execution limits and capture/metric versions. Etude schedules one existing
  eight-leg deck/seat matrix per pairing/deal; evaluation admission checks code and
  checkpoint entrants. Omitted matchup selects the existing Lessons/Allies matrix.
- `GameEvidence`: trace/replay receipt hashes, terminal outcome, setup/decks,
  participant-to-engine-seat mapping, and ordered native facts. Capture opening
  cards before automatic progression; retain card definition, physical copy and
  incarnation, owner/controller, source/destination zone, movement cause, turn,
  and event sequence. Distinguish actual draws, selection/tutor, return to hand,
  outside-game acquisition, and generated cards. Unknown causes fail admission.
- `CardSnapshot` and `CardGame`: content-bound native definitions, physical copies
  and display names. Optional export bindings live separately in `CardMapping`,
  using retained Scryfall printing responses. There is no separate `CardIdentity`
  class. Display names are labels, not join keys; Arena IDs cannot be invented.
- Etude author tables: `card_games` (one game/participant/card, deck copies,
  opening/drawn/GIH counts, outcome and eligibility); `card_evidence` (game,
  participant, immutable artifact reference/hash and format versions).
  Etude adds `run`/`match`; all rows identify `player` or `group` as required.
  Detailed events remain in compressed immutable artifacts, not opaque payloads
  repeated for every card row. References and bytes publish together. SDK 0.2 does not deliver
  arbitrary artifacts: the card publication operation copies verified immutable
  evidence through Etude Storage before delivering evaluation tables/indexes.
  The implementation resolves the source prefix once per run, validates the bytes
  copied, and passes discovered parent directories to SDK sync, which expects
  immediate run children. Nested-run publication/rebuild has focused coverage.

## Metric and query contract

For card c, `GIH WR = sum(win * qualifying_copies) / sum(qualifying_copies)`.
Use opening copies plus subsequent actual draws; retain duplicate copies and
redraws. Tutor, bounce and generated-card acquisitions remain distinguishable.
Version the counting rule and its fixtures. A zero denominator returns null.
First-version policy excludes true draws, crashes, timeouts and replay
failures from this win-rate denominator and reports their counts separately;
it never substitutes the arena's half-point draw score.

Require an evaluation and subject agent. Filters cover opponent, deck matchup,
and play/draw. Default aggregation weights the selected observations, reports
the opponent mix, and never pools incompatible environments. Sort descending
by rate, then stable card identity; expose counts without an implicit sample
cutoff. Historical outcome-only runs report unavailable card evidence.
Completed matches remain reusable after interruption. Rankings require a complete
selected schedule by default; an explicit partial query includes scheduled,
completed, failed and pending counts and never labels its cohort complete.

Intervals resample shared deal blocks, including all correlated legs and, when
pooling opponents, the same deal across pairings. Etude's match bootstrap is
sufficient only when the selected independence unit really is a match. Aggregate
winning and qualifying copies once per card/deal, then include every selected
deal (including zero-opportunity deals) in the estimator.

## Capture and processing authority

The passive native evaluation recorder is enabled before reset during existing
verification replay, outside game-state hashes, policy observations and branch
clones. It records committed root transitions. Capture must preserve RNG,
Commands, event ledgers and outcomes; focused native/replay checks support this
invariant, with full native regression still outstanding. Tables are admitted
only after ordinary replay and capture-completeness checks pass.

One reducer builds `card_games` from `GameEvidence`. An export adapter builds
viewer-specific Arena GRE/log messages from those same facts. Preserve deck,
object/zone, opening-hand, turn, and result semantics; emit no invented mulligan
decisions. The adapter is a bounded, versioned compatibility surface, not a
claim to reproduce every Arena protocol message or replay feature.

Parser conformance runs the pinned upstream client in an isolated harness with
an in-memory submission sink and network disabled. Compare extracted facts,
not just totals. The public client records more hand sightings than count as
draws: its `drawn_cards` field is not the GIH definition. Keep native-to-metric
goldens independent of native-to-log-to-client goldens. Unmapped cards remain
rankable internally but cannot claim successful Arena-ID export.

## Key functions

```python
def play_card_match(match: Match) -> Tables: ...
def capture_verified_game(trace: GameTrace) -> GameEvidence: ...
def reduce_cards(game: GameEvidence, participants: tuple[str, str]) -> list[CardGame]: ...
def rank_cards(source: Storage | Path | str, query: CardQuery) -> CardRanking: ...
def export_arena_log(game: GameEvidence, viewer: Seat, mapping: CardMapping) -> str: ...
```

`GameTrace` is a typed view of the existing receipt, not a replacement
replay format. Query output includes selection, versions and completeness.

## Constraints

managym owns factual capture; manabot owns Magic metrics and log adaptation;
Etude owns evaluation storage/query infrastructure; Academy may consume the
future API. No manabot import from Academy. Use one versioned initial evidence
schema, not intermediate migrations. Installed SDK 0.2 supports scheduling,
resume and query; its arbitrary-artifact delivery gap is handled explicitly.
Existing outcome consumers and frozen evidence remain intact. No existing
scientific run is relaunched.

Package direction, reaffirmed by Jack Heart during interactive concept review:
`manabot/sim/` owns reusable game execution/replay and opponent policies;
`manabot/eval/` owns evaluation and card statistics; `manabot/experiment/`
owns checkpoint-following orchestration. Retire the mixed `manabot/arena/`
namespace; reserve Arena terminology for actual MTG Arena interoperability.
Current commands and APIs above use the migrated owners.

## Forbidden outcomes

Counting presence once per game; treating every hand entry as a draw; leaking
the opponent's private cards into a viewer log; changing policy inputs; counting
both perspectives as independent games; silently dropping failed matches;
duplicating completed matches on resume; claiming full 17Lands server parity
from client acceptance alone; uploading synthetic games to 17Lands.

## Internal slices

1. **Implemented: capture and conformance.** Native setup/committed movements,
   physical copy/incarnation facts, explicit draws, root-only capture, and offline
   pinned-parser tests for both viewers. Unmodified upstream test source and its
   license are retained; sockets are disabled and submissions use an in-memory sink.
2. **Implemented: evaluation and query.** Simulation verification replay
   writes immutable compressed evidence containing its original trace. Etude
   stores `plays`, `card_games`, and `card_evidence`; rank rebuilds reducer rows
   before accepting the scalar tables. Resume skips completed matches and retains
   separate directories for failed attempts. Filters, shared-deal intervals,
   explicit incomplete cohorts and evidence-first publication use the existing SDK.
3. **Implemented: scheduled evidence admission (2026-10-23).** Shared
   `card_schedule.CardConditions` describes production and read-time conditions;
   `card_match_players` owns the eight-leg/two-participant shape and registration
   name binding for both paths. `rank_cards` uses `admit_card_match` to require
   every unique leg before reducing facts in the admitted participant order.
   Each artifact must match its scheduled deal, participant seat, deck label and
   complete deck/sideboard setup. Retained world, engine-source, content-manifest
   and native content identities bind it to the Etude environment; runtime setup
   and registration ABIs must also agree. `GameEvidence.bound` enforces
   capture/trace content agreement on every load. This replaces the insufficient
   unique-game-only admission; no separate scheduler or evidence store is needed.
   Rehashed wrong-deal, duplicate-leg, seat/deck and incompatible-content/runtime
   fixtures with consistently rebuilt scalar rows are rejected for both subjects,
   locally and after Etude delivery.
4. **Implemented: both-participant verification (2026-10-23).** `verify` queries
   both participants before and after trace verification. A focused regression
   rejects corruption confined to the second participant. It previously rebuilt
   only the first participant's scalar rows/ranking despite claiming every row.
   Both this gap and scheduled admission were found by source inspection, not
   observed failures in the preserved bounded runs.

## Remaining implementation and verification

**Prior reconciliation:** no gaps were identified in the 2026-10-23 reconciliation
of the schedule producer, shared admission, evidence loader, query and verifier
against their regression sources. The earlier scheduled-admission gap is closed.
This is a bounded source review supported by the preserved focused proofs, not
full regression or scale certification. The subsequent interactive review found
that this change missed the accepted MTG-179 package direction. The
mixed `arena/` ownership has now been corrected in this checkout; the
log-interchange proposal below remains a separate unresolved scope question.

**Deferred verification — gate/CI.** Full affected Python suites, the complete
debug Rust suite and Clippy, and the declared 1024-game proof remain outstanding.
The manual `card-evaluation.yml` workflow retains scale results/artifacts; its
existence is not a successful run. Existing bounded games and checkpoint proof
do not establish scale acceptance. The broader thousands-of-games demonstration
above remains an intended outcome, not completed evidence. No paid execution is
selected. The implemented keystone remains one change; later product work is
outside its acceptance boundary.

Follow-ups, intent only: OH/GD/GNS/IIH query results with independent conformance;
a read-only 17Lands-style HTTP card-statistics API; Academy ranking/filter/game
inspection screens; additional deck schedules and Arena protocol coverage.
Draft/pick metrics require new draft evidence and are not promised by game logs.

## Done when

Headless acceptance (entry points and schedule admission are implemented):

- `uv run --extra dev --extra evaluation pytest tests/eval/test_card_evaluation.py tests/eval/test_arena_log_conformance.py tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py -q`
  passes hand-calculated weighting, tutor/bounce/redraw, terminal/invalid cases,
  seat swaps, duplicate delivery, interrupted resume, filters, missing evidence,
  parser field equality and replay invariance, plus schedule-binding regressions
  above. Parser tests cannot skip silently.
- `uv run --extra dev --extra evaluation python -m manabot.eval.cards verify --games 1024 --out .runs/card-verification`
  completes bounded native games with small code agents, verifies every trace,
  and rebuilds identical rows/ranks for both participants from artifacts. A tiny
  checkpoint matchup is already in focused coverage. The scale result reports
  elapsed time, artifact bytes and peak parent/largest-child memory, with no
  invented throughput target. Scale proof remains assigned to capable CPU CI.
- Native changes require debug `cargo test`, formatting and Clippy under
  `managym/`, plus the root rebuild and package-area checks in repository guidance.

## Open assumptions and evidence

CLI/SQL first; draw exclusion; current authored layouts; bounded Arena-log
compatibility. Exact redraw/mulligan edge parity beyond published definitions
needs reference fixtures before claiming server equivalence. Current worlds
have no mulligan procedure; adding one is separate work.

Evidence: [source investigation and preserved outcomes](17lands-for-bots-evidence.md).
The prior 16-test focused Python pass, two debug recorder tests, native build,
lint/format, bounded native/checkpoint games, resume, nested publication/rebuild
and both-viewer pinned parser checks remain the supplied proof baseline. They
were not repeated during reconciliation and do not certify the deferred gates.
Check — 2026-10-23: focused game/replay, registration, GIH/parser, monitoring,
follower, search, Academy consumer, failure-recovery, boundary, CI-selection and
release tests pass (failed cases rerun after fixes); integration workflow collects
890 tests; changed Python Ruff/format and whitespace pass; release wheel built,
installed independently, passed dependency check, imported every sim module with
higher layers refused, and passed seven CLI help probes outside the checkout.
Full affected/native suites, Clippy and `verify --games 1024` remain gate/CI scope.

Engineering decisions: use existing arena admission (including checkpoint bytes,
parameter counts and shared observation ABI); local explicit evaluation spec;
SDK 0.2 requires evidence-first publication outside its ordinary sync; failed
matches expose known exclusion counts or explicitly unobserved scheduled games.
The export is an offline card/object/zone/turn/result subset, omitting full rules
replay and public-hand reveal annotations. It reports unavailable mulligan counts
from the real client, not fabricated prompts. Missing printing mappings fail
export but never prevent internal ranking. See `docs/card-evaluation.md`.

## Headless concept review — 2026-10-23

Pursue step 6 outcome: **nothing to change and no open product choice** within
Jack Heart's accepted GIH keystone. This review preserves the accepted design;
it adds no approval for later metrics, API/UI, paid execution or submissions.

The concrete interaction remains:

| Action | Concept and state | Owning API |
| --- | --- | --- |
| Evaluate explicit entrants and deals | One evaluation; Etude completes eight-leg matches | `evaluate_cards(CardEvaluationSpec, directory)` |
| Rank one subject, optionally filtering opponents/decks/position | A selection over admitted game facts; copy-weighted rate, counts and deal uncertainty are derived | `rank_cards(source, CardQuery)` |
| Inspect a contributing game | Stable game ID resolves immutable private evidence | `evidence_refs` / `load_game` |
| Export a chosen viewer offline | A bounded projection with an authored printing map | `export_arena_log(game, viewer, mapping)` |

Normal use follows `evaluate` → `rank --json` → a contributing game →
`export-log`. The existing usage guide and agent instructions describe that
experience without needing a new concept or instruction rewrite. A participant
is distinct from an engine seat because the matrix swaps seats; a physical copy
is distinct from a definition because duplicate copies and redraws affect GIH.
Native facts and Arena projections remain distinct because parser hand sightings
include acquisitions that are not draws. Merging those distinctions would lose
required behavior.

Failure/recovery preserves the same evaluation identity and completed matches:
repeat the unchanged evaluation to retry failed or pending matches; use
`--partial` only to inspect an explicitly incomplete cohort. Attempts retain
their diagnostics, while failed matches supply no admitted ranking rows. Missing
or corrupt retained bytes cannot be repaired by trusting scalar tables or by
ordinary resume, which skips completed matches; restore the original bytes from
a valid copy (or repeat evidence-first publication from an intact source).
Without those bytes the ranking remains unavailable. Missing printing mappings
leave internal ranking usable; supply verified mappings and repeat only export.

The counterexamples remain binding: repeated game presence cannot replace copy
weighting; a bounce/tutor cannot become a draw; eight unique artifacts cannot
replace the scheduled leg matrix; a completed index cannot stand in for missing
evidence; pinned-client acceptance cannot establish downstream server parity.
These follow the existing ownership: managym captures, manabot admits/reduces/
projects, and Etude schedules, persists and delivers. No additional lifecycle or
store is needed. Source and regression inspection found no further conceptual
mismatch; no implementation was changed and no proof runs were repeated.

Next proof remains the declared full affected Python/native checks, Clippy and
1024-game verification at gate/CI. The supplied bounded proof remains evidence
at its recorded scope. Step 7 owns navigation from this outcome.

## Interactive concept review — 2026-10-23

Jack Heart invoked concept-review after pursuit publication. The prior headless
review assessed the accepted bounded export; it did not settle the stronger
interchange experience below. This is a proposal and open product choice, not
approval to change implementation or restart execution.

**Question:** should processing fidelity mean that exported Arena logs can be
read back to reproduce the same GIH card rows and rankings for supported games,
or is the current parser-compatible export sufficient for the first release?

Current behavior has two paths: native facts → `reduce_cards` → rankings, and
native facts → Arena export → upstream-client extracted fields. The conformance
fixture deliberately expects six parser hand sightings but four qualifying GIH
copies for one viewer. That distinction is correct; however, passing the parser
test does not demonstrate metric interchange. The exporter does not retain
draw-versus-selection cause annotations. A library-to-hand move alone cannot
distinguish drawing from tutoring. This is a concrete limit on the stronger
promise, not evidence that the current GIH calculation is incorrect.

Proposed usage wording is drafted in `docs/card-evaluation.md`. A user would
evaluate models, rank a subject, export a game's viewer logs, then import those
logs to obtain the same card facts and derived ranking. The conceptual addition
would be an alternate evidence source, not another ranking implementation:
native and Arena inputs normalize into the same metric-relevant facts and use
one reducer. Native traces remain authoritative for engine replay; imported
logs cannot claim native replay verification.

Recovery counterexample: an import has hand arrivals but no reliable draw/tutor
distinction. It must report unsupported/incomplete metric evidence and preserve
the source bytes; it cannot infer that every arrival was a draw. Reprocessing
after adapter repair should not require replaying the bots or silently replace
an earlier version's derived result. Viewer privacy remains binding.

If selected, next proof is equality of per-card contributions through both
paths for duplicate copies, true draws, tutors, bounce and redraw, plus explicit
rejection of insufficient logs. This would expand the current first-release
boundary and needs Jack Heart's choice; no import API or general Arena replay
support is approved by this note.

### Package ownership correction

Jack Heart clarified that MTG-179 already selected moving away from `arena`
because of the conflict with MTG Arena, with responsibilities split between
`sim/` and `eval/`. The source is
`/Users/jack/src/manabot.training-correctness/scratch/training-credit-audit.md:641`:
it records the naming direction as accepted, the detailed relocation as proposed,
and the split as deferred from that CI repair. The landed
`docs/repository-archive.md` also retains this boundary as remaining work.

The prior design incorrectly treated existing package placement as a constraint.
Accepted target ownership: reusable execution/replay/policies in `sim/`;
evaluation protocols, admission, ratings, GIH reduction/query/CLI and MTG Arena
export in `eval/`; checkpoint-following orchestration in `experiment/`.
Native recording stays in managym. Shared simulation evidence must not import
evaluation, training or experiment orchestration.

**Delete — do not maintain:** retire `manabot/arena/` by cutting its live consumers
over to those owners, including CLI/module references and tests. Do not leave
re-export shims or another implementation behind. Split the new `card_evidence`
module where its replay/capture responsibility meets metric reduction; do not
move the entire mixed package to `eval/`. Actual MTG Arena export may retain that
specific name. Frozen serialized evidence and recorded historical commands retain
their original identities; active APIs, imports and instructions adopt the new
paths. The completed cuts are recorded below.

Jack Heart subsequently requested “lets complete the arena migration now” on
2026-10-23. This authorizes the direct consumer cutover in this checkout. It does
not select the separate log-import proposal.

The migration is indivisible and ships in this PR. **Implemented:** removed the
mixed package and divided its models, execution, capture, metrics and followers
at their dependency boundaries. `sim.registration`, `sim.match`, `sim.replay`,
`sim.card_evidence` and simulation policies remain independent of evaluation,
training and experiment orchestration. `eval` owns admission, schedules, rating,
monitoring schemas, comparisons and GIH processing. `experiment` owns checkpoint
staging/admission orchestration and followers. Canonical byte identities belong
to `infra.identity`. Move exclusive tests and active CLI/workflow consumers in
the same cut; preserve their behavioral assertions. No schema migration or
compatibility import package is needed. Existing serialized arena-named keys,
seed derivation identifiers and evidence paths remain readable unchanged;
relocated source receives its actual new identity.

Migration acceptance is complete at the local implementation boundary. The
simulation probe rejects eval, training, experiment and reports imports, and the
retired package is absent in source and installed wheel. CI follows `tests/eval/`
and the moved simulation tests; release CI probes the new CLI modules.
**This slice:** complete. Remaining verification is the broader GIH gate/CI scope
already declared above. Publication/merge of this migration has not occurred.
The v1 native-evidence envelope retains its historical metric-version tag;
all counting rules and the current metric version constant live in eval.

## Cross-run aggregation direction — 2026-10-23

Jack Heart's direction, given while reviewing the PR walkthrough: GIH should
aggregate as many games as possible, filterable by game properties such as
opponent or play/draw; "however many evals we run at a time shouldnt matter. we
should be aggregating across all the runs into one big db". This supersedes the
per-run, complete-schedule ranking contract above. It is not yet implemented.

- **One queryable database per experiment, explicit selections.** Every run of
  an experiment lands in one store that can be queried together; the number
  or size of its runs does not constrain a query. Jack Heart: "i just
  structurally think you should be able to have a big DB and query it", not
  read every evaluation by default, and "1 DB per experiment. no cross
  experimetn joniing by default". An experiment contains many runs; a query names
  its experiment and by default joins all of that experiment's runs, then
  selects games by subject, opponent, play/draw and decks. A run filter only
  narrows that default. Cross-experiment joins
  require an explicit request. Etude records each run's experiment and its
  `connect` can scope to it, but its storage layout is `<root>/<run>`, not
  per experiment; the current `EXPERIMENT = "gih-card-evaluation"` constant
  puts every card evaluation in one experiment, so the spec must name it.
- **Participants come from joined run tables (Jack Heart: "we need to join it
  against tables saying who the participants were in each eval run").** Etude
  `connect` already materializes `runs(run, experiment, environment, ...)`,
  `seats(run, match, player, agent)` and `agents(agent, name, kind, artifact,
  config)`. Card rows carry `run`/`match`/`player`, so subject and opponent are
  joins, not fields copied into each row: `card_games ⋈ seats ⋈ agents` for the
  subject, the other seat for the opponent, `runs` for the experiment.
  `agents.artifact` is the checkpoint digest, so checkpoint-keyed grouping
  works through this join; the commit would join from run or agent config.
  Proof (2026-10-23, local 16-game verify run): the join returned subject,
  opponent, play/draw, games and GIH copies per experiment.
- **Game-level admission.** Each game is admitted on its own integrity (bytes,
  trace/replay digests, passed replay, content identity). A failed or missing
  game is a reported exclusion; it does not void its deal or block ranking.
  The eight-leg matrix remains one way of producing balanced games, not an
  admission requirement, so schedule binding and the completeness/`--partial`
  gate retire.
- **Identity keys on the checkpoint; the commit is logged (Jack Heart).** An
  initial note kept the source-code digest in agent identity ("easier to stick
  to one commit"); Jack Heart then refined it: "maybe can key on checkpoint but
  just also log the git commit hash so that we can validate it matched when we
  have questions". Proposed shape: checkpoint agents key on checkpoint bytes
  plus registration (player spec, observation/action ABIs), deliberately
  omitting Etude's suggested code list; code agents key on player spec plus
  their policy `source_sha256`. Every game records git commit, dirty flag and a
  digest of uncommitted changes (nothing records the commit today). Pool on
  world, content digest and capture/metric versions; engine source digest and
  commit are per-game provenance and filters, not pooling keys.
- **Subjects are a selection, not a single fixed agent (Jack Heart: "but allow
  people to use different ceckpoints if they want").** A query names one or
  more checkpoints (or code agents) as subjects and optionally opponents; rows
  from every selected subject pool, and output reports the subject mix.
  Selecting one checkpoint is the ordinary case.
- **Persistent experiment database; verify at ingestion (Jack Heart: the
  rebuild-per-connect behavior "cant be teh right thing to do").** Today each
  `etude.eval.query.connect` rebuilds an in-memory DuckDB from every Parquet
  file and match record (downloading them from S3), and every `rank_cards`
  re-reads, re-validates and re-reduces all selected evidence. Proposed: an
  experiment owns one persistent database, updated incrementally as matches
  complete; evidence is verified and reduced once when a match is admitted;
  ordinary queries trust admitted rows; an explicit audit command re-verifies
  evidence on demand. Ownership follows the existing boundary (Etude owns
  evaluation storage/query), so the persistent database is an Etude
  capability; manabot owns admission checks and card reduction. Jack Heart:
  "we write etude. we can issue PRs against etude as necessary" — Etude
  gaps (persistent experiment database, artifact delivery that forced
  evidence-first publication) are fixed in Etude, not worked around here.
- **Normalized tables (proposed).** One per-game `card_plays` row per
  game × participant (game facts, outcome/eligibility, evidence ref); counts
  only in `card_counts` per game × participant × card; a small `cards`
  dimension; participants via Etude `seats`/`agents`. `plays` stays the shared
  outcome table. `leg` and the fixed eight-game layout leave the data model.
- **Uncertainty.** Resample by deal seed across all pooled runs, because games
  sharing a seed share shuffles.

Open: implementation on this branch has not been requested.

### Concept review — experiment, scheduling and games (2026-10-24)

Jack Heart, asked whether the experiment is an Etude tournament: "A tournament
is one way of deciding what eval runs to do and which matches to run. This is
indepedent of how we determine who to play against whom". Accepted reading:
**scheduling produces games; the experiment holds them.** The experiment
database is every eval run declared with that experiment, however it was
scheduled (tournament hold, round robin, benchmark, a plain `evaluate` call).
Card queries never consult a schedule: no schedule binding, no completeness
precondition, no `--partial`. A tournament's `complete` remains a scheduling
fact it can report about its own matches.

Proposed consequences (not implemented): the evaluation spec names its
experiment (replacing the `gih-card-evaluation` constant) and any producer may
write card tables into it; admission is per game at ingestion; subjects and
opponents come from `seats`/`agents` and may be grouped by `agents.artifact`.

Environment pooling (Jack Heart, 2026-10-24: "I just dont think we should
worry about it. i guess 1 is fine here"): within an experiment, games pool
across engine/manabot commits. The pooling key is world + content; engine
source and git commit are logged per run and available as filters, not
environment identity. Card keys already include the content digest.

Next proof (when implementation is requested): games from two runs of one
experiment with different engine commits, different producers and a failed
game rank as one selection; a filter by opponent, play/draw and commit narrows
it; a second experiment's games never appear without an explicit request.

## Etude PR plan — proposed (2026-10-24)

Jack Heart asked to plan the Etude PR. Proposal, not implemented; Etude source
read at `loopflowstudio/etude` `origin/main` `2bd2b75` (one commit past v0.2.0).

**Already on Etude main (unreleased):** #32 made delivery carry every file
under a run (`etude.carriage.listing`), not only Etude's named folders. Evidence
written under the run therefore ships with ordinary delivery and `etude sync`
once manabot pins a release containing it, and `publish_cards` can be deleted.
Remaining gap: only metrics, tables and checkpoints are `immutable`; any other
file is re-hashed on every delivery pass, so thousands of evidence files cost
O(files) reads after every match.

**Not needed in Etude:** an ingestion hook. `play(match)` already is ingestion:
manabot verifies replay/evidence there and returns admitted rows, and a raising
match is recorded failed. Per-game failures should become ineligible rows, not
raised matches (manabot side). Checkpoint-keyed grouping uses
`agents.artifact`; no identity change.

**Etude PR — "Query an experiment's evaluations from a kept database":**
1. *Experiment scope.* `connect(source, experiment=...)` includes only eval runs
   whose `run.json` names that experiment (Etude's own tables too). Omitted keeps
   today's every-run behavior for reports.
2. *Kept, incremental database.* `connect(..., keep=True)` opens a DuckDB file
   under `ETUDE_HOME` keyed by source and experiment, derived and rebuildable
   (storage stays the authority). Each open lists keys, ingests only new ones:
   done match records (`matches/KEY.json`, never rewritten) and failed ones
   superseded by a done record; table Parquet files (immutable, write-time in
   the name). A merged file replaces earlier rows of its (run, match) when its
   write time is later, the same rule `connect` applies today. Deleted keys
   need no action because their rows reappear in the merged file. A schema
   version or Etude-version change rebuilds the file. Bytes still come through
   `ObjectCache`.
3. *Immutable artifacts folder.* Files under a run's `artifacts/` are sent once
   (`immutable`), and authors are told to name them by content. manabot writes
   `artifacts/evidence/<sha256>.json.gz`.
4. *Release.* Jack Heart (2026-10-24): "we can run an etude release (0.2.1)
   whaenever". Step one is releasing current main as 0.2.1 so manabot can pin
   #32's delivery and delete `publish_cards` independently; this PR follows in
   a later release.

*Interface direction (Jack Heart, 2026-10-24: "connect seems like a weird
interface"; "this should be like etude.eval.statsdb()"):* items 1–2 become one
entry point, `etude.eval.statsdb(experiment, store=None)`, that opens the
experiment's kept database (store from Etude config unless given). The
experiment is required; `keep` disappears because keeping is the only behavior.
`rebuild()` recreates it from storage. `curves`/`standings`/`compare` and
manabot's ranking take it where they took the connection; `connect` becomes a
deprecated wrapper or is removed. Jack Heart (2026-10-24): "im ok with it
taking new matches automatically", so opening catches up.

*Landed as loopflowstudio/etude#34 (`a821680`, 2026-10-24) and released in
`etude-sdk` 0.2.1 (PyPI; tag `v0.2.1` at `83797c6`, with #32, #33, #35 and #37).
The pieces below are what manabot pins against:*
- `statsdb(experiment, store=None, *, rebuild=False)` replaces `connect`, which
  is removed (we own its callers). The file lives at
  `ETUDE_HOME/statsdb/EXPERIMENT.STORE.duckdb`, with bookkeeping in the `kept`
  schema. Opening it lists the store once and reads only unseen files. A
  second lock holder gets `statsdb_in_use`. Rebuilding is a flag, not
  `db.rebuild()`, because callers keep a plain DuckDB connection.
- Found while implementing: `hold` named each eval run's experiment after the
  tournament. That contradicts "a tournament is only scheduling" and would
  have kept benchmark results out of their training experiment's database. Now
  `hold(..., experiment=)` and `evaluation(..., experiment=)` require the
  experiment. A batch hold reuses only matches delivered by the experiment's
  own eval runs; that costs one extra `run.json` read per eval run.
- `etude reports eval` writes `reports/eval/EXPERIMENT.html` for each
  experiment, with an optional `--experiment`, matching the training reports.
- Checks: 424 passed and 6 skipped across the full suite, plus mypy and ruff.
  The new tests are in `tests/eval/test_statsdb.py`.
- Not done: the stated "immutable `artifacts/`" delivery item (PR item 3), and
  the manabot follow-ups. Pinning 0.2.1 breaks `card_query.py` and
  `remote/live_evaluation.py` (`connect`) and the `hold` calls in
  `eval/benchmark.py`, `eval/duel.py` and `eval/card_evaluation.py`, which
  must name their experiment.

Failure/recovery: an interrupted ingest leaves the DuckDB transaction
uncommitted, so the next open repeats it; a corrupt or stale file is deleted
and rebuilt from storage; no derived state is authoritative.

Tests (Etude): experiment scoping excludes another experiment's runs; a second
open ingests only a newly delivered match; merge replacement neither drops nor
doubles a match; failed-then-done match counts once as done; rebuild after
schema bump; artifacts delivered once.

**manabot after the release:** delete `publish_cards`, schedule binding,
completeness/`--partial`, and per-query evidence re-reading; normalized
`card_plays`/`card_counts`/`cards`; per-game exclusion rows; spec names its
experiment; `rank --experiment` uses the kept database; `audit` re-verifies
evidence on request; runs record git commit and engine source as provenance.
