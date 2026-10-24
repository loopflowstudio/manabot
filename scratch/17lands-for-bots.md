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
uv run --extra evaluation python -m manabot.arena.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.arena.cards rank --source .runs/cards --run RUN --player AGENT --json
uv run --extra evaluation python -m manabot.arena.cards export-log --source .runs/cards --run RUN --game GAME --viewer 0 --mapping mapping.json --out game.log
```

## Current system

`arena.match.play_cell` owns bounded games and retained Commands;
`arena.replay` checks their execution. `arena.benchmark.arena_tables` emits
outcome-only `plays` rows; `arena.duel.play` uses it for model comparisons.
Etude already owns workers, identities, resume, Parquet delivery, and DuckDB
queries. Its `player` means tournament participant, not physical engine seat.

Native setup deals seven cards without mulligans. Opening cards still bypass
`GameEvent`, and `CardDrawn` omits card identity. The private native recorder now
captures setup before automatic progression and committed zone movements during
verification replay. It closes these capture gaps without inferring draws from
policy snapshots or adding metric events to the rules ledger.

The synced base already consolidates evaluation/tracking into Etude (`8c7f55813`)
and retires historical study tooling (`003644bda`). The implementation uses that
existing arena/Etude path; retired experiment runners are not integration targets.

## Data structures

- `CardEvaluationSpec`: admitted agents, world/content identity, authored matchup,
  seeds, execution limits and capture/metric versions. Etude schedules one existing
  eight-leg deck/seat matrix per pairing/deal; arena admission checks code and
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
def capture_verified_game(trace: ArenaGameTrace) -> GameEvidence: ...
def reduce_cards(game: GameEvidence, participants: tuple[str, str]) -> list[CardGame]: ...
def rank_cards(source: Storage | Path | str, query: CardQuery) -> CardRanking: ...
def export_arena_log(game: GameEvidence, viewer: Seat, mapping: CardMapping) -> str: ...
```

`ArenaGameTrace` is a typed view of the existing receipt, not a replacement
replay format. Query output includes selection, versions and completeness.

## Constraints

managym owns factual capture; manabot owns Magic metrics and log adaptation;
Etude owns evaluation storage/query infrastructure; Academy may consume the
future API. No manabot import from Academy. Use one versioned initial evidence
schema, not intermediate migrations. Installed SDK 0.2 supports scheduling,
resume and query; its arbitrary-artifact delivery gap is handled explicitly.
Existing outcome consumers and frozen evidence remain intact. No existing
scientific run is relaunched.

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
2. **Implemented: evaluation and query.** The arena's existing verification replay
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

**Implementation/concept gaps:** none identified in the 2026-10-23 reconciliation
of the schedule producer, shared admission, evidence loader, query and verifier
against their regression sources. The earlier scheduled-admission gap is closed.
This is a bounded source review supported by the preserved focused proofs, not
full regression or scale certification. No product decision is outstanding for
the current keystone.

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

- `uv run --extra dev --extra evaluation pytest tests/arena/test_card_evaluation.py tests/arena/test_arena_log_conformance.py tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py -q`
  passes hand-calculated weighting, tutor/bounce/redraw, terminal/invalid cases,
  seat swaps, duplicate delivery, interrupted resume, filters, missing evidence,
  parser field equality and replay invariance, plus schedule-binding regressions
  above. Parser tests cannot skip silently.
- `uv run --extra dev --extra evaluation python -m manabot.arena.cards verify --games 1024 --out .runs/card-verification`
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
Check — 2026-10-23: preserved compress proof: `uv run --extra dev --extra evaluation pytest tests/arena/test_card_evaluation.py -q -k 'participant_schedule or schedule_admission or interrupted_schedule or native_evaluation_resume or loaded_evidence or scale_verifier'` → 19 passed, 9 deselected (58.37s); changed-file Ruff/format, `uv run --extra evaluation python -m compileall -q` and whitespace passed; documentation-only realign: `git diff --check` passed, tests not repeated. Full affected/native suites, Clippy and `verify --games 1024` remain deferred to gate/CI.

Engineering decisions: use existing arena admission (including checkpoint bytes,
parameter counts and shared observation ABI); local explicit evaluation spec;
SDK 0.2 requires evidence-first publication outside its ordinary sync; failed
matches expose known exclusion counts or explicitly unobserved scheduled games.
The export is an offline card/object/zone/turn/result subset, omitting full rules
replay and public-hand reveal annotations. It reports unavailable mulligan counts
from the real client, not fabricated prompts. Missing printing mappings fail
export but never prevent internal ranking. See `docs/card-evaluation.md`.

## Concept review — 2026-10-23

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
