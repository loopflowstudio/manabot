# GIH card rankings

Manabot's card evaluation uses Etude's existing scheduling, resume, Parquet and
DuckDB query infrastructure. Each match contains one deal's eight authored
deck/seat legs. The private native recorder runs during arena command replay,
outside policy observations, rule hashes, RNG and search clones.

`CardEvaluationSpec` in `manabot/arena/card_evaluation.py` accepts admitted arena
registrations, explicit checkpoint paths, unique deal seeds, limits and an
optional `AuthoredMatchup`. Omit that matchup for the existing Lessons/Allies
matrix. It uses the same native setup and checkpoint admission as arena play.
The run ID is explicit and resuming requires identical agents and conditions.

```sh
uv run --extra evaluation python -m manabot.arena.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.arena.cards rank --source .runs/cards --run RUN --player AGENT --json
```

Select an agent by its Etude identity or unique registration name. Add repeated
`--opponent` filters and optional `--deck`, `--opponent-deck`, or
`--position play|draw` filters.
Completed schedules are required; `--partial` explicitly reports scheduled,
completed, failed and pending matches. Failed matches have no admitted card
rows. Old outcome-only evaluations report unavailable card evidence.

GIH is **winning qualifying copies / qualifying copies**: opening copies plus
actual subsequent draws, including library redraws. Selection/tutor, bounce,
sideboard acquisition and generated-card acquisition are separate native facts,
not draws. Duplicate card copies count separately. True draws and invalid games
are excluded, never assigned the arena's half-point score. Zero opportunities
produce a null rate. There is no implicit minimum sample filter. These are
descriptive associations, not causal card values.

The deterministic 95% percentile interval uses Etude's ratio estimator on whole
**deal seeds**, pooling every correlated seat/deck leg and opponent pairing.
It is unavailable with fewer than two deals. Output includes counts, opponent
mix, versions, completeness and contributing game IDs.

`card_games` and `card_evidence` are author tables available through
`etude.eval.query.connect(source)`. Etude adds `run` and `match`; `player` is an
Etude participant, not an engine seat. The card key is native content digest plus
definition ID; names are labels. Evidence contains copy/incarnation identity,
owner/controller, movement source/destination/cause, sequence and turn, original
commands and hashes, and the replay receipt. Queries verify artifact bytes,
require exactly one of each scheduled leg, and bind each artifact to its deal,
participant seating, complete deck/sideboard setup and retained runtime/content
identity before rebuilding reducer rows and accepting tables. The same admission
applies to delivered evaluations; missing identities fail admission. No model
inference is needed.
These are private full-game artifacts, not viewer-safe public replay payloads.

```sh
uv run --extra evaluation python -m manabot.arena.cards publish --source .runs/cards --destination /path/to/copied-evaluations
```

Publication copies and verifies immutable compressed evidence using Etude
Storage, then delivers Etude tables and completion records. SDK 0.2's ordinary
`etude sync` omits arbitrary artifacts and is insufficient for these evaluations.
No remote destination is inferred. Keep the full evaluation for replay/rebuild.

```sh
uv run --extra evaluation python -m manabot.arena.cards export-log --source .runs/cards --run RUN --game GAME --viewer 0 --mapping mapping.json --out game.log
```

`CardMapping` binds each exported content/definition key to a retained full
Scryfall card response with a verified `arena_id`. Unmapped cards remain rankable;
export refuses missing or mismatched printings. A mapping is an explicit authored
printing choice, never a name-based join or a fabricated Arena ID.

The `arena-card-log-v1` subset carries own deck, opening hand and subsequent hand
objects, public zone objects, turns and terminal results. Opponent private card
identities and library order are absent. It does not reproduce combat animation,
priority, annotations, life changes, public hand revelations, or a full Arena
replay. It requires a completed verified game with at least four recorded moves.
Current worlds have no mulligan procedure: no mulligan messages are invented,
and the pinned client reports its unavailable count (-1). Parser hand sightings
include tutors/returns and are **not** the GIH metric. These logs are synthetic
offline artifacts and must never be submitted to 17Lands.

The test-only upstream client is pinned with its license under
`tests/fixtures/seventeenlands`. Conformance runs in a separate process with
network disabled and an in-memory sink. Source pin, extracted fields and private
visibility are checked without skips. The fixtures do not establish downstream
server parity or universal Arena compatibility.

The `Card evaluation scale proof` workflow can run the declared 1024-game check
on CPU CI. `verify` reports elapsed time, compressed evidence bytes and peak
parent-process and largest child-process memory; it is an engineering proof, not paid scientific execution.
