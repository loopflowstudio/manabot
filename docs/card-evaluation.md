# Card data from evaluation games

Manabot reports 17Lands-style card statistics from its own evaluation games.
Every card evaluation is an Etude eval run that names an **experiment**. An
experiment's runs share one kept Etude database, so card data pools every game
the experiment has, however many runs produced them and however they were
scheduled. Each match plays one deal's eight authored deck/seat legs. The
private native recorder runs during command replay, outside policy
observations, rule hashes, RNG and search clones.

`CardEvaluationSpec` in `manabot/eval/card_evaluation.py` names the experiment
and run, and takes admitted registrations, explicit checkpoint paths, unique
deal seeds, limits and an optional `AuthoredMatchup`. Omit that matchup for the
existing Lessons/Allies matrix. Resuming a run requires identical players and
conditions.

```sh
uv run --extra evaluation python -m manabot.eval.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.eval.cards data --experiment EXPERIMENT --source .runs/cards
uv run --extra evaluation python -m manabot.eval.cards data --experiment EXPERIMENT --source .runs/cards --html .runs/cards.html
```

`data` prints, as `--json`, or as a page (`--html`) laid out like 17Lands'
Card Data, with opponent, position, deck and opponent-deck filters that switch
between selections computed in Python. Narrow any of them from the command line:

- `--subject` and `--opponent` (repeatable) take a player's name, its Etude
  identity, or at least eight characters of its checkpoint digest. Without a
  subject, every player's games pool, once for each side.
- `--position play|draw`, `--deck`, `--opponent-deck`.
- `--run` and `--commit` (a prefix) narrow to some runs or source commits.

`--source` is a directory or bucket of eval runs; by default, where runs are
written on this computer. The database lives under `ETUDE_HOME/statsdb` and
reads only blocks delivered since it was last opened; `--rebuild` reads everything
again. Another experiment's games never appear. Queries read only the card
rollups (below), never card rows, so the page's every filter combination is one
pass over a subject/opponent pair's per-deal totals.

## Card data from any evaluation

An evaluation that calls `play_cell` itself, such as a checkpoint grid or a
head-to-head, can deliver card data beside its own tables. Pass a scratch
folder as `card_evidence_dir`, merge the card tables and evidence blobs into
the `Result` that `play` returns, and declare the card rollups with the run:

```python
rows, trace, _ = play_cell(..., card_evidence_dir=scratch)
cards = trace_card_tables(scratch, trace, (player_a.player_id, player_b.player_id))
return Result(tables={"games": my_rows(rows), **cards.tables}, blobs=cards.blobs)
...
evaluate(..., rollups=CARD_ROLLUPS)  # manabot.eval.card_metrics
```

The participants are the two registered players in Etude row order (the order
of the match's players). The run then appears in `cards data --experiment` for
its experiment, and `cards audit` rebuilds its games from those participants;
only card evaluations are additionally bound to an eight-leg schedule.
Capture happens during `play_cell`'s existing verification replay and writes
about 90 KB of compressed evidence per game.

## What the columns mean

| Column | Meaning |
| --- | --- |
| # GP, GP WR | Games with the card in the deck, and their win rate. |
| # OH, OH WR | Copies in the opening hand, and the win rate over those copies. |
| # GD, GD WR | Copies drawn after the opening hand, and the win rate over those copies. |
| # GIH, GIH WR | Opening-hand plus drawn copies. GIH WR is **winning qualifying copies / qualifying copies**. |
| # GNS, GNS WR | Games with the card in the deck in which no copy reached the hand. |
| IWD | GIH WR minus GNS WR, in percentage points. |

Counts of copies are copy-weighted: duplicate copies count separately, and a
library redraw counts again. Selection/tutor, bounce, sideboard acquisition and
generated cards are separate native facts, not draws. Only games with a winner
and a passed replay carry a win. Draws, crashes, timeouts, truncations and
failed replays are listed by outcome and never earn a half point. Zero
opportunities give no rate. There is no implicit minimum sample filter. These
are associations between cards and wins, not causal card values.

The GIH interval is Etude's deterministic 95% percentile bootstrap over whole
**deal seeds** across every pooled run, keeping each deal's correlated legs and
opponent pairings together. Games sharing a seed share shuffles. It is
unavailable with fewer than two deals.

## Tables and admission

Each match delivers four author tables; Etude adds `run` and `match`.
`seat` indexes `Match.players`; `game_seat` is the native engine seat (0 is on
the play):

- `card_plays`: one row per game and player, with deal, decks, position,
  outcome, eligibility, win, the evidence blob's SHA256 and size, the trace and
  replay digests, and the source commit and dirty flag that produced it.
- `card_counts`: deck, opening, drawn and GIH copies per card in that game.
- `cards`: names for card keys (native content digest plus definition ID).
- `plays`: the shared outcome table every Etude evaluation report reads.

Etude sums `CARD_ROLLUPS` (`manabot/eval/card_metrics.py`) as each match is
stored and keeps them in the run's blocks: `card_deals` holds each card's
GP/OH/GD/GIH/GNS opportunities and winning copies per deal, position, decks,
source commit, player and opponent over eligible games; `game_deals` holds
player-games and wins per deal and the same splits, with eligibility and
outcome. Open the database with `card_db`, which also sums them once for runs
stored without them. A changed rollup takes a new name.

Subjects and opponents are the rollups' Etude `player`/`opponents` keys, named by `players`, never
fields copied into card rows; `players.artifact` is a checkpoint's digest.
Within an experiment, games pool across engine and manabot commits; the source
commit is provenance and a filter, not part of the pooling key.

A game is admitted when its match is played: its deal, seating, complete
deck/sideboard setup and ABIs must match its match. Each game stands alone. A
crash or a failed replay is an ineligible row and the deal's other games still
count; a match that fails outright is counted as a failed match. Queries trust
admitted rows.

```sh
uv run --extra evaluation python -m manabot.eval.cards audit --experiment EXPERIMENT --source .runs/cards --replay
```

`audit` re-reads every game's evidence, checks its bytes, its schedule binding
and the runtime/content identity its eval run declared, and rebuilds its rows.
`--replay` also replays each trace. Each game's evidence is a blob of its match
(gzip JSON, named by SHA256), kept in the block's `blobs.zip` and delivered with
it (`uv run etude sync`); `audit` and `export-log` read it with Etude's
`read_blobs`, which verifies every digest. Runs written before blocks kept a
file per game; `etude convert-eval RUN --source URI` turns those files into
blobs without rewriting them, and only then can they be read. Evidence contains
copy/incarnation identity, owner/controller, movement source/destination/cause,
sequence and turn, original commands and hashes, and the replay receipt. These
are private full-game artifacts, not viewer-safe public replay payloads.

## Arena logs

```sh
uv run --extra evaluation python -m manabot.eval.cards export-log --experiment EXPERIMENT --source .runs/cards --run RUN --game GAME --viewer 0 --mapping mapping.json --out game.log
```

`CardMapping` binds each exported content/definition key to a retained full
Scryfall card response with a verified `arena_id`. Unmapped cards still have
card data; export refuses missing or mismatched printings. A mapping is an
explicit authored printing choice, never a name-based join or a fabricated
Arena ID.

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

### Proposed log interchange experience — not implemented

An open concept-review proposal would let you read an exported game's viewer
logs back into the card-statistics pipeline and reproduce its GIH contributions.
Native evidence and supported Arena logs would use the same counting rules.
Logs missing the information needed to distinguish draws from other hand
acquisitions would report unavailable metrics. Retained logs could be reprocessed
after a parser improvement without rerunning the models. This proposal does not
add an import command or claim equivalence with 17Lands' downstream service.

## Scale proof

The `Card evaluation scale proof` workflow runs `verify --games 1024` on CPU CI:
bounded native games between two code players, card data for both, and an audit
that replays every retained game. It reports elapsed time and peak parent and
largest child memory; it is an engineering proof, not paid scientific execution.

Package ownership: `sim.card_evidence` retains verified native facts;
`eval.card_metrics` owns the counting rules and table normalization,
`eval.card_query` owns card data and audit, and `eval.card_page` renders the
page. `eval.arena_log` is the MTG Arena export adapter.
