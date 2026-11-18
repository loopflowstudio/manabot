# MTG-353 — checkpoint compatibility and the training corpus

## Outcome

A checkpoint can enter a new evaluation as cards, names and input columns
are added. The input contract determines whether it can play. Its training
corpus determines whether a measured matchup was encountered in training.
The evaluation guide defines when measurements can be lined up.

## Built and verified

- The loader binds learned rows by name, projects appended input columns and
  reads stored programs for every trained definition. Every change in how a
  trained card serializes is treated as a respelling, including a rules change.
  Writer catalogs, token renames, pinned pilots and Academy refusal reasons
  cover compatibility. Exact decks recorded in a binding still compare.
- The evaluation participant records its writer revision and stored program
  count. Historical registrations retain their hashes. The world number
  survives as a serialized historical label. The dedicated strength command
  is gone. The retained frontier evaluation used ordinary admission and both
  arms reproduced their cohorts. The original measurement covers 160 games.
- The docs cover measurements within a revision and across revisions, early
  learning on the raw learner, throughput on the same machine and corpus
  membership. Current docs, refusal messages and code comments describe
  the recorded facts directly. Public API and evidence keys keep their spelling.

## Training corpus verification, 2026-11-18

The collector records each completed self or learning-opponent game twice,
once per seat. The current self-play loop writes `self_play_outcomes` at
each successful non-random update. The minimal loop has no outcome writer.
The frontier executor at `ab2b757d9` drained these counters for self-play
into scalar columns. Absence of outcome evidence leaves coverage unknown.

Etude's training reader exposes the new rows with
`RunData.read_table("self_play_outcomes", "updates")`. The existing offline
projection `updates_from_run` reads historical scalars as well as current
tables, with resumed attempts superseding earlier writes. Its separate,
short-lived `self_play_outcomes/table` reference format still requires the
revision named by its error.

Checked the retained declaration and the first sealed metrics block (updates
0–12) of frontier Small, `tla_frontier/59718fe34945`, from R2. Update 1 had
514 nonempty seat/matchup outcome rows, or 259 games counting the play seat.
All 132 observed deck identities joined by content to the 134 entries in
`training.regime.deck_source.decks` in the declaration. All 262 card names
in that roster resolve through the writer naming catalog to definition keys.

## Corpus query delivered, 2026-11-18

`manabot.eval.training_corpus.training_corpus` takes an Etude checkpoint
record plus storage, or a RunData plus an inclusive update. The `corpus`
view joins each ordered matchup to both decks' definition keys. Missing
update evidence raises UnknownCoverage. New self-play runs retain their
resolved decks at update zero because current recipe arguments need not
include them. Frontier's historical declaration supplies its own roster.

Ran all four SQL questions from the evaluation guide against the local
two-update recipe (`mtg353-training-corpus`) and the frontier Small R2
declaration, writer record and sealed block for updates 0–12. Local had
four games with Island, one ordered matchup and two seats of each deck
per update. Frontier through update 1 had 259 games, 218 with Island;
132 distinct decks occupied 518 seats, from 134 roster entries and 262
distinct definition keys. The ordered matchup starting `7b246825` on
the play and `a3d4b7c3` on the draw occurred twice. For a proposed deck
containing Island and Lightning Bolt, only Lightning Bolt was new in both
runs. DuckDB plans used hash joins on the two deck identities.

`training_corpus` returns the DuckDB connection with `corpus` as a view on
it; callers read every relation with `corpus.sql(...)`
(`scratch/questions.md`, question 43). `decks` and `matchups` stay on the
connection undocumented.

## Sync onto main, 2026-11-18

Merged main through #555 (opponent decision memory) at `945899d893`. Fresh
and rebound decision-memory catalog tables use the same observation capacity
switch (question 44). After the merge and extension rebuild,
`tests/eval/test_training_corpus.py`, `tests/model/test_checkpoint_binding.py`,
`tests/model/test_pinned_pilot.py` and `tests/model/test_writer_catalog.py`
gave 12 passed, 1 skipped. Nothing wider was run locally; CI owns the suite.

## Remaining work

The sixth concept review's leftover (question 46) is done: `roster` and
`CorpusDeck` are deleted, `deck_rows` is the one deck reader, and the guide
prints its deck-entry and card counts from `deck_cards` (`0a772149d7`). QA at
`1e4ebccb3e` ran the guide's blocks as written: frontier Small gave 134 and
262, the local run 2 and 29, from the checkout and from installed wheels
(`scratch/qa.html`).

Corpus settings (item 0 of the seventh concept review; questions 47-51) are
built. `runs` replaces `settings` without `decklists` or `mana_taps`
(`RUN_TABLE = "training_run"`); `game_settings` holds (update_number, seat,
setting, value, games) for `decklists` and `skip_trivial`.
`SelfPlayCounter.record` counts a game's settings with its outcome, from
`_game_settings` kept in `_refresh_roots`; `drain` returns `CompletedGames`
(outcomes and settings) and `log_corpus_outcomes` logs both tables. The
reader raises `UnknownCoverage` when any recorded seat and setting does not
sum to an update's games, and derives `decklists` rows for older runs that
declared `open` or `closed`.

What the slice taught:

- `log_table` indexes a table's digest as a scalar of the same name, so the
  per-update `training_game_settings` digest had to be dropped in
  `manabot.reports.update_metrics.scalar_updates` with the other corpus
  tables, or it would read as a metric.
- `skip_trivial` was a literal in three constructors. The collector now reads
  `env.skip_trivial`; `ShardedVectorEnv` and `TargetPracticeVector` state it
  as an attribute (`CapturedVector` already did).
- A run with outcome tables and no setting table at all reads as a run that
  predates the table: no rows, no error. Only a run that recorded a setting
  somewhere is held to its sum at every update (question 52).
- A child may start from a parent with the other disclosure; nothing gates it.

Checked: `tests/eval/test_training_corpus.py` (always, share on a real mixed
collector, changed at the parent/child boundary through the real loop, and
dropped counts raising), `tests/training/test_self_play.py`,
`test_outcome_tables.py`, `test_minimal_loop.py`, `test_self_play_loop.py`.
The guide's three `game_settings` queries are the SQL those tests assert.
Not run here: the guide's blocks against `etude train manabot/recipes/local.py`
and frontier Small (QA). #559 was not on main; `mana_payment` and
`mana_floats` per seat are added as `GameSetting(seat, ...)` rows in
`_refresh_roots` by whichever PR merges second.

Not in this PR: naming the difference when a checkpoint is seated under other
levels; an admission gate; counts of a setting within one matchup.

1. Etude follow-up, not blocking: its evaluation query database indexes
   `training_checkpoints` only (question 34). Once Etude ingests selected
   training rows, the manabot relation can read them there. File as an Etude
   Task; this PR does not wait for it.
2. File after this PR: publish statistics rows as Etude run artifacts with
   the attempt in their context, so a checkpoint plus storage reaches them
   and supersession applies (question 38).
3. Gate and publish for human review. No landing, paid run, corpus label in
   the checkpoint or change to exact-deck refusal.

## Training settings, deck contents and parent checkpoints (built)

Both loops log resolved deck entries (board, copies and definition key) and
decision settings at update zero and call the same outcome writer. Continued
runs retain the original setup. The reader exposes settings and deck_cards,
and follows a published parent by run, update and SHA-256. Historical rosters
and their outcomes still read as before.

The local parent trained through update 2. Its child started from update 1
and trained one more update. The four membership queries and the settings
query found two games from each run, with separate update-1 coordinates and
v11, current-self, both-seat settings for each. The minimal loop's update 1
had two mirror games, four deck seats and the same decision settings. Both
examples reported Lightning Bolt as new in the proposed two-card list.

## Card statistics in the corpus (built)

Shape chosen in the concept review of 2026-11-18 with nobody present;
reversible (`scratch/concept-review.html`, questions 37 and 39).

`training_corpus(..., statistics=path)` adds `card_statistics` to the same
DuckDB connection: terminal-game sums per update, native seat and definition
key, through the cutoff, mapped through the digest-selected writer catalog
by native definition index (Clue and Ally have no printed-name join). No deck
column: rows hold main-deck copies only. A path with no rows, or fewer
terminal statistics games than corpus games at an update, raises
`UnknownCoverage`; more is allowed. A corpus that follows a parent checkpoint
refuses a single statistics path. Docs paragraph and query are in the corpus
section of `docs/evaluation.md`, including that resumed runs keep superseded
attempts' rows.

Alignment, from the two-update local run with statistics: two terminal games
at capture coordinate 0 and two at 1, plus two nonterminal rows on close;
the outcome table held two games each at updates 1 and 2. The capture
coordinate is one behind the outcome update and the reader maps it. The
combined query found four list games with Island, nine draws and 72 choices.

## Review direction

Direction, Jack Heart, 2026-11-18:

- On the review's finding that "world-bound / deck-source / world pilot" and
  "setup-bound / fixed-match" are five names for two things: "I think this
  needs to be inthis PR".
- "I think we need a notion of like a \"training corpus\". I agree that world
  bound and esuch are wrong."
- "I dont know that its kind, becasue it could be lots of things".
- "ideally it would be (a) automatically derived from how trianing is
  configured (b) queryable in some way that you can then write specific Qs
  about on top of it".
- Offered the words in this PR and the queryable view as a separate Task:
  "no, send back 353 with item 2".

Direction, Jack Heart, 2026-11-18:

- "i feel like we may be layering too kmuch in at at time. I am not sure i
  want a \"manabot recheck\" command. Mostly Etude's report framework is the
  right way to line up data against previous data. The job here is just to
  define when its valid, and also to make it possible to use a previous
  checkpoint in new evals in a way that is safe even when minor evolutions
  have happened"
- "I thought the prpoosal was also getting rid of the world number."
- On a trained card whose engine serialization changed: "i would say we just
  always assume respelling", and "honestly, rules change still probably doesnt
  affect that muc about what the agent thinks".
- "Learning speed and throughput are even more important ... its the early
  part of the curve im more focused on".
- Asked whether to write this into the plan and send the Task back to
  build: "yes".

Direction on the corpus settings, 2026-11-18, relayed from the MTG-385
sessions (PR #559; full text in `.lf/prompts/20261118-113616-*-steer.md`):

- Jack Heart: "also negotiate eith mtg-353 to make sure that this kind of
  change is the kind of thing that is clear to read on how a model was
  trained".
- Jack Heart: "I wonder if we want something for like: - this value was
  always true in its training - this value was true n% of the time - this
  value was true for the first N games and then not true after that", and
  "with the mana payment stuff i can imagine turning on increasingly higher
  levels of mana payment options".
- Agreed between the sessions: the corpus settings (`log_corpus_setup`,
  `SETTINGS_TABLE`) record action-space conditions "as facts and never as
  load gates"; "the settings relation records seat, mana_payment,
  mana_floats, skip_trivial explicitly at defaults. The second PR to merge
  adapts log_corpus_setup at sync." The per-update game-count reading is
  marked "chosen by the MTG-353 session, reversible until merge".
