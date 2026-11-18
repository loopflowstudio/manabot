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

To build in this PR, from the corpus settings direction below (not
started): today `settings` is one row per run whose `decklists` can only
say `mixed`, with a `mana_taps` column read by
`getattr(observation, "mana_taps", False)`, no `skip_trivial` and no
per-seat or per-update values.

0. Corpus settings (shape chosen in the seventh concept review, nobody
   present, reversible until merge; `scratch/concept-review.html`, questions
   47-51). A person asks one thing of a setting: was it always this value,
   in what share of games, or did it change at update N. The corpus answers
   from two relations, split by one rule: could the value differ between two
   games of the same run?

   - `runs` (today's `settings`, renamed): one row per run in the ancestry.
     Columns: experiment, run, policy_input_version, max_actions,
     action_overflow, behavior, opponent_mode, learning_seats,
     matchup_curriculum, parent_run, parent_update, parent_sha256.
     `decklists` and `mana_taps` leave it. Rename `SETTINGS_TABLE` to
     `RUN_TABLE = "training_run"`; no published run wrote the old name.
     `log_corpus_setup` loses its `disclosure` parameter, and both loops
     lose the expression that computed it.
   - `game_settings` (new): experiment, run, update_number, seat, setting,
     value, games. One row per value a setting took among the games
     completed at that update. `seat` is NULL for a setting of the whole
     game (`decklists`, `skip_trivial`) and the native seat (0 is on the
     play, as in `card_statistics`) for a per-seat one; #559's
     `mana_payment` and `mana_floats` arrive as rows with seat 0 and 1, no
     reshape. `value` is text, spelled as the native setup spells it
     (`open`, `closed`; booleans `true`/`false`).

   Writer. A game's settings are counted where its outcome is counted, so
   the two can never disagree on which games an update holds:
   `_refresh_roots` keeps each stream's settings beside
   `_self_play_decks` (decklists from the actual native setup,
   `setups[row].decklists` or the match config; `skip_trivial` read from
   the environment that plays the games, not a literal in the writer);
   `SelfPlayCounter.record` takes them with the decks and winner and counts
   the game once; `drain` returns outcomes and setting counts together;
   `log_corpus_outcomes` logs both tables (`OUTCOME_TABLE` and
   `GAME_SETTINGS_TABLE = "training_game_settings"`) at the same update.
   Pending counts then survive collector snapshots and resume supersession
   exactly as outcomes do. Both loops already call that one function.

   Reader. For each update, seat and setting, the games must sum to that
   update's completed games in `matchups`; otherwise `UnknownCoverage`. A
   run that predates the table gets derived rows only where the answer is
   certain: declared `decklists` of `open` or `closed` gives one row per
   update with that update's games. Declared `mixed`, an absent
   declaration, and `skip_trivial` on older runs give no rows; absence
   means unrecorded, and the guide says so. `declared_settings` drops
   `decklists` and `mana_taps`.

   Tests, in `tests/eval/test_training_corpus.py`, each asserting the SQL
   the guide prints: always (a fixed-disclosure run: one value per update
   covering all its games); share of games (a `mixed` deck source: open
   and closed counts that sum to the update's games); changed at update N
   (a child run started from a parent's checkpoint with the other
   disclosure: the value changes at the parent/child boundary). One writer
   test that a game whose setting counts are dropped makes the reader
   raise. Guide: replace the settings query in `docs/evaluation.md` with a
   `runs` query and the three `game_settings` queries, and state the rule
   that splits them and that seat 0 is on the play.

   If #559 is on main at sync (open at this review), write `mana_payment`
   and `mana_floats` per seat from the native setup, defaults included.
   Not in this PR: naming the difference when a checkpoint is seated under
   other levels; no admission gate; counts of a setting within one matchup
   (the counter would be keyed by matchup and settings together).

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
