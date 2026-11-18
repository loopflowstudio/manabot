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

## Remaining work

1. **Corpus query in manabot, on Etude's public training reader.** The
   installed SDK already selects the rows: `RunData.read_table(
   "self_play_outcomes", "updates")` for current runs and
   `manabot/reports/update_metrics.py::updates_from_run` for historical
   scalars, both honouring resume supersession. Add the entry point beside
   the evaluation queries: accept a checkpoint or run and update, select
   recorded matchups through that update, join their decks to card keys
   through the run's declared deck source and the writer naming catalog, and
   expose the result as a DuckDB relation so specific questions are SQL on
   top. Keep per-update counts for deck-mix questions. Absent outcome
   evidence reports coverage unknown, never empty.
2. Test and demonstrate the four queries against a local run and frontier
   Small (`tla_frontier/59718fe34945`, R2): matchup membership, new cards in
   a deck, games including a card by an update (count a game once, not per
   seat), and deck mix over time. Replace the guide's quoted update-1 counts
   with the executable query. Expected update-1 result: 259 games, 132 of
   134 roster decks observed, 262 card names resolved.
3. Etude follow-up, not blocking: its evaluation query database indexes
   `training_checkpoints` only (question 34). Once Etude ingests selected
   training rows, the manabot relation can read them there. File as an Etude
   Task; this PR does not wait for it.
4. Gate and publish for human review after the query is built. No landing,
   paid run, corpus label in the checkpoint or change to exact-deck refusal.

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
