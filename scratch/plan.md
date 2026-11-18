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
record plus storage, or a RunData plus an inclusive update. The DuckDB
relation joins each ordered matchup to both decks' definition keys. Missing
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

## Remaining work

1. Etude follow-up, not blocking: its evaluation query database indexes
   `training_checkpoints` only (question 34). Once Etude ingests selected
   training rows, the manabot relation can read them there. File as an Etude
   Task; this PR does not wait for it.
2. Card statistics relation, not built (steer, Jack Heart, 2026-11-18:
   "didnt this land?", "i do think we should play well with it"). Old runs:
   "Not a problem"; value-estimate trustworthiness: "agreed, leave this
   out". Design below.
3. Gate and publish for human review after the query is built. No landing,
   paid run, corpus label in the checkpoint or change to exact-deck refusal.

## Card statistics in the corpus (next build)

Shape chosen in the concept review of 2026-11-18 with nobody present;
reversible. Page: `scratch/concept-review.html`.

A run declared with `statistics: true` writes one `GameStatistics` row per
game to `out/statistics/*.jsonl` (`managym.statistics.read_statistics`).
Each row carries the seed, terminal flag, winner, per-seat card counts by
printed name and native definition index, the `card_digest` binding those
indices, and a context holding the update. Rows hold main-deck copies only,
so they cannot name a deck. Etude run storage does not carry the directory.

Build in `manabot/eval/training_corpus.py`:

- `training_corpus(run, through, *, source=None, statistics: Path | None =
  None)`. With a path, the relation's connection also holds
  `card_statistics`: `update_number, seat, card, games, wins, deck_copies,
  opening_copies, drawn_copies, opportunities, selections, resolutions`,
  summed over terminal games per update, native seat (0 is on the play) and
  definition key, for updates 1 through the cutoff. `corpus` is unchanged.
- Names become definition keys through the catalog each row's `card_digest`
  names (`writer_catalog` or `archived_catalog`), with the mapping
  `corpus_decks` already uses; share it, do not copy it.
- Without the keyword there is no `card_statistics` relation. A supplied
  path with no rows raises `UnknownCoverage`. An update through the cutoff
  where statistics hold fewer terminal games than the corpus raises
  `UnknownCoverage` naming both counts. More is allowed.
- No deck column, no new counter, no game record, no read of `run.out`
  from library code, no change to what training writes.

First slice, because it shows soonest whether the design holds: run the
local two-update recipe with `statistics: true`, and compare terminal
statistics games per context update with corpus games per update. The
writer labels a row with the finished-update count when the game ended
(`CaptureScope` context in `manabot/training/loops/self_play.py`); if that
is one behind the outcome table's update, map it in the reader and state
the rule in a comment. If the two cannot be aligned, stop and record what
was seen in `scratch/questions.md` instead of guessing.

Then: one test in `tests/eval/test_training_corpus.py` over a real short
run or its retained rows (join works; missing directory raises; fewer games
raises), and a paragraph plus one query in the corpus section of
`docs/evaluation.md`. State there that a resumed run's statistics keep rows
from superseded attempts, so card counts can exceed the corpus.

Not this PR, to file: publish statistics rows as run artifacts with the
attempt in their context, so a checkpoint plus storage reaches them and
supersession applies.

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
