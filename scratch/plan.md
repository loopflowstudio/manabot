# MTG-353 — checkpoint compatibility and the training corpus

## Human review of 2026-11-18, fourth round — remove the world label (built; gate next)

Jack Heart, on `WORLD_VERSION` staying as a fixed label: "Why do we still need
it". On keeping a private literal where stored formats carry the field: "But
why should stored formats require it anymore? cant we just migrate old data in
the data cache?". On doing the removal in this PR or its own Task: "do it here".

Build: `WORLD_VERSION` goes away. New files stop writing a world label and
readers ignore one in old files. No stored data is rewritten (artifacts and
evaluation blocks are named by the hash of their bytes; frozen evidence keeps
its spelling).

1. Checkpoints: `current_binding_form` (`manabot/model/world.py`) drops
   `world` from a retained binding and the expected binding omits it, so old
   and new checkpoints compare equal. Remove the "trained in X, engine is Y"
   check; an earlier-world checkpoint fails at the input-version or binding
   check instead. The pinned pilots in CI must still load unchanged.
2. Replays and recordings: stop writing the label; delete the label check in
   `managym/replay.py`. The native-bytes check stays.
3. Native: remove `crate::WORLD_VERSION` and the `world_version` fields of
   the pack manifest, pool and decision capture, and the `world` context key,
   where nothing stored is compared against them. Where a stored digest or id
   does include the label, keep reading old data: prefer ignoring the field;
   keep a local string only where the label is part of a stored name. Known
   case: the matchup id `w17-authored-matrix-<hash>` in `manabot/sim/matchup.py`
   keeps its prefix as a plain string so a resumed tournament sees its old
   matchups. Record any other such case in `scratch/questions.md`.
4. Python: remove the `world=` parameter from evaluation, provenance,
   registration and experiment code where it is only compared with itself.
   Frozen experiment files that write `"world": managym.WORLD_VERSION` into a
   config write the string they always wrote.
5. `managym.WORLD_VERSION` public export: a deprecated alias that warns and
   names 0.4.0 as its removal, per the naming rule; nothing in the repository
   reads it.
6. Docs: `WORLDS.md`, `docs/engine.md`, `docs/evaluation.md`, the READMEs and
   CLAUDE.md say there is no world label; `WORLDS.md` stays as history.
7. Rebuild the extension, run the tests of the files changed (debug Rust for
   the touched crates), publish and stop at human review. Do not land.

Not in this round: policy input versions (v9/v10 retirement by distillation
and "v11 is the last number" were discussed, nothing chosen).

### Built in the implement pass of 2026-11-18 (questions 58-62)

Items 1-6 are built; item 7 is partly run (below).

- Native: `WORLD_VERSION` and the `world_version` fields of `WorldIdentity`,
  the pool, the pack manifest and the conformance manifest are gone, with the
  `world` key of the failure context. Decision capture is v6 without the label.
- Checkpoints: `_input_binding` writes no `world`; `current_binding_form`
  drops `world` and `content_manifest.world_version` from a retained binding;
  the "trained in X, engine is Y" check is deleted. A relabelled checkpoint
  loads; one with another input version or shapes is refused by those fields.
- Replays and recordings write no label and compare cards and native bytes.
- Evaluation: `runtime_fingerprints` has no `world`; `PlayerRegistration`,
  `EvaluationKey`, `SelfPlay` and `BaselineReport` keep an optional `world`
  that is omitted when absent. Matchup ids keep `SUITE_PREFIX = "w17"`.
  Frozen experiment files write `"w17"`.
- `managym.WORLD_VERSION` warns and names 0.4.0. Docs and AGENTS.md updated.

What the slice taught:

- The label sat inside three stored digests, not one: the decision capture's
  own SHA-256, a card-evaluation environment's `files` hash, and every
  registration hash. Each is read back with the label when the stored record
  has it; none is rewritten.
- A registration written now differs from a retained one of the same player
  (question 59). Whether Etude then replays a resumed tournament's matches was
  not checked.
- Academy demo fixtures compare a stored pack manifest with the live one; the
  comparison skips the old key and the fixtures were left as they are.

Checked at `47fd4a102f`, extension rebuilt (release), on a machine at load
average ~88 from other checkouts' builds:

- Pinned pilots load unchanged (`tests/model/test_pinned_pilot.py`), with
  `test_world.py`, `test_catalog.py`, `test_attack_choice.py`,
  `test_checkpoint_binding.py`, the alias test in
  `tests/env/test_named_observation_layout.py`, recording and record-writer
  tests, `test_evaluation_models.py`, `test_training_decks.py`: 94 passed.
- Evaluation, capture and verification files: 102 passed; local evaluation and
  arena match: 36 passed; Academy catalog, receipt, trigger order, demos,
  import and watch selections: 28 + 37 + 35 passed.
- Native debug: `world`, `pool`, `decision_capture` and the vector failure
  context unit tests, 8 passed. `cargo check --all-targets --features python`
  is clean. The full debug suite and the conformance replays were not run.
- `conformance/authored-match-w17-v2` receipt regenerated by its script; the
  only change is the dropped `world_version` line of its pack manifest.

Three failures left for gate, none traced to the label:

1. `tests/sim/test_puct_compatibility.py::test_registered_puct_completes_games_and_replays`:
   the audit view adds `pre_state_digest`/`post_state_digest` and the live
   rows lack them (`manabot/sim/frames.py`, `evaluation_records.py`, both
   untouched). Fails every run here; not checked at the previous commit.
2. `tests/verify/test_learning_regression.py::test_recipe_report_preserves_paired_deals_and_policy_inspection`:
   "paired arena evaluation incomplete" after ~240 s; the test gives its
   evaluation 180 s. Failed twice under load.
3. `tests/sim/test_conditional_search.py::test_frozen_fixture_payload_is_stable_and_runtime_drift_is_explicit`:
   `cap_hits` 2 against the fixture's 6.

Not run: Academy UI check/build and Playwright, hosted CI.

Realign at `dce50044ee` (pushed to #558; hosted CI just queued, nothing seen
green or red): the code backs items 1-6. The only readers of the label left
are the ones the plan names (the warning alias, the skipped keys in
`current_binding_form` and `play_demo.py`, the optional unread
`DecisionInput.world_version`); stored JSON keeps its label. `dce50044ee`
merged main (`62c3338f0d`, the Academy interactive board) after the checks
above; nothing was rerun on the merge and the extension was not rebuilt for it.

Remaining, all gate's:

1. Rebuild the extension at the merge and rerun the changed files' tests.
2. The three failures above: run each at `62c3338f0d` (main) to show it is
   already failing there, or fix it. Failure 1's test lost only its `world=`
   argument in this change.
3. Academy UI check/build and Playwright; hosted CI green on the head commit.
4. Publish to #558 and stop at human review. Do not land.

QA at `93f1f371c2` (`scratch/qa.html`): extension rebuilt at the head. Fresh
files carry no label; main's checkpoint, recorded games, incomplete local
evaluation, resumed run and old Academy traces all read. Hosted CI was red in
Integration (evaluation-long) and (experiment): two tests still expected the
removed label check. Both are repaired in the working tree, not committed or
pushed. Of the three failures above, none is in CI's file lists; the first and
third fail on main too; the second failed again under load and was not run on
main. Gate still owes: push the test fixes and see CI green.

Tenth concept review at `fa900158fc` (`scratch/concept-review.html`): the
removal lines up with the direction and the code needs no change. The two
test repairs are committed and pushed in `fa900158fc`; hosted CI on it was
pending in every job when read, nothing seen green or red. Gate owes, besides
CI green:

- Four guide sentences still say a world separates measurements, against
  `docs/evaluation.md`: `README.md` lines 199 (`EvaluationKey (world, ...)`),
  284 ("belong to the world they were measured in") and 308 ("an anchor
  version and a world"), and `docs/engine.md` line 1307 ("Engine and
  checkpoint worlds must remain compatible"). Rewrite them to say revision,
  decks and the input contract, linking the evaluation guide's comparison
  section. Sentences naming a world as a fact about a past run (w13) stay.

Question 59, reasoned and not run: a registration hash already covers
`source_sha256` (code players) and `writer_revision`/`stored_programs`
(checkpoints), so a tournament resumed at another revision already held
different registrations; the missing label adds one more differing field.
Chosen for review: accept it (question 63).

Larger than this PR, on the page only: "world" still names the one card pool
(`ModelConfig.world`, `WORLD_SETUP`), the installed cards' identity
(`world_identity`, `world_binding`) and the old label.

For the person at review, not decided: question 59. A registration written
now has no `world`, so it differs from the one a retained tournament holds
for the same player; whether Etude then replays a resumed tournament's
matches is unchecked.

## Human review of 2026-11-18, third round — items 1 and 4 built, 5 next

Built (questions 54-56): `training_corpus(file, source=storage)` selects the
run and update a `.safetensors` policy records; new files record
`source.etude_experiment`, older ones take `experiment=`; a file missing a
record raises `UnknownCoverage` naming it. Outcome rows are written with
`seat` (0 on the play) and `outcome_rows` in `manabot/reports/update_metrics.py`
reads scalar, `position` and `seat` forms into the same rows. Checked:
`tests/eval/test_training_corpus.py` (all three stored forms; file form on a
real parent and child run), `tests/training/test_self_play.py`,
`tests/training/test_outcome_tables.py`, the checkpoint-source test in
`tests/model/test_world.py`: 18 passed. Not run: the guide's file example
against a real `etude train` output directory (QA).

Realign at `feb228178b`: the code backs items 1 and 4; one leftover
"deck/opponent/position counts" in `docs/experiment-metrics.md` now says
seat. `SelfPlayPoint.position` stays as a warning alias until 0.4.0. Hosted CI
on `feb228178b` was pending, every job including Academy UI; nothing seen
green or red yet.

QA at `6fa4efee06` (`scratch/qa.html`, question 57): the guide's file example
ran on a fresh resumed run, a child, a mixed run and from an installed wheel,
with the counts of the run form. Repaired in the working tree, not pushed: the
guide's first example named an undefined `etude_home` and a path nobody has;
a string path crashed; a storage without the run said "resolved training
decks are unknown". Hosted CI at that commit: 21 passed, none failed, Academy
UI, Engine Budget and Integration (evaluation) still pending.

Ninth concept review at `225d2d4212` (`scratch/concept-review.html`): nothing
to change. "This checkpoint" and "which seat" now each have one owner. Kept as
counterexamples, not work: the corpus reader trusts a file's header for its
run and update and never compares bytes with a registered checkpoint (the raw
policy is not registered); the evaluation card page still filters by
`position` play/draw, outside this PR. Run in that pass:
`tests/eval/test_training_corpus.py`, `tests/training/test_outcome_tables.py`,
`tests/training/test_self_play.py`, 17 passed. Hosted CI at that commit: 2
passed, 17 pending, none failed when read.

Jack Heart answered `scratch/concept-review.html` (the review written with a
person; the graph's eighth review is in git at 309ed9febe): "1a 2a 3a 4 c:
fix it in this PR 5b". The questions and what those answers select:

1. Reaching the corpus from a checkpoint file: (a) build it in this PR.
   `training_corpus` accepts a policy checkpoint file with `source=storage`
   and selects the corpus from the run and update the file records. A file
   that records neither raises, naming what is missing. Etude's reader needs
   the experiment name as well as the run; if the file does not record it,
   new files record it beside the run they already write, and an older file
   takes the experiment as an argument. The guide's first example uses the
   file form.
2. One PR or two: (a) keep #558 as one.
3. A setting within one matchup: (a) leave for later. No change.
4. Seat spelling: "c: fix it in this PR". One spelling for which seat a row
   is about, across the outcome rows, `corpus`, `game_settings` and
   `card_statistics`: the native seat, 0 on the play. New runs write outcome
   rows that way; runs already stored keep their bytes and are read through
   the one reader into the same columns. Choose the column names once, say
   them in the guide, and record the choice in `scratch/questions.md`.
5. The red Academy UI job at the head commit: (b) the Task goes back through
   its gate. Diagnose and fix the failure, or show it is unrelated and
   already failing on main, before publishing.

Publish to PR #558 and stop at human review; do not land. No paid run.

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
  is gone from new files (fourth round, above); stored files keep theirs. The dedicated strength command
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
Realign ran the guide's `runs` query and three `game_settings` queries on a
fresh `etude train manabot/recipes/local.py --until updates=2`: four games,
`decklists` `open` and `skip_trivial` `true` in all four (share 1.0), both
holding over updates 1-2. Its excerpt of the self-play loop in
`docs/training.md` still passed `disclosure=`; removed. Not run: the guide's
blocks against frontier Small, which exercises the derived `decklists` rows
for a run that predates the table (QA). #559 was still open at realign; `mana_payment` and
`mana_floats` per seat are added as `GameSetting(seat, ...)` rows in
`_refresh_roots` by whichever PR merges second.

QA at `90df80d245` (`scratch/qa.html`): the guide's blocks ran as written on a
fresh resumed run (checkout and installed wheel), a closed child of it, a
per-game mixed run (also interrupted and resumed) and the minimal loop;
setting counts summed to games at every update. Frontier Small has no setting
rows (it declared no disclosure). CI's training job failed on that commit:
three scripted test environments did not state `skip_trivial`, which the
collector now reads. Each states it now (uncommitted at the end of QA; the
hosted job has not been seen green). Left as found: runs written by earlier
commits of this branch read with NULL `runs` columns and no parent; three
`test_game_errors.py` replay tests fail on main too and CI does not run that
file.

Eighth concept review at `335eab23d8` (`scratch/concept-review.html`): nothing
to change. `runs` and `game_settings` are built as chosen in questions 47-53
and each product concept has one owner in the code. Kept as counterexamples,
not work: a single run resumed at a revision that adds a setting raises for
the whole corpus (question 52, strict on purpose); a run whose opponent is not
self or learning now logs no outcome table and reads as `UnknownCoverage`
where it used to read as zero games. Run in that pass:
`tests/eval/test_training_corpus.py`, `tests/training/test_self_play.py`,
`tests/training/test_outcome_tables.py`, 16 passed. Hosted CI on `335eab23d8`
was pending; gate owns seeing it green.

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
