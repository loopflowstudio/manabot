# Etude's nouns: a review

2026-11-13. Jack Heart, after `tracked_run` raised the question "the Run
object returend by tracked_run is not a Run?": "i dont know i think we need a
higher level ontology review".

Read from Etude PR #103 at `105abdd` (class names and their docstrings). Main
at v0.3.1 has since renamed `Matchup` to `Seating` and removed `HeadToHead`.
Nothing here is decided; the proposals are mine.

## The sentence the nouns should make

> An **experiment** is a label over **jobs**. A job with a seed is a **run**.
> A run executes as **attempts** on **machines**. While it executes it writes
> **rows** on its **clocks**, keeps **state** so it can carry on, and
> publishes **checkpoints**. A checkpoint with code that plays it is a
> **player**. Players take **seats** in **matches** in an **environment**. A
> **report** shows rows and match scores.

Twelve nouns. A new user should be able to learn Etude from that paragraph.
Today the code has several classes for some of them and one name for several
different things.

## One thing, several names

| Thing | Classes today | Where |
| --- | --- | --- |
| A run | `Run` (a job and a seed), `RunRecord` (the launcher's stored row), `RunDeclaration` / `RunFile` (`run.json`, tracking's identity), `Run` (the live handle `track` returns), `TrainingRun` (as `eval(state)` sees it), `Status` | launch, run, train, experiment |
| An attempt | `RunAttempt` ("Allocation of one run to one machine for a bounded execution window"), `Attempt` ("One process that wrote to a run") | launch, run |
| A checkpoint | `CheckpointRecord` (what a run saved), `Checkpoint` (one with a stored file to evaluate), `CheckpointKey`, `CheckpointRef` (one to seat), `ArtifactRef` (where a file lies) | train, experiment, eval, run |
| A player | `Model` ("A checkpoint and how it plays"), `Code` (a heuristic), `Inference` (a seat any checkpoint fills) | eval |
| A group of runs | experiment (label), `Cohort` (launched together, shared limits), job (shared identity digest), "line" in reports ("per job or external learner line") | launch, report |
| A position in training | "coordinate", `at`, "position", `x`, clock value | train, report |

## One name, several things

| Name | Meanings |
| --- | --- |
| `Run` | the declared job-and-seed record; the live handle inside `train`. Two unrelated classes. Also "eval run" (`TournamentRun`). |
| `Record` | two base classes (`launch.records.Record`, `run.records.Record`) |
| `Storage` | a launcher record ("A named place in a bucket") and a protocol ("A flat key space") |
| state | `RunState` (`state.json`: "how a tracking attempt ended"), `ExperimentState` (what a pass found), `Status`, and now the proposed saved state. My design's `out/state/state.json` reuses a file name that already means something else. |
| model | Etude's `Model` (checkpoint plus how it plays); the author's network (`model=model` in my sketches); Jace's `ModelName` (drafter, builder, player) |
| `Session` | "One start of Etude's process on a machine"; also Loopflow's conversation |
| experiment | a label on runs; the `Experiment` class (an experiment file); in Cloud, a formal record |
| out | `run.out` (the directory that follows a run); `Output` (a named file below it); `--out` in Manabot |

## What I would propose

1. **One noun per thing, with a fixed suffix for its view.** `X` is what the
   author writes. `XRecord` is what Etude stored. Nothing else. So
   `CheckpointRecord` is the stored one and the four other checkpoint classes
   reduce to it plus a reference.
2. **Run.** `Run` stays the declared job-and-seed. The live object in `train`
   is not a second `Run`. Candidates: `TrackedRun` (pairs with
   `tracked_run`), `LiveRun` (its docstring already says "A live run"), or
   naming it for what it strictly is, one attempt at the run. The stored
   forms (`RunRecord`, `RunDeclaration`, `RunFile`, `TrainingRun`) are one
   noun seen by three readers and should converge on `RunRecord`.
3. **Attempt.** One noun. The launcher's and tracking's are the same event
   seen from outside and inside.
4. **Checkpoint and player.** "Model" is the author's word for a network and
   should stop being an Etude noun. A checkpoint is what a run published. A
   **player** is a checkpoint plus the code that plays it (today's `Model`),
   or code alone (today's `Code`). That is also the front line: a checkpoint
   becomes a player.
5. **State.** Reserve the word for what a run keeps to carry on. How an
   attempt ended is an outcome, not state.
6. **Groups.** A user needs two: experiment (the label) and job (same
   function and config, different seeds). `Cohort` is a launch-time detail
   (shared limits). "Line" in reports is a job and can say so.
7. **Position.** One word for where a clock stood; I would use "at", since
   the records already do.
8. **No stage noun.** Stages are edges between runs (`start=`, `continue`).
9. **Cloud's nouns sit above, not beside.** Question, finding and the formal
   experiment are records over these twelve; none of the twelve changes
   meaning when they arrive.

## Said so far

Jack Heart, 2026-11-13: "maybe Run and etude.run want ot change", then, of the
declared job-and-seed record, "ok, RunRequest works". So question 1 is 1e with
`RunRequest`: the live object keeps `Run`. The new name for the internal
`etude.run` package is not given.

On question 2 (Etude says "player", "model" is left to the author): "works
for me". So 2a.

Answers to the six walked-through questions (two classes for an attempt; five
for a checkpoint; the stored run's names; group nouns; the `state.json` clash;
how much renaming goes into EXP-56): "1a -- i think RunAttempt is probably
better than Attempt though", "2a", "3a", "4a, or alternatively use cohort
instead of experiment as a more general word", "5a", "6b". Carried into
`scratch/design-train-api.md` under Names. Those six were numbered as walked
through in conversation, not as the list below.

## Questions

1. The live object inside `train`:
   a. `TrackedRun`, opened by `tracked_run(...)`
   b. `LiveRun`, opened by `live_run(...)` or `open_run(...)`
   c. it is an attempt: named for that
   d. keep two classes called `Run`
   e. it keeps the name `Run`, and the declared job-and-seed record is
      renamed instead (Jack Heart: "maybe Run and etude.run want ot change").
      Candidates for the declared one: `RunRequest`, `Trial`, or no visible
      class at all (`job.seeds(0, 1, 2)`).
2. "Model" as an Etude noun:
   a. drop it; checkpoint and player only
   b. keep `Model` for checkpoint-plus-how-it-plays
3. The stored run (`RunRecord`, `RunDeclaration`, `RunFile`, `TrainingRun`):
   a. converge on one `RunRecord` in this work
   b. agree the target, converge later
   c. leave
4. Group nouns a user sees:
   a. experiment and job; cohort stays launch-only; "line" becomes job
   b. keep all four
5. Is the sentence at the top the right twelve nouns? If not, which is
   missing or should go.
6. How much of this renaming belongs in EXP-56:
   a. only what the train API touches (the live run, state, checkpoint
      folder), the rest as its own Task
   b. all of it now, before the release
   c. none; write the ontology down and rename opportunistically
