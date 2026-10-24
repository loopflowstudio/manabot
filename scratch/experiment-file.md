# Experiment file: train, eval, report

Working design for the production experiment API behind MTG-180's
`etude report EXPERIMENT`. Nothing here is implemented. The prototype in
`experiments/report_prototype/` settles how the report looks; this note covers
how an experiment declares its work.

## Accepted (Jack Heart, 2026-10-24)

- One experiment `.py` file defines `train()`, `eval()` and `report()`.
- `train()` and `eval()` return jobs. `report()` returns a Python `Report` object.
- `train()` stays close to Etude's existing `runs()` job files
  (`launch/job_file.py`, `training(...)`).
- No controller for now. The base APIs must be enough that a person, a loop or a
  generic LLM agent can act as a controller or write one.
- Eval must cover more than checkpoint vs heuristic: policy vs policy, and
  checkpoints rated against each other (MTG-183).
- Evals run as the experiment goes: whatever has become playable gets played.
- Jack Heart accepted the following as the starting shape: `eval()` declares the
  matchups it wants given current state, and a repeatable pass plays what's new.

## Starting shape (accepted to start from; details are proposals)

```python
def eval(state: ExperimentState) -> list[Matchup]:
    """Every matchup the design wants, given what exists now. Called repeatedly."""
```

- **Matchup**: two agents (a checkpoint, or code such as the heuristic) under a
  protocol (deals, games, seat rule), plus tags such as `consecutive` or
  `held-out`. Built on Etude's `Agent` / `Matchup(players, conditions, games)`.
- **ExperimentState** (proposed): the experiment's jobs, their checkpoints with
  clocks (update, learner decisions, hours), whether training has ended, and the
  played and pending matchups.
- **One pass** (proposed `etude eval FILE`, with `--watch` to repeat): call
  `eval(state)`, subtract matchups already played or pending (keyed by agent
  digests and protocol), and launch the rest as evaluation jobs.
- **Failures** stay recorded and aren't replayed unless the file asks for it,
  matching MTG-183's "counted and shown, never replaced".
- **Restart** replays nothing that was already played.
- **`report()`** derives every view (score vs heuristic, head-to-head matrices,
  ratings) from one table of played matches. Tags decide which matches enter a
  fit, so held-out pairs stay out of it.

Covered cases: MTG-183's heuristic grid plus consecutive/reference/endpoint/
held-out pairing, the seat comparison's every-checkpoint-vs-heuristic evals, and
MTG-181's ten-deck grid.

## Etude, first slice (local commit 0d5a671 on `jack-heart/experiment-eval`)

Etude's `Benchmark` already reuses matches by key and its batch hold plays only
undelivered matches. The gap was that every matchup seated the held model. Now
a matchup of two fixed agents (`Model(...).agent(ref)` for a chosen checkpoint)
is scheduled once, so `eval(state)` can return a `Benchmark` and a batch hold is
the pass. On-the-go holds skip those matchups. Not yet: the experiment-file
loader and `ExperimentState`, launched (rented) passes, tags for held-out pairs.

## Etude, second slice (local commit d6f6cbf)

Jack Heart (2026-10-24): this direction supersedes the "declare evaluations on
the training job, every=5, controller launches" plan in Etude's
docs/evaluation.md (now replaced); open to further experimenting. Every-Nth
checkpoint must be easy, and the API must stretch to designs beyond fixed
baselines, post-hoc evals and evals improvised during training.

Built: `etude.experiment` (`ExperimentState`, `TrainingRun`, `Checkpoints` with
`every`/`at`/`tagged`/`final`, `observe`, `advance`), `etude eval FILE
[--pool|--rent] [--watch]`, `evaluation(..., checkpoints=...)`, launched passes
with checkpoint inputs from the training run's folder and an in-flight ledger
(failed launched pass stops the watch), `hold(checkpoints=..., matches=...)`,
`unplayed`, `Matchup.tags` into `schedule.tags`. Post-hoc and improvised evals
need nothing extra: a new matchup or benchmark is played for every chosen
checkpoint, finished runs included, and matches are reused across tournaments.

## Report and evals (proposal, not built)

`report()` reads the one match store (`connect(source)`), never a job or pass.
It names what it shows by benchmark and matchup (or tags), ideally by importing
the same `Benchmark` objects `eval` builds, so names have one owner. Who played
a match (training-time hold, pass, launched pass, post-hoc sweep) never shows.
Prototype panels map: Curve → `curves(benchmark)`; Matrix → `compare` split by
conditions; HeadToHead / Ranking → `schedule` rows of fixed-agent matchups by tag.

## Open

- Where the played-match table and the matchup identity key live in Etude, and
  whether existing `etude.reports.Evaluation` link records already serve as it.
- Whether a matchup becomes one evaluation job or matchups are batched into jobs
  (cost and rental granularity).
- How `train()` exposes the job names and seeds the report labels with
  (`RunDeclaration.job_name` today is display-only).
- Budget limits for a pass, since `--watch` can spend money without a person.

## Next proof

Write MTG-183's design and the seat comparison as `eval()` functions against a
fake `ExperimentState`, and check that a pass launches exactly the expected new
matchups as checkpoints arrive and training ends, with no paid runs.
