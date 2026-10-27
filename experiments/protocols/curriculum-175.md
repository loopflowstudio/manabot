# MTG-175: curriculum selection on the MTG-183 training recipe

Direction updated by Jack Heart, 2026-10-26: build on MTG-183. The intended
outcome remains an actual report comparing uniform U, moderate-matchup M,
value-error E, and their equal-rank mixture ME, at **$5 inclusive per arm,
$20 total**. The separate preparatory allowance is not an arm allocation.

This amendment supersedes the older requirement to wait for the unsuccessful
MTG-174 baseline. Its failed receipts remain historical evidence. MTG-183's
merged training/evaluation implementation is the foundation; its running seed,
evaluation queue and $50 allocation belong to that Task and are not changed.
MTG-183's source protocol, recipe and operator history have been copied into
`scratch/mtg183-context/` so this checkout carries the required context.

## Common foundation

Use `experiments.elemental_long.recipe` and the recipe
`experiments/regimes/elemental-long-183-v1.json`: the ten exact Elemental-v2
decks including Lesson pools, both seats learning, width 64, two layers, four
heads, scalar masked-mean value, damped-policy learning, Adam, EMA .999,
256 streams and 64 minimum decisions per stream. Complete-game credit,
advantage filtering and absolute-iteration schedules are shared across arms.
The final source/runtime freeze must bind the actual admitted world and input
identities; never relabel the old w10 sampler proof or checkpoints as current.

Each arm starts fresh from identical seed/initial weights. MTG-183 checkpoints
are contextual references, not a warm start or a replacement for this study's
uniform control. An arm's trajectories remain fresh current-policy data; a
teacher score changes game selection, never PPO sample reuse or loss weights.

## Four treatments

Retain the detailed selection contract in `scratch/adaptive-curriculum.md`:
initial balanced coverage, protected uniform coverage and age refresh, an
eight-generation score window, shrinkage, inverse averaged ranks, and a 5%
priority cap over 100 ordered cells. U uses uniform game starts; M peaks near
1/3 and 2/3 and is zero at 0, 1/2 and 1; E uses absolute unnormalized target
residuals; ME averages normalized M/E priority distributions. Unknown M cells
retain neutral pre-cap mass. All arms record both signals.

The historical implementation collected one seat. Its native/reset and
collector integration must be ported to the current both-seat collector before
allocation. Sampling happens at terminal game boundaries, preserving unfinished
games, native failures, the existing recovery owner and complete seat trajectories.
Both seats' decisions enter learning under every treatment. Attribute each
seat's outcomes and residuals to its ordered deck role. A physical game remains
one clustered game; mirror seat rows must not inflate independent-game counts.
Retain selection probabilities, assignments, policy coordinates and deck/seat
exposure separately from learner transitions and optimizer exposures.

## Evaluation and publication

Reuse the public experiment interface in `experiments/elemental.py`:
`runs`, `eval(state)` and `report`, current `TrainingJob`, Etude eval matches,
and R2 checkpoint/table storage. Use MTG-183's heuristic-v2 identity, grid
schedule and admitted `play` path. Freeze an affordable common monitoring
cadence, final whole-deal cohort and disjoint development/final deal families
before any new scored game. Compare arm deltas on identical deals with
paired whole-deal uncertainty. Historical heuristic-v1 scores are contextual.
Reference checkpoints and rating machinery can reveal heuristic saturation;
they cannot replace the primary matched uniform-versus-adaptive comparison.

Regular checkpoint evaluations must be real Etude jobs and visible in its
report, with pending, failed, completed and missing points explicit. Use the
owned evaluation pool when available; do not run a new laptop evaluator or
displace MTG-183's evaluator. No additional paid evaluation allocation exists.
Export cadence, finalization, cost limits and report identities must survive
restart without duplicate training or evaluation jobs.

Report target score and arm-minus-U delta versus inclusive dollars, with
update/transition/exposure curves, coverage and forgetting, valid/failed game
counts, exact source/recipe/checkpoint/anchor identities, cumulative receipts
and limitations. One training seed is a bounded screen, not method-level
evidence. Publish retained result bytes through `manabot.data`, commit the
manifest and report, and retain failures rather than substituting seeds/cells.

## Remaining admission

The modern collector integration, recovery, current-world admission and actual
Etude checkpoint-to-evaluation-to-report path still need verification. Before
renting, freeze the concrete four-arm jobs, seed, update/export targets,
inclusive runtime/setup/export/shutdown reserves, evaluation cohorts and
comparison thresholds from current prices and MTG-183 throughput. Each arm's
reservation and eventual receipt must fit $5 including failures. No top-up or
duplicate job is authorized. The current amendment starts implementation; it
does not claim a paid launch, successful integration or scientific result.

## Preserved implementation

The complete pre-MTG-183 working tree is retained at
`refs/archive/mtg175-pre183` (`d4d93186269f7d9858fa1543404512ae6acc891c`): its
first parent is the old base, its tree retains tracked edits, and its third
parent retains untracked implementation/tests/notes. The current checkout was
synced through LF onto main; the obsolete payment implementation was resolved
to main's later shared payment model. Port useful sampler logic selectively;
do not restore superseded training, provider or evaluation modules wholesale.
