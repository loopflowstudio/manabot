# MTG-175: curriculum selection on the MTG-183 training recipe

Direction updated by Jack Heart, 2026-10-26: build on MTG-183. The intended
outcome remains an actual report comparing uniform U, moderate-matchup M,
value-error E, and their equal-rank mixture ME, at **$10 inclusive per arm,
$40 total**. The October 27 amendment below defines reuse and reservations.

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

The adaptive arms start from MTG-183's exact saved initialization; its first
uniform window supplies U as detailed below. An arm's trajectories remain fresh current-policy data; a
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
reservation and eventual receipt must fit $10 including failures. No top-up or
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

## Superseding allocation and reuse amendment — 2026-10-26/27

Jack Heart raised the cap to **$10 inclusive per arm, $40 total**, and directed
reuse of MTG-183 as an arm where possible. This amendment supersedes the $5/$20
allocation and the earlier fresh-U requirement. No paid curriculum arm had started.

**U is the first MTG-183 window**, `mtg183-elemental-long-w1`, training seed
18301, source `8e7df92e4017a0090fb1e1b3e26f75da155c3a97`. It used independent
uniform native roots and both-seat learning. Its 28 retained exports span
initialization through update 1,325 without a continuation/RNG restart. All
28 cached byte hashes were verified; the final digest agrees with its complete
generation-34 manifest. The confirmed cleanup estimate is **$7.2252361304746735**,
not an invoice. The original source ended `paused`; this is a completed data
prefix, not a claim that it reached its original 100,000-update target.
`curriculum-175-uniform.json` freezes its references and coordinates. Original
receipts remain in `.runs/mtg175-reuse-audit/`. There is no new U rental and no
transfer of U's unused allowance to another arm. Show historical attributed cost
separately from new charges; do not bill the original Task twice.

**M, E, ME** each have at most **$9.50** for one training attempt and **$0.50**
reserved for storage/observation/cleanup uncertainty. Thus new training caps
sum to $28.50, new inclusive arm limits to $30, and the reused U costs less
than its own $10 allowance. No retries, replacement seeds or automatic top-up.
All three use seed 18301, the exact saved U initialization (SHA-256
`1004ccdf46f373e833dadcb5278351a758e5328ed7c28049259c8c06be614688`), and the
frozen v1 observation input. A fresh current model is not assumed to reproduce
historical weights: the actual immutable checkpoint is a digest-checked job
input. Adam, EMA and schedules start fresh at update zero. Current saved-recipe
round trips must preserve the omitted v1 version through
`ObservationConfig.from_checkpoint`; the current fresh v2 default is not used.

The proposed jobs target **1,400 updates**, export every 50, and have 15-hour
maximum allocations with the recipe's setup/finalization reserves. Etude SDK
0.2.6 removes the obsolete twelve-hour rental cap; pin it before allocation.
The schedule remains absolute-iteration damped-policy learning. U's independent
uniform policy has no coverage/age teacher. Adaptive arms share the protected
coverage and age safeguards: the comparison therefore measures the complete
adapter, not an isolated priority term. Historical U teacher errors and
propensities unavailable in its records stay unavailable, never reconstructed
from synthetic rows. Selected assignment propensities describe the sampler's
ordered anchor/seat event; they are not the marginal probability of an unlabeled
physical game after combining reverse-role anchor events.

**Fresh evaluation deals.** Reuse MTG-183's grid grouping and replay-checked
player/evaluator, not its already observed deal outcomes. Development uses two
deals per ordered cell, each played in both model seats (400 games per point),
base 2,165,200,000 + 1,000 × cell + deal index. Held-out uses ten deals per cell,
both seats (2,000 games per point), base 2,165,300,000. Cells use `DECKS` order.
The families are disjoint from each other and MTG-183's original heuristic and
cross-checkpoint families. Monitoring is at 0, 100, 200, 400, 800, 1,200 and
actual endpoints. Evaluate initialization on the held-out family as well so
arm-minus-U improvement can be measured. Every match's physical games/paired
seats remain clustered; do not treat seat rows as independent deals.

The primary comparison remains gain over initialization at a common inclusive
cost prefix, using actual retained checkpoints, never interpolated policies or
the best score. The U endpoint supplies a conservative $7.2252361304746735
reference. Record new jobs' costs and newly visible exports prospectively;
select the last admitted checkpoint within the common bound by cost/coordinate,
without consulting evaluation scores. Retain missing cost evidence as missing
and unmatched full endpoints as secondary. The final report must distinguish
receipt estimates from invoices and disclose source/runtime maintenance since
U's training; admitted world/roster/model/input identity is necessary but is
not a claim of byte-reproducible training.

`experiments.curriculum_costs` samples exports before launcher status every
60 seconds. It immutably records each export's first-seen cost estimate plus
the full $0.50 reserve. The primary point is the highest actual noninitial
update with that inclusive estimate at or below U's reference. Late discovery
cannot backdate a cheap prefix. Selection is finalized when a run ends, then
the same held-out family evaluates that export even if it is not the full
endpoint. Runtime cost estimates may lag provider billing; retain their
`estimated` flags, observation times and final cleanup receipts. This is a
conservative sampled estimate comparison, not exact invoice matching. Absence
of an eligible observed export leaves the primary point missing.

Retain earlier predictions (M +1 point, E +3, ME +3 versus U) as speculative.
The practical threshold remains +5 percentage points without any deck mean
worsening by more than 5 points. Use paired whole-deal uncertainty, standardize
ordered cells and seats equally, show all three deltas plus ME−M−E+U, and
state that one historical-control seed cannot establish method-level effects.
Failed, invalid or missing target cells do not disappear into a strength score.
The report remains required even for negative, incomplete or inconclusive work.

Uniform tracking is an explicit imported-evidence projection, not a new training
launch. Preserve the original paused outcome, source coordinates, artifact
identity and historical cost. Its publication wall time is not training time.
Real adaptive exports must drive actual Etude evaluation requests on the owned
CPU pool. One checkpoint per bounded evaluation job keeps queue units finite;
no paid CPU rental or displacement of another Task's evaluator is selected.
