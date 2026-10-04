# Executable training regimes

A `TrainingRegime` is an ordered recipe; a `TrainingRun` is one bounded execution
with resolved settings, named seed streams, artifacts, measured costs and status.
ETU-89 owns this infrastructure. ETU-90 owns its direct-RL treatment correctness;
ETU-91 owns the comparison/ablation protocols and analysis.

```bash
uv run manabot train --regime experiments/regimes/direct-self-play.json --seed 197 --out .runs/direct-pilot
uv run manabot train --regime experiments/regimes/search-distillation.json --seed 197 --out .runs/search-pilot
```

The supplied recipes are bounded pilots, not the proposed week-long experiment.
The ordinary preset command remains available. Combining `--regime` with a
preset or `--set` fails rather than silently ignoring overrides. Choose a new
output directory for every attempt; checkpoints are not process-resume files.

Policy stages are `collect_search`, `train_supervised`, and `train_self_play`.
A supervised stage's `datasets` name earlier collection outputs. Its `initial`
names an earlier supervised stage, carrying learner and Adam state; each source
game keeps its identity and validation assignment as the corpus grows. Game IDs
are unique within the run, and IDs divisible by ten form validation. At least
one complete game must exist in each partition. Self-play `initial` carries the
live collector/learner/optimizer within the same execution; streams pause at the
exact update boundary. External checkpoint reuse is not supported as implicit
resume. Unsupported operations, forward references, and incompatible runtime
worlds fail validation.

`collect_belief` freezes a named earlier policy stage's `raw` or `ema` artifact
and collects complete self-play games with private hidden-hand labels.
Compound policy stages support downstream belief collection from `raw` only;
they do not export EMA. Sampler stages do not update the compound policy.
`train_belief` fits a constrained autoregressive sampler from that collection's
immutable whole-game train/validation/test splits. These stages preserve the
last-complete-raw policy selection; sampler artifacts are separately admitted
and do not turn an observation-only policy into a belief-enabled player.
The [belief sampler guide](belief-sampler.md) describes the bounded pilot,
physical-deal baseline, history-dropout treatment and scientific limits.

The recipe reuses `AgentHypers`, `ObservationSpaceHypers`, and authored
`MatchHypers`, including both sideboards. Architecture and match binding live
once per regime. Every stage declares its CPU device, float32 precision, one
worker, thread count, wall and memory limits. Learning settings independently
select estimators, filtering, regularization reference, schedules, and EMA.
Schedules use `run_elapsed_budget`: elapsed wall time since run creation divided
by the regime's total `wall_seconds`, including generation, collection, updates
and export. Continuation stages do not reset that clock. Stage deadlines remain
independent bounds inside the same total allowance. Evaluation is outside the
training execution and does not consume this clock.

The policy stays on the acting viewer's tensor inputs; private teacher metadata
is not a model input. Learner and behavior weights are raw; EMA is a separately
named evaluation output with its iteration clock, never a hidden behavior swap.

`VerifyStore` is the canonical SQLite owner for runs and stages. `run.json` is
an export of committed state. Files are atomically published before their
SHA-256 identities enter stage records. Each checkpoint uses the ordinary
writer and reload path, including ETU-75's mandatory world/setup binding.
Selected-match recipes explicitly select `semantic_pack="ur-lessons-vs-gw-allies"`;
complete `semantic_cards` and `known_hand` tensors survive collection and training.
EMA uses ETU-90's complete-state helper once per iteration, including filtered
skips, and preserves its clock across continuation.
A failed or interrupted stage retains its error, elapsed cost and previously
published artifacts. The final selection rule is the last complete raw output,
not the best seed or best observed evaluation score.

Collection, optimization and export time are separate from evaluation time.
Completed stages freeze `cumulative_seconds` from run start after artifact
admission and before persisting completion. This includes setup and all prior
persistence overhead. Use this observed cost for checkpoint cutoffs; later
overhead never changes an earlier checkpoint's cost. Stages without completed
artifact admission and older records retain `null`, not an estimated cost.
Records distinguish complete games, environment decisions, learner transitions,
and optimizer exposures. Runtime/source/content/tensor identities and hardware
are saved at creation. The reproducibility contract is resolved settings and
artifact provenance, not byte-identical stochastic training or exact process
continuation. A fresh execution owns fresh costs; no pre-existing corpus is free.

[Learning-speed protocol](../experiments/training-regimes.md) and
[ablation protocol](../experiments/ataraxos-mtg-ablations.md) describe the bounded
smokes and the separately proposed scientific studies.

## Integrated bounded proof — 2026-10-04

ETU-89 integrates ETU-75 provider head `aba61859`, ETU-90 complete-state EMA
and collector match binding, and ETU-91 semantic-pack recipes. Two retained
executions in `.runs/etu89-integrated-world-{1,2}` completed two self-play
stages each, with 1,024 learner transitions each and ordinary reloads of both
raw and EMA outputs at both stages. The second execution used the rebuilt
native extension and completed 14 games in 4.63 seconds. These are pipeline
proofs, not strength or comparative-learning results. The local SQLite and
run manifests retain exact artifact digests and source/runtime identities.

The integrated gate passed 66 tests with one study test skipped; six native
vector tests passed in debug, and scoped Ruff and diff checks passed. ETU-90
owns final RL acceptance and ETU-91 owns final replayed study/notebook evidence.
ETU-75 remains open for its unmet empirical evidence; shared delivery does not
close those claims. No scientific experiment ran in this infrastructure pass.

The separate `ataraxos_move` gradient selector follows the final supplement's
move recipe, including categorical outcome targets and iteration schedules.
See [its source fidelity table and bounded commands](ataraxos.md). Existing PPO
recipes remain controls; the new recipe does not alter a frozen study.

## Compound decisions

`AgentHypers.compound_decisions=true` selects the learned ragged policy, and
`train_compound` collects complete self-play games before each optimizer update.
Use the four `compound-{sequential,grouped}-{bootstrap,outcome}` recipes. These
are separate from the frozen ETU-91 campaign. Their defaults are bounded proof
recipes, not a scientific allocation.

managym's `compound_offers` supplies a complete legal surface. Its existing
atomic attacker declarations and non-kicked, single CreatureOrPlayer casts
are represented jointly; other actions remain action-aligned. In particular,
blocker assignments, kicker/payment, ward, opponent priority, and newly revealed
choices are **new observations**, never inferred continuations. A native
`compound_commands_json` call lowers the sampled submission on an exact fork
into ordinary revision-bound Commands. Sampling and lowering leave the live
root untouched. The ordinary checkpoint player and Etude villain execute the
resulting suffix without asking the model to reconsider it. Interruption rejects
and clears a stale suffix.

The model reuses the viewer-safe object/semantic encoder. Complete priority and
fallback offer rows also use the existing action/focus embeddings. A small GRU
conditions candidate include/exclude logits on the selected prefix, public
candidate labels, and position/count features. Physical IDs route submissions;
they are not learned embeddings. Candidate-specific runtime object binding is
still limited to the native public labels in set-valued choices; this prototype
does not establish optimal representation or strategic strength. The decoder
handles arbitrary candidate counts, but the configured observation capacities
still fail closed rather than truncating game objects.

For each unordered choice, candidates are visited once in native order. Minimum
and maximum cardinalities mask illegal next tokens. This gives each subset one
unique encoding. Joint log probability is the **sum of normalized conditional
log probabilities**. Dynamic dependencies and ordered selections fail closed.
A forced token has probability one and log probability zero. Conditional entropy
and conditional-uniform regularization at sampled prefixes are not exact joint
entropy or a uniform distribution over complete subsets.

`grouping=grouped` uses one joint PPO ratio and root value per submission;
`sequential` uses per-factor ratios and prefix values with the same model and
sampled complete submissions. It is a decoder-credit ablation, not the older
flat-policy baseline. Both keep weights frozen through complete games, separate
seat trajectories, assign only terminal ±1/0 rewards, and never optimize an
incomplete game or partial declaration. `estimator=outcome` uses terminal Monte
Carlo returns; `bootstrapped` uses ETU-90's transition-end GAE with separate
policy/value lambdas. Targets and behavior distributions are detached. Gradients
flow through recomputed normalized logits, recurrent prefixes, and the shared
viewer encoder. Uniform-reference and collection reverse-KL terms use retained
behavior prefixes; summed conditional terms are sampled-prefix regularizers,
not exact joint reverse KL. Raw weights only; unsupported EMA and action-type
reference settings are rejected.

Discount and trace clocks count credit units: submissions for grouped credit,
all decoder factors (including forced factors) for sequential credit. The
comparison recipes fix gamma=1 to preserve the undiscounted outcome objective;
lambda remains an explicit treatment. Changing gamma changes time preference
under grouping and must be reported as an additional confound.

Each stage retains private JSONL Command/transition evidence, exact replay,
whole-game boundaries, world/setup/source identities, optimizer state, and the
ordinary admitted checkpoint. Continuation preserves Adam and cannot branch
from older weights; checkpoints do not promise process resume. Interrupted
JSONL evidence remains incomplete and cannot supply a terminal target.

Accounting distinguishes native microchoices, grouped decisions, decoder
factors, forced factors, native optionless auto-resolution, completed games,
optimizer exposures, collection/learning/export costs, and total/max group
latency. `skip_trivial` is a separate explicit collection setting. Arena traces
retain underlying Commands and per-command latency; report complete-game wall
cost alongside these counts. Fewer exposed prompts alone are not a speedup.

The comparison smoke uses the existing arena and offline notebook/report path:

```bash
OMP_NUM_THREADS=1 uv run --extra notebook -m experiments.runners.run_training_regimes --study compound-decisions --profile smoke --out .runs/compound-smoke
uv run --extra notebook -m experiments.runners.run_training_regimes --report-only .runs/compound-smoke
```

This smoke has one initialization seed and four-leg, held-out complete-game
blocks. Scientific comparison requires a separately authorized, calibrated plan
with independent training seeds, fresh deals, equal wall budgets and measured
inference cost. The ETU-91 scientific-plan generator explicitly rejects borrowing
its campaign allocation for compound work.

Bounded evidence (2026-10-04): the retained one-thread ETU-94 workflow attempts
`.runs/etu94-compound-smoke-1` and `.runs/etu94-compound-smoke-final` completed
in 268 and 224 seconds respectively. The latter ran after integrating main's
Ataraxos and belief-sampler stages: four arms, one seed, eight training games,
eight ordinary admitted checkpoints, and 56 complete arena games in 14 cells.
Every arena cell replayed exactly with zero failed games. Offline regeneration
preserved `cost-comparison.json`, `uncertainty.json`, and `report.md` byte for
byte. The attempts retain source identities, individual costs, underlying
microchoices, grouped decisions, and latency in their manifests and reports.
These ignored local receipts are workflow evidence, not a committed benchmark
or a multi-seed strength result. No ETU-91 run or allocation was changed.
