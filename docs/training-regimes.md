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
