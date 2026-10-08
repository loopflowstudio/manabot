# Declarative training experiments

`manabot.training.experiments.Experiment` declares typed Python experiments.
Resolution returns complete validated `TrainingRegime` values and a provenance
receipt. It never constructs a model, opens a store, collects games or launches
training. Existing `EvaluationProtocol`, `ResolvedStudy`, `TrainingRun`, checkpoint
admission and execution retain their meanings. Training seeds belong to runs;
the evaluation protocol declares the cohort of run seeds.

## Declare and inspect

```python
from manabot.infra.hypers import AgentSpec
from manabot.training.experiments import Axis, Case, Experiment, Model
from manabot.training.presets import ataraxos_mtg_v1

experiment = Experiment(
    name="value-output",
    baseline=ataraxos_mtg_v1(),
    matrix=(Axis("output", (
        Case("scalar", (Model(AgentSpec(value_kind="scalar")),), "Scalar"),
        Case("wdl", (Model(AgentSpec(value_kind="categorical_wdl")),), "WDL"),
    )),),
)
resolved = experiment.resolve()
regimes = resolved.regimes              # fresh, ordinary TrainingRegime values
regime = regimes["value-output-wdl"]
receipt = resolved.receipt()            # JSON-compatible, complete configurations
assert resolved.digests[1] == resolved.cases[1].digest
```

`Experiment.cases` declares coordinated changes that should vary together.
`matrix` crosses independent axes. Cases are the outer loop; the last declared
axis varies fastest. Each ID joins the experiment, case and axis-choice names in
that order, with hyphens. Names must be nonempty lowercase alphanumeric segments;
underscores, duplicate IDs and duplicate comparison labels fail. Axis names
identify provenance; choice names identify cells. An empty experiment prefix is
supported for existing standalone IDs, as in the capacity example. Without cases
or axes, the experiment resolves one baseline cell with the experiment's name.
Labels join case/choice labels with ` / `; omitted labels use their names. Labels
are presentation, never substitutes for configuration or checkpoint identity.

## Components and explicit changes

| Component | Owned properties and update semantics |
| --- | --- |
| `Model(AgentSpec(...))` | `agent.*`; patch only explicitly supplied fields. |
| `Environment(world=..., match=MatchHypers(...), observation=ObservationSpaceHypers(...))` | World, matchup and input capacity; explicit fields only. A supplied deck/sideboard map replaces that entire map. |
| `LearningRule(Learning(...) or AtaraxosMoveLearning(...))` | Replace the complete learning rule on every self-play stage, including the selected rule's defaults. |
| `Resources(execution=Execution(...), wall_seconds=...)` | Explicit execution fields on every stage plus the total run deadline. |
| `Pipeline(stages=(...))` | Replace the complete ordered stage list, including stage-local learning/resources. Use existing typed operation models for heterogeneous stages. |
| `RunControl(schedule_clock=..., recovery_max_microsteps=..., selection=...)` | Replace these three run controls together. New authoring defaults to `iteration_fraction`; `None` explicitly disables recovery. |
| Generated identity | `id` derives from the declaration; `schema_version` stays with TrainingRegime. |

All effective settings have exactly one semantic owner. Stage learning belongs
to learning and stage execution to resources even when supplied inside Pipeline.
The remainder of each stage belongs to pipeline. New top-level TrainingRegime
fields fail resolution until assigned ownership. Nested fields inherit their
existing component; the resolver does not duplicate their schemas.

Common experiment overrides, case overrides and axis choices can replace a
baseline setting, but cannot overwrite **each other**, even with equal values.
Overlapping parent/child writes also fail. A Pipeline replacement therefore
cannot accompany LearningRule or stage execution overrides. Put the desired
rules/resources directly in its typed stages. Disjoint writes commute; there is
no last-write-wins order or hidden conflict priority.

Pydantic models retain which fields were explicitly supplied. Thus
`Model(AgentSpec(hidden_dim=64))` changes only width, while supplying a fully
populated AgentSpec explicitly replaces all its populated fields. An explicit
default or `semantic_pack=None` remains an override in provenance. Whole-rule
LearningRule and whole-pipeline Pipeline replacements deliberately include their
defaults. Reuse existing model validation; typed patches must themselves be valid
models, and every complete cell is revalidated after composition.

Resolution invokes the existing `validate_regime` admission function: incompatible
model/target/continuation combinations, unavailable belief inputs and a world
that differs from the installed native runtime fail before execution. Artifact
existence, exact runtime/source binding, dynamic legal capacity and checkpoint
admission remain execution checks. A resolved recipe is not evidence that its
training or complete-game workflow succeeded.

## Frozen baselines and preset versions

`Baseline.capture("control-v1", regime)` captures all current serialized defaults
into immutable bytes. Mutating the input, a returned regime, another cell or an
exported receipt cannot change that baseline. Baseline identity includes the full
snapshot, not only its display name. Caller-supplied capacity baselines can have
different digests under the same descriptive label; retain the digest.

`ataraxos_mtg_v1()` loads a checked-in, fully resolved snapshot with preset origin.
There is no `latest` alias, mutable registry, inheritance chain or arbitrary import
path. A semantic change needs a new preset version. Loading rejects incomplete
snapshots or schema normalization that would silently change their bytes' meaning;
a golden digest protects v1 in tests. Changes to default values cannot alter fields
already explicit in the snapshot. Algorithm implementation/source identity still
belongs to the existing TrainingRun receipt: a preset is a configuration promise,
not a promise of unchanged executable code or stochastic training bytes.

The pinned `ataraxos-mtg-v1` configuration is a **paper-inspired MTG move-learning
baseline**, not a reproduction or a scientific allocation. It selects categorical
WDL, the supported value token, two attention layers, width 64/four heads, semantic
Allies/Lessons input, current-policy self-play and evaluation EMA. Two cumulative
stages each request one update, four streams and 64 transitions, capped at 80 s;
the whole recipe is capped at 180 s on one CPU thread. Those are small software
workloads, not evidence of sufficient learning. Raw remains the existing default
artifact selection; choosing EMA evaluation requires an explicit compatible
EvaluationProtocol. No history or hidden truth is invented to make the name fit.

Fidelity is grounded in the [final paper](https://www.nature.com/articles/s41586-026-11036-y)
and [supplement S3.4–S3.7, Tables S7/S10](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf),
checked on 2026-10-05; implementation details and unresolved source conventions
are in [the existing fidelity table](ataraxos.md#fidelity-table).

| Disposition | v1 contract |
| --- | --- |
| Matched move mechanism | Clipped surrogate, both reverse KLs, categorical outcome loss, separate trace parameters, filtering, iteration schedules and evaluation EMA use the delivered S3.4 implementation. |
| MTG-adapted | Semantic legal offers, action-type magnet, same-viewer transitions and MTG terminal boundaries replace Stratego inputs and decisions. |
| Compute-reduced | CPU float32, tiny batches/workload and a 2-layer width-64 model replace the paper's 8-layer width-384 move network and GPU scale. |
| Model departure | Delivered attention uses post-normalization and its existing feedforward/ownership conventions; the paper specifies pre-normalization. The value token is a supported local architectural choice, not a claim of matching the paper's architecture. |
| Missing from this preset | Ordered past-move inputs, setup learning, trained belief-network stages and search augmentation. Separate sampler/search tools exist but are not selected here. |

The closest coherent supported choice means retaining the delivered move mechanism
and categorical target contract while exposing model and system departures. It
does not mean selecting unsupported capabilities or importing an unmeasured larger
capacity allocation. Scalar output, historical/masked pooling and capacity changes
are explicit experimental overrides. ETU-102 still owns capacity accounting and
calibration. The value-token experiment remains independently runnable.

## Existing studies and exports

The [value-model example](../experiments/runners/run_value_models.py) declares four
pooling/depth cases crossed with two output heads. Its baseline is a frozen export
of the pre-migration settings, **not** the new Ataraxos preset. All eight complete
regimes, their IDs/digests and the EvaluationProtocol remain identical to PR #223.
The [capacity example](../experiments/runners/model_capacity.py) declares three
explicit cases against a supplied complete baseline; its previous IDs and values
also remain identical. No saved experiment or checkpoint is rewritten.

Export the existing bounded plan and a separate authoring receipt, without training:

```bash
uv run python -m experiments.runners.run_value_models \
  --write-plan /tmp/value-plan.json --write-provenance /tmp/value-provenance.json
```

Both destinations must be new. `--write-provenance` requires `--write-plan`.
The complete regimes and ordered `resolved.digests` feed the existing
EvaluationProtocol and ResolvedStudy exactly as shown in `smoke_plan`; the study
schema and executor are unchanged. General declarations are not a new study type:
current EvaluationProtocol discriminators still limit admitted cell counts,
checkpoint counts, seeds, budgets and raw/EMA selection. Resolve any valid regime
set independently; do not relabel it as an existing scientific study to bypass
those requirements. Only an explicit call to the existing runner executes work.

`ResolvedCase.digest` is the existing canonical TrainingRegime digest, including
its regime ID. `ResolvedCase.identity` additionally hashes the label, baseline
digest and complete setting provenance; `ResolvedExperiment.identity` hashes the
ordered complete receipt. An explicit override equal to a baseline value leaves
the regime digest unchanged but changes provenance identity. Receipts include
values, tuple paths (JSON arrays, with list indexes), semantic component, origin
(`baseline`, `preset`, `override`, `identity`) and source (`experiment`, `case:...`,
`axis:...`, or the baseline name). The implicit serialized omission of
`compound_decisions=False` remains unchanged for compatibility; provenance also
records that effective setting. Exported JSON is evidence, not a second authoring
language. Existing imperative helpers remain available to internal producers.

## Technical walkthrough

1. `Baseline` validates and freezes a complete configuration. The preset loader
   checks the versioned snapshot; no runtime defaults are merged over it.
2. `Experiment.resolve` enumerates declared cases/axes, expands typed components
   into writes and rejects every duplicate or overlapping path before mutation.
   Learning rules and pipeline dependencies still use their existing typed models.
3. A fresh snapshot receives disjoint writes and the descriptive regime ID.
   Existing regime admission validates the whole result, including cross-component
   compatibility. Failure returns no partially resolved experiment.
4. The resolver walks every effective leaf in stable order to attach its owner
   and source. Immutable configuration/value bytes isolate subsequent consumers.
   Configuration digest and provenance identity answer different audit questions.
5. The value example passes the resulting regimes/digests to the unchanged study
   protocol. Resolution has no run seed, collector, optimizer or execution call.

Tests exercise conflict order, full leaf coverage, explicit null/default intent,
mutation isolation, baseline pinning, ordered matrix names, existing identity
parity and admission failures. An execution guard forbids model/store construction
and executor calls during resolution. These tests establish authoring software
behavior; they establish no strength, calibration or training acceptance result.

Capacity early-progress and terminal comparison plans use the same declarative
ladder and ordinary study executor; see the [unexecuted protocol](../experiments/model-capacity.md).

## Execute a comparison

`Experiment.schedule` declares seeds, configured hardware, process/wall budgets
and monitoring independently of regime resolution. The shared explicit
`run_experiment` runner records actual attempts in `ExperimentRun`, evaluates
milestones during learning and creates one editable comparison notebook.
[Execution, continuation and notebook contracts](experiment-execution.md) describe
the supported local CPU placement, history/depth consumers and evidence limits.
Interpretation and repository knowledge updates belong to an experiment skill.

Prospective self-play monitoring uses positive `checkpoint_updates`, shared across
capacity cases. Use `RunControl()` for iteration-based PPO schedules; Ataraxos
already consumes absolute iterations independently of the target count. Existing
baseline snapshots retain their serialized meaning; this authoring default does
not rewrite them. Freeze targets and intervals from an admitted pilot before
launch as described in [calibration](training-calibration.md#freeze-prospective-step-targets).

`TrainingRegime.schedule_clock` retains its historical reader default. New PPO
step declarations must select `iteration_fraction` through `RunControl()` or the
recipe field; cadence admission rejects elapsed-budget recipes before launch.
