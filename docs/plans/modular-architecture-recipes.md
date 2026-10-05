# Modular architecture recipes for manabot

**ETU-104 · 2026-10-05 · Accepted incremental implementation.** Jack Heart
approved the reviewed design and requested implementation and landing. Section 11
is the accepted delivery scope: Python recipes, shared validated composition and
incremental refinement of the existing model configuration. The broader records
and interfaces below remain design sketches unless explicitly listed as delivered.
No scientific campaign or paid compute is authorized.

[HTML technical review](modular-architecture-recipes.html) ·
[System authority](../ARCHITECTURE.md) ·
[Training regimes](../training-regimes.md) ·
[Current chapter](../../wave/intelligence/GOAL.md)

## 1. Recommendation and the observable win

Author Python experiment files that construct TrainingRegime variants through
shared typed helpers. Build on `TrainingRegime.agent: AgentSpec`, extending
its existing fields and validation for the concrete experiments. Reuse current
Agent construction and checkpoint loading; extract shared helpers only where
those experiments expose actual duplication. Keep the public `Agent`
facade and existing player lifecycle. Compose a handful of real modules, with
validated input and output contracts, rather than adding an Agent subclass for
each experimental arm. Keep objectives and collection in TrainingRegime stages.
Do not adopt Hydra, Ray, Transformers, or a general dependency-injection framework.

The intended demo is a small, bounded complete workflow: resolve a baseline and
value-token recipe; print their exact architectural difference and requirements;
train through the existing regime executor; export and reload ordinary
checkpoints; evaluate through the existing four-leg arena with exact replay.
The report identifies which architecture, objective, input, world, data, and cost
produced each result. A missing history field or incompatible value target fails
before collection. A changed label on a recipe cannot change a frozen run.

ETU-106 delivery is integrated at `a371af46`; section 11 records ETU-104 acceptance. Strength and human
challenger acceptance still require the chapter's separately frozen cohorts.
A passing software demo proves the path only.

**Accepted first cut:** reuse existing configuration and construction; adopt the
focused depth/value implementation rather than duplicate it. New history input,
sequence collectors, and weight-transfer tooling remain later interventions.
Strict ordinary reload remains authoritative.

## 2. Evidence from the actual repository

Local observations below are at base `db820056de813a874ea89cb51f213639b007e389`.
They describe code, not inferred model strength. Historical architecture prose
is useful for authority but is not a reliable inventory of today's implementation.

| Surface inspected | What exists | Design consequence |
| --- | --- | --- |
| `manabot/infra/hypers.py:AgentHypers` | Width, heads, attention switch, semantic pack, compound flag, scalar/WDL kind and belief dimensions | AgentHypers already owns model configuration. Extend it in place for the registered experiments; reuse its serialization and validation. Add nested types only where they simplify a concrete group of settings. |
| `manabot/model/agent.py:Agent` | Typed object projections, optional semantic programs and belief rows, one post-norm attention block, focus-aware action scoring | Preserve these implementations; extract a module only when a concrete variant needs that boundary. Attention MLP width currently equals heads × hidden width; heads and feedforward width must become independently explicit. |
| `Agent.value_head`, `MeanPoolingLayer` | Flat critic applies Linear/ReLU to object rows and averages every slot; attention masks its output rows first, but the critic projection can introduce nonzero padded rows after training | Historical fixed-slot averaging is a baseline contract. Masked mean must mask after the per-row value projection and normalize by valid count. It is a treatment, not a silent baseline repair. |
| `Agent.compound`, `model/compound.py` | Masked context; one-root ragged offers; recurrent declaration prefix; differentiable factor log probabilities and scalar prefix values | A game-memory GRU and a compound-prefix GRU are different state owners. Do not force compound output into flat `[B,A]` logits or claim categorical compound training already works. |
| `model/semantic_cards.py` | GRU encodes complete catalog programs with references; definition IDs route into schema-bound buffers | This is program sequence encoding, not game-history memory. Preserve the complete catalog and semantic binding. |
| `env/observation.py:_encode_events` | Seven numeric columns, validity mask, filtered recent window, suffix truncation at max_events, raw source/target IDs | Agent does not consume these events. Adding them changes information access even if tensor dimensions do not change. Raw identity columns cannot become learned semantic features by accident. |
| `training/models.py`, `execution.py` | Strict TrainingRegime stages; TrainingRun attempts, seeds, costs, artifacts; VerifyStore authority; Agent constructed in execution paths | Extend these records and their ordinary writers/readers. Preserve failed attempts and actual device/threads. Do not create an architecture experiment database. |
| `sim/flat_mc.py:load_checkpoint_agent`, `model/world.py` | Reconstruct saved observation/hypers; validate world/setup/content/input; strict state dict; selected compiled content requires semantic pack | Architecture identity supplements, never replaces, world admission. Equal shapes do not mean equal semantics. |
| `sim/value.py`, `training/ataraxos.py` | Existing scalar/value training and WDL training; serving converts WDL logits to signed expectation | Representation and target semantics must be separate. Existing supervised win-logit values cannot be passed off as signed rollout values. |
| `belief/agent.py`, `state.py` | ViewerHistory, AgentMemory containing explicit belief; supplied-belief evaluation does not mutate autonomous belief memory | Preserve the intervention boundary. No unexposed recurrent belief state may bypass a supplied belief. |
| `belief/sampling.py`, `sampling_fit.py` | Separate autoregressive constrained count sampler with its own GRU, schema, frozen generating policy, dataset and artifact admission | Sampler is a second artifact family. It is not the policy's optional belief-input encoder and need not share parameters. |
| `experiments/runners/training_protocol.py` | EvaluationProtocol binds regimes, seeds, deals, four seat/deck legs, anchors, inference and selection | Reuse and version this protocol. Its current study discriminator has no architecture-study member; add one explicitly later, not by relabeling an old scientific study. |

### Play and trace ownership checked before proposing persistence

`etude/server.py:GameSession` owns live prompts, revisions, canonical decision
rows and retained roots. `_step` makes automatic Commands but only appends
canonical decision rows when `auto` is false. `TraceEvent.auto` distinguishes
server passes from deliberate decisions. `_persist_attempt` writes current trace
and canonical replay through `AttemptStore` in `play.sqlite`. The table DTOs in
`etude/experience_protocol.py` follow the Rust-owned experience schema.

Keep those owners. Architecture metadata belongs in existing artifact and player
identity records, not a new session log. Canonical historical decisions alone
are not a complete execution tape or a neural-memory snapshot. This inspection
does not establish atomic durability across an engine transition and a crash.
Reconstruct memory from admitted viewer history only; fail when that history is
unavailable. Real human completion still needs dated human evidence.

`runtime_fingerprints` in `sim/teacher1_evidence.py` defaults to an Interactive
mirror. The regime executor already passes its actual match and observation
space; all new callers must do the same. `checkpoint_world` binds both decks
with their sideboards. Never change frozen fingerprints to make an artifact fit.

## 3. What public model organizations actually expose

### Primary design references: Keras and the JAX ecosystem

**Direction, Jack Heart, 2026-10-05:** prioritize Keras and the JAX community as
sources to borrow from. The earlier framework comparison below remains supporting
research. This changes design emphasis; it does not request a backend migration.
The following official documentation and public source were inspected during the
review on 2026-10-05. These added links are live references, not pinned revisions.

| Reference | Observed mechanism | Consequence for this design |
| --- | --- | --- |
| [Keras serialization](https://keras.io/guides/serialization_and_saving/) and [implementation](https://github.com/keras-team/keras/blob/master/keras/src/saving/serialization_lib.py) | Constructor configuration is exposed by get_config and reconstructed with from_config; weights are separate state. Custom object code must remain available. | Keep model construction readable and make its effective settings sufficient for reconstruction. Extend the existing saved AgentHypers contract rather than serialize arbitrary experiment code. |
| [Keras Functional API](https://keras.io/guides/functional_api/) | Layers compose into models through explicit inputs and outputs; models can be reused as components. | Favor understandable encoder/value/decoder composition and visible shape contracts. A symbolic graph language is unnecessary for the current variants; existing PyTorch modules can provide these boundaries. |
| [Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html) | Ordinary nested Python modules hold state; constructors receive dimensions and RNGs. NNX supports graph/state separation through split, merge and update. | Make architecture settings, learned parameters, buffers and runtime state distinct. Keep initialization inputs explicit. Shared object references matter; do not accidentally duplicate tied parameters during reconstruction. |
| [Equinox overview](https://docs.kidger.site/equinox/all-of-equinox/) and [paper](https://arxiv.org/abs/2111.00254) | Models are callable PyTrees, and filtered transformations select which leaves participate. | Prefer small composable components and explicit gradient selection over a framework-specific model hierarchy. Pydantic validates configuration at boundaries; it need not wrap model tensors or every runtime object. |
| [Equinox serialization example](https://docs.kidger.site/equinox/examples/serialisation/) | Leaf loading needs a compatible model skeleton; the example saves hyperparameters so that skeleton can be rebuilt. | Save resolved settings alongside exact parameter bytes. A matching tensor shape alone cannot establish semantic compatibility. |

For further JAX-side architecture inspection, start with **Flax NNX and Equinox**.
NNX is especially relevant to the existing object-oriented PyTorch Agent; Equinox
provides a useful smaller, functional alternative. The Flax documentation separates
NNX from the older Linen API, so observations about one must not silently describe
the other. big_vision remains a concrete JAX experiment/preset example below.
This selection is about relevant patterns, not a claim of community-wide dominance.

The resulting recommendation is Python experiment composition, ordinary reusable
model modules, and an explicit reconstruction contract inside the existing
TrainingRegime/checkpoint path. These sources do not prescribe Pydantic or a
universal Spec suffix. Pydantic is a local fit because the repository already uses
it. Recipe → spec is useful local terminology: recipe code produces typed settings;
it does not require a new pair of classes for every component. Jack Heart accepted the naming during review: recipe functions construct specs
or enclosing objects, and the refined Hypers types use Spec names. AgentHypers
is now named AgentSpec, retaining the same configuration owner. Existing-type
names in dated inventories below identify the inspected code; they are not a requirement
to retain the Hypers suffix. Preserve serialized compatibility deliberately.


This is research into public implementations, not access to internal practice.
Source revisions were resolved and inspected on 2026-10-05. Papers establish
method and experiment claims; code establishes the configuration mechanism.
Neither establishes what an organization does in its private infrastructure.

| Public example | Observed mechanism | Useful here; cost to avoid |
| --- | --- | --- |
| Meta/PyTorch torchtune | `_component_` paths instantiate nested components after OmegaConf resolution; training recipes orchestrate model, loss, optimizer and data | Borrow separation of builders from training. An unrestricted import-path graph obscures the small set of legal combinations and makes typo/admission errors late. [Config guide](https://docs.pytorch.org/torchtune/main/deep_dives/configs.html), [pinned implementation](https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/config/_instantiate.py). |
| Google Research big_vision | Python ConfigDict presets; ViT variant expansion; width/depth/pooling choices in a model factory; explicit load exclusions and positional-embedding adaptation | Borrow complete resolved variants and named transfer rules. Image-position interpolation does not license remapping MTG definitions or world ABIs. [Config](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/configs/vit_i1k.py), [model/loader](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/models/vit.py). |
| Hugging Face Transformers | Config classes, model-type dispatch and Auto registration; model/config serialization coupled through documented APIs | Borrow an explicit serialized architecture discriminator. A large extensible model zoo and remote custom code are unnecessary for five local variants. [Guide](https://huggingface.co/docs/transformers/custom_models), [pinned registry](https://github.com/huggingface/transformers/blob/263c5eb913849847dcff1885b7b5e387c483f844/src/transformers/models/auto/configuration_auto.py). |
| Ray RLlib | RLModuleSpec carries class, spaces and model configuration; module APIs separate training/inference and expose initial recurrent state; checkpoint constructor arguments are explicit | Borrow the module/learner boundary and explicit state. Ray's distributed execution system would duplicate existing orchestration. [Current docs](https://docs.ray.io/en/latest/rllib/rl-modules.html), [inspected release implementation](https://github.com/ray-project/ray/blob/479fa716904109d9df4b56b98ca3c3350e1ec13c/rllib/core/rl_module/rl_module.py). |
| DreamerV3 authors | Named capacity presets change coordinated dimensions; Agent selects encoder/dynamics components; explicit carries pass through policy and training; stop-gradient choices are visible | Borrow coherent capacity presets and explicit carry contracts. Its latent world model and imagination learning solve a different problem and are not a recipe framework to transplant. [Config](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml), [Agent](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py). |
| Hydra | Ordered defaults composition and structured config validation | Borrow deterministic resolution, with an audit of effective values. Layered defaults, search paths, interpolation and launch plugins are disproportionate until simple named recipes become inadequate. [Composition](https://hydra.cc/docs/advanced/defaults_list/), [structured defaults](https://hydra.cc/docs/tutorials/structured_config/defaults/). |

Primary papers: [Vision Transformer](https://arxiv.org/abs/2010.11929) motivates
comparing encoder depth/width and token-based aggregation, not an expected MTG
win. [Transformers](https://arxiv.org/abs/1910.03771) describes a reusable model
library. [RLlib](https://proceedings.mlr.press/v80/liang18b.html) addresses
composable distributed RL; the 2018 paper is not evidence for today's RLModule
API. [DreamerV3](https://arxiv.org/abs/2301.04104) supplies a relevant example of
model scaling within an integrated learning method, not proof that scaling this
policy pays at equal laptop time.

The [Ataraxos supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf),
S2 and S3.5–S3.6, specifies history-bearing inputs, a distinct belief model,
pre-layernorm networks, and a Stratego move model with depth 8, width 384 and
8 heads. These are paper-reported architecture choices, not evidence of an
internal registry, config language, or MTG benefit. The move and belief networks
must not be conflated. ETU-106 owns focused paper-inspired MTG contrasts;
[the existing fidelity table](../ataraxos.md) owns learning-rule departures.

### Comparison across the requested boundaries

| Boundary | Finding across sources | Proposed manabot choice |
| --- | --- | --- |
| Composition | Torchtune/Hydra compose configuration; big_vision/Dreamer expose ordinary preset code | Named pure constructors with typed patch overrides; resolve fully once; no inheritance chain at execution. |
| Interfaces/factories | RLlib makes operational interfaces explicit; Transformers dispatches model types; torchtune calls components | Closed discriminated unions plus explicit builders. Unknown kind/version fails before allocation. |
| Shapes | RLlib carries spaces; big_vision derives tensor layouts from model dimensions | Validate actual ObservationSpace and semantic capabilities in addition to dimensions; typed tensors still need runtime checks. |
| Checkpoints/migrations | Transformers couples config and model; big_vision has deliberate adaptation; RLlib records reconstruction arguments | Strict same-architecture reload; versioned metadata migration and separately authorized weight transfer. |
| Artifact identity | Public configuration and a path/name are available mechanisms, not a proof of immutable identity | Hash resolved semantics, source, world and bytes separately; keep labels out of semantic hashes. |
| Lineage/reproducibility | A model config does not by itself identify data, optimizer, runtime, or evaluation | TrainingRun/VerifyStore remain lineage authority. Resolve inputs, retain attempts and costs; ETU-101 projects these to W&B. |

Maintenance caveat: inspected public tips were dated 2026-10-05 (Transformers),
2026-10-04 (DreamerV3), 2026-04-23 (torchtune), and 2025-05-19 (big_vision).
The quiet big_vision tip is a historical research reference, not evidence of
frequent current maintenance. Ray's current master path returned 404; the
inspected code is release `ray-2.49.2`, commit above, dated 2025-09-13, alongside
current maintained docs. Do not infer that all APIs are identical to current
master. A guessed SigLIP config URL also returned 404; the verified ViT config
replaced it. Nature's article endpoint failed through its identity redirect;
the publisher supplement was accessible. No inaccessible page supports a claim.

## 4. The proposed architecture data model

**Accepted naming, Jack Heart, 2026-10-05:** a recipe is a Python function that
constructs spec objects or parents containing them, including TrainingRegime.
Use Spec rather than Hypers for the refined configuration types: AgentSpec is
the evolution of AgentHypers; component configuration can use EncoderSpec and
ValueSpec. There is no AgentRecipe data class or additional recipe schema.
Section 11 records the subsequently approved implementation boundary.
Names in source inventories below refer to existing code. Rename related Hypers
types coherently within the affected scope; preserve old serialized meanings and
loader admission, and do not broaden the change into unrelated runtime types.


**Direction, Jack Heart, 2026-10-05:** the existing AgentHypers is probably close
to the needed model configuration; start by refining it. Its exact name, class
identity and field layout are not requirements. Rename, nest or restructure when
that materially improves a concrete experiment or API; a substantially different
design needs a strong reason. Preserve a single model-configuration owner inside
TrainingRegime and the meanings of existing settings.

Adopt the focused ETU-102/106 additions for depth and aggregation as delivered;
add only settings required by a registered experiment. For example, explicit
feedforward width and normalization are incremental AgentHypers fields if the
current focused implementation does not already provide them. Preserve historical
defaults, field names and checkpoint construction. The architecture configuration
still excludes budgets, optimizers, datasets and evaluation policy.

A nested history object is a possible useful addition because its projection,
window and ordering have joint invariants. It is not a reason to reorganize all
existing fields. Illustrative proposed extension (existing AgentHypers fields
are omitted from the sketch, not removed):

```python
class HistorySpec(BaseHypersModel):
    kind: Literal["none", "recent_events"] = "none"
    projection: Literal["none", "kind_amount_role_v1"] = "none"
    window: int = Field(default=0, ge=0)
    order: Literal["none", "relative_position"] = "none"

# Within AgentSpec, when the history experiment needs it:
# history: HistorySpec = Field(default_factory=HistorySpec)
```

Prefer the existing BaseHypersModel and validation methods. Align shared
validation conventions incrementally when justified, preserving historical
admission; do not introduce a new base class merely for naming symmetry.

Delivered validation requires width divisible by heads for attention.
`attention_layers` accepts 1 or 2; disabling attention uses `attention_on=False`
with the historical default layer count of 1. A value token requires attention.
The proposed history fields would additionally require consistent none/zero
settings or a positive admitted event window; those fields are not delivered. For the initial compatibility cut, reject
compound + WDL and compound + value-token until their distinct prefix-value
semantics are implemented and tested. Do not advertise a rectangular cartesian
product of all fields as supported.

`perspective=per_block` means the current ownership vector is applied at every
stacked block. `once` means applied before the stack only. At depth one they
coincide; at greater depth they are different models. ETU-106 uses ownership once and post-normalization; its delivered fields
and implementations remain authoritative.

Additional named records and their authority:

| Proposed value | Fields / ownership |
| --- | --- |
| Recipe constructors | Ordinary typed Python functions returning complete specs. Explicit keyword arguments replace a separate patch language; unknown arguments and invalid combinations fail before execution. |
| `ResolvedArchitecture` | Resolved AgentHypers snapshot, architecture digest, required capabilities and parameter/buffer shape manifest. Built before optimizer allocation. |
| `ModelBinding` | Existing checkpoint world binding plus semantic catalog, consumed-field projection, belief-schema identity if enabled, and history-projection identity. Bind actual setup, not a preset name. |
| `ValueContract` | Actor-relative perspective, scalar units (`signed_return` or existing `win_logit`), WDL class order and expectation mapping. Trainer declares target semantics; builder verifies compatibility. |
| `ArchitectureReceipt` | Resolved spec/digest, builder ABI version, source digest, initialization scheme/seed receipt, observed parameter count and tensor manifest digest. Appended to existing artifact metadata. |
| `TransferPlan` | Source artifact SHA, source/target architecture digests, exact named key mapping, explicit omissions, target initialization receipt and new run relation. Empty/incomplete mappings rejected. |
| `SamplerArchitectureSpec` | Separate `family="hand_sampler"`, count-GRU kind, hidden width, history projection and initialization; SamplerSchema supplied externally. Dataset-stage history dropout remains a treatment. |

Recipe names are human labels. Two aliases resolving to identical semantics get
the same architecture digest. Source/build identity remains separate, so changed
implementation with the same spec is still a different artifact lineage.
Canonical serialization sorts keys, includes defaults, rejects NaN, and does not
include paths, labels or timestamps in the architecture hash. Freeze the resolved
spec in TrainingRun and in every exported checkpoint; never resolve a mutable
preset name while reloading.

### A shared family of specifications and experiment helpers

**Ownership decision, Jack Heart, 2026-10-05:** architecture configuration lives
inside TrainingRegime as the model/hyperparameter specification for a TrainingRun.
Start from `TrainingRegime.agent: AgentHypers` and refine it. The ownership
boundary matters; the exact type name and layout may change with justification.
TrainingRun retains the resolved regime as it already does. Checkpoint metadata
projects the exact model specification from that regime for reconstruction.
Experiment files compose regimes; ResolvedStudy groups them with evaluation and
allocation. Their roles remain distinct. A frozen-policy sampler's architecture
belongs to its train-belief stage, not the policy's `agent` field.

Jack Heart requested cross-experiment helpers and investigation of Pydantic as
the common base during review on 2026-10-05. Python authoring is accepted;
the following broader consolidation remains optional beyond the accepted first cut. Reuse existing base types first; a shared convention is useful only where it
removes concrete repeated validation or serialization work.

Inspection finds three overlapping conventions: `infra/hypers.py:BaseHypersModel`
forbids extra fields; `training/models.py:Strict` also rejects nonfinite numbers;
`EvaluationProtocol` additionally freezes assignment. Consider sharing these policies where duplication causes maintenance problems;
a new inheritance hierarchy is not a prerequisite for the experiments. Do not freeze mutable execution records or silently change historical
parsing/admission. Pydantic's `frozen=True` alone does not freeze nested lists or
dicts; resolved plans must defensively snapshot nested values, use immutable
collections where appropriate, and verify canonical digests at execution.

| Existing type | Proposed relationship |
| --- | --- |
| `AgentHypers` | Start from this type and refine its fields and validation. Names and nesting may change where useful; justify a substantially different abstraction against the existing design. |
| `MatchHypers`, `ObservationSpaceHypers` | Keep their domain identity and existing validation; include resolved snapshots in the same specification family. Rules/setup and input capacities remain separate from architecture. |
| `Schedule`, `Learning`, `AtaraxosMoveLearning`, `Execution`, operation types | Reuse as typed experiment components. Their current algorithms, schedules and discriminator semantics remain authoritative; no second objective schema. |
| `TrainingRegime` | Own the architecture in its existing agent boundary along with match, observation and stages. Shared helpers return complete regime variants. Migrate mutation-dependent authoring before claiming immutable specs. |
| `EvaluationProtocol`, `ResolvedStudy` | Reuse the existing evaluation and whole-study envelope. Generalize shared composition helpers around these rather than introducing a competing Experiment/Study object. |
| `omitted_controls.Contrast` | Existing baseline/treatment composition is a concrete source for common helper extraction. Preserve its declared changes and labels. |
| `TrainingRun`, `StageRecord` | Share serialization/validation conventions where compatible, but retain their execution-record lifecycle and VerifyStore authority. They are outputs, not recipe subclasses. |
| `SamplerSchema`, runtime tensor dataclasses, `nn.Module` | Keep current owners and runtime representations. Wrap persisted sampler architecture in the spec family without converting every tensor object or module to Pydantic. |
| `model/world.py` binding dictionaries | Candidate for a typed boundary wrapper; preserve serialized bytes, digest rules and admission. Never redefine managym world semantics merely to unify Python types. |

Start with shared baseline/model constructors, validated variation helpers,
named comparison construction and resolution/export. Cross only requested axes;
reject unsupported combinations rather than silently dropping cells. Helpers
return new values, never mutate a shared baseline. Reconstruct through normal
Pydantic validation: unchecked `model_copy(update=...)` is not an admission path.
Experiment code must make coupled changes (for example WDL output and its value
target contract) visible in the resolved diff.

Delivered helper API (`manabot/training/recipes.py`):

```python
def with_value_aggregation(
    regime: TrainingRegime, *, id: str,
    aggregation: Literal["historical_mean", "masked_mean", "value_token"],
) -> TrainingRegime: ...

def with_capacity(
    regime: TrainingRegime, *, id: str,
    width: int, depth: Literal[1, 2], heads: int,
) -> TrainingRegime: ...

def value_outputs(
    regimes: Sequence[TrainingRegime],
    outputs: tuple[Literal["scalar", "categorical_wdl"], ...],
) -> dict[str, TrainingRegime]: ...
```

The delivered model fixes post-normalization, injects ownership once, and derives
feedforward width from heads × embedding width. This first cut exposes only the
admitted fields. Independent feedforward width, normalization and history remain
future interventions; a helper cannot invent support for them.

The experiment file constructs complete TrainingRegime objects. These helpers
replace only declared settings inside each regime, validate nested specifications
and stage compatibility, and return independent values with distinct regime IDs.
Each input regime ID supplies its cell prefix; duplicate IDs fail before crossing.
`value_outputs` changes `agent.value_kind` and the regime ID in the resolved
configuration. The existing trainer dispatches the corresponding value targets
and loss from that field; stage learning settings remain unchanged. Unsupported
objectives fail full-regime validation. `with_capacity` preserves normalization, pooling and information
inputs. No helper changes budgets, starts training or silently omits invalid arms.

The existing ResolvedStudy groups the resulting regimes with EvaluationProtocol,
allocation, calibration and runtime fields. Recompute and validate regime digests
when constructing that envelope; preserve declared seeds, budgets and deal cohorts.
Scientific protocols may not silently expand to more arms. The existing executor
remains the only execution path. Add history or other composition helpers when
repeated experiments need them, not an arbitrary configuration language or
automatic hyperparameter search system.

## 5. Example recipes and independent interventions

### Author an experiment in Python; export its resolved configuration

**Decision, Jack Heart, 2026-10-05:** prefer a Python file per experiment;
YAML or similar formats belong in exports and reports. Experiment files compose
typed model specifications, existing TrainingRegime values and EvaluationProtocol.
Shared constructors remove repetition; the experiment file makes the changed and
held-fixed choices visible together. No authored YAML tree or patch language is
required. Model implementations remain ordinary shared PyTorch code.

Imports and definition calls do not train, download data or mutate state. An
explicit entry point resolves and validates the complete plan before execution.
Planning/export works without constructing a collector. Scientific budgets and
cohorts come from a separately frozen protocol, never hidden recipe defaults.
Generated JSON is the machine-readable run record; YAML is an optional readable
report. Reload uses the saved complete spec and admitted local builder, not a
fresh invocation of the experiment file. Record the experiment source digest as
well as the resolved plan: Python can compute a plan, but its name is not identity.

The delivered experiment is `experiments/runners/run_value_models.py`.
`smoke_baseline()` calls the shared `ataraxos_baseline` with the authored
Allies/Lessons setup, semantic pack, width 64/four heads, two linked stages,
one update per stage, four streams of 64 transitions, 30-second stage limits
and a 60-second cell limit. Every input and allocation is explicit in Python.

```python
from experiments.runners.run_value_models import smoke_baseline
from manabot.training.recipes import (
    value_outputs, with_capacity, with_value_aggregation,
)

base = smoke_baseline()
token = with_value_aggregation(base, id="value-value-token-1", aggregation="value_token")
models = (
    base,
    with_value_aggregation(
        base, id="value-masked-mean-1", aggregation="masked_mean",
    ),
    token,
    with_capacity(
        token, id="value-value-token-2", width=64, depth=2, heads=4,
    ),
)
recipes = value_outputs(models, ("scalar", "categorical_wdl"))
```

All eight cells retain ETU-106's resolved regimes and exact digests, seed 1061,
paired deal 910106, anchor deal 920106, four seat/deck legs and 900-second process
cap. Scalar/WDL output selects the existing value targets and loss while keeping
the Ataraxos policy learning settings fixed. Raw and EMA exports retain ordinary
admission. Importing and resolving definitions does not train.

The capacity examples are `experiments/runners/model_capacity.py:regimes`:

```python
from experiments.runners.model_capacity import regimes
from experiments.runners.run_value_models import smoke_baseline

capacity_examples = regimes(smoke_baseline())
assert set(capacity_examples) == {"w64-d1", "w64-d2", "w128-d2"}
```

They change only width, depth and head count plus the regime label. They preserve
pooling, semantic input, learning, setup and the supplied baseline's budgets.
These constructions do not allocate an evaluation protocol or run ETU-102/103.
Width/depth changes create new models; no weight transfer is implied.

ETU-106's merged implementation supplies the exact pooling/depth/initialization
semantics. No second encoder, registry, builder or checkpoint schema is needed.
History remains an explicit later information treatment; no architecture recipe
silently consumes the ignored recent-event tensors.

Registered sources: [ETU-102](https://linear.app/loopflow/issue/ETU-102),
[ETU-103](https://linear.app/loopflow/issue/ETU-103), and
[ETU-106](https://linear.app/loopflow/issue/ETU-106). The inspected focused
`experiments/value-models.md` snapshot has content revision
`9aa850d3ea5e28a350ec1eca25b96e16e26ef91c7fa2ee08122cf9e2a90821d1`.

### Proposed architecture-only configuration

The following is an unimplemented design sketch for a future architecture-only
export. The delivered export remains the ordinary TrainingRegime with its flat
AgentSpec fields, as constructed above. This sketch is not accepted by the CLI:

```json
{
  "schema_version": 1,
  "family": "policy_value",
  "encoder": {
    "kind": "typed_objects", "width": 64, "depth": 1, "heads": 4,
    "ff_width": 256, "norm": "post", "perspective": "per_block",
    "semantic_program": "catalog_gru"
  },
  "history": {"kind": "none", "projection": "none", "window": 0, "order": "none"},
  "memory": "none", "decoder": "flat_focus",
  "value": {"aggregation": "fixed_slots", "output": "scalar"},
  "belief_input": "none", "initialization": "existing_agent_v1"
}
```

The selected Allies/Lessons semantic pack and actual tensor capacities live in
ModelBinding/TrainingRegime, not hidden defaults in this JSON. Generic uncompiled
models resolve `semantic_program=none` only when their world admits it.

| Recipe label | Fully specified difference from the baseline above | What it tests |
| --- | --- | --- |
| `baseline` | None | Reproduces the existing function, including critic pooling. |
| `masked-mean` | `value.aggregation=masked_mean` | Validity-aware critic aggregation, with per-row value projection unchanged. |
| `value-token` | `value.aggregation=value_token` | A learned token participates in attention; read it for the value MLP. This can also change policy object representations. |
| `depth-2` | `encoder.depth=2` | Repeated existing post-norm block; independent parameters and declared perspective injection. |
| `pre-norm-2` | depth 2, norm pre, perspective per_block | A separate norm contrast against depth-2, not baseline alone. |
| `width-128` | width 128, heads 4, ff_width 512 | A proposed capacity rung; not equal parameter count or equal runtime. |
| `events-32` | history recent_events, projection kind_amount_role_v1, window 32, order relative_position | Bounded ordered recent information, no game recurrence. |
| `compound` | decoder compound_gru, aggregation masked_mean, output scalar | Existing native structured support and prefix-value contract; compare with compound control, not just flat action accuracy. |
| `belief-input` | belief_input canonical_marginals | Existing semantic belief-input path with required schema; ordinary generic trainers remain unsupported until they supply these inputs. |
| `wdl-token` | value aggregation value_token, output categorical_wdl | Aggregation × representation crossing; requires existing categorical target path. |

Capacity ladder numbers are illustrative, not a scientific allocation. Ingest
the focused ladder's actual resolved widths/heads/depths when it lands. Never
rename or reconstruct its frozen recipes from these examples. Its minimal local
AgentHypers extension can ship independently, with explicit serialized defaults,
admission and parameter counts. The later resolver must express it losslessly.

ETU-106's live Task was read on 2026-10-05 (revision
`2026-10-05T20:48:12.333Z`). Jack Heart selected value-token aggregation for focused
implementation/testing, retaining historical pooling as control and masked mean
as an alternative. That selection is accepted scope for ETU-106; it is not
approval of this framework. Jack Heart subsequently approved the incremental
ETU-104 design separately. Integration now preserves ETU-106's delivered pooling,
token placement, normalization, state-key and default semantics. Its initial capacity ×
history contrast and larger-rung repeat retain a fixed learning rule and action
domain. Categorical versus scalar remains a separately identifiable contrast.

The sampler example is independently resolved:
`family=hand_sampler, kind=count_gru, hidden_width=64,
history_projection=existing_commitment_summary, initialization=existing_sampler_v1`.
Its SamplerSchema binds vocabulary, count limits and history coordinates. A
future history Transformer is another explicit sampler variant with new data
requirements, not a policy recipe changing the sampler implicitly.

## 6. Module APIs, tensors and state

Proposed signatures use named domain values. `ObservationBatch` is a validated
view of the existing tensor dictionary; validate once at collection/load boundaries
and retain a cheap typed wrapper internally. It must not copy or reshape every
tensor on every forward call. Concrete module implementations remain `nn.Module`
so state dicts, parameter registration and devices have ordinary PyTorch behavior.

```python
from dataclasses import dataclass
from typing import Protocol
from torch import Tensor

@dataclass(frozen=True)
class EncodedObjects:
    rows: Tensor          # float [B,N,D], including optional appended tokens
    valid: Tensor         # bool [B,N]
    object_count: int     # original object span; focus indexes address only it
    value_index: int | None

@dataclass(frozen=True)
class FlatOutput:
    logits: Tensor        # float [B,A], aligned with legal offer rows
    value_raw: Tensor     # float [B,1] or [B,3], never silently converted for loss
    value_expected: Tensor  # float [B], interpreted by ValueContract

class ObjectEncoder(Protocol):
    def __call__(self, batch: "ObservationBatch") -> EncodedObjects: ...

class FlatDecoder(Protocol):
    def __call__(self, batch: "ObservationBatch", encoded: EncodedObjects) -> Tensor: ...

class ValueHead(Protocol):
    def __call__(self, encoded: EncodedObjects) -> Tensor: ...

def resolve_architecture(
    spec: AgentSpec, binding: "ModelBinding"
) -> "ResolvedArchitecture": ...

def build_agent(
    architecture: "ResolvedArchitecture", binding: "ModelBinding",
    initialization: "InitializationPlan"
) -> "Agent": ...

def build_sampler(
    spec: "SamplerArchitectureSpec", schema: "SamplerSchema",
    initialization: "InitializationPlan"
) -> "AutoregressiveBeliefSampler": ...
```

Quoted types denote proposed records or existing domain types identified above;
this is an interface sketch, not an executable module. Flat and compound methods
remain distinct on the facade. Preserve the existing compound call signature:
`compound(obs, batch, *, tokens=None, prefix=(), generator=None,
deterministic=False) -> CompoundOutput`. The teacher-forced tape returns factor
likelihoods and prefix values. Flat `forward` keeps the existing serving tuple;
`forward_distribution` exposes raw critic targets for compatible trainers. Internal
named outputs prevent positional tensor confusion without forcing every consumer
to change at once.

### Shapes and semantic capabilities

- Current objects form `[B,N,D]`, validity `[B,N]`, flat actions `[B,A,F]`, focus
  indexes `[B,A,K]`, legal mask `[B,A]`. Focus indexes retain the original object
  coordinates when event/value tokens are appended. Reject invalid references;
  `-1` is the explicit absent focus.
- Validity is boolean internally; padded object/event rows cannot contribute as
  attention keys. Mask after biased projections before masked pooling. A learned
  value token is always valid and has neutral ownership, not the opponent's
  perspective embedding. Ordinary object ownership remains unchanged.
- For value-token aggregation append one trainable `[D]` token before the shared
  stack, read its final row, then use a two-layer value MLP. Its effect on policy
  is part of the treatment. A read-only query pooling head would be a different
  recipe; do not silently switch to that to avoid policy effects.
- Reject a nonterminal empty legal set; do not softmax all padding. Terminal
  transitions need no policy distribution. Existing finite-mask baseline
  behavior stays versioned; sampling must still prove zero invalid-action mass.
- Flat capacity overflow fails before action selection. Compound offers stay
  ragged with native support and no artificial candidate cap. Length buckets may
  improve batching but must not truncate legal choices or semantic programs.
- Required capabilities include complete semantic catalog when compiled content
  requires it, event projection when history is on, marginal schema when belief
  input is on, and native compound offers when decoding is compound. Validate
  behavior, not just keys with equal shapes.

### History now, recurrence later

Initial event input consumes the already emitted window as an explicitly lossy
input: embed event kind, signed amount and viewer-relative controller category,
plus relative position, into D and append valid rows to the current-state stack.
Source and target raw IDs are excluded; no stable entity semantics are inferred
from allocation numbers. Known viewer/controller conversion must be validated;
unknown controller gets an explicit category. No new claim of complete history
or entity-resolved reasoning follows. The projection version fixes field units,
kind filtering, chronological order and suffix-window policy.

A zero-event window is valid; missing required tensors is an error. Existing
transport truncation is part of the declared bounded-window contract, not a new
silent overflow rule. New projection inputs beyond that window need a new ABI
and receipts. Check hidden-world swap invariance and native/Python event encoding
parity before enabling this path. Do not parse event hashes as meanings.

A persistent memory variant is **designed but not in first-cut schema**. It needs
`MemoryState(hidden[B,D], match_ids, viewers, last_cursors)` and explicit
`step(batch, state, reset_mask) -> (encoded, next_state)` plus sequence training.
The player/collector owns state per match and seat; reset on terminal/new game,
clone on search fork, replay on historical attachment, and reject missing prefix.
Never store batch-global hidden state on Agent. Incremental events require
canonical cursors so overlapping windows are not ingested twice. Store behavior
state/version with collected sequences; train with declared sequence length,
burn-in, padding and detach boundaries. Randomly shuffled single decisions
cannot train recurrence correctly. Existing recovery snapshots must bind this
state before claiming recurrent process recovery. This is why recent-window
attention is the recommended first experiment.

Compound prefix state lives only within one declaration and is discarded at its
end or interruption. The existing Command suffix queue lives in the player,
resets at game boundaries and fails on stale revision; it is not game memory.
Belief intervention evaluation stays pure. A later shared memory system must
explicitly separate public history from belief so supplying a belief cannot
leave an uncontrolled private posterior in the action path.

## 7. Gradient, initialization and value contracts

Model construction knows output families, not PPO, GAE, Ataraxos, or teacher
label generation. The objective owns clipping, loss coefficients, filters,
bootstrapping, credit clocks and schedules. The regime validator checks required
outputs against target semantics before allocating a collector.

Policy and critic gradients reach their heads and the shared encoder. Behavior
probabilities, reference distributions and value targets remain detached under
the existing objective contracts. Compound sampled tokens are constants during
score-function recomputation; factor log probabilities retain gradients. Grouped
versus sequential credit is an objective treatment, not a different decoder.
WDL output uses loss/draw/win order and `p(win)-p(loss)` for serving. A scalar
network's output does not identify its units: preserve win-logit versus signed
return in ValueContract and reject a search consumer requesting the wrong one.

The frozen-policy sampler receives private truth only as supervision. Policy
parameters stay frozen and get no sampler gradients. Optional canonical belief
inputs to policy remain explicit distributions/projections, not hidden sampler
activations. Shared semantic weights or joint optimization would be a new
measured treatment with an explicit gradient graph and transfer contract.

For baseline parity retain existing initialization order, orthogonal projection
initialization, policy output gain 0.01, perspective scale, and native PyTorch
initializers where current modules use them. Do not claim same-seed parity after
changing construction order. New component-seeded initialization derives stable
named seeds from a run seed and versioned component paths, within isolated RNG
scopes; optimizer/data/action RNGs are separate. Shared unchanged components can
then start identically across ablations. Width changes do not promise identical
weights or function-preserving scaling. Token initialization and depth-block
initialization must be explicit; proposed new token uses Normal(0, D^-0.5).
ETU-106's actual accepted initialization supersedes this proposal at integration.

## 8. Checkpoint identity, migration and transfer

There are four different operations:

1. **Reload:** exact architecture and input/value contracts; strict parameter and
   buffer keys/shapes; verify artifact SHA and world binding; instantiate only
   admitted local builders. No arbitrary import path in a checkpoint.
2. **Metadata migration:** deterministic, versioned conversion of old saved
   AgentHypers to a full specification, preserving the old function and validation
   failures. Record original bytes and migration identity. Missing new fields
   mean the historically correct baseline only for a recognized old schema.
3. **Transfer:** create a new run and artifact with a TransferPlan. Permit exact
   named shared tensors only when semantic coordinates and shape agree; new heads
   get recorded initialization. Report every copied, omitted and rejected key.
4. **Resume:** existing recovery contracts restore optimizer, RNG, collector and
   EMA under exact admitted runtime/source/regime. An architecture change is a
   new run, never a recovery of the prior one.

First transfer implementation should support no shape adaptation. Scalar-to-WDL
may reuse compatible encoder weights but initializes a new head and optimizer;
it is not a checkpoint reload. Width or vocabulary changes reject transfer unless
a later named conversion supplies and verifies coordinate semantics. Do not use
`strict=False` as a migration strategy. Fixed-to-masked pooling can share parameter
shapes yet change the function; record it as new architecture even if all tensors
copy. A compound switch cannot pretend the flat head was trained as a GRU.

Proposed checkpoint additions, within the existing checkpoint artifact, are
`architecture_receipt`, `value_contract`, and a versioned consumed-input binding.
Retain `model_state_dict`, ordinary world binding, observation hypers and exact
belief bindings. Saved AgentHypers remain the architectural source of truth;
additional receipts are derived from them, not independently editable. Add
metadata migration only when an actual schema change requires it, with narrowly
tested mappings. Old missing
world binding and removed positional-belief fields must continue to fail. No
frozen artifact is rewritten in place.

Stable state-key naming is part of the builder ABI. Initially preserve baseline
keys using the facade or a single explicit lossless key map; delete duplicated
forward implementations. Test both output and gradient parity against retained
baseline fixtures. An allowlisted same-shape key rename is a migration; changed
normalization/attention/pooling semantics require a new architecture identity.

## 9. One complete construction → train → export → reload → evaluate path

Jack Heart approved the bounded software workflow on 2026-10-05, under the
section-11 cap. The first cut uses existing Agent construction, checkpoint
metadata and admission. Additional architecture receipts, builders and transfer
interfaces below remain design sketches, not prerequisites for that workflow.

1. **Resolve.** `smoke_plan()` constructs complete TrainingRegime values, validates
   model/stage compatibility and binds their canonical digests to EvaluationProtocol.
   `--write-plan` exports the complete plan without constructing a collector.
2. **Construct and train.** Existing `execute_regime` uses ordinary Agent
   construction from `regime.agent: AgentSpec`, the recorded seed and actual
   observation/setup. Training stages retain their learning rules, optimizer,
   continuation, EMA and watchdog ownership.
3. **Export.** Existing writers retain `agent_hypers` as the dictionary key,
   `model_state_dict`, observation capacities and mandatory world binding.
   TrainingRun/StageRecord retain source/runtime, seed, exact artifact bytes,
   costs and failures. Renaming the Python type does not migrate saved fields.
4. **Reload.** `load_checkpoint_agent` reconstructs AgentSpec from the saved
   dictionary, validates setup/input and strictly loads parameters/buffers.
   Missing aggregation/depth fields retain historical one-layer semantics.
   Wrong metadata or weights fail ordinary admission; no aliases or state ports.
5. **Evaluate.** The existing `value-models` study discriminator, immutable player
   registrations and arena run both raw checkpoint cutoffs with exact command
   replay, all four deck/seat legs and random anchors. Report regeneration uses
   retained records without training. Equal thread count is not equal inference
   cost, and smoke scores are not method-level evidence.
6. **Play admission, when requested.** The separate real-demo workflow still uses
   `configured_opponent` and GameSession with both assignments. Arena replay and
   artifact reload do not establish human-game completion or challenger strength.

Separate sampler continuation remains the existing frozen policy → whole-game
private dataset → separately admitted sampler artifact → quality report/local
search path. Its schema, world, policy identity and gradient boundaries are
unchanged. Compound policy decoding retains its existing configuration and
prefix/reset lifecycle; unsupported value/capacity combinations fail validation.

Proposed interfaces at persistence boundaries:

```python
def validate_stage_model(
    stage: "Operation", architecture: "ResolvedArchitecture",
    value: "ValueContract", binding: "ModelBinding"
) -> None: ...

def migrate_checkpoint_metadata(
    metadata: "CheckpointMetadata"
) -> "AdmittedModelMetadata": ...

def apply_weight_transfer(
    source: "AdmittedCheckpoint", target: "Agent", plan: "TransferPlan"
) -> "TransferReceipt": ...
```

Existing APIs such as execute_regime and load_checkpoint_agent remain public.
No new train CLI is required; `uv run manabot train --regime <resolved-file>`
remains the entry point for a resolved regime. The JSON in section 5 is
architecture-only and cannot be passed as a current TrainingRegime.

## 10. Failure behavior, alternatives and proportion

| Failure | Required behavior |
| --- | --- |
| Unknown recipe kind, typo, invalid width/heads or unsupported cross-product | Parse/resolve error with field path before model/collector allocation; retained attempt if execution was requested. |
| Missing semantic/history/belief capability, malformed masks, overflowing flat actions | Admission or collection failure with the actual required schema/capacity; no truncation, neutral belief or decoder fallback. |
| Different current world or content, changed bytes | Existing ordinary loader rejection; preserve historical evidence. |
| Interrupted compound suffix or missing memory prefix | Fail closed; no resampling to manufacture a successful trace. |
| Run timeout/OOM/incomplete game | Existing failed/interrupted records, costs and partial evidence; no replacement games silently added. |
| Architecture hash matches but source/build digest differs | Distinct implementation lineage; exact resume refuses under its existing rules. Evaluation records the difference. |
| Two presets change loss and architecture together | Resolver may represent them, but contrast report marks confounding changes; no isolated architecture conclusion. |

Alternative A: keep adding AgentHypers switches. This is appropriate for the
independent capacity ladder and ETU-106's focused change. It becomes fragile as
history, compound prefix and belief capabilities need validation across exporters
and trainers. Extending these switches with validation and shared regime helpers is the
recommended starting point. Extract modules only when a concrete variant needs it.

Alternative B: adopt generic Hydra/torchtune-style component paths. Flexible,
but it allows configurations the code cannot train/serve and moves semantic
validation into runtime wiring. There is no demonstrated need for external
plugins, arbitrary DAGs or user-supplied model code here.

Alternative C: one subclass per arm. Easy initially, but duplicates gathering,
masking, state and checkpoint behavior; a masking fix can move one arm only.
This defeats the requested reproducible comparisons.

Recommended approach: explicit constructors and strict specs for a small closed
set, growing only when a runnable experiment needs a new boundary. Avoid generic
shape inference, automatic tensor adapters, optimizer-as-model config, a model
registry service, a new tracking backend and distributed execution refactors.
The principal failure risk is a framework that absorbs research time and still
cannot export a real playable checkpoint. The complete-path gate prevents that.

## 11. Delivery cut and acceptance after review

**Accepted by Jack Heart on 2026-10-05:** make registered comparisons easy to
express as Python-built TrainingRegime objects. Keep architecture in the existing
agent configuration; preserve checkpoint meanings and the frozen ETU-91 campaign.
AgentHypers is refined in place as AgentSpec after adopting focused delivery. A
broad configuration hierarchy, alternate executor, registry or backend migration
is not justified by this cut.

Delivered implementation:

- ETU-106 PR #222 (`a371af46`) was merged with CI passing and 120 replayed
  evaluation-recovery games, as reported by Jack Heart. `lf sync` integrated it;
  the earlier dependency review blocker is resolved.
- `AgentSpec` replaces the Python type AgentHypers in ordinary model construction,
  training, serving, exports, scripts and tests. Serialized `agent_hypers` keys,
  field meanings, defaults, strict loading and world admission stay unchanged.
- `ataraxos_baseline`, `with_agent`, `with_value_output`,
  `with_value_aggregation`, `with_capacity` and `value_outputs` compose independent
  validated regimes. The paper-value contrast and eight-cell value study use
  them; all eight ETU-106 recipe digests and its protocol remain unchanged.
- The capacity example constructs width/depth 64/1, 64/2 and 128/2 through the
  delivered model fields. No ETU-102 run, alternative encoder or expanded study.

**Removed in the accepted cut:** the old AgentHypers Python symbol and imports,
JSON-template authoring of these value comparisons, and their unchecked per-cell
model/budget mutation loops are removed. Frozen JSON/checkpoint evidence stays
intact. Existing ordinary construction, learning rules and recovery remain owners.

The retained one-thread workflow at code commit `486739d5` completed all eight
runs, 16 raw plus 16 EMA exports, and 30 arena cells / 120 valid exact-replayed
games in 348.04 charged seconds. Evidence lives in
`.runs/etu104-recipes-smoke`. The first attempt reached all games but failed
reporting at 342.77 seconds because `nbclient` was absent. The original study
JSON is retained as `study-before-report-resume.json`; ordinary `run_study(...,
resume=True)` retained that interruption and completed only reporting after
installing the declared notebook extra. All checkpoint, run and arena-row hashes
remained identical. No retraining, replay replacement or protocol change occurred.
Offline regeneration reproduced report, metrics and cost/uncertainty JSON bytes;
the report's notebook code cells executed.

The bounded workflow requires the declared reporting dependencies:

```bash
uv run --extra notebook python -m experiments.runners.run_value_models --out .runs/etu104-recipes-smoke-next
uv run --extra notebook python -m experiments.runners.run_value_models --report-only .runs/etu104-recipes-smoke-next
```

Jack Heart requested completing delivery and landing after publication as PR #223.
The landing pass reviewed the actual diff and retained acceptance evidence; 64
focused recipe, control, aggregation and world-admission checks passed. This is
agent technical review, not human code review. Required CI and verified merge
remain delivery conditions. The approved software
cap is 15 minutes per complete attempt. Scientific contrasts require a separate
frozen protocol, independent seeds, paired held-out deals and declared costs.
History, sequence training, new sampler architecture and transfer tooling remain
outside this first cut. No strength or byte-identical-training claim follows.

The implementation checks covered recipe composition, categorical reload,
protocol, model/world admission and challenger consumers. After compression at
`e0170f5f`, the focused recipe/value/omitted-controls suite passed 31 tests in
8.56 seconds; all eight recipe digests remained unchanged. Broader verification
remains with gate/CI; this reconciliation does not rerun training.

## 12. Review record and remaining decisions

| Date / attribution | Disposition |
| --- | --- |
| 2026-10-05, Jack Heart | Requested primary-source research and separate substantial design review before architecture framework adoption. |
| 2026-10-05, Jack Heart, ETU-106 live directive | Selected value-token aggregation for focused implementation/testing; historical pooling is compatibility/control. Broad framework remains gated. |
| 2026-10-05, ETU-104 research | Proposed closed typed specs, explicit builders, strict reload, separately receipted transfer, recent-window history first. Not approved. |
| 2026-10-05, Jack Heart, design review | Selected Python files per experiment; YAML/JSON serve as generated exports/reports. Requested concrete walkthroughs of registered Ataraxos-inspired experiments. |
| 2026-10-05, Jack Heart, design review | Requested reusable cross-experiment helpers and consideration of Pydantic plus existing object families. Shared Python helpers were subsequently approved in the incremental first cut; the broader consolidation inventory remains optional. |
| 2026-10-05, Jack Heart, design review | Confirmed architecture belongs inside TrainingRegime as the model/hyperparameter configuration for TrainingRun. Evolve the existing agent field; do not add a competing top-level recipe owner. |
| 2026-10-05, Jack Heart, design review | Preferred refining the existing near-fit AgentHypers, without requiring its exact type or name. A substantially different design needs a strong reason; useful restructuring remains available. |
| 2026-10-05, Jack Heart, design review | Prioritized Keras and JAX-community patterns. The added comparison focuses on Keras, Flax NNX and Equinox; it does not imply a backend migration. |
| 2026-10-05, Jack Heart, design review | Accepted recipe = Python function constructing specs or their enclosing objects, and Hypers → Spec naming. Retain lessons from all earlier research sources alongside the increased Keras/JAX emphasis. |
| 2026-10-05, Jack Heart, approval carried into implementation | Approved the reviewed incremental section-11 design, implementation and landing, with at most 15 minutes on one CPU thread per complete workflow attempt. |

Review approval is complete, and ETU-106 is integrated. The capacity examples
use its delivered width/depth fields; no independent ETU-102 implementation or
scientific run is claimed. ETU-101 retains dashboard
ownership. ETU-91's running campaign, checkout, recipes and evidence are untouched.

The HTML is a self-contained rendering of this design and implementation status.
Visual browser validation is unavailable in the headless environment; structural
and local-link checks are used instead. Research citations and historical source
observations above remain dated evidence rather than claims about merged code.
