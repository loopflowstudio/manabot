# manabot

The trainable agent of Manabot: environment wrappers, models, search,
training, and verification. You train *a* manabot; this package is how.

## Quickstart

```bash
uv run manabot train              # bounded CPU production-path smoke
uv run manabot train --recipe baseline  # ordinary self-play + evaluation
uv run manabot train --regime-name batched_baseline --out .runs/batched/run  # batched GPU baseline
uv run manabot train --learning-seats one  # explicit exact-quota control
uv run manabot sim --preset sim --set sim.hero=random --set sim.villain=default
uv run manabot belief-demo        # engine-derived belief/intervention proof
uv run manabot belief-learn-demo  # held-out frozen-population belief proof
```

The default `local-smoke-v1` recipe runs two CPU collection/update iterations
through the production regime executor: two streams, 16 decisions per stream as
a floor, both seats and complete games, with a 120-second failure deadline.
It uses the baseline model and horizon-scaled damped policy objective, records
actual counts through Etude and exports admitted checkpoints. `--steps`,
`--seconds`, `--seed`, `--out` and `--learning-seats one|both` make local controls
explicit. This is a software smoke, not playing-strength evidence.

Self-play TrainingRegime execution logs cumulative update/transition clocks and
retains admitted initialization, exact monitoring milestones, and raw stage
exports in Etude. These are policy artifacts, not portable training recovery. TrainingRun/VerifyStore still hold
scientific execution records, and the existing scheduler/evaluator has not yet
been replaced by Etude jobs. This is a partial migration, not end-to-end experiment
admission. SDK artifact retention alone does not establish portable collector/RNG
recovery. Remote storage, rentals and scientific runs require explicit allocation.

Training `--preset` and `--set` routes are retired. Their old reward and
optimizer contracts are not translated. Use a recipe or an explicit
`TrainingRegime` JSON. Simulation retains its presets and `--set` overrides.

`--recipe baseline` selects a complete ordinary training run above the regime
layer: current-world Allies/Lessons self-play (128 updates by default), its horizon-dependent schedule,
checkpoint exports, and paired outcome evaluation. `--steps N` counts completed
collection/update iterations (four streams, at least 64 combined seat decisions
each, finishing current games before updating), not optimizer
calls. The same recipe drives the [nightly learning check](../.github/workflows/baseline.yml).
The bare command remains the small `local` smoke run. Use `--out` for a new output
directory; otherwise the recipe creates a timestamped directory under `.runs/`.
`--seconds` sets a failure deadline without changing the step schedule. Saved `--regime` inputs retain their meanings and cannot be combined with
`--recipe`. The baseline is a versioned starting point, not a strength certification.

For GPU training, start from `batched_baseline()` in `experiments/batched_baseline.py`. It is the
same 64×2 model trained on 64 streams for 524,288 learner transitions (128
iterations), with the learning-rate schedule mapped onto 2,048 four-stream
iterations. In the [longer size and batching comparison](../experiments/longer-learning-017.md)
it met the competence thresholds on both seeds within 7 minutes on an RTX 5090, and the
full run takes about 13 minutes versus about 110 at four streams; larger models showed no advantage at this
horizon. The regime is that study's exact tested recipe; it requires CUDA.

Build named curated matches with `MatchConfig.authored(pack_key, hero_deck,
villain_deck)` so both main decks and sideboards come from managym's compiled
setup. Custom `hero_sideboard` / `villain_sideboard` maps default to empty and
require positive integer counts. `Match.swapped()` moves both lists together;
`Env.reset(options={"match": match})` also replaces the setup used by auto-reset.


Train on rented machines through Etude: see [remote training](../docs/remote-training.md).
New self-play jobs use exact absolute update checkpoints (`checkpoint_updates`,
default 1,000); freeze targets and shared intervals before launch. Operational
lease and spending limits remain in hours/dollars. See [cadence and continuation](../docs/training-monitoring.md#new-training-and-live-dashboards).

New self-play training learns from both seats by default (`baseline-v3`).
Use `TrainSelfPlay(learning_seats="one", ...)` for an explicit one-seat control;
saved schema-1 recipes retain their historical one-seat behavior.
Use `games_per_stream` on both arms to match complete-game budgets. Distinct
learning models can use `PairedLearners` to update both independent optimizers
from their own seat trajectories. See [both-seat learning](../docs/training-regimes.md#learning-from-both-seats)
for collection boundaries, accounting and measurement limits.

Single-stage self-play can opt into bounded [game crash recovery](../docs/game-crash-recovery.md).
It retains private replay inputs, discards both seats' uncommitted experience from
failed games and regathers under the same policy. A persisted rolling error rate
strictly above the declared threshold stops training; evaluation remains strict.
Strict scalar/vector training and data-generation paths automatically retain
private native failure inputs before propagating errors. Compound/selection and
simulation retain pending Commands before execution. The same bounded replay command
handles strict failures without enabling tolerance or process recovery; see the
linked capture audit for boundaries and supervision limits.

## World identity

Ordinary checkpoints require `world_binding` from
`manabot.model.world.checkpoint_world(actual_player_configs, observation_space)`.
This binds native `WORLD_VERSION`, rules schemas, full deck/sideboard pairs,
content manifest and input schema. Missing or mismatched bindings fail before
weights load. Seat reversal preserves each deck's sideboard. Policy checkpoint
writers require explicit `player_configs`; never infer training setup from
tensor dimensions or the default deck.

For the compiled Allies/Lessons matchup, use
`ModelConfig(semantic_pack="ur-lessons-vs-gw-allies")`. The ordinary model joins
viewer-safe `semantic_cards` transport IDs to complete checked catalog programs,
including characteristics and referenced definitions; `known_hand` contains
public definition minima, not hidden hand slots. Selected compiled checkpoints
without that semantic input fail admission. Generic uncompiled training keeps
its existing bounded feature model. These models do not certify strength or
semantic transfer; no previous checkpoint is relabeled compatible.

`ModelConfig(policy_film=True)` scales each action embedding by the pooled game
state before the policy head, so offer choices that depend on the state stay
learnable from any start. The local and baseline recipes enable it; the
ordinary baseline recipe is `baseline-v3`.
Omitted, the field serializes as the historical additive policy, so frozen
recipes, presets (`ataraxos-mtg-v1`, `baseline-v1` receipts) and checkpoints
keep their meaning.

An observation/action-shape or rules-meaning change is a world version.
The current development world is **w14** (`managym.WORLD_VERSION`); its rules, tensor shape,
and compatibility limits are in [WORLDS.md](../WORLDS.md). Earlier checkpoints,
including w13 artifacts, are not compatible with w14. Elemental combat,
[sacrifice/landfall](../docs/rules/elemental-sacrifice.md) and [Temur](../docs/rules/elemental-temur.md) compiled admission is implemented;
[ten-deck Elemental admission](../docs/rules/elemental/world.md) is implemented in
`elemental-world-v2`; release certification remains unfinished. The local
challenger workflow records the native world identity and current rules,
content, Lesson-pool and tensor identities with each run. Historical Learn
certification limits remain in the [Learn record](../docs/rules/learn-lesson.md).

`manabot sim` accepts `random`, `default`, a local policy `.pt` path, or an
Etude checkpoint-record `.json` path for each player. Checkpoint records resolve
exact digest-bound local/S3 artifacts before normal world and architecture
admission. Simulation results are recorded locally through Etude.

## Architecture

manabot owns memory, beliefs, planning, policy/value learning,
self-play, evaluation, and promotion evidence over the authoritative managym
world. The cross-package contracts and convergence status are in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   (`VectorEnv`, `ObservationSpace`, `Match`, `Reward`)
2. **`manabot.model`**: trainable policy and value models (`ActorCritic`);
   `manabot.training` owns the regime executor and `Learner`
3. **`manabot.sim`**: players and game runners.
   `sim/action` owns structured offers and compound-action execution.
   `sim/search` owns frozen-policy planning, world sampling and search evidence:
   `puct.py` implements tree search; `conditional.py` compares its action estimates
   under different hidden-information assumptions at the same decision.
4. **`manabot.semantic`**: semantic-program projection and production encoders
5. **`manabot.verify`**: competency scenarios, behavioral probes, and the
   run-provenance store
6. **`manabot.infra`**: shared configuration, runtime setup, provenance and profiling
7. **`manabot.belief`**: canonical world distributions, viewer history,
   exact-world inference, and the `ManabotPlayer` lifecycle

Belief input is opt-in: schema-bound checkpoints load through the ordinary
`checkpoint` player and update belief before each policy/value decision.
`belief-demo` uses a freshly initialized policy to prove intervention wiring;
`belief-learn-demo` trains a belief scorer on one scripted population and holds
out whole episodes. Neither demonstrates strategic strength or opponent
transfer. The generic PPO trainer does not yet produce the
semantic belief inputs required to train a belief-enabled policy. Historical
positional-condition checkpoints are rejected rather than reinterpreted.

Training regimes can [import a published raw/EMA policy](../docs/training-regimes.md#reuse-a-published-policy-without-training)
without retraining, preserving producer provenance and separate sunk costs.
They can freeze a policy, collect private whole-game supervision,
and fit a constrained autoregressive hand sampler without exact enumeration.
See [frozen-policy belief sampling](../docs/belief-sampler.md). This separately
admitted belief artifact does not change the policy's input contract or establish
foreign-opponent calibration or search strength.

[Frozen-policy local search](../docs/local-policy-search.md) plays through the
`local_update` player kind; exact-history search rejects unsupported Rules
likelihood events. Search receipts and simulation replay retain this distinction.

The existing `agent_puct` player accepts current flat and compound checkpoints;
see [PUCT policy/value compatibility](../docs/local-policy-search.md#puct-policyvalue-compatibility)
for critic units, retained prefixes and evaluation registration. This is a search
compatibility path, not a calibrated strength preset.

Declare training comparisons with [typed Python experiments](../docs/training-experiments.md).
Versioned presets, named component overrides and variant matrices resolve to complete
existing regimes with provenance; resolution never launches training.

Each current experiment is one module in [experiments/](../experiments/), holding
its recipes and anything only it needs to run and report; `manabot/` keeps
the reusable instruments. The experiment discipline and ledger are in
[experiments/README.md](../experiments/README.md).

The retired behavioral-study and structural/opcode experiment tools are retained
in Git history. Their archived inputs and receipts are indexed in
[repository archive](../docs/repository-archive.md). Shared target inspection
now belongs to `sim/action/targeting.py`.

`heuristic` (`manabot/sim/heuristic.py`) is a
separately versioned candidate anchor of rules of thumb: highest-mana-value casts,
removal held for a kill, counters held for an opponent's spell, effect polarity
derived from the semantic card source, attacks unless a blocker kills and
survives, and value blocks. Prompts without a rule match `scripted_greedy`, which
stays frozen with its registered source identity. Select it for checkpoint monitoring
with `MonitorProtocol(opponent="heuristic")`. One exploratory check (MTG-152,
GPU seed 12261 at 25k, 200 paired games per anchor) scored the checkpoint
0.57 against `heuristic` and 0.795 against `scripted_greedy`, lower in every
cell; it is not yet a frozen anchor comparison.

For versioned S1–S5 tactical diagnostics through ordinary checkpoint players,
see [checkpoint scenario scoring](../docs/checkpoint-scenarios.md). Each retained
custom-deck root requires its exact checkpoint setup binding.

For capacity configuration, component counts, versioned architecture receipts and
bounded CPU calibration, see [capacity accounting](model/agent.md#capacity-identity-and-accounting)
and [the calibration command](../docs/training-calibration.md#capacity-ladder-software-calibration).

## Research program

[RESEARCH.md](RESEARCH.md) says what we are doing now, what the experiments
have shown, what we plan to do later, and how we measure results. Live work
and ownership remain in Linear; frozen predictions and results remain in
`experiments/`.

## Style

```python
"""
filename.py
One-line purpose of file

Instructions for collaborators on how to approach understanding and editing.
"""

# Standard library
import os

# Third-party imports
from torch import Tensor

# First-party imports
from manabot.env import ObservationSpace

# Local imports
from .sibling import Thing
```

Compound checkpoints use an autoregressive legal-offer decoder and execute via
ordinary checkpoint players. Their complete-game training stages, credit
boundaries and separate comparison recipes are documented under
[compound decisions](../docs/training-regimes.md#compound-decisions).

Self-play update diagnostics live only in the run's Etude metrics;
TrainingRun keeps counters and receipts. The [dashboard guide](../docs/training-monitoring.md)
covers reading them offline and separate checkpoint monitoring.

Optional [public recent-event input](../docs/recent-events.md) uses the existing
semantic catalog and a bounded native history suffix. Recipe history-on/off is
independent of capacity and learning rules; comparison plans remain unexecuted.

Self-play [numerical health](../docs/numerical-health.md) records policy support,
selection, gradients and actual optimizer effects, with bounded private failure
artifacts. CPU correctness does not imply CUDA validation or improved play.

For exploratory same-deck checkpoint play, the [head-to-head runner](../experiments/head-to-head.md#mirror-only-follow-up-mtg-131-2026-10-09)
accepts `--matchups mirrors`. This explicitly permits repeating exact checkpoint
roster decks/sideboards and carries their compiled content pack through replay;
it is not ordinary training-match admission or evidence of mirror competence.

### Etude-owned training jobs

`manabot.experiment.job.TrainingJob` freezes a single fresh self-play
`TrainingRegime` and its interior update milestones into an `etude.launch.Job`.
Its `job(repo=..., revision=..., max_hours=...)` method only builds the job;
`job.runs(*seeds)` only declares runs. Neither submits or reserves resources.
The job command is:

```sh
uv run --locked python -m manabot.experiment.job
```

Run that command through an Etude worker, not as a standalone training CLI: it
requires the SDK's config, seed, target, deadline, output, stop and tracking
environment. The source revision is explicit, the update target must agree with
the recipe, and the remaining lifetime must cover the entire recipe watchdog
plus the declared finalization reserve. A late start fails instead of shortening
the experiment. The machine must supply this source's built native extension;
portable source-build setup and the full complexity campaign remain unfinished.

Etude owns job execution and tracking on this path; the old ExperimentSchedule
runner is not involved. TrainingRun/VerifyStore still retain scientific learning,
world/ABI admission and failure receipts inside the attempt output. Only a
completed target emits `etude-result.json` with the admitted policy path.
Cancellation is checked with the resource watchdog (at most once per second at
check boundaries) and retains a failed receipt, not a successful partial target.
Policy checkpoints cannot resume optimizer, RNG or collector state. Automatic
retries are disabled (`max_attempts=1`); multi-stage and imported-state jobs are
not yet admitted. This worker proof does not establish SDK placement, remote
provisioning, scientific evaluation, artifact upload or learning strength.

### Semantic catalog dimensions

`SemanticCardEncoder` uses a shared GRU over semantic-token embeddings, not an
independent learned embedding per card. Its `catalog_shape` reports definition
count, unique token count, total expanded sequence length, longest sequence and
padded token count. Referenced definitions are expanded into the parent sequence.
The embedding runs over the padded array; the packed GRU skips padding. These
counts describe representation cost, not observed learning difficulty.

`manabot.semantic.policy_catalog` owns explicit native/IR admission. Policy
construction checks compiled pack, IR and source hashes and complete native
coverage before joining transport IDs. The selected product catalog remains the
only registered policy catalog; research catalogs need their own checked IR and
schema admission. In particular, the native Jeong increment alone is not policy
admission: its `Rebel` subtype is absent from the current learning schema.

### Explicit research evaluation matrices

`AuthoredMatchup(pack_key=..., role_a=..., role_b=...)` names two exact native
authored setups. Pass it as `MonitorProtocol(authored_matchup=...,
include_mirrors=True)` to evaluate A–B, B–A, A–A and B–B, each with both player
seat assignments. The training match must equal `matrix.match()`; arbitrary
renaming or substituting a different setup fails admission. Simple roles retain
their own names, never Allies/Lessons labels.

The simulator records the actual decks, explicit content pack and Commands;
its ordinary replay checks still apply. The matrix identity is separate from
the historical selected-world suite. Omitting it preserves existing selected
protocol serialization. A native authored roster does not itself admit a neural
policy catalog. Research report presentation and Etude evaluation recording are
still separate remaining integration work.

Runtime configuration uses `Config` names (`ModelConfig`, `MatchConfig`,
`ObservationConfig`, `RewardConfig`, `RuntimeConfig`). `infra/runtime.py::RunContext`
owns per-run seeding and tracking; `experiment/config.py::Experiment` describes
experiment cases and jobs. Built-in recipes live in `manabot/recipes/`.
Search distillation (the exp-10 BC/value loops and the `train_supervised`
regime stage) is retired; replaying that workflow requires pinned source. Historical checkpoint dictionary keys remain unchanged,
but old `ataraxos_move` recipes require pinned source rather than runtime aliases.


### Training and simulation ownership

Dependencies flow from training to simulation. Simulation does not import
training, experiment scheduling or reports, even when loading a frozen model.
The package-boundary tests import every simulation module and play a checkpoint
with those packages blocked.

- `model/`: network construction/inference and portable checkpoint admission.
  `model.checkpoint.save_checkpoint` and `load_checkpoint` own policy files;
  their serialized historical keys remain unchanged.
- `sim/`: registered players, bounded game execution, replay, native card evidence,
  search and batched games.
- `training/`: regime execution, seat-routed collection, learning rules,
  numerical health and optimizer/collector recovery.
  `regime.py` declares recipes; `records.py` holds measured run records.
  `execution.py` sequences stages and owns the run lifecycle.
- `training/rollout/`: on-policy collection, stream producers, matchups and
  complete-game accounting. Compound collection and replay live here.
- `training/learning/`: Learner, optimization rules, per-seat returns, sample
  selection and numerical health. Compound credit and parameter updates live here.
- `training/state/`: policy admission, exact learning snapshots, recovery and
  retained game-failure state.
- `belief/`: viewer histories, world distributions and belief models;
  `sampling_fit.py` fits and persists hand samplers for `train_belief` stages.
- `recipes/`: packaged local/baseline/belief recipes and composition helpers.
- `experiment/`: authored comparisons, local scheduling, Etude job adapters,
  checkpoint staging, evaluation scheduling/followers and artifact publication.
- `eval/`: evaluation protocols, admission, comparisons, population ratings,
  GIH reduction/query and MTG Arena log export.
- `reports/`: retained-run projections, dashboards, report templates and
  historical evidence readers.
- `infra/`: runtime context, shared file operations and provenance identities.

A policy checkpoint can be loaded for play without a trainer. Resuming a run
also requires the admitted optimizer/collector/RNG snapshot and durable failure
ledger. Moving files changes source identity; it does not admit an old run under
new source or alter its frozen world, protocol or cost limits.

Reports use `reports.records.RetainedRun` to read recorded stage parameters as
JSON without translating historical learning tags or adding current execution
defaults. `TrainingRun` admits the current executable recipe; both share measured
run fields through `RunRecord`. Reading or publishing a historical result does
not admit its recipe for a new run or recovery.

[GIH card evaluation](../docs/card-evaluation.md) uses simulation replay capture and Etude
tables for copy-weighted card rankings, explicit cohorts and offline Arena-log export.

Simulation imports no evaluation, training, experiment or report modules.
Shared registration types live in `sim.registration`; canonical byte identities
live in `infra.identity`. The former `manabot.arena` package is removed. Existing
serialized `arena_key`, `arena_version`, seed identifiers and retained artifact
paths preserve their original meaning; current source identities use the new paths.

For a standalone, inspectable pretrained-model decision over supplied text and
legal references, see the optional [LLM harness](../docs/llm-harness.md). It owns
protocol recovery and inference evidence; observation rendering and Command
execution remain outside it.

### Shared evaluation inference

Frozen arena policies can share a bounded, single-machine inference allocation
through an explicit execution envelope. Monitoring, Etude benchmark and duel
callers retain native isolation, Commands, replay and original training
coordinates. See [shared evaluation inference](../docs/shared-evaluation.md) for
admission, real Perf receipts and the predeclared local calibration command.

A quick local checkpoint comparison needs no TrainingRun or cloud setup:
`uv run python -m manabot.eval.local --checkpoint policy.pt --opponent heuristic --deal-seed 101 --out .runs/local-check --seconds 300`.
The [local evaluation API and CLI](../docs/shared-evaluation.md#quick-local-evaluation)
share arena admission, bounded inference, Commands, replay and Perf receipts with
monitoring and managed comparisons. Pass another local checkpoint as `--opponent`
for checkpoint-versus-checkpoint evaluation.

### Native decision text

`semantic.text_observation.capture_text_decision(env, history_capacity=32)`
produces deterministic English and exact legal A references from one native
capture. `submit_response(env, decision, "A0")` parses an exact reference and
atomically checks origin, reset generation and current captured facts before
executing the ordinary Command. A cloned Env needs its own capture. Retained
JSON can be rendered with `render_decision` but cannot authorize submission.
Capture/text v4 labels each ordered target’s retained “Legal when resolution
began” fact; historical v3 captures retain unknown eligibility. Policy input v4
uses the same native fact, separately from current object presence.

Run the local response-stub proof with
`uv run python -m manabot.verify.text_observation --out .runs/text-observation`.
It retains native inputs, prompts, bindings, malformed/stale checks and execution
receipts. This uses bounded public policy history v1; it loads no model and has
no writable model memory. Tokenizer/model recovery belongs to the caller;
`TextDecision.check_token_budget` checks the complete prompt without truncation.
