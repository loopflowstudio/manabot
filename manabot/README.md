# manabot

The trainable agent of Manabot: environment wrappers, models, search,
training, and verification. You train *a* manabot; this package is how.

## Quickstart

```bash
uv run manabot train              # --preset local: bounded laptop run
uv run manabot train --recipe baseline  # ordinary self-play + evaluation
uv run manabot train --regime-name batched_baseline --out .runs/batched/run  # batched GPU baseline
uv run manabot train --preset simple    # full PPO run (CUDA, W&B)
uv run manabot sim --preset sim --set sim.hero=attention --set sim.villain=simple
uv run manabot belief-demo        # engine-derived belief/intervention proof
uv run manabot belief-learn-demo  # held-out frozen-population belief proof
```

The default `local` preset is the bounded laptop training path. Ordinary
Trainer metrics and exported checkpoints now record through the public
Etude SDK without a W&B account. Standalone tracking lives under
`<runs_dir>/<experiment>/etude/`; under `etude launch`, the SDK supplies
`ETUDE_RUN` and `ETUDE_TRACK_DIR`. The `simple` and `attention` presets still
expect CUDA. Historical W&B model readers and frozen recipe fields remain
readable; those fields no longer select the live Trainer tracker.

Self-play TrainingRegime execution logs cumulative update/transition clocks and
retains admitted initialization, exact monitoring milestones, and raw stage
exports in Etude. These are policy artifacts, not portable training recovery. TrainingRun/VerifyStore still hold
scientific execution records, and the existing scheduler/evaluator has not yet
been replaced by Etude jobs. This is a partial migration, not end-to-end experiment
admission. SDK artifact retention alone does not establish portable collector/RNG
recovery. Remote storage, rentals and scientific runs require explicit allocation.

Override any hyperparameter with `--set dotted.path=value`; presets live in
`manabot/config/presets.py`.

`--recipe baseline` selects a complete ordinary training run above the regime
layer: current-world Allies/Lessons self-play (128 updates by default), its horizon-dependent schedule,
checkpoint exports, and paired outcome evaluation. `--steps N` counts completed
collection/update iterations (four streams, at least 64 combined seat decisions
each, finishing current games before updating), not optimizer
calls. The same recipe drives the [nightly learning check](../docs/nightly-learnability.md).
The bare command remains the small `local` smoke run. Use `--out` for a new output
directory; otherwise the recipe creates a timestamped directory under `.runs/`.
`--seconds` sets a failure deadline without changing the step schedule. Historical
`--regime` and `--preset` routes retain their meanings and cannot be combined with
`--recipe`. The baseline is a versioned starting point, not a strength certification.

For GPU training, start from `batched_baseline()` in `experiments/batched_baseline.py`. It is the
same 64×2 model trained on 64 streams for 524,288 learner transitions (128
iterations), with the learning-rate schedule mapped onto 2,048 four-stream
iterations. In the [longer size and batching comparison](../experiments/longer-learning-017.md)
it met the competence thresholds on both seeds within 7 minutes on an RTX 5090, and the
full run takes about 13 minutes versus about 110 at four streams; larger models showed no advantage at this
horizon. The regime is that study's exact tested recipe; it requires CUDA.

For an identified Allies/Lessons opponent, use the bounded
[train/export/play workflow](../docs/local-trained-challenger.md). It uses the
existing search-distillation trainer and a server-pinned candidate; players see
the bot's identity in the ordinary table. The `local` PPO preset above remains
a separate default-deck training smoke.

Build named curated matches with `MatchHypers.authored(pack_key, hero_deck,
villain_deck)` so both main decks and sideboards come from managym's compiled
setup. Custom `hero_sideboard` / `villain_sideboard` maps default to empty and
require positive integer counts. `Match.swapped()` moves both lists together;
`Env.reset(options={"match": match})` also replaces the setup used by auto-reset.


Submit bounded cloud work and reconnect after the laptop disconnects with
[`manabot deploy submit/status/fetch`](../docs/remote-jobs.md). Training, checkpoint
evaluation, verified S3 snapshots and shutdown belong to the remote job; reconnecting
does not restart the learner or promise CUDA process recovery.
Use `Experiment.cohort` and `manabot deploy cohort start` on an always-on host to
sequence a frozen cohort independently of the submitting terminal or chat.
Cohort status/cancellation reconnect through the same private control storage.
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
arena retain pending Commands before execution. The same bounded replay command
handles strict failures without enabling tolerance or process recovery; see the
linked capture audit for boundaries and supervision limits.

## World identity

Ordinary checkpoints require `world_binding` from
`manabot.model.world.checkpoint_world(actual_player_configs, observation_space)`.
This binds native `WORLD_VERSION`, rules schemas, full deck/sideboard pairs,
content manifest and input schema. Missing or mismatched bindings fail before
weights load. Seat reversal preserves each deck's sideboard. BC and value
writers require explicit `player_configs`; never infer training setup from
tensor dimensions or the default deck.

For the compiled Allies/Lessons matchup, use
`AgentSpec(semantic_pack="ur-lessons-vs-gw-allies")`. The ordinary model joins
viewer-safe `semantic_cards` transport IDs to complete checked catalog programs,
including characteristics and referenced definitions; `known_hand` contains
public definition minima, not hidden hand slots. Selected compiled checkpoints
without that semantic input fail admission. Generic uncompiled training keeps
its existing bounded feature model. These models do not certify strength or
semantic transfer; no previous checkpoint is relabeled compatible.

`AgentSpec(policy_film=True)` scales each action embedding by the pooled game
state before the policy head, so offer choices that depend on the state stay
learnable from any start. New runs train it: `new_agent_spec()` is the
`manabot train` default, and the ordinary baseline recipe is `baseline-v3`.
Omitted, the field serializes as the historical additive policy, so frozen
recipes, presets (`ataraxos-mtg-v1`, `baseline-v1` receipts) and checkpoints
keep their meaning. The learnability gate trains the same FiLM policy.

An observation/action-shape or rules-meaning change is a world version.
The current development world is **w11** (`managym.WORLD_VERSION`); its rules, tensor shape,
and compatibility limits are in [WORLDS.md](../WORLDS.md). Earlier checkpoints,
including w10 artifacts, are not compatible with w11. Elemental combat,
[sacrifice/landfall](../docs/rules/elemental-sacrifice.md) and [Temur](../docs/rules/elemental-temur.md) compiled admission is implemented;
[ten-deck Elemental admission](../docs/rules/elemental/world.md) is implemented in
`elemental-world-v2`; release certification remains unfinished. The local
challenger workflow records the native world identity and current rules,
content, Lesson-pool and tensor identities with each run. Historical Learn
certification limits remain in the [Learn record](../docs/rules/learn-lesson.md).

`train_bc` and `train_search_supervised` take the `observation_hypers` the
shards were encoded with and reject a dataset of any other shape.

## Architecture

manabot owns memory, beliefs, planning, policy/value learning, teacher data,
self-play, evaluation, and promotion evidence over the authoritative managym
world. The cross-package contracts and convergence status are in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   (`VectorEnv`, `ObservationSpace`, `Match`, `Reward`)
2. **`manabot.model`**: trainable policy and value models (`Agent`, PPO
   `Trainer`)
3. **`manabot.sim`**: search, teacher data generation, and game simulation
   (`Player`, `Sim`, determinized PUCT, Teacher-1 evidence)
4. **`manabot.semantic`**: semantic-program encoders and structural
   representation experiments
5. **`manabot.verify`**: competency scenarios, behavioral probes, and the
   run-provenance store
6. **`manabot.infra`**: experiment tracking (W&B/TensorBoard), `Hypers`
   config models, profiling
7. **`manabot.belief`**: canonical world distributions, viewer history,
   supervised exact-world learning, and the `ManabotPlayer` lifecycle
8. **`manabot.study`**: behavioral study of recorded games (`record_games`,
   `measures`, `report`), with one subpackage per matchup

Belief input is opt-in: schema-bound checkpoints load through the ordinary
`checkpoint` player and update belief before each policy/value decision.
`belief-demo` uses a freshly initialized policy to prove intervention wiring;
`belief-learn-demo` trains a belief scorer on one scripted population and holds
out whole episodes. Neither demonstrates strategic strength or opponent
transfer. The generic PPO and teacher-shard trainers do not yet produce the
semantic belief inputs required to train a belief-enabled policy. Historical
positional-condition checkpoints are rejected rather than reinterpreted.

Training regimes can [import a published raw/EMA policy](../docs/training-regimes.md#reuse-a-published-policy-without-training)
without retraining, preserving producer provenance and separate sunk costs.
They can freeze a policy, collect private whole-game supervision,
and fit a constrained autoregressive hand sampler without exact enumeration.
See [frozen-policy belief sampling](../docs/belief-sampler.md). This separately
admitted belief artifact does not change the policy's input contract or establish
foreign-opponent calibration or search strength.

Frozen-policy local search and same-root distillation run through
[`collect_local_update` regimes](../docs/local-policy-search.md). The bounded
compatible-prior recipe completes games; exact-history search rejects unsupported
Rules likelihood events. Target receipts and arena replay retain this distinction.

The existing `agent_puct` player accepts current flat and compound checkpoints;
see [PUCT policy/value compatibility](../docs/local-policy-search.md#puct-policyvalue-compatibility)
for critic units, retained prefixes and arena registration. This is a search
compatibility path, not a calibrated strength preset.

Declare training comparisons with [typed Python experiments](../docs/training-experiments.md).
Versioned presets, named component overrides and variant matrices resolve to complete
existing regimes with provenance; resolution never launches training.

Each current experiment is one module in [experiments/](../experiments/), holding
its recipes and anything only it needs to run and report; `manabot/` keeps
the reusable instruments. The experiment discipline and ledger are in
[experiments/README.md](../experiments/README.md).

Study records retain viewer-safe target source names and target controller sides.
Audit retained Allies/Lessons records without starting games with
`uv run python -m manabot.study.allies_lessons.targeting_audit --games games.jsonl.gz --label greedy`.
The output lists LvA, AvL, LvL and AvA (the selected player's deck first), split by engine
seat, including absent cohorts, unfinished records, declined targets and unknown
identities. Target offers list player 0's objects first, so a first-offer
policy's target side depends on its seat.
Older records remain readable and count as unmeasured target prompts; source-less
trigger choices remain unknown. Labels identify recorded seats, not a verified
policy implementation. Own-side targeting alone does not establish a misplay,
and these dependent choice counts are not game-level strength estimates.

`heuristic` (`manabot/arena/heuristic.py`, Study label `heuristic`) is a
separately versioned candidate anchor of rules of thumb: highest-mana-value casts,
removal held for a kill, counters held for an opponent's spell, effect polarity
derived from the semantic card source, attacks unless a blocker kills and
survives, and value blocks. Prompts without a rule match `scripted_greedy`, which
stays frozen with its arena source identity. Select it for checkpoint monitoring
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

TrainingRun diagnostics can be followed live or backfilled into W&B without
retraining. The [dashboard guide](../docs/training-monitoring.md) covers offline
operation, fixed distillation validation and separate checkpoint monitoring.

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

`manabot.training.etude_job.TrainingJob` freezes a single fresh self-play
`TrainingRegime` and its interior update milestones into an `etude.launch.Job`.
Its `job(repo=..., revision=..., max_hours=...)` method only builds the job;
`job.runs(*seeds)` only declares runs. Neither submits or reserves resources.
The job command is:

```sh
uv run --locked python -m manabot.training.etude_job
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

The shared arena records the actual decks, explicit content pack and Commands;
its ordinary replay checks still apply. The matrix identity is separate from
the historical selected-world suite. Omitting it preserves existing selected
protocol serialization. A native authored roster does not itself admit a neural
policy catalog. Research report presentation and Etude evaluation recording are
still separate remaining integration work.

The [learnability diagnostic](../docs/nightly-learnability.md) trains
three shared tiny generic policies, fans out identity/capacity assertions, and
checks reward-driven production PPO plus checkpoint reload. Its credential-free
CPU runner and failure reports are separate from full-game strength evidence.
