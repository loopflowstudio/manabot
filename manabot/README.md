# manabot

The trainable agent of Manabot: environment wrappers, models, search,
training, and verification. You train *a* manabot; this package is how.

The central loop is self-play: play a batch, turn game results into per-decision
credit, filter, update, repeat. [How a manabot learns](../docs/learning.md)
explains it, [training.md](../docs/training.md) covers running it, and
[training loops](../docs/training-loop.md) lists the updates we maintain,
their pieces and a loop to copy.

## Quickstart

```bash
uv run manabot train              # bounded local production-path smoke
uv run manabot train --recipe baseline  # ordinary self-play + evaluation
uv run manabot train --regime-name batched_baseline --out .runs/batched/run  # batched GPU baseline
uv run manabot train --learning-seats one  # explicit one-seat control
uv run manabot sim --preset sim --set sim.hero=random --set sim.villain=default
uv run manabot belief-demo        # engine-derived belief/intervention proof
uv run manabot belief-learn-demo  # held-out frozen-population belief proof
```

The default `local-smoke-v1` recipe runs two collection/update iterations
through the production regime executor: two streams, 16 decisions per stream as
a floor, both seats and complete games, with a 120-second failure deadline.
It uses the baseline model and horizon-scaled damped policy objective, records
actual counts through Etude and exports admitted checkpoints. `--steps`,
`--seconds`, `--seed`, `--out` and `--learning-seats one|both` make local controls
explicit. This is a software smoke, not playing-strength evidence.

The local recipe selects CUDA when available and CPU otherwise; Apple MPS is used
only with an explicit `--device mps`. `--recipe baseline` requires CUDA or an
explicit `--device mps`. `--device cpu|mps|cuda` overrides the local device;
baseline rejects CPU. MPS self-play uses float32 and requires
Torch 2.15 or newer; evaluation
and search run on CPU. MPS process recovery, belief inputs and frozen opponents
are unsupported. This makes no speed or strength claim.

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
calls. Its CPU test variant drives the [nightly learning check](../.github/workflows/baseline.yml).
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

Build named curated matches with `MatchConfig.authored(setup, hero_deck,
villain_deck)` (a preset or its `pack_key`) so both main decks and sideboards
come from managym's authored setup; like every match, it runs on the world's one
catalog. Custom `hero_sideboard` / `villain_sideboard` maps default to empty and
require positive integer counts. `MatchConfig.decklists` is `"open"` (default:
both players know each other's main deck and sideboard) or `"closed"`; search
players cannot play closed matches. `Match.swapped()` moves both lists together;
`Env.reset(options={"match": match})` also replaces the setup used by later resets.


Train on rented machines through Etude: see [remote training](../docs/training.md#remote-training).
New self-play jobs use exact absolute update checkpoints (`checkpoint_updates`,
default 1,000); freeze targets and shared intervals before launch. Operational
lease and spending limits remain in hours/dollars. See [cadence and continuation](../docs/training.md#monitoring).

New self-play training learns from both seats by default (`baseline-v3`).
Use `TrainSelfPlay(learning_seats="one", ...)` for an explicit one-seat control;
saved schema-1 recipes retain their historical one-seat behavior.
Batches close after complete games by default, whoever learns; `batch_closure`
declares it, and `"exact-quota"` is the one-seat alternative. See
[batch closure](../docs/training.md#batch-closure).
Use `games_per_stream` on both arms to match complete-game budgets. Distinct
learning models can use `PairedLearners` to update both independent optimizers
from their own seat trajectories. See [both-seat learning](../docs/training.md#regimes-and-stages)
for collection boundaries, accounting and measurement limits.

Single-stage self-play can opt into bounded [game crash recovery](../docs/training.md#game-failures).
It retains private replay inputs, discards both seats' uncommitted experience from
failed games and regathers under the same policy. A persisted rolling error rate
strictly above the declared threshold stops training; evaluation remains strict.
Strict scalar/vector training and data-generation paths automatically retain
private native failure inputs before propagating errors. Selection and
simulation retain pending Commands before execution. The same bounded replay command
handles strict failures without enabling tolerance or process recovery; see the
linked capture audit for boundaries and supervision limits.

The TLA frontier run uses `experiments/tla_corpus.py` for the fixed admitted
corpus and `experiments/tla_frontier.py` for Etude training and paired
evaluation ([report](../experiments/tla-frontier-c5fba4-results.md)).
The earlier trophy pilots execute from their pinned source; see the
[pilot evidence](../experiments/tla-trophy-pilot.md).

`UniformDecks.sampling` selects
independent per-seat draws (including mirrors) or the retained distinct-pair
contract. Stable population entry IDs distinguish events even when lists match;
checkpoints still bind the world, never the population.

Run its local evaluation from the training run's pinned source and native world.
Local Etude evaluation needs an explicit directory for its replay evidence
(rented jobs receive this directory from the launcher):

```sh
ETUDE_OUTPUT_DIR="$PWD/.runs/tla-frontier-evaluation" uv run --extra artifacts --extra evaluation etude eval experiments/tla_frontier.py --source r2://manabot/jobs
```

## Local-update search

Construct a search player around an admitted local checkpoint:

```python
from pathlib import Path

from manabot.infra.identity import file_sha256
from manabot.sim.search.flat_mc import make_player

checkpoint = Path(".runs/policy.safetensors")  # Your existing compatible checkpoint.
player, observation_space = make_player(
    {
        "kind": "local_update",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": {"worlds": 16, "depth": 20, "decision_seconds": 30},
    },
    seed=42,
)
```

This constructs a `LocalUpdatePlayer` and its checkpoint's observation space;
it does not start a game. Use it through the simulation player lifecycle, which
starts viewer history and advances it after each transition. The checkpoint
must match the match's world, use observation-only inputs, and supply either a
categorical win/draw/loss value or a scalar declared as `signed_outcome`.
Search requires open decklists.

At each decision, search samples 16 worlds and rolls out every root action with
positive base-policy probability in each world, for up to 20 policy choices
including the root action. Native trivial-decision skipping remains enabled.
All live branches advance together: after each step, branches needing a policy
and value share one network forward. Finished games leave the batch. Each
branch retains its own random stream and uses only its acting player's
observation. Terminal wins, losses and draws score +1, −1 and 0 for the root
player; unfinished leaves use the critic value, sign-flipped when the leaf's
actor is the other player.

Batching is automatic; `worlds` and `depth` still control search work. Defaults
are one world, depth two and a one-second decision budget. Sampling defaults to
`compatible_prior`; `full_probability=1` uses all requested worlds. Lowering it
selects a cheaper allocation of `min(cheap_worlds, worlds)` on the remaining
decisions (`cheap_worlds=1` by default). The local update uses `alpha=0.02` and
`beta=1.0` by default, retaining zero probability outside the base policy's
support. Batching does not establish a playing-strength gain or a fixed latency.
CPU inference omits object columns unused by the entire batch from attention,
then restores their original coordinates before action scoring and value pooling.
It also omits unused fact columns before embedding them, preserving each
program's row order and object joins. Both selections are rebuilt per forward;
no state-derived embeddings are reused between decisions.
Legal choices, observation capacities and checkpoint parameters stay intact.
The native encoder fills fresh batch rows using the checkpoint's selected
input layout and capacities. Search still requires every legal action, including
for checkpoints that allow action truncation during training.

After a successful choice, `player.last_receipt` holds the target distribution,
root-action values and per-rollout seeds, signed values, terminal flags,
actor-observation hashes and branch audits. These are private replay evidence.
Per-ply hashes remain SHA-256 of the semantic observation's fixed-viewer JSON.
Search reads them without constructing the unused composite decision envelope.
Each read rebuilds the projection; presentation events and witness hashes keep
their separate meanings.
Replay requires the original root and viewer history, checkpoint, native
runtime and teacher source; keep their identities with the receipt. Batched float32
inference and padding removal can round slightly differently from
singleton inference over the full padded layout. Rollout seeds
remain independent, but a probability at a sampling threshold can turn that
rounding into a different trajectory; replay uses the retained allocation and
implementation identity.

### Rollout evidence

Search retains compact evidence by default. An audit run selects full evidence
in the search config to keep a per-Command witness and viewer hash for every
rollout step:

```python
player, observation_space = make_player(
    {
        "kind": "local_update",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": {
            "worlds": 16,
            "depth": 20,
            "decision_seconds": 30,
            "rollout_evidence": "full",
        },
    },
    seed=42,
)
```

Omit `"rollout_evidence"` (or set it to `"compact"`) and
`player.last_receipt.schema` is `regularized-local-update/v4`. Every rollout
keeps its seed, signed value, terminal flag and Command count. Its
`actor_observation_hashes` list is empty, and its branch audit records
`audit: false` with empty tapes. During rollouts, search records no native
witness for each Command and computes no viewer hash for each actor. Each
Command it applies still increments a count, and the receipt checks that count
against the native session's counters. Search still checks the root viewer,
validates every sampled world, checks Command preconditions and keeps the same
policy inputs and rollout budgets. Compact receipts cannot show that each
rollout step matches the witness that full evidence would have recorded.
`verify_replay` checks only the evidence a receipt retains.

`rollout_evidence="full"` writes `regularized-local-update/v3` receipts, with
the same `config_json` bytes that receipts had before the field existed. For
the same checkpoint, root, history and seed, compact and full evidence choose
the same Commands and produce the same tensors and values; they differ only in
what the receipt retains.

The evidence mode is part of the implementation and receipt identity. Readers
accept historical v1–v4 receipts, and a v1–v3 receipt is always read as full
evidence. A registration whose config omits `rollout_evidence` predates the
field and also means full evidence, so frozen registrations keep their
meaning; a new registration states the mode, and one selecting compact evidence
declares `search_semantics.branch_audit: false`. Changing the default never
relabels an existing experiment's receipts or registrations.

### Native frontier encoding

Each teacher keeps frontier buffers from
`ObservationEncoder.allocate(capacity)` for its whole life. Native code writes
every live branch's acting-player observation into these buffers with one call,
`managym.encode_observations_into(envs, buffers, capacity)`, which returns each
row's legal-action count. The buffers grow only when a frontier exceeds their
capacity, and rows past the live frontier are left untouched. The tensor layout
is the checkpoint's own, and every legal action is still required. The encoder
leaves inference threads, rollout depth, legal support and the search config
unchanged, and is not recorded in the receipt, so registered evaluation players
use it without changing their identity.

`LocalUpdateTeacher(..., native_encoder=False)`, or a top-level
`"native_encoder": false` beside `"config"` in a `make_player` spec, instead
allocates a fresh batch for each frontier forward and fills its rows one branch
at a time in Python. Use it to compare encoders; both write identical tensors.

A teacher reuses its buffers between calls, so only one search may use it at a
time; give each concurrent search its own teacher. An extension without the
batch writer, or a buffer whose shape, dtype or writability does not match the
selected observation space, raises before any inference. Search never falls
back to the Python encoder: rebuild the extension, or select
`native_encoder=False` explicitly.

### Inference threads

Search never changes Torch's thread settings. A process chooses its intra-op
thread count once, before constructing a teacher, with
`torch.set_num_threads(n)`. Evaluation game children always use one thread.
`scripts/bench_local_update.py --inference-threads N` (default 1) sets the
count before loading the model and records the requested and effective intra-op
counts, plus the inter-op count, which it leaves unchanged. A single-process
measurement with more threads does not predict throughput when several workers
share the machine.

### Budgets and failures

The decision budget includes viewer-history work. Exceeding it raises
`TimeoutError` with no target; increase `decision_seconds` or reduce `worlds` or
`depth` before retrying. The 30-second example is an explicit budget, not a
timing estimate. An incompatible checkpoint, incomplete legal-action encoding
or stale history is an error, not a partial search result. Use a compatible
checkpoint and observation space, and keep the player lifecycle current.
[Removed compound policies](../docs/training.md#compound-decisions-removed)
remain rejected; their historical receipts remain readable.

The [measurement record](../docs/benchmarks/local-update.md) separates the
equal-work batching comparison from the later canonical-writer optimization,
including the unresolved near-singleton latency target.

## World identity

Ordinary checkpoints require `world_binding` from
`manabot.model.world.checkpoint_world(actual_player_configs, observation_space)`.
This binds native `WORLD_VERSION`, rules schemas, full deck/sideboard pairs,
content manifest and input schema. Deck-source runs bind the world itself
(`deck_world`: version, card digest, rules and input contract), so the model
plays any deck the world admits. Missing or mismatched bindings fail before
weights load. Seat reversal preserves each deck's sideboard. Policy checkpoint
writers require explicit `player_configs`; never infer training setup from
tensor dimensions or the default deck.

For the semantic card model, use `ModelConfig(world="world")`. The ordinary
model joins
viewer-safe `semantic_cards` transport IDs to complete checked catalog programs,
including characteristics and referenced definitions; `known_hand` contains
public definition minima, not hidden hand slots, and `decklists` rows hold
each known decklist's main, unseen, sideboard and remaining-sideboard counts. Selected compiled checkpoints
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
The current development world is **w15** (`managym.WORLD_VERSION`); its rules, tensor shape,
and compatibility limits are in [WORLDS.md](../WORLDS.md). Earlier checkpoints,
including w14 artifacts, are not compatible with w15. Elemental combat,
[sacrifice/landfall](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/elemental-sacrifice.md) and [Temur](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/elemental-temur.md) compiled admission is implemented;
[ten-deck Elemental admission](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/elemental/world.md) is implemented in
`elemental-world-v2`; release certification remains unfinished. The local
challenger workflow records the native world identity and current rules,
content, Lesson-pool and tensor identities with each run. Historical Learn
certification limits remain in the [Learn record](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/learn-lesson.md).

`manabot sim` accepts `random`, `default`, a local policy file (`.safetensors`,
or a `.pt` saved by an earlier revision), or an
Etude checkpoint-record `.json` path for each player. Checkpoint records resolve
exact digest-bound local/S3 artifacts before normal world and architecture
admission. Simulation results are recorded locally through Etude.

## Architecture

manabot owns memory, beliefs, planning, policy/value learning,
self-play, evaluation, and promotion evidence over the authoritative managym
world. The cross-package contracts are in the
[README](../README.md#contracts-the-code-enforces).

1. **`manabot.env`**: Gymnasium-compatible wrapper around managym
   (`VectorEnv`, `ObservationSpace`, `Match`, `Reward`)
2. **`manabot.model`**: trainable policy and value models (`ActorCritic`);
   `manabot.training` owns the regime executor and `Learner`
3. **`manabot.sim`**: players and game runners.
   `sim/action` owns the structured-offer prototype decoder and its benchmark.
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

Training regimes can [import a published raw/EMA policy](../docs/training.md#regimes-and-stages)
without retraining, preserving producer provenance and separate sunk costs.
They can freeze a policy, collect private whole-game supervision,
and fit a constrained autoregressive hand sampler without exact enumeration.
See [frozen-policy belief sampling](../README.md#beside-the-loop). This separately
admitted belief artifact does not change the policy's input contract or establish
foreign-opponent calibration or search strength.

[Frozen-policy local search](../README.md#beside-the-loop) plays through the
`local_update` player kind; exact-history search rejects unsupported Rules
likelihood events. Search receipts and simulation replay retain this distinction.

The existing `agent_puct` player accepts flat checkpoints;
see [PUCT policy/value compatibility](../README.md#beside-the-loop)
for critic units and evaluation registration. This is a search
compatibility path, not a calibrated strength preset.

Declare training comparisons with [typed Python experiments](../docs/experiments.md#declaring-a-comparison).
Versioned presets, named component overrides and variant matrices resolve to complete
existing regimes with provenance; resolution never launches training.

Each current experiment is one module in [experiments/](../experiments/), holding
its recipes and anything only it needs to run and report; `manabot/` keeps
the reusable instruments. The experiment discipline and ledger are in
[experiments/README.md](../experiments/README.md).

The reference league's TrainingJob enables private exposure capture through the
ordinary native vector collector. Digest-bound game receipts preserve actual
sampled decks, seed/action replay inputs, per-seat nontrivial options and choices,
card draws and completed effect frames. They live under `training/exposure`,
separate from checkpoints and evaluation evidence. The offline analysis accepts
these downloaded directories with `--training-evidence`; missing or interrupted
capture leaves the report incomplete. Focus opportunities and completed frames
are measurements of decisions/program completion, not strategic benefit. Other
jobs keep capture disabled unless they explicitly select it.

The retired behavioral-study and structural/opcode experiment tools are retained
in Git history. Their archived inputs and receipts are indexed in
[repository archive](../docs/development.md#retained-evidence-and-archived-corpora). Shared target inspection
now belongs to `sim/action/targeting.py`.

Select `MonitorProtocol(opponent="heuristic")` for checkpoint monitoring, or
`VectorEnv(..., player_policies=("agent", "heuristic"))` for a native batched opponent.
The scalar `HeuristicPlayer` adapter and arena anchor call the same engine
implementation. `heuristic` is a separately versioned anchor with rules of thumb
for mana, timing, targets, card choices and combat; see
[selection, rules and information limits](../docs/engine.md#built-in-heuristic).
Historical results keep the heuristic version they evaluated. One exploratory
check (MTG-152, GPU seed 12261 at 25k, 200 paired games per anchor) scored the
checkpoint 0.57 against `heuristic` and 0.795 against `scripted_greedy`, lower
in every cell; it is not yet a frozen anchor comparison.

For versioned S1–S5 tactical diagnostics through ordinary checkpoint players,
see [checkpoint scenario scoring](../docs/evaluation.md#checkpoint-scenarios). Each retained
custom-deck root requires its exact checkpoint setup binding.

For capacity configuration, component counts, versioned architecture receipts and
bounded CPU calibration, see [capacity accounting](model/agent.md#capacity-identity-and-accounting)
and [the calibration command](../docs/experiments.md#freeze-step-targets).

## Shared effect inputs

The policy input encodes each resolved effect once, before per-holder fact
expansion. `env/shared_effects.py` mirrors the native 128 × 18 header,
4,096 × 3 detail and 2,048 × 2 membership buffers. Details carry structured
granted, copy-exception and continuation programs, changed colors and
keyword-counter timestamps. Local counters, subtypes, names, used abilities and
copy exceptions use `facts` (default 512 rows). The model encodes each payload once and routes its contribution into
named objects before attention; action focus remains addressable. Sum reduction
preserves edge multiplicity and shared gradients; it is learned compression,
not a proof of injective embeddings or improved play.

Shared-effect and typed-continuation routing validate one host metadata batch,
then gather live features, project tokens and reduce all records together on the
model device. Ordered GRU packing avoids per-record padded copies; pooling keeps
the unnormalized sum. Input features, object joins and encoder parameters retain
their gradients, including duplicate memberships. An unused continuation module
keeps absent gradients so Adam does not update its momentum spuriously.

The input version is explicit in saved observation and configuration
identities, and main loads only v9; see
[policy inputs](../docs/observation-and-model.md#policy-inputs) for the layout
and the saved-checkpoint and world admission requirements. Text representation
`manabot-text-6` renders each native payload once with exact public source and
member labels and timestamp ranks. Its within-observation token reducer is not
persistent game memory.

## Policy action capacity

Observation overflow raises an error by default. The TLA pilot explicitly uses
`ObservationConfig(max_actions=64, action_overflow="first-legal")` to keep the
first 64 legal actions in native order. Retained indices and Commands are
unchanged; the engine and Academy still expose the complete legal set. Learned
checkpoint evaluation uses the saved policy configuration. The prospective TLA
benchmark gives its heuristic opponent the same prefix before scoring, including
its fallback; the chosen original index is never clamped or remapped.

Capture reports count consumed decisions, overflows and dropped actions by
decision kind across all participating policies, with bounded replay samples.
These aggregate counts are not learner-only or actor-specific rates.
Collector/game close and vector
reseeding flush exact totals, including zero-overflow runs. The approximation
and its evaluation limits are specified in the
[TLA v10 protocol](../experiments/protocols/tla-trophy-213-v10.md).

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
from manabot.env.observation import ObservationSpace

# Local imports
from .sibling import Thing
```

The learned compound-decision decoder has been removed; checkpoints and
regimes that select it fail validation. See
[compound decisions](../docs/training.md#compound-decisions-removed).

Self-play update diagnostics live in the run's Etude metrics and referenced
outcome tables; TrainingRun keeps counters and receipts. The [dashboard guide](../docs/training.md#monitoring)
covers reading them offline and separate checkpoint monitoring.

Optional [public recent-event input](../docs/observation-and-model.md#policy-inputs) uses the existing
semantic catalog and a bounded native history suffix. Recipe history-on/off is
independent of capacity and learning rules; comparison plans remain unexecuted.

Self-play [numerical health](../docs/training.md#numerical-health) records policy support,
selection, gradients and actual optimizer effects, with bounded private failure
artifacts. CPU correctness does not imply CUDA validation or improved play.

For exploratory same-deck checkpoint play, the [head-to-head runner](../experiments/head-to-head.md#mirror-only-follow-up-mtg-131-2026-10-09)
accepts `--matchups mirrors`. This explicitly permits repeating exact checkpoint
roster decks/sideboards and carries them through replay;
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
The [game-failures guide](../docs/training.md#game-failures) describes opt-in
`GameErrorPolicy.recover_capacity`, its recovery boundaries and evaluation limits.

Policy checkpoints cannot resume optimizer, RNG or collector state. Automatic
retries are disabled (`max_attempts=1`); multi-stage and imported-state jobs are
not yet admitted. This worker proof does not establish SDK placement, remote
provisioning, scientific evaluation, artifact upload or learning strength.

### Semantic catalog dimensions

`SemanticCardEncoder` uses a shared GRU over semantic-token embeddings, not an
independent learned embedding per card. Its `catalog_shape` reports definition
count, unique token count, total expanded sequence length, longest sequence and
padded token count. Referenced definitions are expanded into the parent sequence.
Integer token IDs are packed before embedding, so padding receives no embedding
or gradient work. The same GRU processes at most 64 time steps per call, carrying
its hidden state and full gradients across calls. This bounds the temporary
slice-gradient copies in packed CPU GRU backward without truncating a program
or caching trainable outputs across updates. Completed states return to their
original definition rows. These counts describe representation cost, not
observed learning difficulty; chunking changes no checkpoint weights or buffers.

Fresh models (`ModelConfig.record_vocabulary`,
default `catalog`) embed fact and shared-effect ids through the same token
embedding. Native `record_vocabulary` lists every counter kind, color,
subtype, keyword, modification, set name and combat restriction the encoders
hash. Each reads the row of the card token for the same concept (a granted
`flying` and a printed `Keyword::flying` share one row) or a row appended after
the card tokens. Captured program tokens read their card-token rows; integers
and booleans read one row per kind plus their exact value. Fact and effect
sources read the card encoder's row for that definition, and ability or frame
ordinals are numbers. An id outside the vocabulary fails. World checkpoints
that omit the field load the `hashed` layout through `ModelConfig.from_checkpoint`
(elsewhere both values build the same model and omission reads as the default): 4096
learned rows per id column, which held 80% of a w64 L2 model (1,958,402
parameters; 412,482 with catalog ids) and 42% at w160 L8 h8 (9,470,402;
5,605,602). Models without a world keep only the hashed layout.

`manabot.semantic.policy_catalog` owns explicit native admission. Policy
construction checks the engine's learning catalog against the world pool's
content digest and complete native coverage before joining transport IDs. A
policy has one vocabulary, the world's catalog: fresh recipes write
`ModelConfig(world="world")`, and the older labels in
`FROZEN_VOCABULARY_LABELS` are accepted on read, unchanged, because frozen
recipes and architecture receipts record them.

### Explicit research evaluation matrices

`AuthoredMatchup(pack_key=..., role_a=..., role_b=...)` names two exact native
authored setups. Pass it as `MonitorProtocol(authored_matchup=...,
include_mirrors=True)` to evaluate A–B, B–A, A–A and B–B, each with both player
seat assignments. The training match must equal `matrix.match()`; arbitrary
renaming or substituting a different setup fails admission. Simple roles retain
their own names, never Allies/Lessons labels.

The simulator records the actual decks and Commands;
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

- `policy/`: the framework-free policy contract: observation admission, bucket
  trimming, host preparation of the `Prepared` batch every backend's forward
  reads, the safetensors checkpoint container and the backend Protocols.
  It imports no array framework ([policy contract](policy/README.md)).
- `backends/<framework>/`: one folder per array framework with the same files
  (`policy.py`, `rl.py`, `learner.py`, `rollout.py`, `arrays.py`,
  `checkpoint.py`);
  `backends/torch/` is the reference and `backends/parity.py` checks the others
  against it ([backends](backends/README.md)).
  `backends.torch.checkpoint.save_checkpoint` and `load_checkpoint` own policy
  files: `save_checkpoint` writes the shared container (`.safetensors`), and
  `load_checkpoint` reads that or a `.pt` file saved by an earlier revision,
  whose serialized keys keep their meaning.
  Library code receives a backend and never imports one; experiments,
  scripts, recipe entry points and the CLI choose it
  ([who chooses a backend](backends/README.md#who-chooses-a-backend)).
- `model/`: architecture receipts, the checkpoint catalog and world binding.
- `sim/`: registered players, bounded game execution, replay, native card evidence,
  search and batched games.
- `training/`: regime execution, seat-routed collection, learning rules,
  numerical health and optimizer/collector recovery.
  `regime.py` declares recipes; `records.py` holds measured run records.
  `execution.py` sequences stages and owns the run lifecycle.
- `training/rollout/`: on-policy collection, stream producers, matchups and
  complete-game accounting.
- `training/updates/`: an update learns from one collected batch. The damped
  policy update (`damped_policy.py`, the default), the PPO control (`ppo.py`)
  and their host-side pieces: numerical health and selection diagnostics.
  Each takes a backend object or module as its first argument and does its array math
  through it; recipe entry points pass `manabot.backends.torch`
  ([training loops](../docs/training-loop.md#the-updates-we-maintain)).
- `training/loops/`: a loop keeps playing and updating until it is done; a
  one-page example (`minimal.py`).
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

[GIH card evaluation](../docs/evaluation.md#card-data) uses simulation replay capture and Etude
tables for copy-weighted card rankings, explicit cohorts and offline Arena-log export.

Simulation imports no evaluation, training, experiment or report modules.
Shared registration types live in `sim.registration`; canonical byte identities
live in `infra.identity`. The former `manabot.arena` package is removed. Existing
serialized `arena_key`, `arena_version`, seed identifiers and retained artifact
paths preserve their original meaning; current source identities use the new paths.

For a standalone, inspectable pretrained-model decision over supplied text and
legal references, see the optional [LLM players](../docs/evaluation.md#llm-players). It owns
protocol recovery and inference evidence; observation rendering and Command
execution remain outside it.

### Shared evaluation inference

Frozen arena policies can share a bounded, single-machine inference allocation
through an explicit execution envelope. Monitoring, Etude benchmark and duel
callers retain native isolation, Commands, replay and original training
coordinates. See [shared evaluation inference](../docs/evaluation.md) for
admission, real Perf receipts and the predeclared local calibration command.

A quick local checkpoint comparison needs no TrainingRun or cloud setup:
`uv run python -m manabot.eval.local --checkpoint policy.safetensors --opponent heuristic --deal-seed 101 --out .runs/local-check --seconds 300`.
The [local evaluation API and CLI](../docs/evaluation.md)
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
began” fact; historical v3 captures retain unknown eligibility. The policy input
uses the same native fact, separately from current object presence.

Run the local response-stub proof with
`uv run python -m manabot.verify.text_observation --out .runs/text-observation`.
It retains native inputs, prompts, bindings, malformed/stale checks and execution
receipts. This uses bounded public policy history v1; it loads no model and has
no writable model memory. Tokenizer/model recovery belongs to the caller;
`TextDecision.check_token_budget` checks the complete prompt without truncation.

[Decklist file import](../docs/engine.md#decks) admits novel development decks
against the installed world for simulation and Academy play.
