# Functional audit: classes in manabot/ and academy/ (not academy/ui)

Rule: AGENTS.md "Functions and objects (Python)". 755 top-level classes were
enumerated with `ast` (`^class ` in `manabot/` and `academy/`). Most are
configurations, records, enums, protocols, exceptions and `nn.Module`s, which
earn their place; they are counted per module in the last table rather than
listed. This file records what changed here, what was kept and why, and what is
left for later.

## Converted

| Was | Now | File | Why |
|---|---|---|---|
| `GameStats` (mutable accumulator with a lock, 11 methods, dict rows) | frozen `GameRecord` per game; `simulate_models` returns `list[GameRecord]`; `simulation_summary`, `detailed_analysis`, `profiler_summary`, `behavior_summary` functions | `manabot/sim/sim.py` | an object sequencing record/summarise work; summaries are functions of the records |
| `Player` base class + `PlayerType` enum, `to()` (unused) | `Player` `Protocol`; `ModelPlayer`/`RandomPlayer`/`DefaultPlayer` implement it | `manabot/sim/player.py` | interchangeable parts declared as a Protocol, no inheritance for an interface |
| `ArtifactStore` (our class wrapping boto3 clients to add methods) | `publish_artifact`, `download_artifact`, `fetch_artifact` over a private `_client(scheme, profile)` | `manabot/infra/artifacts.py`, `manabot/data.py` | "do not wrap another library's handle in our own class to add methods" |
| `CallbackEndpoint` (class whose `__enter__` returned a different object) | `serve_callbacks(runtime)` `@contextmanager` | `manabot/eval/endpoint.py` | a lifetime with no state for callers to hold |
| `ThreadBudget.account`, `ProcessCensus.take/.record/.enforce` (records that measure the process) | `account_threads`, `take_census`, `record_census`, `enforce_budget`; both records are plain | `manabot/sim/threads.py`, call sites in `sim/native_game.py`, `sim/serving.py`, `sim/supervision.py`, `eval/match.py` | records do not run anything |
| `Opponent.verify()` (record reading a checkpoint) | private `_verify_checkpoint(opponent)` | `academy/opponent.py` | record does not do I/O |

## Kept: notable classes and the reason

Plain configs/records/enums/protocols/exceptions/modules are only counted in
the module table below. Classes with behaviour that were examined and kept:

- Resources with a lifetime: `VectorEnv`, `Env`, `ShardedVectorEnv`,
  `ObservationEncoder` (owns reusable encode buffers), `InferenceRuntime`,
  `InferenceClient`, `PolicyStore`, `PolicyInference`, `ChildProcess`,
  `VerifyStore`, `AttemptStore`, `OpenRouterSession`, `InferenceSession`,
  `VLLMServer`, `LocalUpdateTeacher` (admitted policy plus reused frontier
  buffers), `_ValueLeafScorer` (reused buffers).
- Guarded invariants: `Profiler`/`TimingNode`, `EvalPerformance` (thread-shared
  counters and bounded samples under one lock), `BudgetLedger`,
  `AdmittedRollout`, `AdmittedObservation`/`PreparedObservation`,
  `EvidenceReader` (each dependency read once and digest-bound),
  `BeliefTracker` (posterior and matched prior advance together),
  `belief.agent.Manabot` (policy and belief schema admitted together; `decide`
  already returns the next memory), `PresentationProjector` (per-match sequence
  numbers), `sim.search.rollout.Rollout`.
- Interchangeable parts behind one interface: every player (`HeuristicPlayer`,
  `ExactRangePlayer`, `LocalUpdatePlayer`, `DeterminizedPuctPlayer`,
  `ValueSearchPlayer`, `VGreedyPlayer`, `ManabotPlayer`, the sim/anchor and
  flat-MC players, academy villains), leaf evaluators, hand samplers, search
  session backends in `sim/search/branch.py`, LLM runtimes, scenario trackers
  in `verify/scenarios.py`, `_ReadSnapshot` (read-only Etude `Storage`),
  `FrozenPolicyLikelihood`/`CompatibleDealBeliefModel`.
- Values with pure derivations: `BeliefState`, `PackedActions`, `LiveTokens`,
  `RecordEmbeddings`, `BoundSemanticPack`, `SemanticCatalog`,
  `SemanticDecisionContract`, `ObservationSpace`.
- Required by a library: `_CallbackSupervisor` (BaseManager serves an object).

## Left for later

| Class | File | Reason |
|---|---|---|
| `Reward` | `manabot/env/match.py` | a config wrapper whose `compute` should be a function of `RewardConfig`; constructed in off-limits `training/execution.py`, `training/loops/self_play.py`, `env/target_practice.py` |
| `Match` (`to_rust*` build native configs) | `manabot/env/match.py` | data building runtime objects; called from off-limits files |
| `SelfPlayCounter` | `manabot/training/rollout/self_play.py` | mutable counter owned by off-limits `rollout/collection.py` |
| `AdaptiveSampler` | `manabot/training/rollout/adaptive.py` | draw/commit/snapshot/restore over mutable state; built in off-limits `collection.py` (a `state, draw = draw(state)` shape would fit) |
| `OptimizerHealth` | `manabot/training/learning/health.py` | sequences check/backward/clip/step; frozen `experiments/cuda_precision.py` and `experiments/l4_step_profile.py` monkeypatch `OptimizerHealth.check/.step`, so converting is not a mechanical rename |
| `ReportEvidence` | `manabot/reports/experiment.py` | cached properties read files; it is the notebook API (`tests/fixtures/experiment-demo/report-generator.ipynb`, `docs/experiment-metrics.md`) and holds `manabot.experiment` records slated for deletion |
| `_Capture`, `ScalarCapture`, `CapturedVector`, `CapturedSearch` | `manabot/env/failures.py` | proxy wrappers (`__getattr__`) around envs; woven through off-limits collection and recovery |
| `GameSession`, `StudyBranch` | `academy/server.py`, `academy/study_branch.py` | large stateful sessions sequencing play; high risk, need their own change |
| `_Renderer` | `manabot/semantic/text_observation.py` | 19-method private renderer; could be functions over a lookup record, low value |
| `BehaviorProbe` | `manabot/verify/competency.py` | per-game accumulator; low value |
| `CuratedPack.player_config` | `academy/curated_pack.py` | record building a native config; five call sites, low value |

## Slated for deletion (not converted)

`manabot/infra/runtime.py` `RunContext`; `manabot/experiment/{job,runner,evaluation_queue,monitor_worker,study,config}.py`; `manabot/training/state/admission.py`.

## Off-limits here

`manabot/training/rollout/collection.py`, `training/state/{game_errors,recovery,incidents}.py`,
`training/execution.py`, `training/records.py`, `training/loops/self_play.py`,
`env/target_practice.py` (concurrent rewrite); `cli.py`, `recipes/{elemental,belief_sampler,proposed_etude}.py`,
`training/learning/learner.py` (already converted).

## Every module

"Plain" counts the classes that are configs/records/enums/protocols/exceptions/modules.

| Module | Plain | Notable classes |
|---|---|---|
| `academy/advice.py` | 17 Protocol, 3 config/record, 1 exception |  |
| `academy/advice_identity.py` | 7 Protocol |  |
| `academy/attempts.py` | 1 config/record | `AttemptStore`: resource (SQLite store) |
| `academy/authored_match_parity.py` | 1 config/record, 1 exception |  |
| `academy/authored_match_receipt.py` | 1 exception | `DeterministicServerOfferPolicy`: interchangeable part (policy callable) |
| `academy/bot_decision.py` | 3 config/record |  |
| `academy/curated_pack.py` | - | `CuratedPack`: record; player_config builds a native config (minor, left) |
| `academy/deck_import.py` | 2 config/record |  |
| `academy/opponent.py` | 1 config/record | `Opponent`: record; verify() **moved** to _verify_checkpoint |
| `academy/play_demo.py` | 2 config/record |  |
| `academy/presentation.py` | 2 config/record | `PresentationProjector`: guarded invariant: sequence numbers and pending spells per match |
| `academy/public_commitment_parity.py` | 2 config/record |  |
| `academy/replay_index.py` | 7 Protocol, 1 enum, 4 exception |  |
| `academy/semantic_boundary.py` | 1 config/record |  |
| `academy/server.py` | 5 config/record, 3 exception | `SessionRecord`: record; `GameSession`: left (high risk): 1.5k-line session sequencing play |
| `academy/shared_table_protocol.py` | 37 Protocol, 2 enum |  |
| `academy/study_branch.py` | 2 config/record, 1 exception | `StudyBranch`: left (high risk): stateful study session; `StudyForkProvider`: interchangeable part |
| `academy/study_protocol.py` | 20 Protocol, 3 enum |  |
| `academy/study_runtime.py` | 1 Protocol, 4 config/record, 4 exception |  |
| `academy/trace.py` | 3 config/record |  |
| `academy/villain.py` | - | `RandomVillain`: interchangeable part (villain callable); `CheckpointVillain`: interchangeable part (villain callable) |
| `academy/watch.py` | 2 TypedDict, 1 config/record |  |
| `manabot/_process.py` | 1 config/record |  |
| `manabot/belief/agent.py` | 4 config/record | `Manabot`: guarded invariant: policy and belief schema admitted together; decide(decision, memory) already returns next memory |
| `manabot/belief/audit.py` | 4 config/record |  |
| `manabot/belief/encoding.py` | 4 config/record |  |
| `manabot/belief/likelihood.py` | 2 config/record, 1 exception | `FrozenPolicyLikelihood`: interchangeable part (ActionLikelihood) holding an admitted checkpoint |
| `manabot/belief/player.py` | 1 config/record, 1 enum | `ExactRangePlayer`: interchangeable part (search player interface) |
| `manabot/belief/population.py` | 10 config/record, 1 nn.Module |  |
| `manabot/belief/range.py` | 1 exception | `BeliefState`: value: frozen distribution with pure derivations |
| `manabot/belief/runtime.py` | - | `ManabotPlayer`: interchangeable part (player interface) |
| `manabot/belief/sampling.py` | 2 config/record, 1 nn.Module |  |
| `manabot/belief/sampling_cohort.py` | 7 config/record |  |
| `manabot/belief/sampling_data.py` | 3 config/record |  |
| `manabot/belief/sampling_fit.py` | 5 config/record |  |
| `manabot/belief/state.py` | 1 Protocol, 5 config/record | `CompatibleDealBeliefModel`: interchangeable part (BeliefModel) |
| `manabot/belief/tracker.py` | 1 Protocol, 3 config/record | `BeliefTracker`: guarded invariant: posterior and matched prior advance together per observation |
| `manabot/belief/train.py` | 1 config/record |  |
| `manabot/cli.py` | off-limits here (concurrent rewrite or already converted; 1 classes) | |
| `manabot/data.py` | 3 config/record, 1 exception |  |
| `manabot/env/card_capture.py` | 9 config/record |  |
| `manabot/env/env.py` | - | `Env`: resource; gym.Env interface |
| `manabot/env/failures.py` | 11 config/record | `_Capture`: guarded invariant: failure evidence retained once per game; left (see later); `ScalarCapture`: see _Capture; `CapturedVector`: resource wrapper; left (see later); `CapturedSearch`: see _Capture |
| `manabot/env/match.py` | - | `Match`: left (blocked): to_rust builds runtime objects; called from off-limits files; `Reward`: left (blocked): config wrapper whose compute is a function; constructed in off-limits files |
| `manabot/env/observation.py` | - | `ObservationEncoder`: resource owning reusable encode buffers bound to one space; `ObservationSpace`: gym.spaces.Space value: shapes derived from config |
| `manabot/env/opponents.py` | - | `RandomPolicy`: interchangeable part (policy callable) |
| `manabot/env/replay.py` | 1 config/record, 1 exception |  |
| `manabot/env/target_practice.py` | off-limits here (concurrent rewrite or already converted; 1 classes) | |
| `manabot/env/vector_env.py` | - | `VectorEnv`: resource (native vector env and its buffers) |
| `manabot/eval/admission.py` | 1 config/record |  |
| `manabot/eval/arena_log.py` | 2 config/record |  |
| `manabot/eval/benchmark.py` | 1 config/record |  |
| `manabot/eval/card_evaluation.py` | 1 config/record |  |
| `manabot/eval/card_metrics.py` | 2 config/record |  |
| `manabot/eval/card_query.py` | 9 config/record |  |
| `manabot/eval/card_schedule.py` | 1 config/record |  |
| `manabot/eval/duel.py` | 3 config/record |  |
| `manabot/eval/endpoint.py` | 1 config/record | `_CallbackSupervisor`: required: BaseManager serves an object; `CallbackEndpoint`: **converted** to serve_callbacks context manager |
| `manabot/eval/evaluate.py` | 1 exception |  |
| `manabot/eval/evaluation.py` | 2 config/record |  |
| `manabot/eval/models.py` | 3 config/record |  |
| `manabot/eval/monitor.py` | 2 config/record |  |
| `manabot/eval/rating.py` | 1 config/record |  |
| `manabot/eval/results.py` | 2 config/record |  |
| `manabot/eval/serving_benchmark.py` | 3 config/record |  |
| `manabot/experiment/attack_execution.py` | 2 config/record |  |
| `manabot/experiment/attacks.py` | 2 config/record |  |
| `manabot/experiment/checkpoint_evaluation.py` | 1 config/record |  |
| `manabot/experiment/config.py` | slated for deletion (14 classes) | |
| `manabot/experiment/evaluation_queue.py` | slated for deletion (4 classes) | |
| `manabot/experiment/execution.py` | 7 config/record |  |
| `manabot/experiment/job.py` | slated for deletion (3 classes) | |
| `manabot/experiment/numerics.py` | 2 config/record |  |
| `manabot/experiment/study.py` | slated for deletion (2 classes) | |
| `manabot/infra/artifacts.py` | 1 config/record | `ArtifactStore`: **converted** to publish_artifact/download_artifact/fetch_artifact |
| `manabot/infra/config.py` | 7 config/record, 1 exception |  |
| `manabot/infra/host.py` | 2 config/record |  |
| `manabot/infra/identity.py` | 1 config/record |  |
| `manabot/infra/profiler.py` | - | `TimingNode`: part of Profiler; `Profiler`: guarded invariant (named in AGENTS.md) |
| `manabot/infra/provenance.py` | 1 exception |  |
| `manabot/infra/runtime.py` | - | `RunContext`: slated for deletion |
| `manabot/infra/source.py` | 1 config/record |  |
| `manabot/llm/budget.py` | 3 config/record | `BudgetLedger`: guarded invariant (named in AGENTS.md) |
| `manabot/llm/cache.py` | 3 config/record |  |
| `manabot/llm/openrouter.py` | 14 config/record | `OpenRouterSession`: resource (HTTP session, context manager) |
| `manabot/llm/schema.py` | 12 config/record |  |
| `manabot/llm/selection.py` | 1 Protocol, 1 exception |  |
| `manabot/llm/session.py` | 3 config/record | `InferenceSession`: resource (context manager) |
| `manabot/llm/transformers_runtime.py` | - | `TransformersRuntime`: interchangeable part (LLM runtime) |
| `manabot/llm/vllm_runtime.py` | 4 config/record | `VLLMRuntime`: interchangeable part (LLM runtime) |
| `manabot/llm/vllm_server.py` | - | `VLLMServer`: resource (named in AGENTS.md) |
| `manabot/model/actor_critic.py` | 1 config/record, 5 nn.Module |  |
| `manabot/model/admission.py` | - | `AdmittedObservation`: guarded invariant: marks admitted tensors |
| `manabot/model/architecture.py` | 2 config/record |  |
| `manabot/model/buckets.py` | 1 config/record |  |
| `manabot/model/catalog.py` | 2 config/record, 1 exception |  |
| `manabot/model/continuation.py` | 1 nn.Module |  |
| `manabot/model/facts.py` | 1 nn.Module |  |
| `manabot/model/inference.py` | - | `PreparedObservation`: guarded invariant: marks prepared tensors; `PolicyInference`: resource (named in AGENTS.md) |
| `manabot/model/live_tokens.py` | - | `LiveTokens`: value: packed index plan with pure tensor ops |
| `manabot/model/name_candidates.py` | 1 nn.Module |  |
| `manabot/model/policy_distribution.py` | 1 exception |  |
| `manabot/model/recent_events.py` | 1 nn.Module |  |
| `manabot/model/records.py` | 1 nn.Module |  |
| `manabot/model/semantic_cards.py` | 3 config/record, 1 enum, 1 nn.Module | `RecordEmbeddings`: value with pure derivations |
| `manabot/model/shared_effects.py` | 1 nn.Module |  |
| `manabot/pools.py` | 3 config/record |  |
| `manabot/recipes/baseline.py` | 1 config/record |  |
| `manabot/recipes/belief_sampler.py` | off-limits here (concurrent rewrite or already converted; 1 classes) | |
| `manabot/recipes/builders.py` | 1 config/record |  |
| `manabot/recipes/elemental.py` | off-limits here (concurrent rewrite or already converted; 1 classes) | |
| `manabot/recipes/proposed_etude.py` | off-limits here (concurrent rewrite or already converted; 1 classes) | |
| `manabot/reports/belief.py` | 4 config/record |  |
| `manabot/reports/experiment.py` | 2 config/record | `ReportEvidence`: left (see later): cached properties read files; notebook API |
| `manabot/reports/inference.py` | 2 TypedDict, 22 config/record | `EvidenceReader`: guarded invariant: each dependency read once and digest-bound |
| `manabot/reports/learning.py` | 2 config/record |  |
| `manabot/reports/records.py` | 3 config/record |  |
| `manabot/reports/selection.py` | 5 config/record |  |
| `manabot/reports/self_play.py` | 1 config/record |  |
| `manabot/reports/study.py` | 6 config/record |  |
| `manabot/reports/time_cohort.py` | 3 config/record |  |
| `manabot/reports/time_profile.py` | 3 config/record |  |
| `manabot/reports/time_sources.py` | - | `_ReadSnapshot`: interchangeable part: read-only Storage implementation |
| `manabot/reports/training.py` | 1 config/record |  |
| `manabot/semantic/decision_contract.py` | - | `SemanticDecisionContract`: value: legal commands of one decision |
| `manabot/semantic/learning.py` | 4 config/record, 5 exception | `SemanticCatalog`: value; `BoundSemanticPack`: value bound to one catalog with pure projections |
| `manabot/semantic/text_observation.py` | 2 config/record, 3 exception | `_Renderer`: private renderer caching lookups for one capture; left (see later) |
| `manabot/sim/action/structured.py` | 4 config/record, 1 exception | `SeededSemanticScorer`: interchangeable part (scorer); `RaggedPolicyDecoder`: interchangeable part (decoder) |
| `manabot/sim/anchors.py` | 2 config/record |  |
| `manabot/sim/card_evidence.py` | 3 config/record |  |
| `manabot/sim/child.py` | - | `ChildProcess`: resource (named in AGENTS.md) |
| `manabot/sim/execution.py` | 5 config/record |  |
| `manabot/sim/guidance.py` | - | `PolicyPriorRandomLeafEvaluator`: interchangeable part (leaf evaluator) |
| `manabot/sim/heuristic.py` | 4 config/record | `_Body`: record; `_Observation`: private read-only view over one observation; `_Board`: private read-only view over one observation; `HeuristicPlayer`: interchangeable part (player act interface) |
| `manabot/sim/matchup.py` | - | `AuthoredMatchup`: config; `DeckMatchup`: config |
| `manabot/sim/native_game.py` | - | `_InChildPlayers`: private player builder for a child process |
| `manabot/sim/performance.py` | 1 config/record | `Measurement`: accumulator record owned by EvalPerformance; `EvalPerformance`: guarded invariant: thread-shared counters and bounded samples under one lock |
| `manabot/sim/player.py` | 1 enum | `Player`: **converted** to a Protocol (PlayerType and unused to() removed); `ModelPlayer`: interchangeable part (Player); `RandomPlayer`: interchangeable part (Player); `DefaultPlayer`: interchangeable part (Player) |
| `manabot/sim/players.py` | 1 config/record |  |
| `manabot/sim/registration.py` | 1 config/record | `PlayerRegistration`: config |
| `manabot/sim/replay.py` | 1 config/record |  |
| `manabot/sim/rollout.py` | 4 config/record |  |
| `manabot/sim/search/branch.py` | 2 Protocol, 2 config/record | `_SelectedSession`: interchangeable part (search session backend); `_ReferenceSession`: interchangeable part (search session backend) |
| `manabot/sim/search/conditional.py` | 2 Protocol, 9 config/record, 1 exception |  |
| `manabot/sim/search/flat.py` | 1 config/record |  |
| `manabot/sim/search/flat_mc.py` | 1 Protocol, 4 config/record |  |
| `manabot/sim/search/local_update.py` | 7 config/record | `LocalUpdateTeacher`: resource: admitted policy with reused frontier buffers; `LocalUpdatePlayer`: interchangeable part (player) |
| `manabot/sim/search/puct.py` | 1 Protocol, 5 config/record | `AgentLeafEvaluator`: interchangeable part (leaf evaluator); `DeterminizedPuctPlayer`: interchangeable part (player) |
| `manabot/sim/search/rollout.py` | - | `Rollout`: guarded invariant: one rollout advances once |
| `manabot/sim/search/rollout_receipt.py` | 1 config/record |  |
| `manabot/sim/search/runtime.py` | 2 config/record, 2 exception |  |
| `manabot/sim/search/sampling.py` | 1 Protocol, 4 config/record | `LearnedHandSampler`: interchangeable part (hand sampler) |
| `manabot/sim/search/value.py` | 2 config/record | `_ValueLeafScorer`: resource: reused scoring buffers; `VGreedyPlayer`: interchangeable part (player); `ValueSearchPlayer`: interchangeable part (player) |
| `manabot/sim/serving.py` | 1 config/record | `ResidentPolicy`: resource record; `PolicyStore`: resource: resident policies closed with the store |
| `manabot/sim/serving_protocol.py` | 4 config/record, 1 exception | `InferenceClient`: resource (socket client); `InferenceRuntime`: resource with a lifetime (context manager) |
| `manabot/sim/sim.py` | 1 enum | `GameStats`: **converted** to GameRecord + simulation_summary/detailed_analysis |
| `manabot/sim/supervision.py` | 1 config/record |  |
| `manabot/sim/threads.py` | 1 exception | `ThreadBudget`: record; account() **moved** to account_threads; `ProcessCensus`: record; take/record/enforce **moved** to take_census/record_census/enforce_budget |
| `manabot/training/clock.py` | 1 config/record |  |
| `manabot/training/decks.py` | 1 config/record |  |
| `manabot/training/learning/damped_policy.py` | 1 config/record |  |
| `manabot/training/learning/health.py` | 1 TypedDict | `OptimizerHealth`: left (see later) |
| `manabot/training/learning/learner.py` | off-limits here (concurrent rewrite or already converted; 2 classes) | |
| `manabot/training/learning/selection.py` | 3 TypedDict |  |
| `manabot/training/loops/self_play.py` | off-limits here (concurrent rewrite or already converted; 5 classes) | |
| `manabot/training/records.py` | off-limits here (concurrent rewrite or already converted; 7 classes) | |
| `manabot/training/regime.py` | 1 TypedDict, 20 config/record | `TrainingRegime`: config |
| `manabot/training/rollout/actions.py` | 1 config/record | `PackedActions`: value: packed ragged arrays with pure pack/unpack |
| `manabot/training/rollout/adaptive.py` | 5 TypedDict, 6 config/record | `AdaptiveSampler`: left (blocked: built in collection.py) |
| `manabot/training/rollout/collection.py` | off-limits here (concurrent rewrite or already converted; 8 classes) | |
| `manabot/training/rollout/producers.py` | 1 config/record | `ShardedVectorEnv`: resource with a lifetime (worker processes and shared buffers) |
| `manabot/training/rollout/selection_data.py` | 4 config/record |  |
| `manabot/training/rollout/self_play.py` | 1 TypedDict, 1 config/record | `SelfPlayCounter`: left (blocked: owned by collection.py) |
| `manabot/training/state/admission.py` | slated for deletion (1 classes) | |
| `manabot/training/state/game_errors.py` | off-limits here (concurrent rewrite or already converted; 5 classes) | |
| `manabot/training/state/learning_state.py` | 1 config/record |  |
| `manabot/training/state/recovery.py` | off-limits here (concurrent rewrite or already converted; 2 classes) | |
| `manabot/training/update_metrics.py` | 3 Protocol |  |
| `manabot/verify/action_cases.py` | 2 config/record |  |
| `manabot/verify/checkpoint_scenarios.py` | 1 config/record |  |
| `manabot/verify/choice_corpus.py` | 3 config/record |  |
| `manabot/verify/competency.py` | - | `BehaviorProbe`: accumulating probe over one game; left (low value) |
| `manabot/verify/decision_profile.py` | 1 config/record |  |
| `manabot/verify/learning_regression.py` | 7 config/record |  |
| `manabot/verify/llm_game.py` | 3 config/record | `CheckpointOpponent`: interchangeable part (opponent) |
| `manabot/verify/llm_opponent.py` | 2 config/record |  |
| `manabot/verify/record_coverage.py` | 1 config/record |  |
| `manabot/verify/scenario_validation.py` | 1 TypedDict, 3 config/record |  |
| `manabot/verify/scenarios.py` | 1 Protocol, 2 config/record | `CounterBombTracker`: interchangeable part (named in AGENTS.md); `HoldWipeTracker`: interchangeable part (named in AGENTS.md); `BoltThreatTracker`: interchangeable part (named in AGENTS.md); `RaceBlockTracker`: interchangeable part (named in AGENTS.md); `HoldQuenchTracker`: interchangeable part (named in AGENTS.md) |
| `manabot/verify/store.py` | - | `VerifyStore`: resource (named in AGENTS.md) |
| `manabot/verify/util.py` | 1 config/record |  |
