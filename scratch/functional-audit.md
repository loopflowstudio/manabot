# Historical functional audit: classes in manabot/ and academy/

This records the earlier parallel audit, before the execution cutover. Its
“off-limits” labels describe that pass only. Current deletion scope and retained
record models are in plan.md and implementation.md.

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
| `Opponent.verify()` (record reading a checkpoint) | `verify_checkpoint(opponent)` | `academy/opponent.py` | record does not do I/O |

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

## Subsequent deletion status

Deleted: `RunContext`, `experiment/{job,runner,evaluation_queue,monitor_worker,study}.py`
and `training/state/admission.py`. `experiment/config.py` survives to read and
resolve retained comparison records; it no longer launches runs.

## Off-limits here

`manabot/training/rollout/collection.py`, `training/state/{game_errors,recovery,incidents}.py`,
`training/execution.py`, `training/records.py`, `training/loops/self_play.py`,
`env/target_practice.py` (concurrent rewrite); `cli.py`, `recipes/{elemental,belief_sampler,proposed_etude}.py`,
`training/learning/learner.py` (already converted).

## Historical inventory

The full module-by-module inventory is retained in git at f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653:scratch/functional-audit.md. The remaining candidates above are background, not additional scope for the execution cutover.
