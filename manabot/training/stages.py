"""Stage functions: the units of training work a regime calls over its `Run`.

Each stage opens one `StageScope` (`Run.begin_stage`), records its inputs,
artifacts and costs on that scope's `StageRecord` while it runs, and closes it
with `Run.finish_stage`. There is one implementation of each kind of work:

- `self_play(run, model, ...)` trains a `Model` by self-play, or against
  another model frozen; `import_policy`, `collect_selection`,
  `collect_belief` and `train_belief` admit a published policy, play a frozen
  selection population, collect hidden-hand labels with a frozen policy and
  fit a belief sampler. These are what regime functions call.
- `run_saved_regime(run, regime)` is the regime function of a saved
  `TrainingRegime`: one stage-function call per saved stage, in order;
  `execute_regime` runs it.

Each function admits its own arguments (decks, world, curriculum, input
digests), builds the stage's typed settings record (`TrainSelfPlay`, ...),
shrinks it under a smoke run (`smoke_stage`) and does the work. Self-play's
loop is `_train_self_play`, which reads only the decks, the model and
observation configuration, the settings and the schedule clock.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Literal

import numpy as np
import torch

from manabot.belief.checkpoint import load_belief_sampler
from manabot.belief.sampling_data import (
    collect_frozen_policy,
    read_dataset,
    save_dataset,
)
from manabot.belief.sampling_fit import fit_belief_sampler, save_belief_sampler
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
    RuntimeConfig,
)
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.profiler import Profiler
from manabot.infra.runtime import RunContext
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint, save_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.reports.selection import analyze_selection, write_selection_report
from manabot.training.decks import UniformDecks
from manabot.training.learning.learner import Learner
from manabot.training.learning.objectives import update_ema, update_iteration
from manabot.training.records import PhaseTiming, StageRecord
from manabot.training.regime import (
    ArtifactReference,
    BatchClosure,
    CollectBelief,
    CollectSelection,
    DampedPolicyLearning,
    Execution,
    FrozenOpponent,
    ImportPolicy,
    Learning,
    LearningStateImport,
    Operation,
    PretrainedPolicy,
    ScheduleClock,
    SelectionGameSpec,
    Stage,
    TrainBelief,
    TrainingRegime,
    TrainSelfPlay,
    check_continuation,
    check_self_play,
)
from manabot.training.rollout.adaptive import AdaptiveConfig
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.selection_data import (
    SelectionDataset,
    collect_selection_game,
    replay_selection_game,
)
from manabot.training.run import Model, Run, StageScope, artifact
from manabot.training.state.admission import admit_policy
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
)
from manabot.training.state.learning_state import admit_learning_state, validate_adam
from manabot.training.state.recovery import (
    TrainingPaused,
    restore_learning,
    save_update,
)
from manabot.training.update_metrics import (
    OUTCOME_TABLE,
    metric_row,
    phase_seconds,
    time_since,
)
import managym


def cuda_peak_metrics(device: str) -> dict[str, int]:
    """This process's CUDA high-water marks so far; absent off CUDA.

    Reserved bytes include the allocator's cache, so they are what a card must
    hold; allocated bytes are what the tensors themselves needed.
    """
    if device != "cuda":
        return {}
    return {
        "memory/peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "memory/peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
    }


def mps_memory_metrics(device: str) -> dict[str, int]:
    """Current MPS tensor/driver bytes, not peaks or additive memory totals."""
    if device != "mps":
        return {}
    return {
        "memory/allocated_mps_bytes": torch.mps.current_allocated_memory(),
        "memory/driver_allocated_mps_bytes": torch.mps.driver_allocated_memory(),
    }


def _phase_timings(profiler: Profiler, root: str) -> dict[str, PhaseTiming]:
    """Cumulative subphase timings under `root`, keyed by their path below it."""
    return {
        path.removeprefix(f"{root}/"): PhaseTiming(
            seconds=stats["total_time"], calls=int(stats["count"])
        )
        for path, stats in profiler.get_stats().items()
        if path.startswith(f"{root}/")
    }


Decks = MatchConfig | UniformDecks


def _root(decks: Decks) -> MatchConfig:
    """The match a stage's games open: the authored match, or a deck source's
    representative pair (its sampled roots come from the native draw)."""
    return decks.match_config() if isinstance(decks, UniformDecks) else decks


def checkpoint_setups(decks: Decks) -> list[managym.PlayerConfig] | None:
    """Setups a checkpoint binds; a deck-source run binds the world (`None`)."""
    if isinstance(decks, UniformDecks):
        return None
    return Match(decks).to_rust()


def admit_self_play(
    settings: TrainSelfPlay,
    *,
    decks: Decks,
    agent: ModelConfig,
    recovery_max_microsteps: int | None,
    current_world: bool = True,
) -> None:
    """Admit a self-play stage's decks, curriculum and sampling in this world.

    `validate_regime` runs this for each saved stage before a run opens;
    `self_play` runs it for its own settings. `current_world=False` skips
    admitting a deck source's decks natively (offline authoring).
    """
    if settings.matchup_curriculum == "elemental-uniform":
        elemental_roster(Match(_root(decks)))
        if agent.world != "elemental-world-v2":
            raise ValueError(
                "elemental-uniform requires complete Elemental policy input"
            )
    if settings.adaptive is not None:
        if recovery_max_microsteps is not None:
            raise ValueError("adaptive sampling requires current-game recovery")
        if agent.world == "elemental-world-v2":
            for cell in settings.adaptive.cells:
                elemental_roster(Match(cell))
    if isinstance(decks, UniformDecks):
        if current_world:
            decks.check()
        if agent.world != managym.WORLD_SETUP:
            raise ValueError(
                "a deck source requires the complete world policy vocabulary"
            )
    if agent.belief_count_buckets:
        raise ValueError("regime stages do not produce belief inputs")


def _record_inputs(run: Run, scope: StageScope, producer: str) -> None:
    """Record every artifact of this run's stage `producer` as an input.

    Each is checked against its digest first, so a stage never reads bytes
    that changed on disk since they were published.
    """
    source = next(item for item in run.record.stages if item.id == producer)
    for name, item in source.artifacts.items():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"input artifact changed: {producer}/{name}")
        scope.record.inputs[f"{producer}/{name}"] = dict(item)


def _open_stage(
    run: Run,
    declaration: Operation,
    *,
    arguments: dict[str, object],
    producer: str | None = None,
) -> StageScope:
    """Open the declared stage, record its producer's artifacts and commit."""
    scope = run.begin_stage(
        declaration.id,
        declaration.execution,
        self_play=isinstance(declaration, TrainSelfPlay),
        keep_restored=True,
        operation=declaration.operation,
        arguments=arguments,
    )
    assert scope is not None
    if producer is not None:
        _record_inputs(run, scope, producer)
    run.persist()
    scope.check()
    return scope


# ---------------------------------------------------------------------------
# Self-play


@dataclass
class SelfPlaySession:
    """A self-play learner that a following stage may continue live.

    Continuing keeps the network, Adam state, EMA average, schedule iteration,
    collector (with its open games and RNG) and Etude metrics run.
    """

    learner: Learner
    ema: ActorCritic | None
    iteration: int
    collector: SeatRoutedCollector
    experiment: RunContext
    declaration: TrainSelfPlay


def _train_self_play(
    run: Run,
    scope: StageScope,
    stage: TrainSelfPlay,
    *,
    decks: Decks,
    agent_config: ModelConfig,
    observation: ObservationConfig,
    schedule_clock: ScheduleClock,
    session: SelfPlaySession | None,
    initial: ActorCritic | None,
) -> SelfPlaySession:
    """Train one self-play stage; return the live session it ends with.

    `session` is the live state this stage continues. `initial` supplies the
    starting weights of a fresh learner (the model's). A stage restored only
    to rebuild live state for its successor returns the session without
    training. `schedule_clock` measures schedule progress: the update index
    over `stage.updates`, or elapsed run time over the run's wall budget.
    """
    space = ObservationSpace(observation)
    root = Match(_root(decks))
    setups = checkpoint_setups(decks)
    record = scope.record
    check = scope.check
    out = run.out
    seeds = run.seeds
    imported = None
    if stage.learning_state is not None:
        record.inputs.update(
            {
                f"learning_state/{role}": dict(getattr(stage.learning_state, role))
                for role in ("source_run", *stage.learning_state.roles)
            }
        )
        run.persist()
        tick = time.perf_counter()
        imported = admit_learning_state(
            stage,
            out,
            decks=decks,
            agent=agent_config,
            observation=observation,
            schedule_clock=schedule_clock,
        )
        if imported.seed != run.seed:
            raise ValueError("learning-state continuation must retain training seed")
        record.artifacts.update(imported.artifacts)
        record.learning_state_origin = imported.origin
        record.export_seconds += time.perf_counter() - tick
        # A segment has new stochastic streams, deterministically
        # separated from the producer by its absolute boundary.
        for name in ("initialization", "collection", "minibatches"):
            seeds[name] += imported.origin.iteration
        run.record.seed_streams = dict(seeds)
        run.persist()
    pretrained = None
    if stage.pretrained is not None and run.snapshot is None:
        # A recovered attempt restores its own learner snapshot; the
        # producer was admitted, and its copies retained, only once.
        record.inputs.update(
            {
                "pretrained/source_run": dict(stage.pretrained.source_run),
                "pretrained/checkpoint": dict(stage.pretrained.checkpoint),
            }
        )
        run.persist()
        tick = time.perf_counter()
        admitted = admit_policy(
            stage.pretrained,
            out,
            decks=decks,
            agent=agent_config,
            observation=observation,
            name=f"{stage.id}-pretrained",
        )
        record.artifacts["pretrained"] = admitted.checkpoint
        record.artifacts["pretrained_run"] = admitted.source_run
        record.producer_cost = admitted.producer_cost
        # Fork RNG so loading never shifts this run's own streams.
        with torch.random.fork_rng(devices=[]):
            pretrained, _ = load_checkpoint(admitted.checkpoint["path"])
        if any(
            not torch.isfinite(value).all()
            for value in pretrained.state_dict().values()
        ):
            raise ValueError("pretrained policy contains nonfinite tensors")
        record.export_seconds += time.perf_counter() - tick
        run.persist()
    opponent_agent = None
    if stage.opponent is not None:
        # Validate before every stage, even when retaining the live
        # collector. Changed disk bytes cannot silently change a run.
        opponent_path = Path(stage.opponent.path).resolve()
        if file_sha256(opponent_path) != stage.opponent.sha256:
            raise ValueError("frozen opponent checkpoint digest differs")
        record.inputs["frozen_opponent"] = artifact(opponent_path)
        if session is None or run.snapshot is not None:
            opponent_agent, opponent_space = load_checkpoint(
                str(opponent_path),
                bucketed_shapes=stage.execution.bucketed_shapes,
            )
            validate_agent_setup(opponent_agent, root.to_rust())
            if opponent_space.encoder.config != space.encoder.config:
                raise ValueError("frozen opponent observation ABI differs")
            if opponent_agent.belief_count_buckets:
                raise ValueError("frozen collector does not supply belief inputs")
            opponent_agent.requires_grad_(False)
    if session is not None and run.snapshot is None:
        # Validation permits only the latest live collector to continue.
        learner, ema, iteration = session.learner, session.ema, session.iteration
        collector, experiment = session.collector, session.experiment
    else:
        torch.manual_seed(seeds["initialization"])
        agent = ActorCritic(
            space,
            agent_config,
            bucketed_shapes=stage.execution.bucketed_shapes,
        ).to(stage.execution.device)
        if initial is not None:
            agent.load_state_dict(initial.state_dict())
        if stage.initial_weights is not None:
            reference = stage.initial_weights
            copied = out / f"{stage.id}-initialization-source.pt"
            if (
                file_sha256(reference["path"]) != reference["sha256"]
                or Path(reference["path"]).stat().st_size != reference["bytes"]
            ):
                raise ValueError("initialization checkpoint digest/size differs")
            if not copied.exists():
                shutil.copyfile(reference["path"], copied)
            if file_sha256(copied) != reference["sha256"]:
                raise ValueError("retained initialization checkpoint differs")
            payload = torch.load(copied, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("initial weights require a checkpoint mapping")
            bc = payload.get("bc")
            if (
                payload.get("global_step") != 0
                or not isinstance(bc, dict)
                or bc.get("weights") != "initial_raw"
            ):
                raise ValueError(
                    "initial weights require a retained update-zero initialization"
                )
            with torch.random.fork_rng(devices=[]):
                initialization, initial_space = load_checkpoint(copied)
            if (
                initialization.config != agent_config
                or initial_space.encoder.config != space.encoder.config
            ):
                raise ValueError("initialization model or observation differs")
            validate_agent_setup(initialization, root.to_rust())
            agent.load_state_dict(initialization.state_dict())
            record.inputs["initial_weights"] = dict(reference)
            record.artifacts["initialization_source"] = artifact(copied)
        if imported is not None:
            agent.load_state_dict(imported.raw.state_dict())
        if pretrained is not None:
            # Only weights are inherited: Adam, the EMA average and
            # the schedule iteration below all start fresh.
            agent.load_state_dict(pretrained.state_dict())
            pretrained = None
        if run.snapshot is None:
            tick = time.perf_counter()
            target = out / f"{stage.id}-initial-raw.pt"
            temporary = target.with_suffix(".tmp")
            save_checkpoint(
                agent,
                space,
                temporary,
                player_configs=setups,
                extra={
                    "run_id": run.id,
                    "stage_id": stage.id,
                    "weights": "initial_raw",
                    "value_semantic": "signed_outcome",
                    "regime_digest": run.record.regime_digest,
                },
            )
            os.replace(temporary, target)
            # New self-play monitoring must preserve initialization and
            # action-sampling streams compared with monitoring disabled.
            if stage.opponent is None:
                export_device = next(agent.parameters()).device.type
                with torch.random.fork_rng(
                    devices=[0] if export_device in {"cuda", "mps"} else [],
                    device_type="mps" if export_device == "mps" else "cuda",
                ):
                    load_checkpoint(str(target))
            else:
                load_checkpoint(str(target))
            record.artifacts["initial_raw"] = artifact(target)
            record.export_seconds += time.perf_counter() - tick
            run.persist()
            if run.initial_admission is not None:
                # The supervisor releases this exact checkpoint only
                # after its initialization cohort is complete and
                # durably published. Waiting consumes the watchdog,
                # never collection/learning time or RNG draws.
                admission_started = time.perf_counter()
                try:
                    while not run.initial_admission.exists():
                        check()
                        time.sleep(0.2)
                finally:
                    record.admission_wait_seconds = (
                        time.perf_counter() - admission_started
                    )
                if (
                    run.initial_admission.read_text().strip()
                    != record.artifacts["initial_raw"]["sha256"]
                ):
                    raise ValueError("initial evaluation admission differs")
        run.producers.close()
        ledger = run.game_ledger

        def record_game_attempt(attempt: GameAttempt) -> GameErrorState:
            assert ledger is not None and run.failures is not None
            return ledger.record(attempt, run.failures.context())

        collector = SeatRoutedCollector(
            space,
            root,
            Reward(RewardConfig()),
            num_envs=stage.streams,
            seed=seeds["collection"],
            opponent_mode="frozen"
            if stage.opponent is not None
            else "random"
            if stage.behavior == "random"
            else "self",
            opponent_agent=opponent_agent,
            learning_seats=stage.learning_seats,
            batch_closure=stage.batch_closure,
            games_per_stream=stage.games_per_stream,
            recovery_max_microsteps=run.recovery_max_microsteps,
            game_errors=stage.game_errors,
            record_attempt=record_game_attempt if stage.game_errors else None,
            recovery_max_game_microsteps=run.recovery.max_game_microsteps
            if run.recovery
            else None,
            root=stage.root,
            matchup_curriculum=stage.matchup_curriculum,
            deck_source=decks if isinstance(decks, UniformDecks) else None,
            device=stage.execution.device,
            workers=stage.execution.workers,
            adaptive=stage.adaptive,
            precision=stage.execution.precision,
        )
        run.producers.callback(collector.close)
        resolved: dict[str, object] = {
            "run_id": run.id,
            "identities": run.record.identities,
        }
        if run.record.regime is not None:
            resolved["regime"] = run.record.regime.model_dump(mode="json")
            regime_name = run.record.regime.id
        else:
            assert run.record.regime_function is not None
            resolved["regime_function"] = run.record.regime_function.model_dump(
                mode="json"
            )
            resolved["stage"] = stage.model_dump(mode="json")
            regime_name = run.record.regime_function.function.partition(":")[2]
        experiment = RunContext(
            RuntimeConfig(
                seed=seeds["initialization"] if imported is not None else run.seed,
                runs_dir=out,
                name=stage.id,
                device=stage.execution.device,
                log_level="WARNING",
            ),
            experiment=run.experiment or stage.id,
            regime=regime_name,
            run=run.id,
            # A recovery attempt continues its parent's metrics.
            resume=Path(run.record.metrics_run) if run.record.metrics_run else None,
            resumed_at={
                clock: float(run.coordinates()[clock])
                for clock in ("updates", "learner_transitions")
            }
            if run.record.metrics_run
            else None,
            resolved_config=resolved,
        )
        run.record.metrics_run = str(experiment.run.files.root)
        run.resources.callback(
            lambda exp=experiment: exp.close(
                "finished" if run.record.status in ("completed", "paused") else "failed"
            )
        )
        if run.snapshot is None:
            # Retain the same admitted initialization used by frozen
            # controls. SDK copying does not draw policy RNG samples.
            tick = time.perf_counter()
            point = run.coordinates()
            experiment.log_update(
                {},
                updates=int(point["updates"]),
                transitions=int(point["learner_transitions"]),
            )
            experiment.checkpoint(
                Path(record.artifacts["initial_raw"]["path"]),
                transitions=int(point["learner_transitions"]),
            )
            record.export_seconds += time.perf_counter() - tick
        learner = Learner(
            agent,
            experiment.device,
            minibatch_rows=stage.streams,
            precision=stage.execution.precision,
        )
        ema = deepcopy(agent) if stage.learning.ema is not None else None
        iteration = 0
        if imported is not None:
            # A raw-only import declared a fresh Adam.
            if imported.optimizer is not None:
                learner.optimizer.load_state_dict(imported.optimizer)
                validate_adam(learner.optimizer)
            assert ema is not None
            ema.load_state_dict(imported.ema.state_dict())
            iteration = imported.origin.iteration
            imported = None
    scope.collector, scope.ema, scope.iteration = collector, ema, iteration
    if not scope.restoring_completed:
        record.actual_device = str(next(learner.agent.parameters()).device)
    # Phase timers are always on: each ends at an existing device
    # wait, so they add no synchronization (~1 µs per phase).
    collector.profiler = Profiler()
    before = deepcopy(collector.stats)
    rng = np.random.default_rng(seeds["minibatches"] + iteration)
    scope.rng = rng
    first_update = 0
    if record.learning_state_origin is not None:
        first_update = record.learning_state_origin.iteration
    snapshot = run.snapshot
    if snapshot is not None:
        tick = time.perf_counter()
        try:
            collector.restore(snapshot.collector, check)
            restore_learning(snapshot, learner, ema, rng)
        finally:
            run.record.recovery_seconds += time.perf_counter() - tick
        iteration = snapshot.iteration
        scope.iteration = iteration
        first_update = snapshot.record.updates
        # Restored counts are committed scientific work; failed work
        # remains in the parent attempt and its cost stays charged.
        before = snapshot.collector_before
    live = SelfPlaySession(learner, ema, iteration, collector, experiment, stage)
    if scope.restoring_completed:
        run.snapshot = None
        return live

    if run.game_ledger is not None:
        # The ledger includes attempts after the last committed
        # optimizer snapshot. Restoring learning cannot erase them.
        collector.game_error_state = run.game_ledger.state
        if run.game_ledger.state.stop_reason:
            raise GameErrorLimit(run.game_ledger.state.stop_reason)

    def checkpoint(label: str) -> None:
        if not run.recoverable:
            return
        tick = time.perf_counter()
        record.watchdog_seconds = scope.watchdog()
        suffix = ".pt.gz" if run.recovery else ".pt"
        target = out / f"{stage.id}-{label}{suffix}"
        save_update(
            target,
            learner,
            collector,
            ema,
            rng,
            iteration,
            record,
            run.record,
            before,
        )
        run.record.recovery_artifact = artifact(target)
        record.export_seconds += time.perf_counter() - tick

    if snapshot is None:
        checkpoint("start")
        run.persist()
    run.snapshot = None
    scope.learner = learner
    slowest_update = 0.0
    timed = phase_seconds(record)
    for update_index in range(first_update, stage.updates):
        if stage.active_seconds is not None and (
            record.collection_seconds + record.learning_seconds >= stage.active_seconds
        ):
            break
        check()
        update_start = time.perf_counter()
        tick = update_start
        behavior_agent = ema if stage.behavior == "ema-self" else learner.agent
        assert behavior_agent is not None
        batch = None
        scope.batch = None
        try:
            batch = collector.collect(
                behavior_agent,
                stage.transitions,
                deadline_monotonic=scope.deadline,
                check=check,
            )
            scope.batch = batch
        finally:
            scope.synchronize()
            record.collection_seconds += time.perf_counter() - tick
            # Subphases inherit the existing collection boundary.
            # Copy after it closes; reporting overhead is outside it.
            record.collection_timings = _phase_timings(collector.profiler, "Collection")
        check()
        tick = time.perf_counter()
        try:
            # The final device sync stays inside the Learning phase so
            # queued work (EMA, last step) counts toward learning.
            with learner.profiler.track("Learning"):
                try:
                    diagnostic = (
                        update_iteration(
                            learner,
                            batch,
                            stage.learning,
                            update_index / stage.updates
                            if schedule_clock == "iteration_fraction"
                            else (
                                time.perf_counter()
                                - run.start
                                - run.record.monitoring_export_seconds
                            )
                            / run.wall_seconds,
                            rng,
                            iteration=iteration + 1,
                            bootstrap_agent=behavior_agent,
                        )
                        if stage.trainable == "policy_value"
                        else {
                            "rows": batch.actions.size,
                            "optimizer_exposures": 0,
                            "retained": 0,
                            "numerical": {"optimizer_steps": 0},
                        }
                    )
                    adaptive_diagnostics = collector.commit_adaptive(batch)
                    if adaptive_diagnostics is not None:
                        diagnostic["adaptive"] = adaptive_diagnostics
                    if stage.matchup_curriculum != "fixed":
                        diagnostic["matchup_curriculum"] = stage.matchup_curriculum
                        if stage.matchup_curriculum == "elemental-uniform":
                            diagnostic["cumulative_matchup_exposure"] = deepcopy(
                                collector.stats.exposure
                            )
                        diagnostic["cumulative_games_by_stream"] = {
                            str(k): v
                            for k, v in collector.stats.games_by_stream.items()
                        }
                    diagnostic["behavior"] = stage.behavior
                    diagnostic["behavior_iteration"] = iteration
                    if stage.learning.gradient != "damped_policy":
                        diagnostic["schedule_clock"] = schedule_clock
                    iteration += 1
                    scope.iteration = iteration
                    diagnostic["iteration"] = iteration
                    if ema is not None:
                        with learner.profiler.track("EMA update"):
                            update_ema(ema, learner.agent, stage.learning.ema)
                finally:
                    scope.synchronize()
        finally:
            record.learning_seconds += time.perf_counter() - tick
            record.learning_timings = _phase_timings(learner.profiler, "Learning")
        record.updates += 1
        record.optimizer_exposures += diagnostic["optimizer_exposures"]
        if record.optimizer_updates is not None:
            record.optimizer_updates += diagnostic["numerical"]["optimizer_steps"]
        record.games = collector.stats.games - before.games
        record.environment_decisions = collector.stats.micro_steps - before.micro_steps
        record.idle_stream_steps = (
            collector.stats.idle_stream_steps - before.idle_stream_steps
        )
        record.learner_transitions = (
            collector.stats.learner_transitions - before.learner_transitions
        )
        diagnostic["coordinates"] = run.coordinates()
        spent = phase_seconds(record)
        # Etude holds update rows; the record keeps only counters.
        experiment.log_update(
            metric_row(diagnostic)
            | time_since(timed, spent)
            | cuda_peak_metrics(stage.execution.device)
            | mps_memory_metrics(stage.execution.device),
            updates=int(diagnostic["coordinates"]["updates"]),
            transitions=int(diagnostic["coordinates"]["learner_transitions"]),
        )
        if collector.opponent_mode == "self":
            experiment.log_table(
                OUTCOME_TABLE,
                collector.stats.self_play.drain(),
                updates=int(diagnostic["coordinates"]["updates"]),
                transitions=int(diagnostic["coordinates"]["learner_transitions"]),
            )
        timed = spent
        if stage.active_seconds is None or (
            record.collection_seconds + record.learning_seconds < stage.active_seconds
        ):
            run.monitor_checkpoint(
                scope,
                stage,
                learner.agent,
                space,
                setups,
                experiment,
            )
        pausing = run.pause_path is not None and run.pause_path.exists()
        cadence = (
            run.recovery.checkpoint_updates
            if run.recovery
            else run.recovery_every_updates
        )
        checkpoint_due = (
            iteration % cadence == 0
            or pausing
            or (update_index + 1 == stage.updates if run.recovery else iteration == 1)
        )
        if checkpoint_due:
            checkpoint(f"update-{iteration:08d}")
        if not run.recovery or checkpoint_due:
            run.persist(progress=True)
        if pausing:
            raise TrainingPaused("requested pause committed at an update boundary")
        if run.stop_reserve_seconds is not None and update_index + 1 < stage.updates:
            # Stop while another update and the final exports
            # still fit inside every watchdog clock.
            slowest_update = max(slowest_update, time.perf_counter() - update_start)
            if scope.remaining() < slowest_update + run.stop_reserve_seconds:
                run.wall_bounded = True
                break
    if stage.active_seconds is not None and (
        record.collection_seconds + record.learning_seconds < stage.active_seconds
    ):
        raise RuntimeError(
            "update safety ceiling reached before active training endpoint"
        )
    live = SelfPlaySession(learner, ema, iteration, collector, experiment, stage)
    agent = learner.agent
    optimizer_state = learner.optimizer.state_dict()
    tick = time.perf_counter()
    variants = {"raw": agent}
    if ema is not None:
        variants["ema"] = ema
    for name, model in variants.items():
        target = out / f"{stage.id}-{name}.pt"
        temporary = target.with_suffix(".tmp")
        save_checkpoint(
            model,
            space,
            temporary,
            player_configs=setups,
            extra={
                "run_id": run.id,
                "stage_id": stage.id,
                "weights": name,
                "averaging": {
                    "clock": "collect-update-iteration",
                    "iteration": iteration,
                    "rate": stage.learning.ema,
                }
                if name == "ema"
                else None,
                "regime_digest": run.record.regime_digest,
                "value_semantic": "signed_outcome",
            },
        )
        os.replace(temporary, target)
        candidate = artifact(target)
        try:
            load_checkpoint(str(target))
        except Exception:
            record.rejected_artifacts[name] = candidate
            raise
        record.artifacts[name] = candidate
        if name == "raw":
            # The SDK retains admitted policy bytes, not a promise of
            # collector/RNG recovery. Charge its copy as export cost.
            experiment.checkpoint(
                target,
                transitions=int(run.coordinates()["learner_transitions"]),
            )
    optimizer_path = out / f"{stage.id}-optimizer.pt"
    torch.save(optimizer_state, optimizer_path.with_suffix(".tmp"))
    os.replace(optimizer_path.with_suffix(".tmp"), optimizer_path)
    record.artifacts["optimizer"] = artifact(optimizer_path)
    record.export_seconds += time.perf_counter() - tick
    run.finish_stage(scope, completion=lambda: checkpoint("completed"))
    return live


# ---------------------------------------------------------------------------
# Policy admission, selection and belief stages


def smoke_stage[S: Stage](stage: S, execution: Execution | None) -> S:
    """A stage run tiny under `execution`: one update of one transition per stream.

    Belief collection plays its minimum three games and belief training takes
    one step. Stream counts that a curriculum stratifies are kept. Every stage
    function shrinks its settings here (`run.smoke_execution`; `None` runs the
    stage as declared), so a smoke run means one thing. The result is
    validated again.
    """
    if execution is None:
        return stage
    update: dict[str, object] = {"execution": execution}
    if isinstance(stage, TrainSelfPlay):
        update |= {
            "updates": 1,
            "transitions": 1,
            "active_seconds": None,
            "games_per_stream": None,
            "streams": 2 if stage.matchup_curriculum == "fixed" else stage.streams,
        }
    elif isinstance(stage, CollectBelief):
        update["games"] = 3
    elif isinstance(stage, TrainBelief):
        update["steps"] = 1
    return type(stage).model_validate({**dict(stage), **update})


# ---------------------------------------------------------------------------
# Stage functions


@dataclass(frozen=True)
class SelfPlayResult:
    """What one self-play stage produced, as recorded in its stage record."""

    stage: str
    raw: ArtifactReference
    ema: ArtifactReference | None
    updates: int
    games: int
    learner_transitions: int

    @classmethod
    def of(cls, record: StageRecord) -> "SelfPlayResult":
        return cls(
            stage=record.id,
            raw=record.artifacts["raw"],  # type: ignore[arg-type]
            ema=record.artifacts.get("ema"),  # type: ignore[arg-type]
            updates=record.updates,
            games=record.games,
            learner_transitions=record.learner_transitions,
        )


@dataclass(frozen=True)
class BeliefCorpus:
    """A belief collection stage's published corpus of hidden-hand labels."""

    stage: str
    dataset: ArtifactReference


def self_play(
    run: Run,
    model: Model,
    *,
    decks: Decks,
    updates: int = 2,
    learning: Learning | DampedPolicyLearning = Learning(),
    opponent: Model | FrozenOpponent | None = None,
    stage: str | None = None,
    streams: int = 4,
    transitions: int = 256,
    games_per_stream: int | None = None,
    learning_seats: Literal["one", "both"] | None = None,
    batch_closure: BatchClosure | None = None,
    behavior: Literal["current-self", "ema-self", "random"] = "current-self",
    matchup_curriculum: Literal[
        "fixed", "cross-balanced", "mirrors-balanced", "elemental-uniform"
    ] = "fixed",
    adaptive: AdaptiveConfig | None = None,
    trainable: Literal["policy_value", "none"] = "policy_value",
    root: Literal["lethal-target-v1"] | None = None,
    active_seconds: float | None = None,
    game_errors: GameErrorPolicy | None = None,
    initial_weights: ArtifactReference | None = None,
    pretrained: PretrainedPolicy | None = None,
    learning_state: LearningStateImport | None = None,
    schedule_clock: ScheduleClock = "iteration_fraction",
    execution: Execution = Execution(),
) -> SelfPlayResult:
    """Train `model` by self-play, or against `opponent` held frozen.

    Settings mean what the same-named `TrainSelfPlay` fields mean; omitted
    seats and batch closure take its defaults for the behavior. `opponent`
    is another model of the run (exported first) or a published checkpoint.
    Training the same model again in the next self-play stage continues its
    live learner, Adam state, EMA and collector, which therefore keep the
    settings `check_continuation` names. The model's weights become the
    stage's raw weights (and its EMA, when the learning rule keeps one).
    """
    name = run.stage_name(stage or "self-play")
    recorded = run.completed(name)
    if recorded is not None:
        _restore_model(model, recorded)
        return SelfPlayResult.of(recorded)
    if opponent is model:
        raise ValueError("a model cannot be its own frozen opponent")
    if opponent is not None and behavior != "current-self":
        raise ValueError("a frozen opponent sets the behavior")
    if game_errors is not None and run.game_ledger is None:
        raise ValueError("game-error tolerance requires the run's game-attempt ledger")
    frozen = opponent if isinstance(opponent, FrozenOpponent) else None
    if isinstance(opponent, Model):
        reference = run.export(opponent, decks=decks)
        frozen = FrozenOpponent(path=reference["path"], sha256=reference["sha256"])
    # The run's latest learner continues only the model it last trained; a
    # resumed attempt holds no live learner for the stages it replayed.
    latest = run.live_session
    session = (
        latest
        if latest is not None and latest.declaration.id == model.trained_by
        else None
    )
    implied = {
        key: value
        for key, value in (
            ("learning_seats", learning_seats),
            ("batch_closure", batch_closure),
        )
        if value is not None
    }
    declaration = TrainSelfPlay(
        id=name,
        operation="train_self_play",
        initial=model.trained_by,
        updates=updates,
        learning=learning,
        opponent=frozen,
        streams=streams,
        transitions=transitions,
        games_per_stream=games_per_stream,
        **implied,
        behavior="frozen" if frozen is not None else behavior,
        matchup_curriculum=matchup_curriculum,
        adaptive=adaptive,
        trainable=trainable,
        root=root,
        active_seconds=active_seconds,
        game_errors=game_errors,
        initial_weights=initial_weights,
        pretrained=pretrained,
        learning_state=learning_state,
        execution=execution,
    )
    declaration = smoke_stage(declaration, run.smoke_execution)
    if session is not None:
        check_continuation(session.declaration, declaration)
    elif model.trained_by is not None and (
        run.snapshot is None or run.snapshot.record.id != name
    ):
        # The parent attempt checked this continuation, and its snapshot
        # carries the learner, Adam state and EMA. Without either, refuse
        # rather than start a fresh learner from the weights.
        raise ValueError(
            f"stage {name!r} continues model {model.name!r} from stage "
            f"{model.trained_by!r}, but this attempt holds neither its live "
            "learner nor a recovery snapshot of this stage; train a model in "
            "consecutive self-play stages"
        )
    check_self_play(
        declaration,
        agent=model.agent,
        deck_source=isinstance(decks, UniformDecks),
        schedule_clock=schedule_clock,
        recoverable=run.recoverable,
    )
    admit_self_play(
        declaration,
        decks=decks,
        agent=model.agent,
        recovery_max_microsteps=run.recovery_max_microsteps,
    )
    scope = _open_stage(
        run,
        declaration,
        arguments={
            "model": model.name,
            "opponent": opponent.name if isinstance(opponent, Model) else None,
            "decks": decks,
            "settings": declaration,
            "schedule_clock": schedule_clock,
        },
        producer=model.source,
    )
    live = _train_self_play(
        run,
        scope,
        declaration,
        decks=decks,
        agent_config=model.agent,
        observation=model.observation,
        schedule_clock=schedule_clock,
        session=session,
        initial=model.network if session is None else None,
    )
    run.live_session = live
    model.trained_by = name
    model.network = live.learner.agent
    model.ema = live.ema
    _published(model, scope.record)
    return SelfPlayResult.of(scope.record)


def _published(model: Model, record: StageRecord) -> None:
    """The stage's raw/EMA exports are checkpoints of the model's weights."""
    model.exports = {
        weights: record.artifacts[weights]  # type: ignore[misc]
        for weights in ("raw", "ema")
        if weights in record.artifacts
    }
    model.source = record.id


def _restore_model(model: Model, record: StageRecord) -> None:
    """Load a completed stage's exported weights into `model` on resume.

    The live learner is not restored: the model's next self-play stage
    still continues this stage, from its recovery snapshot.
    """
    with torch.random.fork_rng(devices=[]):
        model.network, _ = load_checkpoint(record.artifacts["raw"]["path"])
        if "ema" in record.artifacts:
            model.ema, _ = load_checkpoint(record.artifacts["ema"]["path"])
    model.trained_by = record.id
    _published(model, record)


def _published_model(run: Run, name: str, producer: str) -> Model:
    """A new model of the run holding the weights stage `producer` published."""
    record = next(item for item in run.record.stages if item.id == producer)
    with torch.random.fork_rng(devices=[]):
        network, space = load_checkpoint(record.artifacts["raw"]["path"])
        model = run.adopt(name, network, space.encoder.config)
        if "ema" in record.artifacts:
            model.ema, _ = load_checkpoint(record.artifacts["ema"]["path"])
    _published(model, record)
    return model


def import_policy(
    run: Run,
    name: str,
    *,
    source_run: ArtifactReference,
    source_stage: str,
    checkpoint: ArtifactReference,
    decks: Decks,
    weights: Literal["raw", "ema"] = "raw",
    agent: ModelConfig | None = None,
    observation: ObservationConfig | None = None,
    stage: str | None = None,
    execution: Execution = Execution(),
) -> Model:
    """Admit a published policy as a new model of the run, named `name`.

    The producer's record must match the checkpoint, its world and setups
    must admit `decks`, and its cost carries into this run. The producer must
    use `agent` and `observation` when given, else the checkpoint's own. The
    model starts from the admitted weights.
    """
    stage_id = run.stage_name(stage or f"import-{name}")
    recorded = run.completed(stage_id)
    if recorded is None:
        if agent is None or observation is None:
            with torch.random.fork_rng(devices=[]):
                producer, producer_space = load_checkpoint(checkpoint["path"])
            agent = agent or producer.config
            observation = observation or producer_space.encoder.config
        declaration = ImportPolicy(
            id=stage_id,
            operation="import_policy",
            source_run=source_run,
            source_stage=source_stage,
            checkpoint=checkpoint,
            weights=weights,
            execution=execution,
        )
        declaration = smoke_stage(declaration, run.smoke_execution)
        scope = _open_stage(
            run,
            declaration,
            arguments={"model": name, "decks": decks, "settings": declaration},
        )
        recorded = scope.record
        tick = time.perf_counter()
        recorded.inputs = {
            "source_run": dict(source_run),
            "checkpoint": dict(checkpoint),
        }
        run.persist()
        admitted = admit_policy(
            declaration, run.out, decks=decks, agent=agent, observation=observation
        )
        recorded.artifacts[weights] = admitted.checkpoint
        recorded.artifacts["source_run"] = admitted.source_run
        recorded.producer_cost = admitted.producer_cost
        recorded.export_seconds = time.perf_counter() - tick
        run.finish_stage(scope)
    with torch.random.fork_rng(devices=[]):
        network, space = load_checkpoint(recorded.artifacts[weights]["path"])
    model = run.adopt(name, network, space.encoder.config)
    _published(model, recorded)
    return model


def collect_selection(
    run: Run,
    policy: str,
    *,
    agent: ModelConfig,
    observation: ObservationConfig,
    decks: MatchConfig,
    population: tuple[SelectionGameSpec, ...],
    source_run: str | None = None,
    weights: Literal["raw", "ema"] = "raw",
    max_steps: int = 2000,
    learning: Learning | DampedPolicyLearning = Learning(),
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 93,
    stage: str | None = None,
    execution: Execution = Execution(),
) -> ArtifactReference:
    """Play `population` with the self-play stage `policy`'s weights frozen.

    `policy` names a completed self-play stage of this run, or of the stored
    run `source_run`; its checkpoint must use `agent` and `observation`.
    Returns the selection dataset; the stage also writes its analysis and
    report.
    """
    name = run.stage_name(stage or "collect-selection")
    recorded = run.completed(name)
    if recorded is not None:
        return recorded.artifacts["dataset"]  # type: ignore[return-value]
    declaration = CollectSelection(
        id=name,
        operation="collect_selection",
        policy=policy,
        source_run=source_run,
        weights=weights,
        population=population,
        max_steps=max_steps,
        learning=learning,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        execution=execution,
    )
    declaration = smoke_stage(declaration, run.smoke_execution)
    scope = _open_stage(
        run, declaration, arguments={"decks": decks, "settings": declaration}
    )
    record = scope.record
    check = scope.check
    out = run.out
    policy_run = (
        run.store.training_run(declaration.source_run)
        if declaration.source_run
        else run.record
    )
    source = next(item for item in policy_run.stages if item.id == declaration.policy)
    self_play_policy = (
        isinstance(
            next(
                item
                for item in policy_run.regime.stages
                if item.id == declaration.policy
            ),
            TrainSelfPlay,
        )
        if policy_run.regime is not None
        else source.operation == "train_self_play"
    )
    if source.status != "completed" or not self_play_policy:
        raise ValueError("selection requires a completed self-play checkpoint stage")
    frozen = source.artifacts[declaration.weights]
    record.inputs["policy"] = dict(frozen)
    run.persist()
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("selection policy artifact changed")
    network, policy_space = load_checkpoint(frozen["path"])
    if network.config != agent or policy_space.encoder.config != observation:
        raise ValueError("selection policy configuration differs from regime")
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("selection policy artifact changed while loading")
    population = [game.model_dump(mode="json") for game in declaration.population]
    population_path = out / f"{declaration.id}-population.json"
    atomic_json(population_path, population)
    record.artifacts["population"] = artifact(population_path)
    run.persist()
    games = []
    root = Match(decks)
    for index, spec in enumerate(declaration.population):
        target = out / f"{declaration.id}-{index}.receipts.jsonl"
        tick = time.perf_counter()
        game = collect_selection_game(
            network,
            policy_space,
            root if spec.assignment == 0 else root.swapped(),
            spec,
            target,
            reference=declaration.learning.reference,
            max_steps=declaration.max_steps,
            check=check,
        )
        record.collection_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        if replay_selection_game(target) != len(game.rows):
            raise ValueError("selection replay decision count mismatch")
        record.diagnostic_seconds += time.perf_counter() - tick
        record.artifacts[f"game-{index}"] = artifact(target)
        record.games += 1
        record.environment_decisions += len(game.rows)
        games.append(game)
        run.persist(progress=True)
    dataset = SelectionDataset(
        run_id=run.id,
        stage_id=declaration.id,
        policy_run_id=policy_run.id,
        policy_stage_id=declaration.policy,
        policy_sha256=frozen["sha256"],
        weights=declaration.weights,
        world_binding_sha256=canonical_sha256(network.world_binding),
        population_sha256=canonical_sha256(population),
        games=tuple(games),
    )
    target = out / f"{declaration.id}-dataset.json"
    atomic_json(target, dataset.model_dump(mode="json"))
    record.artifacts["dataset"] = artifact(target)
    tick = time.perf_counter()
    report = analyze_selection(dataset, declaration, check=check)
    target = out / f"{declaration.id}-analysis.json"
    atomic_json(target, report.model_dump(mode="json"))
    record.artifacts["analysis"] = artifact(target)
    target = out / f"{declaration.id}-report.md"
    write_selection_report(report, target)
    record.artifacts["report"] = artifact(target)
    record.diagnostic_seconds += time.perf_counter() - tick
    record.diagnostics = [
        {
            "selection_groups": report.selection_groups,
            "optimizer_exposures": 0,
            "bootstrapped_tail_fraction": 0,
            "analysis": "frozen-complete-game-associations",
        }
    ]
    run.finish_stage(scope)
    return record.artifacts["dataset"]  # type: ignore[return-value]


def collect_belief(
    run: Run,
    model: Model,
    *,
    decks: MatchConfig,
    games: int,
    weights: Literal["raw", "ema"] = "raw",
    max_steps: int = 2000,
    stage: str | None = None,
    execution: Execution = Execution(),
) -> BeliefCorpus:
    """Collect hidden-hand labels with `model` frozen; return the corpus.

    The model's published weights are the policy (exported first if it has
    none); the stage that published them is recorded as this stage's input.
    """
    name = run.stage_name(stage or "collect-belief")
    recorded = run.completed(name)
    if recorded is not None:
        return BeliefCorpus(name, recorded.artifacts["dataset"])  # type: ignore[arg-type]
    policy = run.export(model, weights, decks=decks)
    assert model.source is not None
    declaration = CollectBelief(
        id=name,
        operation="collect_belief",
        policy=model.source,
        weights=weights,
        games=games,
        max_steps=max_steps,
        execution=execution,
    )
    declaration = smoke_stage(declaration, run.smoke_execution)
    scope = _open_stage(
        run,
        declaration,
        arguments={"model": model.name, "decks": decks, "settings": declaration},
        producer=model.source,
    )
    record = scope.record
    dataset = collect_frozen_policy(
        checkpoint=Path(policy["path"]),
        match_config=decks,
        games=declaration.games,
        seed=run.seeds["belief_collection"] + run.belief_games,
        max_steps=declaration.max_steps,
        check=scope.check,
    )
    record.collection_seconds = time.perf_counter() - scope.stage_start
    record.games = len(dataset.games)
    record.environment_decisions = sum(
        len(game.examples) // 2 for game in dataset.games
    )
    run.belief_games += len(dataset.games)
    tick = time.perf_counter()
    target = run.out / f"{name}-dataset.json"
    save_dataset(dataset, target)
    corpus = BeliefCorpus(name, artifact(target))
    record.artifacts["dataset"] = corpus.dataset
    record.export_seconds = time.perf_counter() - tick
    run.finish_stage(scope)
    return corpus


def train_belief(
    run: Run,
    corpus: BeliefCorpus,
    *,
    steps: int = 32,
    batch_size: int = 32,
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    history_dropout: float = 0.0,
    evaluation_samples: int = 32,
    stage: str | None = None,
    execution: Execution = Execution(),
) -> ArtifactReference:
    """Fit a belief sampler to `corpus`; return its admitted checkpoint.

    The corpus is read back from its published, digest-checked dataset.
    """
    name = run.stage_name(stage or "train-belief")
    recorded = run.completed(name)
    if recorded is not None:
        return recorded.artifacts["sampler"]  # type: ignore[return-value]
    declaration = TrainBelief(
        id=name,
        operation="train_belief",
        dataset=corpus.stage,
        steps=steps,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        history_dropout=history_dropout,
        evaluation_samples=evaluation_samples,
        execution=execution,
    )
    declaration = smoke_stage(declaration, run.smoke_execution)
    scope = _open_stage(
        run,
        declaration,
        arguments={"settings": declaration},
        producer=corpus.stage,
    )
    dataset = read_dataset(Path(corpus.dataset["path"]))
    record = scope.record
    tick = time.perf_counter()
    result = fit_belief_sampler(
        dataset,
        steps=declaration.steps,
        batch_size=declaration.batch_size,
        hidden_size=declaration.hidden_size,
        learning_rate=declaration.learning_rate,
        history_dropout=declaration.history_dropout,
        evaluation_samples=declaration.evaluation_samples,
        seed=run.seeds["belief_initialization"],
        check=scope.check,
    )
    record.learning_seconds = time.perf_counter() - tick
    record.optimizer_exposures = result.optimizer_exposures
    record.diagnostics = [asdict(result.metrics)]
    tick = time.perf_counter()
    target = run.out / f"{name}-sampler.pt"
    checkpoint_identity = save_belief_sampler(target, result, dataset)
    candidate = artifact(target)
    try:
        load_belief_sampler(
            target,
            expected_dataset_identity=dataset.identity,
            expected_policy_identity=dataset.policy_identity,
            expected_schema_identity=dataset.schema.identity,
            expected_world_identity=dataset.world_identity,
            expected_checkpoint_identity=checkpoint_identity,
        )
    except Exception:
        record.rejected_artifacts["sampler"] = candidate
        raise
    record.artifacts["sampler"] = candidate
    record.export_seconds = time.perf_counter() - tick
    run.finish_stage(scope)
    return candidate


# ---------------------------------------------------------------------------
# Saved regimes


def run_saved_regime(run: Run, regime: TrainingRegime) -> None:
    """The regime function of a saved `TrainingRegime`.

    Each saved stage is one stage-function call under its own ID, in order.
    A fresh self-play stage trains a new model initialized from the run's
    initialization stream; a stage whose `initial` names an earlier
    self-play stage trains that stage's model again, continuing its live
    learner (validation admits only the latest self-play stage there). The
    run records the saved regime itself, so its digest and meaning are the
    regime's.
    """
    decks = regime.decks
    models: dict[str, Model] = {}
    corpora: dict[str, BeliefCorpus] = {}
    for stage in regime.stages:
        if isinstance(stage, TrainSelfPlay):
            model = (
                models[stage.initial]
                if stage.initial is not None
                else run.model(
                    stage.id,
                    regime.agent,
                    regime.observation,
                    seed=run.seeds["initialization"],
                )
            )
            models[stage.id] = model
            self_play(
                run,
                model,
                decks=decks,
                stage=stage.id,
                updates=stage.updates,
                learning=stage.learning,
                opponent=stage.opponent,
                streams=stage.streams,
                transitions=stage.transitions,
                games_per_stream=stage.games_per_stream,
                learning_seats=stage.learning_seats,
                batch_closure=stage.batch_closure,
                behavior="current-self"
                if stage.behavior == "frozen"
                else stage.behavior,
                matchup_curriculum=stage.matchup_curriculum,
                adaptive=stage.adaptive,
                trainable=stage.trainable,
                root=stage.root,
                active_seconds=stage.active_seconds,
                game_errors=stage.game_errors,
                initial_weights=stage.initial_weights,
                pretrained=stage.pretrained,
                learning_state=stage.learning_state,
                schedule_clock=regime.schedule_clock,
                execution=stage.execution,
            )
        elif isinstance(stage, ImportPolicy):
            models[stage.id] = import_policy(
                run,
                stage.id,
                source_run=stage.source_run,
                source_stage=stage.source_stage,
                checkpoint=stage.checkpoint,
                decks=decks,
                weights=stage.weights,
                agent=regime.agent,
                observation=regime.observation,
                stage=stage.id,
                execution=stage.execution,
            )
        elif isinstance(stage, CollectSelection):
            collect_selection(
                run,
                stage.policy,
                agent=regime.agent,
                observation=regime.observation,
                decks=regime.root_match,
                population=stage.population,
                source_run=stage.source_run,
                weights=stage.weights,
                max_steps=stage.max_steps,
                learning=stage.learning,
                bootstrap_samples=stage.bootstrap_samples,
                bootstrap_seed=stage.bootstrap_seed,
                stage=stage.id,
                execution=stage.execution,
            )
        elif isinstance(stage, CollectBelief):
            policy = models[stage.policy]
            if policy.source != stage.policy:
                # A later stage continued that model; read the policy stage's
                # own published weights.
                policy = _published_model(run, f"{stage.id}-policy", stage.policy)
            corpora[stage.id] = collect_belief(
                run,
                policy,
                decks=regime.root_match,
                games=stage.games,
                weights=stage.weights,
                max_steps=stage.max_steps,
                stage=stage.id,
                execution=stage.execution,
            )
        else:
            train_belief(
                run,
                corpora[stage.dataset],
                steps=stage.steps,
                batch_size=stage.batch_size,
                hidden_size=stage.hidden_size,
                learning_rate=stage.learning_rate,
                history_dropout=stage.history_dropout,
                evaluation_samples=stage.evaluation_samples,
                stage=stage.id,
                execution=stage.execution,
            )
