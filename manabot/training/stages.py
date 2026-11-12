"""Stage functions: the units of training work a regime calls over its `Run`.

Each stage opens one `StageScope` (`Run.begin_stage`), records its inputs,
artifacts and costs on that scope's `StageRecord` while it runs, and closes it
with `Run.finish_stage`. There is one implementation of each kind of work:

- `self_play(run, model, ...)` trains a `Model` by self-play, or against
  another model frozen; `import_policy`, `collect_belief` and `train_belief`
  admit a published policy, collect hidden-hand labels with a frozen policy
  and fit a belief sampler. These are what regime functions call.
- `run_saved_regime(run, regime)` runs a saved `TrainingRegime` by calling the
  same stage bodies in its stage order; `execute_regime` uses it.

A stage body reads its recipe context as a `TrainingRegime` (`regime`): the
saved regime itself, or for a regime function a one-stage regime built from
the call's arguments and the model's configuration, validated by the same
rules. Its declaration (`TrainSelfPlay`, ...) is the stage's typed settings.
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
    SamplerDataset,
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
from manabot.training.outcome_tables import OUTCOME_TABLE_METRIC, write_outcomes
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
    Stage,
    TrainBelief,
    TrainingRegime,
    TrainSelfPlay,
    check_continuation,
)
from manabot.training.rollout.adaptive import AdaptiveConfig
from manabot.training.rollout.collection import SeatRoutedCollector
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
    GameErrorState,
)
from manabot.training.state.learning_state import admit_learning_state, validate_adam
from manabot.training.state.recovery import (
    TrainingPaused,
    restore_learning,
    save_update,
)
from manabot.training.update_metrics import metric_row, phase_seconds, time_since
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


def checkpoint_setups(regime: TrainingRegime) -> list[managym.PlayerConfig] | None:
    """Setups a checkpoint binds; a deck-source run binds the world (`None`)."""
    if regime.deck_source is not None:
        return None
    return Match(regime.root_match).to_rust()


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
    regime: TrainingRegime,
    space: ObservationSpace,
    stage: TrainSelfPlay,
    session: SelfPlaySession | None,
    *,
    initial: ActorCritic | None = None,
) -> SelfPlaySession:
    """Train one self-play stage; return the live session it ends with.

    `session` is the live state this stage continues (`stage.initial` in a
    saved regime). `initial` supplies starting weights for a fresh learner, as
    a regime function's model does. A stage restored only to rebuild live
    state for its successor returns the session without training.
    """
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
        imported = admit_learning_state(stage, regime, out)
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
            stage.pretrained, regime, out, name=f"{stage.id}-pretrained"
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
            opponent_agent, opponent_space = load_checkpoint(str(opponent_path))
            validate_agent_setup(opponent_agent, Match(regime.root_match).to_rust())
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
        agent = ActorCritic(space, regime.agent).to(stage.execution.device)
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
                initialization.config != regime.agent
                or initial_space.encoder.config != space.encoder.config
            ):
                raise ValueError("initialization model or observation differs")
            validate_agent_setup(initialization, Match(regime.match).to_rust())
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
                player_configs=checkpoint_setups(regime),
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
            Match(regime.root_match),
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
            recovery_max_microsteps=regime.recovery_max_microsteps,
            game_errors=stage.game_errors,
            record_attempt=record_game_attempt if stage.game_errors else None,
            recovery_max_game_microsteps=regime.recovery.max_game_microsteps
            if regime.recovery
            else None,
            root=stage.root,
            matchup_curriculum=stage.matchup_curriculum,
            deck_source=regime.deck_source,
            device=stage.execution.device,
            workers=stage.execution.workers,
            adaptive=stage.adaptive,
            precision=stage.execution.precision,
        )
        run.producers.callback(collector.close)
        resolved: dict[str, object] = {
            "regime": regime.model_dump(mode="json"),
            "run_id": run.id,
            "identities": run.record.identities,
        }
        if run.record.regime_function is not None:
            resolved["regime_function"] = run.record.regime_function.model_dump(
                mode="json"
            )
        experiment = RunContext(
            RuntimeConfig(
                seed=seeds["initialization"] if imported is not None else run.seed,
                runs_dir=out,
                name=stage.id,
                device=stage.execution.device,
                log_level="WARNING",
            ),
            experiment=run.experiment or stage.id,
            regime=regime.id,
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
        if not regime.recoverable:
            return
        tick = time.perf_counter()
        record.watchdog_seconds = scope.watchdog()
        suffix = ".pt.gz" if regime.recovery else ".pt"
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
                            if regime.schedule_clock == "iteration_fraction"
                            else (
                                time.perf_counter()
                                - run.start
                                - run.record.monitoring_export_seconds
                            )
                            / regime.wall_seconds,
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
                        diagnostic["schedule_clock"] = regime.schedule_clock
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
        outcome_metrics: dict[str, str] = {}
        if collector.opponent_mode == "self":
            outcome_metrics[OUTCOME_TABLE_METRIC] = write_outcomes(
                experiment.run.files.root, collector.stats.self_play.drain()
            )
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
            | outcome_metrics
            | time_since(timed, spent)
            | cuda_peak_metrics(stage.execution.device)
            | mps_memory_metrics(stage.execution.device),
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
                checkpoint_setups(regime),
                experiment,
            )
        pausing = run.pause_path is not None and run.pause_path.exists()
        cadence = (
            regime.recovery.checkpoint_updates
            if regime.recovery
            else regime.recovery_every_updates
        )
        checkpoint_due = (
            iteration % cadence == 0
            or pausing
            or (
                update_index + 1 == stage.updates if regime.recovery else iteration == 1
            )
        )
        if checkpoint_due:
            checkpoint(f"update-{iteration:08d}")
        if not regime.recovery or checkpoint_due:
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
            player_configs=checkpoint_setups(regime),
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


def _import_policy(
    run: Run, scope: StageScope, regime: TrainingRegime, stage: ImportPolicy
) -> None:
    record = scope.record
    tick = time.perf_counter()
    record.inputs = {
        "source_run": dict(stage.source_run),
        "checkpoint": dict(stage.checkpoint),
    }
    run.persist()
    admitted = admit_policy(stage, regime, run.out)
    record.artifacts[stage.weights] = admitted.checkpoint
    record.artifacts["source_run"] = admitted.source_run
    record.producer_cost = admitted.producer_cost
    record.export_seconds = time.perf_counter() - tick
    run.finish_stage(scope)


def _collect_selection(
    run: Run, scope: StageScope, regime: TrainingRegime, stage: CollectSelection
) -> None:
    record = scope.record
    check = scope.check
    out = run.out
    policy_run = (
        run.store.training_run(stage.source_run) if stage.source_run else run.record
    )
    source = next(item for item in policy_run.stages if item.id == stage.policy)
    assert policy_run.regime is not None
    policy_stage = next(
        item for item in policy_run.regime.stages if item.id == stage.policy
    )
    if source.status != "completed" or not isinstance(policy_stage, TrainSelfPlay):
        raise ValueError("selection requires a completed self-play checkpoint stage")
    frozen = source.artifacts[stage.weights]
    record.inputs["policy"] = dict(frozen)
    run.persist()
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("selection policy artifact changed")
    agent, policy_space = load_checkpoint(frozen["path"])
    if (
        agent.config != regime.agent
        or policy_space.encoder.config != regime.observation
    ):
        raise ValueError("selection policy configuration differs from regime")
    if file_sha256(frozen["path"]) != frozen["sha256"]:
        raise ValueError("selection policy artifact changed while loading")
    population = [game.model_dump(mode="json") for game in stage.population]
    population_path = out / f"{stage.id}-population.json"
    atomic_json(population_path, population)
    record.artifacts["population"] = artifact(population_path)
    run.persist()
    games = []
    match = Match(regime.root_match)
    for index, spec in enumerate(stage.population):
        target = out / f"{stage.id}-{index}.receipts.jsonl"
        tick = time.perf_counter()
        game = collect_selection_game(
            agent,
            policy_space,
            match if spec.assignment == 0 else match.swapped(),
            spec,
            target,
            reference=stage.learning.reference,
            max_steps=stage.max_steps,
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
        stage_id=stage.id,
        policy_run_id=policy_run.id,
        policy_stage_id=stage.policy,
        policy_sha256=frozen["sha256"],
        weights=stage.weights,
        world_binding_sha256=canonical_sha256(agent.world_binding),
        population_sha256=canonical_sha256(population),
        games=tuple(games),
    )
    target = out / f"{stage.id}-dataset.json"
    atomic_json(target, dataset.model_dump(mode="json"))
    record.artifacts["dataset"] = artifact(target)
    tick = time.perf_counter()
    report = analyze_selection(dataset, stage, check=check)
    target = out / f"{stage.id}-analysis.json"
    atomic_json(target, report.model_dump(mode="json"))
    record.artifacts["analysis"] = artifact(target)
    target = out / f"{stage.id}-report.md"
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


def _collect_belief(
    run: Run,
    scope: StageScope,
    match: MatchConfig,
    stage: CollectBelief,
    policy: ArtifactReference,
) -> SamplerDataset:
    record = scope.record
    dataset = collect_frozen_policy(
        checkpoint=Path(policy["path"]),
        match_config=match,
        games=stage.games,
        seed=run.seeds["belief_collection"] + run.belief_games,
        max_steps=stage.max_steps,
        check=scope.check,
    )
    record.collection_seconds = time.perf_counter() - scope.stage_start
    record.games = len(dataset.games)
    record.environment_decisions = sum(
        len(game.examples) // 2 for game in dataset.games
    )
    run.belief_games += len(dataset.games)
    tick = time.perf_counter()
    target = run.out / f"{stage.id}-dataset.json"
    save_dataset(dataset, target)
    record.artifacts["dataset"] = artifact(target)
    record.export_seconds = time.perf_counter() - tick
    run.finish_stage(scope)
    return dataset


def _train_belief(
    run: Run, scope: StageScope, stage: TrainBelief, dataset: SamplerDataset
) -> ArtifactReference:
    record = scope.record
    tick = time.perf_counter()
    result = fit_belief_sampler(
        dataset,
        steps=stage.steps,
        batch_size=stage.batch_size,
        hidden_size=stage.hidden_size,
        learning_rate=stage.learning_rate,
        history_dropout=stage.history_dropout,
        evaluation_samples=stage.evaluation_samples,
        seed=run.seeds["belief_initialization"],
        check=scope.check,
    )
    record.learning_seconds = time.perf_counter() - tick
    record.optimizer_exposures = result.optimizer_exposures
    record.diagnostics = [asdict(result.metrics)]
    tick = time.perf_counter()
    target = run.out / f"{stage.id}-sampler.pt"
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
    """Run a saved `TrainingRegime`'s stages in order through the stage bodies.

    Stages an earlier attempt completed are skipped; the first open stage
    continues the recovery snapshot. A smoke run runs the regime's tiny form
    (`smoke_form`).
    """
    if run.smoke_execution is not None:
        regime = smoke_form(regime, run.smoke_execution)
    space = ObservationSpace(regime.observation)
    session: SelfPlaySession | None = None
    for stage_index, stage in enumerate(regime.stages):
        if any(item.id == stage.id for item in run.record.stages):
            continue
        following = regime.stages[stage_index + 1 : stage_index + 2]
        scope = run.begin_stage(
            stage.id,
            stage.execution,
            self_play=isinstance(stage, TrainSelfPlay),
            keep_restored=bool(following)
            and getattr(following[0], "initial", None) is not None,
            operation=stage.operation,
        )
        if scope is None:
            continue
        references = []
        if isinstance(stage, CollectBelief):
            references.append(stage.policy)
        if isinstance(stage, TrainBelief):
            references.append(stage.dataset)
        if getattr(stage, "initial", None):
            references.append(stage.initial)
        for reference in references:
            source = next(item for item in run.record.stages if item.id == reference)
            for name, item in source.artifacts.items():
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError(f"input artifact changed: {reference}/{name}")
                scope.record.inputs[f"{reference}/{name}"] = dict(item)
        run.persist()
        scope.check()
        if isinstance(stage, ImportPolicy):
            _import_policy(run, scope, regime, stage)
        elif isinstance(stage, CollectSelection):
            _collect_selection(run, scope, regime, stage)
        elif isinstance(stage, CollectBelief):
            source = next(item for item in run.record.stages if item.id == stage.policy)
            _collect_belief(
                run, scope, regime.root_match, stage, source.artifacts[stage.weights]
            )
        elif isinstance(stage, TrainBelief):
            source = next(
                item for item in run.record.stages if item.id == stage.dataset
            )
            _train_belief(
                run,
                scope,
                stage,
                read_dataset(Path(source.artifacts["dataset"]["path"])),
            )
        elif isinstance(stage, TrainSelfPlay):
            session = _train_self_play(
                run,
                scope,
                regime,
                space,
                stage,
                session if stage.initial else None,
            )


def smoke_stage[S: Stage](stage: S, execution: Execution) -> S:
    """A stage run tiny under `execution`: one update of one transition per stream.

    Belief collection plays its minimum three games and belief training takes
    one step. Stream counts that a curriculum stratifies are kept. Saved and
    function regimes shrink their stages here, so a smoke run means one thing.
    """
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
    return stage.model_copy(update=update)


def smoke_form(regime: TrainingRegime, execution: Execution) -> TrainingRegime:
    """The saved regime with every stage in its `smoke_stage` form."""
    stages = [smoke_stage(stage, execution) for stage in regime.stages]
    return TrainingRegime.model_validate(
        regime.model_copy(update={"stages": stages}).model_dump()
    )


# ---------------------------------------------------------------------------
# Stage functions for regime functions


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


def _stage_regime(
    run: Run,
    name: str,
    decks: MatchConfig | UniformDecks,
    agent: ModelConfig,
    observation: ObservationConfig,
    declaration: TrainSelfPlay | ImportPolicy,
) -> TrainingRegime:
    """The one-stage recipe context a stage body reads for a regime function."""
    regime = TrainingRegime(
        id=name,
        world=run.world,
        match=decks if isinstance(decks, MatchConfig) else None,
        deck_source=decks if isinstance(decks, UniformDecks) else None,
        observation=observation,
        agent=agent,
        stages=[declaration],
        wall_seconds=run.wall_seconds,
        schedule_clock="iteration_fraction",
        recovery=run.recovery,
    )
    # Execution admission (world, deck source, curriculum rosters) is the
    # saved regimes'; a function's stages pass the same checks.
    from manabot.training.execution import validate_regime

    return validate_regime(regime)


def self_play(
    run: Run,
    model: Model,
    *,
    decks: MatchConfig | UniformDecks,
    updates: int = 2,
    learning: Learning | DampedPolicyLearning = Learning(),
    opponent: Model | None = None,
    stage: str | None = None,
    streams: int = 4,
    transitions: int = 256,
    games_per_stream: int | None = None,
    learning_seats: Literal["one", "both"] | None = None,
    batch_closure: BatchClosure | None = None,
    behavior: Literal["current-self", "ema-self", "random"] = "current-self",
    matchup_curriculum: Literal["fixed", "elemental-uniform"] = "fixed",
    adaptive: AdaptiveConfig | None = None,
    trainable: Literal["policy_value", "none"] = "policy_value",
    execution: Execution = Execution(),
) -> SelfPlayResult:
    """Train `model` by self-play, or against `opponent` held frozen.

    Settings mean what the same-named `TrainSelfPlay` fields mean. Training
    the same model again in the next self-play stage continues its live
    learner, Adam state, EMA and collector, which therefore keep the settings
    `check_continuation` names. The model's weights become the stage's raw
    weights (and its EMA, when the learning rule keeps one).
    """
    name = run.stage_name(stage or "self-play")
    recorded = run.completed(name)
    if recorded is not None:
        _restore_model(model, recorded)
        return SelfPlayResult.of(recorded)
    if opponent is model:
        raise ValueError("a model cannot be its own frozen opponent")
    frozen = None
    if opponent is not None:
        reference = run.export(opponent, decks=decks)
        frozen = FrozenOpponent(path=reference["path"], sha256=reference["sha256"])
    # Omitted seats and closure take TrainSelfPlay's defaults for the behavior.
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
        execution=execution,
    )
    if run.smoke_execution is not None:
        declaration = smoke_stage(declaration, run.smoke_execution)
    regime = _stage_regime(
        run, name, decks, model.agent, model.observation, declaration
    )
    session: SelfPlaySession | None = model.session
    if session is not None:
        if run.live_session is not session:
            raise ValueError(
                f"model {model.name!r} can continue only while its collector is "
                "the run's latest; train it in consecutive self-play stages"
            )
        check_continuation(session.declaration, declaration)
    space = ObservationSpace(regime.observation)
    scope = run.begin_stage(
        name,
        declaration.execution,
        self_play=True,
        keep_restored=True,
        operation="train_self_play",
        arguments={
            "model": model.name,
            "opponent": opponent.name if opponent is not None else None,
            "decks": decks,
            "settings": declaration,
        },
    )
    assert scope is not None
    live = _train_self_play(
        run,
        scope,
        regime,
        space,
        declaration,
        session,
        initial=model.network if session is None else None,
    )
    run.live_session = live
    model.session = live
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


def _restore_model(model: Model, record: StageRecord) -> None:
    """Load a completed stage's exported weights into `model` on resume.

    The live learner is not restored, so the model's next self-play stage
    starts a fresh learner from these weights.
    """
    with torch.random.fork_rng(devices=[]):
        model.network, _ = load_checkpoint(record.artifacts["raw"]["path"])
        if "ema" in record.artifacts:
            model.ema, _ = load_checkpoint(record.artifacts["ema"]["path"])
    model.session = None
    _published(model, record)


def import_policy(
    run: Run,
    name: str,
    *,
    source_run: ArtifactReference,
    source_stage: str,
    checkpoint: ArtifactReference,
    decks: MatchConfig | UniformDecks,
    weights: Literal["raw", "ema"] = "raw",
) -> Model:
    """Admit a published policy as a new model of the run, named `name`.

    Admission is the saved `import_policy` stage's: the producer's record must
    match the checkpoint, its world and setups must admit `decks`, and its
    cost carries into this run. The model starts from the admitted weights
    and takes the producer's model and observation configuration.
    """
    stage_id = run.stage_name(f"import-{name}")
    recorded = run.completed(stage_id)
    if recorded is None:
        with torch.random.fork_rng(devices=[]):
            producer, producer_space = load_checkpoint(checkpoint["path"])
        declaration = ImportPolicy(
            id=stage_id,
            operation="import_policy",
            source_run=source_run,
            source_stage=source_stage,
            checkpoint=checkpoint,
            weights=weights,
        )
        regime = _stage_regime(
            run,
            stage_id,
            decks,
            producer.config,
            producer_space.encoder.config,
            declaration,
        )
        scope = run.begin_stage(
            stage_id,
            declaration.execution,
            operation="import_policy",
            arguments={"model": name, "decks": decks, "settings": declaration},
        )
        assert scope is not None
        _import_policy(run, scope, regime, declaration)
        recorded = scope.record
    with torch.random.fork_rng(devices=[]):
        network, space = load_checkpoint(recorded.artifacts[weights]["path"])
    return run.adopt(name, network, space.encoder.config)


def collect_belief(
    run: Run,
    model: Model,
    *,
    decks: MatchConfig,
    games: int,
    weights: Literal["raw", "ema"] = "raw",
    max_steps: int = 2000,
    stage: str | None = None,
) -> SamplerDataset:
    """Collect hidden-hand labels with `model` frozen; return the corpus."""
    name = run.stage_name(stage or "collect-belief")
    recorded = run.completed(name)
    if recorded is not None:
        return read_dataset(Path(recorded.artifacts["dataset"]["path"]))
    declaration = CollectBelief(
        id=name,
        operation="collect_belief",
        policy="model",
        weights=weights,
        games=games,
        max_steps=max_steps,
    )
    if run.smoke_execution is not None:
        declaration = smoke_stage(declaration, run.smoke_execution)
    policy = run.export(model, weights, decks=decks)
    scope = run.begin_stage(
        name,
        declaration.execution,
        operation="collect_belief",
        arguments={"model": model.name, "decks": decks, "settings": declaration},
    )
    assert scope is not None
    scope.record.inputs["policy"] = dict(policy)
    run.persist()
    return _collect_belief(run, scope, decks, declaration, policy)


def train_belief(
    run: Run,
    corpus: SamplerDataset,
    *,
    steps: int = 32,
    batch_size: int = 32,
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    history_dropout: float = 0.0,
    evaluation_samples: int = 32,
    stage: str | None = None,
) -> ArtifactReference:
    """Fit a belief sampler to `corpus`; return its admitted checkpoint."""
    name = run.stage_name(stage or "train-belief")
    recorded = run.completed(name)
    if recorded is not None:
        return recorded.artifacts["sampler"]  # type: ignore[return-value]
    declaration = TrainBelief(
        id=name,
        operation="train_belief",
        dataset="corpus",
        steps=steps,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        history_dropout=history_dropout,
        evaluation_samples=evaluation_samples,
    )
    if run.smoke_execution is not None:
        declaration = smoke_stage(declaration, run.smoke_execution)
    scope = run.begin_stage(
        name,
        declaration.execution,
        operation="train_belief",
        arguments={"corpus": corpus.identity, "settings": declaration},
    )
    assert scope is not None
    scope.record.inputs["dataset"] = {"identity": corpus.identity}
    return _train_belief(run, scope, declaration, corpus)
