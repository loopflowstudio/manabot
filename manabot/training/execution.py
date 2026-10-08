"""Bounded regime execution; VerifyStore owns state, manifests are exports."""

from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import TYPE_CHECKING
import uuid

import numpy as np
import psutil
import torch

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.belief.sampling_data import (
    collect_frozen_policy,
    read_dataset,
    save_dataset,
)
from manabot.belief.sampling_fit import (
    fit_belief_sampler,
    load_belief_sampler,
    save_belief_sampler,
)
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra import Experiment
from manabot.infra.hypers import ExperimentHypers, RewardHypers, TrainHypers
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_identity
from manabot.model.policy_distribution import NumericalError
from manabot.model.world import validate_agent_setup
from manabot.remote.plan import Allocation
from manabot.sim.distill import generate_selfplay_shard, load_shards, save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import NetOpponentTrainer, SeatRoutedCollector
from manabot.sim.search_supervised import (
    SearchSupervisedEpochStats,
    train_search_supervised,
)
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
import managym

from .admission import admit_policy
from .clock import watchdog_seconds
from .compound import CompoundStatistics, collect_game, optimize_games, replay_game
from .incidents import save_incident
from .learning_state import admit_learning_state, validate_adam
from .models import (
    ArtifactReference,
    CheckpointCadence,
    CollectBelief,
    CollectLocalUpdate,
    CollectSearch,
    CollectSelection,
    FixedValidationCohort,
    ImportPolicy,
    MonitoringCheckpoint,
    NumericalFailure,
    StageRecord,
    TrainBelief,
    TrainCompound,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
    TrainSupervised,
)
from .objectives import update_ema, update_iteration
from .recovery import (
    attempt_lock,
    load_update,
    restore_learning,
    save_update,
    settle_orphan,
)
from .selection_analysis import analyze_selection, write_selection_report
from .selection_data import SelectionDataset, collect_selection_game

if TYPE_CHECKING:
    from manabot.verify.store import VerifyStore


# Longest a run.json export may lag the store while a stage is making progress.
PROGRESS_EXPORT_SECONDS = 30.0
# Largest share of a stage's wall time that progress exports may consume.
PROGRESS_EXPORT_SHARE = 0.02


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def export_training_run(
    run_id: str, store: "VerifyStore", out: str | Path
) -> TrainingRun:
    run = store.training_run(run_id)
    atomic_json(Path(out) / "run.json", run.model_dump(mode="json"))
    return run


def validate_regime(regime: TrainingRegime | Mapping[str, object]) -> TrainingRegime:
    regime = TrainingRegime.model_validate(
        regime.model_dump() if isinstance(regime, TrainingRegime) else regime
    )
    if regime.world != managym.WORLD_VERSION:
        raise ValueError("regime world differs from native runtime")
    if regime.agent.belief_count_buckets:
        raise ValueError("regime stages do not produce belief inputs")
    return regime


def _runtime_identities(
    seed: int, regime: TrainingRegime, space: ObservationSpace
) -> dict[str, object]:
    identities = runtime_fingerprints(
        seed, match_hypers=regime.match, observation_space=space
    )
    identities.update(
        architecture=architecture_identity(regime.agent, space),
        hardware={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
        },
        training_source_sha256=source_bundle_sha256(
            sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        ),
        torch=torch.__version__,
        policy_numerics="masked-log-softmax-v1",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    if any(stage.execution.device == "cuda" for stage in regime.stages):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but unavailable; CPU fallback is forbidden"
            )
        gpu = torch.cuda.get_device_properties(0)
        identities["cuda"] = {
            "device": gpu.name,
            "memory_bytes": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
        }
    return identities


def execute_regime(
    regime: TrainingRegime | Mapping[str, object],
    seed: int,
    out: str | Path,
    store: "VerifyStore",
    *,
    resume_from: str | None = None,
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    initial_admission: Path | None = None,
    allocation: Allocation | None = None,
) -> TrainingRun:
    """Execute under a local lease, including admission and crash settlement."""
    regime = validate_regime(regime)
    if initial_admission is not None and (
        (checkpoint_seconds is None and checkpoint_updates is None)
        or resume_from is not None
        or len(regime.stages) != 1
        or not isinstance(regime.stages[0], TrainSelfPlay)
        or initial_admission.exists()
    ):
        raise ValueError("initial admission requires a fresh single self-play stage")
    if checkpoint_seconds is not None or checkpoint_updates is not None:
        CheckpointCadence(
            checkpoint_updates=checkpoint_updates, checkpoint_seconds=checkpoint_seconds
        ).admit_regime(regime)
    if (
        checkpoint_updates is not None
        and initial_admission is not None
        and any(
            isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
            for stage in regime.stages
        )
    ):
        raise ValueError("step continuation cannot repeat initialization admission")
    if allocation is not None:
        if (
            len(regime.stages) != 1
            or not isinstance(regime.stages[0], TrainSelfPlay)
            or regime.stages[0].active_seconds is not None
            or regime.schedule_clock != "iteration_fraction"
        ):
            raise ValueError(
                "allocation execution requires a step-target self-play regime"
            )
        if time.time() >= allocation.pause_at:
            raise ValueError("allocation reserves exhausted before training admission")
    destination = Path(out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lease = destination.with_name(destination.name + ".writer.lock")
    with ExitStack() as leases:
        leases.enter_context(attempt_lock(lease))
        if resume_from is not None:
            parent = store.training_run(resume_from)
            if parent.recovery_lock_path is None:
                raise ValueError("attempt has no recovery writer lease")
            leases.enter_context(
                attempt_lock(Path(parent.recovery_lock_path), existing=True)
            )
            if parent.seed != seed or parent.regime_digest != canonical_sha256(
                regime.model_dump(mode="json")
            ):
                raise ValueError("recovery recipe or seed mismatch")
            if (checkpoint_updates, checkpoint_seconds) != (
                parent.monitoring_checkpoint_updates,
                parent.monitoring_checkpoint_seconds,
            ):
                raise ValueError("recovery checkpoint cadence differs from parent")
            load_update(parent)
            if parent.identities != _runtime_identities(
                seed, regime, ObservationSpace(regime.observation)
            ):
                raise ValueError("recovery runtime/source identity mismatch")
            settled = settle_orphan(parent)
            if settled != parent:
                # Preserve the old run.json and snapshots. The canonical record
                # explains settlement, including the explicitly estimated cost.
                store.save_training_run(settled)
        return _execute_regime(
            regime,
            seed,
            destination,
            store,
            resume_from=resume_from,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_updates=checkpoint_updates,
            initial_admission=initial_admission,
            allocation=allocation,
        )


def _execute_regime(
    regime: TrainingRegime,
    seed: int,
    out: str | Path,
    store: "VerifyStore",
    *,
    resume_from: str | None = None,
    checkpoint_seconds: float | None = None,
    checkpoint_updates: int | None = None,
    initial_admission: Path | None = None,
    allocation: Allocation | None = None,
) -> TrainingRun:
    regime = validate_regime(regime)
    parent = store.training_run(resume_from) if resume_from is not None else None
    if parent is not None:
        if parent.status not in {"failed", "interrupted", "paused"}:
            raise ValueError(
                "recovery requires a stopped failed, interrupted or paused attempt"
            )
        if (
            regime.recovery_max_microsteps is None
            or parent.seed != seed
            or parent.regime_digest != canonical_sha256(regime.model_dump(mode="json"))
        ):
            raise ValueError("recovery recipe or seed mismatch")
    snapshot = load_update(parent) if parent is not None else None
    # A retry can fail during setup before reaching its stage. Walk the retained
    # lineage so that failure cannot erase a stage allowance or completion receipt.
    prior_records: dict[str, StageRecord] = {}
    ancestor = parent
    while ancestor is not None:
        for prior_record in ancestor.stages:
            prior_records.setdefault(prior_record.id, prior_record)
        ancestor = (
            store.training_run(ancestor.parent_run_id)
            if ancestor.parent_run_id
            else None
        )
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    watchdog_start = watchdog_seconds()
    seeds = {
        name: seed + offset
        for name, offset in (
            ("initialization", 0),
            ("collection", 10000),
            ("minibatches", 20000),
            ("evaluation", 30000),
            ("belief_collection", 40000),
            ("belief_initialization", 50000),
        )
    }
    run = TrainingRun(
        id=uuid.uuid4().hex,
        regime=regime,
        regime_digest=canonical_sha256(regime.model_dump(mode="json")),
        seed=seed,
        seed_streams=seeds,
        identities=deepcopy(parent.identities) if parent is not None else {},
        status="running",
        monitoring_checkpoint_seconds=checkpoint_seconds,
        monitoring_checkpoint_updates=checkpoint_updates,
        allocation_deadline=allocation.deadline if allocation is not None else None,
        recovery_lock_path=str(out.with_name(out.name + ".writer.lock")),
        recovery_host=socket.gethostname(),
        last_recorded_wall_seconds=time.time(),
        parent_run_id=parent.id if parent is not None else None,
        prior_seconds=parent.prior_seconds + parent.seconds
        if parent is not None
        else 0,
        prior_watchdog_seconds=(parent.prior_watchdog_seconds + parent.watchdog_seconds)
        if parent is not None
        else 0,
        recovery_artifact=parent.recovery_artifact if parent is not None else None,
        stages=deepcopy(snapshot.completed_stages) if snapshot is not None else [],
    )
    if parent is not None and checkpoint_updates is not None:
        assert snapshot is not None
        restored_updates = sum(
            len(s.diagnostics) for s in (*snapshot.completed_stages, snapshot.record)
        )
        # Retain admitted exports from the committed prefix. Their artifact paths,
        # original costs and coordinates remain unchanged across process attempts.
        run.monitoring_checkpoints = [
            c.model_copy(deep=True)
            for c in parent.monitoring_checkpoints
            if c.updates <= restored_updates
        ]
        for receipt in run.monitoring_checkpoints:
            if receipt.artifact is not None and (
                file_sha256(receipt.artifact["path"]) != receipt.artifact["sha256"]
                or Path(receipt.artifact["path"]).stat().st_size
                != receipt.artifact["bytes"]
            ):
                raise ValueError("recovery monitoring artifact differs")
    if parent is None:
        store.save_training_run(run)
    else:
        store.claim_training_recovery(parent.id, run)
    outputs = {}
    self_play_session = None
    resources = ExitStack()
    game_index = 0
    fixed_validation_dataset: dict[str, np.ndarray] | None = None
    last_monitor_seconds = 0.0

    last_export = float("-inf")
    export_wait = PROGRESS_EXPORT_SECONDS

    def persist(*, progress: bool = False) -> None:
        """Commit the run to the store and refresh its run.json export.

        The store is the authority and is written every time. Rewriting the
        export costs time proportional to the whole record, so saves made only
        to record progress inside a stage refresh it at most once per
        PROGRESS_EXPORT_SECONDS, and less often once one export takes more
        than PROGRESS_EXPORT_SHARE of that interval. Stage boundaries,
        completion and failure always export.
        """
        nonlocal last_export, export_wait
        run.seconds = time.perf_counter() - start
        run.watchdog_seconds = watchdog_seconds() - watchdog_start
        run.last_recorded_wall_seconds = time.time()
        store.save_training_run(run)
        began = time.monotonic()
        if progress and began - last_export < export_wait:
            return
        export_training_run(run.id, store, out)
        last_export = time.monotonic()
        export_wait = max(
            PROGRESS_EXPORT_SECONDS, (last_export - began) / PROGRESS_EXPORT_SHARE
        )

    def artifact(path: Path) -> ArtifactReference:
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }

    def coordinates() -> dict[str, int | float]:
        origins = [
            s.learning_state_origin
            for s in run.stages
            if s.learning_state_origin is not None
        ]
        return {
            "updates": run.updates_through(),
            "training_seconds": run.prior_seconds
            + time.perf_counter()
            - start
            + sum(origin.cumulative_seconds for origin in origins),
            "active_training_seconds": sum(
                s.collection_seconds + s.learning_seconds for s in run.stages
            )
            + sum(origin.active_training_seconds for origin in origins),
            **{
                name: sum(getattr(s, name) for s in run.stages)
                + sum(getattr(origin, name) for origin in origins)
                for name in (
                    "environment_decisions",
                    "learner_transitions",
                    "optimizer_exposures",
                    "games",
                )
            },
            "rss_bytes": psutil.Process().memory_info().rss,
            "host_load_1m": os.getloadavg()[0],
            "process_cpu_seconds": time.process_time()
            - cpu_start
            + sum(s.cpu_seconds for s in run.stages[:-1]),
        }

    def monitor_checkpoint(model: Agent) -> None:
        nonlocal last_monitor_seconds
        active_endpoint = (
            isinstance(stage, TrainSelfPlay) and stage.active_seconds is not None
        )
        elapsed = (
            sum(s.collection_seconds + s.learning_seconds for s in run.stages)
            if active_endpoint
            else time.perf_counter() - start - run.monitoring_export_seconds
        )
        point = coordinates()
        if checkpoint_updates is not None:
            if point["updates"] == 0 or point["updates"] % checkpoint_updates:
                return
        elif (
            checkpoint_seconds is None
            or elapsed - last_monitor_seconds < checkpoint_seconds
        ):
            return
        last_monitor_seconds = elapsed
        receipt = MonitoringCheckpoint(
            stage_id=record.id,
            ordinal=len(run.monitoring_checkpoints),
            **{
                k: point[k]
                for k in (
                    "updates",
                    "training_seconds",
                    "active_training_seconds",
                    "environment_decisions",
                    "learner_transitions",
                    "optimizer_exposures",
                    "games",
                )
            },
        )
        run.monitoring_checkpoints.append(receipt)
        tick = time.perf_counter()
        target = out / f"monitor-{receipt.ordinal:08d}.pt"
        try:
            save_bc_checkpoint(
                model,
                space,
                target.with_suffix(".tmp"),
                player_configs=Match(regime.match).to_rust(),
                extra={
                    "run_id": run.id,
                    "stage_id": record.id,
                    "regime_digest": run.regime_digest,
                    "weights": "raw",
                    "value_semantic": "win_logit"
                    if isinstance(stage, TrainSupervised)
                    and not stage.target.startswith("local_")
                    else "signed_outcome",
                },
            )
            os.replace(target.with_suffix(".tmp"), target)
            # Admission constructs a fresh model. Its initialization must not
            # consume the learner's action-sampling RNG stream.
            with torch.random.fork_rng(
                devices=[0] if next(model.parameters()).is_cuda else []
            ):
                load_checkpoint_agent(str(target))
            receipt.artifact = artifact(target)
        except Exception as error:
            receipt.error = f"{type(error).__name__}: {error}"
            if target.exists():
                record.rejected_artifacts[target.name] = artifact(target)
        finally:
            run.monitoring_export_seconds += time.perf_counter() - tick

    try:
        space = ObservationSpace(regime.observation)
        identities = _runtime_identities(seed, regime, space)
        run.identities = identities
        if snapshot is not None and identities != snapshot.identities:
            raise ValueError("recovery runtime/source identity mismatch")
        run.setup_seconds = time.perf_counter() - start
        persist()
        for stage_index, stage in enumerate(regime.stages):
            if any(item.id == stage.id for item in run.stages):
                continue
            restoring_completed = (
                snapshot is not None and snapshot.record.status == "completed"
            )
            record = (
                snapshot.record.model_copy(deep=True)
                if snapshot is not None
                else StageRecord(id=stage.id)
            )
            if restoring_completed:
                # Canonical completion includes publication cost measured after
                # the snapshot bytes were frozen. Never rewrite that record.
                record = prior_records[stage.id].model_copy(deep=True)
                following = regime.stages[stage_index + 1 : stage_index + 2]
                if not following or following[0].initial is None:
                    run.stages.append(record)
                    snapshot = None
                    continue
            if not restoring_completed:
                record.status = "running"
                record.error = None
            if snapshot is not None and not restoring_completed:
                # Counts describe the continued scientific trajectory; timing
                # describes this attempt only. Prior attempt costs live on run.
                for timing in (
                    "seconds",
                    "collection_seconds",
                    "learning_seconds",
                    "export_seconds",
                    "cpu_seconds",
                ):
                    setattr(record, timing, 0.0)
                record.cumulative_seconds = None
            run.stages.append(record)
            stage_start = time.perf_counter()
            stage_watchdog_start = watchdog_seconds()
            prior_stage_watchdog = (
                prior_records[stage.id].watchdog_seconds
                if stage.id in prior_records
                else 0.0
            )
            cpu_start = time.process_time()
            deadline = min(
                start + regime.wall_seconds - run.prior_watchdog_seconds,
                float("inf")
                if restoring_completed
                else stage_start + stage.execution.wall_seconds - prior_stage_watchdog,
            )
            if allocation is not None:
                deadline = (
                    time.perf_counter() + allocation.checkpoint_deadline - time.time()
                )
            incident_trainer = None
            batch = None
            paused = False
            collection_interrupted = False
            torch.set_num_threads(stage.execution.threads)
            cuda = stage.execution.device == "cuda"
            if cuda and not torch.cuda.is_available():
                raise ValueError(
                    "CUDA requested but unavailable; CPU fallback is forbidden"
                )

            def synchronize() -> None:
                if cuda:
                    torch.cuda.synchronize()

            synchronize()
            if not restoring_completed:
                record.actual_threads = torch.get_num_threads()

            def check() -> None:
                charged = (
                    run.prior_watchdog_seconds + watchdog_seconds() - watchdog_start
                )
                if (
                    time.perf_counter() >= deadline
                    or (
                        allocation is not None
                        and time.time() >= allocation.checkpoint_deadline
                    )
                    or allocation is None
                    and charged >= regime.wall_seconds
                    or (
                        allocation is None
                        and regime.recovery_max_microsteps is not None
                        and not restoring_completed
                        and prior_stage_watchdog
                        + watchdog_seconds()
                        - stage_watchdog_start
                        >= stage.execution.wall_seconds
                    )
                ):
                    raise TimeoutError("training resource wall deadline exceeded")
                process = psutil.Process()
                memory = process.memory_info().rss + sum(
                    p.memory_info().rss
                    for p in process.children(recursive=True)
                    if p.is_running()
                )
                if not restoring_completed:
                    record.sampled_peak_rss_bytes = max(
                        record.sampled_peak_rss_bytes, memory
                    )
                if memory > stage.execution.memory_bytes:
                    raise MemoryError("training process tree memory limit exceeded")

            references = list(getattr(stage, "datasets", []))
            if isinstance(stage, (CollectBelief, CollectLocalUpdate)):
                references.append(stage.policy)
            if isinstance(stage, TrainBelief):
                references.append(stage.dataset)
            if isinstance(stage, CollectLocalUpdate) and stage.sampler is not None:
                references.append(stage.sampler)
            if getattr(stage, "initial", None):
                references.append(stage.initial)
            for reference in references:
                source = next(item for item in run.stages if item.id == reference)
                for name, item in source.artifacts.items():
                    if file_sha256(item["path"]) != item["sha256"]:
                        raise ValueError(f"input artifact changed: {reference}/{name}")
                    record.inputs[f"{reference}/{name}"] = dict(item)
            persist()
            check()
            if isinstance(stage, ImportPolicy):
                tick = time.perf_counter()
                record.inputs = {
                    "source_run": dict(stage.source_run),
                    "checkpoint": dict(stage.checkpoint),
                }
                persist()
                admitted = admit_policy(stage, regime, out)
                record.artifacts[stage.weights] = admitted.checkpoint
                record.artifacts["source_run"] = admitted.source_run
                record.producer_cost = admitted.producer_cost
                record.export_seconds = time.perf_counter() - tick
            elif isinstance(stage, (CollectSearch, CollectLocalUpdate)):
                if isinstance(stage, CollectLocalUpdate):
                    source = next(
                        item for item in run.stages if item.id == stage.policy
                    )
                    frozen = source.artifacts[stage.weights]
                    teacher_spec = {
                        "kind": "local_update",
                        "checkpoint": frozen["path"],
                        "checkpoint_sha256": frozen["sha256"],
                        "config": stage.search.model_dump(),
                    }
                    if stage.sampler is not None:
                        sampler_record = next(
                            item for item in run.stages if item.id == stage.sampler
                        )
                        sampler_stage = next(
                            item for item in regime.stages if item.id == stage.sampler
                        )
                        assert isinstance(sampler_stage, TrainBelief)
                        data_record = next(
                            item
                            for item in run.stages
                            if item.id == sampler_stage.dataset
                        )
                        dataset_artifact = data_record.artifacts["dataset"]
                        if (
                            file_sha256(dataset_artifact["path"])
                            != dataset_artifact["sha256"]
                        ):
                            raise ValueError("sampler source dataset changed")
                        dataset = read_dataset(Path(dataset_artifact["path"]))
                        teacher_spec["sampler"] = {
                            "path": sampler_record.artifacts["sampler"]["path"],
                            "sha256": sampler_record.artifacts["sampler"]["sha256"],
                            "dataset_identity": dataset.identity,
                            "schema_identity": dataset.schema.identity,
                            "policy_identity": dataset.policy_identity,
                            "world_identity": dataset.world_identity,
                        }
                else:
                    teacher_spec = {
                        "kind": "determinized_puct",
                        "sims": stage.simulations,
                        "worlds": stage.worlds,
                        "max_steps": stage.max_steps,
                    }
                shards = []
                for _ in range(stage.games):
                    check()
                    target = out / f"{stage.id}-{game_index}.npz"
                    temporary = target.with_name(target.stem + ".tmp.npz")
                    match = (
                        Match(regime.match).swapped().hypers
                        if game_index % 2
                        else regime.match
                    )
                    summary = generate_selfplay_shard(
                        num_games=1,
                        teacher_spec=teacher_spec,
                        max_steps_per_game=stage.max_steps
                        if isinstance(stage, CollectLocalUpdate)
                        else 5000,
                        seed=seeds["collection"],
                        game_offset=game_index,
                        out_path=temporary,
                        match_hypers=match,
                        observation_hypers=regime.observation,
                        deadline_monotonic=deadline,
                    )
                    os.replace(temporary, target)
                    journal = Path(str(temporary) + ".receipts.jsonl")
                    if journal.exists():
                        record.artifacts[f"receipts-{game_index}"] = artifact(journal)
                    summary["out_path"] = str(target)
                    record.artifacts[f"game-{game_index}"] = artifact(target)
                    record.diagnostics.append(summary)
                    record.games += int(
                        all(summary["terminated"]) and not any(summary["truncated"])
                    )
                    record.environment_decisions += summary["decisions"]
                    record.collection_seconds = time.perf_counter() - stage_start
                    summary["coordinates"] = coordinates()
                    persist(progress=True)
                    if not all(summary["terminated"]) or any(summary["truncated"]):
                        raise RuntimeError(
                            "teacher game lacks authoritative terminal outcome"
                        )
                    shards.append(target)
                    game_index += 1
                outputs[stage.id] = shards
            elif isinstance(stage, TrainCompound):
                if stage.initial:
                    agent, optimizer = outputs[stage.initial]
                else:
                    torch.manual_seed(seeds["initialization"])
                    agent = Agent(space, regime.agent)
                    optimizer = torch.optim.Adam(
                        agent.parameters(), lr=stage.learning.learning_rate.initial
                    )
                sampling = torch.Generator().manual_seed(
                    seeds["collection"] + game_index
                )
                shuffling = torch.Generator().manual_seed(
                    seeds["minibatches"] + game_index
                )
                stats = CompoundStatistics()
                record.diagnostics = [asdict(stats)]
                for _ in range(stage.updates):
                    games = []
                    for _ in range(stage.games_per_update):
                        target = out / f"{stage.id}-game-{game_index}.jsonl"
                        match = Match(regime.match)
                        if game_index % 2:
                            match = match.swapped()
                        tick = time.perf_counter()
                        stats.attempted_games += 1
                        try:
                            game = collect_game(
                                agent,
                                match,
                                seeds["collection"] + game_index,
                                sampling,
                                target,
                                max_commands=stage.max_commands,
                                check=check,
                                skip_trivial=stage.skip_trivial,
                            )
                            # Replay consumes the retained Commands, never another policy.
                            check()
                            replay_game(target)
                        except BaseException:
                            stats.failed_games += 1
                            if target.exists():
                                record.rejected_artifacts[f"game-{game_index}"] = (
                                    artifact(target)
                                )
                                with target.open() as partial:
                                    stats.interrupted_microchoices += sum(
                                        '"command":' in line for line in partial
                                    )
                            record.diagnostics = [asdict(stats)]
                            raise
                        record.collection_seconds += time.perf_counter() - tick
                        record.artifacts[f"game-{game_index}"] = artifact(target)
                        stats.record_game(game)
                        record.games = stats.games
                        record.environment_decisions = stats.microchoices
                        record.learner_transitions = (
                            stats.factors
                            if stage.grouping == "sequential"
                            else stats.decisions
                        )
                        games.append(game)
                        game_index += 1
                        stats.collection_seconds = record.collection_seconds
                        record.diagnostics = [asdict(stats)]
                        persist(progress=True)
                    tick = time.perf_counter()
                    update = optimize_games(
                        agent,
                        optimizer,
                        games,
                        stage.learning,
                        grouped=stage.grouping == "grouped",
                        estimator=stage.estimator,
                        progress=(time.perf_counter() - start) / regime.wall_seconds,
                        generator=shuffling,
                        check=check,
                    )
                    record.learning_seconds += time.perf_counter() - tick
                    record.optimizer_exposures += update.optimizer_exposures
                    stats.learning_seconds = record.learning_seconds
                    stats.optimizer_exposures += update.optimizer_exposures
                    stats.losses.extend(update.losses)
                    record.diagnostics = [asdict(stats)]
                    persist(progress=True)
                outputs[stage.id] = (agent, optimizer)
                optimizer_state = optimizer.state_dict()
            elif isinstance(stage, CollectSelection):
                policy_run = (
                    store.training_run(stage.source_run) if stage.source_run else run
                )
                source = next(
                    item for item in policy_run.stages if item.id == stage.policy
                )
                policy_stage = next(
                    item for item in policy_run.regime.stages if item.id == stage.policy
                )
                if source.status != "completed" or not isinstance(
                    policy_stage, TrainSelfPlay
                ):
                    raise ValueError(
                        "selection requires a completed self-play checkpoint stage"
                    )
                frozen = source.artifacts[stage.weights]
                record.inputs["policy"] = dict(frozen)
                persist()
                if file_sha256(frozen["path"]) != frozen["sha256"]:
                    raise ValueError("selection policy artifact changed")
                agent, policy_space = load_checkpoint_agent(frozen["path"])
                if (
                    agent.hypers != regime.agent
                    or policy_space.encoder.hypers != regime.observation
                ):
                    raise ValueError(
                        "selection policy configuration differs from regime"
                    )
                if file_sha256(frozen["path"]) != frozen["sha256"]:
                    raise ValueError("selection policy artifact changed while loading")
                population = [game.model_dump(mode="json") for game in stage.population]
                population_path = out / f"{stage.id}-population.json"
                atomic_json(population_path, population)
                record.artifacts["population"] = artifact(population_path)
                persist()
                games = []
                match = Match(regime.match)
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
                    if replay_game(target) != len(game.rows):
                        raise ValueError("selection replay decision count mismatch")
                    record.diagnostic_seconds += time.perf_counter() - tick
                    record.artifacts[f"game-{index}"] = artifact(target)
                    record.games += 1
                    record.environment_decisions += len(game.rows)
                    games.append(game)
                    persist(progress=True)
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
            elif isinstance(stage, CollectBelief):
                source = next(item for item in run.stages if item.id == stage.policy)
                policy_artifact = source.artifacts[stage.weights]
                dataset = collect_frozen_policy(
                    checkpoint=Path(policy_artifact["path"]),
                    match_hypers=regime.match,
                    games=stage.games,
                    seed=seeds["belief_collection"] + game_index,
                    max_steps=stage.max_steps,
                    check=check,
                )
                record.collection_seconds = time.perf_counter() - stage_start
                record.games = len(dataset.games)
                record.environment_decisions = sum(
                    len(game.examples) // 2 for game in dataset.games
                )
                game_index += len(dataset.games)
                tick = time.perf_counter()
                target = out / f"{stage.id}-dataset.json"
                save_dataset(dataset, target)
                record.artifacts["dataset"] = artifact(target)
                record.export_seconds = time.perf_counter() - tick
            elif isinstance(stage, TrainBelief):
                source = next(item for item in run.stages if item.id == stage.dataset)
                dataset = read_dataset(Path(source.artifacts["dataset"]["path"]))
                tick = time.perf_counter()
                result = fit_belief_sampler(
                    dataset,
                    steps=stage.steps,
                    batch_size=stage.batch_size,
                    hidden_size=stage.hidden_size,
                    learning_rate=stage.learning_rate,
                    history_dropout=stage.history_dropout,
                    evaluation_samples=stage.evaluation_samples,
                    seed=seeds["belief_initialization"],
                    check=check,
                )
                record.learning_seconds = time.perf_counter() - tick
                record.optimizer_exposures = result.optimizer_exposures
                record.diagnostics = [asdict(result.metrics)]
                tick = time.perf_counter()
                target = out / f"{stage.id}-sampler.pt"
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
            elif isinstance(stage, TrainSelfPlay):
                imported = None
                if stage.learning_state is not None:
                    record.inputs.update(
                        {
                            f"learning_state/{role}": dict(
                                getattr(stage.learning_state, role)
                            )
                            for role in ("source_run", "raw", "ema", "optimizer")
                        }
                    )
                    persist()
                    tick = time.perf_counter()
                    imported = admit_learning_state(stage, regime, out)
                    if imported.seed != seed:
                        raise ValueError(
                            "learning-state continuation must retain training seed"
                        )
                    record.artifacts.update(imported.artifacts)
                    record.learning_state_origin = imported.origin
                    record.export_seconds += time.perf_counter() - tick
                    # A segment has new stochastic streams, deterministically
                    # separated from the producer by its absolute boundary.
                    for name in ("initialization", "collection", "minibatches"):
                        seeds[name] += imported.origin.iteration
                    run.seed_streams = dict(seeds)
                    persist()
                opponent_agent = None
                if stage.opponent is not None:
                    # Validate before every stage, even when retaining the live
                    # collector. Changed disk bytes cannot silently change a run.
                    opponent_path = Path(stage.opponent.path).resolve()
                    if file_sha256(opponent_path) != stage.opponent.sha256:
                        raise ValueError("frozen opponent checkpoint digest differs")
                    record.inputs["frozen_opponent"] = artifact(opponent_path)
                    if not stage.initial or snapshot is not None:
                        opponent_agent, opponent_space = load_checkpoint_agent(
                            str(opponent_path)
                        )
                        validate_agent_setup(
                            opponent_agent, Match(regime.match).to_rust()
                        )
                        if opponent_space.encoder.hypers != space.encoder.hypers:
                            raise ValueError("frozen opponent observation ABI differs")
                        if opponent_agent.hypers.compound_decisions:
                            raise ValueError(
                                "frozen collector does not support compound opponent submissions"
                            )
                        if opponent_agent.belief_count_buckets:
                            raise ValueError(
                                "frozen collector does not supply belief inputs"
                            )
                        opponent_agent.requires_grad_(False)
                if stage.initial and snapshot is None:
                    # Validation permits only the latest live collector to continue.
                    trainer, ema, iteration = self_play_session
                else:
                    torch.manual_seed(seeds["initialization"])
                    agent = Agent(space, regime.agent).to(stage.execution.device)
                    if imported is not None:
                        agent.load_state_dict(imported.raw.state_dict())
                    if (
                        stage.opponent is not None
                        or checkpoint_seconds is not None
                        or (checkpoint_updates is not None and imported is None)
                    ) and snapshot is None:
                        tick = time.perf_counter()
                        target = out / f"{stage.id}-initial-raw.pt"
                        temporary = target.with_suffix(".tmp")
                        save_bc_checkpoint(
                            agent,
                            space,
                            temporary,
                            player_configs=Match(regime.match).to_rust(),
                            extra={
                                "run_id": run.id,
                                "stage_id": stage.id,
                                "weights": "initial_raw",
                                "regime_digest": run.regime_digest,
                            },
                        )
                        os.replace(temporary, target)
                        # New self-play monitoring must preserve initialization and
                        # action-sampling streams compared with monitoring disabled.
                        if stage.opponent is None:
                            with torch.random.fork_rng(
                                devices=[0] if next(agent.parameters()).is_cuda else []
                            ):
                                load_checkpoint_agent(str(target))
                        else:
                            load_checkpoint_agent(str(target))
                        record.artifacts["initial_raw"] = artifact(target)
                        record.export_seconds += time.perf_counter() - tick
                        persist()
                        if initial_admission is not None:
                            # The supervisor releases this exact checkpoint only
                            # after its initialization cohort is complete and
                            # durably published. Waiting consumes the watchdog,
                            # never collection/learning time or RNG draws.
                            while not initial_admission.exists():
                                check()
                                time.sleep(0.2)
                            if (
                                initial_admission.read_text().strip()
                                != record.artifacts["initial_raw"]["sha256"]
                            ):
                                raise ValueError("initial evaluation admission differs")
                    collector = SeatRoutedCollector(
                        space,
                        Match(regime.match),
                        Reward(RewardHypers()),
                        num_envs=stage.streams,
                        seed=seeds["collection"],
                        opponent_mode="frozen"
                        if stage.opponent is not None
                        else "self",
                        opponent_agent=opponent_agent,
                        recovery_max_microsteps=regime.recovery_max_microsteps,
                        device=stage.execution.device,
                    )
                    experiment = Experiment(
                        ExperimentHypers(
                            wandb=False,
                            seed=seeds["initialization"]
                            if imported is not None
                            else seed,
                            runs_dir=out,
                            exp_name=stage.id,
                            device=stage.execution.device,
                            log_level="WARNING",
                        )
                    )
                    resources.callback(experiment.close)
                    trainer = NetOpponentTrainer(
                        agent,
                        experiment,
                        collector,
                        TrainHypers(
                            num_envs=stage.streams, num_steps=stage.transitions
                        ),
                    )
                    ema = deepcopy(agent) if stage.learning.ema is not None else None
                    iteration = 0
                    if imported is not None:
                        trainer.optimizer.load_state_dict(imported.optimizer)
                        validate_adam(trainer.optimizer)
                        assert ema is not None
                        ema.load_state_dict(imported.ema.state_dict())
                        iteration = imported.origin.iteration
                        imported = None
                if not restoring_completed:
                    record.actual_device = str(next(trainer.agent.parameters()).device)
                before = deepcopy(trainer.collector.stats)
                rng = np.random.default_rng(seeds["minibatches"] + iteration)
                first_update = 0
                if record.learning_state_origin is not None:
                    first_update = record.learning_state_origin.iteration
                if snapshot is not None:
                    tick = time.perf_counter()
                    try:
                        trainer.collector.restore(snapshot.collector, check)
                        restore_learning(snapshot, trainer, ema, rng)
                    finally:
                        run.recovery_seconds += time.perf_counter() - tick
                    iteration = snapshot.iteration
                    first_update = len(snapshot.record.diagnostics)
                    # Restored counts are committed scientific work; failed work
                    # remains in the parent attempt and its cost stays charged.
                    before = snapshot.collector_before
                if restoring_completed:
                    self_play_session = (trainer, ema, iteration)
                    snapshot = None
                    continue

                def checkpoint(label: str) -> None:
                    if regime.recovery_max_microsteps is None:
                        return
                    tick = time.perf_counter()
                    record.watchdog_seconds = (
                        prior_stage_watchdog + watchdog_seconds() - stage_watchdog_start
                    )
                    target = out / f"{stage.id}-{label}.pt"
                    save_update(
                        target, trainer, ema, rng, iteration, record, run, before
                    )
                    run.recovery_artifact = artifact(target)
                    record.export_seconds += time.perf_counter() - tick

                if snapshot is None:
                    checkpoint("start")
                    persist()
                snapshot = None
                incident_trainer = trainer
                for update_index in range(first_update, stage.updates):
                    # Stop only at an exact collector/update boundary. The saved
                    # snapshot includes Adam, EMA, RNGs and paused native streams.
                    if allocation is not None and time.time() >= allocation.pause_at:
                        paused = True
                        break
                    if stage.active_seconds is not None and (
                        record.collection_seconds + record.learning_seconds
                        >= stage.active_seconds
                    ):
                        break
                    check()
                    tick = time.perf_counter()
                    behavior_agent = (
                        ema if stage.behavior == "ema-self" else trainer.agent
                    )
                    assert behavior_agent is not None
                    batch = None
                    try:
                        batch = trainer.collector.collect(
                            behavior_agent,
                            stage.transitions,
                            deadline_monotonic=min(
                                deadline,
                                time.perf_counter() + allocation.pause_at - time.time(),
                            )
                            if allocation is not None
                            else deadline,
                            check=check
                            if regime.recovery_max_microsteps is not None
                            else None,
                        )
                    except TimeoutError:
                        if allocation is None or time.time() < allocation.pause_at:
                            raise
                        paused = collection_interrupted = True
                        break
                    finally:
                        synchronize()
                        record.collection_seconds += time.perf_counter() - tick
                    check()
                    tick = time.perf_counter()
                    try:
                        diagnostic = update_iteration(
                            trainer,
                            batch,
                            stage.learning,
                            update_index / stage.updates
                            if regime.schedule_clock == "iteration_fraction"
                            else (
                                time.perf_counter()
                                - start
                                - run.monitoring_export_seconds
                            )
                            / regime.wall_seconds,
                            rng,
                            iteration=iteration + 1,
                            bootstrap_agent=behavior_agent,
                        )
                        diagnostic["behavior"] = stage.behavior
                        diagnostic["behavior_iteration"] = iteration
                        if stage.learning.gradient != "ataraxos_move":
                            diagnostic["schedule_clock"] = regime.schedule_clock
                        iteration += 1
                        if ema is not None:
                            update_ema(ema, trainer.agent, stage.learning.ema)
                    finally:
                        synchronize()
                        record.learning_seconds += time.perf_counter() - tick
                    record.diagnostics.append(diagnostic)
                    record.optimizer_exposures += diagnostic["optimizer_exposures"]
                    record.games = trainer.collector.stats.games - before.games
                    record.environment_decisions = (
                        trainer.collector.stats.micro_steps - before.micro_steps
                    )
                    record.learner_transitions = (
                        trainer.collector.stats.learner_transitions
                        - before.learner_transitions
                    )
                    diagnostic["coordinates"] = coordinates()
                    # The completed stage export owns an active-time endpoint;
                    # do not enqueue a duplicate monitoring cohort at that point.
                    if checkpoint_updates is not None:
                        # The final raw export owns an exact endpoint, including
                        # one that also lies on the regular cadence grid.
                        if update_index + 1 < stage.updates:
                            monitor_checkpoint(trainer.agent)
                    elif stage.active_seconds is None or (
                        record.collection_seconds + record.learning_seconds
                        < stage.active_seconds
                    ):
                        monitor_checkpoint(trainer.agent)
                    checkpoint(f"update-{iteration:08d}")
                    persist(progress=True)
                if stage.active_seconds is not None and (
                    record.collection_seconds + record.learning_seconds
                    < stage.active_seconds
                ):
                    raise RuntimeError(
                        "update safety ceiling reached before active training endpoint"
                    )
                self_play_session = (trainer, ema, iteration)
                agent = trainer.agent
                optimizer_state = trainer.optimizer.state_dict()
            else:
                dataset = load_shards(
                    [p for ref in stage.datasets for p in outputs[ref]],
                    globally_unique_games=True,
                )
                validation = {
                    int(g) for g in np.unique(dataset["game_index"]) if int(g) % 10 == 0
                }
                if fixed_validation_dataset is None:
                    mask = np.isin(dataset["game_index"], list(validation))
                    if not mask.any():
                        raise ValueError("fixed validation cohort is empty")
                    fixed_validation_dataset = {
                        k: v[mask].copy() for k, v in dataset.items()
                    }
                    cohort = out / "fixed-validation.npz"
                    np.savez_compressed(cohort, **fixed_validation_dataset)
                    run.fixed_validation = FixedValidationCohort(
                        artifact=artifact(cohort),
                        games=sorted(validation),
                        policy_target_kind=stage.target,
                        source_inputs=deepcopy(record.inputs),
                    )
                assert run.fixed_validation is not None
                validation.update(
                    int(g) for g in fixed_validation_dataset["game_index"]
                )
                previous = outputs.get(stage.initial, {})
                if stage.initial and not previous:
                    source = next(
                        item for item in run.stages if item.id == stage.initial
                    )
                    initial_agent, _ = load_checkpoint_agent(
                        source.artifacts[stage.initial_weights]["path"]
                    )
                    previous = {"agent": initial_agent}
                continuation = {}
                tick = time.perf_counter()

                def report_epoch(
                    stats: SearchSupervisedEpochStats, model: Agent
                ) -> None:
                    record.actual_device = str(next(model.parameters()).device)
                    record.learning_seconds = time.perf_counter() - tick
                    record.optimizer_exposures += int(
                        (~np.isin(dataset["game_index"], list(validation))).sum()
                    )
                    row = asdict(stats)
                    record.diagnostics.append(row)
                    row["coordinates"] = coordinates()
                    monitor_checkpoint(model)
                    persist(progress=True)

                agent, _, _, history = train_search_supervised(
                    dataset,
                    policy_target_kind=stage.target,
                    value_weight=0,
                    agent_hypers=regime.agent,
                    observation_hypers=regime.observation,
                    epochs=stage.epochs,
                    batch_size=stage.batch_size,
                    lr=stage.learning_rate,
                    seed=seeds["initialization"],
                    minibatch_seed=seeds["minibatches"],
                    validation_games=validation,
                    initial_agent_state=previous["agent"].state_dict()
                    if previous
                    else None,
                    optimizer_state=previous.get("optimizer_state"),
                    continuation=continuation,
                    deadline_monotonic=deadline,
                    fixed_validation_dataset=fixed_validation_dataset,
                    fixed_validation_target_kind=run.fixed_validation.policy_target_kind,
                    on_epoch=report_epoch,
                )
                record.learning_seconds = time.perf_counter() - tick
                record.optimizer_exposures = int(
                    (~np.isin(dataset["game_index"], list(validation))).sum()
                ) * len(history)
                outputs[stage.id] = {"agent": agent, **deepcopy(continuation)}
                optimizer_state = continuation["optimizer_state"]
            if isinstance(stage, (TrainSelfPlay, TrainSupervised, TrainCompound)):
                tick = time.perf_counter()
                variants = {"raw": agent}
                if isinstance(stage, TrainSelfPlay) and ema is not None:
                    variants["ema"] = ema
                for name, model in variants.items():
                    target = out / f"{stage.id}-{name}.pt"
                    temporary = target.with_suffix(".tmp")
                    save_bc_checkpoint(
                        model,
                        space,
                        temporary,
                        player_configs=Match(regime.match).to_rust(),
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
                            "regime_digest": run.regime_digest,
                            "value_semantic": "signed_outcome"
                            if isinstance(stage, (TrainSelfPlay, TrainCompound))
                            or stage.target.startswith("local_")
                            else "win_logit",
                        },
                    )
                    os.replace(temporary, target)
                    candidate = artifact(target)
                    try:
                        if allocation is not None:
                            # Admission constructs a model; pause exports must not
                            # perturb the RNG resumed by the next allocation.
                            with torch.random.fork_rng(devices=[0] if cuda else []):
                                load_checkpoint_agent(str(target))
                        else:
                            load_checkpoint_agent(str(target))
                    except Exception:
                        record.rejected_artifacts[name] = candidate
                        raise
                    record.artifacts[name] = candidate
                optimizer_path = out / f"{stage.id}-optimizer.pt"
                torch.save(
                    optimizer_state,
                    optimizer_path.with_suffix(".tmp"),
                )
                os.replace(optimizer_path.with_suffix(".tmp"), optimizer_path)
                record.artifacts["optimizer"] = artifact(optimizer_path)
                record.export_seconds += time.perf_counter() - tick
            check()
            record.seconds = time.perf_counter() - stage_start
            record.cpu_seconds = time.process_time() - cpu_start
            record.status = "paused" if paused else "completed"
            # Freeze the admission cost; later persistence belongs to later outputs.
            record.cumulative_seconds = run.prior_seconds + time.perf_counter() - start
            record.watchdog_seconds = (
                prior_stage_watchdog + watchdog_seconds() - stage_watchdog_start
            )
            if isinstance(stage, TrainSelfPlay):
                try:
                    if not collection_interrupted:
                        checkpoint("paused" if paused else "completed")
                except BaseException:
                    record.status = "running"
                    raise
                record.seconds = time.perf_counter() - stage_start
                record.cpu_seconds = time.process_time() - cpu_start
                record.watchdog_seconds = (
                    prior_stage_watchdog + watchdog_seconds() - stage_watchdog_start
                )
                record.cumulative_seconds = (
                    run.prior_seconds + time.perf_counter() - start
                )
            persist()
            if paused:
                run.status = "paused"
                run.error = "Allocation ended before the update target; explicit recovery required"
                persist()
                return run
        completed_models = [
            item.artifacts["raw"] for item in run.stages if "raw" in item.artifacts
        ]
        run.selected_artifact = completed_models[-1] if completed_models else None
        run.status = "completed"
        persist()
    except BaseException as error:
        run.status = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, TimeoutError))
            else "failed"
        )
        run.error = f"{type(error).__name__}: {error}"
        if run.stages and run.stages[-1].status != "completed":
            run.stages[-1].watchdog_seconds = (
                prior_stage_watchdog + watchdog_seconds() - stage_watchdog_start
            )
            run.stages[-1].status = run.status
            run.stages[-1].error = run.error
            run.stages[-1].seconds = time.perf_counter() - stage_start
            record = run.stages[-1]
            retained_paths = {item["path"] for item in record.artifacts.values()}
            for journal in out.glob(f"{record.id}-*.receipts.jsonl"):
                if str(journal) not in retained_paths:
                    record.rejected_artifacts[journal.name] = artifact(journal)
            record.cpu_seconds = time.process_time() - cpu_start
            unaccounted = max(
                0.0,
                record.seconds
                - record.collection_seconds
                - record.learning_seconds
                - record.export_seconds
                - record.diagnostic_seconds,
            )
            # Unmeasured setup, persistence and waits are never active training.
            # Self-play operation timers settle in finally, including failed work.
            record.diagnostic_seconds += unaccounted
            if isinstance(error, NumericalError) and incident_trainer is not None:
                record.numerical_failure = NumericalFailure(
                    invariant=error.invariant,
                    iteration=iteration + 1,
                    health={
                        **error.health,
                        "rejected_updates": 1,
                        "nonfinite_failure": int(
                            error.invariant.startswith("nonfinite")
                        ),
                    },
                )
                capture_start = time.perf_counter()
                try:
                    paths = save_incident(
                        out / f"{record.id}-numerical-incident",
                        error,
                        incident_trainer,
                        batch,
                        run,
                        iteration + 1,
                        ema,
                        rng,
                    )
                    for path in paths:
                        record.rejected_artifacts[f"numerical/{path.name}"] = artifact(
                            path
                        )
                except Exception as capture_error:
                    # Failure capture cannot replace the primary training error.
                    run.error += (
                        f"; incident capture failed: {type(capture_error).__name__}"
                    )
                    record.error = run.error
                record.diagnostic_seconds += time.perf_counter() - capture_start
                record.seconds = time.perf_counter() - stage_start
        run.seconds = time.perf_counter() - start
        run.watchdog_seconds = watchdog_seconds() - watchdog_start
        run.last_recorded_wall_seconds = time.time()
        try:
            store.save_training_run(run)
            export_training_run(run.id, store, out)
        except Exception as telemetry_error:
            error.add_note(
                f"terminal telemetry failed: {type(telemetry_error).__name__}"
            )
            try:
                atomic_json(
                    out / "terminal-error.json", {"error": run.error, "run": run.id}
                )
            except OSError:
                pass
        raise
    finally:
        resources.close()
    return run
