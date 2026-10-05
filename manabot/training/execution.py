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
from manabot.model.world import validate_agent_setup
from manabot.sim.distill import generate_selfplay_shard, load_shards, save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import NetOpponentTrainer, SeatRoutedCollector
from manabot.sim.search_supervised import train_search_supervised
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
import managym

from .clock import watchdog_seconds
from .compound import CompoundStatistics, collect_game, optimize_games, replay_game
from .models import (
    ArtifactReference,
    CollectBelief,
    CollectLocalUpdate,
    CollectSearch,
    CollectSelection,
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


def execute_regime(
    regime: TrainingRegime | Mapping[str, object],
    seed: int,
    out: str | Path,
    store: "VerifyStore",
    *,
    resume_from: str | None = None,
) -> TrainingRun:
    """Execute under a local lease, including admission and crash settlement."""
    regime = validate_regime(regime)
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
            load_update(parent)
            settled = settle_orphan(parent)
            if settled != parent:
                # Preserve the old run.json and snapshots. The canonical record
                # explains settlement, including the explicitly estimated cost.
                store.save_training_run(settled)
        return _execute_regime(
            regime, seed, destination, store, resume_from=resume_from
        )


def _execute_regime(
    regime: TrainingRegime,
    seed: int,
    out: str | Path,
    store: "VerifyStore",
    *,
    resume_from: str | None = None,
) -> TrainingRun:
    regime = validate_regime(regime)
    parent = store.training_run(resume_from) if resume_from is not None else None
    if parent is not None:
        if parent.status not in {"failed", "interrupted"}:
            raise ValueError(
                "recovery requires a stopped failed or interrupted attempt"
            )
        if (
            regime.recovery_max_microsteps is None
            or parent.seed != seed
            or parent.regime_digest != canonical_sha256(regime.model_dump(mode="json"))
        ):
            raise ValueError("recovery recipe or seed mismatch")
    snapshot = load_update(parent) if parent is not None else None
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
        identities={},
        status="running",
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
    )
    if parent is None:
        store.save_training_run(run)
    else:
        store.claim_training_recovery(parent.id, run)
    outputs = {}
    self_play_session = None
    resources = ExitStack()
    game_index = 0

    def persist() -> None:
        run.seconds = time.perf_counter() - start
        run.watchdog_seconds = watchdog_seconds() - watchdog_start
        run.last_recorded_wall_seconds = time.time()
        store.save_training_run(run)
        export_training_run(run.id, store, out)

    def artifact(path: Path) -> ArtifactReference:
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }

    try:
        space = ObservationSpace(regime.observation)
        identities = runtime_fingerprints(
            seed, match_hypers=regime.match, observation_space=space
        )
        identities.update(
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
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        )
        run.identities = identities
        if snapshot is not None and identities != snapshot.identities:
            raise ValueError("recovery runtime/source identity mismatch")
        run.setup_seconds = time.perf_counter() - start
        persist()
        for stage in regime.stages:
            record = (
                snapshot.record.model_copy(deep=True)
                if snapshot is not None
                else StageRecord(id=stage.id)
            )
            record.status = "running"
            record.error = None
            if snapshot is not None:
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
            cpu_start = time.process_time()
            deadline = min(
                start + regime.wall_seconds - run.prior_watchdog_seconds,
                stage_start + stage.execution.wall_seconds - run.prior_watchdog_seconds,
            )
            phase = "collection_seconds"
            torch.set_num_threads(stage.execution.threads)
            record.actual_threads = torch.get_num_threads()

            def check() -> None:
                charged = (
                    run.prior_watchdog_seconds + watchdog_seconds() - watchdog_start
                )
                if (
                    time.perf_counter() >= deadline
                    or charged >= regime.wall_seconds
                    or (
                        regime.recovery_max_microsteps is not None
                        and charged >= stage.execution.wall_seconds
                    )
                ):
                    raise TimeoutError("training resource wall deadline exceeded")
                process = psutil.Process()
                memory = process.memory_info().rss + sum(
                    p.memory_info().rss
                    for p in process.children(recursive=True)
                    if p.is_running()
                )
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
            if isinstance(stage, (CollectSearch, CollectLocalUpdate)):
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
                    persist()
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
                        phase = "collection_seconds"
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
                        persist()
                    phase = "learning_seconds"
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
                    persist()
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
                    phase = "diagnostic_seconds"
                    tick = time.perf_counter()
                    if replay_game(target) != len(game.rows):
                        raise ValueError("selection replay decision count mismatch")
                    record.diagnostic_seconds += time.perf_counter() - tick
                    record.artifacts[f"game-{index}"] = artifact(target)
                    record.games += 1
                    record.environment_decisions += len(game.rows)
                    games.append(game)
                    persist()
                    phase = "collection_seconds"
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
                phase = "diagnostic_seconds"
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
                phase = "export_seconds"
                tick = time.perf_counter()
                target = out / f"{stage.id}-dataset.json"
                save_dataset(dataset, target)
                record.artifacts["dataset"] = artifact(target)
                record.export_seconds = time.perf_counter() - tick
            elif isinstance(stage, TrainBelief):
                source = next(item for item in run.stages if item.id == stage.dataset)
                dataset = read_dataset(Path(source.artifacts["dataset"]["path"]))
                phase = "learning_seconds"
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
                phase = "export_seconds"
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
                opponent_agent = None
                if stage.opponent is not None:
                    # Validate before every stage, even when retaining the live
                    # collector. Changed disk bytes cannot silently change a run.
                    opponent_path = Path(stage.opponent.path).resolve()
                    if file_sha256(opponent_path) != stage.opponent.sha256:
                        raise ValueError("frozen opponent checkpoint digest differs")
                    record.inputs["frozen_opponent"] = artifact(opponent_path)
                    if not stage.initial:
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
                if stage.initial:
                    # Validation permits only the latest live collector to continue.
                    trainer, ema, iteration = self_play_session
                else:
                    torch.manual_seed(seeds["initialization"])
                    agent = Agent(space, regime.agent)
                    if stage.opponent is not None:
                        phase = "export_seconds"
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
                        load_checkpoint_agent(str(target))
                        record.artifacts["initial_raw"] = artifact(target)
                        record.export_seconds += time.perf_counter() - tick
                        persist()
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
                    )
                    experiment = Experiment(
                        ExperimentHypers(
                            wandb=False,
                            seed=seed,
                            runs_dir=out,
                            exp_name=stage.id,
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
                record.actual_device = str(next(trainer.agent.parameters()).device)
                before = deepcopy(trainer.collector.stats)
                rng = np.random.default_rng(seeds["minibatches"] + iteration)
                first_update = 0
                if snapshot is not None:
                    tick = time.perf_counter()
                    trainer.collector.restore(snapshot.collector, check)
                    restore_learning(snapshot, trainer, ema, rng)
                    iteration = first_update = snapshot.iteration
                    # Restored counts are committed scientific work; failed work
                    # remains in the parent attempt and its cost stays charged.
                    before = type(before)()
                    run.recovery_seconds = time.perf_counter() - tick
                for update_index in range(first_update, stage.updates):
                    check()
                    tick = time.perf_counter()
                    phase = "collection_seconds"
                    behavior_agent = (
                        ema if stage.behavior == "ema-self" else trainer.agent
                    )
                    assert behavior_agent is not None
                    batch = trainer.collector.collect(
                        behavior_agent,
                        stage.transitions,
                        deadline_monotonic=deadline,
                        check=check
                        if regime.recovery_max_microsteps is not None
                        else None,
                    )
                    record.collection_seconds += time.perf_counter() - tick
                    check()
                    phase = "learning_seconds"
                    tick = time.perf_counter()
                    diagnostic = update_iteration(
                        trainer,
                        batch,
                        stage.learning,
                        update_index / stage.updates
                        if regime.schedule_clock == "iteration_fraction"
                        else (time.perf_counter() - start) / regime.wall_seconds,
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
                    if regime.recovery_max_microsteps is not None:
                        tick = time.perf_counter()
                        target = out / f"update-{iteration:08d}.pt"
                        save_update(target, trainer, ema, rng, iteration, record, run)
                        run.recovery_artifact = artifact(target)
                        record.export_seconds += time.perf_counter() - tick
                    persist()
                self_play_session = (trainer, ema, iteration)
                agent = trainer.agent
                optimizer_state = trainer.optimizer.state_dict()
            else:
                dataset = load_shards(
                    [p for ref in stage.datasets for p in outputs[ref]]
                )
                validation = {
                    int(g) for g in np.unique(dataset["game_index"]) if int(g) % 10 == 0
                }
                previous = outputs.get(stage.initial, {})
                if stage.initial and not previous:
                    source = next(
                        item for item in run.stages if item.id == stage.initial
                    )
                    initial_agent, _ = load_checkpoint_agent(
                        source.artifacts["raw"]["path"]
                    )
                    previous = {"agent": initial_agent}
                continuation = {}
                phase = "learning_seconds"
                tick = time.perf_counter()
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
                )
                record.learning_seconds = time.perf_counter() - tick
                record.optimizer_exposures = int(
                    (~np.isin(dataset["game_index"], list(validation))).sum()
                ) * len(history)
                record.diagnostics = [asdict(item) for item in history]
                outputs[stage.id] = {"agent": agent, **deepcopy(continuation)}
                optimizer_state = continuation["optimizer_state"]
            if isinstance(stage, (TrainSelfPlay, TrainSupervised, TrainCompound)):
                phase = "export_seconds"
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
            record.status = "completed"
            # Freeze the admission cost; later persistence belongs to later outputs.
            record.cumulative_seconds = run.prior_seconds + time.perf_counter() - start
            persist()
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
        if run.stages:
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
            setattr(record, phase, getattr(record, phase) + unaccounted)
        run.seconds = time.perf_counter() - start
        run.watchdog_seconds = watchdog_seconds() - watchdog_start
        run.last_recorded_wall_seconds = time.time()
        store.save_training_run(run)
        try:
            export_training_run(run.id, store, out)
        except OSError:
            pass
        raise
    finally:
        resources.close()
    return run
