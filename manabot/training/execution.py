"""Bounded regime execution; VerifyStore owns state, manifests are exports."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any
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
from manabot.sim.distill import generate_selfplay_shard, load_shards, save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import NetOpponentTrainer, SeatRoutedCollector
from manabot.sim.search_supervised import train_search_supervised
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
from manabot.verify.store import VerifyStore
import managym

from .models import (
    CollectBelief,
    CollectSearch,
    StageRecord,
    TrainBelief,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
    TrainSupervised,
)
from .objectives import update_ema, update_iteration


def atomic_json(path: Path | str, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def export_training_run(
    run_id: str, store: VerifyStore, out: Path | str
) -> TrainingRun:
    run = store.training_run(run_id)
    atomic_json(Path(out) / "run.json", run.model_dump(mode="json"))
    return run


def validate_regime(regime: TrainingRegime | dict[str, Any]) -> TrainingRegime:
    regime = TrainingRegime.model_validate(
        regime.model_dump() if isinstance(regime, TrainingRegime) else regime
    )
    if regime.world != managym.WORLD_VERSION:
        raise ValueError("regime world differs from native runtime")
    if regime.agent.belief_count_buckets:
        raise ValueError("regime stages do not produce belief inputs")
    return regime


def execute_regime(
    regime: TrainingRegime | dict[str, Any],
    seed: int,
    out: Path | str,
    store: VerifyStore,
) -> TrainingRun:
    regime = validate_regime(regime)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
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
    )
    store.save_training_run(run)
    outputs = {}
    self_play_session = None
    resources = ExitStack()
    game_index = 0

    def persist() -> None:
        run.seconds = time.perf_counter() - start
        store.save_training_run(run)
        export_training_run(run.id, store, out)

    def artifact(path: Path) -> dict[str, str | int]:
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
        run.setup_seconds = time.perf_counter() - start
        persist()
        for stage in regime.stages:
            record = StageRecord(id=stage.id)
            run.stages.append(record)
            stage_start = time.perf_counter()
            cpu_start = time.process_time()
            deadline = min(
                start + regime.wall_seconds, stage_start + stage.execution.wall_seconds
            )
            phase = "collection_seconds"
            torch.set_num_threads(stage.execution.threads)

            def check() -> None:
                if time.perf_counter() >= deadline:
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
            if isinstance(stage, CollectBelief):
                references.append(stage.policy)
            if isinstance(stage, TrainBelief):
                references.append(stage.dataset)
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
            if isinstance(stage, CollectSearch):
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
                        teacher_spec={
                            "kind": "determinized_puct",
                            "sims": stage.simulations,
                            "worlds": stage.worlds,
                            "max_steps": stage.max_steps,
                        },
                        seed=seeds["collection"],
                        game_offset=game_index,
                        out_path=temporary,
                        match_hypers=match,
                        observation_hypers=regime.observation,
                        deadline_monotonic=deadline,
                    )
                    os.replace(temporary, target)
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
                if stage.initial:
                    # Validation permits only the latest live collector to continue.
                    trainer, ema, iteration = self_play_session
                else:
                    torch.manual_seed(seeds["initialization"])
                    agent = Agent(space, regime.agent)
                    collector = SeatRoutedCollector(
                        space,
                        Match(regime.match),
                        Reward(RewardHypers()),
                        num_envs=stage.streams,
                        seed=seeds["collection"],
                        opponent_mode="self",
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
                before = deepcopy(trainer.collector.stats)
                rng = np.random.default_rng(seeds["minibatches"] + iteration)
                for _ in range(stage.updates):
                    check()
                    tick = time.perf_counter()
                    phase = "collection_seconds"
                    batch = trainer.collector.collect(
                        trainer.agent, stage.transitions, deadline_monotonic=deadline
                    )
                    record.collection_seconds += time.perf_counter() - tick
                    check()
                    phase = "learning_seconds"
                    tick = time.perf_counter()
                    diagnostic = update_iteration(
                        trainer,
                        batch,
                        stage.learning,
                        (time.perf_counter() - start) / regime.wall_seconds,
                        rng,
                    )
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
                continuation = {}
                phase = "learning_seconds"
                tick = time.perf_counter()
                agent, _, _, history = train_search_supervised(
                    dataset,
                    policy_target_kind="visit_distribution",
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
            if isinstance(stage, (TrainSelfPlay, TrainSupervised)):
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
                record.export_seconds = time.perf_counter() - tick
            check()
            record.seconds = time.perf_counter() - stage_start
            record.cpu_seconds = time.process_time() - cpu_start
            record.status = "completed"
            # Freeze the admission cost; later persistence belongs to later outputs.
            record.cumulative_seconds = time.perf_counter() - start
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
            record.cpu_seconds = time.process_time() - cpu_start
            unaccounted = max(
                0.0,
                record.seconds
                - record.collection_seconds
                - record.learning_seconds
                - record.export_seconds,
            )
            setattr(record, phase, getattr(record, phase) + unaccounted)
        run.seconds = time.perf_counter() - start
        store.save_training_run(run)
        try:
            export_training_run(run.id, store, out)
        except OSError:
            pass
        raise
    finally:
        resources.close()
    return run
