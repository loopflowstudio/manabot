"""Admit completed training exports and orchestrate bounded checkpoint evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import resource
import time

import psutil

from manabot.eval.admission import register_anchor, register_checkpoint
from manabot.eval.evaluation import evaluate
from manabot.eval.execution import bind_execution, resolve_execution
from manabot.eval.models import EvaluationKey
from manabot.eval.monitor import MonitorProtocol, MonitorResult, _request, _summarize
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import authored_matchup_sha256, source_bundle_sha256
from manabot.sim import players
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import SELECTED_SUITE, selected_match
from manabot.sim.serving_protocol import InferenceRuntime
from manabot.training.records import TrainingCoordinates, TrainingRun
from manabot.training.regime import ArtifactReference, Strict


class Checkpoint(Strict):
    artifact: ArtifactReference
    coordinates: TrainingCoordinates


def stage_checkpoint(run: TrainingRun, stage_id: str) -> Checkpoint | None:
    """Return an admitted completed or paused export with original coordinates."""
    for index, stage in enumerate(run.stages):
        if stage.id != stage_id:
            continue
        if (
            stage.status not in {"completed", "paused"}
            or "raw" not in stage.artifacts
            or stage.cumulative_seconds is None
        ):
            return None
        prefix = run.stages[: index + 1]
        operation = run.regime.stages[index].operation
        self_play = operation == "train_self_play"
        origins = [
            s.learning_state_origin
            for s in prefix
            if s.learning_state_origin is not None
        ]
        return Checkpoint(
            artifact=stage.artifacts["raw"],
            coordinates=TrainingCoordinates(
                stage_id=stage.id,
                updates=run.updates_through(stage.id),
                segment_updates=stage.updates
                if self_play and stage.optimizer_updates is not None
                else None,
                optimizer_updates=run.optimizer_updates_through(stage.id),
                training_seconds=stage.cumulative_seconds
                + sum(origin.cumulative_seconds for origin in origins),
                # As the stage's last update coordinate measured it: collection
                # and learning time of this attempt's stages plus inherited time.
                active_training_seconds=sum(
                    s.collection_seconds + s.learning_seconds for s in prefix
                )
                + sum(origin.active_training_seconds for origin in origins)
                if stage.updates and operation == "train_self_play"
                else None,
                **{
                    name: sum(getattr(s, name) for s in prefix)
                    + sum(getattr(origin, name) for origin in origins)
                    for name in (
                        "environment_decisions",
                        "learner_transitions",
                        "optimizer_exposures",
                        "games",
                    )
                },
            ),
        )
    return None


def checkpoint_manifest(
    run: TrainingRun,
    artifact: ArtifactReference,
    coordinates: TrainingCoordinates,
    protocol: MonitorProtocol,
    *,
    execution: EvalExecution | None = None,
) -> MonitorResult:
    """Admit exact checkpoint bytes and runtime identities before playing games."""
    path = Path(artifact["path"]).resolve()
    matchup = protocol.authored_matchup
    expected_match = matchup.match() if matchup else selected_match()
    suite = matchup.content_suite if matchup else SELECTED_SUITE
    if authored_matchup_sha256(run.identities, "monitoring") != canonical_sha256(
        expected_match.model_dump()
    ):
        raise ValueError("monitoring requires its exact declared authored setup")
    admission = register_checkpoint(
        path,
        match=run.regime.root_match,
        content_suite=suite,
        player_id="monitor-checkpoint",
        display_name="Monitored checkpoint",
        training_seed=run.seed,
        artifact_id=f"monitor/{artifact['sha256']}",
        expected_sha256=artifact["sha256"],
        expected_bytes=artifact["bytes"],
        allow_deck_repetition=protocol.include_mirrors,
    )
    runtime = admission.identities
    for name in (
        "world",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
        "engine_source_sha256",
        "content_manifest_sha256",
    ):
        if (
            name == "engine_source_sha256"
            and protocol.purpose == "supplemental-matchup-diagnostic"
        ):
            # Artifact loading already admitted the exact rules/content/input binding.
            # Preserve the historical producer source separately from this instrument.
            continue
        if runtime[name] != run.identities.get(name):
            raise ValueError(f"monitoring runtime differs from training: {name}")
    runtime["monitor_source_sha256"] = source_bundle_sha256(
        [
            Path(__file__),
            *sorted(Path(players.__file__).parent.rglob("*.py")),
            *sorted((Path(__file__).resolve().parents[1] / "eval").glob("*.py")),
        ]
    )
    candidate = admission.registration
    opponent = register_anchor(
        candidate,
        protocol.opponent,
        player_id="monitor-" + protocol.opponent.replace("_", "-"),
    )
    key = EvaluationKey(
        world=run.regime.world,
        content_suite=suite,
        viewer_boundary="acting-viewer",
        arena_version="training-monitor-v1",
        rating_model_version="unrated-monitoring",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    if execution is not None:
        key, (candidate, opponent) = bind_execution(
            key, (candidate, opponent), execution
        )
    return MonitorResult(
        purpose=protocol.result_purpose,
        run_id=run.id,
        regime_digest=run.regime_digest,
        training_seed=run.seed,
        artifact={**artifact, "path": str(path)},
        coordinates=coordinates,
        protocol=protocol,
        key=key,
        candidate=candidate,
        opponent=opponent,
        expected_games=protocol.legs * len(protocol.deal_seeds),
        evaluation_identities=runtime,
    )


def evaluate_checkpoint(
    run: TrainingRun,
    artifact: ArtifactReference,
    coordinates: TrainingCoordinates,
    output_dir: Path,
    *,
    protocol: MonitorProtocol | None = None,
    concurrent_activity: str = "unknown; host load is not contention attribution",
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | None = None,
) -> MonitorResult:
    """Run bounded arena games. Use a new directory per attempt; never overwrite failures."""
    protocol = protocol or MonitorProtocol()
    execution = resolve_execution(execution, inference)
    output_dir.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    started_unix = time.time()
    process = psutil.Process()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        result = checkpoint_manifest(
            run,
            artifact,
            coordinates,
            protocol,
            execution=execution,
        )
    except Exception as exc:
        atomic_json(
            output_dir / "admission-failure.json",
            {
                "run_id": run.id,
                "artifact": artifact,
                "coordinates": coordinates.model_dump(mode="json"),
                "error": f"{type(exc).__name__}: {exc}",
                "evaluation_seconds": time.perf_counter() - start,
            },
        )
        raise
    result.concurrent_activity = concurrent_activity
    result.started_unix = started_unix
    result.coordinator_rss_before_bytes = process.memory_info().rss
    result.host_load_before = os.getloadavg()
    atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    try:
        # A served panel covers every deal; scalar play keeps one block per deal.
        panels = (
            [result.protocol.deal_seeds]
            if execution
            else [(seed,) for seed in result.protocol.deal_seeds]
        )
        for block, seeds in enumerate(panels):
            directory = output_dir / f"deal-{block:03d}"
            # The arena persists command traces after every attempted game.
            panel = evaluate(
                _request(result, tuple(seeds)),
                out_dir=directory,
                execution=execution,
                inference=inference,
            )
            for row in panel.rows:
                row.trace_path = str(Path(directory.name) / row.trace_path)
                result.rows.append(row)
            if panel.error:
                result.error = panel.error
                break
            atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.evaluation_seconds = time.perf_counter() - start
        result.finished_unix = time.time()
        result.coordinator_cpu_seconds = time.process_time() - cpu
        child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.reaped_worker_cpu_seconds = (
            child_end.ru_utime
            + child_end.ru_stime
            - child_start.ru_utime
            - child_start.ru_stime
        )
        result.coordinator_rss_after_bytes = process.memory_info().rss
        result.host_load_after = os.getloadavg()
        try:
            _summarize(result)
        except ValueError as error:
            result.error = f"{type(error).__name__}: {error}"
            result.status = "incomplete"
            result.win = result.draw = result.score = None
        atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    return result
