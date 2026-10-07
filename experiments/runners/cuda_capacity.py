"""Frozen ETU-103 CUDA capacity cohort over the existing rental and arena owners.

Calibration chooses work counts before scoring. CapacityPlan binds Experiment
recipes, EvaluationProtocol, source and timing receipts. The controller retrieves
committed exports during training and gives the ordinary checkpoint queue one
CPU evaluator; each rental still owns its deadlines, verification and deletion.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Literal

from pydantic import Field, model_validator

from experiments.runners.cuda_live_calibration import LiveCalibrationReceipt
from experiments.runners.model_capacity import experiment
from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.infra.artifacts import S3ArtifactStore
from manabot.remote.bundle import Bundle
from manabot.remote.deploy import Receipt, current_source, deploy
from manabot.remote.plan import DeploymentPlan, HardwareMix, Source, compile_plan
from manabot.remote.progress import LiveExports
from manabot.remote.provider import RunPod
from manabot.remote.transport import Transport
from manabot.training.artifacts import publish_run
from manabot.training.checkpoint_queue import CheckpointQueue, MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.experiments import Baseline, Case, Experiment, Model, Pipeline
from manabot.training.models import (
    ArtifactReference,
    Strict,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.training.monitor_evaluation import Checkpoint, MonitorProtocol
from manabot.training.monitoring import publish_dashboard, training_dashboard


class PublicationAttempt(Strict):
    status: Literal["completed", "failed", "timeout"]
    seconds: float
    directory: str
    accounting: str = "report reserve"


class Timing(Strict):
    capacity: str
    path: str
    sha256: str
    seconds_per_update: float = Field(gt=0)
    updates: int = Field(ge=20)


class CapacityPlan(Strict):
    source: Source
    calibration_root: str
    calibration_receipts: dict[str, str]
    timings: list[Timing]
    live_workflow_sha256: str
    experiment_receipt_sha256: str
    protocol: EvaluationProtocol
    deployments: list[DeploymentPlan]
    created_unix: float
    total_seconds: Literal[41700] = 41700
    evaluator_seconds: Literal[28800] = 28800
    cohort_seconds: Literal[1200] = 1200
    report_seconds: Literal[1800] = 1800
    dollar_ceiling: Literal[12] = 12
    storage_reserve_dollars: Literal[1] = 1
    checkpoint_seconds: Literal[3600] = 3600
    endpoint: str = (
        "frozen calibrated updates, two equal linked halves; no convergence claim"
    )
    pause: str = "between complete seed/arm rentals only; failures retain prefixes, delete rental and stop without process recovery or retry"

    @model_validator(mode="after")
    def admitted(self) -> "CapacityPlan":
        expected = [
            (10351, "cuda-w64-d2"),
            (10351, "cuda-w384-d8"),
            (10352, "cuda-w384-d8"),
            (10352, "cuda-w64-d2"),
            (10353, "cuda-w64-d2"),
            (10353, "cuda-w384-d8"),
        ]
        if [(d.seed, d.regime.id) for d in self.deployments] != expected:
            raise ValueError(
                "capacity cohort must retain the six paired runs and alternating order"
            )
        digests = {
            canonical_sha256(d.regime.model_dump(mode="json")) for d in self.deployments
        }
        if digests != set(self.protocol.regime_digests):
            raise ValueError("evaluation protocol does not bind the deployed recipes")
        if any(
            d.source != self.source or d.mix != self.deployments[0].mix
            for d in self.deployments
        ):
            raise ValueError(
                "all scientific arms require one frozen source and hardware mix"
            )
        controls = []
        for deployment in self.deployments:
            payload = deployment.regime.model_dump(mode="json")
            payload.pop("id")
            for field in (
                "hidden_dim",
                "attention_layers",
                "attention_feedforward_dim",
            ):
                payload["agent"].pop(field, None)
            for stage in payload["stages"]:
                stage.pop("updates")
            controls.append(payload)
        if any(c != controls[0] for c in controls):
            raise ValueError("non-capacity scientific controls differ")
        if (
            sum(d.mix.wall_seconds for d in self.deployments)
            + self.evaluator_seconds
            + self.report_seconds
            > self.total_seconds
        ):
            raise ValueError("full cohort exceeds total time allocation")
        return self


def _receipts(root: Path) -> list[Receipt]:
    return [
        Receipt.model_validate_json(p.read_text())
        for p in root.glob("calibration-*/deployment.json")
    ]


def freeze(root: Path, out: Path, gpu: str, streams: int, batch: int) -> CapacityPlan:
    """Freeze timing-only counts; the caller selects a common measured workload."""
    if out.exists():
        raise ValueError("freeze output already exists")
    root, out = root.resolve(), out.resolve()
    receipts = _receipts(root)
    if len(receipts) < 4 or any(
        r.phase != "deleted" or r.estimated_dollars is None for r in receipts
    ):
        raise ValueError(
            "all four calibration rentals need confirmed deletion and costs"
        )
    for receipt_path in root.glob("calibration-*/deployment.json"):
        receipt = Receipt.model_validate_json(receipt_path.read_text())
        if not receipt.complete:
            recovered = receipt_path.parent / "bulk-return"
            if not (recovered / "receipt.json").exists():
                raise ValueError(
                    "incomplete rental requires a separately verified closed bundle"
                )
            Bundle.model_validate_json((recovered / "bundle.json").read_text()).verify(
                recovered / "evidence"
            )
    amendment = json.loads((root / "calibration-amendment.json").read_text())
    if (
        amendment["calibration_seconds"] != 8700
        or amendment["comparison_seconds"] != 41700
        or amendment["combined_seconds"] != 50400
    ):
        raise ValueError("calibration reallocation differs from recorded plan")
    if time.time() - min(r.started for r in receipts) > 8700:
        raise ValueError("calibration clock exhausted before freeze")
    if streams not in (4, 16, 64) or batch not in (128, 512) or batch % streams:
        raise ValueError("selected workload must be a declared measured sweep cell")
    live_path = root / "live-workflow-control/receipt.json"
    live_proof = LiveCalibrationReceipt.model_validate_json(live_path.read_text())
    if not any(
        o.run_status == "running" and o.attempts == 1 for o in live_proof.observations
    ):
        raise ValueError("live evaluation was not demonstrated during training")
    suffix = "l4" if gpu == "NVIDIA L4" else "a40"
    base = TrainingRegime.model_validate_json(
        (root / "inputs/cuda-calibration-small.json").read_text()
    )
    resolved = experiment(base, include_ataraxos=True).resolve()
    chosen: list[TrainingRegime] = []
    timings: list[Timing] = []
    for capacity in ("w64-d2", "w384-d8"):
        rental = root / f"calibration-{suffix}-sweep"
        evidence = (
            rental / "bulk-return/evidence"
            if (rental / "bulk-return/receipt.json").exists()
            else rental / "evidence"
        )
        path = (
            evidence / f"performance/{capacity}-loop-b{batch}-s{streams}/run/run.json"
        )
        run = TrainingRun.model_validate_json(path.read_text())
        if run.status != "completed" or run.updates_through() != 3:
            raise ValueError("selected complete-loop calibration did not complete")
        stage = run.stages[0]
        points = [
            float(d["coordinates"]["training_seconds"]) for d in stage.diagnostics
        ]
        rate = max(
            run.seconds / 3,
            stage.seconds / 3,
            points[1] - points[0],
            points[2] - points[1],
        )
        # 900 scheduled seconds with a 50% timing margin. Counts stay fixed even
        # if later runs are faster, slower or have different filtering exposure.
        updates = 20 * int(900 / (1.5 * rate * 20))
        timings.append(
            Timing(
                capacity=capacity,
                path=str(path),
                sha256=file_sha256(path),
                seconds_per_update=rate,
                updates=updates,
            )
        )
        recipe = resolved.regimes[capacity].model_copy(deep=True)
        recipe.id = f"cuda-{capacity}"
        recipe.wall_seconds = 1000
        for stage in recipe.stages:
            assert isinstance(stage, TrainSelfPlay)
            stage.streams, stage.transitions, stage.updates = (
                streams,
                batch // streams,
                updates // 2,
            )
            stage.execution.wall_seconds = 490
            stage.execution.device, stage.execution.threads = "cuda", 1
        chosen.append(TrainingRegime.model_validate(recipe.model_dump()))
    base.wall_seconds = 1000
    resolved = Experiment(
        name="cuda",
        baseline=Baseline.capture("cuda-capacity-control-v1", base),
        cases=tuple(
            Case(t.capacity, (Model(r.agent), Pipeline(tuple(r.stages))))
            for t, r in zip(timings, chosen, strict=True)
        ),
    ).resolve()
    chosen = list(resolved.regimes.values())
    source = current_source(Path.cwd())
    template = HardwareMix.model_validate_json((root / "inputs/mix.json").read_text())
    mix = HardwareMix.model_validate(
        template.model_dump()
        | dict(
            gpu_types=[gpu],
            wall_seconds=1850,
            setup_seconds=330,
            transfer_seconds=360,
            cleanup_seconds=120,
        )
    )
    seeds = (10351, 10352, 10353)
    deployments = [
        compile_plan(recipe.model_dump_json(), mix, source, seed)
        for i, seed in enumerate(seeds)
        for recipe in (chosen if i % 2 == 0 else chosen[::-1])
    ]
    protocol = EvaluationProtocol(
        study="cuda-capacity",
        purpose="scientific",
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in chosen
        ),
        training_seeds=seeds,
        paired_deals=(),
        anchor_deals=tuple(range(1_910_103_510, 1_910_103_535)),
        endpoint_anchor_deals=tuple(range(1_910_103_610, 1_910_103_635)),
        random_diagnostic_deals=tuple(range(1_910_103_710, 1_910_103_735)),
        anchors=("scripted-greedy", "random"),
        checkpoint_count=3,
        cost_cutoffs_seconds=(300, 600),
        early_progress_seconds=600,
        progress_score=0.5,
        process_seconds=41700,
        uncertainty="paired-seed-descriptive",
        game_seconds=60,
    )
    plan = CapacityPlan(
        source=source,
        calibration_root=str(root),
        calibration_receipts={
            str(p): file_sha256(p) for p in root.glob("calibration-*/deployment.json")
        },
        timings=timings,
        experiment_receipt_sha256=resolved.identity,
        live_workflow_sha256=file_sha256(live_path),
        protocol=protocol,
        deployments=deployments,
        created_unix=time.time(),
    )
    if (
        sum(p.mix.wall_seconds for p in deployments)
        + plan.evaluator_seconds
        + plan.report_seconds
        > plan.total_seconds
    ):
        raise ValueError("complete cohort does not fit the time allocation")
    if (
        sum(r.estimated_dollars or 0 for r in receipts)
        + sum(p.projected_dollars for p in deployments)
        + 1
        > plan.dollar_ceiling
    ):
        raise ValueError("complete cohort does not fit aggregate dollar ceiling")
    out.mkdir(parents=True)
    atomic_json(out / "authoring.json", resolved.receipt())
    atomic_json(out / "plan.json", plan.model_dump(mode="json"))
    return plan


def protocols(
    plan: CapacityPlan, run: TrainingRun, checkpoint: Checkpoint
) -> list[MonitorProtocol]:
    p = plan.protocol
    common = dict(
        game_seconds=p.game_seconds,
        max_commands=p.max_commands,
        bootstrap_seed=10351,
        bootstrap_replicates=10000,
    )
    if (
        checkpoint.coordinates.updates == 0
        or checkpoint.coordinates.stage_id == "policy-0"
    ):
        return [MonitorProtocol(deal_seeds=p.anchor_deals, **common)]
    if (
        checkpoint.coordinates.stage_id != "policy-1"
        or run.stages[-1].status != "completed"
    ):
        return []
    return [
        MonitorProtocol(
            deal_seeds=p.endpoint_anchor_deals,
            purpose="frozen-study-evaluation",
            study_protocol_sha256=canonical_sha256(p.model_dump(mode="json")),
            **common,
        ),
        MonitorProtocol(
            deal_seeds=p.random_diagnostic_deals, opponent="random", **common
        ),
    ]


def publish(directory: Path, destination: Path) -> None:
    """Called in a bounded child after rental deletion; originals stay immutable."""
    run_path = directory / "evidence/run/run.json"
    run = TrainingRun.model_validate_json(run_path.read_text())
    bundle = Bundle.model_validate_json((directory / "bundle.json").read_text())
    receipt: dict[str, str | None] = {"run_id": run.id}
    try:
        manifest = publish_run(
            run_path,
            "s3://etudefantasia/manabot/",
            destination / "artifacts.json",
            S3ArtifactStore(),
            resolve_artifact=lambda a: bundle.resolve(directory / "evidence", a.path),
        )
        receipt["s3"] = "verified"
        dashboard = training_dashboard(run)
        dashboard.summary["artifact_storage"] = manifest.model_dump(mode="json")
        receipt["wandb"] = publish_dashboard(
            dashboard,
            destination,
            project="etude",
            entity="loopflow-studio",
        )
    except Exception as exc:
        receipt["error"] = type(exc).__name__
    atomic_json(destination / "publication.json", receipt)


def _bounded_command(
    command: list[str], out: Path, seconds: float
) -> PublicationAttempt:
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    status: Literal["completed", "failed", "timeout"] = "completed"
    with (out / "worker.log").open("ab") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=seconds)
            if code:
                status = "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return PublicationAttempt(
        status=status, seconds=time.monotonic() - began, directory=str(out)
    )


def execute(out: Path) -> None:
    """No resume/retry: every attempt remains immutable and stops on actual failure."""
    out = out.resolve()
    plan = CapacityPlan.model_validate_json((out / "plan.json").read_text())
    if (out / "campaign.json").exists():
        raise ValueError("campaign already attempted; automatic recovery unsupported")
    if current_source(Path.cwd()) != plan.source:
        raise ValueError("scientific source changed after freeze")
    if (
        file_sha256(Path(plan.calibration_root) / "live-workflow-control/receipt.json")
        != plan.live_workflow_sha256
    ):
        raise ValueError("live-workflow admission changed")
    for path, sha in plan.calibration_receipts.items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("calibration receipt changed")
    for timing in plan.timings:
        if file_sha256(Path(timing.path)) != timing.sha256:
            raise ValueError("calibration timing changed")
    started = time.time()
    deadline = started + plan.total_seconds
    references: dict[str, ArtifactReference] = {}
    sources: dict[str, Path] = {}
    rental_seconds = 0.0
    initial_cost = sum(
        r.estimated_dollars or 0 for r in _receipts(Path(plan.calibration_root))
    )
    dollars = initial_cost
    status = "running"
    error: str | None = None
    publications: list[PublicationAttempt] = []
    publication_seconds = 0.0
    queue: CheckpointQueue | None = None
    last_reported = 0

    def save() -> None:
        atomic_json(
            out / "campaign.json",
            dict(
                status=status,
                error=error,
                started_unix=started,
                deadline_unix=deadline,
                elapsed_seconds=time.time() - started,
                rental_seconds=rental_seconds,
                evaluator_seconds=queue.charged_seconds if queue else 0,
                estimated_dollars=dollars,
                calibration_estimated_dollars=initial_cost,
                storage_reserve_dollars=1,
                publications=[p.model_dump(mode="json") for p in publications],
                nonoverlapping_publication_seconds=publication_seconds,
                plan_sha256=file_sha256(out / "plan.json"),
            ),
        )

    def refresh_report(*, overlapping: bool) -> None:
        nonlocal last_reported, publication_seconds
        if queue is None:
            return
        finished = sum(a.status != "running" for a in queue.attempts)
        if finished == last_reported:
            return
        last_reported = finished
        delivery = _bounded_command(
            [
                sys.executable,
                "-m",
                "experiments.runners.cuda_capacity_report",
                "--root",
                plan.calibration_root,
                "--science",
                str(out),
            ],
            out / "reporting",
            30,
        )
        delivery.accounting = "subset of rental" if overlapping else "report reserve"
        if not overlapping:
            publication_seconds += delivery.seconds
        publications.append(delivery)

    def check() -> None:
        if time.time() >= deadline - plan.report_seconds:
            raise TimeoutError(
                "comparison work deadline reached; report reserve retained"
            )
        if queue is not None and any(
            a.status in ("failed", "interrupted") for a in queue.attempts
        ):
            raise ValueError("evaluation failed; retained attempt stops cohort")

    try:
        # Acquire before renting. An ETU-118 owner makes admission fail immediately;
        # never displace its learner or hold a paid GPU waiting for evaluator access.
        queue = CheckpointQueue(
            out / "evaluation",
            MonitoringBudget(
                seconds=plan.evaluator_seconds,
                attempt_seconds=plan.cohort_seconds,
                include_initial=True,
            ),
            resolve_artifact=lambda a: references[a["sha256"]],
            protocols_for=lambda r, c: protocols(plan, r, c),
        )
        save()
        for index, deployment in enumerate(plan.deployments):
            check()
            remaining_rentals = sum(
                d.mix.wall_seconds for d in plan.deployments[index:]
            )
            remaining_evaluation = plan.evaluator_seconds - queue.charged_seconds
            if (
                rental_seconds
                + queue.charged_seconds
                + remaining_rentals
                + remaining_evaluation
                + plan.report_seconds
                > plan.total_seconds
            ):
                raise ValueError("remaining full cohort cannot fit")
            quote = RunPod().prices().get(deployment.mix.gpu_types[0])
            atomic_json(
                out / f"quote-{index}.json",
                dict(
                    at_unix=time.time(),
                    gpu=deployment.mix.gpu_types[0],
                    quote=quote,
                    prior_estimate=dollars,
                    remaining_projection=sum(
                        d.projected_dollars for d in plan.deployments[index:]
                    ),
                ),
            )
            if (
                dollars + sum(d.projected_dollars for d in plan.deployments[index:]) + 1
                > plan.dollar_ceiling
            ):
                raise ValueError("aggregate rental ceiling would be exceeded")
            live = LiveExports(out / f"live-{index}")
            last_publish = 0.0

            def observe(transport: Transport) -> None:
                nonlocal last_publish
                check()
                source = live.retrieve(transport)
                references.update(live.references)
                if source is not None:
                    run = TrainingRun.model_validate_json(source.read_text())
                    sources[run.id] = source
                queue.tick(list(sources.values()))
                refresh_report(overlapping=True)
                if source is not None and time.time() - last_publish >= 180:
                    last_publish = time.time()
                    publication = _bounded_command(
                        [
                            sys.executable,
                            "-m",
                            "manabot.training.monitoring",
                            "--run",
                            str(source),
                            "--out",
                            str(out / f"wandb-live-{index}"),
                            "--online",
                            "--project",
                            "etude",
                            "--entity",
                            "loopflow-studio",
                        ],
                        out / f"wandb-live-{index}",
                        35,
                    )
                    publication.accounting = "subset of rental time; not added twice"
                    publications.append(publication)
                save()

            directory = out / f"rental-{index}"
            began = time.time()
            try:
                receipt = deploy(
                    deployment,
                    directory,
                    Path.cwd(),
                    observe=observe,
                    checkpoint_seconds=plan.checkpoint_seconds,
                    bulk_return=True,
                )
            finally:
                rental_seconds += time.time() - began
                receipt_path = directory / "deployment.json"
                if receipt_path.exists():
                    retained = Receipt.model_validate_json(receipt_path.read_text())
                    if retained.estimated_dollars is not None:
                        dollars += retained.estimated_dollars
                save()
            if not receipt.complete or receipt.phase != "deleted":
                raise ValueError("rental completion/deletion not admitted")
            run_path = directory / "evidence/run/run.json"
            run = TrainingRun.model_validate_json(run_path.read_text())
            bundle = Bundle.model_validate_json((directory / "bundle.json").read_text())
            sources[run.id] = run_path
            for stage in run.stages:
                for name, a in stage.artifacts.items():
                    if name in ("raw", "initial_raw"):
                        references[a["sha256"]] = {
                            **a,
                            "path": str(
                                bundle.resolve(directory / "evidence", a["path"])
                            ),
                        }
            queue.tick(list(sources.values()))
            publication = _bounded_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.runners.cuda_capacity",
                    "--out",
                    str(out / f"publication-{index}"),
                    "--publish",
                    str(directory),
                ],
                out / f"publication-{index}",
                120,
            )
            publication_seconds += publication.seconds
            publications.append(publication)
            if publication_seconds > plan.report_seconds:
                raise TimeoutError("publication consumed report reserve")
            save()
        while True:
            check()
            queue.tick(list(sources.values()))
            refresh_report(overlapping=False)
            save()
            if queue.process is None and queue.pending == 0:
                break
            if (
                queue.process is None
                and queue.pending
                and plan.evaluator_seconds - queue.charged_seconds < plan.cohort_seconds
            ):
                raise TimeoutError(
                    "remaining evaluator allowance cannot admit the next complete cohort"
                )
            time.sleep(5)
        if len(queue.attempts) != 24 or any(
            a.status != "completed" for a in queue.attempts
        ):
            raise ValueError("frozen 24-cohort schedule incomplete")
        status = "completed"
    except BaseException as exc:
        status, error = "failed", type(exc).__name__
        raise
    finally:
        if queue is not None:
            queue.close()
        save()
        remaining_report = max(1, plan.report_seconds - publication_seconds)
        delivery = _bounded_command(
            [
                sys.executable,
                "-m",
                "experiments.runners.cuda_capacity_report",
                "--root",
                plan.calibration_root,
                "--science",
                str(out),
                "--notebook",
            ],
            out / "reporting",
            min(120, remaining_report),
        )
        publication_seconds += delivery.seconds
        publications.append(delivery)
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--gpu", choices=("NVIDIA L4", "NVIDIA A40"))
    parser.add_argument("--streams", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.publish:
        publish(args.publish, args.out)
    elif args.freeze:
        if None in (args.calibration_root, args.gpu, args.streams, args.batch):
            parser.error("freeze requires calibration root, GPU, streams and batch")
        freeze(args.calibration_root, args.out, args.gpu, args.streams, args.batch)
    else:
        parser.error(
            "scientific launch unavailable until ETU-123 disconnect-safe lifecycle "
            "is integrated; $3 of the shared $15 is reserved for its proof"
        )


if __name__ == "__main__":
    main()
