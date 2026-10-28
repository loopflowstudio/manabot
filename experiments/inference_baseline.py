"""Freeze and execute bounded local inference recipes without provisioning.

`freeze` resolves source, native, runtime and input bytes before measurement;
`validate` only checks those pins. `run` executes games and `replay` executes
retained inputs. Both reserve every attempt first, retain failures, and emit the
existing report manifest for game receipts. Input witnesses remain diagnostics,
never fresh game throughput. No command downloads missing models.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from manabot._process import bounded_call
from manabot.reports.inference import (
    BaselineAttempt,
    BaselineManifest,
    EvidenceFile,
    inventory,
    pin_file,
)

if TYPE_CHECKING:
    import numpy as np


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PolicyWorkload(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    kind: Literal["policy"]
    request: Path
    legs: tuple[int, ...] = Field(min_length=1)
    trace: Path | None = None
    roots: int = Field(default=64, ge=1, le=64)


class LLMWorkload(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    kind: Literal["llm"]
    config: Path
    cache_manifest: Path
    games: tuple[Path, ...] = Field(min_length=1, max_length=5)
    # One sequential resident session owns this entire bounded game set.
    corpus: tuple[Path, ...] = Field(default=(), max_length=160)
    load_seconds: float = Field(default=180, gt=0, le=180)


Workload = Annotated[PolicyWorkload | LLMWorkload, Field(discriminator="kind")]


class Cell(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    workload: str
    # Existing EvalAllocation JSON; None selects ordinary scalar CPU policy play.
    execution: Path | None = None
    instrumentation: bool = True
    mode: Literal["ordinary", "diagnostic"] = "ordinary"


class Recipe(Record):
    workloads: tuple[Workload, ...] = Field(min_length=1, max_length=32)
    cells: tuple[Cell, ...] = Field(min_length=1, max_length=64)
    repetitions: int = Field(default=3, ge=1, le=3)
    warmup_passes: Literal[2] = 2
    cell_seconds: float = Field(default=300, gt=2, le=1800)
    total_seconds: float = Field(default=1800, gt=2, le=1800)
    process_tree_bytes: int = Field(default=8 * 2**30, gt=0, le=8 * 2**30)
    sample_device: bool = False
    absolute_tolerance: float = Field(default=1e-5, ge=0)
    relative_tolerance: float = Field(default=1e-4, ge=0)

    @model_validator(mode="after")
    def references(self) -> Recipe:
        if any(w.kind == "policy" for w in self.workloads) and self.cell_seconds > 300:
            raise ValueError("policy cell ceiling is 300 seconds including cleanup")
        names = {w.id for w in self.workloads}
        if len(names) != len(self.workloads) or len({c.id for c in self.cells}) != len(
            self.cells
        ):
            raise ValueError("workload and cell identities must be unique")
        if any(c.workload not in names for c in self.cells):
            raise ValueError("cell references an unknown workload")
        return self


class Protocol(Record):
    version: Literal["inference-execution-v1"] = "inference-execution-v1"
    recipe: Recipe
    source: tuple[EvidenceFile, ...]
    inputs: tuple[EvidenceFile, ...]
    runtime: dict[str, str]
    cache_policy: Literal["offline-existing-no-eviction"] = (
        "offline-existing-no-eviction"
    )
    os_page_cache: Literal["unknown"] = "unknown"
    threads: Literal[1] = 1
    tf32: Literal[False] = False


def _write(
    path: Path, value: BaseModel | dict[str, JsonValue] | list[JsonValue]
) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _runtime() -> dict[str, str]:
    facts = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in ("torch", "numpy", "transformers", "huggingface-hub"):
        try:
            facts[package] = version(package)
        except PackageNotFoundError:
            facts[package] = "unavailable"
    facts["cpu"] = platform.processor() or "unavailable"
    facts["logical_cpus"] = str(os.cpu_count())
    try:
        facts["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        facts["gpu"] = "unavailable"
    return facts


def freeze(recipe: Recipe) -> Protocol:
    """Resolve local inputs only; missing checkpoints/cache snapshots fail closed."""
    paths: set[Path] = set()
    for cell in recipe.cells:
        if cell.execution:
            paths.add(cell.execution)
    for workload in recipe.workloads:
        if isinstance(workload, PolicyWorkload):
            paths.add(workload.request)
            # Admission remains with the policy owner; freeze binds its local bytes.
            from manabot.eval.evaluation import EvaluationRequest

            request = EvaluationRequest.model_validate_json(
                workload.request.read_bytes()
            )
            paths.update(Path(p) for p in request.checkpoint_paths.values())
            if workload.trace:
                paths.add(workload.trace)
        else:
            from huggingface_hub import snapshot_download

            from manabot.llm.schema import InferenceConfig

            config = InferenceConfig.model_validate_json(workload.config.read_bytes())
            paths.update(
                (
                    workload.config,
                    *workload.games,
                    workload.cache_manifest,
                    *workload.corpus,
                )
            )
            for model, revision in (
                (config.model_id, config.model_revision),
                (config.tokenizer_id, config.tokenizer_revision),
            ):
                snapshot = Path(
                    snapshot_download(model, revision=revision, local_files_only=True)
                )
                paths.update(p for p in snapshot.rglob("*") if p.is_file())
    source_paths = [
        p
        for directory in (
            "manabot",
            "managym/src",
            "experiments/inference_baselines",
            "content/semantic",
        )
        for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix in {".py", ".rs", ".so", ".json"}
    ]
    source_paths.extend(
        Path(p)
        for p in (
            "experiments/inference_baseline.py",
            "experiments/policy_inference_gpu.py",
            "uv.lock",
            "pyproject.toml",
            "managym/Cargo.lock",
        )
    )
    return Protocol(
        recipe=recipe,
        source=tuple(pin_file(p) for p in sorted(set(source_paths))),
        inputs=tuple(pin_file(p) for p in sorted(paths)),
        runtime=_runtime(),
    )


def validate(protocol: Protocol) -> None:
    if not protocol.source or not protocol.inputs:
        raise ValueError("resolved source and input inventories are required")
    current = freeze(protocol.recipe)
    if current.runtime != protocol.runtime:
        raise ValueError("runtime differs from frozen protocol; freeze a new treatment")
    if current.source != protocol.source or current.inputs != protocol.inputs:
        raise ValueError("frozen digest/size/inventory mismatch")


class _DeviceSampler:
    """Optional bounded 1 Hz samples, including UUID; no mandatory GPU utility."""

    def __init__(self, out: Path, enabled: bool) -> None:
        self.out = out
        self.enabled = enabled
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.error: str | None = None
        self.samples = 0

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id="
                        + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                        "--query-gpu=timestamp,uuid,name,memory.total,memory.used,driver_version",
                        "--format=csv",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=1,
                )
                lines = result.stdout.splitlines()
                with (self.out / "gpu-memory.csv").open("a") as stream:
                    stream.write(
                        "\n".join(lines if self.samples == 0 else lines[1:]) + "\n"
                    )
                self.samples += max(0, len(lines) - 1)
            except (OSError, subprocess.SubprocessError) as error:
                self.error = f"{type(error).__name__}: {error}"
                break
            self.stop.wait(1)

    def __enter__(self) -> _DeviceSampler:
        self.started = time.monotonic()
        if self.enabled:
            self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        if self.enabled:
            self.thread.join(timeout=2)
        _write(
            self.out / "device-sampling.json",
            {
                "samples": self.samples,
                "interval_seconds": 1,
                "window_seconds": time.monotonic() - self.started,
                "status": "recorded" if self.samples else "unavailable",
                "reason": self.error or (None if self.enabled else "not requested"),
            },
        )


def _worker(
    sender: Connection,
    protocol: Protocol,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> None:
    # Set before runtime imports or spawned workers. There is no network fallback.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    workload = next(w for w in protocol.recipe.workloads if w.id == cell.workload)
    with _DeviceSampler(out, protocol.recipe.sample_device):
        try:
            verification_start = time.monotonic()
            validate(protocol)
            _write(
                out / "protocol-verification.json",
                {
                    "seconds": time.monotonic() - verification_start,
                    "status": "verified",
                },
            )
            if isinstance(workload, PolicyWorkload):
                from experiments.inference_baselines.policy import execute
            else:
                from experiments.inference_baselines.llm import execute
            result = execute(protocol, workload, cell, out, replay, deadline)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    _write(out / "result.json", result)
    sender.send(result)


def _collect(
    sender: Connection,
    protocol: Protocol,
    directory: Path,
    references: tuple[Path, ...],
) -> None:
    """Pin durable worker output before comparison, streaming partial evidence.

    The execution process tree has stopped before this child starts. Each pin
    is small; arrays and hashing stay here under the original deadline/RSS cap.
    An interrupted comparison cannot discard already received raw witnesses.
    """

    def retain(file: EvidenceFile) -> None:
        sender.send({"progress": file.model_dump(mode="json")})

    complete = False
    result: dict[str, JsonValue] = {"status": "completed"}
    try:
        inventory(directory, on_file=retain)
        try:
            if references:
                _compare_outputs(references, directory, protocol)
        finally:
            comparison = directory / "comparison.json"
            if comparison.exists():
                retain(pin_file(comparison, relative_to=directory))
            complete = True
    except Exception as error:
        result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    result["inventory_complete"] = complete
    sender.send(result)


def run(
    protocol: Protocol,
    out: Path,
    *,
    replay: bool = False,
    worker: Callable[..., None] = _worker,
    collector: Callable[..., None] = _collect,
) -> BaselineManifest:
    """Execute once; tests may inject a controlled worker, never a load generator."""
    invocation_start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "protocol.json", protocol)
    protocol_file = pin_file(out / "protocol.json")
    protocol_digest = protocol_file.sha256
    scheduled: list[tuple[Cell, BaselineAttempt]] = []
    for repeat in range(protocol.recipe.repetitions):
        cells = protocol.recipe.cells
        shift = repeat % len(cells)
        for cell in (*cells[shift:], *cells[:shift]):
            workload = next(
                w for w in protocol.recipe.workloads if w.id == cell.workload
            )
            scheduled.append(
                (
                    cell,
                    BaselineAttempt(
                        id=f"{repeat}-{cell.id}",
                        kind="retained-input"
                        if replay
                        else "evaluation"
                        if workload.kind == "policy"
                        else "llm",
                        cohort=workload.id,
                        repeat=repeat,
                        mode="diagnostic" if replay else cell.mode,
                        status="planned",
                        reason="reserved before worker startup",
                    ),
                )
            )
    manifest = BaselineManifest(
        version="inference-baseline-v1",
        source=protocol_digest,
        protocol=protocol_file,
        attempts=[a for _, a in scheduled],
    )
    _write(out / "manifest.json", manifest)
    total_deadline = time.time() + max(
        0, protocol.recipe.total_seconds - (time.monotonic() - invocation_start)
    )
    stopped_reason = None
    for index, (cell, attempt) in enumerate(scheduled):
        if stopped_reason is not None or time.time() >= total_deadline - 2:
            manifest.attempts[index] = attempt.model_copy(
                update={
                    "status": "unattempted",
                    "reason": stopped_reason or "invocation deadline exhausted",
                }
            )
            _write(out / "manifest.json", manifest)
            continue
        directory = out / attempt.id
        directory.mkdir()
        started = time.monotonic()
        deadline = min(total_deadline, time.time() + protocol.recipe.cell_seconds)
        manifest.attempts[index] = attempt.model_copy(
            update={"status": "attempted", "reason": "worker starting"}
        )
        _write(out / "manifest.json", manifest)
        outcome = bounded_call(
            worker,
            (protocol, cell, directory, replay, deadline - 2),
            deadline_unix=deadline,
            process_tree_bytes=protocol.recipe.process_tree_bytes,
        )
        result = outcome.payload or {}
        reason = outcome.error or str(result.get("reason") or "finished")
        status = (
            "completed"
            if result.get("status") == "completed"
            and not outcome.error
            and outcome.cleanup_complete
            else "failed"
        )
        if not outcome.cleanup_complete:
            reason = "child cleanup incomplete; " + reason
        collection_start = time.monotonic()
        references: tuple[Path, ...] = ()
        if replay and status == "completed" and (directory / "inputs.npz").exists():
            references = tuple(
                out / a.id
                for a in manifest.attempts[:index]
                if a.cohort == attempt.cohort and a.status == "completed"
            )
        files: list[EvidenceFile] = []

        def retain(progress: dict[str, JsonValue]) -> None:
            files.append(EvidenceFile.model_validate(progress))

        # Reuse the absolute deadline: collection gets only what execution and
        # cleanup left, never a fresh cell budget. Only the small final receipt
        # and ledger writes remain in this parent after supervision.
        collection = (
            bounded_call(
                collector,
                (protocol, directory, references),
                deadline_unix=deadline,
                process_tree_bytes=protocol.recipe.process_tree_bytes,
                on_progress=retain,
            )
            if outcome.cleanup_complete
            else None
        )
        collected = (collection.payload or {}) if collection is not None else {}
        inventory_complete = collected.get("inventory_complete") is True
        if collection is None:
            reason += "; collection not started"
        elif (
            collection.error
            or collected.get("status") != "completed"
            or not collection.cleanup_complete
            or not inventory_complete
        ):
            status = "failed"
            reason += "; collection: " + (
                collection.error or str(collected.get("reason") or "incomplete")
            )
            if not collection.cleanup_complete:
                reason += "; collection child cleanup incomplete"
        if not outcome.cleanup_complete or (
            collection is not None and not collection.cleanup_complete
        ):
            stopped_reason = "child cleanup incomplete; invocation stopped"
        _write(
            directory / "attempt.json",
            {
                "version": "inference-attempt-v1",
                "id": attempt.id,
                "protocol_sha256": protocol_digest,
                "status": status,
                "reason": reason,
                "whole_attempt_seconds": time.monotonic() - started,
                "collection_seconds": time.monotonic() - collection_start,
                "boundary": "parent dispatch through worker preparation, play/replay, final collection and child cleanup; excludes final ledger write",
                "supervision": {
                    **asdict(outcome),
                    "collection": asdict(collection)
                    if collection is not None
                    else None,
                },
            },
        )
        files.append(pin_file(directory / "attempt.json", relative_to=directory))
        files.sort(key=lambda file: file.path)
        # Retained-input execution is a diagnostic corpus, not a game receipt.
        # Keep its evidence in the ledger without asking the game reader to invent rows.
        root = str(directory.resolve())
        manifest.attempts[index] = attempt.model_copy(
            update={
                "status": status,
                "reason": reason,
                "root": root,
                "files": files,
                "inventory_complete": inventory_complete,
                "corpus": sorted(
                    f.path for f in files if f.path.endswith("/selection.json")
                )
                if attempt.kind == "llm"
                else [],
            }
        )
        _write(out / "manifest.json", manifest)
    _write(
        out / "invocation.json",
        {
            "seconds": time.monotonic() - invocation_start,
            "attempts": len(scheduled),
            "replay": replay,
            "inclusive_billing": None,
            "billing_reason": "no provider settlement supplied",
        },
    )
    return manifest


class _OutputWitness(TypedDict):
    logits: np.ndarray
    masks: np.ndarray
    actions: np.ndarray


def _compare_outputs(
    references: tuple[Path, ...], directory: Path, protocol: Protocol
) -> None:
    """Compare identical retained arrays; changed trajectories never enter here."""
    import numpy as np

    from manabot.sim.execution import EvalExecution

    def settings(root: Path) -> EvalExecution:
        return EvalExecution.model_validate(
            json.loads((root / "preparation.json").read_bytes())["execution"]
        )

    current_settings = settings(directory)
    compute = current_settings.model_dump(exclude={"deadline_unix", "instrumentation"})
    reference = references[0]
    instrumentation_pair = False
    # Cell ordering rotates between repeats. Prefer an otherwise identical
    # treatment so a scalar-first recipe cannot hide instrumentation changes.
    # Metadata reads and witness comparison remain within supervised collection.
    for candidate in reversed(references):
        candidate_settings = settings(candidate)
        if (
            candidate_settings.instrumentation != current_settings.instrumentation
            and candidate_settings.model_dump(
                exclude={"deadline_unix", "instrumentation"}
            )
            == compute
        ):
            reference = candidate
            instrumentation_pair = True
            break

    with (
        np.load(reference / "inputs.npz", allow_pickle=False) as left,
        np.load(directory / "inputs.npz", allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files) or any(
            not np.array_equal(left[key], right[key]) for key in left.files
        ):
            raise ValueError("retained input/mask/seed corpus differs")

    def outputs(root: Path) -> _OutputWitness:
        logits, masks, actions = [], [], []
        paths = sorted(
            root.glob(f"pass-{protocol.recipe.warmup_passes}-batch-*.npz"),
            key=lambda p: int(p.stem.rsplit("-", 1)[1]),
        )
        for path in paths:
            with np.load(path, allow_pickle=False) as values:
                logits.append(values["logits"])
                masks.append(values["masks"])
                actions.append(values["actions"])
        return _OutputWitness(
            logits=np.concatenate(logits),
            masks=np.concatenate(masks),
            actions=np.concatenate(actions),
        )

    reference_output, current_output = outputs(reference), outputs(directory)
    expected, actual = reference_output["logits"], current_output["logits"]
    mask = reference_output["masks"] != 0
    close = np.isclose(
        expected[mask],
        actual[mask],
        atol=protocol.recipe.absolute_tolerance,
        rtol=protocol.recipe.relative_tolerance,
    )
    actions = reference_output["actions"] == current_output["actions"]
    measured = [
        json.loads((root / f"pass-{protocol.recipe.warmup_passes}.json").read_bytes())[
            "seconds"
        ]
        for root in (reference, directory)
    ]
    identical = bool(np.array_equal(expected, actual) and actions.all())
    overhead = None
    if instrumentation_pair and min(measured) > 0:
        enabled = 1 if current_settings.instrumentation else 0
        overhead = measured[enabled] / measured[1 - enabled] - 1
    _write(
        directory / "comparison.json",
        {
            "reference": str(reference.resolve()),
            "legal_logits_close": bool(close.all()),
            "maximum_absolute_error": float(
                np.abs(expected[mask] - actual[mask]).max()
            ),
            "action_agreement": float(actions.mean()),
            "numerically_equivalent": bool(close.all() and actions.all()),
            "instrumentation_outputs_identical": identical
            if instrumentation_pair
            else None,
            "instrumentation_host_overhead": overhead,
            "overhead_boundary": "measured-pass wall including output witness writes; no speed threshold",
            "roots": len(actions),
            "absolute_tolerance": protocol.recipe.absolute_tolerance,
            "relative_tolerance": protocol.recipe.relative_tolerance,
            "boundary": "same input arrays, masks and row seeds; legal logits only; action ties retained in raw witnesses",
        },
    )
    if instrumentation_pair and not identical:
        raise ValueError("instrumentation changed same-device retained outputs")
    # An unequal action can arise within logit tolerance; retain both facts.
    # Numerical disagreement is evidence, not permission to retry a treatment.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run", "replay"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.out is None:
            parser.error("freeze requires --out")
        protocol = freeze(Recipe.model_validate_json(args.plan.read_bytes()))
        with args.out.open("x") as stream:
            stream.write(protocol.model_dump_json(indent=2) + "\n")
    else:
        protocol = Protocol.model_validate_json(args.plan.read_bytes())
        if args.command == "validate":
            validate(protocol)
        else:
            if args.out is None:
                parser.error("execution requires --out")
            manifest = run(protocol, args.out, replay=args.command == "replay")
            print(manifest.model_dump_json(indent=2))
            if any(a.status != "completed" for a in manifest.attempts):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
