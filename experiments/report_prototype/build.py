"""Render retained evaluation and LLM performance without executing inference.

Training reports belong to Etude experiment files. This builder preserves the
existing Perf surface and copies byte-bound evidence into a portable report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

from manabot.reports.inference import (
    AttemptObservation,
    AttemptProjection,
    BaselineAllocation,
    EvidenceDependency,
    EvidenceFile,
    EvidenceReader,
    evidence_identity,
    inventory,
    project_llm,
    read_manifest,
)

ROOT = Path(__file__).parent
ASSETS = ("report.html", "model.js", "charts.js")


@dataclass(frozen=True)
class _Receipt:
    """Keep a projection beside the source inventory used to validate and copy it."""

    root: Path
    files: list[EvidenceFile]
    projection: dict[str, Any]


def write_evaluations(directories: list[Path], out: Path) -> Path:
    """Calibration runner's existing adapter to the shared Perf view."""
    return write_report(directories, [], out)


def write_report(
    evaluations: list[Path],
    llm: list[Path],
    out: Path,
    baseline_manifest: Path | None = None,
) -> Path:
    evaluations, llm = list(evaluations), list(llm)
    explicit = {
        p.resolve(): kind
        for kind, paths in (("evaluation", evaluations), ("llm", llm))
        for p in paths
    }
    if len(explicit) != len(evaluations) + len(llm):
        raise ValueError("duplicate receipt directory")
    manifest = read_manifest(baseline_manifest) if baseline_manifest else None
    attempts: list[AttemptProjection] = []
    if manifest and baseline_manifest:
        for attempt in manifest.attempts:
            root = (
                (baseline_manifest.parent / attempt.root).resolve()
                if attempt.root
                else None
            )
            attempts.append(AttemptProjection(attempt, root))
            if not attempt.inventory_complete:
                if root in explicit:
                    raise ValueError(
                        "incomplete inventory cannot project game receipts"
                    )
                continue
            if root in explicit:
                if explicit[root] != attempt.kind:
                    raise ValueError(
                        f"attempt {attempt.id} source {root}: receipt kind conflicts "
                        f"with declared {attempt.kind}"
                    )
            elif root and root.is_dir() and attempt.kind != "retained-input":
                # A worker may fail before producing a game receipt. Its outer
                # attempt remains evidence without inventing an evaluation row.
                if (root / "attempt.json").exists():
                    if attempt.kind == "llm":
                        game_receipt = any(
                            (root / name).exists() for name in ("game", "config.json")
                        ) or any(root.glob("game-*/config.json"))
                    else:
                        game_receipt = all(
                            (root / name).exists()
                            for name in ("performance.json", "rows.json")
                        )
                    if not game_receipt:
                        continue
                target = llm if attempt.kind == "llm" else evaluations
                target.append(root)
    if not evaluations and not llm and not attempts:
        raise ValueError("at least one receipt or baseline manifest is required")
    pinned = {a.source: a for a in attempts if a.source is not None}
    policy_files = {root: inventory(root) for root in evaluations}
    pins: dict[Path, EvidenceFile] = {}
    allocations: dict[Path, BaselineAllocation] = {}

    def pin(root: Path, files: list[EvidenceFile]) -> None:
        for file in files:
            path = (root / file.path).resolve()
            if previous := pins.get(path):
                if (previous.sha256, previous.bytes) != (file.sha256, file.bytes):
                    raise ValueError(f"conflicting evidence pins: {path}")
            pins[path] = file

    if manifest and baseline_manifest:
        for allocation in manifest.allocations:
            root = (baseline_manifest.parent / allocation.root).resolve()
            serving = root / "serving.json"
            if serving in allocations:
                raise ValueError(f"duplicate allocation source: {root}")
            allocations[serving] = allocation
            pin(root, allocation.files)
    shared_paths = set(pins)
    for root, files in policy_files.items():
        if relationship := pinned.get(root.resolve()):
            # Shared files may live beneath a cell root. Their absence affects
            # only dependent measurements, even in a historical full inventory.
            available = {f.path for f in files}
            missing_shared = [
                f
                for f in relationship.declaration.files
                if f.path not in available and (root / f.path).resolve() in shared_paths
            ]
            relationship.declaration.validate_inventory(files + missing_shared)
        pin(root, files)
    dependencies = EvidenceReader(pins, allow_unpinned=manifest is None)
    # Validate declared shared bytes even if another missing receipt prevents
    # traversal to them. Absent files remain unavailable and keep their pins.
    for path in sorted(shared_paths):
        dependencies.read(path)
    # Keep the simulation performance dependency out of LLM-only historical
    # reporting, which also works in the lightweight installation.
    if evaluations:
        from manabot.reports.evaluation import project_evaluations

        experiment = project_evaluations(
            evaluations,
            evidence=dependencies,
            declared_allocations=allocations,
        )
    else:
        experiment = _empty_report()
    policy_receipts = [
        _Receipt(root, policy_files[root], projection)
        for root, projection in zip(
            evaluations, experiment["eval_performance"]["runs"], strict=True
        )
    ]
    llm_receipts: list[_Receipt] = []
    for root in llm:
        projection = project_llm(root)
        files = [EvidenceFile.model_validate(f) for f in projection["evidence"]]
        projection["run"] = len(experiment["runs"])
        # LLM inputs have no evaluation milestones or score rows.
        experiment["runs"].append(
            {
                "id": projection["identity"],
                "label": root.name,
                "job_name": f"LLM {root.name} {projection['identity'][:8]}",
                "seed": (projection["config"] or {}).get("seed"),
                "evals": [],
            }
        )
        llm_receipts.append(_Receipt(root, files, projection))
    identities = [r.projection["identity"] for r in llm_receipts]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate LLM evidence is not an independent run")
    eval_ids = [evidence_identity(r.files) for r in policy_receipts]
    if len(set(eval_ids)) != len(eval_ids):
        raise ValueError("duplicate evaluation evidence is not an independent run")
    receipts = {r.root.resolve(): r for r in policy_receipts + llm_receipts}
    for receipt in receipts.values():
        receipt.projection["attempt_id"] = None
    for relationship in attempts:
        root = relationship.source
        if root is None or not root.is_dir():
            continue
        receipt = receipts.get(root)
        observation_path = root / "attempt.json"
        # Policy inventories were validated with their shared dependencies above.
        # Reuse projected inventories; only attempts without a game need a scan.
        if receipt is not None:
            if relationship.declaration.kind == "llm":
                relationship.declaration.validate_inventory(receipt.files)
        elif observation_path.exists():
            relationship.declaration.validate_inventory(inventory(root))
            if not relationship.declaration.inventory_complete and not any(
                f.path == "attempt.json" for f in relationship.declaration.files
            ):
                raise ValueError(
                    "partial inventory requires a pinned attempt observation"
                )
        if observation_path.exists():
            observation = AttemptObservation.model_validate_json(
                observation_path.read_bytes()
            )
            if (
                observation.id != relationship.declaration.id
                or observation.protocol_sha256 != manifest.source
            ):
                raise ValueError(
                    "attempt observation differs from declaration/protocol identity"
                )
            relationship.execution = observation
        if receipt is not None:
            projection = receipt.projection
            if relationship.execution is not None:
                if (
                    relationship.execution.status == "completed"
                    and projection["observed_status"] == "failed"
                ):
                    raise ValueError("completed attempt contains failed game evidence")
                projection["game_observed_status"] = projection["observed_status"]
                projection["observed_status"] = relationship.execution.status
            relationship.bind(projection["run"], projection["observed_status"])
            projection["attempt_id"] = relationship.declaration.id
        elif relationship.execution is not None:
            relationship.bind(None, relationship.execution.status)
    experiment["inference_performance"] = {
        "runs": [r.projection for r in llm_receipts],
        "attempts": [a.to_dict() for a in attempts],
    }
    sources = {p.resolve() for p in evaluations + llm}
    sources.update(a.source for a in attempts if a.source is not None)
    sources.update(path.parent for path in allocations)
    sources.update(path.parent for path in dependencies.files)
    if any(out.resolve().is_relative_to(source) for source in sources):
        raise ValueError("report output must be outside receipt directories")
    out.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        shutil.copyfile(ROOT / asset, out / asset)
    _retain_evidence(
        experiment,
        policy_receipts,
        llm_receipts,
        baseline_manifest,
        out,
        attempts,
        manifest.protocol if manifest else None,
    )
    if llm_receipts:
        experiment["title"] = "Inference performance"
        experiment["question"] = "What do the retained inference receipts measure?"
        experiment["source"] = (
            "Retained evaluation and LLM receipts"
            if evaluations
            else "Retained LLM receipts"
        )
        experiment["context"]["status"] = "incomplete"
        experiment["context"]["caveat"] = (
            "LLM and policy workloads are separate cohorts, never a speedup ratio. Missing startup, warmup and billing remain unavailable."
        )
    payload = json.dumps({"experiments": [experiment]}, allow_nan=False)
    (out / "data.js").write_text("window.REPORT=" + payload + ";\n")
    (out / "data.json").write_text(payload + "\n")
    return out / "report.html"


def _retain_evidence(
    experiment: dict[str, Any],
    policy_receipts: list[_Receipt],
    llm_receipts: list[_Receipt],
    baseline_manifest: Path | None,
    out: Path,
    attempts: list[AttemptProjection],
    protocol: EvidenceFile | None,
) -> None:
    """Verify copied bytes and replace source links with portable report links."""
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    copied: dict[Path, EvidenceFile] = {}

    def retain(source: Path, expected: EvidenceFile | None = None) -> str:
        source = source.resolve()
        actual = copied.get(source)
        data = None
        if actual is None:
            data = source.read_bytes()
            digest = sha256(data).hexdigest()
            target = evidence / f"{digest}-{source.name}"
            actual = EvidenceFile(
                path=str(target.relative_to(out)), sha256=digest, bytes=len(data)
            )
        # A file may explain both local and shared measurements. Both snapshots
        # must match the retained bytes, including when the copy is reused.
        if expected and (actual.sha256, actual.bytes) != (
            expected.sha256,
            expected.bytes,
        ):
            raise ValueError(f"evidence changed during report build: {source}")
        if data is not None:
            (out / actual.path).write_bytes(data)
            copied[source] = actual
        return actual.path

    def retain_files(root: Path, files: list[EvidenceFile]) -> list[dict[str, Any]]:
        return [dict(f.model_dump(), href=retain(root / f.path, f)) for f in files]

    def retain_dependencies(
        files: list[EvidenceDependency],
    ) -> list[EvidenceDependency]:
        result: list[EvidenceDependency] = []
        for file in files:
            href = None
            if file["status"] == "available":
                href = retain(
                    Path(file["path"]),
                    EvidenceFile.model_validate(
                        {key: file[key] for key in ("path", "sha256", "bytes")}
                    ),
                )
            result.append({**file, "href": href})
        return result

    for receipt in policy_receipts:
        run = receipt.projection
        file_refs = retain_files(receipt.root, receipt.files)
        run["dependencies"] = retain_dependencies(run["dependencies"])
        path = out / f"evaluation-{evidence_identity(receipt.files)}.json"
        path.write_text(
            json.dumps(
                {
                    "files": file_refs,
                    "dependencies": run["dependencies"],
                    "allocation_ids": run["allocation_ids"],
                },
                indent=2,
            )
            + "\n"
        )
        run["evidence_manifest"] = path.name
        run["evidence"] = retain(Path(run["evidence"]))
        run["optimization"]["evidence"] = run["evidence"]
    for allocation in experiment["eval_performance"]["allocations"]:
        allocation["dependencies"] = retain_dependencies(allocation["dependencies"])
        links = {f["path"]: f["href"] for f in allocation["dependencies"]}
        allocation["evidence"] = links.get(allocation["source"])
        if serving := allocation["serving"]:
            serving["traces"] = [
                links[path] for path in serving["traces"] if links[path]
            ]
    for receipt in llm_receipts:
        run = receipt.projection
        run["evidence"] = retain_files(receipt.root, receipt.files)
        manifest_path = out / f"corpus-{run['identity']}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "identity": run["identity"],
                    "corpus": run["corpus"],
                    "files": run["evidence"],
                },
                indent=2,
            )
            + "\n"
        )
        run["evidence_manifest"] = manifest_path.name
    available_files = {
        r.root.resolve(): r.files for r in policy_receipts + llm_receipts
    }
    for relationship, projected in zip(
        attempts, experiment["inference_performance"]["attempts"], strict=True
    ):
        if relationship.execution is not None and relationship.source is not None:
            projected["evidence"] = retain_files(
                relationship.source,
                available_files.get(
                    relationship.source, relationship.declaration.files
                ),
            )
            projected["execution_evidence"] = retain(
                relationship.source / "attempt.json"
            )
    if baseline_manifest and protocol:
        source = baseline_manifest.parent / protocol.path
        experiment["inference_performance"]["protocol_status"] = (
            "available" if source.is_file() else "unavailable"
        )
        if source.is_file():
            experiment["inference_performance"]["protocol"] = retain(source, protocol)
    if baseline_manifest:
        experiment["inference_performance"]["baseline_manifest"] = retain(
            baseline_manifest
        )


def _empty_report() -> dict[str, Any]:
    """The existing browser report DTO; no score rows are invented for inference."""
    return {
        "id": "inference-performance",
        "title": "Inference performance",
        "question": "What do the retained inference receipts measure?",
        "source": "Retained inference receipts",
        "synthetic": False,
        "decision_unit": "model",
        "decks": [],
        "deck_names": [],
        "anchors": [],
        "runs": [],
        "cells": [],
        "direct": [],
        "baseline": None,
        "training": {"series": []},
        "timing": [],
        "eval_performance": {"runs": [], "allocations": []},
        "context": {
            "status": "incomplete",
            "world": "recorded; no current-world admission",
            "device": "see receipts",
            "heartbeat_at": None,
            "captured_at": None,
            "caveat": "Separate workloads are not speedup comparisons. Missing measurements are not zero.",
        },
        "layout": {"clock": "updates", "evals": [], "ml": [], "perf": []},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(".runs/eval/report"))
    parser.add_argument("--eval-receipt", type=Path, action="append", default=[])
    parser.add_argument("--llm-receipt", type=Path, action="append", default=[])
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    try:
        report = write_report(
            args.eval_receipt, args.llm_receipt, args.out, args.baseline_manifest
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Receipt validation failed: {error}\n")
    print(report)
    result = json.loads((args.out / "data.json").read_text())["experiments"][0]
    for receipt in result["inference_performance"]["runs"]:
        coverage = receipt["coverage"]
        print(
            f"{receipt['label']}: {coverage['selection_files']} selections, "
            f"{coverage['committed_commands']} committed Commands, "
            f"{coverage['replay_verified_games']} replay-verified games; {receipt['observed_status']}. "
            "Warmup and inclusive billing unavailable."
        )


if __name__ == "__main__":
    main()
