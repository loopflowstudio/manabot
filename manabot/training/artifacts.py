"""Publish retained TrainingRun artifacts to S3 without rewriting training evidence.

ArtifactManifest binds the exact producer export and named artifact roles to
verified S3 bytes. The CLI publishes completed snapshots or fetches one named
artifact into a cache; neither path runs training or evaluation. A manifest is
storage provenance, never policy admission or a portable training recovery claim.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from manabot.infra.artifacts import S3ArtifactStore, StoredArtifact, verify_file
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun


class LocalArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    training_run_id: str
    source_run: StoredArtifact
    artifacts: dict[str, StoredArtifact]


def retained_artifacts(run: TrainingRun) -> dict[str, LocalArtifact]:
    """Keep role and rejection provenance; do not infer files from directories."""
    result: dict[str, LocalArtifact] = {}
    for stage in run.stages:
        for category in ("artifacts", "rejected_artifacts"):
            for name, reference in getattr(stage, category).items():
                role = f"stages/{stage.id}/{category}/{name}"
                if role in result:
                    raise ValueError(f"duplicate artifact role: {role}")
                result[role] = LocalArtifact.model_validate(reference)
    for item in run.monitoring_checkpoints:
        if item.artifact is not None:
            role = f"monitoring/{item.ordinal}"
            if role in result:
                raise ValueError(f"duplicate artifact role: {role}")
            result[role] = LocalArtifact.model_validate(item.artifact)
    if run.recovery_artifact is not None:
        result["recovery"] = LocalArtifact.model_validate(run.recovery_artifact)
    if run.fixed_validation is not None:
        result["fixed-validation"] = LocalArtifact.model_validate(
            run.fixed_validation.artifact
        )
    return result


def publish_run(
    source: Path,
    destination: str,
    output: Path,
    store: S3ArtifactStore,
    *,
    resolve_artifact: Callable[[LocalArtifact], Path] | None = None,
) -> ArtifactManifest:
    """Preflight every local artifact before uploading; preserve exact run bytes."""
    payload = source.read_bytes()
    run = TrainingRun.model_validate_json(payload)
    if run.status in {"running", "pending"}:
        raise ValueError("publish a stopped or completed TrainingRun snapshot")
    references = retained_artifacts(run)
    local_paths = {
        role: resolve_artifact(reference)
        if resolve_artifact is not None
        else Path(reference.path)
        for role, reference in references.items()
    }
    for role, reference in references.items():
        verify_file(local_paths[role], reference.sha256, reference.bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(payload).hexdigest()
    if output.exists():
        previous = ArtifactManifest.model_validate_json(output.read_text())
        if previous.source_run.sha256 != source_sha:
            raise ValueError("manifest output already belongs to a different snapshot")
    published: dict[str, StoredArtifact] = {}
    by_digest: dict[str, StoredArtifact] = {}
    for role, reference in references.items():
        if reference.sha256 not in by_digest:
            by_digest[reference.sha256] = store.publish(
                local_paths[role],
                destination,
                sha256=reference.sha256,
                size=reference.bytes,
            )
        published[role] = by_digest[reference.sha256]
    source_reference = store.publish(
        source, destination, sha256=source_sha, size=len(payload)
    )
    manifest = ArtifactManifest(
        training_run_id=run.id, source_run=source_reference, artifacts=published
    )
    atomic_json(output, manifest.model_dump(mode="json"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--run", type=Path, required=True)
    publish.add_argument("--destination", required=True)
    publish.add_argument("--out", type=Path, required=True)
    publish.add_argument("--profile")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument(
        "--artifact", required=True, help="Exact role in manifest, or source-run"
    )
    fetch.add_argument("--cache", type=Path, default=Path(".runs/artifact-cache"))
    fetch.add_argument("--profile")
    args = parser.parse_args()
    try:
        store = S3ArtifactStore(profile=args.profile)
        if args.command == "publish":
            publish_run(args.run, args.destination, args.out, store)
            payload = args.out.read_bytes()
            reference = store.publish(
                args.out,
                args.destination,
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
            )
            atomic_json(
                args.out.with_suffix(".receipt.json"), reference.model_dump(mode="json")
            )
            print(reference.uri)
        else:
            manifest = ArtifactManifest.model_validate_json(args.manifest.read_text())
            reference = (
                manifest.source_run
                if args.artifact == "source-run"
                else manifest.artifacts[args.artifact]
            )
            print(store.fetch(reference, args.cache).resolve())
    except Exception as error:
        # SDK messages can contain signed URLs; expose only a credential-safe type.
        raise SystemExit(
            f"Artifact storage failed ({type(error).__name__}); local evidence retained"
        ) from None


if __name__ == "__main__":
    main()
