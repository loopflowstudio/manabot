"""Keep immutable, byte-verified copies of the task-local scientific programs.

The archive records original relative paths and includes transitive local
helpers. It contains code only: checkpoints, games, runtime/ledger receipts and
the pinned repository remain separate required inputs. Nothing is executed.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(".runs/grid-007/learning-017")
RECEIPT = ROOT / "analysis-source.json"
NAMES = (
    "verify_learning.py",
    "verify_grouped.py",
    "audit_completed_numerics.py",
    "evaluate_learning.py",
    "evaluation_runtime.py",
    "analyze_learning.py",
    "learning_selection.py",
    "learning_figures.py",
    "learning_resources.py",
    "render_learning_report.py",
    "export_learning_recipes.py",
    "test_learning_analysis.py",
    "check_analysis_readiness.py",
    "check_resource_readiness.py",
    "check_report_readiness.py",
    "preview_learning_figures.py",
    "retain_analysis_source.py",
)
PATHS = (
    *[Path(".runs/grid-007") / name for name in NAMES],
    Path(".runs/gpu-validation/cli.py"),
)


class SourceFile(BaseModel):
    path: str
    stored_path: str
    bytes: int
    sha256: str


class SourceArchive(BaseModel):
    directory: Path
    contents_sha256: str
    manifest_sha256: str
    files: list[SourceFile]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_current() -> SourceArchive:
    archive = SourceArchive.model_validate_json(RECEIPT.read_text())
    if digest(archive.directory / "manifest.json") != archive.manifest_sha256:
        raise ValueError("Archived source manifest changed")
    manifest = json.loads((archive.directory / "manifest.json").read_text())
    serialized = [record.model_dump() for record in archive.files]
    contents = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest["files"] != serialized
        or manifest["contents_sha256"] != contents
        or archive.contents_sha256 != contents
    ):
        raise ValueError("Archived source inventory differs from its receipt")
    if manifest["readme_sha256"] != digest(archive.directory / "README.md"):
        raise ValueError("Archived source instructions changed")
    expected_paths = {str(path) for path in PATHS}
    if {record.path for record in archive.files} != expected_paths or len(
        archive.files
    ) != len(PATHS):
        raise ValueError("Analysis source inventory is incomplete or duplicated")
    for record in archive.files:
        source = Path(record.path)
        expected_storage = Path("files") / source.relative_to(".runs")
        if record.stored_path != str(expected_storage):
            raise ValueError("Unexpected archive storage path")
        retained = archive.directory / record.stored_path
        if (
            source.stat().st_size != record.bytes
            or digest(source) != record.sha256
            or retained.stat().st_size != record.bytes
            or digest(retained) != record.sha256
        ):
            raise ValueError(
                f"Analysis source changed; retain a new snapshot: {record.path}"
            )
    return archive


def main() -> None:
    records = [
        SourceFile(
            path=str(path),
            stored_path=str(Path("files") / path.relative_to(".runs")),
            bytes=path.stat().st_size,
            sha256=digest(path),
        )
        for path in sorted(PATHS)
    ]
    serialized = [record.model_dump() for record in records]
    contents = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory = ROOT / f"analysis-source-{contents[:20]}"
    if not directory.exists():
        directory.mkdir()
        for record in records:
            target = directory / record.stored_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(record.path).read_bytes())
        (directory / "README.md").write_text("""# Protocol017 analysis source archive

This is a code archive, not a complete study dataset or a standalone executable.
The manifest records each file's original and archived paths, byte count and SHA256. The final
analysis receipts separately bind the programs actually used to produce results.
Snapshots are immutable; changed programs receive another contents-derived path.
Archived paths omit the leading `.runs` directory so normal Git ignore rules do
not discard the source when a report package is published. Restore the original
path from each manifest entry when reproducing the analysis.

## Required inputs

Use the pinned repository runtime from source
`3c3681d2b26a7fc8167928fa63fe24a6cd3d058a` and its locked environment. Restore the
archived files to their original relative paths in an isolated checkout, preserving
any newer work. Restore the retained data at the paths named in the analysis and
allocation: protocol017 training directories, allocation/frozen inputs, complete
evaluation tree and manifest, resolved recipes and resource receipts.

The verifier also reads the locked-wheel upload receipt. Runtime admission reads
the retained route012 baseline monitor and the short-screen small-four-stream
initial checkpoint, plus `learning-017/evaluation-runtime.json`. The exact CPU
binary/source/content identities and Python/Torch/NumPy versions must match;
rebuilding from a commit alone does not prove byte-identical native runtime.
The readiness scripts additionally read the named earlier screen013 datasets.

## Ordinary offline checks

From that restored repository root, the selector/manifest invariants are checked by:

```sh
uv run --no-sync pytest .runs/grid-007/test_learning_analysis.py -q
```

To audit existing retained game evidence and apply the frozen selection rule:

```sh
uv run --no-sync python .runs/grid-007/analyze_learning.py
```

The analysis consumes saved games; it does not launch training or replay missing
panels. The report renderer consumes the verified analysis and resources. Rendering
also requires a matching current source-archive receipt; it refuses to overwrite
an existing report directory. Inspect any existing report before replacing it.

`evaluate_learning.py` executes the declared CPU arena panel and is not needed to
inspect already retained results. It rejects an existing admission; do not remove
that guard to rerun a failed panel. `verify_learning.py` calls the earlier numerical
audit through `verify_grouped.py`, which imports definitions with AST selection.
Do not execute `audit_completed_numerics.py` directly: its top-level historical
driver writes an older aggregate.

`learning_resources.py` reads existing SDK rental records only after full retention
and deletion settlement. Its configuration helper expects the operator's existing
credential environment and launcher history, which are intentionally absent from
this source archive. It is unnecessary for rendering an already retained resource
receipt. No credential file, SDK state, wheel archive or paid-run launcher is copied.

The full result still needs retained data, successful checks and scientific/visual
review. Merely archiving these files does not establish any training outcome.
""")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "observed_at": datetime.now(UTC).isoformat(),
                    "contents_sha256": contents,
                    "files": serialized,
                    "readme_sha256": digest(directory / "README.md"),
                },
                indent=2,
            )
            + "\n"
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["contents_sha256"] != contents or manifest["files"] != serialized:
        raise ValueError(
            "An existing source snapshot differs from its contents address"
        )
    if manifest["readme_sha256"] != digest(directory / "README.md"):
        raise ValueError("Archived instructions changed")
    archive = SourceArchive(
        directory=directory,
        contents_sha256=contents,
        manifest_sha256=digest(directory / "manifest.json"),
        files=records,
    )
    RECEIPT.write_text(archive.model_dump_json(indent=2) + "\n")
    verify_current()
    print(
        json.dumps(
            {
                "source_archive": str(directory),
                "files": len(records),
                "bytes": sum(record.bytes for record in records),
                "status": "verified exact source copies",
            }
        )
    )


if __name__ == "__main__":
    main()
