"""Retain protocol017 placement evidence without touching either evaluation.

This copies immutable admissions, both completed calibration attempts, and the
logging repair. The active Mini panel and its later scientific results remain
owned by their existing processes. No model or cloud client is imported.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

STUDY = Path(".runs/grid-007/learning-017")
OUTPUT = STUDY / "mini-operations"


@dataclass(frozen=True)
class Copy:
    source: Path
    destination: Path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_manifest(root: Path, manifest: Path, *, includes_manifest: bool) -> None:
    # Retained JSON is validated here before it controls any filesystem access.
    value = json.loads(manifest.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError(f"Invalid manifest: {manifest}")
    seen: set[str] = set()
    for row in value["files"]:
        if not isinstance(row, dict):
            raise ValueError("Manifest entry is not a record")
        name, size, sha = row.get("path"), row.get("bytes"), row.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(size, int)
            or not isinstance(sha, str)
        ):
            raise ValueError("Invalid manifest entry types")
        path = root / name
        if name in seen or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Duplicate or escaping manifest path")
        if path.stat().st_size != size or _digest(path) != sha:
            raise ValueError(f"Retained bytes changed: {path}")
        seen.add(name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if not includes_manifest:
        actual.remove("manifest.json")
    if actual != seen:
        raise ValueError(f"Manifest coverage differs: {root}")


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("Operations package exists; inspect it without replacing it")
    failed = STUDY / "mini-calibration"
    passed = STUDY / "mini-calibration-standard"
    transfer = STUDY / "mini-transfer"
    _verify_manifest(
        failed, transfer / "failed-control-retention.json", includes_manifest=True
    )
    _verify_manifest(passed, passed / "manifest.json", includes_manifest=False)
    parity = json.loads(
        (transfer / "calibration-standard-verification.json").read_text()
    )
    if not (
        parity["status"] == "verified"
        and parity["games"] == 32
        and parity["all_decisions_identities_and_replays_equal"]
        and parity["manifest_sha256"] == _digest(passed / "manifest.json")
        and parity["comparison_sha256"] == _digest(passed / "comparison.json")
    ):
        raise ValueError("Complete Mini control parity is not established")
    cancellation = json.loads((transfer / "laptop-cancellation.json").read_text())
    for name, entry in cancellation["files"].items():
        path = STUDY / "evaluation" / name
        if path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
            raise ValueError(f"Cancelled laptop evidence changed: {path}")
    patch = json.loads((transfer / "worker-logging-fix.json").read_text())
    if patch["patch_sha256"] != _digest(transfer / "worker-event-lines.patch"):
        raise ValueError("Logging repair differs from its receipt")
    mini = json.loads((STUDY / "mini-plan.json").read_text())
    admission_path = transfer / "owner-receipts/evaluation/admission.json"
    admission = json.loads(admission_path.read_text())
    if admission["mini_plan_sha256"] != _digest(STUDY / "mini-plan.json"):
        raise ValueError("Mini admission differs from its prospective plan")
    if (admission["games"], admission["commands"]) != (18304, 44):
        raise ValueError("Mini admission changed the scientific allocation")
    audit_path = transfer / "owner-receipts/mini-analysis-worker/admission.json"
    audit = json.loads(audit_path.read_text())
    if audit["evaluation_admission_sha256"] != _digest(admission_path):
        raise ValueError("Audit owner refers to another evaluation admission")
    if audit["controller_sha256"] != _digest(
        Path(__file__).with_name("finish_mini_learning.py")
    ):
        raise ValueError("Audit owner source changed after admission")
    if admission["controller_sha256"] != _digest(
        Path(__file__).with_name("evaluate_learning_on_mini.py")
    ):
        raise ValueError("Evaluation owner source changed after admission")

    copies: list[Copy] = []
    for root, destination in (
        (failed, "failed-background-control"),
        (passed, "passed-standard-control"),
    ):
        copies.extend(
            Copy(p, Path(destination) / p.relative_to(root))
            for p in sorted(root.rglob("*"))
            if p.is_file()
        )
    for name in (
        "failed-control-retention.json",
        "calibration-standard-verification.json",
        "calibration-standard-launch.json",
        "worker-logging-fix.json",
        "worker-event-lines.patch",
        "evaluation-launch.json",
        "verification.json",
        "supplement-verification.json",
        "laptop-cancellation.json",
        "calibration-background-source.py",
    ):
        copies.append(Copy(transfer / name, Path("receipts") / name))
    for name in cancellation["files"]:
        copies.append(
            Copy(STUDY / "evaluation" / name, Path("cancelled-laptop") / name)
        )
    for name in (
        "mini-transfer/analysis-launch.json",
        "mini-analysis-worker/admission.json",
        "evaluation/admission.json",
    ):
        copies.append(
            Copy(transfer / "owner-receipts" / name, Path("mini-owners") / name)
        )
    for name in ("mini-plan.json", "evaluation-runtime.json"):
        copies.append(Copy(STUDY / name, Path(name)))
    for name in (
        "evaluate_learning_on_mini.py",
        "finish_mini_learning.py",
        "calibrate_mini_evaluation.py",
        "prepare_mini_plan.py",
        "verify_mini_transfer.py",
        "retain_mini_operations.py",
    ):
        copies.append(Copy(Path(__file__).with_name(name), Path("controllers") / name))
    for name in (
        "verification.json",
        "comparison.json",
        "manifest.json",
        "protocol.txt",
        "protocol-random.json",
        "protocol-scripted_greedy.json",
    ):
        copies.append(
            Copy(
                Path(".runs/grid-007/local-evaluation-parallel") / name,
                Path("original-parallel-control") / name,
            )
        )

    temporary = OUTPUT.with_name(OUTPUT.name + ".assembling")
    temporary.mkdir(exist_ok=False)
    for item in copies:
        target = temporary / item.destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.source, target)
        if _digest(target) != _digest(item.source):
            raise ValueError(f"Copy differs: {item.source}")
    readme = f"""# Protocol017 placement and operational evidence

This package records admissions and completed controls. It does not claim the
live 18,304-game Mini panel or subsequent scientific/visual review has finished.
The scientific report and final recommendation must establish those separately.

- The laptop attempt was explicitly cancelled by Jack Heart. Its 352 replayed
  games are retained and excluded from the replacement Mini scores. Its exit124
  is not evidence that the original scientific deadline expired. Original logs,
  admission, status and manifest are copied; partial game traces remain in the
  original retained attempt and are not included in this small package.
- The first Mini control used Background scheduling and recorded watchdog
  failures. It was stopped and retained. The final controller log differs from
  the original manifest because a traceback was appended after that manifest;
  `receipts/failed-control-retention.json` verifies all 27 final files, including
  the unchanged original manifest. No historical receipt was rewritten.
- A separately admitted Standard-scheduling control completed 32 games, with
  all recorded decisions, identities and replays matching the earlier control.
  Its parallel command interval was {parity["parallel_seconds"]:.6f} seconds.
  Complete failed and successful control trees are included separately.
- PR311/commit `{patch["commit"]}` serializes supervisor event output. The exact
  patch and receipt are included. Scientific source remains
  `{patch["scientific_source"]}`; no model, game or learning change was made.
- The Mini plan scales operational command/work limits from matched controls
  while preserving the full panel and 120-second game watchdog. Four commands
  run concurrently; command limit {mini["evaluation"]["command_seconds"]:,}s,
  work limit {mini["evaluation"]["work_seconds"]:,}s, retention240s. These are
  bounds, not an ETA. Launchd uses Standard scheduling and no automatic restart.
- The separate one-shot audit owner waits for successful full-panel completion
  before running the frozen analysis and renderer. Source and admission hashes
  are retained here. Its launch is not proof that the later audit succeeded.

The Mini move used direct SSH file transfer and no new GPU rental or S3 artifact
download. The renderer's historical budget paragraph describes rental accounting
and an ancillary reserve. That reserve is not measured S3 attribution or proof
that total all-inclusive campaign spending is within $50; shared storage/transfer
charges remain partly unattributed. Do not present quote/SDK costs as invoices.

Treat the frozen protocol's original laptop-sized limits and placement as
historical, with this explicit Mini plan/admission as the operational amendment.
Scientific endpoints, seeds, deal panel and frozen selection rule are unchanged.
No failed/cancelled control or partial laptop score is combined with Mini scores.
"""
    (temporary / "README.md").write_text(readme)
    records = [
        {
            "path": str(p.relative_to(temporary)),
            "bytes": p.stat().st_size,
            "sha256": _digest(p),
        }
        for p in sorted(temporary.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "observed_at": datetime.now(UTC).isoformat(),
        "status": "verified operational history; final scientific outcome pending",
        "files": records,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _verify_manifest(temporary, temporary / "manifest.json", includes_manifest=False)
    temporary.rename(OUTPUT)
    print(
        json.dumps(
            {
                "directory": str(OUTPUT),
                "files": len(records),
                "bytes": sum(row["bytes"] for row in records),
                "manifest_sha256": _digest(OUTPUT / "manifest.json"),
                "new_games": 0,
                "new_rentals": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
