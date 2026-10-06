"""Export compact evidence from the completed ETU-106 campaign, without execution.

Saved receipts remain authoritative. The export keeps every file hash, resolved
configuration, checkpoint, paired deal score and run cost; large private traces
and per-update diagnostics stay in the retained campaign and its backup.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    # Saved JSON is a heterogeneous evidence boundary, not a training input.
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: Path, output: Path, backup: Path | None) -> None:
    study = _read(source / "study/study.json")
    supervisor = _read(source / "supervisor.json")
    factorial = _read(source / "study/factorial.json")
    assert supervisor["status"] == study["status"] == factorial["status"] == "completed"
    assert len(study["comparisons"]) == 24 and len(study["runs"]) == 12
    manifest = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        sha = _sha(path)
        if backup is not None:
            assert _sha(backup / relative) == sha, relative
        manifest.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha))
    cells = []
    for cell in study["comparisons"]:
        rows = cell["rows"]
        assert len(rows) == 100 and cell["replay"]["passed"]
        assert {(r["deal_seed"], r["leg"]) for r in rows} == {
            (deal, leg) for deal in range(961160, 961185) for leg in range(4)
        }
        assert all(
            r["terminated"]
            and r["replay_passed"]
            and not r["truncated"]
            and r["failure"] is None
            for r in rows
        )
        latency = [r["latency"][r["player_a"]] for r in rows]
        cells.append(
            {
                **{
                    k: cell[k]
                    for k in (
                        "a",
                        "training_seed",
                        "cutoff",
                        "evaluation_seconds",
                        "replay",
                        "trace",
                    )
                },
                "deal_leg_scores": [
                    [r["deal_seed"], r["leg"], r["score_a"]] for r in rows
                ],
                "candidate_inference_calls": sum(r["count"] for r in latency),
                "candidate_inference_seconds": sum(r["seconds"] for r in latency),
            }
        )
    runs = []
    for entry in study["runs"]:
        path = Path(entry["path"])
        assert _sha(path) == entry["sha256"]
        run = _read(path)
        assert run["status"] == "completed" and run["error"] is None
        stages = []
        for stage in run["stages"]:
            assert stage["status"] == "completed" and stage["error"] is None
            stages.append({k: v for k, v in stage.items() if k != "diagnostics"})
        runs.append(
            {
                **{
                    k: run[k]
                    for k in (
                        "id",
                        "seed",
                        "status",
                        "error",
                        "regime_digest",
                        "identities",
                        "seconds",
                        "setup_seconds",
                        "seed_streams",
                    )
                },
                "stages": stages,
            }
        )
    # Full value-loss time series remain in the hash-bound original factorial/run receipts.
    exposures = [
        {k: v for k, v in row.items() if k != "recorded_value_losses"}
        for row in factorial["exposures"]
    ]
    result = dict(
        schema="etu106-pooling-filter-evidence-v1",
        source=str(source.resolve()),
        supervisor=supervisor,
        calibration=_read(source / "calibration.json"),
        resolved_plan=_read(source / "resolved-plan.json"),
        study_seconds=study["seconds"],
        cells=cells,
        runs=runs,
        measurements=study["measurements"],
        factorial={**factorial, "exposures": exposures},
        common_cost=_read(source / "study/cost-comparison.json"),
        diagnostic=_read(source / "study/diagnostic-2h.json"),
        files=manifest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        f"Exported {len(manifest)} file identities, 12 runs, 24 cells, 2400 verified game rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    export(args.source, args.output, args.backup)


if __name__ == "__main__":
    main()
