"""Historical evidence projects offline; incomplete work never becomes success."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

from experiments.report_prototype.build import write_report
from manabot.reports import evaluation
from manabot.reports.inference import (
    BaselineAllocation,
    EvidenceReader,
    inventory,
    project_llm,
)


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _fixture(root: Path, *, failure: bool = False) -> Path:
    """Schema-faithful, controlled-clock receipt; never performance evidence."""
    config = {
        "model_id": "fixture/model",
        "model_revision": "a" * 40,
        "tokenizer_id": "fixture/model",
        "tokenizer_revision": "b" * 40,
        "device": "cpu",
        "precision": "float32",
        "context_tokens": 64,
        "max_new_tokens": 4,
        "deadline_seconds": 10.0,
        "fallback_seed": 1,
        "decoding": "greedy",
        "enable_thinking": False,
    }
    _write(
        root / "game/config.json",
        {"seed": 23, "model_seat": 0, "deck": {"Mountain": 60}},
    )
    _write(
        root / "game/runtime.json",
        {"runtime": "fixture", "runtime_version": "1", "device_name": "CPU"},
    )
    generation = {
        "prompt": {"text": "prompt", "token_ids": [1, 2]},
        "response": {"text": "A0", "token_ids": [3]},
        "prompt_tokens": 2,
        "output_tokens": 1,
        "validation_error": None,
        "seconds": 0.5,
    }
    for i, origin in enumerate(
        ["initial", "retry", "fallback"] + ([None] if failure else [])
    ):
        attempts = [
            deepcopy(generation)
            for _ in range(2 if origin in {"retry", "fallback"} else 1)
        ]
        if len(attempts) == 2:
            attempts[0]["validation_error"] = "not a legal reference"
        if origin == "fallback":
            attempts[-1]["validation_error"] = "not a legal reference"
        if origin is None:
            attempts[-1].update(response=None, output_tokens=None)
        step = root / "game" / f"decision-{i:05d}"
        _write(
            step / "selection.json",
            {
                "version": "llm-decision-v1",
                "decision": {
                    "text": "prompt",
                    "legal_references": ["A0"],
                    "representation_id": "fixture",
                    "source_identity": {"capture_digest": f"capture-{i}"},
                },
                "config": config,
                "runtime": None,
                "attempts": attempts,
                "selected_reference": "A0" if origin else None,
                "origin": origin,
                "failure": {"kind": "runtime", "message": "injected worker failure"}
                if origin is None
                else None,
                "fallback": {"seed": 1, "selected_index": 0, "state_before": "fixture"}
                if origin == "fallback"
                else None,
                "seconds": float(i + 1),
                "peak_gpu_bytes": None,
            },
        )
        _write(
            step / "capture.json",
            {"digest": f"capture-{i}", "world_version": "fixture-world"},
        )
        if origin:
            _write(
                step / "command.json",
                {"command_id": f"command-{i}", "expected_revision": i},
            )
            _write(
                step / "transition.json",
                {
                    "receipt": {
                        "command_id": f"command-{i}",
                        "before_revision": i,
                        "after_revision": i + 1,
                    }
                },
            )
    _write(
        root / "game/summary.json",
        {
            "completed": not failure,
            "failure": "runtime failure" if failure else None,
            "decisions": 3,
            "model_decisions": 4 if failure else 3,
            "game_seconds": 20.0,
            "model_load_seconds": 2.0,
            "peak_gpu_bytes": None,
            "replay_matches": True,
            "decision_seconds_median": 2.0,
        },
    )
    _write(root / "cache.json", {"seconds": 5.0, "files": []})
    (root / "gpu-memory.csv").write_text(
        "timestamp, name, memory.total [MiB], memory.used [MiB]\n2026/01/01 00:00:00.000, fixture, 8 MiB, 3 MiB\n2026/01/01 00:00:01.000, fixture, 8 MiB, 4 MiB\n"
    )
    return root


def _report(path: Path) -> dict[str, Any]:
    return json.loads(path.with_name("data.json").read_text())["experiments"][0]


def _baseline(
    path: Path, root: Path, *, status: str = "completed", policy: Path | None = None
) -> Path:
    _write(
        path,
        {
            "version": "inference-baseline-v1",
            "source": "fixture",
            "attempts": [
                {
                    "id": "llm-1",
                    "kind": "llm",
                    "cohort": "fixture-llm",
                    "repeat": 0,
                    "status": status,
                    "reason": "controlled fixture",
                    "root": str(root),
                    "files": [f.model_dump() for f in inventory(root)],
                    "corpus": [
                        str(p.relative_to(root))
                        for p in sorted(root.glob("game/decision-*/selection.json"))
                    ],
                },
                {
                    "id": "gpu-1",
                    "kind": "evaluation",
                    "cohort": "policy-cuda",
                    "repeat": 0,
                    "status": "unattempted",
                    "reason": "no execution authorization",
                },
            ],
        },
    )
    if policy is not None:
        data = json.loads(path.read_text())
        data["attempts"].append(
            {
                "id": "policy-1",
                "kind": "evaluation",
                "cohort": "fixture-policy",
                "repeat": 0,
                "status": "completed",
                "reason": "controlled policy panel",
                "root": str(policy),
                "files": [f.model_dump() for f in inventory(policy)],
            }
        )
        _write(path, data)
    return path


def test_latency_counts_boundaries_and_byte_preservation(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "llm")
    before = inventory(root)
    report = write_report([], [root], tmp_path / "report")
    data = _report(report)
    assert data["cells"] == data["runs"][0]["evals"] == []
    receipt = data["inference_performance"]["runs"][0]
    assert receipt["attempt_id"] is None
    assert data["inference_performance"]["attempts"] == []
    assert receipt["coverage"]["generation_attempts"] == 5
    assert (
        receipt["coverage"]["initial"]
        == receipt["coverage"]["repaired"]
        == receipt["coverage"]["fallback"]
        == 1
    )
    assert receipt["coverage"]["committed_commands"] == 3
    assert receipt["coverage"]["replay_verified_games"] == 1
    assert receipt["selection_latency"] | {"method": ""} == {
        "n": 3,
        "population": 3,
        "p50": 2,
        "p95": 2,
        "maximum": 3,
        "method": "",
    }
    metrics = receipt["metrics"]
    assert metrics["minimum_prompt_tokens"]["value"] == 2
    assert metrics["output_tokens"]["value"] == 5
    assert metrics["game_wall"]["value"] == 20
    assert metrics["cache_preparation"]["value"] == 5
    assert metrics["worker_readiness"]["value"] == 2
    assert metrics["model_decisions_per_game_second"]["value"] == 0.15
    assert metrics["replay_verified_games_per_game_second"]["value"] == 0.05
    assert metrics["device_peak_used"]["value"] == 4 * 1024 * 1024
    assert metrics["process_peak_allocated"]["value"] is None
    for name in [
        "inclusive_cost",
        "games_per_dollar",
        "whole_attempt",
        "warmup",
        "prefill",
        "native_work",
    ]:
        assert metrics[name]["value"] is None
        assert metrics[name]["status"] == "unavailable"
    for evidence in receipt["evidence"]:
        copied = report.parent / evidence["href"]
        assert copied.read_bytes() == (root / evidence["path"]).read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == evidence["sha256"]
    assert inventory(root) == before


def test_failed_and_pending_work_keep_distinct_populations(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "failed", failure=True)
    result = project_llm(root)
    assert result["observed_status"] == "failed"
    assert result["coverage"]["failed_selections"] == 1
    assert result["coverage"]["generation_attempts"] == 6
    assert result["selection_latency"]["n"] == 3
    assert result["failed_selection_latency"]["maximum"] == 4
    assert result["metrics"]["selection_sum"]["value"] == 10
    assert result["metrics"]["replay_verified_games_per_game_second"]["value"] == 0
    assert result["metrics"]["model_decisions_per_game_second"]["value"] == 0.15
    # A failed-only cohort cannot emit a success latency quantile.
    for path in sorted((root / "game").glob("decision-*"))[:3]:
        shutil.rmtree(path)
    result = project_llm(root)
    assert result["selection_latency"]["n"] == 0
    assert result["selection_latency"]["p95"] is None
    assert result["metrics"]["model_decisions_per_game_second"]["value"] is None
    (root / "game/summary.json").unlink()
    result = project_llm(root)
    assert result["observed_status"] == "unknown"
    assert result["metrics"]["game_wall"]["value"] is None


def test_missing_files_do_not_become_zero_or_completed_game(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "partial")
    (root / "cache.json").unlink()
    (root / "gpu-memory.csv").unlink()
    (root / "game/decision-00000/transition.json").unlink()
    result = project_llm(root)
    assert result["observed_status"] == "completed"
    assert result["coverage"]["committed_commands"] == 2
    assert result["coverage"]["replay_verified_games"] is None
    for key in [
        "cache_preparation",
        "device_peak_used",
        "committed_commands_per_game_second",
    ]:
        assert result["metrics"][key]["value"] is None


def test_duplicate_copies_and_digest_mutation_fail(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "first")
    clone = tmp_path / "copied"
    shutil.copytree(root, clone)
    with pytest.raises(ValueError, match="duplicate LLM"):
        write_report([], [root, clone], tmp_path / "out")
    manifest = _baseline(tmp_path / "baseline.json", root)
    report = _report(write_report([], [root], tmp_path / "out", manifest))
    assert len(report["inference_performance"]["runs"]) == 1
    assert (
        report["inference_performance"]["attempts"][1]["evidence_status"]
        == "unavailable"
    )
    (root / "cache.json").write_text('{"seconds": 99, "files": []}')
    with pytest.raises(ValueError, match="digest/size/inventory mismatch"):
        write_report([], [], tmp_path / "changed", manifest)


@pytest.mark.parametrize("pinned", [False, True])
def test_evidence_changed_after_projection_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    root = _fixture(tmp_path / "llm")
    manifest = _baseline(tmp_path / "baseline.json", root) if pinned else None

    def project_then_mutate(directory: Path) -> dict[str, Any]:
        receipt = project_llm(directory)
        _write(directory / "cache.json", {"seconds": 99, "files": []})
        return receipt

    monkeypatch.setattr(
        "experiments.report_prototype.build.project_llm", project_then_mutate
    )
    with pytest.raises(ValueError, match="evidence changed during report build"):
        write_report([], [root], tmp_path / "out", manifest)
    assert not (tmp_path / "out/data.json").exists()


def test_manifest_preserves_corpus_order(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "llm")
    manifest = _baseline(tmp_path / "baseline.json", root)
    data = json.loads(manifest.read_text())
    data["attempts"][0]["corpus"].reverse()
    _write(manifest, data)
    with pytest.raises(ValueError, match="ordinal order"):
        write_report([], [], tmp_path / "out", manifest)


@pytest.mark.parametrize(
    "mutation", ["negative", "nan", "tokens", "illegal", "binding", "json"]
)
def test_malformed_receipts_fail_closed(tmp_path: Path, mutation: str) -> None:
    root = _fixture(tmp_path / "bad")
    path = root / "game/decision-00000/selection.json"
    data = json.loads(path.read_text())
    if mutation == "negative":
        data["seconds"] = -1
    elif mutation == "nan":
        data["seconds"] = float("nan")
    elif mutation == "tokens":
        data["attempts"][0]["prompt_tokens"] = 9
    elif mutation == "illegal":
        data["selected_reference"] = "A99"
    elif mutation == "binding":
        data["decision"]["source_identity"]["capture_digest"] = "wrong"
    _write(path, data)
    if mutation == "json":
        path.write_text("{broken")
    with pytest.raises(ValueError):
        project_llm(root)


def test_cli_never_imports_model_or_native_runtime(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "offline")
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text("""import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'managym'} or fullname.startswith(('manabot.sim', 'manabot.infra', 'manabot.env', 'manabot.model', 'manabot.llm')):
            raise AssertionError('Forbidden report import: ' + fullname)
sys.meta_path.insert(0, Block())
""")
    env = dict(os.environ, PYTHONPATH=str(blocker) + os.pathsep + str(Path.cwd()))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.report_prototype.build",
            "--llm-receipt",
            str(root),
            "--out",
            str(tmp_path / "report"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "3 selections, 3 committed Commands, 1 replay-verified games" in result.stdout
    )


def test_manifest_only_unattempted_cells_render_without_fake_runs(
    tmp_path: Path,
) -> None:
    manifest = _baseline(tmp_path / "baseline.json", _fixture(tmp_path / "missing"))
    shutil.rmtree(tmp_path / "missing")
    data = _report(write_report([], [], tmp_path / "out", manifest))
    assert data["runs"] == []
    assert all(
        a["evidence_status"] == "unavailable"
        for a in data["inference_performance"]["attempts"]
    )
    completed, unattempted = data["inference_performance"]["attempts"]
    assert completed["declared_status"] == "completed"
    assert unattempted["declared_status"] == "unattempted"
    assert all(
        a["observed_status"] == "unknown" and a["run"] is None
        for a in (completed, unattempted)
    )


def _evaluation_fixture(root: Path) -> Path:
    worker = {
        "elapsed_seconds": 10.0,
        "process_cpu_seconds": 4.0,
        "measurements": {
            "native_step_and_encoding": {"seconds": 4.0},
            "inference_round_trip": {"seconds": 3.0},
        },
        "spans": [],
        "latency_boundary": "game worker submission through reply identity validation",
        "latency_sample_count": 3,
        "request_p50_seconds": 0.5,
        "request_p95_seconds": 0.75,
    }
    _write(
        root / "performance.json",
        {
            "elapsed_seconds": 20.0,
            "counters": {"decisions": 3, "legs_per_block": 1},
            "measurements": {},
            "spans": [],
            "sample_coverage": "controlled fixture",
        },
    )
    _write(
        root / "rows.json",
        [
            {
                "failure": None,
                "terminated": True,
                "replay_passed": True,
                "integrity": {},
                "deal_seed": 1,
                "termination_reason": "completed",
                "trace_path": "games.jsonl.gz",
            }
        ],
    )
    (root / "games.jsonl.gz").write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "match_id": "fixture-game",
                    "decisions": [{}, {}, {}],
                    "worker_performance": worker,
                }
            ).encode()
        )
    )
    return root


@pytest.mark.parametrize("phases", [True, False])
def test_worker_latency_remains_per_game_without_optimization_phases(
    tmp_path: Path, phases: bool
) -> None:
    policy = _evaluation_fixture(tmp_path / "policy")
    trace = policy / "games.jsonl.gz"
    first = json.loads(gzip.decompress(trace.read_bytes()))
    second = deepcopy(first)
    second["match_id"] = "second-game"
    second["worker_performance"]["request_p50_seconds"] = 2.0
    second["worker_performance"]["request_p95_seconds"] = 3.0
    games = [first, second]
    if not phases:
        for game in games:
            game["worker_performance"]["measurements"] = {}
    trace.write_bytes(gzip.compress("\n".join(map(json.dumps, games)).encode()))
    result = _report(write_report([policy], [], tmp_path / "report"))
    (run,) = result["eval_performance"]["runs"]
    assert [g["game"] for g in run["worker_request_latency"]] == [
        "fixture-game",
        "second-game",
    ]
    assert [g["request_p95_seconds"] for g in run["worker_request_latency"]] == [
        0.75,
        3.0,
    ]
    assert bool(run["opportunities"]) == phases
    assert all("worker_request_latency" not in o for o in run["opportunities"])


def test_mixed_report_keeps_workloads_and_evidence_separate(tmp_path: Path) -> None:
    llm = _fixture(tmp_path / "llm", failure=True)
    policy = _evaluation_fixture(tmp_path / "policy")
    report = write_report([policy], [llm], tmp_path / "mixed")
    result = _report(report)
    assert len(result["runs"]) == 2
    assert result["eval_performance"]["runs"][0]["valid_games"] == 1
    assert (
        result["inference_performance"]["runs"][0]["coverage"]["replay_verified_games"]
        == 0
    )
    assert result["inference_performance"]["runs"][0]["run"] == 1
    assert result["runs"][1]["evals"] == []
    assert result["cells"] == []
    assert result["timing"][0]["run"] == 0
    clone = tmp_path / "policy-copy"
    shutil.copytree(policy, clone)
    with pytest.raises(ValueError, match="duplicate evaluation"):
        write_report([policy, clone], [], tmp_path / "duplicate")


@pytest.mark.parametrize("kind", ["llm", "evaluation"])
@pytest.mark.parametrize("failed", [False, True])
def test_terminal_contradictions_name_attempt_and_source(
    tmp_path: Path, kind: str, failed: bool
) -> None:
    llm = _fixture(tmp_path / "llm", failure=failed)
    policy = _evaluation_fixture(tmp_path / "policy")
    if failed:
        rows = json.loads((policy / "rows.json").read_text())
        rows.append({**rows[0], "failure": "injected failure", "terminated": False})
        _write(policy / "rows.json", rows)
    manifest = _baseline(
        tmp_path / "baseline.json",
        llm,
        status="failed" if failed else "completed",
        policy=policy,
    )
    data = json.loads(manifest.read_text())
    data["attempts"][2]["status"] = "failed" if failed else "completed"
    attempt = data["attempts"][0 if kind == "llm" else 2]
    attempt["status"] = "completed" if failed else "failed"
    _write(manifest, data)
    with pytest.raises(ValueError) as error:
        write_report([], [], tmp_path / "out", manifest)
    assert attempt["id"] in str(error.value)
    assert attempt["root"] in str(error.value)
    assert f"declared {attempt['status']} contradicts observed" in str(error.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("status", ["planned", "attempted", "unknown", "unattempted"])
def test_nonterminal_declarations_can_gain_observations(
    tmp_path: Path, status: str
) -> None:
    root = _fixture(tmp_path / "llm", failure=True)
    manifest = _baseline(tmp_path / "baseline.json", root, status=status)
    data = _report(write_report([], [], tmp_path / "out", manifest))
    attempt = data["inference_performance"]["attempts"][0]
    assert attempt["declared_status"] == status
    assert attempt["observed_status"] == "failed"
    assert attempt["run"] == data["inference_performance"]["runs"][0]["run"]


def test_missing_completed_evidence_restores_same_attempt_links(tmp_path: Path) -> None:
    llm = _fixture(tmp_path / "llm")
    # Retained summary completion does not certify missing replay evidence.
    (llm / "game/decision-00000/transition.json").unlink()
    policy = _evaluation_fixture(tmp_path / "policy")
    manifest = _baseline(tmp_path / "baseline.json", llm, policy=policy)
    original = manifest.read_bytes()
    for source in (llm, policy):
        source.rename(source.with_suffix(".absent"))
    missing = _report(write_report([], [], tmp_path / "missing", manifest))
    assert missing["runs"] == []
    for attempt in missing["inference_performance"]["attempts"]:
        assert attempt["observed_status"] == "unknown"
        assert attempt["evidence_status"] == "unavailable"
        assert attempt["run"] is None
    for source in (llm, policy):
        source.with_suffix(".absent").rename(source)
    restored = _report(write_report([], [], tmp_path / "restored", manifest))
    attempts = restored["inference_performance"]["attempts"]
    assert [a["id"] for a in attempts] == [
        a["id"] for a in missing["inference_performance"]["attempts"]
    ]
    for projection, attempt in zip(
        [
            restored["inference_performance"]["runs"][0],
            restored["eval_performance"]["runs"][0],
        ],
        [attempts[0], attempts[2]],
    ):
        assert projection["attempt_id"] == attempt["id"]
        assert projection["run"] == attempt["run"]
        assert attempt["evidence_status"] == "available"
        assert attempt["declared_status"] == attempt["observed_status"] == "completed"
    assert (
        restored["inference_performance"]["runs"][0]["coverage"][
            "replay_verified_games"
        ]
        is None
    )
    assert manifest.read_bytes() == original


def test_partial_policy_panel_does_not_claim_terminal_outcome(tmp_path: Path) -> None:
    llm = _fixture(tmp_path / "llm")
    policy = _evaluation_fixture(tmp_path / "policy")
    performance = json.loads((policy / "performance.json").read_text())
    performance["counters"]["unattempted_games"] = 1
    _write(policy / "performance.json", performance)
    manifest = _baseline(tmp_path / "baseline.json", llm, policy=policy)
    data = _report(write_report([], [], tmp_path / "out", manifest))
    assert data["eval_performance"]["runs"][0]["observed_status"] == "partial"
    assert (
        data["inference_performance"]["attempts"][2]["declared_status"] == "completed"
    )


def test_manifest_kind_must_match_explicit_receipt_input(tmp_path: Path) -> None:
    llm = _fixture(tmp_path / "llm")
    manifest = _baseline(tmp_path / "baseline.json", llm)
    with pytest.raises(ValueError, match="llm-1.*receipt kind conflicts"):
        write_report([llm], [], tmp_path / "out", manifest)


@pytest.mark.parametrize("available", [False, True])
def test_report_cannot_occupy_declared_evidence_directory(
    tmp_path: Path, available: bool
) -> None:
    root = _fixture(tmp_path / "llm")
    manifest = _baseline(tmp_path / "baseline.json", root)
    if not available:
        root.rename(tmp_path / "retained")
    with pytest.raises(ValueError, match="output must be outside receipt directories"):
        write_report([], [], root / "report", manifest)
    assert not (root / "report").exists()


def _shared_fixture(base: Path, *, llm_failure: bool = False) -> Path:
    """Two policy cells share one injected charge; no workers are executed."""
    llm = _fixture(base / "llm", failure=llm_failure)
    shared = base / "shared"
    trace = base / "diagnostics/trace.json"
    _write(trace, {"traceEvents": []})
    _write(
        shared / "serving.json",
        {
            "identity": "allocation-1",
            "elapsed_seconds": 10.0,
            "schema_version": 2,
            "measurements": {
                "dispatch_queue": {
                    "seconds": 18.0,
                    "population": 6,
                    "maximum_seconds": 3.0,
                    "partitions_elapsed": False,
                },
                "model_load": {"seconds": 2.0, "population": 1, "maximum_seconds": 2.0},
            },
            "counters": {"decisions": 6},
            "spans": [],
            "traces": [str(trace)],
            "unavailable": [],
            "request_p50_seconds": 0.1,
            "request_p95_seconds": 0.2,
            "latency_sample_count": 6,
            "sample_coverage": "controlled fixture",
        },
    )
    _write(
        shared / "execution.json",
        {
            "execution": {"device": "cpu"},
            "cost": {
                "measured_usd": 2.0,
                "estimated_usd": None,
                "allocation_method": "unallocated",
                "unallocated_fraction": 1.0,
            },
        },
    )
    _write(
        shared / "allocation.json",
        {
            "elapsed_seconds": 40.0,
            "measurements": {
                "service_start_to_ready": {
                    "seconds": 1.0,
                    "population": 1,
                    "partitions_elapsed": False,
                },
                "admission_round_trip": {
                    "seconds": 3.0,
                    "population": 1,
                    "partitions_elapsed": False,
                },
                "cleanup": {
                    "seconds": 0.5,
                    "population": 1,
                    "partitions_elapsed": False,
                },
            },
        },
    )
    policies = [_evaluation_fixture(base / name) for name in ("policy", "policy-2")]
    for index, policy in enumerate(policies):
        _write(
            policy / "report-run.json",
            {
                "job": f"policy-{index}",
                "seed": index,
                "serving_path": str(shared / "serving.json"),
            },
        )
    manifest = _baseline(base / "baseline.json", llm, policy=policies[0])
    data = json.loads(manifest.read_text())
    data["attempts"].append(
        {
            **data["attempts"][-1],
            "id": "policy-2",
            "root": str(policies[1]),
            "files": [f.model_dump() for f in inventory(policies[1])],
        }
    )
    trace_file = inventory(trace.parent)[0].model_copy(update={"path": str(trace)})
    data["allocations"] = [
        {
            "identity": "allocation-1",
            "root": str(shared),
            "files": [f.model_dump() for f in [*inventory(shared), trace_file]],
        }
    ]
    _write(manifest, data)
    return manifest


@pytest.mark.parametrize("relative_trace", [False, True])
def test_shared_charge_and_copied_explanatory_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_trace: bool
) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    if relative_trace:
        monkeypatch.chdir(tmp_path)
        path = manifest.parent / "shared/serving.json"
        serving = json.loads(path.read_text())
        serving["traces"] = ["sources/diagnostics/trace.json"]
        _write(path, serving)
        declaration = json.loads(manifest.read_text())
        files = declaration["allocations"][0]["files"]
        files[:] = [f for f in files if f["path"] != "serving.json"]
        files.extend(
            f.model_dump() for f in inventory(path.parent) if f.path == "serving.json"
        )
        _write(manifest, declaration)
    report = write_report([], [], tmp_path / "report", manifest)
    data = _report(report)
    (allocation,) = data["eval_performance"]["allocations"]
    assert allocation["identity"] == "allocation-1"
    assert allocation["included_cells"] == allocation["valid_games"] == 2
    assert allocation["cost"]["measured_usd"] == 2
    assert allocation["games_per_measured_dollar"] == 1
    assert allocation["games_per_allocation_hour"] == 180
    assert all(
        r["allocation_ids"] == ["allocation-1"]
        for r in data["eval_performance"]["runs"]
    )
    assert all(r["games_per_dollar"] is None for r in data["eval_performance"]["runs"])
    assert {Path(f["path"]).name for f in allocation["dependencies"]} == {
        "serving.json",
        "execution.json",
        "allocation.json",
        "trace.json",
    }
    shutil.rmtree(manifest.parent)
    for file in allocation["dependencies"]:
        payload = (report.parent / file["href"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == file["sha256"]
        assert len(payload) == file["bytes"]
    execution = next(
        f for f in allocation["dependencies"] if f["path"].endswith("execution.json")
    )
    assert (
        json.loads((report.parent / execution["href"]).read_text())["cost"]
        == allocation["cost"]
    )
    assert (report.parent / allocation["serving"]["traces"][0]).is_file()


@pytest.mark.parametrize(
    "name",
    ["serving.json", "execution.json", "allocation.json", "../diagnostics/trace.json"],
)
def test_shared_pinned_mutation_fails_before_output(tmp_path: Path, name: str) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    path = manifest.parent / "shared" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="dependency digest/size mismatch"):
        write_report([], [], tmp_path / "out", manifest)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name",
    ["serving.json", "execution.json", "allocation.json", "../diagnostics/trace.json"],
)
def test_shared_missing_restored_preserves_observations_and_identity(
    tmp_path: Path, name: str
) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    declaration = manifest.read_bytes()
    before = _report(write_report([], [], tmp_path / "before", manifest))
    path = (manifest.parent / "shared" / name).resolve()
    payload = path.read_bytes()
    path.unlink()
    missing = _report(write_report([], [], tmp_path / "missing", manifest))
    assert (
        missing["inference_performance"]["attempts"]
        == before["inference_performance"]["attempts"]
    )
    assert missing["eval_performance"]["runs"] == before["eval_performance"]["runs"]
    (allocation,) = missing["eval_performance"]["allocations"]
    assert allocation["identity"] == "allocation-1"
    assert allocation["valid_games"] == 2
    file = next(f for f in allocation["dependencies"] if f["path"] == str(path))
    assert file["status"] == "unavailable" and file["href"] is None
    assert "missing" in file["reason"]
    if name == "execution.json":
        assert allocation["cost"]["measured_usd"] is None
        assert allocation["games_per_measured_dollar"] is None
        assert allocation["games_per_allocation_hour"] == 180
    elif name == "allocation.json":
        assert allocation["games_per_allocation_hour"] is None
        assert allocation["games_per_measured_dollar"] == 1
    elif name == "serving.json":
        assert allocation["serving"] is None
    else:
        assert allocation["serving"]["traces"] == []
    path.write_bytes(payload)
    restored = _report(write_report([], [], tmp_path / "restored", manifest))
    assert restored == before
    assert manifest.read_bytes() == declaration


def test_legacy_policy_manifest_does_not_consume_unpinned_shared_cost(
    tmp_path: Path,
) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    declaration = json.loads(manifest.read_text())
    declaration.pop("allocations")
    _write(manifest, declaration)
    data = _report(write_report([], [], tmp_path / "legacy", manifest))
    (allocation,) = data["eval_performance"]["allocations"]
    assert allocation["serving"] is None
    assert allocation["cost"]["measured_usd"] is None
    assert allocation["games_per_measured_dollar"] is None
    assert all("not pinned" in f["reason"] for f in allocation["dependencies"])
    assert [r["valid_games"] for r in data["eval_performance"]["runs"]] == [1, 1]
    # Direct reports still inventory and copy the same shared sources.
    direct = _report(
        write_report(
            [manifest.parent / "policy", manifest.parent / "policy-2"],
            [],
            tmp_path / "direct",
        )
    )
    assert direct["eval_performance"]["allocations"][0]["cost"]["measured_usd"] == 2


def test_unpinned_cost_does_not_hide_pinned_serving(tmp_path: Path) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    declaration = json.loads(manifest.read_text())
    declaration["allocations"][0]["files"] = [
        f
        for f in declaration["allocations"][0]["files"]
        if f["path"] != "execution.json"
    ]
    _write(manifest, declaration)
    (manifest.parent / "shared/execution.json").write_text("not even valid JSON")
    data = _report(write_report([], [], tmp_path / "out", manifest))
    (allocation,) = data["eval_performance"]["allocations"]
    assert allocation["serving"]["counters"]["decisions"] == 6
    assert allocation["cost"]["measured_usd"] is None
    assert allocation["games_per_measured_dollar"] is None
    assert allocation["games_per_allocation_hour"] == 180
    execution = next(
        f for f in allocation["dependencies"] if f["path"].endswith("execution.json")
    )
    assert execution["href"] is None and "not pinned" in execution["reason"]


def test_shared_copy_time_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    project = evaluation.project_evaluations

    def project_then_mutate(
        directories: list[Path],
        *,
        evidence: EvidenceReader | None = None,
        declared_allocations: dict[Path, BaselineAllocation] | None = None,
    ) -> dict[str, Any]:
        result = project(
            directories,
            evidence=evidence,
            declared_allocations=declared_allocations,
        )
        _write(
            manifest.parent / "shared/execution.json", {"cost": {"measured_usd": 99}}
        )
        return result

    monkeypatch.setattr(evaluation, "project_evaluations", project_then_mutate)
    with pytest.raises(ValueError, match="evidence changed during report build"):
        write_report([], [], tmp_path / "out", manifest)
    assert not (tmp_path / "out/data.json").exists()


def _attempt_observation(root: Path, identity: str, status: str) -> None:
    _write(
        root / "attempt.json",
        {
            "version": "inference-attempt-v1",
            "id": identity,
            "protocol_sha256": "a" * 64,
            "status": status,
            "reason": "controlled whole-attempt outcome",
            "whole_attempt_seconds": 30,
            "collection_seconds": 1,
            "boundary": "controlled parent wall including cleanup",
            "supervision": {"cleanup_complete": status == "completed"},
        },
    )


@pytest.mark.parametrize("kind", ["llm", "evaluation"])
def test_interrupted_collection_projects_only_pinned_attempt_evidence(
    tmp_path: Path, kind: str
) -> None:
    root = (
        _fixture(tmp_path / "source")
        if kind == "llm"
        else _evaluation_fixture(tmp_path / "source")
    )
    _attempt_observation(root, "llm-1", "failed")
    manifest = _baseline(tmp_path / "baseline.json", root, status="failed")
    declaration = json.loads(manifest.read_text())
    declaration["source"] = "a" * 64
    attempt = declaration["attempts"][0]
    raw_pin = "game/config.json" if kind == "llm" else "rows.json"
    attempt.update(
        kind=kind,
        inventory_complete=False,
        files=[f for f in attempt["files"] if f["path"] in {"attempt.json", raw_pin}],
        corpus=[],
    )
    _write(manifest, declaration)
    unpinned = (
        "game/decision-00000/selection.json" if kind == "llm" else "performance.json"
    )
    (root / unpinned).write_text("unfinished, unpinned JSON")
    out = tmp_path / "report"
    data = _report(write_report([], [], out, manifest))
    projected = data["inference_performance"]["attempts"][0]
    assert projected["observed_status"] == "failed"
    assert projected["inventory_complete"] is False
    assert projected["execution"]["whole_attempt_seconds"] == 30
    assert projected["run"] is None and data["runs"] == []
    assert (out / projected["execution_evidence"]).is_file()
    assert len(projected["evidence"]) == 2
    with pytest.raises(ValueError, match="incomplete inventory"):
        write_report(
            [root] if kind == "evaluation" else [],
            [root] if kind == "llm" else [],
            tmp_path / "explicit",
            manifest,
        )
    (root / raw_pin).write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        write_report([], [], tmp_path / "mutated", manifest)
    attempt["status"] = "completed"
    _write(manifest, declaration)
    with pytest.raises(
        ValueError, match="completed attempt requires a complete inventory"
    ):
        write_report([], [], tmp_path / "completed", manifest)


@pytest.mark.parametrize(
    ("game_failed", "attempt_failed"), [(False, True), (True, True), (True, False)]
)
def test_whole_attempt_and_game_outcomes_remain_distinct(
    tmp_path: Path, game_failed: bool, attempt_failed: bool
) -> None:
    root = _fixture(tmp_path / "llm", failure=game_failed)
    status = "failed" if attempt_failed else "completed"
    _attempt_observation(root, "llm-1", status)
    manifest = _baseline(tmp_path / "baseline.json", root, status=status)
    declaration = json.loads(manifest.read_text())
    declaration["source"] = "a" * 64
    _write(manifest, declaration)
    if game_failed and not attempt_failed:
        with pytest.raises(ValueError, match="completed attempt contains failed game"):
            write_report([], [], tmp_path / "report", manifest)
        assert not (tmp_path / "report").exists()
        return
    data = _report(write_report([], [], tmp_path / "report", manifest))
    run = data["inference_performance"]["runs"][0]
    attempt = data["inference_performance"]["attempts"][0]
    assert run["game_observed_status"] == ("failed" if game_failed else "completed")
    assert run["observed_status"] == attempt["observed_status"] == status
    assert run["coverage"]["replay_verified_games"] == int(not game_failed)
    assert run["attempt_id"] == attempt["id"]
    assert run["run"] == attempt["run"]


@pytest.mark.parametrize("whole_attempt", [False, True])
def test_nested_shared_evidence_can_disappear_without_hiding_cell(
    tmp_path: Path,
    whole_attempt: bool,
) -> None:
    manifest = _shared_fixture(tmp_path / "sources")
    shared = manifest.parent / "shared"
    nested = manifest.parent / "policy/allocation"
    shared.rename(nested)
    declaration = json.loads(manifest.read_text())
    declaration["source"] = "a" * 64
    declaration["allocations"][0]["root"] = str(nested)
    for attempt in declaration["attempts"]:
        if attempt["id"] not in {"policy-1", "policy-2"}:
            continue
        root = Path(attempt["root"])
        metadata = json.loads((root / "report-run.json").read_text())
        metadata["serving_path"] = str(nested / "serving.json")
        _write(root / "report-run.json", metadata)
        if whole_attempt:
            _attempt_observation(root, attempt["id"], "completed")
        attempt["files"] = [f.model_dump() for f in inventory(root)]
    _write(manifest, declaration)
    shutil.rmtree(nested)
    data = _report(write_report([], [], tmp_path / "out", manifest))
    assert len(data["eval_performance"]["runs"]) == 2
    (allocation,) = data["eval_performance"]["allocations"]
    assert allocation["identity"] == "allocation-1"
    assert allocation["included_cells"] == 2
    assert allocation["cost"]["measured_usd"] is None
    if whole_attempt:
        for attempt in data["inference_performance"]["attempts"][2:]:
            assert attempt["observed_status"] == "completed"
            assert attempt["execution"]["whole_attempt_seconds"] == 30
            assert (tmp_path / "out" / attempt["execution_evidence"]).is_file()


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_sequential_llm_games_charge_readiness_once_and_keep_game_populations(
    tmp_path: Path,
    mode: str,
) -> None:
    root = tmp_path / "sequence"
    for name in ("game-000", "game-001"):
        _fixture(root / name)
    _write(
        root / "games.json",
        {
            "version": "llm-game-sequence-v1",
            "mode": mode,
            "games": ["game-000", "game-001"],
        },
    )
    _write(root / "readiness.json", {"seconds": 7.0, "ready": True})
    projection = project_llm(root)
    assert projection["observed_status"] == "completed"
    assert projection["coverage"]["completed_games"] == 2
    assert projection["coverage"]["replay_verified_games"] == 2
    assert projection["selection_latency"]["n"] == 6
    assert [g["projection"]["selection_latency"]["n"] for g in projection["games"]] == [
        3,
        3,
    ]
    assert projection["metrics"]["worker_readiness"]["value"] == 7
    assert projection["metrics"]["game_wall"]["value"] == 40
    if mode == "concurrent":
        assert "not campaign wall" in projection["metrics"]["game_wall"]["boundary"]
        assert "overlap" in projection["notes"][0]
    report = write_report([], [root], tmp_path / "report")
    assert report.is_file()
    # The session stopped before its second planned game: retained first-game
    # success survives, but cannot establish completed many-game execution.
    shutil.rmtree(root / "game-001")
    partial = project_llm(root)
    assert partial["observed_status"] == "partial"
    assert partial["coverage"]["planned_games"] == 2
    assert partial["coverage"]["observed_games"] == 1
    _fixture(root / "game-001")
    _write(root / "game-001/game/runtime.json", {"runtime": "different-device"})
    with pytest.raises(ValueError, match="runtime identity"):
        project_llm(root)


def test_device_samples_retain_uuid_and_declared_interval(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "llm")
    (root / "gpu-memory.csv").write_text(
        "timestamp, uuid, name, memory.total [MiB], memory.used [MiB], driver_version\n"
        "2026/10/26 12:00:00, GPU-fixture, fixture, 100, 10, test\n"
        "2026/10/26 12:00:01, GPU-fixture, fixture, 100, 20, test\n"
    )
    _write(root / "device-sampling.json", {"interval_seconds": 1, "samples": 2})
    memory = project_llm(root)["device_memory"]
    assert memory["device_uuid"] == "GPU-fixture"
    assert memory["sample_interval_seconds"] == 1
    assert memory["peak_used_bytes"] == 20 * 1024**2


def _matchup_fixture(root: Path) -> Path:
    """Two controlled matchup games and one missing planned game."""
    for i in range(2):
        game = _fixture(root / f"game-{i:03d}") / "game"
        path = game / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            mode="qwen-v-checkpoint",
            winner=i,
            opponent_latencies=[0.01, 0.03],
            observations={
                "llm_wait_and_dispatch": 12.0,
                "native_command": 3.0,
                "policy_inference": 0.04,
                "unattributed": 4.96,
            },
            llm_phases={"generation_seconds": 10.0},
            reasoning_tokens=30,
            answer_tokens=3,
            output_tokens=40,
            token_split_attempts=5,
            generation_attempts=5,
        )
        _write(path, summary)
        for selection in game.glob("decision-*/selection.json"):
            data = json.loads(selection.read_text())
            data["config"]["enable_thinking"] = True
            _write(selection, data)
    _write(
        root / "games.json",
        {
            "version": "llm-game-sequence-v1",
            "games": ["game-000", "game-001", "game-002"],
        },
    )
    _write(root / "readiness.json", {"seconds": 4.0, "policy_cpu_seconds": 0.2})
    return root


def test_same_game_matchup_keeps_exclusive_and_nested_timers_separate(
    tmp_path: Path,
) -> None:
    root = _matchup_fixture(tmp_path / "sequence")
    projection = project_llm(root)
    matchup = projection["matchup"]
    assert projection["observed_status"] == "partial"
    assert projection["coverage"]["planned_games"] == 3
    assert (matchup["wins"], matchup["losses"], matchup["valid"]) == (1, 1, 2)
    assert sum(matchup["exclusive"].values()) == matchup["game_seconds"] == 40
    assert matchup["llm_phases"]["generation_seconds"] == 20
    assert matchup["policy_latency"]["n"] == 4
    assert matchup["policy_latency"]["p50"] == 0.01
    assert projection["metrics"]["worker_readiness"]["value"] == 4
    assert projection["metrics"]["policy_readiness"]["value"] == 0.2
    path = write_report([], [root], tmp_path / "report")
    assert _report(path)["question"] == "Where does evaluation time go?"
    assert _report(path)["inference_performance"]["runs"][0]["matchup"] == matchup
