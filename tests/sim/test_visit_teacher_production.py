"""Production contract and decision checks for the INT-4 iteration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.runners import run_visit_teacher_production as production_runner
from experiments.runners.run_visit_teacher_production import (
    ResourceCapReached,
    _admission,
    _atomic_json,
    _control_receipt,
    _execute_job,
    _load_bound_stage_result,
    _load_contracts,
    _production_runtime,
    _read_resource_ledger,
    _resource_ledger_receipt,
    _run_stage,
    _validate_production_runtime,
    _verify,
    _verify_job_reference,
)
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.teacher1_evidence import ContractError, file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONTRACT = (
    REPO_ROOT / "experiments/contracts/int-4-visit-teacher-iteration-v1.json"
)
PRODUCTION_CONTRACT = (
    REPO_ROOT / "experiments/contracts/int-4-visit-teacher-production-v1.json"
)
SCENARIOS = [
    "s1_counter_the_bomb",
    "s2_hold_the_wipe",
    "s3_bolt_the_threat",
    "s4_race_vs_block",
    "s5_hold_up_quench",
]


def _job(result: dict) -> dict:
    return {"result": result}


def _cell(*, wins: int, games: int = 48, draws: int = 0) -> dict:
    return {
        "num_games": float(games),
        "wins": float(wins),
        "win_rate": wins / games,
        "draws_or_caps": float(draws),
    }


def _profile() -> dict:
    return {"training_seeds": [197, 419, 887]}


def _production() -> dict:
    return {
        "competencies": {"scenarios": SCENARIOS},
        "admission_gates": {
            "median_win_rate": 0.55,
            "winning_training_seeds": 2,
            "per_seed_win_rate": 0.5,
            "competency_max_regression": 0.1,
        },
        "resource_accounting": {
            "workers": 1,
            "wall_hours": 1,
            "core_hours": 1,
            "artifact_bytes": 64 * 1024 * 1024,
        },
    }


def _resources(*, wall: float = 0.01) -> dict:
    return {
        "wall_seconds": wall,
        "user_cpu_seconds": 0.01,
        "system_cpu_seconds": 0.01,
        "peak_rss_bytes": 1024,
    }


def _arena(
    *, search_wins: list[int], policy_only_wins: list[int], policy_value_wins: list[int]
) -> dict:
    competencies = {}
    for scenario in SCENARIOS:
        block = {"control-policy-value": {"correct_rate": 0.5}}
        for seed in _profile()["training_seeds"]:
            block[f"visit-policy-value-{seed}"] = {"correct_rate": 0.5}
            block[f"visit-policy-value-search-{seed}"] = {"correct_rate": 0.5}
        competencies[scenario] = block
    per_seed = {}
    for index, seed in enumerate(_profile()["training_seeds"]):
        # The searched candidate is the villain in this paired cell.
        student_wins = 48 - search_wins[index]
        per_seed[str(seed)] = {
            "round_robin": {
                "student-vs-student+search": _job(_cell(wins=student_wins))
            },
            "controls": {
                "student-vs-control-policy_only": _job(
                    _cell(wins=policy_only_wins[index])
                ),
                "student-vs-control-policy_value": _job(
                    _cell(wins=policy_value_wins[index])
                ),
            },
        }
    return {"per_seed": per_seed, "competencies": _job(competencies)}


def test_production_contract_binds_frozen_iteration_and_runtime() -> None:
    contract, _, _, _, production, _ = _load_contracts(
        BASE_CONTRACT, PRODUCTION_CONTRACT
    )
    runtime = _production_runtime(seed=int(contract["runtime_seed"]))
    _validate_production_runtime(production, runtime)
    assert production["base_iteration"]["profile"] == "iteration"
    assert production["calibration"]["max_realized_p50_gap"] == 0.1
    assert production["competencies"]["runs_per_agent_scenario"] == 100


def test_control_receipt_rejects_wrong_hash_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ContractError, match="SHA-256"):
        _control_receipt(
            path,
            {
                "arm": "policy_only",
                "sha256": "0" * 64,
                "deterministic": True,
            },
        )


def test_control_receipt_binds_checkpoint_arm_and_model(tmp_path: Path) -> None:
    path = tmp_path / "policy-value.pt"
    obs_space = ObservationSpace()
    agent = Agent(obs_space, AgentHypers())
    save_bc_checkpoint(agent, obs_space, path, extra={"arm": "policy_value"})
    receipt = _control_receipt(
        path,
        {
            "arm": "policy_value",
            "sha256": file_sha256(path),
            "deterministic": True,
        },
    )
    assert receipt["arm"] == "policy_value"
    assert receipt["parameter_count"] > 0
    with pytest.raises(ContractError, match="metadata arm"):
        _control_receipt(
            path,
            {
                "arm": "policy_only",
                "sha256": file_sha256(path),
                "deterministic": True,
            },
        )


def test_admission_prefers_student_search_when_complete_gates_pass() -> None:
    arena = _arena(
        search_wins=[27, 26, 28],
        policy_only_wins=[30, 30, 30],
        policy_value_wins=[30, 30, 30],
    )
    result = _admission(arena, _profile(), _production())
    assert result["decision"] == "admit_student_search"
    assert result["disposition"] == "continue"


def test_admission_falls_back_to_student_then_honest_failure() -> None:
    student = _arena(
        search_wins=[20, 21, 22],
        policy_only_wins=[27, 26, 28],
        policy_value_wins=[28, 27, 26],
    )
    assert _admission(student, _profile(), _production())["decision"] == "admit_student"

    failed = _arena(
        search_wins=[20, 21, 22],
        policy_only_wins=[20, 21, 22],
        policy_value_wins=[21, 20, 22],
    )
    result = _admission(failed, _profile(), _production())
    assert result["decision"] == "prototype_failure"
    assert result["disposition"] == "revise"


def test_completed_child_job_is_reused_only_for_exact_inputs(tmp_path: Path) -> None:
    job = {
        "kind": "play_cell",
        "hero": {"kind": "random"},
        "villain": {"kind": "random"},
        "blocks": [{"id": "focused", "seed": 17, "games": 1}],
    }
    first = _execute_job(tmp_path, "cached", job, _production())
    ledger_entries = len(_read_resource_ledger(tmp_path))
    second = _execute_job(tmp_path, "cached", job, _production())
    assert second["output_sha256"] == first["output_sha256"]
    assert len(_read_resource_ledger(tmp_path)) == ledger_entries

    with pytest.raises(ContractError, match="different inputs"):
        _execute_job(
            tmp_path,
            "cached",
            {**job, "blocks": [{"id": "changed", "seed": 18, "games": 1}]},
            _production(),
        )


def test_spawned_matchup_job_records_result_and_resources(tmp_path: Path) -> None:
    result = _execute_job(
        tmp_path,
        "random-mirror",
        {
            "kind": "play_cell",
            "hero": {"kind": "random"},
            "villain": {"kind": "random"},
            "blocks": [{"id": "focused", "seed": 17, "games": 2}],
        },
        _production(),
    )
    assert result["result"]["num_games"] == 2
    assert result["result"]["draws_or_caps"] == 0
    assert result["resources"]["wall_seconds"] > 0
    assert result["resources"]["peak_rss_bytes"] > 0
    assert result["output_sha256"] == file_sha256(tmp_path / result["result_path"])


def test_mid_stage_cap_prevents_launching_another_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = {
        "kind": "play_cell",
        "hero": {"kind": "random"},
        "villain": {"kind": "random"},
        "blocks": [{"id": "focused", "seed": 31, "games": 1}],
    }
    monkeypatch.setattr(
        production_runner,
        "_resource_receipt",
        lambda before, started: _resources(wall=3601.0),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {"status": "running", "stages": {}}
    _atomic_json(manifest_path, manifest)

    def stage_work() -> dict:
        _execute_job(tmp_path, "first", job, _production())
        _execute_job(tmp_path, "second", job, _production())
        return {"unexpected": True}

    with pytest.raises(ResourceCapReached, match="cumulative cap reached"):
        _run_stage(
            manifest,
            manifest_path,
            tmp_path,
            _production(),
            "teacher",
            {"input": "frozen"},
            stage_work,
        )
    events = _read_resource_ledger(tmp_path)
    assert [(event["scope"], event["status"]) for event in events] == [
        ("job", "completed"),
        ("stage", "failed"),
    ]
    assert manifest["status"] == "inconclusive_resource_cap"
    assert not list((tmp_path / "jobs").glob("second.attempt-*.result.json"))
    event_count = len(events)
    with pytest.raises(ResourceCapReached, match="cumulative cap reached"):
        _run_stage(
            manifest,
            manifest_path,
            tmp_path,
            _production(),
            "teacher",
            {"input": "frozen"},
            stage_work,
        )
    assert len(_read_resource_ledger(tmp_path)) == event_count


def test_failed_attempt_remains_charged_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_child(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(production_runner.subprocess, "run", fail_child)
    monkeypatch.setattr(
        production_runner,
        "_resource_receipt",
        lambda before, started: _resources(wall=3601.0),
    )
    job = {"kind": "play_cell", "hero": {"kind": "random"}}
    with pytest.raises(RuntimeError, match="failed"):
        _execute_job(tmp_path, "failed", job, _production())
    assert _read_resource_ledger(tmp_path)[0]["status"] == "failed"
    with pytest.raises(ResourceCapReached, match="cumulative cap reached"):
        _execute_job(tmp_path, "failed", job, _production())
    assert calls == 1


def test_job_result_tampering_breaks_immutable_digest(tmp_path: Path) -> None:
    reference = _execute_job(
        tmp_path,
        "tamper-job",
        {
            "kind": "play_cell",
            "hero": {"kind": "random"},
            "villain": {"kind": "random"},
            "blocks": [{"id": "focused", "seed": 43, "games": 1}],
        },
        _production(),
    )
    result_path = tmp_path / reference["result_path"]
    payload = json.loads(result_path.read_text())
    payload["result"]["wins"] = 999
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="digest binding drifted"):
        _verify_job_reference(tmp_path, reference)


@pytest.mark.parametrize("target", ["stage_file", "manifest_copy"])
def test_stage_result_and_manifest_tampering_break_binding(
    tmp_path: Path, target: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {"stages": {}}
    _atomic_json(manifest_path, manifest)
    _run_stage(
        manifest,
        manifest_path,
        tmp_path,
        _production(),
        "teacher",
        {"input": "frozen"},
        lambda: {"gate": {"passed": True}},
    )
    if target == "stage_file":
        path = tmp_path / manifest["stages"]["teacher"]["result_path"]
        path.write_text('{"gate":{"passed":false}}\n')
    else:
        manifest["stages"]["teacher"]["result"]["gate"]["passed"] = False
    with pytest.raises(ContractError, match="result binding drifted"):
        _load_bound_stage_result(manifest, tmp_path, "teacher")


def test_resume_rejects_a_truncated_resource_ledger(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {"stages": {}}
    _atomic_json(manifest_path, manifest)
    _run_stage(
        manifest,
        manifest_path,
        tmp_path,
        _production(),
        "first",
        {"input": "frozen"},
        lambda: {"complete": True},
    )
    (tmp_path / "resource-ledger.jsonl").write_text("")
    with pytest.raises(ContractError, match="shorter than its persisted receipt"):
        _run_stage(
            manifest,
            manifest_path,
            tmp_path,
            _production(),
            "second",
            {"input": "frozen"},
            lambda: {"must_not_run": True},
        )


def test_verify_reads_bound_artifacts_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    runtime = {
        "production_source_sha256": "source",
        "experience_content_hash": "content",
        "asset_manifest_hash": "assets",
    }
    controls = {"policy_only": {"sha256": "a"}, "policy_value": {"sha256": "b"}}
    manifest = {
        "schema_version": 1,
        "experiment": "int-4-visit-teacher-production-v1",
        "status": "running",
        "contract_sha256": "contract",
        "profile_sha256": "profile",
        "production_contract_sha256": "production",
        "runtime": runtime,
        "controls": controls,
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)
    audit_path = tmp_path / "audit.json"
    _atomic_json(audit_path, {})
    study_path = tmp_path / "study.json"
    _atomic_json(study_path, {})
    replay = {"passed": True}
    teacher_gate = {"passed": True}
    admission = {"decision": "prototype_failure"}
    stage_results = {
        "calibration": {},
        "teacher": {"gate": teacher_gate},
        "dataset": {
            "shards": [],
            "audit_path": str(audit_path),
            "audit_sha256": file_sha256(audit_path),
            "replay": replay,
        },
        "training": {"checkpoints": {}},
        "arena": {"admission": admission},
        "study": {"path": str(study_path), "sha256": file_sha256(study_path)},
    }
    for name, result in stage_results.items():
        _run_stage(
            manifest,
            manifest_path,
            tmp_path,
            _production(),
            name,
            {"name": name},
            lambda result=result: result,
        )
    manifest["status"] = "completed"
    manifest["admission"] = admission
    manifest["resource_ledger"] = _resource_ledger_receipt(tmp_path)
    _atomic_json(manifest_path, manifest)

    def generation_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verification generated new evidence")

    monkeypatch.setattr(production_runner, "_execute_job", generation_forbidden)
    monkeypatch.setattr(
        production_runner.iteration, "_run_dataset", generation_forbidden
    )
    monkeypatch.setattr(
        production_runner.iteration, "_run_training", generation_forbidden
    )
    monkeypatch.setattr(production_runner.iteration, "_run_study", generation_forbidden)
    monkeypatch.setattr(
        production_runner, "replay_teacher_trajectories", lambda *a, **k: replay
    )
    monkeypatch.setattr(production_runner, "receipt_dict", lambda value: value)
    monkeypatch.setattr(
        production_runner,
        "StudyArtifact",
        SimpleNamespace(model_validate_json=lambda value: None),
    )
    monkeypatch.setattr(
        production_runner.iteration, "_validate_study_in_rust", lambda path: None
    )
    monkeypatch.setattr(
        production_runner, "_teacher_gate", lambda *a, **k: teacher_gate
    )
    monkeypatch.setattr(production_runner, "_admission", lambda *a, **k: admission)
    before = {
        str(path.relative_to(tmp_path)): file_sha256(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = _verify(
        tmp_path,
        "contract",
        {"sampled_search_roots": []},
        "profile",
        "production",
        runtime,
        controls,
        _production(),
    )
    after = {
        str(path.relative_to(tmp_path)): file_sha256(path)
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "verification.json"
    }
    assert result["verified"] is True
    assert after == before
