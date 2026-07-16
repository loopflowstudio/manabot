from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "bench_branching", ROOT / "scripts/bench_branching.py"
)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def worker(index: int, checksum: str) -> dict:
    return {
        "seed_path": {"worker_index": index, "worker_seed": 100 + index},
        "metrics": {
            "result_checksum": checksum,
            "elapsed_seconds": 999.0,
        },
        "rss_peak_bytes": 123,
    }


def test_repeat_checksum_is_ordered_but_excludes_timing_and_rss() -> None:
    first = [worker(1, "b"), worker(0, "a")]
    second = [worker(0, "a"), worker(1, "b")]
    second[0]["metrics"]["elapsed_seconds"] = 0.001
    second[0]["rss_peak_bytes"] = 999_999
    assert bench.deterministic_repeat_checksum("cell", 7, first) == (
        bench.deterministic_repeat_checksum("cell", 7, second)
    )


def test_repeat_checksum_changes_for_logical_outcome() -> None:
    first = [worker(0, "a")]
    second = [worker(0, "different")]
    assert bench.deterministic_repeat_checksum("cell", 7, first) != (
        bench.deterministic_repeat_checksum("cell", 7, second)
    )


def test_artifact_hash_omits_only_its_own_field() -> None:
    payload = {"schema": "x", "value": 1, "artifact_sha256": "old"}
    original = bench.artifact_hash(payload)
    payload["artifact_sha256"] = "new"
    assert bench.artifact_hash(payload) == original
    payload["value"] = 2
    assert bench.artifact_hash(payload) != original


def test_percentile_uses_nearest_rank() -> None:
    assert bench.percentile([4, 1, 3, 2], 0.50) == 2
    assert bench.percentile([4, 1, 3, 2], 0.99) == 4


class FakeProcess:
    def __init__(self, *, pid: int = 123, stdin: object | None = None) -> None:
        self.pid = pid
        self.stdin = stdin
        self.returncode = None

    def poll(self) -> None:
        return None


def test_sampler_failure_invalidates_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(_pid: int) -> None:
        raise psutil.AccessDenied(pid=_pid)

    monkeypatch.setattr(bench.psutil, "Process", denied)
    with pytest.raises(RuntimeError, match="RSS sampler cannot inspect"):
        bench.sample_process_rss(
            [FakeProcess()], sequence=0, phase="baseline", offset_seconds=0.0
        )


def test_barrier_failure_invalidates_measurement() -> None:
    broken_stdin = SimpleNamespace(
        write=lambda _value: (_ for _ in ()).throw(BrokenPipeError()),
        flush=lambda: None,
        close=lambda: None,
    )
    with pytest.raises(RuntimeError, match="start barrier failed"):
        bench.release_barrier([FakeProcess(stdin=broken_stdin)])


def load_artifact() -> dict:
    payload = json.loads(bench.DEFAULT_RAW.read_text())
    identity = bench.source_identity(check_clean=False)
    payload["run"].setdefault("source_digest_method", bench.SOURCE_DIGEST_METHOD)
    payload["run"].setdefault("source_paths", list(identity.paths))
    payload["run"].setdefault("measurement_code_revision", "0" * 40)
    payload["build"].setdefault("binary_sha256", "0" * 64)
    loopflow = payload["run"]["loopflow"]
    loopflow.setdefault("status_snapshot", None)
    loopflow.setdefault(
        "status_unavailable_reason",
        None
        if loopflow["status_snapshot"] is not None
        else "legacy fixture has no local Loopflow snapshot",
    )
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    return payload


@pytest.mark.parametrize(
    "path",
    [
        ("run", "started_at"),
        ("run", "argv"),
        ("run", "source_digest_method"),
        ("run", "source_paths"),
        ("run", "measurement_code_revision"),
        ("build", "binary_sha256"),
        ("cells", 0, "repeats", 0, "process_group_id"),
        ("cells", 0, "repeats", 0, "commands"),
        ("cells", 0, "repeats", 0, "rss_samples"),
        ("cells", 0, "repeats", 0, "sampler"),
    ],
)
def test_verifier_rejects_missing_contract_fields(path: tuple[str | int, ...]) -> None:
    payload = load_artifact()
    cursor = payload
    for component in path[:-1]:
        cursor = cursor[component]
    del cursor[path[-1]]
    payload["run"]["canonical"] = False
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    with pytest.raises(RuntimeError, match="missing required"):
        bench.verify(payload)


def test_verifier_rejects_worker_failure() -> None:
    payload = load_artifact()
    repeat = payload["cells"][0]["repeats"][0]
    repeat["rss_samples"][-1]["workers"][0]["returncode"] = -9
    payload["run"]["canonical"] = False
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    with pytest.raises(RuntimeError, match="worker failure"):
        bench.verify(payload)


def test_verifier_rejects_reused_process_group() -> None:
    payload = load_artifact()
    first = payload["cells"][0]["repeats"][0]
    second = payload["cells"][1]["repeats"][0]
    reused = first["process_group_id"]
    second["process_group_id"] = reused
    for ready in second["ready"]:
        ready["process_group_id"] = reused
    second["ready"][0]["pid"] = reused
    for sample in second["rss_samples"]:
        sample["workers"][0]["pid"] = reused
    payload["run"]["canonical"] = False
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    with pytest.raises(RuntimeError, match="reused across cell/repeat"):
        bench.verify(payload)


def test_expected_worker_simulations_is_a_times_l_times_r() -> None:
    dimensions = {
        "actors_per_worker": 8,
        "worlds": 16,
        "rollouts_per_world": 2,
    }
    assert bench.expected_worker_simulations(dimensions, action_count=6) == 1_536


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def commit_all(root: Path, message: str) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-m", message)


def make_source_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    files = {
        "content/semantic/v1/coverage.evidence.json": "{}\n",
        "content/semantic/v1/generated/two_deck.ir.json": "{}\n",
        "content/semantic/v1/two_deck.source.json": "{}\n",
        "docs/benchmarks/search-branching-contract-v1.md": "# contract\n",
        "docs/benchmarks/dense-page-cow-prereg-v1.md": "# prereg\n",
        "managym/Cargo.lock": "# lock\n",
        "managym/Cargo.toml": (
            '[package]\nname = "source-fixture"\nversion = "0.1.0"\n'
        ),
        "managym/src/lib.rs": (
            'const IR: &str = include_str!("../../content/semantic/v1/generated/two_deck.ir.json");\n'
            'const SOURCE: &[u8] = include_bytes!("../../content/semantic/v1/two_deck.source.json");\n'
            'const COVERAGE: &[u8] = include_bytes!("../../content/semantic/v1/coverage.evidence.json");\n'
        ),
        "managym/tests/smoke.rs": "#[test]\nfn smoke() {}\n",
        "pyproject.toml": '[project]\nname = "source-fixture"\nversion = "0.1.0"\n',
        "scripts/bench_branching.py": "# harness\n",
        "tests/bench/test_branching_benchmark.py": "# tests\n",
        "uv.lock": "version = 1\n",
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Branching Test")
    git(root, "config", "user.email", "branching@example.test")
    commit_all(root, "fixture")
    return root


def test_selected_git_tree_digest_is_checkout_independent(tmp_path: Path) -> None:
    root = make_source_repo(tmp_path)
    noise = root / ".claude/worktrees/agent/noise.txt"
    noise.parent.mkdir(parents=True)
    noise.write_text("incidental worktree bytes\n")
    linked = tmp_path / "linked"
    git(root, "worktree", "add", "-b", "linked", str(linked))

    assert bench.source_identity(root=root) == bench.source_identity(root=linked)
    assert (linked / ".git").is_file()


def test_unrelated_tracked_commit_does_not_change_digest(tmp_path: Path) -> None:
    root = make_source_repo(tmp_path)
    baseline = bench.source_identity(root=root)
    (root / "notes.md").write_text("unrelated\n")
    commit_all(root, "unrelated")
    assert bench.source_identity(root=root) == baseline


@pytest.mark.parametrize("relative", bench.COMPILE_TIME_SOURCE_INPUTS)
def test_external_compile_time_input_mutation_changes_digest(
    tmp_path: Path, relative: str
) -> None:
    root = make_source_repo(tmp_path)
    baseline = bench.source_identity(root=root)
    target = root / relative
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="admitted source inputs are dirty"):
        bench.source_identity(root=root)
    commit_all(root, f"mutate {relative}")
    assert bench.source_identity(root=root).sha256 != baseline.sha256


@pytest.mark.parametrize("state", ["unstaged", "staged", "untracked", "ignored"])
def test_dirty_admitted_input_fails_closed(tmp_path: Path, state: str) -> None:
    root = make_source_repo(tmp_path)
    if state == "unstaged":
        (root / "scripts/bench_branching.py").write_text("# dirty\n")
    elif state == "staged":
        (root / "scripts/bench_branching.py").write_text("# staged\n")
        git(root, "add", "scripts/bench_branching.py")
    elif state == "untracked":
        (root / "managym/src/untracked.rs").write_text("// untracked\n")
    else:
        (root / ".gitignore").write_text("managym/src/ignored.rs\n")
        commit_all(root, "ignore fixture")
        (root / "managym/src/ignored.rs").write_text("// ignored\n")
    with pytest.raises(RuntimeError, match="admitted source inputs are dirty"):
        bench.source_identity(root=root)


def test_new_external_compile_time_input_fails_inventory(tmp_path: Path) -> None:
    root = make_source_repo(tmp_path)
    extra = root / "content/semantic/v1/new.json"
    extra.write_text("{}\n")
    library = root / "managym/src/lib.rs"
    library.write_text(
        library.read_text()
        + 'const NEW: &str = include_str!("../../content/semantic/v1/new.json");\n'
    )
    with pytest.raises(RuntimeError, match="input inventory mismatch"):
        bench.production_compile_time_inputs(root=root)


def test_nonliteral_compile_time_input_fails_inventory(tmp_path: Path) -> None:
    root = make_source_repo(tmp_path)
    library = root / "managym/src/lib.rs"
    library.write_text(
        library.read_text()
        + 'const NEW: &str = include_str!(concat!("../../content", "/new.json"));\n'
    )
    with pytest.raises(
        RuntimeError, match="unresolvable production compile-time include"
    ):
        bench.production_compile_time_inputs(root=root)


def test_new_cargo_build_script_must_be_admitted(tmp_path: Path) -> None:
    root = make_source_repo(tmp_path)
    (root / "managym/build.rs").write_text("fn main() {}\n")
    with pytest.raises(RuntimeError, match="build script is not admitted"):
        bench.production_compile_time_inputs(root=root)


def test_verifier_rejects_wrong_source_method_and_path_closure() -> None:
    payload = load_artifact()
    payload["run"]["canonical"] = False
    payload["run"]["source_digest_method"] = "filesystem-walk-v0"
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    with pytest.raises(RuntimeError, match="source digest method mismatch"):
        bench.verify(payload)

    payload = load_artifact()
    payload["run"]["canonical"] = False
    payload["run"]["source_paths"] = payload["run"]["source_paths"][1:]
    payload["artifact_sha256"] = bench.artifact_hash(payload)
    with pytest.raises(RuntimeError, match="source path closure mismatch"):
        bench.verify(payload)


def test_matrix_projection_ignores_only_physical_measurements() -> None:
    payload = load_artifact()
    physical = copy.deepcopy(payload)
    physical["cells"][2]["repeats"][0]["workers"][0]["metrics"][
        "elapsed_seconds"
    ] = 0.001
    physical["cells"][2]["repeats"][0]["workers"][0]["metrics"][
        "cow_bytes"
    ] = 4096
    physical["cells"][2]["repeats"][0]["rss_peak_bytes"] += 1234
    assert bench.matched_matrix_projection(payload) == bench.matched_matrix_projection(
        physical
    )

    physical["cells"][2]["repeats"][0]["workers"][0]["metrics"][
        "result_checksum"
    ] = "logical-drift"
    assert bench.matched_matrix_projection(payload) != bench.matched_matrix_projection(
        physical
    )


def decision_candidate(
    *,
    flat_speed: float,
    retained_speed: float,
    flat_rss: int = 100,
    retained_rss: int = 100,
    single_delta: int = 100,
    saturated_delta: int = 100,
) -> dict:
    cells = []
    for cell in bench.WHOLE_CELLS:
        retained = cell.startswith("retained")
        cells.append(
            {
                "id": cell,
                "summary": {
                    "simulations_per_second": retained_speed
                    if retained
                    else flat_speed,
                    "root_latency_seconds_p99": 1.0,
                    "rss_peak_bytes": retained_rss if retained else flat_rss,
                    "rss_peak_delta_bytes": saturated_delta
                    if cell == "retained-saturated-16-v1"
                    else single_delta,
                },
            }
        )
    return {"cells": cells}


def test_decision_thresholds_select_general_hybrid_or_baseline() -> None:
    full = decision_candidate(flat_speed=100, retained_speed=100)
    undo = decision_candidate(flat_speed=121, retained_speed=100)
    page = decision_candidate(
        flat_speed=95,
        retained_speed=95,
        retained_rss=80,
        single_delta=70,
        saturated_delta=50,
    )
    payloads = {
        bench.FULL_CLONE_DRIVER: full,
        bench.CLONE_PLUS_UNDO_DRIVER: undo,
        bench.PAGE_COW_DRIVER: page,
    }
    assert bench.decision_outcome(payloads)["page_general"] is True

    payloads[bench.PAGE_COW_DRIVER] = decision_candidate(
        flat_speed=80,
        retained_speed=95,
        retained_rss=80,
        single_delta=70,
        saturated_delta=50,
    )
    assert "hybrid" in bench.decision_outcome(payloads)["selection"]

    payloads[bench.PAGE_COW_DRIVER] = decision_candidate(
        flat_speed=80, retained_speed=80
    )
    payloads[bench.CLONE_PLUS_UNDO_DRIVER] = decision_candidate(
        flat_speed=100, retained_speed=100
    )
    assert bench.decision_outcome(payloads)["selection"].startswith("retain")


def test_loopflow_metadata_selects_ambient_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "projects": [
            {
                "tasks": [
                    {
                        "task": {"identifier": "W2-264"},
                        "reference": {
                            "workspace": {
                                "worktree": str(bench.ROOT),
                                "branch": "jack-heart/reproducible",
                            }
                        },
                        "runtime": {
                            "session_id": "ts_current",
                            "project_session_id": "ps_search",
                            "status": "running",
                            "reason": "working",
                        },
                    }
                ]
            }
        ]
    }

    def fake_command(command: list[str], *, required: bool = True) -> str | None:
        if command[:2] == ["lf", "status"]:
            return json.dumps(status)
        if command[:2] == ["git", "symbolic-ref"]:
            return "fallback-branch"
        raise AssertionError(command)

    monkeypatch.setenv("LF_TASK_SESSION_ID", "ts_current")
    monkeypatch.setattr(bench, "command_output", fake_command)
    metadata = bench.loopflow_metadata()
    assert metadata["task"] == "W2-264"
    assert metadata["task_session"] == "ts_current"
    assert metadata["branch"] == "jack-heart/reproducible"
    assert metadata["status_unavailable_reason"] is None
