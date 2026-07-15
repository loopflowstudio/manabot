from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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
    return json.loads(bench.DEFAULT_RAW.read_text())


@pytest.mark.parametrize(
    "path",
    [
        ("run", "started_at"),
        ("run", "argv"),
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
