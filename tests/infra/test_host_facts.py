"""The receipt's machine facts read cgroup limits as a container reports them."""

from pathlib import Path

import pytest

from manabot.infra.host import (
    HostFacts,
    MemoryLimit,
    _cpu_model,
    _cpu_quota,
    memory_limit,
)


@pytest.mark.parametrize(
    ("text", "quota"),
    [("600000 100000\n", 6.0), ("max 100000\n", None), ("", None)],
)
def test_cpu_quota_reads_cgroup_limits(
    tmp_path: Path, text: str, quota: float | None
) -> None:
    path = tmp_path / "cpu.max"
    path.write_text(text)
    assert _cpu_quota(path) == quota


def test_missing_cgroup_file_means_no_limit(tmp_path: Path) -> None:
    assert _cpu_quota(tmp_path / "absent") is None


def test_cpu_model_reads_the_first_processor(tmp_path: Path) -> None:
    path = tmp_path / "cpuinfo"
    path.write_text("processor : 0\nmodel name : AMD EPYC 7763\nmodel name : other\n")
    assert _cpu_model(path) == "AMD EPYC 7763"


def test_cpu_model_is_absent_without_cpuinfo(tmp_path: Path) -> None:
    assert _cpu_model(tmp_path / "absent") is None


def _limit(
    root: Path,
    v2: str | None = None,
    v1: str | None = None,
    environ: bytes | None = None,
) -> MemoryLimit | None:
    paths = (root / "memory.max", root / "limit_in_bytes", root / "environ")
    for path, text in zip(paths[:2], (v2, v1), strict=True):
        if text is not None:
            path.write_text(text)
    if environ is not None:
        paths[2].write_bytes(environ)
    return memory_limit(*paths)


def test_memory_limit_reads_each_source(tmp_path: Path) -> None:
    assert _limit(tmp_path, v2="1000\n") == MemoryLimit(bytes=1000, source="cgroup-v2")
    (tmp_path / "memory.max").unlink()
    assert _limit(tmp_path, v1="2000\n") == MemoryLimit(bytes=2000, source="cgroup-v1")
    (tmp_path / "limit_in_bytes").unlink()
    assert _limit(
        tmp_path, environ=b"PATH=/bin\0RUNPOD_MEM_GB=48\0HOME=/root\0"
    ) == MemoryLimit(bytes=48 * 10**9, source="runpod-env")


def test_memory_limit_takes_the_smallest_source(tmp_path: Path) -> None:
    assert _limit(
        tmp_path, v2="3000\n", v1="2000\n", environ=b"RUNPOD_MEM_GB=48\0"
    ) == MemoryLimit(bytes=2000, source="cgroup-v1")


@pytest.mark.parametrize(
    ("v2", "v1", "environ"),
    [
        (None, None, None),
        ("max\n", "9223372036854771712\n", b"PATH=/bin\0"),
        ("", "not a number", b"RUNPOD_MEM_GB=unknown\0"),
    ],
)
def test_memory_limit_is_absent_when_nothing_states_one(
    tmp_path: Path, v2: str | None, v1: str | None, environ: bytes | None
) -> None:
    assert _limit(tmp_path, v2=v2, v1=v1, environ=environ) is None


def test_absent_memory_limit_leaves_retained_receipts_unchanged() -> None:
    retained = {
        "cpu_model": "AMD EPYC 7763",
        "logical_cpus": 64,
        "schedulable_cpus": 6,
        "cpu_quota": None,
    }
    assert HostFacts.model_validate(retained).model_dump() == retained
