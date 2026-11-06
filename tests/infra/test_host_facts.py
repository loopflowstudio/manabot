"""The receipt's machine facts read cgroup limits as a container reports them."""

from pathlib import Path

import pytest

from manabot.infra.host import _cpu_model, _cpu_quota


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
