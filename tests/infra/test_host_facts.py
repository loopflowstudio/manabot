"""Retained machine receipts keep their original field spellings."""

from manabot.infra.host import HostFacts


def test_absent_memory_limit_leaves_retained_receipts_unchanged() -> None:
    retained = {
        "cpu_model": "AMD EPYC 7763",
        "logical_cpus": 64,
        "schedulable_cpus": 6,
        "cpu_quota": None,
    }
    assert HostFacts.model_validate(retained).model_dump() == retained


def test_retained_memory_allowance_keeps_its_source() -> None:
    retained = {
        "cpu_model": "AMD EPYC 7763",
        "logical_cpus": 64,
        "schedulable_cpus": 6,
        "cpu_quota": 6.0,
        "memory_limit": {"bytes": 48_000_000_000, "source": "runpod-env"},
    }
    assert HostFacts.model_validate(retained).model_dump() == retained
