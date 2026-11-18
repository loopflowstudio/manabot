"""The spend ledger keeps its ceiling across restarts; no paid requests."""

from decimal import Decimal
from pathlib import Path

import pytest

from manabot.llm.budget import BudgetConfig, BudgetLedger


def budget_config() -> BudgetConfig:
    return BudgetConfig(
        ceiling_usd=Decimal("1"),
        request_bound_usd=Decimal("0.20"),
        overhead_usd=Decimal("0.1"),
        charge_multiplier=Decimal("1.1"),
        bound_evidence="controlled fixture charge ceiling",
    )


@pytest.mark.parametrize("overhead", ["1.1", "1.0000000001"])
def test_budget_rejects_non_api_allowance_above_the_ceiling(overhead: str) -> None:
    values = budget_config().model_dump()
    values["overhead_usd"] = Decimal(overhead)
    with pytest.raises(ValueError, match="non-API allowance exceeds"):
        BudgetConfig.model_validate(values)


def test_interrupted_reservation_survives_restart_and_configuration_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.sqlite"
    ledger = BudgetLedger(path, budget_config())
    identifier = ledger.reserve("pilot-game", "digest")
    restarted = BudgetLedger(path, budget_config())
    assert restarted.snapshot().accounted_usd == Decimal("0.3")
    with pytest.raises(RuntimeError, match="unsettled"):
        restarted.reserve("replacement", "digest")
    with pytest.raises(ValueError, match="cannot change"):
        BudgetLedger(
            path, budget_config().model_copy(update={"ceiling_usd": Decimal(10)})
        )
    restarted.settle(identifier, 0.1)
    restarted.settle(identifier, 0.1)
    assert restarted.snapshot().accounted_usd == Decimal("0.21")


def test_budget_counts_inclusive_cost_and_stops_before_dispatch(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "spend.sqlite", budget_config())
    for index in range(4):
        identifier = ledger.reserve(str(index), "digest")
        ledger.settle(identifier, 0.18)
    assert ledger.snapshot().accounted_usd == Decimal("0.892")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        ledger.reserve("next", "digest")


def test_exceeded_bound_is_retained_and_blocks_restart(tmp_path: Path) -> None:
    path = tmp_path / "spend.sqlite"
    ledger = BudgetLedger(path, budget_config())
    identifier = ledger.reserve("first", "digest")
    with pytest.raises(RuntimeError, match="exceeded"):
        ledger.settle(identifier, 0.3)
    restarted = BudgetLedger(path, budget_config())
    assert restarted.snapshot().accounted_usd == Decimal("0.43")
    with pytest.raises(RuntimeError, match="violated"):
        restarted.reserve("next", "digest")


def test_budget_never_rounds_up_the_authorized_ceiling(tmp_path: Path) -> None:
    config = BudgetConfig(
        ceiling_usd=Decimal("0.1000000001"),
        request_bound_usd=Decimal("0.100000001"),
        overhead_usd=Decimal("0"),
        charge_multiplier=Decimal("1"),
        bound_evidence="controlled fixture charge ceiling",
    )
    ledger = BudgetLedger(tmp_path / "spend.sqlite", config)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        ledger.check_admission()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        ledger.reserve("first", "digest")
    assert ledger.snapshot().requests == ()
