"""Controlled campaign outcomes and interrupted billing; no paid requests."""

from decimal import Decimal
import json
from pathlib import Path

import pytest

from experiments.jev_mtg183 import (
    Attempt,
    PinnedFile,
    Protocol,
    ScheduledGame,
    execute,
    _report,
    schedule,
)
from manabot.llm.budget import BudgetConfig, BudgetLedger
from manabot.llm.openrouter import OpenRouterConfig
from manabot.verify.llm_game import GameSummary


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


def protocol() -> Protocol:
    pin = PinnedFile(path=Path("fixture"), sha256="0" * 64, bytes=1)
    return Protocol(
        inference=OpenRouterConfig(
            model_id="typesafe/jev-1.14",
            transport="decisions",
            objective="win-game",
        ),
        checkpoint=pin,
        checkpoint_artifact_id="fixture",
        checkpoint_selection="controlled fixture only",
        billing_evidence=pin,
        budget=budget_config().model_copy(update={"bound_evidence": pin.sha256}),
        world="fixture",
        source_sha256=pin.sha256,
        engine_extension_sha256=pin.sha256,
        driver_sha256=pin.sha256,
    )


def test_schedule_is_frozen_and_balanced_with_distinct_seeded_games() -> None:
    games = schedule(protocol())
    assert games == schedule(protocol())
    assert len(games) == len({g.config.seed for g in games}) == 1000
    assert sum(g.config.model_seat == 0 for g in games) == 500
    assert len({(g.jev_deck, g.policy_deck, g.config.model_seat) for g in games}) == 200


def test_report_uses_actual_seat_and_excludes_failed_replay(tmp_path: Path) -> None:
    games = schedule(protocol())
    first = next(g for g in games if g.config.model_seat == 1)
    second = next(g for g in games if g.config.model_seat == 0)
    attempts = [
        Attempt(
            game=first, summary=GameSummary(mode="fixture", completed=True, winner=1)
        ),
        Attempt(
            game=second,
            summary=GameSummary(
                mode="fixture", completed=True, winner=0, replay_matches=False
            ),
        ),
        Attempt(game=games[2]),
    ]
    ledger = BudgetLedger(tmp_path / "spend.sqlite", budget_config())
    ledger.settle(ledger.reserve(first.name, "paid"), 0.1)
    ledger.settle(ledger.reserve(second.name, "free"), 0)
    ledger.reserve(first.name, "digest")
    result = _report(games, attempts, ledger.snapshot(), 20, "fixture")
    assert result.overall.valid == result.overall.wins == 1
    assert result.overall.replay_failures == result.overall.interrupted == 1
    assert result.by_seat["1"].wins == 1 and result.by_seat["0"].wins == 0
    assert result.overall.wilson95 is not None
    assert result.unattempted == 997 and result.unpriced_attempts == 1
    assert result.spend.accounted_usd == Decimal("0.41")
    assert result.game_spend_usd == {
        first.name: Decimal("0.31"),
        second.name: Decimal(0),
        games[2].name: Decimal(0),
    }
    assert result.pilot.accounted_api_usd == Decimal("0.31")


def test_campaign_interruption_keeps_attempt_and_never_replays_it(
    tmp_path: Path,
) -> None:
    config = protocol()

    def interrupted(game: object, out: Path, ledger: BudgetLedger) -> GameSummary:
        ledger.reserve("interrupted-game", "digest")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(config, tmp_path, interrupted, admit=lambda: None)
    retained = json.loads((tmp_path / "report.json").read_text())
    assert retained["overall"]["interrupted"] == 1
    assert retained["unattempted"] == 999
    recovered = execute(config, tmp_path, interrupted, admit=lambda: None)
    assert recovered.overall.attempted == 1
    assert recovered.spend.pending_usd == Decimal("0.2")


def test_campaign_pilot_and_later_games_share_one_inclusive_ceiling(
    tmp_path: Path,
) -> None:
    config = protocol().model_copy(
        update={
            "budget": BudgetConfig(
                ceiling_usd=Decimal("1"),
                request_bound_usd=Decimal("0.04"),
                overhead_usd=Decimal("0"),
                charge_multiplier=Decimal("1"),
                bound_evidence="0" * 64,
            ),
        }
    )

    def player(game: ScheduledGame, out: Path, ledger: BudgetLedger) -> GameSummary:
        identifier = ledger.reserve(game.name, "digest")
        ledger.settle(identifier, 0.04)
        return GameSummary(
            mode="fixture",
            completed=True,
            winner=game.config.model_seat,
            model_decisions=1,
            prompt_tokens=100,
            completion_tokens=10,
            reported_cost_usd=0.04,
            model_latencies=[0.01],
        )

    result = execute(config, tmp_path, player, admit=lambda: None)
    assert result.overall.attempted == result.overall.valid == 25
    assert result.unattempted == 975
    assert result.spend.accounted_usd == Decimal("1")
    assert result.pilot.attempted == result.pilot.model_decisions == 20
    assert result.pilot.input_tokens == 2000
    assert result.pilot.accounted_api_usd == Decimal("0.8")
    assert result.pilot.projected_1000_inclusive_usd == Decimal("40")
    assert (tmp_path / "pilot.json").exists()
    assert result.stop_reason == "inclusive campaign budget exhausted"
    retained = execute(config, tmp_path, player, admit=lambda: None)
    assert retained == result


def test_campaign_stops_before_admitting_another_game_with_pending_spend(
    tmp_path: Path,
) -> None:
    def player(game: ScheduledGame, out: Path, ledger: BudgetLedger) -> GameSummary:
        ledger.reserve(game.name, "digest")
        # A retained outcome cannot establish that its API reservation settled.
        return GameSummary(mode="fixture", completed=True, winner=0)

    result = execute(protocol(), tmp_path, player, admit=lambda: None)
    assert result.overall.attempted == 1
    assert result.spend.pending_usd == Decimal("0.2")
    assert result.stop_reason == "unsettled request reservation; campaign stopped"


@pytest.mark.parametrize(
    "summary",
    [
        GameSummary(mode="fixture", completed=False),
        GameSummary(mode="fixture", completed=True, winner=0, replay_matches=False),
        GameSummary(mode="fixture", completed=True, winner=0, failure="fixture"),
    ],
)
def test_invalid_games_are_retained_unscored_until_consecutive_failures_stop(
    tmp_path: Path, summary: GameSummary
) -> None:
    played: list[str] = []

    def player(game: ScheduledGame, out: Path, ledger: BudgetLedger) -> GameSummary:
        played.append(game.name)
        if len(played) == 2:
            return GameSummary(mode="fixture", completed=True, winner=0)
        return summary

    result = execute(protocol(), tmp_path, player, admit=lambda: None)
    assert len(played) == result.overall.attempted == 5
    assert result.overall.failures == 4 and result.overall.valid == 1
    assert result.pilot.projected_1000_api_usd is None
    assert result.stop_reason and result.stop_reason.startswith("3 consecutive")


def test_admission_failure_leaves_the_campaign_runnable(tmp_path: Path) -> None:
    def refuse() -> None:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    def player(game: ScheduledGame, out: Path, ledger: BudgetLedger) -> GameSummary:
        return GameSummary(mode="fixture", completed=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        execute(protocol(), tmp_path, player, admit=refuse)
    assert not (tmp_path / "report.json").exists()
    result = execute(protocol(), tmp_path, player, admit=lambda: None)
    assert result.overall.attempted == protocol().max_consecutive_failures


def test_abandoned_reservation_counts_its_bound_without_blocking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.sqlite"
    ledger = BudgetLedger(path, budget_config())
    ledger.abandon(ledger.reserve("game", "digest"))
    identifier = ledger.reserve("game", "digest")
    ledger.settle(identifier, 0.1)
    ledger.abandon(identifier)
    spend = BudgetLedger(path, budget_config()).snapshot()
    assert spend.accounted_usd == Decimal("0.41")
    assert [r.abandoned for r in spend.requests] == [True, False]
    with pytest.raises(ValueError, match="unknown"):
        ledger.abandon("missing")
