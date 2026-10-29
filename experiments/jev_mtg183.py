"""One frozen Jev campaign with durable inclusive request admission.

Jev plays a digest-bound checkpoint (MTG-212) or the `heuristic` anchor, pinned
by its source digest. The ordinary native game runner owns Commands and replay. This module owns the
seeded schedule, pilot boundary, checkpoint/runtime admission and retained report.
Restart never replaces an attempted game or clears an uncertain API reservation.
The protocol must be reviewed and populated from dated evidence before execution.
"""

import argparse
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
import fcntl
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Literal

from pydantic import Field, model_validator
import torch

from experiments.elemental_grid import DECKS, PACK
from manabot.env.failures import runtime_identities
from manabot.infra.artifacts import verify_file
from manabot.infra.config import MatchConfig
from manabot.infra.identity import file_sha256
from manabot.llm.budget import BudgetConfig, BudgetLedger, Spend
from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession
from manabot.llm.schema import Record
from manabot.verify.decision_profile import wilson_interval
from manabot.verify.llm_game import (
    CheckpointOpponent,
    GameConfig,
    GameSummary,
    HeuristicOpponent,
    heuristic_sha256,
    play,
)

DEALS_PER_PAIRING = 5


class PinnedFile(Record):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)

    def verify(self) -> None:
        verify_file(self.path, self.sha256, self.bytes)


class Protocol(Record):
    version: Literal["jev-mtg183-v3"] = "jev-mtg183-v3"
    inference: OpenRouterConfig
    # Exactly one opponent: a checkpoint with its selection, or the heuristic.
    checkpoint: PinnedFile | None = None
    checkpoint_artifact_id: str | None = Field(default=None, min_length=1)
    # Who selected the checkpoint and by what rule; "final" is not "best".
    checkpoint_selection: str | None = Field(default=None, min_length=1)
    heuristic_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    billing_evidence: PinnedFile
    budget: BudgetConfig
    world: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_extension_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    driver_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_games: int = Field(default=20, ge=10, le=20)
    # A slice of the frozen shuffled schedule: `planned_games` from position
    # `first_game`; a continuation after an outage starts past retained work.
    first_game: int = Field(default=0, ge=0, lt=1000)
    planned_games: int | None = Field(default=None, ge=20, le=1000)
    pilot_included: Literal[True] = True
    decks: tuple[str, ...] = DECKS
    pack: Literal["elemental-world-v2"] = PACK
    schedule_seed: int = 212183
    deal_base: int = 212183000
    opponent_seed_base: int = 212595000
    history_capacity: int = Field(default=32, ge=0, le=32)
    skip_trivial: bool = True
    max_decisions: int = Field(default=2000, gt=0)
    game_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    campaign_seconds: float = Field(default=86400, gt=0, allow_inf_nan=False)
    # A failed game is retained unscored; consecutive failures suggest an outage.
    max_consecutive_failures: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def treatment(self) -> "Protocol":
        # The Decisions transport itself requires the win-game objective.
        if (self.inference.transport, self.inference.model_id) != (
            "decisions",
            "typesafe/jev-1.14",
        ):
            raise ValueError("campaign requires the frozen Jev winning treatment")
        if len(self.decks) != len(DECKS) or set(self.decks) != set(DECKS):
            raise ValueError("campaign requires the complete ten-deck Elemental suite")
        checkpoint = (
            self.checkpoint,
            self.checkpoint_artifact_id,
            self.checkpoint_selection,
        )
        if self.heuristic_sha256 is None:
            if None in checkpoint:
                raise ValueError("checkpoint opponent requires its pin and selection")
        elif checkpoint != (None, None, None):
            raise ValueError("campaign has exactly one opponent")
        if self.budget.bound_evidence != self.billing_evidence.sha256:
            raise ValueError("budget must name the pinned billing evidence digest")
        return self

    def admit(self) -> None:
        self.billing_evidence.verify()
        if self.checkpoint is not None:
            self.checkpoint.verify()
        if (
            self.heuristic_sha256 is not None
            and heuristic_sha256() != self.heuristic_sha256
        ):
            raise ValueError("campaign runtime mismatch: heuristic_sha256")
        actual = runtime_identities()
        for name in ("world", "source_sha256", "engine_extension_sha256"):
            if actual[name] != getattr(self, name):
                raise ValueError(f"campaign runtime mismatch: {name}")
        if file_sha256(Path(__file__)) != self.driver_sha256:
            raise ValueError("campaign driver differs from frozen protocol")
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is required")


class ScheduledGame(Record):
    index: int
    jev_deck: str
    policy_deck: str
    config: GameConfig

    @property
    def name(self) -> str:
        return f"game-{self.index:04d}"

    def admission(self, out: Path) -> Path:
        return out / f"{self.name}.json"


class Attempt(Record):
    game: ScheduledGame
    # None is an admitted but interrupted attempt, never an unattempted game.
    summary: GameSummary | None = None

    @property
    def outcome(self) -> Literal["win", "loss", "draw"] | None:
        """Only completed, replay-matching, failure-free games can score."""
        summary = self.summary
        if (
            summary is None
            or not summary.completed
            or not summary.replay_matches
            or summary.failure is not None
        ):
            return None
        if summary.winner is None:
            return "draw"
        return "win" if summary.winner == self.game.config.model_seat else "loss"


def schedule(protocol: Protocol) -> tuple[ScheduledGame, ...]:
    """Five independent deals per ordered matchup/seat, shuffled before scoring.

    Each game has its own deal seed; no claim of paired-deal independence.
    The affordable prefix can be unbalanced, so report its actual seat counts.
    """
    games: list[ScheduledGame] = []
    for jev in protocol.decks:
        for policy in protocol.decks:
            for seat in (0, 1):
                for _ in range(DEALS_PER_PAIRING):
                    index = len(games)
                    decks = (jev, policy) if seat == 0 else (policy, jev)
                    games.append(
                        ScheduledGame(
                            index=index,
                            jev_deck=jev,
                            policy_deck=policy,
                            config=GameConfig(
                                seed=protocol.deal_base + index,
                                opponent_seed=protocol.opponent_seed_base + index,
                                model_seat=seat,
                                match=MatchConfig.authored(protocol.pack, *decks),
                                history_capacity=protocol.history_capacity,
                                skip_trivial=protocol.skip_trivial,
                                max_decisions=protocol.max_decisions,
                                max_seconds=protocol.game_seconds,
                            ),
                        )
                    )
    random.Random(protocol.schedule_seed).shuffle(games)
    end = (
        None
        if protocol.planned_games is None
        else protocol.first_game + protocol.planned_games
    )
    return tuple(games[protocol.first_game : end])


def _write(path: Path, text: str) -> None:
    """Persist admission/protocol/report across process interruption."""
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as output:
        output.write(text + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Scores(Record):
    attempted: int
    valid: int
    wins: int
    losses: int
    draws: int
    failures: int
    interrupted: int
    replay_failures: int
    win_rate: float | None
    wilson95: tuple[float, float] | None


def _scores(attempts: list[Attempt]) -> Scores:
    outcomes = Counter(a.outcome for a in attempts)
    wins, draws = outcomes["win"], outcomes["draw"]
    n = len(attempts) - outcomes[None]
    return Scores(
        attempted=len(attempts),
        valid=n,
        wins=wins,
        losses=outcomes["loss"],
        draws=draws,
        failures=len(attempts) - n,
        interrupted=sum(a.summary is None for a in attempts),
        replay_failures=sum(
            a.summary is not None and not a.summary.replay_matches for a in attempts
        ),
        win_rate=wins / n if n else None,
        wilson95=wilson_interval(wins, n) if n else None,
    )


class Pilot(Record):
    attempted: int
    model_decisions: int
    input_tokens: int
    output_tokens: int
    accounted_api_usd: Decimal
    projected_1000_api_usd: Decimal | None
    projected_1000_inclusive_usd: Decimal | None


class Report(Record):
    planned: int
    unattempted: int
    overall: Scores
    by_seat: dict[str, Scores]
    attempts: tuple[Attempt, ...]
    spend: Spend
    game_spend_usd: dict[str, Decimal]
    inclusive_cost_per_attempt_usd: Decimal | None
    pilot: Pilot
    repairs: int
    fallbacks: int
    unpriced_attempts: int
    # Unsent requests resent after connection failure; each keeps its bound.
    connect_retries: int
    decision_seconds_median: float | None
    decision_seconds_p95: float | None
    stop_reason: str | None
    interpretation: str = (
        "Seeded games are the sampling units. Wilson intervals describe observed "
        "wins among replay-verified completed games; draws are non-wins. Budget "
        "stopping, incomplete games and the unbalanced schedule prefix prevent a "
        "fixed-sample coverage or full-suite strength claim. Pilot games count "
        "in the same cohort and budget. Pending charges retain their full bound."
    )


def report(
    games: tuple[ScheduledGame, ...],
    attempts: list[Attempt],
    spend: Spend,
    pilot_games: int,
    stop_reason: str | None,
) -> Report:
    owner_costs = spend.cost_by_owner
    costs = {a.game.name: owner_costs.get(a.game.name, Decimal(0)) for a in attempts}
    pilot = attempts[:pilot_games]
    summaries = [a.summary for a in attempts if a.summary is not None]
    pilot_summaries = [a.summary for a in pilot if a.summary is not None]
    pilot_cost = sum((costs[a.game.name] for a in pilot), Decimal(0))
    full_pilot = len(pilot) == pilot_games and all(a.outcome is not None for a in pilot)
    projection = pilot_cost * 1000 / pilot_games if full_pilot else None
    latencies = sorted(t for summary in summaries for t in summary.model_latencies)
    return Report(
        planned=len(games),
        unattempted=len(games) - len(attempts),
        overall=_scores(attempts),
        by_seat={
            str(seat): _scores(
                [a for a in attempts if a.game.config.model_seat == seat]
            )
            for seat in (0, 1)
        },
        attempts=tuple(attempts),
        spend=spend,
        game_spend_usd=costs,
        inclusive_cost_per_attempt_usd=spend.accounted_usd / len(attempts)
        if attempts
        else None,
        pilot=Pilot(
            attempted=len(pilot),
            model_decisions=sum(s.model_decisions for s in pilot_summaries),
            input_tokens=sum(s.prompt_tokens for s in pilot_summaries),
            output_tokens=sum(s.completion_tokens for s in pilot_summaries),
            accounted_api_usd=pilot_cost,
            projected_1000_api_usd=projection,
            projected_1000_inclusive_usd=(projection + spend.overhead_usd)
            if projection is not None
            else None,
        ),
        repairs=sum(s.origins.get("retry", 0) for s in summaries),
        fallbacks=sum(s.origins.get("fallback", 0) for s in summaries),
        unpriced_attempts=sum(c.charged_usd is None for c in spend.requests),
        connect_retries=sum(c.abandoned for c in spend.requests),
        decision_seconds_median=statistics.median(latencies) if latencies else None,
        decision_seconds_p95=latencies[math.ceil(len(latencies) * 0.95) - 1]
        if latencies
        else None,
        stop_reason=stop_reason,
    )


def execute(
    protocol: Protocol,
    out: Path,
    player: Callable[[ScheduledGame, Path, BudgetLedger], GameSummary],
    *,
    admit: Callable[[], None],
) -> Report:
    """Single writer, one attempt per schedule entry; injected player for fixtures."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = out / "protocol.json"
        if frozen.exists():
            if Protocol.model_validate_json(frozen.read_text()) != protocol:
                raise ValueError("campaign protocol cannot change")
        else:
            _write(frozen, protocol.model_dump_json(indent=2))
        games = schedule(protocol)
        schedule_path = out / "schedule.json"
        schedule_data = [g.model_dump(mode="json") for g in games]
        if schedule_path.exists():
            if json.loads(schedule_path.read_text()) != schedule_data:
                raise ValueError("retained schedule differs from frozen protocol")
        else:
            _write(schedule_path, json.dumps(schedule_data, indent=2))
        ledger = BudgetLedger(out / "spend.sqlite", protocol.budget)
        attempts: list[Attempt] = []
        for game in games:
            path = game.admission(out)
            if path.exists():
                attempt = Attempt.model_validate_json(path.read_text())
                if attempt.game != game:
                    raise ValueError("retained attempt differs from frozen schedule")
                attempts.append(attempt)
        # Re-running collects interrupted work; it never silently restarts a cohort.
        prior = out / "report.json"

        def retain_report(reason: str | None) -> Report:
            result = report(
                games, attempts, ledger.snapshot(), protocol.pilot_games, reason
            )
            _write(prior, result.model_dump_json(indent=2))
            return result

        if attempts or prior.exists():
            reason = "existing campaign retained; no automatic continuation"
            if prior.exists():
                reason = (
                    Report.model_validate_json(prior.read_text()).stop_reason or reason
                )
            return retain_report(reason)
        # Admission failure precedes any attempt or spend and leaves the
        # directory runnable once the missing input is supplied.
        admit()
        stop_reason = None
        failed_run = 0
        started = time.monotonic()
        try:
            for game in games:
                if (
                    time.monotonic() - started + protocol.game_seconds
                    > protocol.campaign_seconds
                ):
                    stop_reason = "campaign time allowance reached"
                    break
                try:
                    ledger.check_admission()
                except RuntimeError as error:
                    stop_reason = str(error)
                    break
                attempt = Attempt(game=game)
                path = game.admission(out)
                _write(path, attempt.model_dump_json(indent=2))
                attempts.append(attempt)
                summary = player(game, out / game.name, ledger)
                attempt = Attempt(game=game, summary=summary)
                attempts[-1] = attempt
                _write(path, attempt.model_dump_json(indent=2))
                current = retain_report(None)
                if len(attempts) == protocol.pilot_games:
                    _write(out / "pilot.json", current.pilot.model_dump_json(indent=2))
                failed_run = 0 if attempt.outcome is not None else failed_run + 1
                if failed_run == protocol.max_consecutive_failures:
                    stop_reason = (
                        f"{failed_run} consecutive failed games; last: "
                        f"{summary.failure or 'game incomplete or replay failed'}"
                    )
                    break
        except BaseException as error:
            stop_reason = f"{type(error).__name__}: campaign interrupted"
            raise
        finally:
            result = retain_report(stop_reason)
        return result


def run(protocol: Protocol, out: Path) -> Report:
    opponent: CheckpointOpponent | HeuristicOpponent | None = None

    def admit() -> None:
        nonlocal opponent
        protocol.admit()
        torch.set_num_threads(1)
        pin = protocol.checkpoint
        opponent = (
            HeuristicOpponent()
            if pin is None
            else CheckpointOpponent(pin.path, pin.sha256, pin.bytes)
        )

    def player(
        game: ScheduledGame, directory: Path, ledger: BudgetLedger
    ) -> GameSummary:
        assert opponent is not None
        with OpenRouterSession(
            protocol.inference, budget=ledger, budget_owner=game.name
        ) as session:
            return play(game.config, directory, session, opponent=opponent)

    return execute(protocol, out, player, admit=admit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(Protocol.model_validate_json(args.protocol.read_text()), args.out)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
