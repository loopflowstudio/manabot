"""Native games inside Etude's game process, with durable Command evidence.

Cooperative limits are checked between native steps. Until Etude's per-match
watchdog lands, a true native hang or hard process exit retries the grouped
job; this module creates no supervisor or nested process.
"""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

from etude.errors import RetryRequested

from manabot.sim.agent import GamePlayer
from manabot.sim.card_evidence import EvidenceRef
from manabot.sim.native_game import execute_game
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import ReplayReceipt
from manabot.sim.supervision import ReplayOutcome, is_game_error, verify_game


def play_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    agents: Mapping[str, GamePlayer],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path,
    *,
    keep_completed: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
    game["registrations"] = [r.model_dump(mode="json") for r in registrations]
    with evidence_path.open("x") as journal:

        def send(event: tuple[str, Any]) -> None:
            journal.write(json.dumps(event, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        send(("game", game))
        execute_game(
            game,
            registrations,
            {},
            max_commands,
            send,
            agents=agents,
            deadline=started + game_seconds,
            propagate_errors=(RetryRequested,),
        )
        game["execution_seconds"] = time.monotonic() - started
        send(("settled", game))
    if game["failure"] is None and not keep_completed:
        evidence_path.unlink()
    return game


def replay_in_process(
    games: list[dict[str, Any]],
    card_evidence_dir: Path | None,
) -> ReplayOutcome:
    """Verify Commands and capture cards in the owning Etude game process."""
    receipts: list[dict[str, Any]] = []
    references: list[EvidenceRef] = []
    errors: list[str] = []
    for game in games:
        try:
            receipt, reference = verify_game(game, card_evidence_dir)
            receipts.append(receipt)
            if reference is not None:
                references.append(reference)
        except BaseException as error:
            if not is_game_error(error):
                raise
            errors.append(f"{type(error).__name__}: {error}")
            receipts.append({**ReplayReceipt().to_dict(), "passed": False})
    return ReplayOutcome(receipts, tuple(references), "; ".join(errors) or None)
