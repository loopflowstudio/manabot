"""Deprecated training-record helpers and retained exposure-receipt export.

New collectors write managym records directly. The receipt-to-Command transform
remains only for historical exposure bytes; it never writes new exposure facts.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any
import warnings

from manabot.env.failures import CaptureScope, TrainingExposureReceipt, capture_scope
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.infra.run_settings import RunSettings
from manabot.sim.frames import build_viewer_frame, decision_row
from manabot.sim.replay import replay_environment, write_trace
import managym
from managym.experience import UNVERSIONED_CONTENT_HASH
from managym.journal import json_line
from managym.recording import GameInputs, GameRecord, read_records, record_from_row


def record_actions(
    setup: GameInputs,
    actions: Iterable[int],
    *,
    players: tuple[str, str],
    source: dict[str, object],
    option_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reconstruct one complete game, refusing missing or unoffered decisions."""
    if len(setup.players) != 2:
        raise ValueError("a game requires two player setups")
    native = setup.setup()
    first, second = native
    game: dict[str, Any] = {
        "match_id": f"recorded:{canonical_sha256({'setup': setup.model_dump(), 'source': source})}",
        "deal_seed": setup.seed,
        "leg": 0,
        "seat_players": list(players),
        "match_hypers": MatchConfig.from_players(
            first,
            second,
            decklists=native.decklists,
            mana_taps=native.mana_taps,
        ).model_dump(),
        "observation_hypers": ObservationConfig().model_dump(),
        "world": managym.world_identity(),
        "source": source,
        "decisions": [],
        "winner": None,
        "terminated": False,
        "truncated": False,
        "failure": None,
    }
    env, _ = replay_environment(game)
    try:
        game["initial_state_digest"] = env._engine.state_digest()
        for revision, action in enumerate(actions):
            if game["terminated"]:
                raise ValueError(
                    f"decision {revision}: actions continue after game ended"
                )
            frame = build_viewer_frame(
                env.last_raw_obs,
                match_id=game["match_id"],
                revision=revision,
                content_hash=UNVERSIONED_CONTENT_HASH,
                asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
            )
            if option_counts is not None and (
                revision >= len(option_counts)
                or len(frame["offers"]) != option_counts[revision]
            ):
                raise ValueError(
                    f"decision {revision}: option count differs from receipt"
                )
            row = decision_row(env, frame, action)
            row["player_id"] = players[row["actor"]]
            _, _, terminal, truncated, _ = env.step(action)
            row["post_state_digest"] = env._engine.state_digest()
            game["decisions"].append(row)
            if truncated:
                raise ValueError(f"decision {revision}: game truncated")
            game["terminated"] = bool(terminal)
        if option_counts is not None and len(game["decisions"]) != len(option_counts):
            raise ValueError("decision count differs from receipt")
        if not game["terminated"]:
            raise ValueError("game is cut off; export requires a complete game")
        game["winner"] = env._engine.winner_index()
    finally:
        env.close()
    game["game_trace_sha256"] = canonical_sha256(game)
    return game


@dataclass(frozen=True)
class TrainingGame:
    """One digest-bound game; export rereads the bytes to detect later edits."""

    path: Path
    receipt_sha256: str
    seed: int
    stream: int
    decks: tuple[dict[str, int], dict[str, int]]
    decisions: int
    record_index: int | None = None

    def export(self, path: Path, players: tuple[str, str] | None = None) -> Path:
        """Write one selected record, or transform a retained receipt into a Command trace.

        Seat names come from `players`, the recorded context or the setup.
        The selected source bytes stay untouched.
        """
        if path.resolve() == self.path.resolve():
            raise ValueError("export must not overwrite its receipt")
        if self.record_index is not None:
            record = list(read_records(self.path))[self.record_index]
            if record.sha256 != self.receipt_sha256:
                raise ValueError("record changed after selection")
            if players is not None:
                record.context = {**record.context, "players": list(players)}
            body = json_line(record.model_dump(mode="json", by_alias=True))
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with os.fdopen(
                os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb"
            ) as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(
                    gzip.compress(body, mtime=0) if path.suffix == ".gz" else body
                )
            return path
        receipt = _read_receipt(self.path)
        if receipt.sha256 != self.receipt_sha256:
            raise ValueError("receipt changed after selection")
        if not receipt.game.terminal:
            raise ValueError("game is cut off; export requires a complete game")
        capture = receipt.game.cards.capture
        running = managym.world_identity()
        if (
            receipt.identities.get("world") != running["world_version"]
            or capture.content_digest != running["card_digest"]
        ):
            raise ValueError(
                f"receipt world {receipt.identities.get('world')} differs from running {running['world_version']}: "
                f"retained cards {capture.content_digest}, running cards {running['card_digest']}"
            )
        assert capture.exposure is not None  # validated by the receipt model
        trace = receipt.game.trace
        first, second = (
            players
            or receipt.context.get("players")
            or [setup.name for setup in trace.player_configs]
        )
        if not (
            isinstance(first, str) and isinstance(second, str) and first and second
        ):
            raise ValueError("two nonempty player names required")
        game = record_actions(
            GameInputs(
                seed=receipt.game.seed,
                players=trace.player_configs,
                decision_memory=trace.decision_memory,
                decision_memory_include_own=trace.decision_memory_include_own,
            ),
            trace.actions,
            players=(first, second),
            source={
                "context": receipt.context,
                "kind": "training",
                "receipt_sha256": receipt.sha256,
                "stream": receipt.stream,
                "seed": receipt.game.seed,
            },
            option_counts=[len(choice.options) for choice in capture.exposure.choices],
        )
        write_trace(path, [game])
        return path


def _read_receipt(path: Path) -> TrainingExposureReceipt:
    body = path.read_bytes()
    receipt = TrainingExposureReceipt.model_validate_json(body)
    digest = hashlib.sha256(body).hexdigest()
    if digest != receipt.sha256 or path.stem != digest:
        raise ValueError(f"receipt digest mismatch: {path}")
    return receipt


class TrainingGames:
    """Read a capture directory or one receipt, without importing a learner."""

    def __init__(self, source: Path) -> None:
        warnings.warn(
            "TrainingGames is deprecated; use managym.recording.read_records; removal in 0.4.0",
            DeprecationWarning,
            stacklevel=2,
        )
        self.source = source

    def _records(self) -> Iterator[tuple[Path, int | None, GameRecord, str]]:
        if not self.source.exists():
            raise ValueError(f"record source does not exist: {self.source}")
        paths = (
            [self.source]
            if self.source.is_file()
            else sorted(self.source.glob("*.jsonl"))
        )
        for path in paths:
            with path.open() as source:
                first = json.loads(source.readline())
            if first.get("schema") == "training-exposure-v1":
                receipt = _read_receipt(path)
                yield path, None, record_from_row(first), receipt.sha256
            else:
                for index, record in enumerate(read_records(path)):
                    yield path, index, record, record.sha256

    @property
    def cut_off(self) -> int:
        """Games the loop stopped before finishing; they are never exported."""
        return sum(
            record.games[0].end is None or record.games[0].end.status != "complete"
            for _, _, record, _ in self._records()
        )

    def games(self) -> list[TrainingGame]:
        """Every complete game, in file and row order."""
        games: list[TrainingGame] = []
        for path, index, record, digest in self._records():
            inputs = record.games[0]
            if inputs.end is not None and inputs.end.status == "complete":
                first, second = inputs.players
                games.append(
                    TrainingGame(
                        path=path,
                        receipt_sha256=digest,
                        seed=inputs.seed,
                        stream=inputs.stream,
                        decks=(first.decklist, second.decklist),
                        decisions=len(inputs.actions),
                        record_index=index,
                    )
                )
        return games

    def sample(self, seed: int | None = None) -> TrainingGame:
        games = self.games()
        if not games:
            raise ValueError(f"no complete games ({self.cut_off} cut off)")
        return random.Random(seed).choice(games)


@contextmanager
def record_games(
    directory: Path,
    *,
    players: tuple[str, str] | None = None,
    context: Callable[[], dict[str, object]] | None = None,
) -> Iterator[TrainingGames]:
    """Capture collectors constructed in this scope; close them before leaving it.

    `players` and `context()` are retained in each record's context.
    """
    warnings.warn(
        "record_games is deprecated; use managym recording games_path; removal in 0.4.0",
        DeprecationWarning,
        stacklevel=2,
    )
    (directory / "games").mkdir(parents=True, exist_ok=True, mode=0o700)

    def coordinates() -> dict[str, object]:
        return {
            **(context() if context else {}),
            **({"players": list(players)} if players else {}),
        }

    with capture_scope(
        CaptureScope(
            directory / "failures",
            context=coordinates,
            settings=RunSettings(record_games=1.0),
        )
    ):
        yield TrainingGames(directory / "games")
