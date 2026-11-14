"""Export actual collector games as policy-free Command traces.

TrainingGames reads immutable exposure receipts; record_games enables that same
capture for collectors created inside its scope. Export reconstructs a completed
game, never a policy, and checks each retained choice before writing the trace.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any

from manabot.env.failures import (
    CaptureScope,
    GameInputs,
    TrainingExposureReceipt,
    capture_scope,
)
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.sim.frames import build_viewer_frame, decision_row
from manabot.sim.replay import replay_environment, write_trace
import managym
from managym.experience import UNVERSIONED_CONTENT_HASH


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
    first, second = setup.players
    if first.world != second.world or first.disclosure != second.disclosure:
        raise ValueError("player setups require the same world and disclosure")
    game: dict[str, Any] = {
        "match_id": f"recorded:{canonical_sha256({'setup': setup.model_dump(), 'source': source})}",
        "deal_seed": setup.seed,
        "leg": 0,
        "seat_players": list(players),
        "match_hypers": MatchConfig(
            hero=first.name,
            villain=second.name,
            hero_deck=first.decklist,
            villain_deck=second.decklist,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
            world=first.world,
            decklists=first.disclosure,
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
    """One digest-bound receipt; export rereads the bytes to detect later edits."""

    path: Path
    receipt_sha256: str
    seed: int
    stream: int
    decks: tuple[dict[str, int], dict[str, int]]
    decisions: int

    def export(self, path: Path, players: tuple[str, str] | None = None) -> Path:
        """Write this game as one Command trace, leaving the receipt untouched.

        Seat names are `players`, else the names `record_games` retained in the
        receipt, else the receipt's deck seat names.
        """
        if path.resolve() == self.path.resolve():
            raise ValueError("export must not overwrite its receipt")
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
            GameInputs(seed=receipt.game.seed, players=trace.player_configs),
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

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _receipts(self) -> list[tuple[Path, TrainingExposureReceipt]]:
        if not self.directory.exists():
            raise ValueError(f"receipt source does not exist: {self.directory}")
        paths = (
            [self.directory]
            if self.directory.is_file()
            else sorted(self.directory.glob("*.jsonl"))
        )
        return [(path, _read_receipt(path)) for path in paths]

    @property
    def cut_off(self) -> int:
        """Games the loop stopped before finishing; they are never exported."""
        return sum(not receipt.game.terminal for _, receipt in self._receipts())

    def games(self) -> list[TrainingGame]:
        """Every complete game, in receipt-digest order."""
        games: list[TrainingGame] = []
        for path, receipt in self._receipts():
            if receipt.game.terminal:
                first, second = receipt.game.trace.player_configs
                games.append(
                    TrainingGame(
                        path=path,
                        receipt_sha256=receipt.sha256,
                        seed=receipt.game.seed,
                        stream=receipt.stream,
                        decks=(first.decklist, second.decklist),
                        decisions=len(receipt.game.trace.actions),
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

    `players` and `context()` are retained in each receipt's context.
    """
    directory.mkdir(parents=True, exist_ok=True)

    def coordinates() -> dict[str, object]:
        return {
            **(context() if context else {}),
            **({"players": list(players)} if players else {}),
        }

    with capture_scope(
        CaptureScope(
            directory / "failures",
            context=coordinates,
            training_exposure=True,
            exposure_directory=directory,
        )
    ):
        yield TrainingGames(directory)
