"""One language-model game against a frozen checkpoint, as an Etude match.

An experiment's `play(match)` opens its model session and calls `play_game`.
The match seats two players: the language model, as Etude `Code` named for the
model with its inference config, and the checkpoint. Its config is one `Game`
of a fixed schedule (`schedule`): the decks, the model's native seat and the
deal. `manabot.verify.llm_game.play` owns the game itself: capture, selection,
Commands and replay.

`play_game` returns Etude tables, one match of one game:

- `games`: a row per player, `score` null unless the game completed and
  replayed, so a failed game is retained and never scored.
- `llm_usage`: the model's decisions, tokens, reported API dollars and timing
  for the game, with the SHA-256 of its record.

The record (every capture, prompt, selection receipt and Command) leaves as one
blob, a gzip tar of the game folder; per-request detail is read from there.
"""

from collections.abc import Sequence
import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from etude.eval.records import CheckpointPlayer, Code, Match
from etude.eval.result import Result, Row
from pydantic import BaseModel, ConfigDict, Field

from manabot.infra.artifacts import fetch_etude_checkpoint
from manabot.infra.config import MatchConfig
from manabot.llm.session import InferenceSession
from manabot.verify.llm_game import CheckpointOpponent, GameConfig, GameSummary, play

if TYPE_CHECKING:
    from manabot.llm.openrouter import OpenRouterSession
    from manabot.policy.protocol import Backend


class Game(BaseModel):
    """One scheduled game: who pilots which deck from which seat on which deal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pack: str
    llm_deck: str
    opponent_deck: str
    # The model's native seat; 0 is on the play.
    llm_seat: int = Field(ge=0, le=1)
    deal: int
    opponent_seed: int
    # A game past either limit is retained as a failure, never scored.
    max_decisions: int = Field(default=2000, gt=0)
    max_seconds: float = Field(default=3600, gt=0)

    def config(self) -> GameConfig:
        decks = (self.llm_deck, self.opponent_deck)
        return GameConfig(
            seed=self.deal,
            opponent_seed=self.opponent_seed,
            model_seat=self.llm_seat,
            match=MatchConfig.authored(
                self.pack, *decks[:: 1 if not self.llm_seat else -1]
            ),
            max_decisions=self.max_decisions,
            max_seconds=self.max_seconds,
        )


def schedule(
    pack: str,
    decks: Sequence[str],
    *,
    deals: int,
    deal_base: int,
    opponent_seed_base: int,
    max_seconds: float = 3600,
) -> tuple[Game, ...]:
    """Every ordered deck pairing once per deal, the model's seat alternating.

    Within a deal each deck is on the play in half its games on either side,
    and a pairing's seat flips from one deal to the next, so two deals cover
    both seats of each pairing.
    Seeds count up from the bases in schedule order, so a longer schedule
    extends a shorter one and a game keeps its match key.
    """
    games: list[Game] = []
    for deal in range(deals):
        for row, llm_deck in enumerate(decks):
            for column, opponent_deck in enumerate(decks):
                index = len(games)
                games.append(
                    Game(
                        pack=pack,
                        llm_deck=llm_deck,
                        opponent_deck=opponent_deck,
                        llm_seat=(row + column + deal) % 2,
                        deal=deal_base + index,
                        opponent_seed=opponent_seed_base + index,
                        max_seconds=max_seconds,
                    )
                )
    return tuple(games)


def seated(match: Match) -> tuple[int, Code, CheckpointPlayer]:
    """The language model's index among the match's players, it and the checkpoint."""
    found = {type(player): index for index, player in enumerate(match.players)}
    if len(match.players) != 2 or set(found) != {Code, CheckpointPlayer}:
        raise ValueError("an LLM match seats one language model and one checkpoint")
    index = found[Code]
    llm, checkpoint = match.players[index], match.players[1 - index]
    assert isinstance(llm, Code) and isinstance(checkpoint, CheckpointPlayer)
    return index, llm, checkpoint


def checkpoint_opponent(player: CheckpointPlayer, cache: Path) -> CheckpointOpponent:
    """The seated checkpoint, verified and loaded on the CPU once.

    A local file is verified in place; a stored object is downloaded into
    `cache`, named by its digest.
    """
    artifact = player.checkpoint.artifact
    if artifact.sha256 is None or artifact.size_bytes is None:
        raise ValueError(f"{player.name} is not a checkpoint with a digest and size")
    cache.mkdir(parents=True, exist_ok=True)
    path = fetch_etude_checkpoint(artifact, cache / artifact.sha256)
    return CheckpointOpponent(path, artifact.sha256, artifact.size_bytes)


def play_game(
    match: Match,
    session: "InferenceSession | OpenRouterSession",
    opponent: CheckpointOpponent,
    *,
    backend: "Backend",
) -> Result:
    """Play the match's one game and return its rows and record; see the module docstring.

    `backend` runs the seated checkpoint.
    """
    if match.games != 1:
        raise ValueError("an LLM match is one game")
    game = Game.model_validate(match.config)
    index, _, _ = seated(match)
    with TemporaryDirectory() as scratch:
        record = Path(scratch) / match.key
        summary = play(
            game.config(), record, session, opponent=opponent, backend=backend
        )
        archive = _archive(record)
    return Result(
        tables=tables(game, summary, llm=index, record=archive), blobs=[archive]
    )


def tables(
    game: Game, summary: GameSummary, *, llm: int, record: bytes
) -> dict[str, list[Row]]:
    """`games` rows for both players and the model's `llm_usage` row."""
    valid = summary.completed and summary.replay_matches and summary.failure is None
    score = None
    if valid:
        score = (
            0.5 if summary.winner is None else float(summary.winner == game.llm_seat)
        )
    sides = (
        (llm, game.llm_deck, game.opponent_deck, game.llm_seat, score),
        (
            1 - llm,
            game.opponent_deck,
            game.llm_deck,
            1 - game.llm_seat,
            None if score is None else 1.0 - score,
        ),
    )
    games: list[Row] = [
        {
            "seat": seat,
            "deal": game.deal,
            "deck": deck,
            "opponent_deck": opponent_deck,
            "game_seat": game_seat,
            "score": player_score,
            "valid": valid,
            "failure": summary.failure,
            "decisions": summary.decisions,
            "seconds": summary.game_seconds,
        }
        for seat, deck, opponent_deck, game_seat, player_score in sides
    ]
    usage: Row = {
        "seat": llm,
        "deal": game.deal,
        "model_decisions": summary.model_decisions,
        "repairs": summary.origins.get("retry", 0),
        "fallbacks": summary.origins.get("fallback", 0),
        "generation_attempts": summary.generation_attempts,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "output_tokens": summary.output_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "cached_tokens": summary.cached_tokens,
        "max_prompt_tokens": summary.max_prompt_tokens,
        "reported_cost_usd": summary.reported_cost_usd,
        "unpriced_attempts": summary.unpriced_attempts,
        "decision_seconds_median": summary.decision_seconds_median,
        "decision_seconds_max": summary.decision_seconds_max,
        "record_sha256": hashlib.sha256(record).hexdigest(),
    }
    return {"games": games, "llm_usage": [usage]}


def _archive(folder: Path) -> bytes:
    """The game folder as gzip tar bytes, members in name order."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(folder.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(folder.parent)), recursive=False)
    return buffer.getvalue()
