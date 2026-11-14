"""Private completed-game publication and self-contained immutable fit inputs.

A SealedGame is one successful closure, including zero-row coverage and private
replay provenance. SnapshotBinding is the membership authority; packs contain
exact verified shard bytes. There is no launcher, mutable fit directory, queue
or checkpoint store here. Callers publish only after a whole native batch has
succeeded, and propagate storage failures before advancing the learner.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from manabot.belief.sampling_collection import CaptureCoverage
from manabot.belief.sampling_data import (
    BehaviorIdentity,
    BehaviorSegment,
    GameKey,
    GameMembership,
    ProducerSelection,
    SamplerDataset,
    SamplerGame,
    SnapshotBinding,
    SplitPlan,
    _validate_example,
)
from manabot.belief.sampling_types import SamplerSchema
from managym.decision import Command

if TYPE_CHECKING:
    from manabot.infra.artifacts import ArtifactStore, StoredArtifact

# ArtifactStore's single-PUT limit. No archive extraction or filesystem paths
# supplied by a producer are needed: packs contain JSON values, not tar members.
_MAX_PACK_BYTES = 5 * 1024**3


def _encoded(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _identity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish_bytes(path: Path, data: bytes) -> None:
    """No-clobber atomic publication, including identical concurrent redelivery."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(
        prefix=".belief-", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(
                    "conflicting bytes for an immutable belief identity"
                ) from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SealedGame:
    key: GameKey
    group: str
    schema: SamplerSchema
    world_identity: str
    segments: tuple[BehaviorSegment, ...]
    game: SamplerGame
    coverage: CaptureCoverage
    commands: tuple[Command, ...]
    status: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        if self.status != "completed" or not self.group or not self.world_identity:
            raise ValueError("only completed games can be sealed")
        if self.key.identity != self.game.game_id:
            raise ValueError("sealed game differs from original producer identity")
        step = 0
        for segment in self.segments:
            if segment.first_step != step:
                raise ValueError(
                    "behavior segments must cover the game without gaps or overlap"
                )
            step = segment.stop_step
        if (
            not self.segments
            or step != self.coverage.decisions
            or step != len(self.commands)
        ):
            raise ValueError(
                "behavior segments and replay must cover every exposed decision"
            )
        retained = tuple(
            sum(row.viewer == seat for row in self.game.examples) for seat in (0, 1)
        )
        if retained != self.coverage.retained_rows or sum(retained) != len(
            self.game.examples
        ):
            raise ValueError("sealed rows disagree with completed coverage")
        if self.coverage.selected_moments > self.coverage.decisions:
            raise ValueError("selected moments exceed the completed game")
        if any(row.viewer not in (0, 1) for row in self.game.examples):
            raise ValueError("unknown sealed viewer")
        views = [(row.viewer, row.revision) for row in self.game.examples]
        if len(set(views)) != len(views):
            raise ValueError("duplicate sealed viewer observation")


def seal_game(directory: Path, game: SealedGame) -> GameMembership:
    """Persist closure, provenance and supervision together, then acknowledge.

    Reconstructed prefixes must not call this function as new games. Redelivery
    with the original key is safe; changed completion under that key is corruption.
    """
    data = _encoded(asdict(game))
    member = GameMembership(
        game.key, game.group, game.game.split, _identity(data), len(data)
    )
    # Validate supervision before publishing, including zero-row completed games.
    _validate_rows(game)
    _publish_bytes(directory / f"{game.key.identity}.json", data)
    return member


def _validate_rows(game: SealedGame) -> None:
    # One validator for v1 evidence and producer snapshots, including private targets.
    for row in game.game.examples:
        _validate_example(row, game.schema)


def _producers(games: tuple[SealedGame, ...]) -> tuple[BehaviorIdentity, ...]:
    producers = {p.identity: p for g in games for s in g.segments for p in s.seats}
    return tuple(producers[key] for key in sorted(producers))


def _read_shard(data: bytes, member: GameMembership) -> SealedGame:
    if len(data) != member.size or _identity(data) != member.sha256:
        raise ValueError("snapshot shard byte identity mismatch")
    game = TypeAdapter(SealedGame).validate_json(data)
    if (game.key, game.group, game.game.split) != (
        member.key,
        member.group,
        member.split,
    ):
        raise ValueError("snapshot shard membership mismatch")
    _validate_rows(game)
    return game


def validate_snapshot_groups(bindings: tuple[SnapshotBinding, ...]) -> None:
    """A declared group never changes split, even across producer snapshots."""
    groups: dict[str, str] = {}
    plans: dict[str, SplitPlan] = {}
    games: dict[str, tuple[str, str, str]] = {}
    for binding in bindings:
        for game in binding.games:
            group = game.group
            if groups.setdefault(group, game.split) != game.split:
                raise ValueError("snapshot group crosses splits")
            identity = (binding.split_plan.family, game.group, game.sha256)
            if games.setdefault(game.key.identity, identity) != identity:
                raise ValueError("snapshot game changed its group or bytes")
    for binding in bindings:
        plan = binding.split_plan
        if plans.setdefault(plan.family, plan) != plan:
            raise ValueError("prospective split plan changed across snapshots")


def seal_snapshot(
    directory: Path,
    members: tuple[GameMembership, ...],
    *,
    selection: tuple[ProducerSelection, ...],
    split_plan: SplitPlan,
    watermark: str,
    destination: Path,
    previous: tuple[SnapshotBinding, ...] = (),
    failure_references: tuple[str, ...] = (),
) -> SamplerDataset:
    """Select complete games only when every seat and segment qualifies.

    `members` is the producer's acknowledged publication index at the watermark,
    not a directory scan. Failed attempts are immutable private evidence references.
    The returned snapshot never reads the producer directory again.
    """
    selected: list[SealedGame] = []
    references: list[GameMembership] = []
    seen: dict[str, GameMembership] = {}
    for member in members:
        if member.key.identity in seen:
            if seen[member.key.identity] != member:
                raise ValueError("conflicting duplicate game publication")
            continue
        seen[member.key.identity] = member
        game = _read_shard(
            (directory / f"{member.key.identity}.json").read_bytes(), member
        )
        if all(
            any(s.includes(p) for s in selection)
            for segment in game.segments
            for p in segment.seats
        ):
            selected.append(game)
            references.append(member)
    if not selected:
        raise ValueError("snapshot selection has no completed games")
    first = selected[0]
    if any(
        g.schema != first.schema or g.world_identity != first.world_identity
        for g in selected
    ):
        raise ValueError("snapshot schema/world mismatch")
    games = tuple(selected)
    binding = SnapshotBinding(
        selection,
        _producers(games),
        split_plan,
        watermark,
        tuple(references),
        failure_references,
    )
    validate_snapshot_groups((*previous, binding))
    payload = {
        "format": "manabot.belief-snapshot/v1",
        "schema": asdict(first.schema),
        "world_identity": first.world_identity,
        "manifest": asdict(binding),
        "shards": [asdict(game) for game in games],
    }
    data = _encoded(payload)
    if len(data) > _MAX_PACK_BYTES:
        raise ValueError("snapshot exceeds the artifact store's single-PUT limit")
    _publish_bytes(destination, data)
    return read_snapshot(destination)


def read_snapshot(path: Path) -> SamplerDataset:
    """Verify membership and each contained shard; never extract supplied paths."""
    if path.stat().st_size > _MAX_PACK_BYTES:
        raise ValueError("snapshot exceeds the supported input size")
    return _read_snapshot_payload(json.loads(path.read_bytes()))


def _read_snapshot_payload(payload: dict[str, Any]) -> SamplerDataset:
    """Narrow untyped JSON once at the private transport boundary."""
    if payload["format"] != "manabot.belief-snapshot/v1":
        raise ValueError("unknown belief snapshot format")
    schema = TypeAdapter(SamplerSchema).validate_python(payload["schema"])
    binding = TypeAdapter(SnapshotBinding).validate_python(payload["manifest"])
    if len(payload["shards"]) != len(binding.games):
        raise ValueError("snapshot shard membership count mismatch")
    games = tuple(
        _read_shard(_encoded(shard), member)
        for shard, member in zip(payload["shards"], binding.games, strict=True)
    )
    if _producers(games) != binding.producers or any(
        not any(s.includes(p) for s in binding.selection) for p in binding.producers
    ):
        raise ValueError("snapshot producer selection mismatch")
    if any(
        g.schema != schema or g.world_identity != payload["world_identity"]
        for g in games
    ):
        raise ValueError("snapshot schema/world mismatch")
    return SamplerDataset(
        schema, None, payload["world_identity"], tuple(g.game for g in games), binding
    )


def publish_snapshot(
    path: Path, destination: str, *, store: "ArtifactStore"
) -> "StoredArtifact":
    """Acknowledge only a verified storage write to a caller-verified private URI."""
    data = path.read_bytes()
    if len(data) > _MAX_PACK_BYTES:
        raise ValueError("snapshot exceeds the supported input size")
    _read_snapshot_payload(json.loads(data))
    return store.publish(path, destination, sha256=_identity(data), size=len(data))
