"""Producer-owned publication outputs, independent of learner checkpoints.

ProducerPublication is the acknowledged game index and fixed split plan. Its
snapshot records retain exact pending bytes before storage is contacted, and
verified references before fits may use them. Restore this output with Etude's
saved outputs, never from an older learner checkpoint or a directory scan.
These local transactions do not establish remote saved-output delivery.
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from manabot.belief.sampling_data import (
    GameMembership,
    ProducerSelection,
    SnapshotBinding,
    SplitPlan,
)
from manabot.belief.sampling_snapshot import (
    SealedGame,
    _encoded,
    _identity,
    _publish_bytes,
    _sync_directory,
    publish_snapshot,
    read_snapshot,
    seal_game,
    seal_snapshot,
    validate_snapshot_groups,
)
from manabot.belief.sampling_types import SamplerSchema
from manabot.infra.artifacts import StoredArtifact, split_bucket_uri, verify_file
from manabot.infra.files import exclusive_file_lock

if TYPE_CHECKING:
    from manabot.infra.artifacts import ArtifactStore


@dataclass(frozen=True, slots=True)
class SnapshotPublication:
    """A retained pack; only a verified reference is eligible as a fit Input."""

    binding: SnapshotBinding
    destination: str
    sha256: str
    size: int
    reference: StoredArtifact | None = None

    def __post_init__(self) -> None:
        scheme, bucket, prefix = split_bucket_uri(self.destination)
        key = "/".join(filter(None, (prefix.rstrip("/"), "sha256", self.sha256)))
        expected = StoredArtifact(
            uri=f"{scheme}://{bucket}/{key}", sha256=self.sha256, bytes=self.size
        )
        if self.size < 1:
            raise ValueError("publication requires nonempty snapshot bytes")
        if self.reference is not None and (
            self.reference.uri,
            self.reference.sha256,
            self.reference.bytes,
        ) != (expected.uri, expected.sha256, expected.bytes):
            raise ValueError("verified reference differs from retained pending pack")


@dataclass(frozen=True, slots=True)
class ProducerPublication:
    run: str
    schema: SamplerSchema
    world_identity: str
    split_plan: SplitPlan
    members: tuple[GameMembership, ...] = ()
    failure_references: tuple[str, ...] = ()
    snapshots: tuple[SnapshotPublication, ...] = ()
    format: Literal["manabot.belief-producer/v1"] = "manabot.belief-producer/v1"

    def __post_init__(self) -> None:
        if not self.run or not self.world_identity:
            raise ValueError("producer requires run and world bindings")
        if len({m.key.identity for m in self.members}) != len(self.members):
            raise ValueError("duplicate acknowledged game")
        if any(
            m.key.run != self.run or m.split != self.split_plan.assign(m.group)
            for m in self.members
        ):
            raise ValueError("game differs from producer or prospective split")
        if len(set(self.failure_references)) != len(self.failure_references) or any(
            len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref)
            for ref in self.failure_references
        ):
            raise ValueError("failure references require unique evidence digests")
        members = {m.key.identity: m for m in self.members}
        for index, snapshot in enumerate(self.snapshots):
            binding = snapshot.binding
            if (
                binding.split_plan != self.split_plan
                or any(members.get(m.key.identity) != m for m in binding.games)
                or not set(binding.failure_references) <= set(self.failure_references)
            ):
                raise ValueError("snapshot differs from retained producer outputs")
            if snapshot.reference is None and index != len(self.snapshots) - 1:
                raise ValueError("pending publication must be reconciled first")
        validate_snapshot_groups(tuple(s.binding for s in self.snapshots))

    @property
    def watermark(self) -> str:
        # Outputs, not optimizer updates: rollback must never erase this frontier
        # or use it to advance the learner. Failure digests resolve at their owner.
        return f"{self.run}:{len(self.members)}:{len(self.failure_references)}"

    @property
    def verified(self) -> tuple[StoredArtifact, ...]:
        return tuple(s.reference for s in self.snapshots if s.reference is not None)


_STATE = TypeAdapter(ProducerPublication)


def _state_bytes(state: ProducerPublication) -> bytes:
    return _encoded(_STATE.dump_python(state, mode="json"))


def create_producer(
    directory: Path,
    *,
    run: str,
    schema: SamplerSchema,
    world_identity: str,
    split_plan: SplitPlan,
) -> ProducerPublication:
    """Fix the split plan before capture; reopening cannot change its bindings."""
    initial = ProducerPublication(run, schema, world_identity, split_plan)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "producer.json"
    lock = directory / "producer.lock"
    first_use = not lock.exists()
    # Initialization and mutation share a fence, including concurrent first use.
    with exclusive_file_lock(lock):
        if not path.exists():
            if not first_use:
                raise FileNotFoundError("retained producer index is missing")
            _publish_bytes(path, _state_bytes(initial))
        retained = read_producer(directory)
        if (
            retained.run,
            retained.schema,
            retained.world_identity,
            retained.split_plan,
        ) != (run, schema, world_identity, split_plan):
            raise ValueError("retained producer bindings differ")
    return retained


def read_producer(directory: Path) -> ProducerPublication:
    """Read the authoritative output index, even after learner rollback.

    A missing index is missing state, never permission to adopt orphan files.
    """
    return _STATE.validate_json((directory / "producer.json").read_bytes())


def _retain(
    directory: Path, before: ProducerPublication, after: ProducerPublication
) -> None:
    # Construct/validate and fsync outside the short writer fence. Snapshot I/O
    # and remote storage never hold this lock. Stale writers fail without retry.
    data = _state_bytes(_STATE.validate_python(after))
    descriptor, name = tempfile.mkstemp(prefix=".producer-", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        with exclusive_file_lock(directory / "producer.lock"):
            if read_producer(directory) != before:
                raise RuntimeError("producer outputs changed under another writer")
            os.replace(temporary, directory / "producer.json")
            _sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def retain_game(directory: Path, game: SealedGame) -> GameMembership:
    """Seal a successful batch's closure before acknowledging membership.

    Replay may redeliver the original key, never mint a new key for that prefix.
    Call only after the native batch succeeded; this does not tolerate failures.
    """
    state = read_producer(directory)
    if (
        game.key.run != state.run
        or game.schema != state.schema
        or game.world_identity != state.world_identity
        or game.game.split != state.split_plan.assign(game.group)
    ):
        raise ValueError(
            "game differs from retained producer bindings or prospective split"
        )
    member = seal_game(directory / "games", game)
    earlier = next((m for m in state.members if m.key == member.key), None)
    if earlier is not None:
        if earlier != member:
            raise ValueError("conflicting acknowledged game")
        return earlier
    _retain(directory, state, replace(state, members=(*state.members, member)))
    return member


def retain_failure(directory: Path, evidence_digest: str) -> None:
    """Retain a reference after its evidence owner acknowledged immutable bytes."""
    state = read_producer(directory)
    if evidence_digest not in state.failure_references:
        _retain(
            directory,
            state,
            replace(
                state, failure_references=(*state.failure_references, evidence_digest)
            ),
        )


def prepare_snapshot(
    directory: Path,
    *,
    selection: tuple[ProducerSelection, ...],
    destination: str,
) -> SnapshotPublication:
    """Retain exact pending bytes and destination before attempting publication."""
    state = read_producer(directory)
    if state.snapshots:
        latest = state.snapshots[-1]
        if (
            latest.binding.watermark == state.watermark
            and latest.binding.selection == selection
            and latest.destination == destination
        ):
            _verify_pack(directory, latest)
            return latest
        if latest.reference is None:
            raise ValueError(
                "reconcile pending publication before selecting another cut"
            )
    packs = directory / "snapshots"
    packs.mkdir(exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=packs) as temporary:
        path = Path(temporary) / "pack.json"
        dataset = seal_snapshot(
            directory / "games",
            state.members,
            selection=selection,
            split_plan=state.split_plan,
            watermark=state.watermark,
            destination=path,
            previous=tuple(s.binding for s in state.snapshots),
            failure_references=state.failure_references,
        )
        assert dataset.binding is not None
        data = path.read_bytes()
        pending = SnapshotPublication(
            dataset.binding, destination, _identity(data), len(data)
        )
        _publish_bytes(packs / f"{pending.sha256}.json", data)
    _retain(directory, state, replace(state, snapshots=(*state.snapshots, pending)))
    return pending


def _verify_pack(directory: Path, publication: SnapshotPublication) -> Path:
    path = directory / "snapshots" / f"{publication.sha256}.json"
    verify_file(path, publication.sha256, publication.size)
    if read_snapshot(path).binding != publication.binding:
        raise ValueError("retained snapshot binding differs from pending bytes")
    return path


def publish_pending_snapshot(
    directory: Path, *, store: "ArtifactStore"
) -> StoredArtifact:
    """Reconcile the retained content key, including a lost storage reply.

    ArtifactStore confirms existing bytes by size/checksum on retry. A storage
    error leaves the pending record intact and must stop the next learner update.
    No reference is returned before its acknowledgement is durably retained.
    """
    state = read_producer(directory)
    if not state.snapshots:
        raise ValueError("producer has no prepared snapshot")
    latest = state.snapshots[-1]
    path = _verify_pack(directory, latest)
    if latest.reference is not None:
        return latest.reference
    reference = publish_snapshot(path, latest.destination, store=store)
    verified = replace(latest, reference=reference)
    _retain(
        directory, state, replace(state, snapshots=(*state.snapshots[:-1], verified))
    )
    return reference
