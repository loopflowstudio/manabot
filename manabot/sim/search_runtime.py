"""Dependency-light limits, receipts, and retained search registrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_MAX_PLAYOUT_STEPS = 2000

REPO_ROOT = Path(__file__).resolve().parents[2]
INT7_VALUE_TARGET_MANIFEST_SHA256 = (
    "3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf"
)
INT7_VALUE_TARGET_RESULT = (
    REPO_ROOT
    / "experiments"
    / "data"
    / "int-7-value-target-comparison-v1"
    / "sha256"
    / INT7_VALUE_TARGET_MANIFEST_SHA256
    / "result"
)
INT7_VALUE_TARGET_MANIFEST = INT7_VALUE_TARGET_RESULT / "manifest.json"

_INT7_POLICY_ONLY_SHA256 = {
    197: "1673a237ef2460d0e699667987c29fe6b42c28711bdb2041989f37692edbd1e6",
    198: "5b3dab6517534047d899704d44c839276c5cf74c7c56b6e29ce0a52180bf5223",
    199: "72cad2028861a7dd422f3e9ae18a98a09e9e911b6e3ca908f46b95a4c35c7fd3",
}


class RetainedCheckpointUnavailableError(FileNotFoundError):
    """A registered checkpoint or its manifest is not retained locally."""


class RetainedCheckpointMismatchError(ValueError):
    """Retained checkpoint identity or bytes differ from the frozen registry."""


@dataclass(frozen=True)
class RetainedCheckpointRegistration:
    """One repository-owned checkpoint resolved from frozen experiment evidence."""

    checkpoint_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    manifest_sha256: str
    training_seed: int
    world_id: str
    observation_abi_sha256: str
    action_abi_sha256: str
    parameter_count: int
    value_mode: str
    branch_driver_id: str
    simulations: int
    sampled_worlds: int
    c_puct: float
    max_steps: int


@dataclass(frozen=True)
class RetainedFileSnapshot:
    """Content identity for one retained evidence file."""

    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RetainedCheckpointSnapshot:
    """One selected checkpoint registration and its current file identity."""

    checkpoint_id: str
    training_seed: int
    checkpoint_sha256: str
    checkpoint_bytes: int
    manifest_sha256: str
    world_id: str
    observation_abi_sha256: str
    action_abi_sha256: str
    parameter_count: int
    value_mode: str
    branch_driver_id: str
    simulations: int
    sampled_worlds: int
    c_puct: float
    max_steps: int
    file: RetainedFileSnapshot


@dataclass(frozen=True)
class RetainedInt7EvidenceSnapshot:
    """Closed before/after receipt for the retained INT-7 evidence root."""

    manifest_digest: str
    files: tuple[RetainedFileSnapshot, ...]
    selected_checkpoints: tuple[RetainedCheckpointSnapshot, ...]

    @property
    def manifest_file(self) -> RetainedFileSnapshot:
        return next(
            file for file in self.files if file.relative_path == "manifest.json"
        )

    @property
    def retention_tree_files(self) -> int:
        return len(self.files)

    @property
    def retention_tree_bytes(self) -> int:
        return sum(file.bytes for file in self.files)

    @property
    def retention_tree_content_sha256(self) -> str:
        return _canonical_sha256([asdict(file) for file in self.files])

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(
            {
                "manifest_digest": self.manifest_digest,
                "files": [asdict(file) for file in self.files],
                "selected_checkpoints": [
                    asdict(checkpoint) for checkpoint in self.selected_checkpoints
                ],
            }
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def retained_int7_manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash canonical INT-7 manifest contents without its self-declared digest."""

    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return _canonical_sha256(unsigned)


def model_observation_abi_sha256(observation_space: Any) -> str:
    """Hash the exact tensor names, shapes, and dtypes consumed by a manabot."""

    schema = {
        name: {"shape": list(shape), "dtype": "float32"}
        for name, shape in sorted(observation_space.shapes.items())
    }
    return _canonical_sha256(schema)


def model_action_abi_sha256(observation_space: Any) -> str:
    """Hash the model action enum and encoded action tensor limits."""

    from manabot.env.observation import ActionEnum

    action_abi = {
        "enum": {member.name: int(member.value) for member in ActionEnum},
        "max_actions": observation_space.encoder.max_actions,
        "actions_shape": list(observation_space.shapes["actions"]),
        "action_focus_shape": list(observation_space.shapes["action_focus"]),
        "actions_valid_shape": list(observation_space.shapes["actions_valid"]),
    }
    return _canonical_sha256(action_abi)


def _load_int7_manifest() -> dict[str, Any]:
    if not INT7_VALUE_TARGET_MANIFEST.is_file():
        raise RetainedCheckpointUnavailableError(
            "retained INT-7 checkpoint manifest is unavailable"
        )
    try:
        manifest = json.loads(INT7_VALUE_TARGET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 checkpoint manifest is invalid"
        ) from exc
    actual_digest = retained_int7_manifest_digest(manifest)
    if (
        actual_digest != INT7_VALUE_TARGET_MANIFEST_SHA256
        or manifest.get("manifest_sha256") != INT7_VALUE_TARGET_MANIFEST_SHA256
    ):
        raise RetainedCheckpointMismatchError(
            "retained INT-7 checkpoint manifest identity drifted"
        )
    return manifest


def retained_int7_policy_only_checkpoint(
    training_seed: int,
) -> RetainedCheckpointRegistration:
    """Resolve and verify one admissible INT-7 policy-only checkpoint.

    The caller selects only a frozen seed. Manifest lookup derives the exact
    repository path; arbitrary checkpoint paths never enter this boundary.
    """

    expected_sha256 = _INT7_POLICY_ONLY_SHA256.get(int(training_seed))
    if expected_sha256 is None:
        raise RetainedCheckpointMismatchError(
            "INT-7 checkpoint seed is not in the retained policy-only registry"
        )
    manifest = _load_int7_manifest()
    rows = [
        row
        for row in manifest.get("candidates", [])
        if row.get("arm") == "visit_policy_only"
        and row.get("training_seed") == int(training_seed)
    ]
    if len(rows) != 1:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 policy-only registration is missing or ambiguous"
        )
    row = rows[0]
    checkpoint_bytes = 428_629
    checkpoint_id = (
        f"int-7-visit_policy_only-seed-{training_seed}-{expected_sha256[:16]}"
    )
    required = {
        "artifact_id": checkpoint_id,
        "checkpoint_sha256": expected_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "world": "w2",
        "observation_abi_sha256": (
            "d7da94347d956dd492039d55a10291bcdc1b2605e59f67dfe7af66d3fbbbaaa5"
        ),
        "action_abi_sha256": (
            "9f1290e4db495c9a299df0bbf06f19fd02cf1eabc62b435f8b029ddd5f6257b2"
        ),
        "parameter_count": 102_722,
        "value_mode": "neutral",
    }
    drifted = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in required.items()
        if row.get(key) != expected
    }
    player_spec = row.get("player_spec")
    if not isinstance(player_spec, dict):
        drifted["player_spec"] = {"expected": "object", "actual": player_spec}
        player_spec = {}
    expected_spec = {
        "kind": "int7_checkpoint_puct",
        "branch_driver_id": "full_clone/current_game_v1",
        "sims": 32,
        "worlds": 4,
        "c_puct": 1.5,
        "max_steps": DEFAULT_MAX_PLAYOUT_STEPS,
        "value_mode": "neutral",
    }
    drifted.update(
        {
            f"player_spec.{key}": {"expected": expected, "actual": player_spec.get(key)}
            for key, expected in expected_spec.items()
            if player_spec.get(key) != expected
        }
    )
    if drifted:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 policy-only registration drifted: "
            + json.dumps(drifted, sort_keys=True)
        )

    checkpoint_path = (
        INT7_VALUE_TARGET_RESULT
        / "checkpoints"
        / f"visit_policy_only-seed-{training_seed}.pt"
    )
    if not checkpoint_path.is_file():
        raise RetainedCheckpointUnavailableError(
            "retained INT-7 policy-only checkpoint bytes are unavailable"
        )
    if checkpoint_path.stat().st_size != checkpoint_bytes:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 policy-only checkpoint byte length drifted"
        )
    if _sha256_file(checkpoint_path) != expected_sha256:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 policy-only checkpoint SHA-256 drifted"
        )

    return RetainedCheckpointRegistration(
        checkpoint_id=checkpoint_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=expected_sha256,
        checkpoint_bytes=checkpoint_bytes,
        manifest_sha256=INT7_VALUE_TARGET_MANIFEST_SHA256,
        training_seed=int(training_seed),
        world_id=str(row["world"]),
        observation_abi_sha256=str(row["observation_abi_sha256"]),
        action_abi_sha256=str(row["action_abi_sha256"]),
        parameter_count=int(row["parameter_count"]),
        value_mode=str(row["value_mode"]),
        branch_driver_id=str(player_spec["branch_driver_id"]),
        simulations=int(player_spec["sims"]),
        sampled_worlds=int(player_spec["worlds"]),
        c_puct=float(player_spec["c_puct"]),
        max_steps=int(player_spec["max_steps"]),
    )


def _retained_file_snapshot(path: Path) -> RetainedFileSnapshot:
    try:
        digest = _sha256_file(path)
        byte_length = path.stat().st_size
    except FileNotFoundError as exc:
        raise RetainedCheckpointUnavailableError(
            "retained INT-7 evidence file is unavailable"
        ) from exc
    try:
        relative_path = path.relative_to(INT7_VALUE_TARGET_RESULT).as_posix()
    except ValueError as exc:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 evidence escaped its frozen root"
        ) from exc
    return RetainedFileSnapshot(
        relative_path=relative_path,
        bytes=byte_length,
        sha256=digest,
    )


def retained_int7_evidence_snapshot(
    training_seeds: tuple[int, ...],
) -> RetainedInt7EvidenceSnapshot:
    """Snapshot the closed retained root and exact selected registrations."""

    selected_seeds = tuple(sorted({int(seed) for seed in training_seeds}))
    if not selected_seeds:
        raise RetainedCheckpointMismatchError(
            "retained INT-7 evidence snapshot requires a selected checkpoint"
        )
    manifest = _load_int7_manifest()
    registrations = tuple(
        retained_int7_policy_only_checkpoint(seed) for seed in selected_seeds
    )
    if not INT7_VALUE_TARGET_RESULT.is_dir():
        raise RetainedCheckpointUnavailableError(
            "retained INT-7 evidence root is unavailable"
        )
    files = tuple(
        _retained_file_snapshot(path)
        for path in sorted(INT7_VALUE_TARGET_RESULT.rglob("*"))
        if path.is_file()
    )
    by_relative_path = {file.relative_path: file for file in files}
    manifest_file = by_relative_path.get("manifest.json")
    if manifest_file is None:
        raise RetainedCheckpointUnavailableError(
            "retained INT-7 checkpoint manifest is unavailable"
        )
    selected = tuple(
        RetainedCheckpointSnapshot(
            checkpoint_id=registration.checkpoint_id,
            training_seed=registration.training_seed,
            checkpoint_sha256=registration.checkpoint_sha256,
            checkpoint_bytes=registration.checkpoint_bytes,
            manifest_sha256=registration.manifest_sha256,
            world_id=registration.world_id,
            observation_abi_sha256=registration.observation_abi_sha256,
            action_abi_sha256=registration.action_abi_sha256,
            parameter_count=registration.parameter_count,
            value_mode=registration.value_mode,
            branch_driver_id=registration.branch_driver_id,
            simulations=registration.simulations,
            sampled_worlds=registration.sampled_worlds,
            c_puct=registration.c_puct,
            max_steps=registration.max_steps,
            file=by_relative_path[
                registration.checkpoint_path.relative_to(
                    INT7_VALUE_TARGET_RESULT
                ).as_posix()
            ],
        )
        for registration in registrations
    )
    return RetainedInt7EvidenceSnapshot(
        manifest_digest=retained_int7_manifest_digest(manifest),
        files=files,
        selected_checkpoints=selected,
    )


@dataclass
class SearchStats:
    """Accumulated cost and behavior for one search player."""

    decisions: int = 0
    seconds: float = 0.0
    simulations: int = 0
    cap_hits: int = 0
    decision_seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "decisions": float(self.decisions),
            "seconds": self.seconds,
            "simulations": float(self.simulations),
            "cap_hits": float(self.cap_hits),
        }
