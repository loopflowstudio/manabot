"""Dependency-light limits, receipts, and retained search registrations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from manabot.infra.identity import canonical_sha256, file_sha256

DEFAULT_MAX_PLAYOUT_STEPS = 2000

REPO_ROOT = Path(__file__).resolve().parents[3]
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


def retained_int7_manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash canonical INT-7 manifest contents without its self-declared digest."""

    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


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
    if file_sha256(checkpoint_path) != expected_sha256:
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
