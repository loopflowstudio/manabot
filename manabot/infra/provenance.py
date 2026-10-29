"""Shared runtime fingerprints and replay identity contracts.

Frozen registrations retain their existing field names and digest algorithms.
Source identities bind relative paths as well as bytes; relocating source changes
identity and never grants automatic admission to an older execution.
"""

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from managym.experience import UNVERSIONED_CONTENT_HASH

from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ActionEnum, ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.verify.games import INTERACTIVE_DECK
import managym

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SCHEMA = REPO_ROOT / "protocol" / "experience-v1.schema.json"
# Recorded protocol-v1 frames name the curated asset pack Academy serves by
# default. manabot records its digest as provenance and never loads the pack;
# tests/academy/test_curated_pack.py checks it equals Academy's served pack.
RECORDED_ASSET_MANIFEST = (
    REPO_ROOT
    / "academy"
    / "ui"
    / "src"
    / "lib"
    / "packs"
    / "tla-ur-lessons-vs-gw-allies"
    / "v2"
    / "manifest.json"
)


class ContractError(ValueError):
    """The checked-in experiment contract does not match the runtime."""


RECORDED_ASSET_MANIFEST_HASH = file_sha256(RECORDED_ASSET_MANIFEST)


def source_bundle_sha256(paths: Iterable[Path]) -> str:
    """Hash relative names and bytes for a stable, self-contained source ID."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(REPO_ROOT))):
        relative = str(path.relative_to(REPO_ROOT)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def engine_source_paths() -> list[Path]:
    paths = list((REPO_ROOT / "managym" / "src").rglob("*.rs"))
    paths.extend(
        [
            REPO_ROOT / "managym" / "Cargo.lock",
            REPO_ROOT / "manabot" / "sim" / "search" / "puct.py",
            REPO_ROOT / "manabot" / "sim" / "search" / "flat_mc.py",
        ]
    )
    return paths


def runtime_fingerprints(
    seed: int = 197,
    *,
    match_config: MatchConfig | None = None,
    observation_space: ObservationSpace | None = None,
    world: str = managym.WORLD_VERSION,
) -> dict[str, Any]:
    """Return identities that the pre-registration freezes before a run."""
    if world != managym.WORLD_VERSION:
        raise ValueError(
            f"Runtime is {managym.WORLD_VERSION}; cannot register it as {world}"
        )

    obs_space = observation_space or ObservationSpace()
    matchup = match_config or MatchConfig(
        hero="teacher-a",
        villain="teacher-b",
        hero_deck=dict(INTERACTIVE_DECK),
        villain_deck=dict(INTERACTIVE_DECK),
    )
    env = Env(
        Match(matchup),
        obs_space,
        Reward(RewardConfig()),
        seed=seed,
        auto_reset=False,
    )
    env.reset(seed=seed)
    content = env.content_pack_manifest()
    observation_schema = {
        name: {"shape": list(shape), "dtype": "float32"}
        for name, shape in sorted(obs_space.shapes.items())
    }
    action_abi = {
        "enum": {member.name: int(member.value) for member in ActionEnum},
        "max_actions": obs_space.encoder.max_actions,
        "actions_shape": list(obs_space.shapes["actions"]),
        "action_focus_shape": list(obs_space.shapes["action_focus"]),
        "actions_valid_shape": list(obs_space.shapes["actions_valid"]),
    }
    extension_path = Path(managym._managym.__file__)
    return {
        "world": world,
        "engine_source_sha256": source_bundle_sha256(engine_source_paths()),
        "engine_extension_sha256": file_sha256(extension_path),
        "engine_extension_name": extension_path.name,
        "content_schema_version": content.get("schema_version"),
        "content_digest": content.get("content_digest"),
        "content_manifest_sha256": canonical_sha256(content),
        "observation_abi_sha256": canonical_sha256(observation_schema),
        "action_abi_sha256": canonical_sha256(action_abi),
        "experience_protocol_sha256": file_sha256(PROTOCOL_SCHEMA),
        "experience_content_hash": UNVERSIONED_CONTENT_HASH,
        "asset_manifest_hash": RECORDED_ASSET_MANIFEST_HASH,
        "matchup_sha256": canonical_sha256(matchup.model_dump()),
        "matchup": matchup.model_dump(),
    }


def authored_matchup_sha256(identities: Mapping[str, Any], consumer: str) -> str:
    """The authored setup a run trained on, for consumers registered against one.

    A deck run samples a supply (`training_decks_sha256`) and has no single
    matchup; such consumers reject it rather than read a representative pair.
    """
    if "training_decks_sha256" in identities:
        raise ValueError(
            f"{consumer} requires an authored matchup; this run sampled training "
            f"decks {identities['training_decks_sha256']} from pool "
            f"{identities['content_pool']}"
        )
    return identities["matchup_sha256"]


def validate_runtime_fingerprints(
    expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    """Fail closed for every identity explicitly frozen by the contract."""

    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ContractError(
            "runtime does not match the pre-registration: "
            + json.dumps(mismatches, sort_keys=True)
        )
