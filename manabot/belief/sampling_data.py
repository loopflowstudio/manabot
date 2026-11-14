"""Frozen-policy self-play evidence for constrained hidden-hand sampling.

Inputs are snapshotted from the viewer before accessing supervision-only hands.
Games are the split unit; artifacts retain checkpoint and complete world bindings.
The native constraint projection shares exact-range semantics without enumeration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from manabot.belief.sampling_types import SamplerInput, SamplerSchema, _digest
from manabot.belief.state import ViewerHistory
from managym.decision import (
    PUBLIC_COMMITMENT_KINDS,
)

if TYPE_CHECKING:
    from manabot.infra.config import MatchConfig


class ConstraintSource(Protocol):
    def hidden_hand_constraints_json(self, viewer: int) -> str: ...


class UnsupportedSamplerRow(ValueError):
    """A recognized native domain outside the count-only sampler's support."""


Split = Literal["train", "validation", "test"]
HISTORY_SCHEMA = "manabot.public-commitment-counts/actor-kind-card-sqrt-v1"


@dataclass(frozen=True, slots=True)
class SamplerExample:
    """Joint hand label is private supervision, never a sampler input."""

    inputs: SamplerInput
    target_hand: tuple[int, ...]
    viewer: int
    revision: int
    observation_identity: str


@dataclass(frozen=True, slots=True)
class SamplerGame:
    game_id: str
    seed: int
    assignment: int
    split: Split
    examples: tuple[SamplerExample, ...]


@dataclass(frozen=True, slots=True)
class SamplerDataset:
    schema: SamplerSchema
    policy_identity: str
    world_identity: str
    games: tuple[SamplerGame, ...]

    @property
    def identity(self) -> str:
        return _digest(asdict(self))

    def __post_init__(self) -> None:
        if len({game.game_id for game in self.games}) != len(self.games) or len(
            {(game.seed, game.assignment) for game in self.games}
        ) != len(self.games):
            raise ValueError("duplicate game identities in belief dataset")
        if {game.split for game in self.games} != {"train", "validation", "test"}:
            raise ValueError("belief dataset requires three nonempty whole-game splits")
        if not self.policy_identity or not self.world_identity:
            raise ValueError(
                "belief dataset requires frozen policy and world identities"
            )
        for game in self.games:
            if not game.examples:
                raise ValueError("belief dataset contains an empty game")
            for row in game.examples:
                row.inputs.validate(self.schema)
                if len(row.target_hand) != len(self.schema.card_names):
                    raise ValueError("belief hand label vocabulary mismatch")
                if sum(row.target_hand) != row.inputs.hand_size or any(
                    type(count) is not int or count < low or count > high
                    for count, low, high in zip(
                        row.target_hand,
                        row.inputs.known_minima,
                        row.inputs.pool_counts,
                        strict=True,
                    )
                ):
                    raise ValueError("belief hand label violates authority constraints")


def save_dataset(dataset: SamplerDataset, path: Path) -> None:
    """Create immutable private training evidence; refuse accidental overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only fully flushed bytes. Linking is atomic and refuses overwrite.
    descriptor, temporary = tempfile.mkstemp(prefix=".belief-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(
                {
                    "schema_version": 1,
                    "identity": dataset.identity,
                    "dataset": asdict(dataset),
                },
                output,
                sort_keys=True,
                allow_nan=False,
            )
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def read_dataset(path: Path) -> SamplerDataset:
    """Load typed immutable rows and verify the complete content digest."""
    payload = json.loads(path.read_text())
    if payload["schema_version"] != 1:
        raise ValueError("unsupported belief dataset version")
    data = payload["dataset"]
    schema = SamplerSchema(
        **{**data["schema"], "card_names": tuple(data["schema"]["card_names"])}
    )
    games = tuple(_read_game(game) for game in data["games"])
    dataset = SamplerDataset(
        schema, data["policy_identity"], data["world_identity"], tuple(games)
    )
    if dataset.identity != payload["identity"]:
        raise ValueError("belief dataset content digest mismatch")
    return dataset


def _read_game(game: dict[str, Any]) -> SamplerGame:
    """Narrow JSON fields at the private artifact boundary."""
    rows: list[SamplerExample] = []
    for row in game["examples"]:
        inputs = row["inputs"]
        rows.append(
            SamplerExample(
                inputs=SamplerInput(
                    **{
                        **inputs,
                        "pool_counts": tuple(inputs["pool_counts"]),
                        "known_minima": tuple(inputs["known_minima"]),
                        "history_features": tuple(inputs["history_features"]),
                    }
                ),
                target_hand=tuple(row["target_hand"]),
                viewer=row["viewer"],
                revision=row["revision"],
                observation_identity=row["observation_identity"],
            )
        )
    return SamplerGame(
        game["game_id"], game["seed"], game["assignment"], game["split"], tuple(rows)
    )


def _counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or type(v) is not int or v < 0 for k, v in value.items()
    ):
        raise ValueError("native belief constraint count map is malformed")
    return cast(dict[str, int], value)


def history_features(
    history: ViewerHistory, schema: SamplerSchema
) -> tuple[float, ...]:
    """Order-invariant typed commitment counts; no opaque hashes or truth features.

    Index order is actor role × public commitment kind × (no-card, vocabulary).
    Square-root event-count normalization preserves multiplicity. This restricted
    summary cannot express event ordering; policy shift evaluation remains required.
    """
    width = len(schema.card_names) + 1
    features = [0.0] * schema.history_size
    names = {name: index + 1 for index, name in enumerate(schema.card_names)}
    for event in history.semantic_events:
        if event.commitment.card is not None and event.commitment.card not in names:
            raise ValueError("public history card is outside frozen sampler vocabulary")
        card = 0 if event.commitment.card is None else names[event.commitment.card]
        kind = PUBLIC_COMMITMENT_KINDS.index(event.commitment.kind)
        features[
            (event.actor_role_id * len(PUBLIC_COMMITMENT_KINDS) + kind) * width + card
        ] += 1.0
    scale = math.sqrt(max(1, len(history.semantic_events)))
    return tuple(value / scale for value in features)


def public_sampler_input(
    engine: ConstraintSource,
    history: ViewerHistory,
    schema: SamplerSchema,
) -> tuple[SamplerInput, str]:
    """Bind public features to their native constraint snapshot, without truth access.

    The returned JSON is the authority token for direct hand materialization.
    Callers must retain history from the source trajectory; opaque event hashes
    are not decoded into invented public commitments.
    """
    encoded = engine.hidden_hand_constraints_json(history.viewer)
    projected = json.loads(encoded)
    if projected.get("retained_selection", False) or projected["foretold"]:
        raise UnsupportedSamplerRow(
            "count-only belief sampler cannot represent retained selections or foretold slots"
        )
    pool = _counts(projected["pool"])
    known = _counts(projected["known_hand"])
    if (set(pool) | set(known)) - set(schema.card_names):
        raise ValueError("native constraint vocabulary changed")
    source = projected["source_observation"]
    if (
        source["revision"] != history.current_revision
        or source["viewer"] != history.viewer
        or source["schema_version"] != history.schema_version
        or source["viewer_state_hash"] != history.current_viewer_state_hash
    ):
        raise ValueError("belief input history does not match current observation")
    inputs = SamplerInput(
        schema.vocabulary_identity,
        tuple(pool.get(name, 0) for name in schema.card_names),
        tuple(known.get(name, 0) for name in schema.card_names),
        projected["hand_size"],
        history_features(history, schema),
    )
    inputs.validate(schema)
    return inputs, encoded


def collect_frozen_policy(
    *,
    checkpoint: Path,
    match_config: MatchConfig,
    games: int,
    seed: int,
    max_steps: int = 2000,
    check: Callable[[], None] | None = None,
) -> SamplerDataset:
    """Complete frozen self-play games; cap/deadline failures reject the dataset.

    Both seats are observed, and deck assignments alternate. Labels are read only
    after immutable viewer inputs are constructed. No optimizer or exact support
    enumeration occurs. A caller owns attempt/failure and elapsed-cost receipts.
    """
    # Keep runtime imports at the entry point: saved data and scheduled capture
    # also run in an isolated historical policy runtime without today's loader.
    import torch

    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.belief.sampling_collection import CaptureSlot, capture_game
    from manabot.belief.sampling_runtime import EnvCaptureRuntime
    from manabot.env.env import Env
    from manabot.env.match import Match, Reward
    from manabot.infra.config import RewardConfig
    from manabot.sim.search.flat_mc import AgentMatchupPlayer

    if games < 3 or max_steps < 1:
        raise ValueError("three-way whole-game split requires at least three games")
    policy_identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, obs_space = load_checkpoint(str(checkpoint))
    if agent.belief_count_buckets:
        raise ValueError(
            "frozen sampler collection requires an observation-only behavior policy"
        )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != policy_identity:
        raise ValueError("behavior checkpoint changed while loading")
    match = Match(match_config)
    env = Env(match, obs_space, Reward(RewardConfig()), seed=seed)
    runtime = EnvCaptureRuntime(
        env,
        AgentMatchupPlayer(agent),
        lambda slot: match if slot.assignment == 0 else match.swapped(),
    )
    order = list(range(games))
    random.Random(seed).shuffle(order)
    held_out = max(1, games // 5)
    split_by_game: dict[int, Split] = {index: "train" for index in order}
    for index in order[:held_out]:
        split_by_game[index] = "validation"
    for index in order[held_out : 2 * held_out]:
        split_by_game[index] = "test"
    # The public wrapper retains its matchup-specific schema and strict behavior.
    names = tuple(
        sorted(
            set(match.hero_deck)
            | set(match.villain_deck)
            | set(match.hero_sideboard)
            | set(match.villain_sideboard)
        )
    )
    runtime.reset(CaptureSlot("schema", seed, 0, "train", ("hero", "villain")))
    manifest = env._engine.content_pack_manifest()
    identity = _digest(
        {
            "content": manifest["content_digest"],
            "names": names,
            "history": HISTORY_SCHEMA,
            "kinds": PUBLIC_COMMITMENT_KINDS,
        }
    )
    schema = SamplerSchema(
        identity, names, 2 * len(PUBLIC_COMMITMENT_KINDS) * (len(names) + 1)
    )
    records: list[SamplerGame] = []
    try:
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(seed)
            for index in range(games):
                slot = CaptureSlot(
                    f"{policy_identity}:{seed + index}:{index % 2}",
                    seed + index,
                    index % 2,
                    split_by_game[index],
                    ("hero", "villain"),
                )
                result = capture_game(
                    runtime,
                    slot,
                    schema,
                    max_steps=max_steps,
                    moments=None,
                    check=check,
                )
                if result.failure is not None:
                    raise RuntimeError(result.failure)
                if result.game is None:
                    raise ValueError("completed game has no representable rows")
                records.append(result.game)
    finally:
        env.close()
    return SamplerDataset(
        schema,
        policy_identity,
        _digest(getattr(agent, "world_binding")),
        tuple(records),
    )
