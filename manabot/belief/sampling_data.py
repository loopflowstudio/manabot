"""Frozen-policy self-play evidence for constrained hidden-hand sampling.

Inputs are snapshotted from the viewer before accessing supervision-only hands.
Games are the split unit; artifacts retain checkpoint and complete world bindings.
The native constraint projection shares exact-range semantics without enumeration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Literal, cast

import torch

from manabot.belief.sampling import SamplerInput, SamplerSchema
from manabot.belief.state import ViewerHistory
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.flat_mc import AgentMatchupPlayer, load_checkpoint_agent
from managym.decision import (
    PUBLIC_COMMITMENT_KINDS,
    Command,
    DecisionFrame,
    Observation,
)

Split = Literal["train", "validation", "test"]
HISTORY_SCHEMA = "manabot.public-commitment-counts/actor-kind-card-sqrt-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
    games: list[SamplerGame] = []
    for game in data["games"]:
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
        games.append(
            SamplerGame(
                game["game_id"],
                game["seed"],
                game["assignment"],
                game["split"],
                tuple(rows),
            )
        )
    dataset = SamplerDataset(
        schema, data["policy_identity"], data["world_identity"], tuple(games)
    )
    if dataset.identity != payload["identity"]:
        raise ValueError("belief dataset content digest mismatch")
    return dataset


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


def collect_frozen_policy(
    *,
    checkpoint: Path,
    match_hypers: MatchHypers,
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
    if games < 3 or max_steps < 1:
        raise ValueError("three-way whole-game split requires at least three games")
    policy_identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, obs_space = load_checkpoint_agent(str(checkpoint))
    if agent.belief_count_buckets:
        raise ValueError(
            "frozen sampler collection requires an observation-only behavior policy"
        )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != policy_identity:
        raise ValueError("behavior checkpoint changed while loading")
    player = AgentMatchupPlayer(agent)
    match = Match(match_hypers)
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed)
    order = list(range(games))
    random.Random(seed).shuffle(order)
    held_out = max(1, games // 5)
    split_by_game: dict[int, Split] = {index: "train" for index in order}
    for index in order[:held_out]:
        split_by_game[index] = "validation"
    for index in order[held_out : 2 * held_out]:
        split_by_game[index] = "test"
    schema: SamplerSchema | None = None
    records: list[SamplerGame] = []
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(seed)
        for game_index in range(games):
            if check is not None:
                check()
            obs, _ = env.reset(
                seed=seed + game_index,
                options={"match": match if game_index % 2 == 0 else match.swapped()},
            )
            # The collector owns game boundaries; discard any queued compound suffix.
            player.start_game(env, seat=0)
            manifest = env._engine.content_pack_manifest()
            if schema is None:
                # Restrict work to configured definitions; generated out-of-roster
                # history or hidden cards fail explicitly rather than truncating.
                names = tuple(
                    sorted(
                        set(match.hero_deck)
                        | set(match.villain_deck)
                        | set(match.hero_sideboard)
                        | set(match.villain_sideboard)
                    )
                )
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
            histories = [
                ViewerHistory.from_observation(
                    Observation.from_json(env._engine.semantic_observation_json(viewer))
                )
                for viewer in range(2)
            ]
            examples: list[SamplerExample] = []
            for step in range(max_steps):
                if check is not None:
                    check()
                for viewer in range(2):
                    projected = json.loads(
                        env._engine.hidden_hand_constraints_json(viewer)
                    )
                    pool = _counts(projected["pool"])
                    known = _counts(projected["known_hand"])
                    if (set(pool) | set(known)) - set(schema.card_names):
                        raise ValueError("native constraint vocabulary changed")
                    source = projected["source_observation"]
                    history = histories[viewer]
                    if (
                        source["revision"] != history.current_revision
                        or source["viewer_state_hash"]
                        != history.current_viewer_state_hash
                    ):
                        raise ValueError(
                            "belief input history does not match current observation"
                        )
                    inputs = SamplerInput(
                        schema.vocabulary_identity,
                        tuple(pool.get(name, 0) for name in schema.card_names),
                        tuple(known.get(name, 0) for name in schema.card_names),
                        projected["hand_size"],
                        history_features(history, schema),
                    )
                    inputs.validate(schema)
                    # Privileged label access begins only after all inference inputs exist.
                    authority = env._engine.observation_for_player(1 - viewer)
                    hand = Counter(
                        str(card.name)
                        for card in authority.agent_cards
                        if int(card.zone) == 1
                    )
                    examples.append(
                        SamplerExample(
                            inputs,
                            tuple(hand[name] for name in schema.card_names),
                            viewer,
                            history.current_revision,
                            history.current_viewer_state_hash,
                        )
                    )
                frame = DecisionFrame.from_json(
                    env._engine.semantic_decision_frame_json()
                )
                if len(frame.offers) != int(obs["actions_valid"].sum()):
                    raise ValueError("frozen policy cannot encode every legal offer")
                action = player.act(env, obs)
                command = Command(
                    f"sampler-{seed}-{game_index}-{step}",
                    frame.revision,
                    int(frame.offers[action]["id"]),
                )
                obs, _, terminated, truncated, _, transition = env.step_semantic(
                    command
                )
                histories = [
                    history.advance(
                        transition.receipt,
                        Observation.from_json(
                            env._engine.semantic_observation_json(viewer)
                        ),
                        acting=frame.actor,
                    )
                    for viewer, history in enumerate(histories)
                ]
                if truncated:
                    raise RuntimeError("belief self-play game truncated")
                if terminated:
                    break
            else:
                raise RuntimeError("belief self-play exceeded the declared step cap")
            records.append(
                SamplerGame(
                    f"{policy_identity}:{seed + game_index}:{game_index % 2}",
                    seed + game_index,
                    game_index % 2,
                    split_by_game[game_index],
                    tuple(examples),
                )
            )
    assert schema is not None
    return SamplerDataset(
        schema,
        policy_identity,
        _digest(getattr(agent, "world_binding")),
        tuple(records),
    )
