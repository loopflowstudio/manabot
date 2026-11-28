"""Game-side Etude players: admitted host preparation and seeded action choice.

Etude owns model forwards. Game workers verify checkpoint bytes and keep only
host preparation metadata, then sample legal actions with each seat's seed.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Literal, cast

from etude.checkpoints import CheckpointRef
from etude.eval.agents import Forward, ForwardError, ModelForward
from etude.eval.records import CheckpointPlayer, Code, Match as EtudeMatch, Player
from etude.jobs import Agents
import numpy as np

from manabot.env.decision_memory import blank_memory
from manabot.env.env import Env
from manabot.env.match import Match
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.eval.agent_loader import load, load_bf16, load_tf32
from manabot.infra.config import MatchConfig
from manabot.infra.identity import file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.world import validate_setup_binding
from manabot.policy.admission import Spec, admit_native, check, legal_mask
from manabot.policy.checkpoint import read_container
from manabot.policy.prepare import Layout, Prepared, Tables, prepare
from manabot.policy.spans import ObservationSelection
from manabot.sim.agent import GameChoice, GamePlayer
from manabot.sim.heuristic import HEURISTIC_VERSION, HeuristicPlayer
from manabot.sim.players import CodePlayer, build_code_player
from manabot.sim.registration import (
    ETUDE_SAMPLER_SPEC,
    EvalPrecision,
    PlayerRegistration,
    etude_compute_class,
)


@dataclass(frozen=True)
class CheckpointInputs:
    """Admitted host metadata for a checkpoint, without its weights."""

    spec: Spec
    world_binding: Mapping[str, Any]
    sha256: str
    size_bytes: int
    writer_revision: str | None
    stored_programs: int
    parameter_count: int

    @property
    def space(self) -> ObservationSpace:
        return self.spec.space

    def admit(self, match: MatchConfig, *, allow_deck_repetition: bool) -> None:
        validate_setup_binding(
            self.world_binding,
            Match(match).to_rust(),
            allow_deck_repetition=allow_deck_repetition,
        )


def read_checkpoint_inputs(
    path: Path, sha256: str | None, size_bytes: int | None
) -> CheckpointInputs:
    digest = file_sha256(path)
    if size_bytes is None or sha256 != digest or size_bytes != path.stat().st_size:
        raise ValueError("checkpoint differs from its recorded bytes")
    checkpoint = read_container(path)
    # NumPy views of these buffers otherwise keep the checkpoint's weights alive.
    tables = Tables.from_buffers(
        {key: value.copy() for key, value in checkpoint.buffers.items()},
        Layout(checkpoint.readings),
        fact_kinds=checkpoint.fact_kinds,
    )
    return CheckpointInputs(
        spec=Spec(
            checkpoint.model,
            checkpoint.space,
            tables,
            checkpoint.parameters["card_definitions.weight"].shape[0]
            if "card_definitions.weight" in checkpoint.parameters
            else 0,
        ),
        world_binding=json.loads(checkpoint.metadata["world_binding"]),
        sha256=digest,
        size_bytes=size_bytes,
        writer_revision=checkpoint.writer_revision,
        stored_programs=checkpoint.stored_programs,
        parameter_count=sum(int(p.size) for p in checkpoint.parameters.values()),
    )


type CheckpointReader = Callable[[CheckpointRef], CheckpointInputs]


def checkpoint_reader(capacity: int) -> CheckpointReader:
    """Reuse verified metadata within one arena, while bounding resident tables."""
    read = lru_cache(maxsize=capacity)(read_checkpoint_inputs)

    def inputs(ref: CheckpointRef) -> CheckpointInputs:
        return read(ref.path, ref.artifact.sha256, ref.artifact.size_bytes)

    return inputs


def prepare_batch(
    batch: ObservationBuffers | ObservationSelection, spec: Spec
) -> Prepared[np.ndarray]:
    """Admit and prepare the batch after its callers apply seat-level ablations."""
    if isinstance(batch, ObservationSelection):
        admitted = admit_native(batch, spec, bucketed=False)
    else:
        check(batch, spec, legal_mask(batch["actions_valid"]))
        admitted = batch
    model = spec.model
    return replace(
        prepare(
            admitted,
            spec.tables,
            binding=spec.binding,
            memory_summaries=model.decision_memory_summaries
            if model.decision_memory != "none"
            else 0,
        ),
        programs=None,
    )


@dataclass(frozen=True)
class Sampler:
    checkpoint_key: str
    forward: Forward
    deterministic: bool = False
    allow_deck_repetition: bool = False
    # Memory ablation: the policy sees the live board with an empty decision log.
    blank_decision_memory: bool = False

    async def logits(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        outputs = await self.forward([batch], rows=batch.batch)
        elapsed = time.perf_counter() - started
        try:
            logits = checked_logits(outputs, batch)
        except (TypeError, ValueError) as exc:
            raise ForwardError(self.checkpoint_key, str(exc)) from exc
        return logits, elapsed

    def sample(self, logits: np.ndarray, legal: np.ndarray, seed: int) -> int:
        slots = np.flatnonzero(legal)
        if self.deterministic:
            return int(slots[logits[slots].argmax()])
        weights = np.exp(logits[slots].astype(np.float64) - logits[slots].max())
        return int(np.random.default_rng(seed).choice(slots, p=weights / weights.sum()))


def checked_logits(outputs: list[object], batch: Prepared[np.ndarray]) -> np.ndarray:
    if len(outputs) != 1:
        raise ValueError(
            f"forward returned {len(outputs)} outputs for 1 prepared input"
        )
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.shape != batch.legal.shape or not np.isfinite(logits[batch.legal]).all():
        raise ValueError("forward returned malformed legal logits")
    return logits


@dataclass(frozen=True)
class ScalarSampler:
    """Synchronous player for the scalar arena, using the same host sampler."""

    sampler: Sampler
    inputs: CheckpointInputs
    logits: Callable[[Prepared[np.ndarray]], np.ndarray]

    @property
    def space(self) -> ObservationSpace:
        return self.inputs.space

    def admit(self, match: MatchConfig) -> None:
        self.inputs.admit(
            match, allow_deck_repetition=self.sampler.allow_deck_repetition
        )

    def choose(self, env: Env, obs: ObservationBuffers, seed: int) -> int:
        if self.sampler.blank_decision_memory:
            obs = blank_memory(obs)
        batch = prepare_batch(
            {key: value[None] for key, value in obs.items()}, self.inputs.spec
        )
        return self.sampler.sample(self.logits(batch)[0], batch.legal[0], seed)

    def open_game(self, seed: int) -> GameChoice:
        return self.choose


def code_choice(player: CodePlayer) -> GameChoice:
    def choose(env: Env, obs: ObservationBuffers, seed: int) -> int:
        return player.act(env, obs)

    return choose


@dataclass(frozen=True)
class CodeSeat:
    factory: Callable[[int], CodePlayer]
    space: None = None

    def admit(self, match: MatchConfig) -> None:
        pass

    def open_game(self, seed: int) -> GameChoice:
        return code_choice(self.factory(seed))


@dataclass(frozen=True)
class NativeHeuristic:
    max_actions: int | None
    space: None = None

    def __post_init__(self) -> None:
        if self.max_actions is not None and (
            type(self.max_actions) is not int or not 0 < self.max_actions < 2**32
        ):
            raise ValueError("heuristic max_actions must be a positive u32")

    def admit(self, match: MatchConfig) -> None:
        pass

    @property
    def player_policy(self) -> str:
        return (
            "heuristic" if self.max_actions is None else f"heuristic:{self.max_actions}"
        )

    def open_game(self, seed: int) -> GameChoice:
        return code_choice(HeuristicPlayer(max_actions=self.max_actions))


def blank_decision_memory(player: CheckpointPlayer) -> bool:
    """Whether the seat declares the memory ablation; only a literal true enables it."""
    blank = player.config.get("blank_decision_memory", False)
    if not isinstance(blank, bool):
        raise ValueError("blank_decision_memory must be a boolean")
    return blank


type Seat = Sampler | CodeSeat | NativeHeuristic


def require_heuristic_version(player: Code) -> None:
    """Bind the built-in heuristic's behavior to the Etude player key."""
    version = player.config.get("version")
    if version != HEURISTIC_VERSION:
        raise ValueError(
            f"built-in heuristic requires version {HEURISTIC_VERSION!r}; "
            f"got {version!r}. Use a new tournament for the current version"
        )


def build(player: object, leaves: Mapping[str, Forward]) -> Seat:
    if isinstance(player, CheckpointPlayer):
        registration = player.config.get("registration", {})
        player_spec = (
            registration.get("player_spec", {})
            if isinstance(registration, dict)
            else {}
        )
        return Sampler(
            checkpoint_key=player.checkpoint.key,
            forward=leaves[player.checkpoint.key],
            deterministic=bool(player_spec.get("deterministic", False)),
            allow_deck_repetition=bool(player_spec.get("allow_deck_repetition", False)),
            blank_decision_memory=blank_decision_memory(player),
        )
    if isinstance(player, Code):
        if "registration" in player.config:
            registration = PlayerRegistration.model_validate(
                player.config["registration"]
            )
            return CodeSeat(lambda seed: build_code_player(registration, seed=seed))
        if player.name != "heuristic":
            raise TypeError(f"Code player {player.name} has no Manabot builder")
        require_heuristic_version(player)
        return NativeHeuristic(player.config.get("max_actions"))
    raise TypeError(f"unsupported Manabot player {type(player).__name__}")


AGENTS = Agents(load=load, build=build)
AGENTS_TF32 = Agents(load=load_tf32, build=build)
AGENTS_BF16 = Agents(load=load_bf16, build=build)

_AGENTS_BY_PRECISION: dict[EvalPrecision, Agents] = {
    "float32": AGENTS,
    "tf32": AGENTS_TF32,
    "bf16": AGENTS_BF16,
}


def agents_for(precision: EvalPrecision = "float32") -> Agents:
    """Return the Etude `Agents` declaration for a declared evaluation precision."""
    if precision not in _AGENTS_BY_PRECISION:
        allowed = ", ".join(_AGENTS_BY_PRECISION)
        raise ValueError(
            f"unsupported eval precision {precision!r}; choose from {allowed}"
        )
    return _AGENTS_BY_PRECISION[precision]


def game_players(players: Sequence[Player], agents: Sequence[object]) -> list[Seat]:
    """Bind bare Etude checkpoint forwards, preserving configured seats."""
    return [
        build(player, {player.checkpoint.key: cast(Forward, agent)})
        if isinstance(player, CheckpointPlayer) and not isinstance(agent, Sampler)
        else cast(Seat, agent)
        for player, agent in zip(players, agents, strict=True)
    ]


def local_scalar_sampler(
    player: CheckpointPlayer, forward: ModelForward, read: CheckpointReader
) -> ScalarSampler:
    sampler = build(player, {player.checkpoint.key: local_forward(forward)})
    assert isinstance(sampler, Sampler)
    return ScalarSampler(
        sampler,
        read(player.checkpoint),
        lambda batch: checked_logits(forward([batch]), batch),
    )


async def play_scalar[MatchT: EtudeMatch, Outcome](
    play: Callable[[MatchT, Sequence[GamePlayer]], Outcome],
    match: MatchT,
    agents: Sequence[object],
) -> Outcome:
    """Run the scalar arena in a thread, sending forwards through Etude's event loop."""
    loop = asyncio.get_running_loop()
    read = checkpoint_reader(len(match.players))

    def play_in_thread() -> Outcome:
        seated: list[GamePlayer] = []
        for player, agent in zip(
            match.players, game_players(match.players, agents), strict=True
        ):
            if isinstance(agent, Sampler):
                assert isinstance(player, CheckpointPlayer)

                def logits(
                    batch: Prepared[np.ndarray], sampler: Sampler = agent
                ) -> np.ndarray:
                    future = asyncio.run_coroutine_threadsafe(
                        sampler.logits(batch), loop
                    )
                    return future.result()[0]

                seated.append(ScalarSampler(agent, read(player.checkpoint), logits))
            else:
                seated.append(agent)
        return play(match, seated)

    return await asyncio.to_thread(play_in_thread)


def local_forward(forward: ModelForward) -> Forward:
    """Present a synchronous local model to the vector runner's forward interface."""

    async def call(inputs: list[object], *, rows: int | None = None) -> list[object]:
        return forward(inputs)

    return call


def registration(
    player: CheckpointPlayer,
    *,
    name: str,
    match: MatchConfig,
    content_suite: str,
    role: Literal["anchor", "incumbent", "challenger"],
    precision: EvalPrecision = "float32",
    read: CheckpointReader | None = None,
) -> PlayerRegistration:
    """Admit local checkpoint bytes and its world without importing a framework.

    A seat declaring `blank_decision_memory` registers the same bytes as a
    distinct memory-ablation player.
    """
    ref = player.checkpoint
    checkpoint = (
        read(ref)
        if read is not None
        else read_checkpoint_inputs(
            ref.path, ref.artifact.sha256, ref.artifact.size_bytes
        )
    )
    checkpoint.admit(match, allow_deck_repetition=True)
    runtime = runtime_fingerprints(
        match_config=match, observation_space=checkpoint.space
    )
    return PlayerRegistration(
        player_id=name,
        display_name=player.name,
        role=role,
        content_suite=content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=runtime["observation_abi_sha256"],
        action_abi_sha256=runtime["action_abi_sha256"],
        matchup_sha256=runtime["matchup_sha256"],
        player_seed_derivation_id="arena-pair-deal-player-v1",
        compute_class_id=etude_compute_class(precision),
        runner_kind="checkpoint",
        player_spec=dict(ETUDE_SAMPLER_SPEC)
        | ({"blank_decision_memory": True} if blank_decision_memory(player) else {}),
        checkpoint_sha256=checkpoint.sha256,
        writer_revision=checkpoint.writer_revision,
        stored_programs=checkpoint.stored_programs,
        checkpoint_bytes=checkpoint.size_bytes,
        parameter_count=checkpoint.parameter_count,
        training_seed=None,
        artifact_id=f"etude/{checkpoint.sha256}",
    )


def etude_registration(
    player: PlayerRegistration, *, precision: EvalPrecision = "float32"
) -> PlayerRegistration:
    """An execution copy naming the Etude sampler, leaving retained bytes alone."""
    if player.runner_kind == "code":
        return player
    return PlayerRegistration.model_validate(
        {
            **player.model_dump(),
            "compute_class_id": etude_compute_class(precision),
            "player_spec": dict(ETUDE_SAMPLER_SPEC),
        }
    )
