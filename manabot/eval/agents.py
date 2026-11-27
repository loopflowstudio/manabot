"""Game-side Etude players: admitted host preparation and seeded action choice.

Model forwards belong to Etude. Builders read only local checkpoint metadata;
no Torch, device probe, remote fetch or sampling happens in the loader protocol.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from typing import Any, Literal, cast

from etude.eval.agents import Forward, ModelForward
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
from manabot.policy.admission import Spec, check, legal_mask
from manabot.policy.checkpoint import read_container
from manabot.policy.prepare import Layout, Tables, prepare
from manabot.sim.agent import GamePlayer
from manabot.sim.heuristic import HeuristicPlayer
from manabot.sim.players import CodePlayer, build_code_player
from manabot.sim.registration import (
    ETUDE_SAMPLER_SPEC,
    EvalPrecision,
    PlayerRegistration,
    etude_compute_class,
)


@dataclass(frozen=True)
class Sampler:
    forward: Forward | ModelForward
    spec: Spec
    world_binding: Mapping[str, Any]
    deterministic: bool = False
    allow_deck_repetition: bool = False
    # Memory ablation: the policy sees the live board with an empty decision log.
    blank_decision_memory: bool = False

    @property
    def space(self) -> ObservationSpace:
        return self.spec.space

    def admit(self, match: MatchConfig) -> None:
        validate_setup_binding(
            self.world_binding,
            Match(match).to_rust(),
            allow_deck_repetition=self.allow_deck_repetition,
        )

    def choose(self, env: Env, obs: ObservationBuffers, seed: int) -> int:
        batch = {key: value[None] for key, value in obs.items()}
        if self.blank_decision_memory:
            batch = blank_memory(batch)
        legal = legal_mask(batch["actions_valid"])
        check(batch, self.spec, legal)
        model = self.spec.model
        row = replace(
            prepare(
                batch,
                self.spec.tables,
                binding=self.spec.binding,
                memory_summaries=model.decision_memory_summaries
                if model.decision_memory != "none"
                else 0,
                memory_fold=model.decision_memory == "gru",
            ),
            programs=None,
        )
        [output] = cast(ModelForward, self.forward)([row])
        logits = np.asarray(output, dtype=np.float32)
        if logits.shape != legal.shape or not np.isfinite(logits[legal]).all():
            raise ValueError("forward returned malformed legal logits")
        row_logits = logits[0]
        slots = np.flatnonzero(legal[0])
        if self.deterministic:
            return int(slots[row_logits[slots].argmax()])
        weights = np.exp(row_logits[slots].astype(np.float64) - row_logits[slots].max())
        return int(np.random.default_rng(seed).choice(slots, p=weights / weights.sum()))


@dataclass(frozen=True)
class Heuristic:
    factory: Callable[[int], CodePlayer]
    space: None = None

    def admit(self, match: MatchConfig) -> None:
        pass

    def choose(self, env: Env, obs: ObservationBuffers, seed: int) -> int:
        return self.factory(seed).act(env, obs)


def blank_decision_memory(player: CheckpointPlayer) -> bool:
    """Whether the seat declares the memory ablation; only a literal true enables it."""
    blank = player.config.get("blank_decision_memory", False)
    if not isinstance(blank, bool):
        raise ValueError("blank_decision_memory must be a boolean")
    return blank


def build(player: object, leaves: Mapping[str, Forward | ModelForward]) -> GamePlayer:
    if isinstance(player, CheckpointPlayer):
        checkpoint = read_container(player.checkpoint.path)
        # Keep only copied integer metadata in the game process: NumPy views
        # from read_container otherwise retain the entire weights file.
        tables = Tables.from_buffers(
            {key: value.copy() for key, value in checkpoint.buffers.items()},
            Layout(checkpoint.readings),
            fact_kinds=checkpoint.fact_kinds,
        )
        spec = Spec(
            checkpoint.model,
            checkpoint.space,
            tables,
            checkpoint.parameters["card_definitions.weight"].shape[0]
            if "card_definitions.weight" in checkpoint.parameters
            else 0,
        )
        registration = player.config.get("registration", {})
        player_spec = (
            registration.get("player_spec", {})
            if isinstance(registration, dict)
            else {}
        )
        return Sampler(
            leaves[player.checkpoint.key],
            spec,
            json.loads(checkpoint.metadata["world_binding"]),
            deterministic=bool(player_spec.get("deterministic", False)),
            allow_deck_repetition=bool(player_spec.get("allow_deck_repetition", False)),
            blank_decision_memory=blank_decision_memory(player),
        )
    if isinstance(player, Code):
        if "registration" in player.config:
            registration = PlayerRegistration.model_validate(
                player.config["registration"]
            )
            return Heuristic(lambda seed: build_code_player(registration, seed=seed))
        if player.name != "heuristic":
            raise TypeError(f"Code player {player.name} has no Manabot builder")
        maximum = player.config.get("max_actions")
        HeuristicPlayer(max_actions=maximum)  # refuse a bad cap here, not mid-game
        return Heuristic(lambda seed: HeuristicPlayer(max_actions=maximum))
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


def game_players(
    players: Sequence[Player], agents: Sequence[object]
) -> list[GamePlayer]:
    """A match's Etude agents as game players, in seating order.

    Etude gives a checkpoint player without config its bare forward instead of
    calling `build`, so build that seat's sampler here; every other seat
    already arrives built.
    """
    if len(players) != len(agents):
        raise ValueError("one agent per seated player")
    return [
        build(player, {player.checkpoint.key: cast(Forward | ModelForward, agent)})
        if isinstance(player, CheckpointPlayer) and not player.config
        else cast(GamePlayer, agent)
        for player, agent in zip(players, agents, strict=True)
    ]


async def play_scalar[MatchT: EtudeMatch, Outcome](
    play: Callable[[MatchT, Sequence[object]], Outcome],
    match: MatchT,
    agents: Sequence[object],
) -> Outcome:
    """Run the scalar arena in a thread, sending forwards through Etude's event loop."""
    loop = asyncio.get_running_loop()

    def blocking(forward: Forward) -> ModelForward:
        def call(inputs: list[object]) -> list[object]:
            async def send() -> list[object]:
                return await forward(inputs, rows=len(inputs))

            return asyncio.run_coroutine_threadsafe(send(), loop).result()

        return call

    seated = []
    for player, agent in zip(match.players, agents, strict=True):
        if isinstance(agent, Sampler):
            agent = replace(agent, forward=blocking(cast(Forward, agent.forward)))
        elif isinstance(player, CheckpointPlayer) and not player.config:
            agent = blocking(cast(Forward, agent))
        seated.append(agent)
    return await asyncio.to_thread(play, match, seated)


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
) -> PlayerRegistration:
    """Admit local checkpoint bytes and its world without importing a framework.

    A seat declaring `blank_decision_memory` registers the same bytes as a
    distinct memory-ablation player.
    """
    checkpoint = read_container(player.checkpoint.path)
    validate_setup_binding(
        json.loads(checkpoint.metadata["world_binding"]),
        Match(match).to_rust(),
        allow_deck_repetition=True,
    )
    path = player.checkpoint.path
    digest = file_sha256(path)
    artifact = player.checkpoint.artifact
    if artifact.sha256 != digest or artifact.size_bytes != path.stat().st_size:
        raise ValueError("checkpoint differs from its recorded bytes")
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
        checkpoint_sha256=digest,
        writer_revision=checkpoint.writer_revision,
        stored_programs=checkpoint.stored_programs,
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(int(p.size) for p in checkpoint.parameters.values()),
        training_seed=None,
        artifact_id=f"etude/{digest}",
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
