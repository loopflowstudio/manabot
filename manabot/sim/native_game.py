"""The native game child: one bounded game, its Commands and pre-command evidence.

`supervise_game` launches `game_worker` as a `ChildProcess`. The child builds
viewer-safe frames, sends each pending Command for durable acknowledgement
before native execution, and reports the settled game. Checkpoint players
are served: they reach their policy over the game's own connection to the
allocation's inference service, not through the supervisor.
Code players (anchors, heuristic) are built Torch-free, so a game between them
and served players never imports Torch. Search players (flat-MC, PUCT, local
update) keep their policy in-child; `_InChildPlayers` is the one place this
module imports Torch-bearing code.

Before reporting, the child counts its threads against the declared game
budget; exceeding it is an infrastructure failure, not a forfeit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
import time
import traceback
from typing import Any
from uuid import uuid4

from managym.decision import Command, DecisionFrame
from managym.experience import UNVERSIONED_CONTENT_HASH

from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import (
    RECORDED_ASSET_MANIFEST_HASH,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.sim.execution import EvalExecution
from manabot.sim.frames import build_command, build_viewer_frame
from manabot.sim.performance import EvalPerformance
from manabot.sim.players import CODE_PLAYER_KINDS, build_code_player
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import replay_environment
from manabot.sim.serving_protocol import (
    DecisionReply,
    DecisionRequest,
    InferenceClient,
    InferenceFailure,
    PolicyKey,
)
from manabot.sim.supervision import _failure
from manabot.sim.threads import GAME_BUDGET, ThreadBudgetExceeded


class _InChildPlayers:
    """Builds in-process players and answers which game hooks they take.

    Code players (anchors, heuristic, demo search) come from the Torch-free
    `players` module. Search players load Torch on their first build. Only
    search players that track the whole game receive `start_game`, and only
    local-update players observe each step, exactly as registered policies
    always have.
    """

    def __init__(self) -> None:
        self._torch: Any = None
        self._game_trackers: tuple[type, ...] = ()
        self._step_observers: tuple[type, ...] = ()

    def build(
        self,
        registration: PlayerRegistration,
        *,
        seed: int,
        checkpoint_path: str | None,
    ) -> tuple[Any, ObservationSpace | None]:
        if registration.player_spec["kind"] in CODE_PLAYER_KINDS:
            return build_code_player(registration, seed=seed), None
        if registration.player_spec["kind"] == "checkpoint":
            raise InferenceFailure("evaluation checkpoint players play served")
        if self._torch is None:
            import torch

            from manabot.sim.search.local_update import LocalUpdatePlayer
            from manabot.sim.search.puct import DeterminizedPuctPlayer

            torch.set_num_threads(1)
            self._torch = torch
            self._game_trackers = (LocalUpdatePlayer, DeterminizedPuctPlayer)
            self._step_observers = (LocalUpdatePlayer,)
        from manabot.sim.guidance import build_registered_player

        return build_registered_player(
            registration, seed=seed, checkpoint_path=checkpoint_path
        )

    def starts_game(self, player: Any) -> bool:
        return isinstance(player, self._game_trackers)

    def observes_steps(self, player: Any) -> bool:
        return isinstance(player, self._step_observers)

    def seed(self, seed: int) -> None:
        """Seed Torch's global generator before a checkpoint-guided search decision."""
        self._torch.manual_seed(seed)


def _census(game: dict[str, Any]) -> None:
    """Count this child's threads into its result; over budget fails the game."""
    try:
        GAME_BUDGET.account(game["worker_performance"]["counters"])
    except ThreadBudgetExceeded as exc:
        _failure(game, "crash", str(exc), None)
        game["infrastructure_failure"] = True


def execute_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    send: Callable[[tuple[str, Any]], None],
    infer: Callable[[DecisionRequest], DecisionReply] | None = None,
) -> None:
    """Play one game, sending phase, pending and decision events, then its result.

    Served players reach a policy only through `infer`; code and search players
    are built in-process (`_InChildPlayers`); only search players import Torch.
    """
    responsible = None
    perf = EvalPerformance(
        str(game.get("match_id", "game")),
        game.get("instrumentation", True),
        latency_boundary="game worker submission through reply identity validation; excludes Command validation and native execution",
    )
    try:
        built = {}
        execution = (
            EvalExecution.model_validate(game["execution"])
            if "execution" in game
            else None
        )
        policies: dict[str, PolicyKey] = {}
        in_child = _InChildPlayers()
        for registration in registrations:
            responsible = registration.player_id
            send(("phase", responsible))
            if "execution" in registration.player_spec:
                if execution is None or infer is None:
                    raise InferenceFailure(
                        "served checkpoint requires an inference endpoint"
                    )
                policies[responsible] = PolicyKey.from_registration(
                    registration, execution
                )
                config = ObservationConfig.from_checkpoint(
                    game["served_observations"][responsible]
                )
                built[responsible] = (None, ObservationSpace(config))
            else:
                with perf.span("player_startup"):
                    built[responsible] = in_child.build(
                        registration,
                        seed=game["player_seeds"][responsible],
                        checkpoint_path=checkpoint_paths.get(responsible),
                    )
        responsible = None
        send(("phase", None))
        spaces = [space for _, space in built.values() if space is not None]
        if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
            raise ValueError("players have different observation ABIs")
        space = spaces[0] if spaces else ObservationSpace()
        game["observation_hypers"] = space.encoder.config.model_dump()
        send(("observation", game["observation_hypers"]))
        game["runtime_identities"] = runtime_fingerprints(
            game["deal_seed"],
            match_config=MatchConfig.model_validate(game["match_hypers"])
            if "match_hypers" in game
            else None,
            observation_space=space,
        )
        game["runtime_identities"]["arena_source_sha256"] = source_bundle_sha256(
            sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        )
        send(("identities", game["runtime_identities"]))
        with perf.span("native_reset_and_encoding"):
            env, obs = replay_environment(game, space)
        game["initial_state_digest"] = env._engine.state_digest()
        send(("initial", game["initial_state_digest"]))
        for seat, player_id in enumerate(game["seat_players"]):
            player = built[player_id][0]
            if in_child.starts_game(player):
                player.start_game(env, seat)
        local_players = [
            player for player, _ in built.values() if in_child.observes_steps(player)
        ]
        decision_counts = dict.fromkeys(built, 0)
        for revision in range(max_commands):
            raw = env.last_raw_obs
            actor = int(raw.agent.player_index)
            player_id = game["seat_players"][actor]
            frame_started = time.monotonic()
            frame = build_viewer_frame(
                raw,
                match_id=game["match_id"],
                revision=revision,
                content_hash=UNVERSIONED_CONTENT_HASH,
                asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
            )
            perf.record("frame_projection", frame_started)
            if frame["projection"]["opponent"].get("hand"):
                game["integrity"]["private_exposures"] += 1
                raise RuntimeError("viewer frame exposes a private hand")
            pre_digest = env._engine.state_digest()
            responsible = player_id
            send(("phase", player_id))
            ordinal = decision_counts[player_id]
            policy_seed = (game["player_seeds"][player_id] + ordinal) % (2**63)
            if registrations[actor].runner_kind == "checkpoint" and (
                player_id not in policies
            ):
                in_child.seed(policy_seed)
            started = time.perf_counter()
            if player_id in policies:
                assert execution is not None and infer is not None
                request = DecisionRequest(
                    uuid4().hex,
                    game["match_id"],
                    actor,
                    revision,
                    policies[player_id],
                    ordinal,
                    policy_seed,
                    {k: v.copy() for k, v in obs.items()},
                    min(
                        execution.deadline_unix, time.time() + execution.request_seconds
                    ),
                    time.monotonic(),
                )
                with perf.span("inference_round_trip", request=request.request_id):
                    try:
                        reply = infer(request)
                        reply.validate(request)
                        action = reply.action
                    except BaseException:
                        perf.latency(
                            time.monotonic() - request.submitted, outcome="failed"
                        )
                        raise
                    else:
                        perf.latency(time.monotonic() - request.submitted)
                perf.count(
                    "ipc_input_bytes",
                    sum(v.nbytes for v in request.observation.values()),
                )
            else:
                with perf.span("scalar_policy"):
                    action = int(built[player_id][0].act(env, obs))
            elapsed = time.perf_counter() - started
            validation_started = time.monotonic()
            if env._engine.state_digest() != pre_digest:
                game["integrity"]["root_mutations"] += 1
                raise RuntimeError("player mutated the authoritative root")
            offers = {int(offer["id"]): offer for offer in frame["offers"]}
            if action not in offers:
                game["integrity"]["illegal_actions"] += 1
                raise RuntimeError(f"illegal offer {action}")
            command = build_command(frame, action)
            perf.record("command_validation", validation_started)
            responsible = None
            send(("phase", None))
            decision = {
                "revision": revision,
                "actor": actor,
                "player_id": player_id,
                "player_seed": game["player_seeds"][player_id],
                "player_decision_ordinal": ordinal,
                "policy_rng_seed": policy_seed,
                "action_space_kind": frame["action_space"],
                "frame_sha256": canonical_sha256(frame),
                "command": command,
                "command_sha256": canonical_sha256(command),
                "chosen_offer": offers[action],
                "pre_state_digest": pre_digest,
                "latency_seconds": elapsed,
            }
            if player_id in policies:
                decision["inference_request_id"] = request.request_id
                decision["inference_batch_id"] = reply.batch_id
            # The parent acknowledges durable pre-command evidence before native
            # execution. Successful Commands alone cannot reproduce a rejection.
            game["pending_decision"] = decision
            with perf.span("command_durable_acknowledgement"):
                send(("pending", decision))
            native_started = time.monotonic()
            if local_players:
                for player in local_players:
                    player.prepare_step(env, actor, action)
                semantic_frame = DecisionFrame.from_json(
                    env._engine.semantic_decision_frame_json()
                )
                semantic_command = Command(
                    f"arena-{revision}",
                    semantic_frame.revision,
                    int(semantic_frame.offers[action]["id"]),
                )
                obs, _, terminated, truncated, info, transition = env.step_semantic(
                    semantic_command
                )
            else:
                obs, _, terminated, truncated, info = env.step(action)
            perf.record("native_step_and_encoding", native_started)
            perf.count("native_step_calls")
            decision["post_state_digest"] = env._engine.state_digest()
            player = built[player_id][0]
            if in_child.observes_steps(player) and player.last_receipt is not None:
                # Arena traces are private. Advice uses the explicit public projection.
                decision["local_update_receipt"] = player.last_receipt.to_json()
                decision["latency_seconds"] += player.last_receipt.belief_seconds
                decision["belief_model_mismatch"] = (
                    "opponent_behavior_not_assumed_equal_to_frozen_policy"
                )
            game.pop("pending_decision", None)
            game["decisions"].append(decision)
            decision_counts[player_id] += 1
            send(("decision", decision))
            for player in local_players:
                player.observe_step(env, actor, transition)
            if truncated or any(
                info.get(name)
                for name in (
                    "action_space_truncated",
                    "card_space_truncated",
                    "permanent_space_truncated",
                )
            ):
                game["integrity"]["truncations"] += 1
                _failure(
                    game, "truncated", "environment or observation truncated", None
                )
                break
            if terminated:
                winner = env._engine.winner_index()
                game.update(
                    winner=winner,
                    terminated=True,
                    termination_reason="draw" if winner is None else "terminal",
                )
                break
        else:
            _failure(game, "command_cap", "game exceeded its Command cap", None)
    except BaseException as exc:
        if not isinstance(exc, Exception) and type(exc).__name__ != "PanicException":
            raise
        game["failure_trace"] = traceback.format_exc()
        game["error_class"] = type(exc).__name__
        _failure(
            game,
            "crash",
            f"{type(exc).__name__}: {exc}",
            None if isinstance(exc, InferenceFailure) else responsible,
        )
    game["worker_performance"] = perf.snapshot()
    send(("result", game))


def game_worker(
    connection: Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    inference: int | None,
    deadline_unix: float,
) -> None:
    """Game child entry; `inference` is this game's service connection descriptor.

    Inference waiting consumes the game's wall allowance, so each request's
    deadline is clamped to `deadline_unix`. An inference failure is reported to
    the supervisor before the game settles, so it fails the allocation.
    """
    client = None if inference is None else InferenceClient(inference)

    def send(event: tuple[str, Any]) -> None:
        if event[0] == "result":
            _census(event[1])
        connection.send(event)
        if event[0] == "pending" and connection.recv() != "retained":
            raise RuntimeError("pre-command evidence was not retained")

    def infer(request: DecisionRequest) -> DecisionReply:
        if client is None:
            raise InferenceFailure("no inference connection attached")
        try:
            return client.infer(
                replace(
                    request,
                    deadline_unix=min(request.deadline_unix, deadline_unix),
                )
            )
        except InferenceFailure as exc:
            connection.send(("inference_failure", str(exc)))
            raise

    try:
        execute_game(game, registrations, checkpoint_paths, max_commands, send, infer)
    finally:
        if client is not None:
            client.close()
        connection.close()
