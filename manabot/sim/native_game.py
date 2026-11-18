"""Native game stepping, Command binding and replay evidence.

execute_game runs in Etude's owning game process with supplied GamePlayers.
The standalone game_worker entry builds scalar/search players for local CPU
inspection. The game algorithm and disclosure checks are shared; there is no
inference service or model process here. Ordinary games journal native
commitments through their capture before execution; constructed scenarios
retain their acknowledged Command trace because a seed and decks cannot
recreate their root.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any

from manabot.backends import resolve
from manabot.env.failures import ScalarCapture
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig
from manabot.infra.provenance import (
    RECORDED_ASSET_MANIFEST_HASH,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.policy.protocol import Backend
from manabot.sim.agent import GamePlayer
from manabot.sim.evaluation_records import (
    bind_record,
    decision_annotation,
    evaluation_metadata,
    finalize_game,
)
from manabot.sim.frames import build_viewer_frame, decision_row
from manabot.sim.performance import EvalPerformance
from manabot.sim.players import CODE_PLAYER_KINDS, build_code_player
from manabot.sim.registration import PlayerRegistration, require_open_decklists
from manabot.sim.replay import replay_environment
from manabot.sim.supervision import failure_record, is_game_error
from managym import ManaTaps
from managym.bots import BeliefSummary
from managym.decision import Command, DecisionFrame
from managym.experience import UNVERSIONED_CONTENT_HASH
from managym.recording import read_record


class _InChildPlayers:
    """Builds in-process players and answers which game hooks they take.

    Code players (anchors, heuristic, demo search) come from the Torch-free
    `players` module. Scalar checkpoint and search players load Torch on their
    first build. Only search players that track the whole game receive
    `start_game`, and only local-update players observe each step; until a
    backend is built there are no such players, so neither hook applies to
    supplied agents.
    """

    def __init__(self, backend_factory: Callable[[], Backend] | None) -> None:
        self._backend_factory = backend_factory
        self._backend: Backend | None = None
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
        if self._backend is None:
            if self._backend_factory is None:
                raise ValueError(
                    "in-child checkpoint players require an injected backend"
                )
            from manabot.sim.search.local_update import LocalUpdatePlayer
            from manabot.sim.search.puct import DeterminizedPuctPlayer

            self._backend = self._backend_factory()
            self._game_trackers = (LocalUpdatePlayer, DeterminizedPuctPlayer)
            self._step_observers = (LocalUpdatePlayer,)
        from manabot.sim.guidance import build_registered_player

        return build_registered_player(
            registration,
            seed=seed,
            checkpoint_path=checkpoint_path,
            backend=self._backend,
        )

    def starts_game(self, player: Any) -> bool:
        return isinstance(player, self._game_trackers)

    def observes_steps(self, player: Any) -> bool:
        return isinstance(player, self._step_observers)

    def seed(self, seed: int) -> None:
        """Seed Torch's global generator before a checkpoint-guided search decision."""
        assert self._backend is not None
        self._backend.seed(seed)


def execute_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    send: Callable[[tuple[str, Any]], None],
    *,
    backend_factory: Callable[[], Backend] | None = None,
    agents: Mapping[str, GamePlayer] | None = None,
    deadline: float | None = None,
) -> None:
    """Play one game, sending phase, pending and decision events, then its result.

    Supplied agents own policy preparation/choice; standalone scalar players
    are built in-process. Only native AgentError/PanicException failures become
    invalid game rows; other errors propagate to the caller.
    """
    responsible = None
    env = None
    temporary = tempfile.TemporaryDirectory(prefix="evaluation-record-")
    capture = None
    perf = EvalPerformance(
        str(game.get("match_id", "game")),
        game.get("instrumentation", True),
    )
    try:
        built = {}
        tap_players: dict[str, ManaTaps] = {}
        in_child = _InChildPlayers(backend_factory)
        match = (
            MatchConfig.model_validate(game["match_hypers"])
            if "match_hypers" in game
            else None
        )
        if match is not None:
            require_open_decklists(match.decklists, registrations)
        for registration in registrations:
            responsible = registration.player_id
            send(("phase", responsible))
            if agents is not None:
                player = agents[responsible]
                assert match is not None
                player.admit(match)
                built[responsible] = (player, player.space)
            else:
                with perf.span("player_startup"):
                    built[responsible] = in_child.build(
                        registration,
                        seed=game["player_seeds"][responsible],
                        checkpoint_path=checkpoint_paths.get(responsible),
                    )
            player_space = built[responsible][1]
            if (
                registration.player_spec["kind"] in {"checkpoint", "etude_checkpoint"}
                and player_space is not None
            ):
                tap_players[responsible] = player_space.encoder.config.mana_taps
        responsible = None
        send(("phase", None))
        spaces = [space for _, space in built.values() if space is not None]
        if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
            raise ValueError("players have different observation ABIs")
        space = spaces[0] if spaces else ObservationSpace()
        if match is not None:
            first, second = game["seat_players"]
            match = match.model_copy(
                update={
                    "mana_taps": (
                        tap_players.get(first, "off"),
                        tap_players.get(second, "off"),
                    )
                }
            )
            resolved = match.model_dump(mode="json")
            if resolved != game["match_hypers"]:
                game["match_hypers"] = resolved
                send(("match", resolved))
        game["observation_hypers"] = space.encoder.config.model_dump()
        send(("observation", game["observation_hypers"]))
        game["runtime_identities"] = runtime_fingerprints(
            game["deal_seed"],
            match_config=match,
            observation_space=space,
        )
        game["runtime_identities"]["arena_source_sha256"] = source_bundle_sha256(
            sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        )
        send(("identities", game["runtime_identities"]))
        with perf.span("native_reset_and_encoding"):
            if "scenario_root" not in game:
                capture_path = Path(
                    game.get("capture_path", Path(temporary.name) / "journal.jsonl")
                )
                capture = ScalarCapture(
                    game["deal_seed"],
                    journal_path=capture_path,
                    context={"source_kind": "evaluation"},
                    context_provider=lambda: {"evaluation": evaluation_metadata(game)},
                    failure_directory=capture_path.parent,
                )
            env, obs = replay_environment(game, space, capture=capture)
        if capture is None:
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
            if deadline is not None and time.monotonic() >= deadline:
                failure_record(
                    game, "timeout", "game wall-clock budget exhausted", None
                )
                break
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
            if agents is None and registrations[actor].runner_kind == "checkpoint":
                in_child.seed(policy_seed)
            started = time.perf_counter()
            if agents is not None:
                action = agents[player_id].choose(env, obs, policy_seed)
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
            perf.record("command_validation", validation_started)
            responsible = None
            send(("phase", None))
            decision = {
                **decision_row(env, frame, action, retain_digest=capture is None),
                "revision": revision,
                "player_id": player_id,
                "player_seed": game["player_seeds"][player_id],
                "player_decision_ordinal": ordinal,
                "policy_rng_seed": policy_seed,
                "latency_seconds": elapsed,
            }
            player = built[player_id][0]
            if in_child.observes_steps(player) and player.last_receipt is not None:
                # Arena traces are private. Advice uses the explicit public projection.
                decision["local_update_receipt"] = player.last_receipt.to_json()
                decision["latency_seconds"] += player.last_receipt.belief_seconds
                decision["belief_model_mismatch"] = (
                    "opponent_behavior_not_assumed_equal_to_frozen_policy"
                )
            annotation = decision_annotation(decision)
            if in_child.observes_steps(player) and player.last_receipt is not None:
                receipt = player.last_receipt
                annotation.update(
                    probabilities=list(receipt.target),
                    search_scores=list(receipt.values),
                    belief=asdict(
                        BeliefSummary(
                            receipt.belief_model,
                            receipt.belief_digest,
                            receipt.condition_mass,
                            len(
                                {
                                    (r.world_index, r.world_seed)
                                    for r in receipt.rollouts
                                }
                            ),
                        )
                    ),
                )
            # Scenarios use the parent's acknowledged Command journal. Ordinary
            # games fsync their capture journal immediately before native execution.
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
                    semantic_command, annotation=annotation
                )
            else:
                obs, _, terminated, truncated, info = env.step(
                    action, annotation=annotation
                )
            perf.record("native_step_and_encoding", native_started)
            perf.count("native_step_calls")
            if capture is None:
                decision["post_state_digest"] = env._engine.state_digest()
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
                failure_record(
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
            failure_record(game, "command_cap", "game exceeded its Command cap", None)
    except BaseException as exc:
        if not is_game_error(exc):
            if env is not None:
                env.close()
            raise
        game["failure_trace"] = traceback.format_exc()
        game["error_class"] = type(exc).__name__
        failure_record(
            game,
            "crash",
            f"{type(exc).__name__}: {exc}",
            responsible,
        )
    if env is not None:
        if env.obs_space.encoder.config.action_overflow == "first-legal":
            game["policy_action_cutoff"] = env._capture.policy_action_report()
        env.close(reason=game["termination_reason"])
        if capture is not None:
            # The journal includes a rejected/pending call; viewing never executes it.
            record = (
                read_record(capture_path)
                if game.get("pending_decision")
                else capture.snapshot()
            )
            bind_record(game, record)
    temporary.cleanup()
    game["worker_performance"] = perf.snapshot()
    finalize_game(game)
    send(("result", game))


def game_worker(
    connection: Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    deadline_unix: float,
) -> None:
    """Standalone scalar child; Etude games call execute_game in their own process."""

    def backend_factory() -> Backend:
        import torch

        torch.set_num_threads(1)
        return resolve("torch")

    def send(event: tuple[str, Any]) -> None:
        connection.send(event)
        if (
            event[0] == "pending"
            and "scenario_root" in game
            and connection.recv() != "retained"
        ):
            raise RuntimeError("pre-command evidence was not retained")

    try:
        execute_game(
            game,
            registrations,
            checkpoint_paths,
            max_commands,
            send,
            backend_factory=backend_factory,
            deadline=time.monotonic() + max(0, deadline_unix - time.time()),
        )
    finally:
        connection.close()
