"""Compact Command traces and policy-free simulation replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from managym.experience import UNVERSIONED_CONTENT_HASH, Command

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.env.replay import (
    ReplayAgreement,
    matches_retained_error,
    require_replay_runtime,
    run_bounded_replay,
)
from manabot.infra.config import MatchConfig, ObservationConfig, RewardConfig
from manabot.infra.identity import canonical_json, canonical_sha256, file_sha256
from manabot.infra.provenance import RECORDED_ASSET_MANIFEST_HASH
from manabot.model.world import checkpoint_world
from manabot.sim.search.teacher1_evidence import build_command, build_viewer_frame
from manabot.verify.competency import SCENARIOS, build_scenario_env
from manabot.verify.scenario_validation import scenario_identity
from manabot.verify.util import INTERACTIVE_DECK
import managym


@dataclass(frozen=True)
class ReplayReceipt:
    games: int = 0
    decisions: int = 0
    frame_mismatches: int = 0
    offer_mismatches: int = 0
    command_mismatches: int = 0
    state_mismatches: int = 0
    outcome_mismatches: int = 0
    actor_mismatches: int = 0
    missing_decisions: int = 0
    trace_mismatches: int = 0
    private_exposures: int = 0

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.frame_mismatches,
                self.offer_mismatches,
                self.command_mismatches,
                self.state_mismatches,
                self.outcome_mismatches,
                self.actor_mismatches,
                self.missing_decisions,
                self.trace_mismatches,
                self.private_exposures,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def write_trace(path: Path, games: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(game) + b"\n" for game in games)
    # Replace atomically: a later failed game must not tear earlier evidence.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(gzip.compress(payload, compresslevel=9, mtime=0))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": file_sha256(path), "games": len(games)}


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
    ]


def replay_environment(
    game: dict[str, Any],
    observation_space: ObservationSpace | None = None,
    *,
    capture_cards: bool = False,
) -> tuple[Env, dict[str, Any]]:
    if "scenario_root" in game:
        if capture_cards:
            raise ValueError("card capture requires ordinary deck setup")
        root = game["scenario_root"]
        name = root["name"]
        if root != scenario_identity(name):
            raise ValueError("scenario root version/world/source mismatch")
        space = observation_space or ObservationSpace(
            ObservationConfig(**game["observation_hypers"])
        )
        env, observation, _ = build_scenario_env(
            SCENARIOS[name], space, int(game["deal_seed"])
        )
        if checkpoint_world(env.match.to_rust(), space) != game["world_binding"]:
            env.close()
            raise ValueError("scenario world/setup/input mismatch")
        return env, observation
    names = game["seat_players"]
    match = Match(
        MatchConfig.model_validate(game["match_hypers"])
        if "match_hypers" in game
        else MatchConfig(
            hero=str(names[0]),
            villain=str(names[1]),
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        observation_space
        or ObservationSpace(ObservationConfig(**game.get("observation_hypers", {}))),
        Reward(RewardConfig()),
        seed=int(game["deal_seed"]),
        auto_reset=False,
    )
    if capture_cards:
        env._engine.enable_card_capture()
    observation, _ = env.reset(seed=int(game["deal_seed"]))
    return env, observation


def replay_games(
    games: list[dict[str, Any]], *, require_terminal: bool = True
) -> ReplayReceipt:
    """Verify saved Commands; diagnostics may explicitly request prefix replay.

    A passing prefix receipt proves only the retained trajectory, never game
    completion. Arena callers retain the complete-game requirement by default.
    """
    counts = asdict(ReplayReceipt())
    for game in games:
        receipt, _ = replay_game(game, require_terminal=require_terminal)
        for name, count in asdict(receipt).items():
            counts[name] += count
    return ReplayReceipt(**counts)


def replay_game(
    game: dict[str, Any],
    *,
    require_terminal: bool = True,
    capture_cards: bool = False,
) -> tuple[ReplayReceipt, dict[str, Any] | None]:
    """Verify one game, optionally retaining native facts from the same replay."""
    env, _ = replay_environment(game, capture_cards=capture_cards)
    try:
        receipt = _replay_game(game, env, require_terminal=require_terminal)
        captured = (
            json.loads(env._engine.card_capture_json()) if capture_cards else None
        )
        return receipt, captured
    finally:
        env.close()


def _replay_game(
    game: dict[str, Any], env: Env, *, require_terminal: bool
) -> ReplayReceipt:
    """Validate once in the caller's environment, leaving it at the prefix end.

    Failure replay can then attempt the pending Command on this same checked
    state. The caller owns closing the environment, including on native errors.
    """
    counts = asdict(ReplayReceipt(games=1))
    unsigned_game = dict(game)
    stored_trace_sha256 = unsigned_game.pop("game_trace_sha256", None)
    if stored_trace_sha256 != canonical_sha256(unsigned_game):
        counts["trace_mismatches"] += 1
    initial_digest = env._engine.state_digest()
    if game.get("initial_state_digest", initial_digest) != initial_digest:
        counts["state_mismatches"] += 1
        return ReplayReceipt(**counts)
    done = False
    for revision, expected in enumerate(game["decisions"]):
        if done:
            counts["command_mismatches"] += 1
            break
        frame = build_viewer_frame(
            env.last_raw_obs,
            match_id=str(game["match_id"]),
            revision=revision,
            content_hash=UNVERSIONED_CONTENT_HASH,
            asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
        )
        if canonical_sha256(frame) != expected["frame_sha256"]:
            counts["frame_mismatches"] += 1
        if frame["projection"]["opponent"].get("hand"):
            counts["private_exposures"] += 1
        actor = int(env.last_raw_obs.agent.player_index)
        if (
            actor != int(expected["actor"])
            or game["seat_players"][actor] != expected["player_id"]
        ):
            counts["actor_mismatches"] += 1
        if env._engine.state_digest() != expected["pre_state_digest"]:
            counts["state_mismatches"] += 1
        command = Command.model_validate(expected["command"])
        if canonical_sha256(expected["command"]) != expected.get("command_sha256"):
            counts["command_mismatches"] += 1
        offer_ids = {int(offer["id"]) for offer in frame["offers"]}
        if (
            command.match_id != frame["match_id"]
            or command.expected_revision != revision
            or command.prompt_id != frame["prompt"]["id"]
            or command.offer_id not in offer_ids
        ):
            counts["command_mismatches"] += 1
            break
        rebuilt_command = build_command(frame, int(command.offer_id))
        if rebuilt_command != expected["command"]:
            counts["command_mismatches"] += 1
        selected_offer = next(
            (offer for offer in frame["offers"] if offer["id"] == command.offer_id),
            None,
        )
        if (
            selected_offer is None
            or selected_offer != expected.get("chosen_offer")
            or frame["action_space"] != expected.get("action_space_kind")
        ):
            counts["offer_mismatches"] += 1
        _, _, terminated, truncated, _ = env.step(int(command.offer_id))
        done = bool(terminated or truncated)
        if env._engine.state_digest() != expected["post_state_digest"]:
            counts["state_mismatches"] += 1
        counts["decisions"] += 1
    failed = game.get("failure") is not None
    if require_terminal and not done and not failed:
        counts["missing_decisions"] += 1
    winner = env._engine.winner_index() if done else None
    if not failed and (
        winner != game["winner"]
        or bool(game.get("terminated")) != bool(done and not game.get("truncated"))
    ):
        counts["outcome_mismatches"] += 1
    return ReplayReceipt(**counts)


def apply_game_event(
    game: dict[str, Any] | None, kind: str, value: Any
) -> dict[str, Any]:
    """Project worker events identically in the live parent and journal reader.

    Event payloads are private wire data. Pending decisions become completed
    only on a decision event; phase events carry no game-state mutation.
    """
    if kind in {"game", "result", "settled"}:
        return value
    if game is None:
        raise ValueError("journal event precedes game setup")
    if kind == "decision":
        game.pop("pending_decision", None)
        game["decisions"].append(value)
    elif kind == "initial":
        game["initial_state_digest"] = value
    elif kind == "identities":
        game["runtime_identities"] = value
    elif kind == "observation":
        game["observation_hypers"] = value
    elif kind == "pending":
        game["pending_decision"] = value
    return game


def read_game_journal(path: Path) -> dict[str, Any]:
    """Recover acknowledged inputs even when the parent never published a trace.

    Torn JSON fails closed. An unobserved outcome is explicitly unknown, never a
    successful evaluation or a claim that the pending Command actually failed.
    """
    game: dict[str, Any] | None = None
    with path.open() as stream:
        for line in stream:
            kind, value = json.loads(line)
            game = apply_game_event(game, kind, value)
    if game is None:
        raise ValueError("journal has no game setup")
    if game.get("pending_decision") is not None and game.get("failure") is None:
        game.update(failure="execution_outcome_unknown", winner=None, terminated=False)
    game["game_trace_sha256"] = canonical_sha256(game)
    return game


def replay_failure(game: dict[str, Any]) -> dict[str, Any]:
    """Replay a rejected offered Command without a policy or outcome credit.

    Run through the CLI for a process wall-time bound. The original trace remains
    immutable; passing the retained action after repair is diagnostic advancement.
    """
    pending = game.get("pending_decision")
    if pending is None or game.get("failure") is None:
        raise ValueError("game has no retained failed Command")
    agreement = ReplayAgreement(game["failure"])
    # Arena's historical error field combines class and message. Worker loss,
    # timeout and unknown journal outcomes supply no comparable native error.
    error_class, separator, message = game["failure"].partition(": ")
    retained_class = error_class if separator else None
    retained_message = message if separator else None
    env: Env | None = None
    try:
        env, _ = replay_environment(game)
        if not _replay_game(game, env, require_terminal=False).passed:
            agreement.matches_binding = False
            return {**asdict(agreement), "status": "prefix_mismatch"}
        frame = build_viewer_frame(
            env.last_raw_obs,
            match_id=game["match_id"],
            revision=len(game["decisions"]),
            content_hash=UNVERSIONED_CONTENT_HASH,
            asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
        )
        action = int(pending["command"]["offer_id"])
        offer = next((o for o in frame["offers"] if int(o["id"]) == action), None)
        if offer is None:
            agreement.matches_binding = False
            return {**asdict(agreement), "status": "pending_mismatch"}
        bindings = {
            "frame_sha256": canonical_sha256(frame),
            "pre_state_digest": env._engine.state_digest(),
            "chosen_offer": offer,
            "command": build_command(frame, action),
            "command_sha256": canonical_sha256(pending["command"]),
        }
        if any(
            pending[key] != value
            for key, value in bindings.items()
            if pending.get(key) is not None
        ):
            agreement.matches_binding = False
            return {**asdict(agreement), "status": "pending_mismatch"}
        if all(pending.get(key) is not None for key in bindings):
            agreement.matches_binding = True
        agreement.at_retained_failure = True
        _, _, terminated, truncated, _ = env.step(action)
        return {
            **asdict(agreement),
            "status": "completed"
            if terminated
            else "truncated"
            if truncated
            else "advanced",
        }
    except BaseException as error:
        if (
            not isinstance(error, managym.AgentError)
            and type(error).__name__ != "PanicException"
        ):
            raise
        agreement.matches_error = matches_retained_error(
            error, retained_class, retained_message
        )
        # The boundary flag records native execution, so a prefix rejection
        # cannot acquire pending-command status through shared error handling.
        result = {**asdict(agreement), "status": "prefix_failed", "error": str(error)}
        if agreement.at_retained_failure:
            result.update(
                status="failed",
                action_index=len(game["decisions"]),
                error_class=type(error).__name__,
            )
        return result
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded policy-free arena failure replay"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--game", type=int, required=True, help="One-based game index")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--allow-changed-runtime", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.game < 1 or not 0 < args.seconds <= 300:
        parser.error("game must be positive and seconds in (0, 300]")
    if not args.worker:
        run_bounded_replay("manabot.sim.replay", args.seconds)
    games = (
        [read_game_journal(args.trace)]
        if args.trace.suffix == ".jsonl"
        else read_trace(args.trace)
    )
    if args.game > len(games):
        parser.error("game is beyond the end of the trace")
    game = games[args.game - 1]
    try:
        require_replay_runtime(
            game["runtime_identities"], allow_changed_runtime=args.allow_changed_runtime
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(replay_failure(game)))


if __name__ == "__main__":
    main()
