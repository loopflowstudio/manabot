"""Compact Command trace serialization and policy-free arena replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from typing import Any

from etude.experience_protocol import Command
from etude.server import ASSET_MANIFEST_HASH, CONTENT_HASH
from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.teacher1_evidence import build_command, build_viewer_frame
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

from .models import canonical_json, canonical_sha256, file_sha256


@dataclass(frozen=True)
class ArenaReplayReceipt:
    games: int
    decisions: int
    frame_mismatches: int
    offer_mismatches: int
    command_mismatches: int
    state_mismatches: int
    outcome_mismatches: int
    actor_mismatches: int
    missing_decisions: int
    trace_mismatches: int
    private_exposures: int

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
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return {"path": str(path), "sha256": file_sha256(path), "games": len(games)}


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
    ]


def replay_environment(
    game: dict[str, Any], observation_space: ObservationSpace | None = None
) -> tuple[Env, dict[str, Any]]:
    names = game["seat_players"]
    match = Match(
        MatchHypers(
            hero=str(names[0]),
            villain=str(names[1]),
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        observation_space or ObservationSpace(),
        Reward(RewardHypers()),
        seed=int(game["deal_seed"]),
        auto_reset=False,
    )
    observation, _ = env.reset(seed=int(game["deal_seed"]))
    return env, observation


def replay_prefix(
    game: dict[str, Any],
    decisions: int,
    observation_space: ObservationSpace | None = None,
) -> tuple[Env, dict[str, Any]]:
    """Rebuild one retained decision root without invoking either player."""

    env, observation = replay_environment(game, observation_space)
    if decisions < 0 or decisions > len(game["decisions"]):
        raise ValueError("replay prefix is outside the retained trace")
    for expected in game["decisions"][:decisions]:
        command = Command.model_validate(expected["command"])
        observation, _, terminated, truncated, _ = env.step(int(command.offer_id))
        if terminated or truncated:
            raise RuntimeError("retained profile root occurs after game termination")
    return env, observation


def replay_games(games: list[dict[str, Any]]) -> ArenaReplayReceipt:
    counts = {
        "decisions": 0,
        "frame_mismatches": 0,
        "offer_mismatches": 0,
        "command_mismatches": 0,
        "state_mismatches": 0,
        "outcome_mismatches": 0,
        "actor_mismatches": 0,
        "missing_decisions": 0,
        "trace_mismatches": 0,
        "private_exposures": 0,
    }
    for game in games:
        unsigned_game = dict(game)
        stored_trace_sha256 = unsigned_game.pop("game_trace_sha256", None)
        if stored_trace_sha256 != canonical_sha256(unsigned_game):
            counts["trace_mismatches"] += 1
        env, _ = replay_environment(game)
        info: dict[str, Any] = {}
        done = False
        for revision, expected in enumerate(game["decisions"]):
            if done:
                counts["command_mismatches"] += 1
                break
            frame = build_viewer_frame(
                env.last_raw_obs,
                match_id=str(game["match_id"]),
                revision=revision,
                content_hash=CONTENT_HASH,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
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
            _, _, terminated, truncated, info = env.step(int(command.offer_id))
            done = bool(terminated or truncated)
            if env._engine.state_digest() != expected["post_state_digest"]:
                counts["state_mismatches"] += 1
            counts["decisions"] += 1
        if not done:
            counts["missing_decisions"] += 1
        winner = winner_from_info_or_obs(info, env.last_raw_obs) if done else None
        if winner != game["winner"] or bool(game.get("terminated")) != bool(
            done and not game.get("truncated")
        ):
            counts["outcome_mismatches"] += 1
    return ArenaReplayReceipt(games=len(games), **counts)
