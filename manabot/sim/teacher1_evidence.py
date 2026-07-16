"""Viewer-safe and reproducible evidence helpers for the Teacher-1 pilot.

The training observation and the authority audit are deliberately separate.
The former is the existing fixed tensor ABI.  The latter uses protocol-v1
frames and prompt-bound commands so a result can be replayed without exposing
the opponent's private hand to the learner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gui import trace as trace_store
from gui.experience_protocol import PROTOCOL_VERSION, Command, ExperienceFrame
from gui.server import (
    ASSET_MANIFEST_HASH,
    CONTENT_HASH,
    _offer_verb,
    describe_actions,
    serialize_observation,
)
from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.env.observation import ActionEnum, ActionSpaceEnum
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.mcts import DeterminizedPuctPlayer, determinized_puct
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs
import managym

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SCHEMA = REPO_ROOT / "protocol" / "experience-v1.schema.json"


class ContractError(ValueError):
    """The checked-in experiment contract does not match the runtime."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            REPO_ROOT / "manabot" / "sim" / "mcts.py",
            REPO_ROOT / "manabot" / "sim" / "flat_mc.py",
        ]
    )
    return paths


def runtime_fingerprints(seed: int = 197) -> dict[str, Any]:
    """Return identities that the pre-registration freezes before a run."""

    obs_space = ObservationSpace()
    matchup = MatchHypers(
        hero="teacher-a",
        villain="teacher-b",
        hero_deck=dict(INTERACTIVE_DECK),
        villain_deck=dict(INTERACTIVE_DECK),
    ).model_dump()
    env = Env(
        Match(MatchHypers(**matchup)),
        obs_space,
        Reward(RewardHypers()),
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
        "world": "w2",
        "engine_source_sha256": source_bundle_sha256(engine_source_paths()),
        "pilot_source_sha256": source_bundle_sha256(
            [
                Path(__file__).resolve(),
                REPO_ROOT / "experiments" / "runners" / "run_teacher1_pilot.py",
            ]
        ),
        "engine_extension_sha256": file_sha256(extension_path),
        "engine_extension_name": extension_path.name,
        "content_schema_version": content.get("schema_version"),
        "content_digest": content.get("content_digest"),
        "content_manifest_sha256": canonical_sha256(content),
        "observation_abi_sha256": canonical_sha256(observation_schema),
        "action_abi_sha256": canonical_sha256(action_abi),
        "experience_protocol_sha256": file_sha256(PROTOCOL_SCHEMA),
        "experience_content_hash": CONTENT_HASH,
        "asset_manifest_hash": ASSET_MANIFEST_HASH,
        "matchup_sha256": canonical_sha256(matchup),
        "matchup": matchup,
    }


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


def _action_space_name(raw: Any) -> str:
    return ActionSpaceEnum(int(raw.action_space.action_space_type)).name


def build_viewer_frame(
    raw: Any,
    *,
    match_id: str,
    revision: int,
    content_hash: str,
    asset_manifest_hash: str,
) -> dict[str, Any]:
    """Build the protocol frame for exactly the acting player's observation."""

    projection = serialize_observation(raw)
    trace_store.redact_observation(projection)
    actor = int(raw.agent.player_index)
    actions = describe_actions(raw)
    action_space = _action_space_name(raw)
    offers = [
        {
            "id": int(action["index"]),
            "actor": actor,
            "verb": _offer_verb(str(action["type"])),
            "source": None,
            "label": str(action["description"]),
            "help": None,
            "choices": [],
            "confirm_label": str(action["description"]),
            "action_type": str(action["type"]),
            "focus": [int(value) for value in action["focus"]],
        }
        for action in actions
    ]
    prompt = {
        "id": revision,
        "actor": actor,
        "kind": action_space.lower(),
        "title": "Teacher decision",
        "instruction": "Choose an action",
    }
    core = {
        "protocol": PROTOCOL_VERSION,
        "match_id": match_id,
        "revision": revision,
        "content_hash": content_hash,
        "asset_manifest_hash": asset_manifest_hash,
        "status": "ready",
        "prompt": prompt,
        "projection": projection,
        "offers": offers,
    }
    frame = {
        **core,
        "frame_hash": canonical_sha256(core),
        "winner": None,
        "action_space": action_space,
        "stops": {
            "my": [],
            "opponent": [],
            "stop_on_stack": False,
            "auto_pass": False,
        },
    }
    validated = ExperienceFrame.model_validate(frame)
    return validated.model_dump(mode="json", exclude_unset=True)


def build_command(frame: dict[str, Any], action: int) -> dict[str, Any]:
    offer_ids = {int(offer["id"]) for offer in frame["offers"]}
    if action not in offer_ids:
        raise RuntimeError(f"teacher action {action} is absent from legal offers")
    command = Command(
        command_id=f"{frame['match_id']}:{frame['revision']}",
        match_id=str(frame["match_id"]),
        expected_revision=int(frame["revision"]),
        prompt_id=int(frame["prompt"]["id"]),
        offer_id=action,
        answers=[],
    )
    return command.model_dump(mode="json")


def _fresh_env(seed: int) -> Env:
    match = Match(
        MatchHypers(
            hero="teacher-a",
            villain="teacher-b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(
        match,
        ObservationSpace(),
        Reward(RewardHypers()),
        seed=seed,
        auto_reset=False,
    )
    env.reset(seed=seed)
    return env


@dataclass(frozen=True)
class ReplayReceipt:
    games: int
    decisions: int
    frame_mismatches: int
    command_mismatches: int
    outcome_mismatches: int
    opponent_private_cards_exposed: int

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.frame_mismatches,
                self.command_mismatches,
                self.outcome_mismatches,
                self.opponent_private_cards_exposed,
            )
        )


def record_teacher_trajectories(
    *,
    games: int,
    simulations: int,
    worlds: int,
    c_puct: float,
    seed: int,
    content_hash: str,
    asset_manifest_hash: str,
    max_steps: int = 2000,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a small authoritative self-play audit, not a training shard."""

    if games < 1:
        raise ValueError("games must be positive")
    env = _fresh_env(seed)
    players = [
        DeterminizedPuctPlayer(
            simulations,
            worlds=worlds,
            c_puct=c_puct,
            max_steps=max_steps,
            seed=seed * 2 + seat + 1,
        )
        for seat in range(2)
    ]
    game_records: list[dict[str, Any]] = []
    for game_index in range(games):
        deal_seed = seed + game_index
        obs, _ = env.reset(seed=deal_seed)
        match_id = f"w2-t1-audit-{deal_seed}"
        decisions: list[dict[str, Any]] = []
        done = False
        info: dict[str, Any] = {}
        revision = 0
        while not done:
            raw = env.last_raw_obs
            actor = int(raw.agent.player_index)
            encoded_legal_count = int(np.count_nonzero(obs["actions_valid"]))
            frame = build_viewer_frame(
                raw,
                match_id=match_id,
                revision=revision,
                content_hash=content_hash,
                asset_manifest_hash=asset_manifest_hash,
            )
            root_digest = env._engine.state_digest()
            search_call_index = players[actor].stats.decisions + 1
            action = players[actor].act(env, obs)
            root_unchanged = env._engine.state_digest() == root_digest
            command = build_command(frame, action)
            result = players[actor].last_result
            if result is None:
                raise RuntimeError("Teacher-1 did not publish a search result")
            decisions.append(
                {
                    "frame": frame,
                    "command": command,
                    "actor": actor,
                    "seat": actor,
                    "opponent_class": "search",
                    "search": {
                        "simulations": result.simulations,
                        "worlds": result.worlds,
                        "visit_counts": result.visit_counts.astype(int).tolist(),
                        "q_values": result.q_values.astype(float).tolist(),
                        "root_value": result.root_value,
                        "tree_nodes": result.tree_nodes,
                        "max_depth": result.max_depth,
                        "cap_hits": result.cap_hits,
                        "encoded_legal_count": encoded_legal_count,
                        "root_unchanged": root_unchanged,
                        "player_seed": seed * 2 + actor + 1,
                        "call_index": search_call_index,
                    },
                }
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            revision += 1
        winner = winner_from_info_or_obs(info, env.last_raw_obs)
        game_records.append(
            {
                "game_index": game_index,
                "deal_seed": deal_seed,
                "match_id": match_id,
                "decisions": decisions,
                "winner": winner,
            }
        )
    return {
        "schema_version": 1,
        "teacher": {
            "kind": "determinized_puct",
            "simulations": simulations,
            "worlds": worlds,
            "c_puct": c_puct,
            "max_steps": max_steps,
        },
        "seed": seed,
        "provenance": provenance or {},
        "games": game_records,
    }


def replay_teacher_trajectories(
    artifact: dict[str, Any], *, content_hash: str, asset_manifest_hash: str
) -> ReplayReceipt:
    """Replay commands without rerunning search and compare viewer-safe frames."""

    frame_mismatches = 0
    command_mismatches = 0
    outcome_mismatches = 0
    private_exposures = 0
    decisions_seen = 0
    for game in artifact["games"]:
        env = _fresh_env(int(game["deal_seed"]))
        info: dict[str, Any] = {}
        done = False
        for revision, expected in enumerate(game["decisions"]):
            if done:
                command_mismatches += 1
                break
            frame = build_viewer_frame(
                env.last_raw_obs,
                match_id=str(game["match_id"]),
                revision=revision,
                content_hash=content_hash,
                asset_manifest_hash=asset_manifest_hash,
            )
            if frame != expected["frame"]:
                frame_mismatches += 1
            opponent = frame["projection"]["opponent"]
            if opponent.get("hand"):
                private_exposures += 1
            command = Command.model_validate(expected["command"])
            if (
                command.match_id != frame["match_id"]
                or command.expected_revision != revision
                or command.prompt_id != frame["prompt"]["id"]
                or command.offer_id not in {offer["id"] for offer in frame["offers"]}
            ):
                command_mismatches += 1
            _, _, terminated, truncated, info = env.step(int(command.offer_id))
            done = bool(terminated or truncated)
            decisions_seen += 1
        winner = winner_from_info_or_obs(info, env.last_raw_obs) if done else None
        if winner != game["winner"]:
            outcome_mismatches += 1
    return ReplayReceipt(
        games=len(artifact["games"]),
        decisions=decisions_seen,
        frame_mismatches=frame_mismatches,
        command_mismatches=command_mismatches,
        outcome_mismatches=outcome_mismatches,
        opponent_private_cards_exposed=private_exposures,
    )


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)


def _teacher_action(result: Any) -> int:
    """Apply Teacher-1's visit-first, Q-value tie break exactly."""

    candidates = np.flatnonzero(result.visit_counts == int(result.visit_counts.max()))
    if len(candidates) == 1:
        return int(candidates[0])
    return int(candidates[np.argmax(result.q_values[candidates])])


def evaluate_root_stability(
    *,
    budgets: list[int],
    worlds: int,
    c_puct: float,
    roots: int,
    repeats: int,
    seed: int,
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Measure repeated-search stability on a fixed teacher-driven root stream."""

    if roots < 1 or repeats < 2:
        raise ValueError("roots must be positive and repeats must be at least two")
    if not budgets or any(budget < worlds for budget in budgets):
        raise ValueError("every budget must be at least the number of worlds")
    env = _fresh_env(seed)
    obs, _ = env.reset(seed=seed)
    del obs
    per_budget: dict[int, dict[str, list[float] | int]] = {
        budget: {
            "pair_js": [],
            "pair_action_agreement": [],
            "target_entropy": [],
            "visit_coverage": [],
            "tree_nodes": [],
            "max_depth": [],
            "cap_hits": 0,
        }
        for budget in budgets
    }
    roots_seen = 0
    game_index = 0
    while roots_seen < roots:
        if env.last_raw_obs.game_over:
            game_index += 1
            env.reset(seed=seed + game_index)
        repeat_results: dict[int, list[Any]] = {}
        for budget in budgets:
            repeat_results[budget] = []
            for repeat in range(repeats):
                root_digest = env._engine.state_digest()
                result = determinized_puct(
                    env._engine,
                    simulations=budget,
                    worlds=worlds,
                    c_puct=c_puct,
                    seed=seed + roots_seen * 10_000 + repeat,
                    max_steps=max_steps,
                )
                if env._engine.state_digest() != root_digest:
                    raise RuntimeError(
                        "Teacher-1 search mutated its authoritative root"
                    )
                repeat_results[budget].append(result)
            block = per_budget[budget]
            distributions = [
                result.visit_counts.astype(np.float64) / result.visit_counts.sum()
                for result in repeat_results[budget]
            ]
            actions = [_teacher_action(result) for result in repeat_results[budget]]
            for left in range(repeats):
                for right in range(left + 1, repeats):
                    block["pair_js"].append(
                        _js_divergence(distributions[left], distributions[right])
                    )
                    block["pair_action_agreement"].append(
                        float(actions[left] == actions[right])
                    )
            for distribution, result in zip(distributions, repeat_results[budget]):
                positive = distribution > 0
                block["target_entropy"].append(
                    float(
                        -np.sum(distribution[positive] * np.log(distribution[positive]))
                    )
                )
                block["visit_coverage"].append(float(np.mean(positive)))
                block["tree_nodes"].append(float(result.tree_nodes))
                block["max_depth"].append(float(result.max_depth))
                block["cap_hits"] += int(result.cap_hits)
        high = repeat_results[max(budgets)][0]
        action = _teacher_action(high)
        _, _, terminated, truncated, _ = env.step(action)
        roots_seen += 1
        if terminated or truncated:
            game_index += 1
            env.reset(seed=seed + game_index)

    aggregate: dict[str, Any] = {}
    for budget, block in per_budget.items():
        aggregate[str(budget)] = {
            "roots": roots_seen,
            "searches": roots_seen * repeats,
            "top_action_agreement": float(np.mean(block["pair_action_agreement"])),
            "median_js_divergence": float(np.median(block["pair_js"])),
            "mean_target_entropy": float(np.mean(block["target_entropy"])),
            "mean_visit_coverage": float(np.mean(block["visit_coverage"])),
            "mean_tree_nodes": float(np.mean(block["tree_nodes"])),
            "mean_max_depth": float(np.mean(block["max_depth"])),
            "cap_hits": int(block["cap_hits"]),
        }
    return aggregate


def receipt_dict(receipt: ReplayReceipt) -> dict[str, Any]:
    return {**asdict(receipt), "passed": receipt.passed}
