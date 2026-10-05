"""Score bound checkpoints on retained tactical roots without training.

ScenarioScore owns diagnostic results; arena traces own Command evidence. These
injected prefixes measure behavior against a fixed script, not full-game strength.
The ordinary checkpoint loader and encoder retain world/setup/capacity authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue
import torch

from etude.server import ASSET_MANIFEST_HASH, CONTENT_HASH
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.arena.replay import read_trace, replay_games, write_trace
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.sim.flat_mc import AgentMatchupPlayer, make_player
from manabot.sim.teacher1_evidence import build_command, build_viewer_frame
from manabot.verify.competency import (
    SCENARIOS,
    ScriptedVillain,
    _action_casts,
    _battlefield_pairs,
    _hero_view,
    build_scenario_env,
)
from manabot.verify.scenario_validation import (
    _outcome,
    scenario_identity,
    validate_scenarios,
)


class ScenarioScore(BaseModel):
    """One retained attempt, including unavailable bindings and execution errors."""

    model_config = ConfigDict(extra="forbid")
    scenario: str
    seed: int
    checkpoint_sha256: str | None = None
    status: Literal["scored", "unsupported", "failed"] = "failed"
    correct: bool | None = None
    outcome: str = "unavailable"
    error: str | None = None
    decisions: int = 0
    behaviors: dict[str, JsonValue] = Field(default_factory=dict)
    seconds: float = 0
    trace_sha256: str | None = None
    replay_passed: bool = False


def score_checkpoint(
    scenario_name: str, checkpoint: Path, *, seed: int = 2, max_steps: int = 1000
) -> tuple[ScenarioScore, dict[str, Any] | None]:
    """Run deterministic ordinary policy decisions with a fixed scripted opponent.

    The returned unstructured payload is the existing arena trace wire format.
    No checkpoint is rewritten or admitted against a substitute setup. Failed
    attempts retain all successfully executed Commands and elapsed cost.
    """
    if scenario_name not in SCENARIOS or seed < 0 or max_steps < 1:
        raise ValueError("known scenario, nonnegative seed and positive bound required")
    score = ScenarioScore(scenario=scenario_name, seed=seed)
    started = time.perf_counter()
    env = None
    game: dict[str, Any] | None = None
    try:
        score.checkpoint_sha256 = file_sha256(checkpoint)
        player, space = make_player(
            {"kind": "checkpoint", "path": str(checkpoint), "deterministic": True},
            seed=seed,
        )
        if not isinstance(player, AgentMatchupPlayer):
            score.status = "unsupported"
            score.error = "Injected roots do not supply belief checkpoint history"
            return score, None
        assert space is not None
        scenario = SCENARIOS[scenario_name]
        env, observation, _ = build_scenario_env(scenario, space, seed)
        validate_agent_setup(player.agent, env.match.to_rust())
        player.start_game(env, 0)
        game = {
            "match_id": f"scenario:{scenario_name}:{seed}:{score.checkpoint_sha256}",
            "scenario_root": scenario_identity(scenario_name),
            "world_binding": checkpoint_world(env.match.to_rust(), space),
            "observation_hypers": space.encoder.hypers.model_dump(),
            "deal_seed": seed,
            "seat_players": [score.checkpoint_sha256, "scripted-villain"],
            "checkpoint_sha256": score.checkpoint_sha256,
            "deterministic": True,
            "max_steps": max_steps,
            "initial_state_digest": env._engine.state_digest(),
            "decisions": [],
            "winner": None,
            "terminated": False,
            "truncated": False,
            "failure": None,
        }
        villain = ScriptedVillain(scenario.villain_casts)
        tracker = scenario.tracker()
        cast: str | None = None
        creatures_at_cast = 0
        for revision in range(max_steps + 1):
            raw = env.last_raw_obs
            tracker.observe_state(raw)
            resolution = _outcome(scenario, raw, True, cast, creatures_at_cast)
            contrast = _outcome(scenario, raw, False, cast, creatures_at_cast)
            if resolution is not None or contrast is not None:
                score.correct = resolution[0] if resolution is not None else False
                score.outcome = (resolution or contrast)[1]
                score.status = "scored"
                break
            if raw.game_over or int(raw.turn.turn_number) > scenario.max_turns:
                score.correct = False
                score.outcome = (
                    "reference behavior not achieved within fixture turn bound"
                )
                score.status = "scored"
                break
            if revision == max_steps:
                raise RuntimeError("scenario exceeded Command cap")
            actor = int(raw.agent.player_index)
            frame = build_viewer_frame(
                raw,
                match_id=game["match_id"],
                revision=revision,
                content_hash=CONTENT_HASH,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
            )
            if frame["projection"]["opponent"].get("hand"):
                raise RuntimeError("viewer frame exposes hidden hand")
            pre = env._engine.state_digest()
            action = player.act(env, observation) if actor == 0 else villain.act(raw)
            if env._engine.state_digest() != pre:
                raise RuntimeError("player mutated scenario root")
            offers = {int(offer["id"]): offer for offer in frame["offers"]}
            if action not in offers:
                raise ValueError("player selected illegal offer")
            if actor == 0:
                tracker.observe_hero(raw, action)
                spell = _action_casts(raw, action)
                if spell is not None:
                    cast = spell
                    _, side = _hero_view(raw)
                    creatures_at_cast = sum(
                        bool(card.card_types.is_creature)
                        for card, _ in _battlefield_pairs(raw, side)
                    )
            command = build_command(frame, action)
            observation, _, terminated, truncated, _ = env.step(action)
            game["decisions"].append(
                {
                    "actor": actor,
                    "player_id": game["seat_players"][actor],
                    "action_space_kind": frame["action_space"],
                    "frame_sha256": canonical_sha256(frame),
                    "command": command,
                    "command_sha256": canonical_sha256(command),
                    "chosen_offer": offers[action],
                    "pre_state_digest": pre,
                    "post_state_digest": env._engine.state_digest(),
                }
            )
            game["terminated"] = bool(terminated)
            game["truncated"] = bool(truncated)
            game["winner"] = env._engine.winner_index() if terminated else None
            if truncated:
                raise RuntimeError("scenario environment truncated")
        if hasattr(tracker, "set_winner"):
            tracker.set_winner(game["winner"])
        score.behaviors = tracker.result()
        # A favorable resolution alone does not prove the named behavior: S4
        # requires the flying attack/ground hold, S5 the main-phase mana hold.
        # S2 additionally needs the resolved removal checked above; its legacy
        # tracker only observes cast intent.
        intended_behavior = bool(score.behaviors.pop("correct"))
        score.correct = bool(score.correct and intended_behavior)
        if file_sha256(checkpoint) != score.checkpoint_sha256:
            raise RuntimeError("checkpoint bytes changed during scoring")
    except Exception as error:
        score.error = f"{type(error).__name__}: {error}"
        # Invalid identity/capacity is a failed admission, never a tactical loss.
        score.status = "failed"
        score.correct = None
        if game is not None:
            game["failure"] = score.error
    finally:
        if env is not None:
            env.close()
        if game is not None:
            score.decisions = len(game["decisions"])
            game["scenario_outcome"] = score.outcome
            game["game_trace_sha256"] = canonical_sha256(game)
            score.trace_sha256 = game["game_trace_sha256"]
            try:
                score.replay_passed = replay_games(
                    [game], require_terminal=False
                ).passed
            except Exception as error:
                score.error = f"replay failed: {type(error).__name__}: {error}"
            if not score.replay_passed:
                score.status = "failed"
                score.correct = None
                score.error = score.error or "scenario Command replay failed"
        score.seconds = time.perf_counter() - started
    return score, game


def write_report(directory: Path) -> None:
    """Regenerate a readable report from saved evidence without loading policies."""
    payload = json.loads((directory / "results.json").read_text())
    scores = [ScenarioScore.model_validate(row) for row in payload["scores"]]
    games = read_trace(directory / "traces.jsonl.gz")
    if file_sha256(directory / "traces.jsonl.gz") != payload["trace"]["sha256"]:
        raise ValueError("scenario trace artifact digest mismatch")
    replay = replay_games(games, require_terminal=False)
    if not replay.passed:
        raise ValueError("offline scenario replay failed")
    lines = [
        "# Checkpoint tactical behaviors",
        "",
        "Injected custom-deck prefixes, deterministic policy, fixed scripted opponent. "
        "These are local behavior diagnostics, not optimal-strategy, full-game strength "
        "or robustness conclusions. ETU-99 owns empirical comparisons.",
        "",
        f"Exact prefix replay: {replay.decisions} Commands in {replay.games} attempts.",
        f"Premise validation: {payload['validation_seconds']:.3f} seconds (separate from checkpoint attempts).",
        "",
        "| Scenario | Status | Reference achieved | Outcome / error |",
        "| --- | --- | --- | --- |",
    ]
    for score in scores:
        detail = (score.error or score.outcome).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {score.scenario} | {score.status} | {score.correct} | {detail} |"
        )
    lines.extend(["", "## Behavior details", ""])
    for score in scores:
        lines.extend(
            [
                f"### {score.scenario}",
                "",
                f"Checkpoint: `{score.checkpoint_sha256}`; seed {score.seed}; "
                f"{score.decisions} Commands; {score.seconds:.3f} seconds including replay.",
                "",
                f"Measurements: `{json.dumps(score.behaviors, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Premise validation",
            "",
            "Reference/contrast scripts check resolved effects; they do not prove global optimality.",
        ]
    )
    for record in payload["validation"]:
        lines.append(
            f"- {record['scenario']}: {record['status']} — {record['premise']}"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        help="JSON mapping scenario names to checkpoint paths",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.report_only:
        write_report(args.out)
        return
    if args.checkpoints is None:
        parser.error("--checkpoints is required for scoring")
    assignments = json.loads(args.checkpoints.read_text())
    if not isinstance(assignments, dict) or any(
        name not in SCENARIOS or not isinstance(path, str)
        for name, path in assignments.items()
    ):
        parser.error(
            "checkpoint mapping requires known scenario names and path strings"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    validation_started = time.perf_counter()
    validation = validate_scenarios(seeds=(args.seed,), max_steps=args.max_steps)
    validation_seconds = time.perf_counter() - validation_started
    scores: list[ScenarioScore] = []
    games: list[dict[str, Any]] = []  # Existing arena trace wire boundary.
    for record in validation:
        if record.status != "validated_fixture":
            score = ScenarioScore(
                scenario=record.scenario,
                seed=args.seed,
                status="unsupported",
                error="strategic premise validation failed",
            )
        elif record.scenario not in assignments:
            score = ScenarioScore(
                scenario=record.scenario,
                seed=args.seed,
                status="unsupported",
                error="no checkpoint bound to this scenario setup",
            )
        else:
            score, game = score_checkpoint(
                record.scenario,
                Path(assignments[record.scenario]),
                seed=args.seed,
                max_steps=args.max_steps,
            )
            if game is not None:
                games.append(game)
        scores.append(score)
        trace = write_trace(args.out / "traces.jsonl.gz", games)
        (args.out / "results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "validation_seconds": validation_seconds,
                    "validation": [asdict(item) for item in validation],
                    "scores": [item.model_dump() for item in scores],
                    "trace": trace,
                },
                indent=2,
            )
            + "\n"
        )
    write_report(args.out)
    if any(score.status == "failed" for score in scores):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
