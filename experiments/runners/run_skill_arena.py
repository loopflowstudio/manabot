#!/usr/bin/env python3
"""Freeze and challenge the world-pinned manabot skill arena."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from itertools import combinations
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from manabot.arena.competency import run_competencies
from manabot.arena.match import play_cell
from manabot.arena.models import (
    ArenaContract,
    MatchRow,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.arena.players import build_player
from manabot.arena.profile import (
    native_gameplay_profiles,
    profile_players,
    select_profile_roots,
    verify_profile,
)
from manabot.arena.rating import bootstrap_population, fit_population, payoff_matrix
from manabot.arena.replay import read_trace, replay_games
from manabot.sim.teacher1_evidence import (
    REPO_ROOT,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.verify.competency import SCENARIOS, aggregate_scenario_results


class ArenaError(RuntimeError):
    """The closed arena contract or artifact failed validation."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


def source_digest(paths: tuple[str, ...] | list[str]) -> str:
    return source_bundle_sha256([REPO_ROOT / path for path in paths])


def arena_runtime_fingerprints() -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    engine_runtime = runtime_fingerprints()
    engine_runtime.pop("matchup")
    engine_runtime.pop("pilot_source_sha256")
    return {
        **engine_runtime,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": metadata.version("numpy"),
        "torch_version": metadata.version("torch"),
        "pydantic_version": metadata.version("pydantic"),
        "psutil_version": metadata.version("psutil"),
        "torch_threads_per_player": torch.get_num_threads(),
        "inference_device": "cpu",
    }


def registration_source_paths(registration: PlayerRegistration) -> list[str]:
    kind = registration.player_spec["kind"]
    if kind == "scripted_greedy":
        return ["manabot/arena/players.py"]
    if kind == "determinized_puct":
        return [
            "manabot/sim/flat_mc.py",
            "manabot/sim/mcts.py",
            "manabot/sim/search_branch.py",
            "managym/src/python/bindings.rs",
        ]
    return ["manabot/sim/flat_mc.py"]


def validate_registration_source(registration: PlayerRegistration) -> None:
    if registration.runner_kind != "code":
        return
    actual = source_digest(registration_source_paths(registration))
    if registration.source_sha256 != actual:
        raise ArenaError(f"source drift for {registration.player_id}: {actual}")


def preflight_contract(path: Path) -> tuple[ArenaContract, str, dict[str, Any]]:
    raw = load_json(path)
    contract = ArenaContract.model_validate(raw)
    contract_sha = file_sha256(path)
    actual_source = source_digest(contract.source_paths)
    if actual_source != contract.source_sha256:
        raise ArenaError(f"arena source drift: {actual_source}")
    actual_runtime = arena_runtime_fingerprints()
    mismatches = {
        key: {"expected": expected, "actual": actual_runtime.get(key)}
        for key, expected in contract.runtime.items()
        if actual_runtime.get(key) != expected
    }
    if mismatches:
        raise ArenaError("runtime drift: " + json.dumps(mismatches, sort_keys=True))
    for registration in contract.anchors:
        validate_registration_source(registration)
    return contract, contract_sha, actual_runtime


def preflight_candidate(
    path: Path,
    checkpoint_path: Path | None,
    *,
    contract: ArenaContract,
    profile_name: str,
    validate_checkpoint_load: bool = True,
) -> tuple[PlayerRegistration, dict[str, str]]:
    registration = PlayerRegistration.model_validate(load_json(path))
    if registration.role != "challenger":
        raise ArenaError("candidate registration role must be challenger")
    if (
        registration.world != contract.key.world
        or registration.content_suite != contract.key.content_suite
        or registration.information_boundary != contract.key.viewer_boundary
        or registration.observation_abi_sha256
        != contract.runtime["observation_abi_sha256"]
        or registration.action_abi_sha256 != contract.runtime["action_abi_sha256"]
        or registration.matchup_sha256 != contract.runtime["matchup_sha256"]
    ):
        raise ArenaError("candidate world/content/viewer compatibility mismatch")
    if registration.evidence_class == "fixture" and profile_name != "smoke":
        raise ArenaError("fixture checkpoint registration is smoke-only")
    checkpoint_paths: dict[str, str] = {}
    if registration.runner_kind == "code":
        if checkpoint_path is not None:
            raise ArenaError("code candidate rejects --candidate-checkpoint")
        validate_registration_source(registration)
    else:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ArenaError("checkpoint candidate bytes are unavailable")
        if file_sha256(checkpoint_path) != registration.checkpoint_sha256:
            raise ArenaError("checkpoint candidate SHA-256 mismatch")
        if checkpoint_path.stat().st_size != registration.checkpoint_bytes:
            raise ArenaError("checkpoint candidate byte-size mismatch")
        checkpoint_paths[registration.player_id] = str(checkpoint_path)
        if validate_checkpoint_load:
            player, observation_space = build_player(
                registration, seed=0, checkpoint_path=str(checkpoint_path)
            )
            if observation_space is None:
                raise ArenaError("checkpoint candidate has no observation space")
            observation_schema = {
                name: {"shape": list(shape), "dtype": "float32"}
                for name, shape in sorted(observation_space.shapes.items())
            }
            if canonical_sha256(observation_schema) != (
                registration.observation_abi_sha256
            ):
                raise ArenaError("checkpoint candidate observation-shape mismatch")
            agent = getattr(player, "agent", None)
            parameter_count = (
                sum(parameter.numel() for parameter in agent.parameters())
                if agent is not None
                else None
            )
            if parameter_count != registration.parameter_count:
                raise ArenaError("checkpoint candidate parameter-count mismatch")
    return registration, checkpoint_paths


def append_ledger(out_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    path = out_dir / "resource-ledger.jsonl"
    previous = None
    if path.exists() and path.read_text().splitlines():
        previous = json.loads(path.read_text().splitlines()[-1])["event_sha256"]
    unsigned = {
        "stage": stage,
        "recorded_unix": time.time(),
        "previous_event_sha256": previous,
        "payload": payload,
    }
    event = {**unsigned, "event_sha256": canonical_sha256(unsigned)}
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def play_cells(
    *,
    contract: ArenaContract,
    pairs: list[tuple[PlayerRegistration, PlayerRegistration]],
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    jobs = [
        {
            "key": contract.key,
            "player_a": first,
            "player_b": second,
            "deal_seeds": deal_seeds,
            "out_dir": out_dir,
            "checkpoint_paths": checkpoint_paths,
        }
        for first, second in pairs
    ]
    with ProcessPoolExecutor(
        max_workers=contract.resource_caps.outcome_workers
    ) as executor:
        return list(executor.map(_play_cell_job, jobs))


def _play_cell_job(
    job: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return play_cell(**job)


def artifact_receipts(out_dir: Path, names: list[str]) -> dict[str, Any]:
    return {
        name: {"path": name, "sha256": file_sha256(out_dir / name)} for name in names
    }


def portable_trace_receipts(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = []
    for trace in traces:
        receipt = dict(trace)
        receipt["path"] = receipt.pop("artifact_path")
        receipts.append(receipt)
    return receipts


def artifact_bytes(out_dir: Path) -> int:
    return sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())


def resource_cap_receipt(
    out_dir: Path, contract: ArenaContract, *, started: float
) -> dict[str, Any]:
    wall_seconds = time.perf_counter() - started
    wall_hours = wall_seconds / 3600.0
    core_hours = wall_hours * contract.resource_caps.outcome_workers
    current_artifact_bytes = artifact_bytes(out_dir)
    clauses = {
        "wall_hours": {
            "actual": wall_hours,
            "maximum": contract.resource_caps.wall_hours,
            "passed": wall_hours <= contract.resource_caps.wall_hours,
        },
        "core_hours_conservative": {
            "actual": core_hours,
            "maximum": contract.resource_caps.core_hours,
            "passed": core_hours <= contract.resource_caps.core_hours,
        },
        "artifact_bytes_at_decision": {
            "actual": current_artifact_bytes,
            "maximum": contract.resource_caps.artifact_bytes,
            "passed": current_artifact_bytes <= contract.resource_caps.artifact_bytes,
        },
    }
    return {
        "clauses": clauses,
        "passed": all(row["passed"] for row in clauses.values()),
    }


def finalize_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {**manifest, "manifest_sha256": manifest_digest(manifest)}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def verify_ledger(path: Path) -> dict[str, Any]:
    previous = None
    events = read_jsonl(path)
    for event in events:
        unsigned = dict(event)
        event_sha256 = unsigned.pop("event_sha256", None)
        if unsigned.get("previous_event_sha256") != previous:
            raise ArenaError("resource ledger chain mismatch")
        if event_sha256 != canonical_sha256(unsigned):
            raise ArenaError("resource ledger event digest mismatch")
        previous = event_sha256
    if not events or events[-1]["stage"] != "complete":
        raise ArenaError("resource ledger is incomplete")
    return {"events": len(events), "head_sha256": previous}


def report_markdown(
    *,
    fit: Any,
    promotion: dict[str, Any],
    matrix: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    lines = [
        "# Manabot Skill Arena",
        "",
        f"Disposition: **{promotion['disposition']}**",
        f"Reason: `{promotion.get('reason')}`",
        "",
        "## Ratings",
        "",
    ]
    for player, rating in sorted(fit.ratings.items(), key=lambda item: -item[1]):
        lines.append(f"- `{player}`: {rating:.1f} Elo")
    lines.extend(
        [
            "",
            f"Seat-0 effect: {fit.seat0_elo:.1f} Elo",
            f"Log loss: {fit.log_loss:.6f}",
            f"Payoff cells: {len(matrix)}",
            f"Replay passed: {replay.get('passed', False)}",
            "",
        ]
    )
    return "\n".join(lines)


def rating_payload(
    rows: list[dict[str, Any]], schedule: Any, contract: ArenaContract
) -> tuple[Any, dict[str, Any]]:
    model = contract.rating_model
    fit = fit_population(
        rows,
        anchor=model.anchor_player_id,
        prior_elo_std=model.prior_elo_std,
        tolerance=model.tolerance,
        max_iterations=model.max_iterations,
    )
    if not fit.converged:
        raise ArenaError("Bradley-Terry MAP fit did not converge")
    bootstrap = bootstrap_population(
        rows,
        replicates=schedule.bootstrap_replicates,
        seed=schedule.bootstrap_seed,
        anchor=model.anchor_player_id,
        prior_elo_std=model.prior_elo_std,
    )
    matrix = payoff_matrix(rows)
    largest_residual_cells = sorted(
        matrix.values(),
        key=lambda cell: abs(float(cell["raw_residual_percentage_points"])),
        reverse=True,
    )[:10]
    payload = {
        **model.model_dump(),
        "rating_model_sha256": contract.key.rating_prior_sha256,
        "ratings": fit.ratings,
        "seat0_elo": fit.seat0_elo,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "gradient_norm": fit.gradient_norm,
        "hessian_condition": fit.hessian_condition,
        "log_loss": fit.log_loss,
        "deviance": fit.deviance,
        "residual_rows": list(fit.rows),
        "largest_residual_cells": largest_residual_cells,
        "bootstrap": bootstrap,
    }
    return fit, payload


def promotion_payload(
    candidate: PlayerRegistration | None,
    rows: list[dict[str, Any]],
    *,
    smoke: bool,
    profile: dict[str, Any],
    competencies: dict[str, Any],
    contract: ArenaContract,
) -> dict[str, Any]:
    row_integrity_passed = all(
        row.get("replay_passed")
        and not any(int(value) for value in row["integrity"].values())
        for row in rows
    )
    matched_players = profile["matched_root"]["players"]
    profile_integrity_passed = all(
        int(player["root_mutations"]) == 0
        and int(player["illegal_actions"]) == 0
        and float(player["playout_cap_rate"]) <= contract.promotion.playout_cap_rate_max
        for player in matched_players.values()
    )
    resource_caps_passed = bool(profile["resource_caps"]["passed"])
    integrity_passed = (
        row_integrity_passed and profile_integrity_passed and resource_caps_passed
    )
    if not integrity_passed:
        disposition = {
            "disposition": "invalid_integrity",
            "reason": "zero_tolerance_gate",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    if smoke:
        disposition = {
            "disposition": "engineering_smoke_non_promotion",
            "reason": "fixture_or_smoke_profile",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    if candidate is None:
        disposition = {
            "disposition": "rated_not_promotion_eligible",
            "reason": "anchor_freeze",
        }
        return {**disposition, "input_sha256": canonical_sha256(disposition)}
    matched = profile["matched_root"]["players"][candidate.player_id]
    integrity = {
        "match_rows_passed": row_integrity_passed,
        "matched_root_mutations": int(matched["root_mutations"]),
        "matched_root_illegal_actions": int(matched["illegal_actions"]),
        "playout_cap_rate": float(matched["playout_cap_rate"]),
        "playout_cap_rate_max": contract.promotion.playout_cap_rate_max,
        "passed": int(matched["root_mutations"]) == 0
        and int(matched["illegal_actions"]) == 0
        and float(matched["playout_cap_rate"])
        <= contract.promotion.playout_cap_rate_max
        and resource_caps_passed,
        "resource_caps": profile["resource_caps"],
    }
    disposition = {
        "disposition": "rated_not_promotion_eligible",
        "reason": "incumbent_not_in_cohort",
        "candidate": candidate.player_id,
        "compute_class_id": candidate.compute_class_id,
        "eligibility": {
            "available": False,
            "reason": "incumbent_not_in_cohort",
            "fixed_compute": {"available": False},
            "rating_delta": {"available": False},
            "competency_noninferiority": {"available": False},
        },
        "integrity": integrity,
        "candidate_competencies_sha256": canonical_sha256(
            competencies["players"][candidate.player_id]
        ),
    }
    return {**disposition, "input_sha256": canonical_sha256(disposition)}


def verify_manifest(
    out_dir: Path, contract: ArenaContract, contract_sha: str
) -> dict[str, Any]:
    manifest = load_json(out_dir / "manifest.json")
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise ArenaError("manifest digest mismatch")
    if manifest.get("contract_sha256") != contract_sha:
        raise ArenaError("manifest contract mismatch")
    if manifest.get("arena_key") != contract.key.model_dump():
        raise ArenaError("manifest arena key mismatch")
    if manifest.get("runtime") != contract.runtime:
        raise ArenaError("manifest runtime identity mismatch")
    profile_name = str(manifest.get("profile"))
    if profile_name not in contract.profiles:
        raise ArenaError("manifest profile mismatch")
    schedule = contract.schedules[profile_name]
    for receipt in manifest["artifacts"].values():
        path = out_dir / receipt["path"]
        if not path.is_file() or file_sha256(path) != receipt["sha256"]:
            raise ArenaError(f"artifact digest mismatch: {path}")
    ledger = verify_ledger(out_dir / "resource-ledger.jsonl")
    replay_receipts = []
    traced_games: dict[str, str] = {}
    source_profile_games: list[dict[str, Any]] | None = None
    for trace in manifest["traces"]:
        path = out_dir / trace.get("artifact_path", trace["path"])
        if not path.is_file() or file_sha256(path) != trace["sha256"]:
            raise ArenaError(f"trace digest mismatch: {path}")
        games = read_trace(path)
        if path.name == "random-v1__scripted-greedy-v1.commands.jsonl.gz":
            source_profile_games = games
        if len(games) != trace["games"]:
            raise ArenaError(f"trace game-count mismatch: {path}")
        replay_receipts.append(replay_games(games).to_dict())
        for game in games:
            digest = str(game["game_trace_sha256"])
            if digest in traced_games:
                raise ArenaError("duplicate game trace identity")
            traced_games[digest] = str(trace["sha256"])
    if replay_receipts and not all(receipt["passed"] for receipt in replay_receipts):
        raise ArenaError("Command replay mismatch")
    rows = read_jsonl(out_dir / "matches.jsonl")
    try:
        validated_rows = [MatchRow.model_validate(row) for row in rows]
    except ValueError as error:
        raise ArenaError(f"match row identity mismatch: {error}") from error
    expected_cells = 10 if manifest["kind"] == "anchor-freeze" else 15
    expected_games = expected_cells * len(schedule.deal_seeds) * 2
    if len(rows) != expected_games:
        raise ArenaError("match schedule game-count mismatch")
    if len({row.cell_id for row in validated_rows}) != expected_cells:
        raise ArenaError("match schedule cell-count mismatch")
    if {row.deal_seed for row in validated_rows} != set(schedule.deal_seeds):
        raise ArenaError("match schedule deal-seed mismatch")
    row_keys = {(row.cell_id, row.deal_block, row.leg) for row in validated_rows}
    if len(row_keys) != expected_games:
        raise ArenaError("match schedule contains duplicate or missing seat legs")
    if any(
        row.deal_block >= len(schedule.deal_seeds)
        or schedule.deal_seeds[row.deal_block] != row.deal_seed
        for row in validated_rows
    ):
        raise ArenaError("match deal-block mapping mismatch")
    expected_seed_set = canonical_sha256(list(schedule.deal_seeds))
    if any(row.deal_seed_set_sha256 != expected_seed_set for row in validated_rows):
        raise ArenaError("match row schedule identity mismatch")
    local_rows = [
        row for row in validated_rows if row.game_trace_sha256 in traced_games
    ]
    expected_local_cells = 10 if manifest["kind"] == "anchor-freeze" else 5
    if len(local_rows) != expected_local_cells * len(schedule.deal_seeds) * 2:
        raise ArenaError("local trace-to-match coverage mismatch")
    if any(
        row.trace_shard_sha256 != traced_games[row.game_trace_sha256]
        for row in local_rows
    ):
        raise ArenaError("match row trace-shard binding mismatch")
    players = [
        PlayerRegistration.model_validate(player)
        for player in load_json(out_dir / "players.json")
    ]
    expected_players = list(contract.anchors)
    if manifest["kind"] == "challenge":
        expected_players.append(
            PlayerRegistration.model_validate(manifest["candidate"])
        )
    if players != expected_players:
        raise ArenaError("player registry artifact mismatch")
    candidate = (
        PlayerRegistration.model_validate(manifest["candidate"])
        if manifest["kind"] == "challenge"
        else None
    )
    _, recomputed_rating = rating_payload(rows, schedule, contract)
    stored_rating = load_json(out_dir / "rating.json")
    if recomputed_rating != stored_rating:
        raise ArenaError("rating recomputation mismatch")
    if payoff_matrix(rows) != load_json(out_dir / "payoff-matrix.json"):
        raise ArenaError("payoff matrix/residual recomputation mismatch")
    competencies = load_json(out_dir / "competencies.json")
    if (
        competencies.get("scenario_seeds") != list(schedule.competency_seeds)
        or competencies.get("scenario_seed_set_sha256")
        != canonical_sha256(list(schedule.competency_seeds))
        or set(competencies["players"])
        != {player.player_id for player in expected_players}
    ):
        raise ArenaError("competency schedule or player identity mismatch")
    for player_id, scenarios in competencies["players"].items():
        if set(scenarios) != set(SCENARIOS):
            raise ArenaError(f"competency scenario identity mismatch: {player_id}")
        for scenario_name, evidence in scenarios.items():
            if [run["run_seed"] for run in evidence["runs"]] != list(
                schedule.competency_seeds
            ):
                raise ArenaError(
                    f"competency seed-row mismatch: {player_id}/{scenario_name}"
                )
            if aggregate_scenario_results(evidence["runs"]) != evidence["aggregate"]:
                raise ArenaError(
                    f"competency aggregate mismatch: {player_id}/{scenario_name}"
                )
    profile_payload = load_json(out_dir / "profile.json")
    native_rows = (
        rows
        if manifest["kind"] == "anchor-freeze"
        else [
            row
            for row in rows
            if manifest["candidate"]["player_id"] in {row["player_a"], row["player_b"]}
        ]
    )
    if profile_payload["native_gameplay"] != native_gameplay_profiles(
        native_rows, worker_count=contract.resource_caps.outcome_workers
    ):
        raise ArenaError("native gameplay cost recomputation mismatch")
    verify_profile(profile_payload["matched_root"])
    if manifest["kind"] == "anchor-freeze":
        if source_profile_games is None:
            raise ArenaError("anchor artifact has no matched-root source trace")
        expected_roots = select_profile_roots(
            source_profile_games,
            warmup=contract.profile_roots.warmup,
            measured=contract.profile_roots.measured,
        )
        if profile_payload["matched_root"]["root_ids"] != [
            root["root_id"] for root in expected_roots
        ]:
            raise ArenaError("matched-root selection recomputation mismatch")
    expected_profile_players = (
        {player.player_id for player in contract.anchors}
        if candidate is None
        else {candidate.player_id}
    )
    if set(profile_payload["matched_root"]["players"]) != expected_profile_players:
        raise ArenaError("matched-root profile player identity mismatch")
    resource_receipt = profile_payload["resource_caps"]
    if not resource_receipt.get("passed") or not all(
        clause["passed"] and float(clause["actual"]) <= float(clause["maximum"])
        for clause in resource_receipt["clauses"].values()
    ):
        raise ArenaError("resource cap receipt mismatch")
    if artifact_bytes(out_dir) > contract.resource_caps.artifact_bytes:
        raise ArenaError("artifact resource cap exceeded")
    disposition_rows = (
        [
            row
            for row in rows
            if candidate.player_id in {row["player_a"], row["player_b"]}
        ]
        if candidate is not None
        else rows
    )
    recomputed_promotion = promotion_payload(
        candidate,
        disposition_rows,
        smoke=contract.profiles[profile_name].disposition != "production",
        profile=profile_payload,
        competencies=competencies,
        contract=contract,
    )
    if recomputed_promotion != load_json(out_dir / "promotion.json"):
        raise ArenaError("promotion receipt recomputation mismatch")
    return {
        "verified": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "games": len(rows),
        "replay": replay_receipts,
        "ledger": ledger,
    }


def freeze_anchors(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    contract, contract_sha, runtime = preflight_contract(args.contract)
    profile = contract.profiles[args.profile]
    schedule = contract.schedules[args.profile]
    if args.out_dir.exists():
        raise ArenaError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    append_ledger(
        args.out_dir,
        "preflight",
        {"contract_sha256": contract_sha, "profile": args.profile},
    )
    rows: list[dict[str, Any]] = []
    traces = []
    replays = []
    pairs = list(combinations(contract.anchors, 2))
    for cell_rows, trace, replay in play_cells(
        contract=contract,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=args.out_dir,
    ):
        rows.extend(cell_rows)
        traces.append(trace)
        replays.append(replay)
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(
        args.out_dir / "players.json",
        [player.model_dump() for player in contract.anchors],
    )
    competencies = run_competencies(
        list(contract.anchors), seeds=schedule.competency_seeds
    )
    write_json(args.out_dir / "competencies.json", competencies)
    fit, rating = rating_payload(rows, schedule, contract)
    matrix = payoff_matrix(rows)
    source_trace = next(
        trace
        for trace in traces
        if Path(trace["path"]).name == "random-v1__scripted-greedy-v1.commands.jsonl.gz"
    )
    source_games = read_trace(args.out_dir / source_trace["path"])
    profile_payload = {
        "native_gameplay": native_gameplay_profiles(
            rows, worker_count=contract.resource_caps.outcome_workers
        ),
        "matched_root": profile_players(
            list(contract.anchors),
            source_games=source_games,
            profile_roots=contract.profile_roots,
        ),
    }
    profile_payload["resource_caps"] = resource_cap_receipt(
        args.out_dir, contract, started=started
    )
    promotion = promotion_payload(
        None,
        rows,
        smoke=profile.disposition != "production",
        profile=profile_payload,
        competencies=competencies,
        contract=contract,
    )
    replay_payload = {
        "passed": all(item["passed"] for item in replays),
        "cells": replays,
    }
    write_json(args.out_dir / "replay.json", replay_payload)
    write_json(args.out_dir / "rating.json", rating)
    write_json(args.out_dir / "payoff-matrix.json", matrix)
    write_json(args.out_dir / "profile.json", profile_payload)
    write_json(args.out_dir / "promotion.json", promotion)
    (args.out_dir / "report.md").write_text(
        report_markdown(
            fit=fit, promotion=promotion, matrix=matrix, replay=replay_payload
        )
    )
    append_ledger(args.out_dir, "complete", {"games": len(rows), "cells": len(traces)})
    names = [
        "resource-ledger.jsonl",
        "players.json",
        "matches.jsonl",
        "replay.json",
        "competencies.json",
        "profile.json",
        "rating.json",
        "payoff-matrix.json",
        "promotion.json",
        "report.md",
    ]
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "anchor-freeze",
            "profile": args.profile,
            "contract_sha256": contract_sha,
            "arena_key": contract.key.model_dump(),
            "runtime": runtime,
            "traces": portable_trace_receipts(traces),
            "artifacts": artifact_receipts(args.out_dir, names),
        },
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "manifest": str(args.out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "disposition": promotion["disposition"],
            },
            sort_keys=True,
        )
    )


def challenge(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    contract, contract_sha, runtime = preflight_contract(args.contract)
    retained_checkpoint = None
    candidate_raw = PlayerRegistration.model_validate(load_json(args.candidate))
    if args.verify and candidate_raw.runner_kind == "checkpoint":
        retained_checkpoint = (
            args.out_dir / "checkpoints" / f"{candidate_raw.player_id}.pt"
        )
    checkpoint_argument = args.candidate_checkpoint or retained_checkpoint
    candidate, checkpoint_paths = preflight_candidate(
        args.candidate,
        checkpoint_argument,
        contract=contract,
        profile_name=args.profile,
        validate_checkpoint_load=not args.verify,
    )
    if args.anchor_artifact.name != "manifest.json":
        raise ArenaError("--anchor-artifact must name the frozen manifest.json")
    anchor_dir = args.anchor_artifact.parent
    anchor_verification = verify_manifest(anchor_dir, contract, contract_sha)
    anchor_manifest = load_json(args.anchor_artifact)
    if anchor_manifest["kind"] != "anchor-freeze":
        raise ArenaError("challenge requires an anchor-freeze artifact")
    if anchor_manifest["profile"] != args.profile:
        raise ArenaError("challenge and anchor artifact profiles differ")
    if args.verify:
        verification = verify_manifest(args.out_dir, contract, contract_sha)
        challenge_manifest = load_json(args.out_dir / "manifest.json")
        if (
            challenge_manifest.get("anchor_manifest_sha256")
            != anchor_verification["manifest_sha256"]
        ):
            raise ArenaError("challenge anchor binding mismatch")
        anchor_corpus = load_json(anchor_dir / "profile.json")["matched_root"][
            "root_corpus_sha256"
        ]
        challenge_corpus = load_json(args.out_dir / "profile.json")["matched_root"][
            "root_corpus_sha256"
        ]
        if challenge_corpus != anchor_corpus:
            raise ArenaError("challenge matched-root corpus drift")
        print(json.dumps(verification, sort_keys=True))
        return
    profile = contract.profiles[args.profile]
    schedule = contract.schedules[args.profile]
    if args.out_dir.exists():
        raise ArenaError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    if candidate.runner_kind == "checkpoint":
        retained = args.out_dir / "checkpoints" / f"{candidate.player_id}.pt"
        retained.parent.mkdir(parents=True)
        shutil.copyfile(checkpoint_argument, retained)
        checkpoint_paths[candidate.player_id] = str(retained)
    append_ledger(
        args.out_dir,
        "preflight",
        {
            "contract_sha256": contract_sha,
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidate_sha256": candidate.identity_sha256,
        },
    )
    new_rows: list[dict[str, Any]] = []
    traces = []
    replays = []
    pairs = [(candidate, anchor) for anchor in contract.anchors]
    for cell_rows, trace, replay in play_cells(
        contract=contract,
        pairs=pairs,
        deal_seeds=schedule.deal_seeds,
        out_dir=args.out_dir,
        checkpoint_paths=checkpoint_paths,
    ):
        new_rows.extend(cell_rows)
        traces.append(trace)
        replays.append(replay)
    anchor_rows = read_jsonl(anchor_dir / "matches.jsonl")
    rows = anchor_rows + new_rows
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(
        args.out_dir / "players.json",
        [player.model_dump() for player in (*contract.anchors, candidate)],
    )
    anchor_competencies = load_json(anchor_dir / "competencies.json")
    candidate_competencies = run_competencies(
        [candidate], seeds=schedule.competency_seeds, checkpoint_paths=checkpoint_paths
    )
    competencies = dict(anchor_competencies)
    competencies["players"] = dict(anchor_competencies["players"])
    competencies["players"].update(candidate_competencies["players"])
    write_json(args.out_dir / "competencies.json", competencies)
    fit, rating = rating_payload(rows, schedule, contract)
    matrix = payoff_matrix(rows)
    source_trace = (
        anchor_dir / "traces" / ("random-v1__scripted-greedy-v1.commands.jsonl.gz")
    )
    profile_payload = {
        "native_gameplay": native_gameplay_profiles(
            new_rows, worker_count=contract.resource_caps.outcome_workers
        ),
        "matched_root": profile_players(
            [candidate],
            source_games=read_trace(source_trace),
            profile_roots=contract.profile_roots,
            checkpoint_paths=checkpoint_paths,
        ),
    }
    profile_payload["resource_caps"] = resource_cap_receipt(
        args.out_dir, contract, started=started
    )
    promotion = promotion_payload(
        candidate,
        new_rows,
        smoke=profile.disposition != "production",
        profile=profile_payload,
        competencies=competencies,
        contract=contract,
    )
    replay_payload = {
        "passed": all(item["passed"] for item in replays),
        "cells": replays,
        "anchor_verification": anchor_verification,
    }
    write_json(args.out_dir / "replay.json", replay_payload)
    write_json(args.out_dir / "rating.json", rating)
    write_json(args.out_dir / "payoff-matrix.json", matrix)
    write_json(args.out_dir / "profile.json", profile_payload)
    write_json(args.out_dir / "promotion.json", promotion)
    (args.out_dir / "report.md").write_text(
        report_markdown(
            fit=fit, promotion=promotion, matrix=matrix, replay=replay_payload
        )
    )
    append_ledger(
        args.out_dir,
        "complete",
        {"new_games": len(new_rows), "combined_games": len(rows)},
    )
    names = [
        "resource-ledger.jsonl",
        "players.json",
        "matches.jsonl",
        "replay.json",
        "competencies.json",
        "profile.json",
        "rating.json",
        "payoff-matrix.json",
        "promotion.json",
        "report.md",
    ]
    if candidate.runner_kind == "checkpoint":
        names.append(f"checkpoints/{candidate.player_id}.pt")
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "challenge",
            "profile": args.profile,
            "contract_sha256": contract_sha,
            "arena_key": contract.key.model_dump(),
            "runtime": runtime,
            "anchor_manifest_sha256": anchor_verification["manifest_sha256"],
            "candidate": candidate.model_dump(),
            "traces": portable_trace_receipts(traces),
            "artifacts": artifact_receipts(args.out_dir, names),
        },
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "manifest": str(args.out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "candidate_elo": fit.ratings[candidate.player_id],
                "disposition": promotion["disposition"],
                "reason": promotion.get("reason"),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-anchors")
    freeze.add_argument("--contract", type=Path, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    freeze.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    challenge_parser = subparsers.add_parser("challenge")
    challenge_parser.add_argument("--contract", type=Path, required=True)
    challenge_parser.add_argument("--anchor-artifact", type=Path, required=True)
    challenge_parser.add_argument("--candidate", type=Path, required=True)
    challenge_parser.add_argument("--candidate-checkpoint", type=Path)
    challenge_parser.add_argument("--out-dir", type=Path, required=True)
    challenge_parser.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    challenge_parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "freeze-anchors":
            freeze_anchors(args)
        else:
            challenge(args)
    except (ArenaError, ValueError, FileNotFoundError) as error:
        print(
            json.dumps(
                {
                    "state": "failed",
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
