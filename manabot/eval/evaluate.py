"""Evaluate a candidate against the demo Search-64 on the selected matchup.

`uv run python -m manabot.eval.evaluate --out-dir DIR [--candidate REG --candidate-checkpoint PT]`
plays the authored Allies/Lessons setup with both sideboards, four seat/deck legs
per deal, and retains registrations, traces, an exact replay and a manifest
under a new directory. Without a candidate it plays Random. It grants no
promotion; the result is an instrument proof, not a rating."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

from manabot.env import ObservationSpace
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey, MatchRow
from manabot.eval.rating import bootstrap_population, fit_population
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import (
    REPO_ROOT,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.sim.guidance import build_registered_player
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import SELECTED_SUITE, selected_match
from manabot.sim.registration import PlayerRegistration
from manabot.sim.search.flat import DEFAULT_ROLLOUTS_PER_WORLD, DEFAULT_SEARCH_SIMS
from manabot.sim.search.runtime import DEFAULT_MAX_PLAYOUT_STEPS
from managym import WORLD_VERSION


class EvaluationError(RuntimeError):
    """The closed arena contract or artifact failed validation."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return canonical_sha256(unsigned)


def source_digest(paths: tuple[str, ...] | list[str]) -> str:
    return source_bundle_sha256([REPO_ROOT / path for path in paths])


def evaluation_runtime_fingerprints(**kwargs) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    engine_runtime = runtime_fingerprints(**kwargs)
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
    if kind == "demo_search":
        return ["manabot/sim/search/flat.py", "manabot/sim/players.py"]
    if kind == "scripted_greedy":
        return ["manabot/sim/players.py"]
    if kind == "heuristic":
        return [str(path.relative_to(REPO_ROOT)) for path in heuristic_source_paths()]
    if kind == "determinized_puct":
        return [
            "manabot/sim/search/flat_mc.py",
            "manabot/sim/search/puct.py",
            "manabot/sim/search/branch.py",
            "managym/src/python/bindings.rs",
        ]
    return ["manabot/sim/search/flat_mc.py"]


def validate_registration_source(registration: PlayerRegistration) -> None:
    if registration.runner_kind != "code":
        return
    actual = source_digest(registration_source_paths(registration))
    if registration.source_sha256 != actual:
        raise EvaluationError(f"source drift for {registration.player_id}: {actual}")


def validate_registered_player(
    registration: PlayerRegistration,
    checkpoint_path: Path | None,
    *,
    validate_checkpoint_load: bool = True,
) -> tuple[dict[str, str], ObservationSpace | None]:
    """Validate bytes and inference once, retaining the loaded observation bounds."""
    checkpoint_paths: dict[str, str] = {}
    observation_space = None
    if registration.runner_kind == "code":
        if checkpoint_path is not None:
            raise EvaluationError("code candidate rejects --candidate-checkpoint")
        validate_registration_source(registration)
    else:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise EvaluationError("checkpoint candidate bytes are unavailable")
        if file_sha256(checkpoint_path) != registration.checkpoint_sha256:
            raise EvaluationError("checkpoint candidate SHA-256 mismatch")
        if checkpoint_path.stat().st_size != registration.checkpoint_bytes:
            raise EvaluationError("checkpoint candidate byte-size mismatch")
        checkpoint_paths[registration.player_id] = str(checkpoint_path)
        if validate_checkpoint_load:
            player, observation_space = build_registered_player(
                registration, seed=0, checkpoint_path=str(checkpoint_path)
            )
            if observation_space is None:
                raise EvaluationError("checkpoint candidate has no observation space")
            observation_schema = {
                name: {"shape": list(shape), "dtype": "float32"}
                for name, shape in sorted(observation_space.shapes.items())
            }
            if canonical_sha256(observation_schema) != (
                registration.observation_abi_sha256
            ):
                raise EvaluationError("checkpoint candidate observation-shape mismatch")
            agent = getattr(player, "agent", None)
            if agent is None:
                agent = getattr(getattr(player, "evaluator", None), "agent", None)
            parameter_count = (
                sum(parameter.numel() for parameter in agent.parameters())
                if agent is not None
                else None
            )
            if parameter_count != registration.parameter_count:
                raise EvaluationError("checkpoint candidate parameter-count mismatch")
    return checkpoint_paths, observation_space


def validate_player_compatibility(
    player: PlayerRegistration, key: EvaluationKey, runtime: dict[str, Any]
) -> None:
    if (
        player.world != key.world
        or player.content_suite != key.content_suite
        or player.information_boundary != key.viewer_boundary
        or any(
            getattr(player, name) != runtime[name]
            for name in (
                "observation_abi_sha256",
                "action_abi_sha256",
                "matchup_sha256",
            )
        )
    ):
        raise EvaluationError("candidate world/content/viewer compatibility mismatch")


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


def finalize_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = {**manifest, "manifest_sha256": manifest_digest(manifest)}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def evaluate_matchup(args: argparse.Namespace) -> None:
    """Bounded instrument proof on authored Allies/Lessons, without promotion."""
    if (
        not math.isfinite(args.game_seconds)
        or args.game_seconds <= 0
        or args.max_commands < 1
    ):
        raise EvaluationError("game limits must be finite and positive")
    if len(set(args.deal_seeds)) != len(args.deal_seeds) or any(
        not 0 <= seed < 2**64 for seed in args.deal_seeds
    ):
        raise EvaluationError("deal seeds must be unique unsigned 64-bit integers")
    if args.candidate_checkpoint and not args.candidate:
        raise EvaluationError(
            "--candidate-checkpoint requires --candidate registration"
        )
    candidate = (
        PlayerRegistration.model_validate(load_json(args.candidate))
        if args.candidate
        else None
    )
    checkpoint_paths = {}
    space = None
    if candidate is not None:
        checkpoint_paths, space = validate_registered_player(
            candidate,
            args.candidate_checkpoint.resolve() if args.candidate_checkpoint else None,
        )
    space = space or ObservationSpace()
    match = selected_match()
    runtime = evaluation_runtime_fingerprints(
        match_config=match, observation_space=space, world=WORLD_VERSION
    )
    common = dict(
        runner_kind="code",
        world=WORLD_VERSION,
        content_suite=SELECTED_SUITE,
        information_boundary="acting-viewer-history-only-v1",
        observation_abi_sha256=runtime["observation_abi_sha256"],
        action_abi_sha256=runtime["action_abi_sha256"],
        matchup_sha256=runtime["matchup_sha256"],
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    baseline = PlayerRegistration(
        **common,
        player_id="demo-search-64-v1",
        display_name="Demo Search-64",
        role="incumbent",
        compute_class_id="demo-search-cpu-s64-r4-v1",
        player_spec={
            "kind": "demo_search",
            "sims": DEFAULT_SEARCH_SIMS,
            "rollouts_per_world": DEFAULT_ROLLOUTS_PER_WORLD,
            "max_steps": DEFAULT_MAX_PLAYOUT_STEPS,
        },
        source_sha256=source_digest(
            ["manabot/sim/search/flat.py", "manabot/sim/players.py"]
        ),
    )
    candidate = (
        candidate
        if candidate is not None
        else PlayerRegistration(
            **common,
            player_id="random-v1",
            display_name="Random control",
            role="challenger",
            compute_class_id="random-cpu-v1",
            player_spec={"kind": "random"},
            source_sha256=source_digest(["manabot/sim/search/flat_mc.py"]),
        )
    )
    # No inherited INT-6 predictions, anchor population, budget or promotion rule.
    source_paths = sorted(
        {
            *[
                str(path.relative_to(REPO_ROOT))
                for path in (REPO_ROOT / "manabot").rglob("*.py")
            ],
            "managym/experience.py",
            "managym/experience_projection.py",
            "uv.lock",
        }
    )
    source_sha256 = source_digest(source_paths)
    key = EvaluationKey(
        world=WORLD_VERSION,
        content_suite=SELECTED_SUITE,
        viewer_boundary=common["information_boundary"],
        arena_version="allies-lessons-"
        + canonical_sha256({"runtime": runtime, "source": source_sha256})[:16],
        rating_model_version="seat-aware-gaussian-map-bradley-terry-v1",
        rating_prior_sha256=canonical_sha256({"prior_elo_std": 400.0}),
        anchor_cohort_sha256=canonical_sha256([baseline.model_dump()]),
        evaluation_compute_envelope_id="local-cpu-one-worker-one-thread-v1",
    )
    for player in (candidate, baseline):
        validate_player_compatibility(player, key, runtime)
    if candidate.role != "challenger":
        raise EvaluationError("candidate must have challenger role")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "arena_key": key.model_dump(),
        "runtime": runtime,
        "source_paths": source_paths,
        "source_sha256": source_sha256,
        "match_hypers": match.model_dump(),
        "observation": space.encoder.config.model_dump(),
        "deal_seeds": list(args.deal_seeds),
        "legs_per_block": 4,
        "game_seconds": args.game_seconds,
        "max_commands": args.max_commands,
        "checkpoint_rng": "Torch CPU reseeded by (player_seed + player_decision_ordinal) modulo 2**63",
        "baseline": baseline.model_dump(),
        "candidate": candidate.model_dump(),
        "baseline_selection": "Shipped default Search-64; no configured trained bundle substituted",
        "failure_rule": "Player crashes forfeit; total-game limits and unattributed failures reject statistics; terminal draws score 0.5",
        "disposition": "engineering instrument only; no strength or w3 certification claim",
    }
    write_json(args.out_dir / "protocol.json", protocol)
    write_json(
        args.out_dir / "players.json", [candidate.model_dump(), baseline.model_dump()]
    )
    append_ledger(
        args.out_dir,
        "start",
        {"protocol_sha256": file_sha256(args.out_dir / "protocol.json")},
    )
    rows, trace, replay = play_cell(
        key=key,
        player_a=candidate,
        player_b=baseline,
        deal_seeds=tuple(args.deal_seeds),
        out_dir=args.out_dir,
        checkpoint_paths=checkpoint_paths,
        game_seconds=args.game_seconds,
        max_commands=args.max_commands,
    )
    for row in rows:
        MatchRow.model_validate(row)
    write_jsonl(args.out_dir / "matches.jsonl", rows)
    write_json(args.out_dir / "replay.json", replay)
    failures = sum(row["failure"] is not None for row in rows)
    source_stable = source_digest(source_paths) == protocol["source_sha256"]
    # Ratings remain in the existing statistical consumer. An unscored failure
    # invalidates the cohort instead of disappearing from its denominator.
    statistics = {"admitted": False, "failures": failures}
    if source_stable and all(
        row["score_a"] is not None and row["replay_passed"] for row in rows
    ):
        fit = fit_population(rows, anchor=baseline.player_id)
        statistics.update(
            admitted=True,
            scores=[row["score_a"] for row in rows],
            ratings=fit.ratings,
            seat0_elo=fit.seat0_elo,
            bootstrap=bootstrap_population(
                rows, anchor=baseline.player_id, seed=83, replicates=100
            ),
        )
    write_json(args.out_dir / "rating.json", statistics)
    append_ledger(args.out_dir, "complete", {"games": len(rows), "failures": failures})
    manifest = finalize_manifest(
        args.out_dir,
        {
            "schema_version": 1,
            "kind": "selected-match-instrument",
            "arena_key": key.model_dump(),
            "runtime": runtime,
            "source_stable": source_stable,
            "promotion_eligible": False,
            "admission_eligible": False,
            "traces": portable_trace_receipts([trace]),
            "artifacts": artifact_receipts(
                args.out_dir,
                [
                    "protocol.json",
                    "players.json",
                    "matches.jsonl",
                    "replay.json",
                    "rating.json",
                    "resource-ledger.jsonl",
                ],
            ),
        },
    )
    print(
        json.dumps(
            {
                "games": len(rows),
                "failures": failures,
                "replay_passed": replay["passed"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    )
    if failures or not replay["passed"] or not source_stable:
        raise EvaluationError(
            "instrument retained failed attempts; inspect matches and replay"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="PlayerRegistration JSON")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--deal-seeds", type=int, nargs="+", default=[83001])
    parser.add_argument("--game-seconds", type=float, default=120.0)
    parser.add_argument("--max-commands", type=int, default=10000)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evaluate_matchup(args)
    except (EvaluationError, ValueError, FileNotFoundError) as error:
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
