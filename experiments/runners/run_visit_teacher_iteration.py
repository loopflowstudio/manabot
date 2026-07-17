"""Run the resumable INT-4 visit teacher, student, arena, and Study iteration.

The checked-in ``iteration`` profile is the preregistered experiment. The
``smoke`` profile exercises the same artifact boundaries at bounded scale and
is explicitly non-admission evidence.

Usage:
    uv run experiments/runners/run_visit_teacher_iteration.py \
      --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
      --out-dir .runs/int-4-visit-teacher-iteration-v1

Engineering evidence:
    uv run experiments/runners/run_visit_teacher_iteration.py \
      --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
      --profile smoke --out-dir .runs/int-4-visit-teacher-smoke-v1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch

from etude.study_protocol import StudyArtifact
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import (
    OBS_KEYS,
    ROOT_VALUE_KEY,
    SCORE_KEY,
    VISIT_COUNT_KEY,
    generate_selfplay_shard,
    load_shards,
    save_bc_checkpoint,
)
from manabot.sim.flat_mc import (
    aggregate_records,
    load_checkpoint_agent,
    play_games,
)
from manabot.sim.search_supervised import (
    CHOSEN_ACTION_TARGET,
    ROOT_VALUE_TARGET,
    VISIT_DISTRIBUTION_TARGET,
    train_search_supervised,
)
from manabot.sim.study_evidence import (
    build_study_artifact,
    select_evidence_complete_decision,
)
from manabot.sim.teacher1_evidence import (
    ContractError,
    canonical_sha256,
    evaluate_root_stability,
    file_sha256,
    receipt_dict,
    record_teacher_trajectories,
    replay_teacher_trajectories,
    runtime_fingerprints,
    source_bundle_sha256,
    validate_runtime_fingerprints,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "int-4-visit-teacher-iteration-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ContractError(f"required JSON does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return payload


def _iteration_runtime(seed: int) -> dict[str, Any]:
    runtime = runtime_fingerprints(seed=seed)
    sources = [
        Path(__file__).resolve(),
        REPO_ROOT / "manabot" / "sim" / "study_evidence.py",
        REPO_ROOT / "manabot" / "sim" / "search_supervised.py",
        REPO_ROOT / "manabot" / "sim" / "distill.py",
        REPO_ROOT / "etude" / "study_protocol.py",
        REPO_ROOT / "protocol" / "study-v1.schema.json",
        REPO_ROOT / "managym" / "src" / "bin" / "validate_study_artifact.rs",
    ]
    runtime.update(
        iteration_source_sha256=source_bundle_sha256(sources),
        study_schema_sha256=file_sha256(REPO_ROOT / "protocol/study-v1.schema.json"),
        python=platform.python_version(),
        torch=torch.__version__,
        machine=platform.machine(),
        mps_available=bool(torch.backends.mps.is_available()),
    )
    return runtime


def _load_contract(
    path: Path, profile_name: str
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    contract = _load_json(path)
    if contract.get("schema_version") != 1 or contract.get("experiment") != EXPERIMENT:
        raise ContractError("unexpected INT-4 contract identity")
    profiles = contract.get("profiles") or {}
    if profile_name not in profiles:
        raise ContractError(f"contract has no profile {profile_name!r}")
    profile = profiles[profile_name]
    budgets = [int(value) for value in profile.get("teacher_budgets") or []]
    worlds = int(profile.get("worlds", 0))
    if not budgets or worlds < 1 or any(budget < worlds for budget in budgets):
        raise ContractError("every teacher budget must cover every sampled world")
    if int(profile.get("high_budget", 0)) != max(budgets):
        raise ContractError("high_budget must be the largest teacher budget")
    if int(profile.get("dataset_games", 0)) < int(profile.get("audit_games", 0)):
        raise ContractError("dataset_games must cover the replay audit games")
    seeds = [int(value) for value in profile.get("training_seeds") or []]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ContractError("training seeds must be nonempty and unique")
    return (
        contract,
        canonical_sha256(contract),
        profile,
        canonical_sha256({"name": profile_name, "profile": profile}),
    )


def _teacher_spec(profile: dict[str, Any], budget: int) -> dict[str, Any]:
    return {
        "kind": "determinized_puct",
        "name": f"t1-{budget}-w{profile['worlds']}",
        "sims": budget,
        "worlds": int(profile["worlds"]),
        "c_puct": float(profile["c_puct"]),
        "max_steps": int(profile["max_steps"]),
    }


def _search_stats(stats: Any) -> dict[str, Any] | None:
    if stats is None:
        return None
    seconds = np.asarray(stats.decision_seconds, dtype=np.float64)
    payload = stats.to_dict()
    payload.update(
        p50_decision_ms=(
            float(np.quantile(seconds, 0.50)) * 1000 if len(seconds) else None
        ),
        p95_decision_ms=(
            float(np.quantile(seconds, 0.95)) * 1000 if len(seconds) else None
        ),
        labels_per_second=(stats.decisions / stats.seconds if stats.seconds else None),
        traversals_per_second=(
            stats.simulations / stats.seconds if stats.seconds else None
        ),
        cap_rate=(stats.cap_hits / stats.simulations if stats.simulations else 0.0),
    )
    return payload


def _play_cell(
    hero: dict[str, Any],
    villain: dict[str, Any],
    *,
    blocks: list[dict[str, Any]],
    seed_offset: int = 0,
) -> dict[str, Any]:
    results = []
    for block in blocks:
        result = play_games(
            hero,
            villain,
            num_games=int(block["games"]),
            seed=int(block["seed"]) + seed_offset,
        )
        results.append((block, result))
    records = [record for _, result in results for record in result.records]
    return {
        **aggregate_records(records),
        "hero": results[0][1].hero,
        "villain": results[0][1].villain,
        "wall_seconds": sum(result.wall_seconds for _, result in results),
        "blocks": {
            str(block["id"]): {
                **aggregate_records(result.records),
                "seed": int(block["seed"]) + seed_offset,
                "hero_search": _search_stats(result.hero_search),
                "villain_search": _search_stats(result.villain_search),
            }
            for block, result in results
        },
        "hero_search": _merge_search_stats(
            [result.hero_search for _, result in results]
        ),
        "villain_search": _merge_search_stats(
            [result.villain_search for _, result in results]
        ),
    }


def _merge_search_stats(stats: list[Any]) -> dict[str, Any] | None:
    stats = [item for item in stats if item is not None]
    if not stats:
        return None
    decision_seconds = np.asarray(
        [value for item in stats for value in item.decision_seconds], dtype=np.float64
    )
    decisions = sum(int(item.decisions) for item in stats)
    seconds = sum(float(item.seconds) for item in stats)
    simulations = sum(int(item.simulations) for item in stats)
    cap_hits = sum(int(item.cap_hits) for item in stats)
    payload: dict[str, Any] = {
        "decisions": decisions,
        "seconds": seconds,
        "simulations": simulations,
        "cap_hits": cap_hits,
        "p50_decision_ms": float(np.quantile(decision_seconds, 0.50)) * 1000,
        "p95_decision_ms": float(np.quantile(decision_seconds, 0.95)) * 1000,
        "labels_per_second": decisions / seconds if seconds else None,
        "traversals_per_second": simulations / seconds if seconds else None,
        "cap_rate": cap_hits / simulations if simulations else 0.0,
    }
    if hasattr(stats[0], "tree_nodes"):
        payload.update(
            tree_nodes=sum(int(item.tree_nodes) for item in stats),
            worlds_sampled=sum(int(item.worlds_sampled) for item in stats),
            mean_max_depth=(
                sum(int(item.max_depth_sum) for item in stats) / decisions
                if decisions
                else 0.0
            ),
            max_depth_max=max(int(item.max_depth_max) for item in stats),
        )
    return payload


def _run_teacher(profile: dict[str, Any]) -> dict[str, Any]:
    cells = {}
    for budget in profile["teacher_budgets"]:
        cells[str(budget)] = _play_cell(
            _teacher_spec(profile, int(budget)),
            {"kind": "random"},
            blocks=list(profile["teacher_seed_blocks"]),
        )
    stability = evaluate_root_stability(
        budgets=[int(value) for value in profile["teacher_budgets"]],
        worlds=int(profile["worlds"]),
        c_puct=float(profile["c_puct"]),
        roots=int(profile["stability_roots"]),
        repeat_seeds=[int(value) for value in profile["stability_repeat_seeds"]],
        seed=int(profile["stability_seed"]),
        max_steps=int(profile["max_steps"]),
    )
    return {"cells_vs_random": cells, "root_stability": stability}


def _dataset_diagnostics(dataset: dict[str, np.ndarray], budget: int) -> dict[str, Any]:
    visits = np.asarray(dataset[VISIT_COUNT_KEY])
    valid = np.asarray(dataset["actions_valid"]) > 0
    root_values = np.asarray(dataset[ROOT_VALUE_KEY])
    scores = np.asarray(dataset[SCORE_KEY])
    checks = {
        "finite_visits": bool(np.isfinite(visits).all()),
        "finite_q_values": bool(np.isfinite(scores).all()),
        "finite_root_values": bool(np.isfinite(root_values).all()),
        "visit_mass": bool(np.all(visits.sum(axis=1) == budget)),
        "visits_within_legal_mask": bool(np.all(visits[~valid] == 0)),
        "root_values_in_range": bool(
            np.all((root_values >= 0.0) & (root_values <= 1.0))
        ),
        "terminal_outcomes_present": bool(np.all(dataset["winner"] >= 0)),
    }
    return {
        "games": int(len(np.unique(dataset["game_index"]))),
        "decisions": int(len(dataset["action"])),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_dataset(
    out_dir: Path,
    profile: dict[str, Any],
    *,
    profile_hash: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    progress_path = dataset_dir / "manifest.json"
    progress = (
        _load_json(progress_path)
        if progress_path.exists()
        else {
            "profile_sha256": profile_hash,
            "status": "running",
            "shards": [],
        }
    )
    if progress.get("profile_sha256") != profile_hash:
        raise ContractError("existing dataset belongs to a different profile")
    for item in progress["shards"]:
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ContractError(f"completed dataset shard drifted: {path}")

    games = int(profile["dataset_games"])
    per_shard = int(profile["games_per_shard"])
    teacher = _teacher_spec(profile, int(profile["high_budget"]))
    completed = {int(item["game_offset"]) for item in progress["shards"]}
    for game_offset in range(0, games, per_shard):
        if game_offset in completed:
            continue
        shard_games = min(per_shard, games - game_offset)
        shard = dataset_dir / f"shard_{game_offset // per_shard:03d}.npz"
        if shard.exists():
            raise ContractError(f"unregistered dataset shard already exists: {shard}")
        summary = generate_selfplay_shard(
            num_games=shard_games,
            teacher_spec=teacher,
            seed=int(profile["dataset_seed"]),
            game_offset=game_offset,
            out_path=shard,
            dataset_run_fingerprint=profile_hash,
        )
        progress["shards"].append(
            {
                "game_offset": game_offset,
                "games": shard_games,
                "path": str(shard),
                "sha256": file_sha256(shard),
                "bytes": shard.stat().st_size,
                "summary": summary,
            }
        )
        _atomic_json(progress_path, progress)
    progress["status"] = "completed"
    _atomic_json(progress_path, progress)

    shard_paths = [Path(item["path"]) for item in progress["shards"]]
    dataset = load_shards(shard_paths)
    diagnostics = _dataset_diagnostics(dataset, int(profile["high_budget"]))
    if not diagnostics["passed"]:
        raise RuntimeError("visit dataset failed integrity checks")

    audit_path = out_dir / "trajectory-audit.json"
    if audit_path.exists():
        audit = _load_json(audit_path)
    else:
        audit = record_teacher_trajectories(
            games=int(profile["audit_games"]),
            simulations=int(profile["high_budget"]),
            worlds=int(profile["worlds"]),
            c_puct=float(profile["c_puct"]),
            seed=int(profile["dataset_seed"]),
            content_hash=runtime["experience_content_hash"],
            asset_manifest_hash=runtime["asset_manifest_hash"],
            max_steps=int(profile["max_steps"]),
            provenance={"profile_sha256": profile_hash},
        )
        _atomic_json(audit_path, audit)
    replay = replay_teacher_trajectories(
        audit,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=list(profile["sampled_search_roots"]),
    )
    if not replay.passed:
        raise RuntimeError("trajectory or sampled-search replay failed")

    audit_decisions = [
        decision for game in audit["games"] for decision in game["decisions"]
    ]
    mismatches = 0
    for row, decision in enumerate(audit_decisions):
        search = decision["search"]
        legal = len(decision["frame"]["offers"])
        mismatches += int(
            int(dataset["action"][row]) != decision["command"]["offer_id"]
        )
        mismatches += int(
            not np.array_equal(
                dataset[VISIT_COUNT_KEY][row, :legal].astype(np.int64),
                np.asarray(search["visit_counts"], dtype=np.int64),
            )
        )
        mismatches += int(
            not np.array_equal(
                dataset[SCORE_KEY][row, :legal].astype(np.float32),
                np.asarray(search["q_values"], dtype=np.float32),
            )
        )
        mismatches += int(
            np.float32(dataset[ROOT_VALUE_KEY][row]) != np.float32(search["root_value"])
        )
    if mismatches:
        raise RuntimeError("learner shard does not align with its replay audit")
    return {
        "manifest_path": str(progress_path),
        "manifest_sha256": file_sha256(progress_path),
        "shards": progress["shards"],
        "diagnostics": diagnostics,
        "audit_path": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "replay": receipt_dict(replay),
        "learner_audit_mismatches": mismatches,
    }


ARMS = {
    "chosen_policy_only": (CHOSEN_ACTION_TARGET, 0.0),
    "chosen_policy_value": (CHOSEN_ACTION_TARGET, 1.0),
    "visit_policy_only": (VISIT_DISTRIBUTION_TARGET, 0.0),
    "visit_policy_value": (VISIT_DISTRIBUTION_TARGET, 1.0),
}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        array = value.detach().cpu().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _initialization_sha256(seed: int) -> str:
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(), AgentHypers())
    return _state_sha256(agent.state_dict())


def _run_training(
    out_dir: Path,
    profile: dict[str, Any],
    dataset_stage: dict[str, Any],
    *,
    profile_hash: str,
) -> dict[str, Any]:
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    progress_path = training_dir / "manifest.json"
    progress = (
        _load_json(progress_path)
        if progress_path.exists()
        else {
            "profile_sha256": profile_hash,
            "dataset_manifest_sha256": dataset_stage["manifest_sha256"],
            "status": "running",
            "checkpoints": {},
        }
    )
    if (
        progress.get("profile_sha256") != profile_hash
        or progress.get("dataset_manifest_sha256") != dataset_stage["manifest_sha256"]
    ):
        raise ContractError("existing training artifacts have different inputs")
    for item in progress["checkpoints"].values():
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ContractError(f"completed checkpoint drifted: {path}")

    dataset = load_shards([Path(item["path"]) for item in dataset_stage["shards"]])
    for seed in profile["training_seeds"]:
        seed = int(seed)
        initialization = _initialization_sha256(seed)
        for arm, (policy_target, value_weight) in ARMS.items():
            key = f"{seed}:{arm}"
            if key in progress["checkpoints"]:
                continue
            started = time.perf_counter()
            agent, obs_space, initial, history = train_search_supervised(
                dataset,
                policy_target_kind=policy_target,
                value_target_kind=ROOT_VALUE_TARGET,
                policy_weight=1.0,
                value_weight=value_weight,
                lr=float(profile["learning_rate"]),
                epochs=int(profile["epochs"]),
                batch_size=int(profile["batch_size"]),
                val_fraction=float(profile["val_fraction"]),
                seed=seed,
                device=str(profile["training_device"]),
            )
            temporary = training_dir / f"{arm}-seed-{seed}.pt"
            save_bc_checkpoint(
                agent,
                obs_space,
                temporary,
                extra={
                    "experiment": EXPERIMENT,
                    "profile_sha256": profile_hash,
                    "arm": arm,
                    "seed": seed,
                    "initialization_sha256": initialization,
                    "policy_target_kind": policy_target,
                    "value_target_kind": ROOT_VALUE_TARGET,
                    "value_weight": value_weight,
                    "dataset_manifest_sha256": dataset_stage["manifest_sha256"],
                },
            )
            checkpoint_hash = file_sha256(temporary)
            checkpoint = training_dir / (f"{arm}-seed-{seed}-{checkpoint_hash[:16]}.pt")
            temporary.replace(checkpoint)
            progress["checkpoints"][key] = {
                "arm": arm,
                "seed": seed,
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "bytes": checkpoint.stat().st_size,
                "initialization_sha256": initialization,
                "policy_target_kind": policy_target,
                "value_target_kind": ROOT_VALUE_TARGET,
                "value_weight": value_weight,
                "seconds": time.perf_counter() - started,
                "initial_validation": asdict(initial),
                "history": [asdict(epoch) for epoch in history],
            }
            _atomic_json(progress_path, progress)
    progress["status"] = "completed"
    _atomic_json(progress_path, progress)
    return {
        "manifest_path": str(progress_path),
        "manifest_sha256": file_sha256(progress_path),
        "checkpoints": progress["checkpoints"],
    }


def _checkpoint(training: dict[str, Any], seed: int, arm: str) -> dict[str, Any]:
    return training["checkpoints"][f"{seed}:{arm}"]


def _run_arena(
    profile_name: str,
    profile: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    per_seed = {}
    for seed_value in profile["training_seeds"]:
        seed = int(seed_value)
        policy = _checkpoint(training, seed, "visit_policy_only")
        student = _checkpoint(training, seed, "visit_policy_value")
        players = {
            "teacher": _teacher_spec(profile, int(profile["high_budget"])),
            "policy-only": {
                "kind": "checkpoint",
                "name": f"visit-policy-only-{seed}",
                "path": policy["path"],
                "deterministic": True,
            },
            "student": {
                "kind": "checkpoint",
                "name": f"visit-policy-value-{seed}",
                "path": student["path"],
                "deterministic": True,
            },
            "student+search": {
                "kind": "agent_puct",
                "name": f"visit-policy-value-search-{seed}",
                "checkpoint": student["path"],
                "sims": int(profile["student_search_simulations"]),
                "worlds": int(profile["worlds"]),
                "c_puct": float(profile["c_puct"]),
                "max_steps": int(profile["max_steps"]),
                "device": "cpu",
            },
        }
        cells = {}
        for left, right in itertools.combinations(players, 2):
            cells[f"{left}-vs-{right}"] = _play_cell(
                players[left],
                players[right],
                blocks=list(profile["arena_seed_blocks"]),
                seed_offset=seed * 10_000,
            )
        chosen_policy = _checkpoint(training, seed, "chosen_policy_only")
        chosen_value = _checkpoint(training, seed, "chosen_policy_value")
        ablations = {
            "visit_policy_only-vs-chosen_policy_only": _play_cell(
                players["policy-only"],
                {
                    "kind": "checkpoint",
                    "name": f"chosen-policy-only-{seed}",
                    "path": chosen_policy["path"],
                    "deterministic": True,
                },
                blocks=list(profile["ablation_seed_blocks"]),
                seed_offset=seed * 10_000,
            ),
            "visit_policy_value-vs-chosen_policy_value": _play_cell(
                players["student"],
                {
                    "kind": "checkpoint",
                    "name": f"chosen-policy-value-{seed}",
                    "path": chosen_value["path"],
                    "deterministic": True,
                },
                blocks=list(profile["ablation_seed_blocks"]),
                seed_offset=seed * 10_000,
            ),
        }
        per_seed[str(seed)] = {"round_robin": cells, "ablations": ablations}

    paired_rates = [
        float(
            per_seed[str(seed)]["round_robin"]["student-vs-student+search"]["win_rate"]
        )
        for seed in profile["training_seeds"]
    ]
    search_rates = [1.0 - rate for rate in paired_rates]
    if profile_name == "smoke":
        verdict = "revise"
        diagnosis = "engineering_smoke_only_no_admission_claim"
    elif (
        np.median(search_rates) >= 0.55
        and sum(rate > 0.5 for rate in search_rates) >= 2
    ):
        verdict = "continue"
        diagnosis = "student_guided_search_improves_the_paired_student"
    else:
        verdict = "revise"
        diagnosis = "student_guided_search_did_not_clear_the_predeclared_strength_bar"
    return {
        "per_seed": per_seed,
        "student_search_win_rates": search_rates,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "limitation": (
            "This vertical slice does not yet execute the frozen Teacher-0 "
            "incumbent or competency cells."
        ),
    }


def _policy_mass_for_study(
    audit: dict[str, Any],
    dataset_stage: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[int, float]:
    game_index, decision_index, decision = select_evidence_complete_decision(audit)
    row = decision_index
    for game in sorted(audit["games"], key=lambda item: int(item["game_index"])):
        if int(game["game_index"]) == game_index:
            break
        row += len(game["decisions"])
    dataset = load_shards([Path(item["path"]) for item in dataset_stage["shards"]])
    agent, _ = load_checkpoint_agent(checkpoint["path"])
    obs = {
        key: torch.as_tensor(dataset[key][row], dtype=torch.float32).unsqueeze(0)
        for key in OBS_KEYS
    }
    with torch.inference_mode():
        logits, _ = agent(obs)
        probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy()
    return {
        int(offer["id"]): float(probabilities[int(offer["id"])])
        for offer in decision["frame"]["offers"]
    }


def _validate_study_in_rust(path: Path) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(REPO_ROOT / "managym/Cargo.toml"),
        "--bin",
        "validate_study_artifact",
        "--",
        str(path),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rust Study validation failed: {result.stderr.strip()}")
    return {"passed": True, "validator": "managym::study::StudyArtifact::validate"}


def _run_study(
    out_dir: Path,
    contract: dict[str, Any],
    profile: dict[str, Any],
    dataset_stage: dict[str, Any],
    training: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    audit = _load_json(Path(dataset_stage["audit_path"]))
    study_seed = int(contract["study_model_seed"])
    if study_seed not in [int(value) for value in profile["training_seeds"]]:
        study_seed = int(profile["training_seeds"][0])
    checkpoint = _checkpoint(training, study_seed, "visit_policy_value")
    artifact = build_study_artifact(
        audit,
        policy_mass_by_offer=_policy_mass_for_study(audit, dataset_stage, checkpoint),
        source_replay_sha256=dataset_stage["audit_sha256"],
        checkpoint_sha256=checkpoint["sha256"],
        engine_build_sha256=runtime["engine_extension_sha256"],
        content_pack_id=str(contract["content_pack"]["id"]),
        content_pack_version=str(contract["content_pack"]["version"]),
        model_id=f"visit-policy-value-seed-{study_seed}",
        producer_version=runtime["iteration_source_sha256"],
    )
    path = out_dir / "study-artifact.json"
    _atomic_json(path, artifact.model_dump(mode="json"))
    StudyArtifact.model_validate_json(path.read_text())
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "python_validation": {"passed": True, "model": "etude.StudyArtifact"},
        "rust_validation": _validate_study_in_rust(path),
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    teacher = manifest["stages"]["teacher"]["result"]
    training = manifest["stages"]["training"]["result"]
    arena = manifest["stages"]["arena"]["result"]
    lines = [
        "# INT-4 Visit Teacher Iteration",
        "",
        f"- Profile: `{manifest['profile']}`",
        f"- Status: `{manifest['status']}`",
        f"- Verdict: `{arena['verdict']}` — {arena['diagnosis']}",
        f"- Contract SHA-256: `{manifest['contract_sha256']}`",
        f"- Runtime source SHA-256: `{manifest['runtime']['iteration_source_sha256']}`",
        "",
        "## Teacher versus random",
        "",
        "| Traversals | Games | Win rate | p50 ms | p95 ms | Labels/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, cell in teacher["cells_vs_random"].items():
        search = cell["hero_search"]
        lines.append(
            f"| {budget} | {int(cell['num_games'])} | {cell['win_rate']:.3f} | "
            f"{search['p50_decision_ms']:.2f} | {search['p95_decision_ms']:.2f} | "
            f"{search['labels_per_second']:.2f} |"
        )
    dataset = manifest["stages"]["dataset"]["result"]
    lines.extend(
        [
            "",
            "## Dataset and replay",
            "",
            f"- Games: {dataset['diagnostics']['games']}",
            f"- Decisions: {dataset['diagnostics']['decisions']}",
            f"- Exact trajectory/search replay: `{dataset['replay']['passed']}`",
            f"- Learner/audit mismatches: {dataset['learner_audit_mismatches']}",
            "",
            "## Student validation",
            "",
            "| Seed | Arm | Policy CE | Policy KL | Root-value Brier | Checkpoint |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in training["checkpoints"].values():
        final = item["history"][-1]["validation"]
        lines.append(
            f"| {item['seed']} | {item['arm']} | {final['policy_loss']:.4f} | "
            f"{final['policy_kl']:.4f} | {final['value_brier']:.4f} | "
            f"`{item['sha256'][:16]}` |"
        )
    lines.extend(["", "## Four-agent arena", ""])
    for seed, block in arena["per_seed"].items():
        lines.extend(
            [
                f"### Seed {seed}",
                "",
                "| Cell | Games | Hero win rate |",
                "|---|---:|---:|",
            ]
        )
        for cell_name, cell in block["round_robin"].items():
            lines.append(
                f"| {cell_name} | {int(cell['num_games'])} | {cell['win_rate']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            arena["limitation"],
            "The smoke profile validates plumbing and integrity only; its game-level "
            "rates are not method-level evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _verify(
    out_dir: Path,
    contract_hash: str,
    profile_hash: str,
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(out_dir / "manifest.json")
    if (
        manifest.get("contract_sha256") != contract_hash
        or manifest.get("profile_sha256") != profile_hash
        or manifest.get("runtime") != runtime
    ):
        raise ContractError("run manifest identity drifted")
    dataset = manifest["stages"]["dataset"]["result"]
    for shard in dataset["shards"]:
        if file_sha256(shard["path"]) != shard["sha256"]:
            raise RuntimeError(f"dataset shard changed: {shard['path']}")
    audit = _load_json(Path(dataset["audit_path"]))
    if file_sha256(dataset["audit_path"]) != dataset["audit_sha256"]:
        raise RuntimeError("trajectory audit changed")
    replay = replay_teacher_trajectories(
        audit,
        content_hash=runtime["experience_content_hash"],
        asset_manifest_hash=runtime["asset_manifest_hash"],
        sampled_search_roots=list(profile["sampled_search_roots"]),
    )
    if receipt_dict(replay) != dataset["replay"]:
        raise RuntimeError("exact replay receipt changed")
    training = manifest["stages"]["training"]["result"]
    for checkpoint in training["checkpoints"].values():
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint changed: {checkpoint['path']}")
        load_checkpoint_agent(checkpoint["path"])
    study = manifest["stages"]["study"]["result"]
    if file_sha256(study["path"]) != study["sha256"]:
        raise RuntimeError("Study artifact changed")
    StudyArtifact.model_validate_json(Path(study["path"]).read_text())
    _validate_study_in_rust(Path(study["path"]))
    receipt = {
        "verified": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract_hash,
        "profile_sha256": profile_hash,
        "trajectory_replay": receipt_dict(replay),
        "study_sha256": study["sha256"],
    }
    _atomic_json(out_dir / "verification.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("iteration", "smoke"), default="iteration"
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--print-runtime", action="store_true")
    args = parser.parse_args()

    contract, contract_hash, profile, profile_hash = _load_contract(
        args.contract, args.profile
    )
    runtime = _iteration_runtime(seed=int(contract["runtime_seed"]))
    if args.print_runtime:
        print(json.dumps(runtime, indent=2, sort_keys=True))
        return
    validate_runtime_fingerprints(contract["expected_fingerprints"], runtime)
    out_dir = args.out_dir.resolve()
    if args.verify:
        print(
            json.dumps(
                _verify(out_dir, contract_hash, profile_hash, profile, runtime),
                indent=2,
            )
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if (
            manifest.get("contract_sha256") != contract_hash
            or manifest.get("profile_sha256") != profile_hash
            or manifest.get("runtime") != runtime
        ):
            raise ContractError("existing run belongs to different frozen inputs")
    else:
        manifest = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "profile": args.profile,
            "evidence_class": profile["evidence_class"],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "contract_path": str(args.contract),
            "contract_sha256": contract_hash,
            "profile_sha256": profile_hash,
            "runtime": runtime,
            "stages": {},
        }
        _atomic_json(manifest_path, manifest)

    def run_stage(name: str, inputs: dict[str, Any], function: Any) -> dict[str, Any]:
        input_hash = canonical_sha256(inputs)
        existing = manifest["stages"].get(name)
        if existing and existing.get("status") == "completed":
            if existing.get("input_sha256") != input_hash:
                raise ContractError(f"completed stage {name} has different inputs")
            return existing["result"]
        manifest["stages"][name] = {
            "status": "running",
            "input_sha256": input_hash,
            "started_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(manifest_path, manifest)
        started = time.perf_counter()
        result = function()
        manifest["stages"][name] = {
            "status": "completed",
            "input_sha256": input_hash,
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "result": result,
        }
        _atomic_json(manifest_path, manifest)
        return result

    teacher = run_stage(
        "teacher",
        {"profile": profile_hash, "runtime": runtime},
        lambda: _run_teacher(profile),
    )
    dataset = run_stage(
        "dataset",
        {
            "profile": profile_hash,
            "runtime": runtime,
            "teacher": canonical_sha256(teacher),
        },
        lambda: _run_dataset(
            out_dir, profile, profile_hash=profile_hash, runtime=runtime
        ),
    )
    training = run_stage(
        "training",
        {"profile": profile_hash, "dataset": dataset["manifest_sha256"]},
        lambda: _run_training(out_dir, profile, dataset, profile_hash=profile_hash),
    )
    arena = run_stage(
        "arena",
        {"profile": profile_hash, "training": training["manifest_sha256"]},
        lambda: _run_arena(args.profile, profile, training),
    )
    study = run_stage(
        "study",
        {
            "profile": profile_hash,
            "audit": dataset["audit_sha256"],
            "training": training["manifest_sha256"],
        },
        lambda: _run_study(out_dir, contract, profile, dataset, training, runtime),
    )
    manifest["status"] = "completed"
    manifest["verdict"] = arena["verdict"]
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["study_sha256"] = study["sha256"]
    _atomic_json(manifest_path, manifest)
    _write_report(out_dir / "report.md", manifest)
    print(
        f"{EXPERIMENT} {args.profile}: completed with {arena['verdict']} -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
