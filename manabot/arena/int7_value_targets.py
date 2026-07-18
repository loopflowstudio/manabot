"""Additive INT-7 value-target experiment mechanics.

This module deliberately sits outside the frozen INT-6 and INT-8 arena source
bundles. It reuses their match, replay, rating, competency, and root-selection
authorities while giving newly trained checkpoints an explicit learned or
neutral value-leaf identity.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import resource
import threading
import time
from typing import Any, Literal

import numpy as np
import psutil
from pydantic import Field, model_validator
import torch

from manabot.env import ObservationSpace
from manabot.sim.distill import OBS_KEYS, ROOT_VALUE_KEY
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.mcts import AgentLeafEvaluator, DeterminizedPuctPlayer
from manabot.sim.search_supervised import outcome_targets
from manabot.sim.teacher1_evidence import source_bundle_sha256
from manabot.verify.competency import (
    SCENARIOS,
    aggregate_scenario_results,
    run_scenario_once,
)

from .guidance import build_arena_player as build_frozen_arena_player
from .models import (
    PlayerRegistration,
    ProfileRoots,
    SearchSemantics,
    StrictModel,
    canonical_sha256,
    file_sha256,
)
from .profile import (
    _root_mechanism_metrics,
    _summarize_isolated,
    select_profile_roots,
    verify_profile,
)
from .replay import replay_prefix

EVIDENCE_CLASS = "engineering_smoke_only_no_admission_claim"
MODEL_SEEDS = (197, 198, 199)
ARM_ORDER = (
    "visit_policy_only",
    "visit_terminal",
    "visit_blend_50_50",
    "visit_teacher_root",
)
PHASE_NAMES = (
    "beginning",
    "precombat_main",
    "combat",
    "postcombat_main",
    "ending",
)
RESOURCE_CAPS = {
    "wall_hours": 6.0,
    "core_hours": 24.0,
    "workers": 4,
    "artifact_bytes": 2147483648,
}


class Int7PlayerRegistration(StrictModel):
    """Closed checkpoint-PUCT identity used only by the additive experiment."""

    player_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    display_name: str
    role: Literal["challenger"]
    runner_kind: Literal["checkpoint"]
    player_spec: dict[str, Any]
    compute_class_id: Literal["checkpoint-puct-cpu-s32-w4-v1"]
    information_boundary: Literal["acting-viewer-history-only-v1"]
    world: Literal["w2"]
    content_suite: Literal["w2-interactive-mirror-v1"]
    observation_abi_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_abi_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matchup_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_bytes: int = Field(gt=0)
    parameter_count: int = Field(gt=0)
    training_seed: int
    artifact_id: str
    evidence_class: Literal["fixture"] = "fixture"
    player_seed_derivation_id: Literal["arena-comparison-alias-player-v1"]
    search_call_seed_derivation_id: Literal["mcts-mix-comparison-seed-decision-v1"]
    search_semantics: SearchSemantics
    arm: Literal[
        "visit_policy_only",
        "visit_terminal",
        "visit_teacher_root",
        "visit_blend_50_50",
    ]
    value_mode: Literal["neutral", "learned"]
    profile_only: bool = False

    @model_validator(mode="after")
    def validate_frozen_compute(self) -> "Int7PlayerRegistration":
        required = {
            "kind",
            "sims",
            "worlds",
            "c_puct",
            "max_steps",
            "branch_driver_id",
            "device",
            "batch_size",
            "deterministic",
            "root_noise",
            "implementation_source_sha256",
            "value_mode",
        }
        if set(self.player_spec) != required:
            raise ValueError("INT-7 checkpoint PUCT spec must be fully explicit")
        expected = {
            "kind": "int7_checkpoint_puct",
            "sims": 32,
            "worlds": 4,
            "c_puct": 1.5,
            "max_steps": 2000,
            "branch_driver_id": "full_clone/current_game_v1",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": True,
            "root_noise": "none",
            "implementation_source_sha256": int7_player_source_sha256(),
            "value_mode": self.value_mode,
        }
        if self.player_spec != expected:
            raise ValueError("INT-7 checkpoint PUCT parameters drifted")
        expected_leaf = {
            "learned": "checkpoint-sigmoid-value-v1",
            "neutral": "neutral-0.5-after-checkpoint-forward-v1",
        }[self.value_mode]
        if self.search_semantics.model_dump() != {
            "branch_audit": False,
            "root_prior": "checkpoint-policy-softmax-v1",
            "leaf_evaluator": expected_leaf,
        }:
            raise ValueError("INT-7 search semantics drifted")
        if self.arm == "visit_policy_only" and self.value_mode != "neutral":
            raise ValueError("policy-only must serve a neutral value")
        if (
            not self.profile_only
            and self.arm != "visit_policy_only"
            and (self.value_mode != "learned")
        ):
            raise ValueError("joint gameplay candidates must serve learned values")
        return self

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.model_dump())


def int7_player_source_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    return source_bundle_sha256(
        [root / "manabot/arena/int7_value_targets.py", root / "manabot/sim/mcts.py"]
    )


def build_int7_player(
    registration: Any, *, seed: int, checkpoint_path: str | None = None
) -> tuple[Any, ObservationSpace | None]:
    if not isinstance(registration, Int7PlayerRegistration):
        return build_frozen_arena_player(
            registration, seed=seed, checkpoint_path=checkpoint_path
        )
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise FileNotFoundError("INT-7 checkpoint candidate bytes are unavailable")
    if file_sha256(checkpoint_path) != registration.checkpoint_sha256:
        raise RuntimeError("INT-7 checkpoint SHA-256 drifted")
    if Path(checkpoint_path).stat().st_size != registration.checkpoint_bytes:
        raise RuntimeError("INT-7 checkpoint byte size drifted")
    agent, observation_space = load_checkpoint_agent(checkpoint_path)
    parameter_count = sum(parameter.numel() for parameter in agent.parameters())
    if parameter_count != registration.parameter_count:
        raise RuntimeError("INT-7 checkpoint parameter count drifted")
    evaluator = AgentLeafEvaluator(
        agent, observation_space, value_mode=registration.value_mode
    )
    spec = registration.player_spec
    player = DeterminizedPuctPlayer(
        int(spec["sims"]),
        worlds=int(spec["worlds"]),
        c_puct=float(spec["c_puct"]),
        max_steps=int(spec["max_steps"]),
        seed=seed,
        evaluator=evaluator,
        branch_driver_id=str(spec["branch_driver_id"]),
        branch_audit=False,
    )
    return player, observation_space


def _play_cell_job(
    job: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import manabot.arena.match as match_module

    original = match_module.build_arena_player
    match_module.build_arena_player = build_int7_player
    try:
        return match_module.play_cell(**job)
    finally:
        match_module.build_arena_player = original


def play_int7_cells(
    *,
    key: Any,
    pairs: list[tuple[Any, Any]],
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str],
    comparison_seed_aliases: dict[str, str],
    workers: int = 4,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    jobs = [
        {
            "key": key,
            "player_a": first,
            "player_b": second,
            "deal_seeds": deal_seeds,
            "out_dir": out_dir,
            "checkpoint_paths": checkpoint_paths,
            "comparison_seed_aliases": comparison_seed_aliases,
        }
        for first, second in pairs
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_play_cell_job, jobs))


def _competency_seed(alias: str, scenario: str, run_seed: int) -> int:
    return int(canonical_sha256([alias, scenario, run_seed])[:16], 16)


def run_int7_competencies(
    registrations: list[Int7PlayerRegistration],
    *,
    seeds: tuple[int, ...],
    checkpoint_paths: dict[str, str],
    comparison_seed_aliases: dict[str, str],
) -> dict[str, Any]:
    players: dict[str, Any] = {}
    for registration in registrations:
        scenario_rows: dict[str, Any] = {}
        for scenario_name, scenario in SCENARIOS.items():
            runs = []
            for run_seed in seeds:
                seed = _competency_seed(
                    comparison_seed_aliases[registration.player_id],
                    scenario_name,
                    run_seed,
                )
                player, observation_space = build_int7_player(
                    registration,
                    seed=seed,
                    checkpoint_path=checkpoint_paths[registration.player_id],
                )
                result = run_scenario_once(
                    scenario, player, observation_space, seed=seed
                )
                runs.append({"run_seed": run_seed, "player_seed": seed, **result})
            scenario_rows[scenario_name] = {
                "correct_line": scenario.correct_line,
                "runs": runs,
                "aggregate": aggregate_scenario_results(runs),
            }
        players[registration.player_id] = scenario_rows
    return {
        "schema_version": 1,
        "scenario_authority": "manabot.verify.competency.SCENARIOS",
        "scenario_seeds": list(seeds),
        "scenario_seed_set_sha256": canonical_sha256(list(seeds)),
        "players": players,
    }


def _ru_maxrss_bytes() -> int:
    import platform

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _profile_worker(
    connection: Any,
    registration_payload: dict[str, Any],
    is_int7: bool,
    roots: list[dict[str, Any]],
    warmup: int,
    checkpoint_path: str | None,
    sampler_interval_ms: int,
    comparison_seed_alias: str,
) -> None:
    try:
        torch.set_num_threads(1)
        registration = (
            Int7PlayerRegistration.model_validate(registration_payload)
            if is_int7
            else PlayerRegistration.model_validate(registration_payload)
        )
        process = psutil.Process()
        baseline_rss = int(process.memory_info().rss)
        player_seed = int(
            canonical_sha256([comparison_seed_alias, "matched-root-profile-v1"])[:16],
            16,
        )
        player, observation_space = build_int7_player(
            registration, seed=player_seed, checkpoint_path=checkpoint_path
        )
        post_load_rss = int(process.memory_info().rss)
        peak_rss = max(baseline_rss, post_load_rss)
        stop = threading.Event()

        def sample_rss() -> None:
            nonlocal peak_rss
            while not stop.wait(sampler_interval_ms / 1000.0):
                peak_rss = max(peak_rss, int(process.memory_info().rss))

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        samples = []
        root_mutations = 0
        illegal_actions = 0
        for root_index, root in enumerate(roots):
            env, observation = replay_prefix(
                root["game"], root["revision"], observation_space
            )
            pre_digest = env._engine.state_digest()
            stats = getattr(player, "stats", None)
            before_simulations = int(getattr(stats, "simulations", 0))
            before_cap_hits = int(getattr(stats, "cap_hits", 0))
            before_tree_nodes = int(getattr(stats, "tree_nodes", 0))
            evaluator = getattr(player, "evaluator", None)
            before_forwards = int(getattr(evaluator, "forward_calls", 0))
            legal_action_count = len(env.last_raw_obs.action_space.actions)
            started = time.perf_counter()
            action = int(player.act(env, observation))
            elapsed = time.perf_counter() - started
            simulations = int(getattr(stats, "simulations", 0)) - before_simulations
            cap_hits = int(getattr(stats, "cap_hits", 0)) - before_cap_hits
            tree_nodes = int(getattr(stats, "tree_nodes", 0)) - before_tree_nodes
            forward_calls = (
                int(getattr(evaluator, "forward_calls", 0)) - before_forwards
            )
            mutated = env._engine.state_digest() != pre_digest
            root_mutations += int(mutated)
            legal = 0 <= action < legal_action_count
            illegal_actions += int(not legal)
            if root_index >= warmup:
                sample = {
                    "root_id": root["root_id"],
                    "decision_ordinal": root_index,
                    "action_space_kind": root["game"]["decisions"][root["revision"]][
                        "action_space_kind"
                    ],
                    "legal_action_count": legal_action_count,
                    "latency_seconds": elapsed,
                    "action": action,
                    "simulations": simulations,
                    "tree_nodes": tree_nodes,
                    "forward_calls": forward_calls,
                    "cap_hits": cap_hits,
                    "root_mutated": mutated,
                    "legal": legal,
                }
                result = getattr(player, "last_result", None)
                if result is not None:
                    visits = np.asarray(result.visit_counts, dtype=np.float64)
                    priors = getattr(evaluator, "last_root_priors", None)
                    if priors is None:
                        priors = np.full(len(visits), 1.0 / len(visits))
                    sample.update(
                        _root_mechanism_metrics(
                            visits=visits,
                            input_prior=np.asarray(priors, dtype=np.float64),
                        )
                    )
                    sample["root_value"] = float(result.root_value)
                samples.append(sample)
        stop.set()
        sampler.join(timeout=1.0)
        peak_rss = max(peak_rss, int(process.memory_info().rss), _ru_maxrss_bytes())
        connection.send(
            {
                "ok": True,
                "player_id": registration.player_id,
                "player_seed": player_seed,
                "checkpoint_bytes": getattr(registration, "checkpoint_bytes", None),
                "parameter_count": getattr(registration, "parameter_count", None),
                "samples": samples,
                "root_mutations": root_mutations,
                "illegal_actions": illegal_actions,
                "baseline_rss_bytes": baseline_rss,
                "post_load_rss_bytes": post_load_rss,
                "peak_rss_bytes": peak_rss,
                "ru_maxrss_bytes": _ru_maxrss_bytes(),
            }
        )
    except BaseException as error:
        connection.send(
            {"ok": False, "error": type(error).__name__, "message": str(error)}
        )
    finally:
        connection.close()


def profile_int7_players(
    registrations: list[Any],
    *,
    source_games: list[dict[str, Any]],
    profile_roots: ProfileRoots,
    checkpoint_paths: dict[str, str],
    comparison_seed_aliases: dict[str, str],
) -> dict[str, Any]:
    roots = select_profile_roots(
        source_games, warmup=profile_roots.warmup, measured=profile_roots.measured
    )
    root_ids = [root["root_id"] for root in roots]
    context = multiprocessing.get_context("spawn")
    players: dict[str, Any] = {}
    for registration in registrations:
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_profile_worker,
            args=(
                child,
                registration.model_dump(),
                isinstance(registration, Int7PlayerRegistration),
                roots,
                profile_roots.warmup,
                checkpoint_paths.get(registration.player_id),
                profile_roots.sampler_interval_ms,
                comparison_seed_aliases.get(
                    registration.player_id, registration.player_id
                ),
            ),
        )
        process.start()
        child.close()
        result = parent.recv()
        process.join()
        if process.exitcode != 0 or not result.get("ok"):
            raise RuntimeError(
                f"INT-7 matched-root profile failed for {registration.player_id}: "
                f"{result.get('error')}: {result.get('message')}"
            )
        result.pop("ok")
        players[registration.player_id] = _summarize_isolated(result)
    payload = {
        "schema_version": 1,
        "method": "isolated-matched-root-v1",
        "promotion_authority": False,
        "source_cell": profile_roots.source_cell,
        "selection": profile_roots.selection,
        "warmup_roots": profile_roots.warmup,
        "measured_roots": profile_roots.measured,
        "root_ids": root_ids,
        "root_corpus_sha256": canonical_sha256(root_ids),
        "sampler_interval_ms": profile_roots.sampler_interval_ms,
        "players": players,
    }
    verify_profile(payload)
    return payload


class ResourceCapExceeded(RuntimeError):
    def __init__(self, evidence: dict[str, Any]):
        super().__init__(json.dumps(evidence, sort_keys=True))
        self.evidence = evidence


class CumulativeResourceLedger:
    """Persistent chained cap ledger checked before every registered launch."""

    def __init__(self, out_dir: Path, *, started: float, caps: dict[str, Any]):
        self.out_dir = out_dir
        self.path = out_dir / "resource-ledger.jsonl"
        self.started = started
        self.caps = dict(caps)
        self.core_seconds = 0.0

    def _artifact_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self.out_dir.rglob("*") if path.is_file()
        )

    def _append(self, kind: str, payload: dict[str, Any]) -> None:
        previous = None
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            if lines:
                previous = json.loads(lines[-1])["event_sha256"]
        unsigned = {
            "kind": kind,
            "recorded_unix": time.time(),
            "previous_event_sha256": previous,
            "payload": payload,
        }
        event = {**unsigned, "event_sha256": canonical_sha256(unsigned)}
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def check(
        self,
        stage: str,
        *,
        projected_wall_seconds: float,
        projected_workers: int,
        projected_artifact_bytes: int = 0,
        worker_launch: int | None = None,
    ) -> None:
        wall_seconds = time.perf_counter() - self.started
        artifact_bytes = self._artifact_bytes()
        projected = {
            "wall_hours": (wall_seconds + projected_wall_seconds) / 3600.0,
            "core_hours": (
                self.core_seconds + projected_wall_seconds * projected_workers
            )
            / 3600.0,
            "artifact_bytes": artifact_bytes + projected_artifact_bytes,
            "workers": projected_workers,
        }
        passed = (
            projected["wall_hours"] <= self.caps["wall_hours"]
            and projected["core_hours"] <= self.caps["core_hours"]
            and projected["artifact_bytes"] <= self.caps["artifact_bytes"]
            and projected_workers <= self.caps["workers"]
        )
        payload = {
            "stage": stage,
            "worker_launch": worker_launch,
            "actual": {
                "wall_hours": wall_seconds / 3600.0,
                "core_hours": self.core_seconds / 3600.0,
                "artifact_bytes": artifact_bytes,
            },
            "projected": projected,
            "caps": self.caps,
            "passed": passed,
        }
        self._append("preflight", payload)
        if not passed:
            evidence = {
                "status": "resource_cap_exceeded",
                "stage": stage,
                "receipt": payload,
            }
            self._append("resource_cap_exceeded", evidence)
            raise ResourceCapExceeded(evidence)

    def finish(self, stage: str, *, elapsed_seconds: float, workers: int) -> None:
        self.core_seconds += elapsed_seconds * workers
        wall_seconds = time.perf_counter() - self.started
        actual = {
            "wall_hours": wall_seconds / 3600.0,
            "core_hours": self.core_seconds / 3600.0,
            "artifact_bytes": self._artifact_bytes(),
            "workers": workers,
        }
        passed = (
            actual["wall_hours"] <= self.caps["wall_hours"]
            and actual["core_hours"] <= self.caps["core_hours"]
            and actual["artifact_bytes"] <= self.caps["artifact_bytes"]
            and workers <= self.caps["workers"]
        )
        self._append(
            "stage_complete", {"stage": stage, "actual": actual, "passed": passed}
        )
        if not passed:
            evidence = {
                "status": "resource_cap_exceeded",
                "stage": stage,
                "receipt": actual,
            }
            self._append("resource_cap_exceeded", evidence)
            raise ResourceCapExceeded(evidence)

    def complete(self, payload: dict[str, Any]) -> None:
        self._append("complete", payload)


def verify_resource_ledger(path: Path, caps: dict[str, Any]) -> dict[str, Any]:
    previous = None
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    for event in events:
        unsigned = dict(event)
        event_sha = unsigned.pop("event_sha256", None)
        if unsigned.get("previous_event_sha256") != previous:
            raise ValueError("INT-7 resource ledger chain mismatch")
        if canonical_sha256(unsigned) != event_sha:
            raise ValueError("INT-7 resource ledger digest mismatch")
        previous = event_sha
        if event["kind"] == "preflight":
            if event["payload"]["caps"] != caps or not event["payload"]["passed"]:
                raise ValueError("INT-7 resource ledger preflight failed")
    if not events or events[-1]["kind"] != "complete":
        raise ValueError("INT-7 resource ledger is incomplete")
    worker_events = [
        event
        for event in events
        if event["kind"] == "preflight"
        and event["payload"].get("worker_launch") is not None
    ]
    if len(worker_events) < 4:
        raise ValueError("INT-7 resource ledger lacks worker-launch checks")
    return {"events": len(events), "head_sha256": previous}


def phase_indices(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    phase_start = 2 + 7
    one_hot = np.asarray(dataset["agent_player"])[:, 0, phase_start : phase_start + 5]
    phases = np.argmax(one_hot, axis=1)
    if not np.allclose(one_hot.sum(axis=1), 1.0):
        raise ValueError("retained observation phase one-hot is invalid")
    return {
        name: np.flatnonzero(phases == index) for index, name in enumerate(PHASE_NAMES)
    }


def _reliability(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(predictions) == 0:
        return {
            "count": 0,
            "status": "insufficient_n",
            "mean_prediction": None,
            "mean_target": None,
            "brier": None,
            "binary_cross_entropy": None,
            "expected_calibration_error": None,
            "bins": [],
        }
    clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
    bins = []
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        mask = (predictions >= lower) & (
            predictions <= upper if index == 9 else predictions < upper
        )
        count = int(mask.sum())
        mean_prediction = float(predictions[mask].mean()) if count else None
        mean_target = float(targets[mask].mean()) if count else None
        if count:
            ece += count / len(predictions) * abs(mean_prediction - mean_target)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
            }
        )
    return {
        "count": len(predictions),
        "status": "sufficient" if len(predictions) >= 10 else "insufficient_n",
        "mean_prediction": float(predictions.mean()),
        "mean_target": float(targets.mean()),
        "brier": float(np.mean((predictions - targets) ** 2)),
        "binary_cross_entropy": float(
            np.mean(-(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))
        ),
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def checkpoint_calibration(
    checkpoint_path: Path,
    dataset: dict[str, np.ndarray],
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    agent, _ = load_checkpoint_agent(str(checkpoint_path))
    obs = {
        key: torch.as_tensor(dataset[key][validation_indices], dtype=torch.float32)
        for key in OBS_KEYS
    }
    with torch.inference_mode():
        _logits, value_logits = agent(obs)
        predictions = torch.sigmoid(value_logits).cpu().numpy().astype(np.float64)
    outcome_usable, terminal = outcome_targets(dataset["winner"], dataset["seat"])
    roots = np.asarray(dataset[ROOT_VALUE_KEY], dtype=np.float64)
    targets = {
        "terminal_outcome": terminal.astype(np.float64),
        "teacher_root_value": roots,
        "blend_50_50": 0.5 * terminal + 0.5 * roots,
    }
    phases = phase_indices(dataset)
    result: dict[str, Any] = {
        "validation_indices_sha256": canonical_sha256(validation_indices.tolist()),
        "reliability_bins": 10,
        "sparse_threshold": 10,
        "target_sources": {},
    }
    for source, target in targets.items():
        finite = outcome_usable & np.isfinite(target)
        overall_indices = validation_indices[finite[validation_indices]]
        source_rows: dict[str, Any] = {
            "interpretation": (
                "calibration"
                if source == "terminal_outcome"
                else "target_source_agreement_not_ground_truth"
            ),
            "overall": _reliability(
                predictions[finite[validation_indices]], target[overall_indices]
            ),
            "phases": {},
        }
        for phase_name, phase_rows in phases.items():
            selected = np.intersect1d(overall_indices, phase_rows, assume_unique=False)
            positions = np.flatnonzero(np.isin(validation_indices, selected))
            source_rows["phases"][phase_name] = _reliability(
                predictions[positions], target[selected]
            )
        result["target_sources"][source] = source_rows
    return result


def mechanism_payload(
    profile: dict[str, Any], registrations: list[Int7PlayerRegistration]
) -> dict[str, Any]:
    by_key = {
        (row.arm, row.training_seed, row.value_mode): row for row in registrations
    }
    neutral_learned = {}
    policy_drift = {}
    for arm in ARM_ORDER:
        for seed in MODEL_SEEDS:
            primary_mode = "neutral" if arm == "visit_policy_only" else "learned"
            primary = by_key[(arm, seed, primary_mode)]
            primary_samples = profile["players"][primary.player_id]["samples"]
            if arm != "visit_policy_only":
                neutral = by_key[(arm, seed, "neutral")]
                neutral_samples = profile["players"][neutral.player_id]["samples"]
                paired = list(zip(primary_samples, neutral_samples, strict=True))
                neutral_learned[f"{arm}-seed-{seed}"] = {
                    "roots": len(paired),
                    "action_agreement": float(
                        np.mean(
                            [
                                left["action"] == right["action"]
                                for left, right in paired
                            ]
                        )
                    ),
                    "mean_absolute_root_value_delta": float(
                        np.mean(
                            [
                                abs(left["root_value"] - right["root_value"])
                                for left, right in paired
                            ]
                        )
                    ),
                    "forward_count_match": all(
                        left["forward_calls"] == right["forward_calls"]
                        for left, right in paired
                    ),
                }
            if arm != "visit_policy_only":
                control = by_key[("visit_policy_only", seed, "neutral")]
                control_samples = profile["players"][control.player_id]["samples"]
                joint_neutral = by_key[(arm, seed, "neutral")]
                joint_samples = profile["players"][joint_neutral.player_id]["samples"]
                paired = list(zip(control_samples, joint_samples, strict=True))
                kls = []
                for left, right in paired:
                    p = np.asarray(left["input_prior"], dtype=np.float64)
                    q = np.asarray(right["input_prior"], dtype=np.float64)
                    kls.append(float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))))
                policy_drift[f"{arm}-seed-{seed}"] = {
                    "roots": len(paired),
                    "mean_control_to_joint_prior_kl": float(np.mean(kls)),
                    "neutral_action_agreement": float(
                        np.mean(
                            [
                                left["action"] == right["action"]
                                for left, right in paired
                            ]
                        )
                    ),
                }
    return {
        "neutral_vs_learned": neutral_learned,
        "shared_encoder_policy_drift": policy_drift,
    }
