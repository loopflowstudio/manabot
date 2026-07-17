"""Fixed real-consumer evidence for the selected search BranchDriver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any, Iterable

import numpy as np
import psutil

from manabot.sim.mcts import _mix_seed, determinized_puct
from manabot.sim.search_branch import (
    REFERENCE_BRANCH_DRIVER_ID,
    SELECTED_BRANCH_DRIVER_ID,
)
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "rul-2-selected-branchdriver-teacher-v1"
SOURCE_PATHS = (
    "managym/src/agent/env.rs",
    "managym/src/agent/structured_offer.rs",
    "managym/src/experience.rs",
    "managym/src/python/bindings.rs",
    "managym/src/search_state.rs",
    "manabot/sim/mcts.py",
    "manabot/sim/search_branch.py",
    "manabot/sim/selected_branchdriver_teacher.py",
    "experiments/runners/run_selected_branchdriver_teacher.py",
)


class SelectedTeacherError(RuntimeError):
    """The fixed workload, provenance, or verification contract failed."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(paths: Iterable[str] = SOURCE_PATHS) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        data = (REPO_ROOT / relative).read_bytes()
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _player_configs(ur_seat: int) -> list[managym.PlayerConfig]:
    decks = [GW_ALLIES_DECK, GW_ALLIES_DECK]
    decks[ur_seat] = UR_LESSONS_DECK
    return [
        managym.PlayerConfig(
            "ur" if seat == ur_seat else "gw",
            dict(decks[seat]),
        )
        for seat in range(2)
    ]


def _fresh_engine(seed: int, ur_seat: int) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True)
    env.reset(_player_configs(ur_seat))
    return env


def runtime_identity(seed: int = 1197) -> dict[str, Any]:
    env = _fresh_engine(seed, 0)
    manifest = env.content_pack_manifest()
    extension = Path(managym._managym.__file__)
    return {
        "source_sha256": source_sha256(),
        "source_paths": list(SOURCE_PATHS),
        "extension_name": extension.name,
        "extension_sha256": file_sha256(extension),
        "pack_manifest": manifest,
        "pack_manifest_sha256": canonical_sha256(manifest),
        "deck_counts": {
            "ur": sum(UR_LESSONS_DECK.values()),
            "gw": sum(GW_ALLIES_DECK.values()),
        },
        "drivers": {
            "selected": SELECTED_BRANCH_DRIVER_ID,
            "reference": REFERENCE_BRANCH_DRIVER_ID,
        },
    }


def validate_runtime(contract: dict[str, Any], identity: dict[str, Any]) -> None:
    expected = contract["expected_runtime"]
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise SelectedTeacherError(
            "runtime differs from contract: " + json.dumps(mismatches, sort_keys=True)
        )


def _choose_action(result: Any) -> int:
    candidates = np.flatnonzero(result.visit_counts == int(result.visit_counts.max()))
    if len(candidates) == 1:
        return int(candidates[0])
    return int(candidates[np.argmax(result.q_values[candidates])])


def _normalized_branch_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(receipt))
    normalized.pop("driver_id", None)
    for site in ("world", "child", "leaf"):
        for row in normalized["tapes"][site]:
            row["native_receipt"].pop("driver_id", None)
    return normalized


def _sum_site_counts(target: dict[str, int], source: dict[str, int]) -> dict[str, int]:
    for site in ("world", "child", "leaf"):
        target[site] += int(source[site])
    return target


def run_teacher_game(
    *,
    driver_id: str,
    deal_seed: int,
    ur_seat: int,
    simulations: int,
    worlds: int,
    max_steps: int,
    max_decisions: int,
    audit: bool,
) -> dict[str, Any]:
    """Play one authored game while every label uses real visit search."""

    env = _fresh_engine(deal_seed, ur_seat)
    manifest = env.content_pack_manifest()
    match_id = f"rul-2-{deal_seed}-ur{ur_seat}"
    decision_seconds: list[float] = []
    logical_trace: list[dict[str, Any]] = []
    counters = {
        "forks": {"world": 0, "child": 0, "leaf": 0},
        "applies": {"world": 0, "child": 0, "leaf": 0},
        "marks": 0,
        "rollbacks": 0,
        "random_playouts": 0,
        "random_playout_cap_hits": 0,
        "indexed_fallbacks": 0,
    }
    cap_hits = 0
    decision = 0
    while not env.is_game_over():
        if decision >= max_decisions:
            raise SelectedTeacherError(
                f"game {deal_seed} exceeded {max_decisions} decisions"
            )
        root_context = json.loads(env.search_context_json(audit))
        action_count = int(env.action_count())
        if action_count != len(root_context["offers"]["offers"]):
            raise SelectedTeacherError("root action and structured offer counts differ")
        root_witness = env.search_witness_json() if audit else env.state_digest()
        search_seed = _mix_seed(deal_seed, decision + 1)
        started = time.perf_counter()
        result = determinized_puct(
            env,
            simulations=simulations,
            worlds=worlds,
            seed=search_seed,
            max_steps=max_steps,
            branch_driver_id=driver_id,
            branch_audit=audit,
            branch_match_id=f"{match_id}-search-{decision}",
        )
        elapsed = time.perf_counter() - started
        decision_seconds.append(elapsed)
        if (env.search_witness_json() if audit else env.state_digest()) != root_witness:
            raise SelectedTeacherError("search mutated its authoritative source root")
        action = _choose_action(result)
        root_offer = root_context["offers"]["offers"][action]
        command = {
            "command_id": f"root.{match_id}.{decision + 1}",
            "match_id": match_id,
            "expected_revision": root_context["revision"],
            "prompt_id": root_context["prompt_id"],
            "offer_id": root_offer["id"],
            "answers": [],
        }
        offers = env.structured_search_offers()
        observation, _, terminated, truncated, _, native_actions = env.step_structured(
            offers,
            json.dumps(
                {"offer_id": command["offer_id"], "answers": []},
                separators=(",", ":"),
            ),
        )
        if native_actions != 1:
            raise SelectedTeacherError(
                "root structured Command was not one native apply"
            )
        raw_counters = result.branch_receipt["counters"]
        _sum_site_counts(counters["forks"], raw_counters["forks"])
        _sum_site_counts(counters["applies"], raw_counters["applies"])
        for name in (
            "marks",
            "rollbacks",
            "random_playouts",
            "random_playout_cap_hits",
            "indexed_fallbacks",
        ):
            counters[name] += int(raw_counters[name])
        cap_hits += int(result.cap_hits)
        normalized_branch = _normalized_branch_receipt(result.branch_receipt)
        logical_trace.append(
            {
                "decision": decision,
                "actor": int(observation.agent.player_index)
                if not (terminated or truncated)
                else None,
                "source": root_context
                if audit
                else {
                    "revision": root_context["revision"],
                    "prompt_id": root_context["prompt_id"],
                    "offers": root_context["offers"],
                },
                "search_seed": search_seed,
                "visit_counts": result.visit_counts.astype(int).tolist(),
                "q_values": result.q_values.astype(float).tolist(),
                "root_value": result.root_value,
                "world_visit_counts": result.world_visit_counts.astype(int).tolist(),
                "world_q_values": result.world_q_values.astype(float).tolist(),
                "world_root_values": result.world_root_values.astype(float).tolist(),
                "tree_nodes": result.tree_nodes,
                "max_depth": result.max_depth,
                "cap_hits": result.cap_hits,
                "command": command,
                "branch_receipt_sha256": canonical_sha256(normalized_branch),
                "branch_tape_sha256": {
                    site: canonical_sha256(normalized_branch["tapes"][site])
                    for site in ("world", "child", "leaf")
                },
                "branch_counters": normalized_branch["counters"],
                "post_root_witness": (
                    json.loads(env.search_witness_json())
                    if audit
                    else env.state_digest()
                ),
            }
        )
        decision += 1

    winner = env.winner_index()
    return {
        "driver_id": driver_id,
        "deal_seed": deal_seed,
        "ur_seat": ur_seat,
        "pack_key": manifest.get("pack_key")
        or manifest["compiled_semantics"]["pack_key"],
        "decisions": decision,
        "traversals": decision * simulations,
        "decision_seconds": decision_seconds,
        "cap_hits": cap_hits,
        "counters": counters,
        "winner": winner,
        "logical_trace_sha256": canonical_sha256(logical_trace),
        "logical_trace": logical_trace if audit else [],
    }


def compare_exact_games(
    selected: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    fields = (
        "deal_seed",
        "ur_seat",
        "pack_key",
        "decisions",
        "traversals",
        "cap_hits",
        "winner",
        "logical_trace_sha256",
    )
    return [
        {"field": field, "selected": selected[field], "reference": reference[field]}
        for field in fields
        if selected[field] != reference[field]
    ]


def _warmup(driver_id: str, seed: int) -> None:
    env = _fresh_engine(seed, 0)
    determinized_puct(
        env,
        simulations=1,
        worlds=1,
        seed=_mix_seed(seed, 0xCAFE),
        max_steps=20,
        branch_driver_id=driver_id,
    )


def _cell_worker(
    output: Any,
    barrier: Any,
    driver_id: str,
    cell: dict[str, Any],
    assignments: list[tuple[int, int]],
) -> None:
    try:
        _warmup(driver_id, assignments[0][0])
        output.put({"kind": "ready", "pid": mp.current_process().pid})
        barrier.wait()
        cpu_started = time.process_time()
        games = [
            run_teacher_game(
                driver_id=driver_id,
                deal_seed=seed,
                ur_seat=ur_seat,
                simulations=int(cell["simulations"]),
                worlds=int(cell["worlds"]),
                max_steps=int(cell["max_steps"]),
                max_decisions=int(cell["max_decisions"]),
                audit=False,
            )
            for seed, ur_seat in assignments
        ]
        output.put(
            {
                "kind": "result",
                "pid": mp.current_process().pid,
                "cpu_seconds": time.process_time() - cpu_started,
                "games": games,
            }
        )
    except BaseException as error:
        output.put(
            {
                "kind": "error",
                "pid": mp.current_process().pid,
                "error": f"{type(error).__name__}: {error}",
            }
        )


def _assign_seeds(cell: dict[str, Any]) -> list[list[tuple[int, int]]]:
    workers = int(cell["workers"])
    assignments: list[list[tuple[int, int]]] = [[] for _ in range(workers)]
    for index, seed in enumerate(cell["deal_seeds"]):
        assignments[index % workers].append((int(seed), index % 2))
    if any(not assignment for assignment in assignments):
        raise SelectedTeacherError("every measurement worker needs a game assignment")
    return assignments


def _rss_bytes(process: psutil.Process) -> int:
    try:
        return process.memory_info().rss if process.is_running() else 0
    except psutil.NoSuchProcess:
        return 0


def run_measurement_cell(driver_id: str, cell: dict[str, Any]) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    output = ctx.Queue()
    workers = int(cell["workers"])
    barrier = ctx.Barrier(workers + 1)
    processes = [
        ctx.Process(
            target=_cell_worker,
            args=(output, barrier, driver_id, cell, assignment),
        )
        for assignment in _assign_seeds(cell)
    ]
    for process in processes:
        process.start()

    ready: list[int] = []
    while len(ready) < workers:
        message = output.get(timeout=120)
        if message["kind"] == "error":
            raise SelectedTeacherError(message["error"])
        if message["kind"] == "ready":
            ready.append(int(message["pid"]))
    tracked = [psutil.Process(pid) for pid in ready]
    rss_baseline = sum(_rss_bytes(process) for process in tracked)
    rss_samples = [rss_baseline]
    barrier.wait()
    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    while len(results) < workers:
        try:
            message = output.get(timeout=0.005)
        except queue.Empty:
            message = None
        rss_samples.append(sum(_rss_bytes(process) for process in tracked))
        if message is None:
            continue
        if message["kind"] == "error":
            raise SelectedTeacherError(message["error"])
        if message["kind"] == "result":
            results.append(message)
    wall_seconds = time.perf_counter() - wall_started
    for process in processes:
        process.join(timeout=30)
        if process.exitcode != 0:
            raise SelectedTeacherError(
                f"measurement worker {process.pid} exited {process.exitcode}"
            )
    raw = {
        "driver_id": driver_id,
        "cell": cell,
        "wall_seconds": wall_seconds,
        "rss_baseline_bytes": rss_baseline,
        "rss_samples_bytes": rss_samples,
        "workers": results,
        "physical_cores": psutil.cpu_count(logical=False),
    }
    return {"raw": raw, "derived": derive_measurement(raw)}


def derive_measurement(raw: dict[str, Any]) -> dict[str, Any]:
    games = [game for worker in raw["workers"] for game in worker["games"]]
    decisions = sum(int(game["decisions"]) for game in games)
    traversals = sum(int(game["traversals"]) for game in games)
    durations = [
        float(duration) for game in games for duration in game["decision_seconds"]
    ]
    wall = float(raw["wall_seconds"])
    cpu = sum(float(worker["cpu_seconds"]) for worker in raw["workers"])
    counters = {
        "forks": {"world": 0, "child": 0, "leaf": 0},
        "applies": {"world": 0, "child": 0, "leaf": 0},
        "marks": 0,
        "rollbacks": 0,
        "random_playouts": 0,
        "random_playout_cap_hits": 0,
        "indexed_fallbacks": 0,
    }
    for game in games:
        _sum_site_counts(counters["forks"], game["counters"]["forks"])
        _sum_site_counts(counters["applies"], game["counters"]["applies"])
        for name in (
            "marks",
            "rollbacks",
            "random_playouts",
            "random_playout_cap_hits",
            "indexed_fallbacks",
        ):
            counters[name] += int(game["counters"][name])
    rss_samples = [int(value) for value in raw["rss_samples_bytes"]]
    baseline = int(raw["rss_baseline_bytes"])
    cap_hits = sum(int(game["cap_hits"]) for game in games)
    return {
        "decisions": decisions,
        "traversals": traversals,
        "decisions_per_second": decisions / wall,
        "traversals_per_second": traversals / wall,
        "p50_decision_ms": float(np.percentile(durations, 50)) * 1000,
        "p95_decision_ms": float(np.percentile(durations, 95)) * 1000,
        "cpu_ms_per_label": cpu * 1000 / decisions,
        "inner_search_ms_per_label": sum(durations) * 1000 / decisions,
        "wall_ms_per_label": wall * 1000 / decisions,
        "peak_rss_bytes": max(rss_samples),
        "peak_rss_delta_bytes": max(rss_samples) - baseline,
        "cap_hits": cap_hits,
        "cap_rate": cap_hits / traversals,
        "counters": counters,
        "outcomes_sha256": canonical_sha256(
            [
                {
                    "deal_seed": game["deal_seed"],
                    "ur_seat": game["ur_seat"],
                    "winner": game["winner"],
                    "logical_trace_sha256": game["logical_trace_sha256"],
                }
                for game in games
            ]
        ),
    }


def evaluate_verdict(
    contract: dict[str, Any], measurements: dict[str, Any], exact_mismatches: int
) -> dict[str, Any]:
    failures: list[str] = []
    if exact_mismatches:
        failures.append(f"{exact_mismatches} exactness mismatch(es)")
        return {
            "decision": "remove",
            "split_justified": False,
            "failures": failures,
        }
    for cell_name, cell in contract["cells"].items():
        selected = measurements[cell_name][SELECTED_BRANCH_DRIVER_ID]["derived"]
        reference = measurements[cell_name][REFERENCE_BRANCH_DRIVER_ID]["derived"]
        gates = cell["gates"]
        if selected["decisions_per_second"] < float(gates["decisions_per_second_min"]):
            failures.append(f"{cell_name}: decisions/s absolute gate")
        if selected["traversals_per_second"] < float(
            gates["traversals_per_second_min"]
        ):
            failures.append(f"{cell_name}: traversals/s absolute gate")
        if selected["p95_decision_ms"] > float(gates["p95_decision_ms_max"]):
            failures.append(f"{cell_name}: p95 absolute gate")
        if selected["peak_rss_bytes"] > int(gates["peak_rss_bytes_max"]):
            failures.append(f"{cell_name}: RSS absolute gate")
        if selected["cap_rate"] != 0 or selected["counters"]["indexed_fallbacks"] != 0:
            failures.append(f"{cell_name}: cap or fallback exactness gate")
        if selected["decisions_per_second"] < 0.9 * reference["decisions_per_second"]:
            failures.append(f"{cell_name}: decisions/s matched gate")
        if selected["traversals_per_second"] < 0.9 * reference["traversals_per_second"]:
            failures.append(f"{cell_name}: traversals/s matched gate")
        if selected["p95_decision_ms"] > 1.15 * reference["p95_decision_ms"]:
            failures.append(f"{cell_name}: p95 matched gate")
        if selected["peak_rss_bytes"] > 1.10 * reference["peak_rss_bytes"]:
            failures.append(f"{cell_name}: RSS matched gate")
    return {
        "decision": "remain" if not failures else "remove",
        "split_justified": False,
        "failures": failures,
    }


def execute_contract(contract: dict[str, Any]) -> dict[str, Any]:
    identity = runtime_identity(int(contract["runtime_seed"]))
    validate_runtime(contract, identity)
    exactness: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    audit = contract["exactness"]
    for index, seed in enumerate(audit["deal_seeds"]):
        kwargs = {
            "deal_seed": int(seed),
            "ur_seat": index % 2,
            "simulations": int(audit["simulations"]),
            "worlds": int(audit["worlds"]),
            "max_steps": int(audit["max_steps"]),
            "max_decisions": int(audit["max_decisions"]),
            "audit": True,
        }
        selected = run_teacher_game(driver_id=SELECTED_BRANCH_DRIVER_ID, **kwargs)
        reference = run_teacher_game(driver_id=REFERENCE_BRANCH_DRIVER_ID, **kwargs)
        game_mismatches = compare_exact_games(selected, reference)
        exactness.append(
            {
                "deal_seed": seed,
                "selected": selected,
                "reference": reference,
                "mismatches": game_mismatches,
            }
        )
        mismatches.extend(
            {"deal_seed": seed, **mismatch} for mismatch in game_mismatches
        )
        if game_mismatches:
            break

    measurements: dict[str, Any] = {}
    if not mismatches:
        for cell_name, cell in contract["cells"].items():
            measurements[cell_name] = {
                driver_id: run_measurement_cell(driver_id, cell)
                for driver_id in (
                    REFERENCE_BRANCH_DRIVER_ID,
                    SELECTED_BRANCH_DRIVER_ID,
                )
            }
    verdict = evaluate_verdict(contract, measurements, len(mismatches))
    receipt = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "contract_sha256": canonical_sha256(contract),
        "runtime": identity,
        "exactness": exactness,
        "exactness_mismatches": mismatches,
        "measurements": measurements,
        "verdict": verdict,
    }
    receipt["artifact_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_receipt(contract: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("contract_sha256") != canonical_sha256(contract):
        raise SelectedTeacherError("receipt does not bind to the contract")
    validate_runtime(contract, runtime_identity(int(contract["runtime_seed"])))
    expected_artifact = receipt.get("artifact_sha256")
    without_artifact = dict(receipt)
    without_artifact.pop("artifact_sha256", None)
    if expected_artifact != canonical_sha256(without_artifact):
        raise SelectedTeacherError("receipt artifact digest mismatch")
    for cell in receipt["measurements"].values():
        for measurement in cell.values():
            if measurement["derived"] != derive_measurement(measurement["raw"]):
                raise SelectedTeacherError("measurement derivation mismatch")
    expected_verdict = evaluate_verdict(
        contract,
        receipt["measurements"],
        len(receipt["exactness_mismatches"]),
    )
    if receipt["verdict"] != expected_verdict:
        raise SelectedTeacherError("receipt verdict does not recompute")
    return {
        "verified": True,
        "artifact_sha256": expected_artifact,
        "decision": expected_verdict["decision"],
    }


@dataclass(frozen=True)
class FailureReplay:
    reproduced: bool
    selected_sha256: str
    reference_sha256: str


def replay_failure(capsule: dict[str, Any]) -> FailureReplay:
    kwargs = dict(capsule["workload"])
    selected = run_teacher_game(
        driver_id=SELECTED_BRANCH_DRIVER_ID, audit=True, **kwargs
    )
    reference = run_teacher_game(
        driver_id=REFERENCE_BRANCH_DRIVER_ID, audit=True, **kwargs
    )
    selected_hash = selected["logical_trace_sha256"]
    reference_hash = reference["logical_trace_sha256"]
    if capsule.get("corrupt_reference_sha256"):
        reference_hash = str(capsule["corrupt_reference_sha256"])
    return FailureReplay(
        reproduced=selected_hash != reference_hash,
        selected_sha256=selected_hash,
        reference_sha256=reference_hash,
    )
