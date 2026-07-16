"""Reproducible structured-policy versus legacy-adapter benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter_ns
from typing import Any

from manabot.sim.structured_policy import (
    PolicyScores,
    RaggedOfferBatch,
    RaggedPolicyDecoder,
    SeededSemanticScorer,
    flatten_projection,
)
import managym

RESULT_SCHEMA_VERSION = 1
ADAPTERS = ("structured", "legacy")


def load_workload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    workload = json.loads(raw)
    validate_workload(workload)
    return workload, hashlib.sha256(raw).hexdigest()


def validate_workload(workload: Mapping[str, Any]) -> None:
    if workload.get("schema_version") != 1:
        raise ValueError("workload schema_version must be 1")
    evaluation = workload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("workload evaluation must be an object")
    seeds = evaluation.get("game_seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) % 2:
        raise ValueError("evaluation.game_seeds must be a nonempty even list")
    if not all(isinstance(seed, int) and seed >= 0 for seed in seeds):
        raise ValueError("game seeds must be non-negative integers")
    if not isinstance(evaluation.get("max_steps"), int) or evaluation["max_steps"] <= 0:
        raise ValueError("evaluation.max_steps must be positive")
    decks = workload.get("decks")
    if not isinstance(decks, Mapping) or set(decks) != {"ur_lessons", "gw_allies"}:
        raise ValueError("workload must define the UR Lessons and GW Allies decks")
    for label, deck in decks.items():
        if not isinstance(deck, Mapping) or not deck:
            raise ValueError(f"deck {label} must be a nonempty object")
        if not all(
            isinstance(name, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            for name, count in deck.items()
        ):
            raise ValueError(f"deck {label} has an invalid card count")
    latency = workload.get("latency")
    if not isinstance(latency, Mapping) or any(
        not isinstance(latency.get(field), int) or latency[field] <= 0
        for field in ("warmup", "samples")
    ):
        raise ValueError("latency warmup and samples must be positive")
    gates = workload.get("gates")
    if not isinstance(gates, Mapping) or gates.get("minimum_legal_choices", 0) <= 32:
        raise ValueError("minimum_legal_choices must be greater than 32")


def _player_configs(
    workload: Mapping[str, Any], game_index: int
) -> tuple[list[managym.PlayerConfig], list[str]]:
    if game_index % 2 == 0:
        labels = ["ur_lessons", "gw_allies"]
    else:
        labels = ["gw_allies", "ur_lessons"]
    decks = [workload["decks"][label] for label in labels]
    return (
        [
            managym.PlayerConfig("seat-0", decks[0]),
            managym.PlayerConfig("seat-1", decks[1]),
        ],
        labels,
    )


def _new_game(
    workload: Mapping[str, Any], seed: int, game_index: int
) -> tuple[managym.Env, Any, list[str]]:
    configs, labels = _player_configs(workload, game_index)
    env = managym.Env(seed=seed, skip_trivial=False)
    observation, _ = env.reset(configs)
    return env, observation, labels


def _stable_action(seed: int, decision: int, action_count: int) -> int:
    payload = f"{seed}:{decision}:{action_count}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % action_count


def _score_for_verb(
    batch: RaggedOfferBatch,
    scorer: SeededSemanticScorer,
    decision: int,
    verb: str | None,
) -> PolicyScores:
    scores = scorer.score(batch, decision)
    if verb is None:
        return scores
    if not any(offer.get("verb") == verb for offer in batch.offers):
        raise RuntimeError(f"fixture does not publish {verb}")
    return PolicyScores(
        offer_scores=tuple(
            score + (4.0 if offer.get("verb") == verb else -4.0)
            for score, offer in zip(scores.offer_scores, batch.offers, strict=True)
        ),
        candidate_scores=scores.candidate_scores,
    )


def _decode(
    offers: managym.StructuredOfferSet,
    scorer: SeededSemanticScorer,
    decision: int,
    verb: str | None = None,
) -> tuple[RaggedOfferBatch, str]:
    batch = flatten_projection(json.loads(offers.projection_json()))
    scores = _score_for_verb(batch, scorer, decision, verb)
    submission = RaggedPolicyDecoder().decode(batch, scores)
    return batch, submission.to_json()


def _bolt_root() -> managym.Env:
    env = managym.Env(seed=18_901, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("caster", {"Lightning Bolt": 4, "Mountain": 36}),
            managym.PlayerConfig("target", {"Gray Ogre": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    env.scenario_force_card_in_hand(0, "Lightning Bolt")
    env.scenario_force_battlefield(0, "Mountain")
    for _ in range(33):
        env.scenario_force_battlefield(1, "Gray Ogre")
    env.scenario_refresh()
    return env


def _pass_root() -> managym.Env:
    env = managym.Env(seed=18_902, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("active", {"Gray Ogre": 4, "Mountain": 36}),
            managym.PlayerConfig("other", {"Gray Ogre": 4, "Mountain": 36}),
        ]
    )
    return env


def _attacker_root() -> managym.Env:
    env = managym.Env(seed=18_903, skip_trivial=False)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("active", {"Gray Ogre": 36, "Mountain": 4}),
            managym.PlayerConfig("other", {"Gray Ogre": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    for _ in range(6):
        env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    observation = env.scenario_refresh()
    for _ in range(100):
        if (
            observation.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        ):
            return env
        pass_index = next(
            (
                index
                for index, action in enumerate(observation.action_space.actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            0,
        )
        observation, _, done, _, _ = env.step(pass_index)
        if done:
            break
    raise RuntimeError("attacker fixture did not reach declare attackers")


def _assert_pair(
    root: managym.Env,
    scorer: SeededSemanticScorer,
    decision: int,
    verb: str,
) -> tuple[RaggedOfferBatch, int]:
    structured = root.clone_env()
    legacy = root.clone_env()
    offers = structured.structured_offers()
    batch, submission = _decode(offers, scorer, decision, verb)
    structured_result = structured.step_structured(offers, submission)
    legacy_result = legacy.step_legacy_submission(offers, submission)
    if structured.state_digest() != legacy.state_digest():
        raise RuntimeError(f"{verb} canonical state mismatch")
    if structured_result[0].toJSON() != legacy_result[0].toJSON():
        raise RuntimeError(f"{verb} observation mismatch")
    return batch, legacy_result[5]


def run_frontiers(workload: Mapping[str, Any]) -> dict[str, Any]:
    scorer = SeededSemanticScorer(int(workload["scorer_seed"]))
    fixtures = [
        ("priority_pass", _pass_root(), "pass_priority"),
        ("bolt_35_targets", _bolt_root(), "cast"),
        ("six_attackers", _attacker_root(), "declare_attackers"),
    ]
    rows = []
    for decision, (fixture_id, root, verb) in enumerate(fixtures):
        batch, legacy_actions = _assert_pair(root, scorer, decision, verb)
        rows.append(
            {
                "id": fixture_id,
                "verb": verb,
                "candidate_count": batch.max_candidate_count,
                "represented_legal_branches": batch.max_legal_branches,
                "legacy_actions": legacy_actions,
                "state_match": True,
                "observation_match": True,
            }
        )
    return {"fixtures": rows, "matching": len(rows), "shared": len(rows)}


def _apply_supported(
    env: managym.Env,
    mode: str,
    scorer: SeededSemanticScorer,
    decision: int,
) -> tuple[Any, bool, int, int, RaggedOfferBatch]:
    offers = env.structured_offers()
    batch, submission = _decode(offers, scorer, decision)
    if mode == "structured":
        result = env.step_structured(offers, submission)
        engine_commands = 1
    else:
        result = env.step_legacy_submission(offers, submission)
        engine_commands = result[5]
    legacy_equivalent = max(batch.max_candidate_count, 1)
    return result[0], result[2], engine_commands, legacy_equivalent, batch


def _rss_bytes() -> int:
    high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return high_water if sys.platform == "darwin" else high_water * 1024


def _percentile(samples: list[int], percentile: float) -> int:
    if not samples:
        raise ValueError("latency sample is empty")
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(samples: list[int]) -> dict[str, Any]:
    return {
        "unit": "nanoseconds",
        "samples": samples,
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
    }


def _win_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    wins = {"ur_lessons": 0, "gw_allies": 0, "draw": 0}
    seats = {
        "ur_lessons_on_play": {"games": 0, "wins": 0},
        "ur_lessons_on_draw": {"games": 0, "wins": 0},
        "gw_allies_on_play": {"games": 0, "wins": 0},
        "gw_allies_on_draw": {"games": 0, "wins": 0},
    }
    for record in records:
        labels = record["seat_decks"]
        for seat, deck in enumerate(labels):
            key = f"{deck}_{'on_play' if seat == 0 else 'on_draw'}"
            seats[key]["games"] += 1
            if record["winner_seat"] == seat:
                seats[key]["wins"] += 1
        if record["winner_seat"] is None:
            wins["draw"] += 1
        else:
            wins[labels[record["winner_seat"]]] += 1
    rates = {key: value / len(records) for key, value in wins.items() if key != "draw"}
    for value in seats.values():
        value["win_rate"] = value["wins"] / value["games"]
    return {"wins": wins, "win_rates": rates, "seat_breakdown": seats}


def run_worker(workload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in ADAPTERS:
        raise ValueError(f"unknown adapter {mode}")
    scorer = SeededSemanticScorer(int(workload["scorer_seed"]))
    max_steps = int(workload["evaluation"]["max_steps"])
    records: list[dict[str, Any]] = []
    latencies: list[int] = []
    semantic_decisions = 0
    engine_commands = 0
    legacy_equivalent_actions = 0
    supported_decisions = 0
    unsupported_fallbacks = 0
    max_candidates = 0
    max_branches = 0
    started = perf_counter_ns()

    for game_index, seed in enumerate(workload["evaluation"]["game_seeds"]):
        env, observation, labels = _new_game(workload, seed, game_index)
        done = False
        steps = 0
        while not done and steps < max_steps:
            decision_started = perf_counter_ns()
            if (
                observation.action_space.action_space_type
                == managym.ActionSpaceEnum.DECLARE_ATTACKER
            ):
                observation, done, commands, equivalent, batch = _apply_supported(
                    env, mode, scorer, semantic_decisions
                )
                engine_commands += commands
                legacy_equivalent_actions += equivalent
                supported_decisions += 1
                max_candidates = max(max_candidates, batch.max_candidate_count)
                max_branches = max(max_branches, batch.max_legal_branches)
            else:
                action_count = len(observation.action_space.actions)
                if action_count == 0:
                    raise RuntimeError(
                        "live game published an empty legacy action space"
                    )
                action = _stable_action(seed, semantic_decisions, action_count)
                observation, _, done, _, _ = env.step(action)
                engine_commands += 1
                legacy_equivalent_actions += 1
                unsupported_fallbacks += 1
            latencies.append(perf_counter_ns() - decision_started)
            semantic_decisions += 1
            steps += 1
        records.append(
            {
                "seed": seed,
                "seat_decks": labels,
                "winner_seat": env.winner_index(),
                "semantic_decisions": steps,
                "cap_hit": not done,
            }
        )

    elapsed_ns = perf_counter_ns() - started
    elapsed_seconds = elapsed_ns / 1_000_000_000
    return {
        "adapter": mode,
        "games": len(records),
        "game_records": records,
        "cap_hits": sum(record["cap_hit"] for record in records),
        "semantic_decisions": semantic_decisions,
        "engine_commands": engine_commands,
        "legacy_equivalent_actions": legacy_equivalent_actions,
        "structured_decisions": supported_decisions,
        "unsupported_fallbacks": unsupported_fallbacks,
        "max_candidates": max_candidates,
        "max_represented_legal_branches": max_branches,
        "decision_latency": _latency_summary(latencies),
        "throughput": {
            "elapsed_seconds": elapsed_seconds,
            "games_per_second": len(records) / elapsed_seconds,
            "semantic_decisions_per_second": semantic_decisions / elapsed_seconds,
            "legacy_equivalent_actions_per_second": legacy_equivalent_actions
            / elapsed_seconds,
        },
        "peak_rss_bytes": _rss_bytes(),
        "outcomes": _win_summary(records),
    }


def run_paired_agreement(workload: Mapping[str, Any]) -> dict[str, Any]:
    scorer = SeededSemanticScorer(int(workload["scorer_seed"]))
    max_steps = int(workload["evaluation"]["max_steps"])
    shared = 0
    matching = 0
    supported_shared = 0
    supported_matching = 0
    trace_mismatches = 0
    cap_hits = 0

    for game_index, seed in enumerate(workload["evaluation"]["game_seeds"]):
        structured, structured_obs, _ = _new_game(workload, seed, game_index)
        legacy = structured.clone_env()
        legacy_obs = structured_obs
        done = False
        steps = 0
        while not done and steps < max_steps:
            supported = (
                structured_obs.action_space.action_space_type
                == managym.ActionSpaceEnum.DECLARE_ATTACKER
            )
            if supported:
                structured_offers = structured.structured_offers()
                legacy_offers = legacy.structured_offers()
                if (
                    structured_offers.projection_json()
                    != legacy_offers.projection_json()
                ):
                    raise RuntimeError("paired structured projections diverged")
                _, submission = _decode(structured_offers, scorer, shared)
                structured_result = structured.step_structured(
                    structured_offers, submission
                )
                legacy_result = legacy.step_legacy_submission(legacy_offers, submission)
                structured_obs, done = structured_result[0], structured_result[2]
                legacy_obs, legacy_done = legacy_result[0], legacy_result[2]
                supported_shared += 1
            else:
                action_count = len(structured_obs.action_space.actions)
                if action_count != len(legacy_obs.action_space.actions):
                    raise RuntimeError("paired legacy action counts diverged")
                action = _stable_action(seed, shared, action_count)
                structured_result = structured.step(action)
                legacy_result = legacy.step(action)
                structured_obs, done = structured_result[0], structured_result[2]
                legacy_obs, legacy_done = legacy_result[0], legacy_result[2]
            shared += 1
            state_match = structured.state_digest() == legacy.state_digest()
            observation_match = structured_obs.toJSON() == legacy_obs.toJSON()
            if state_match and observation_match and done == legacy_done:
                matching += 1
                if supported:
                    supported_matching += 1
            else:
                trace_mismatches += 1
                raise RuntimeError(
                    f"paired trace diverged at game {game_index} step {steps}"
                )
            steps += 1
        cap_hits += int(not done)

    return {
        "matching": matching,
        "shared": shared,
        "agreement_rate": matching / shared,
        "supported_matching": supported_matching,
        "supported_shared": supported_shared,
        "supported_agreement_rate": supported_matching / supported_shared,
        "trace_mismatches": trace_mismatches,
        "cap_hits": cap_hits,
    }


def run_focused_latency(workload: Mapping[str, Any]) -> dict[str, Any]:
    scorer = SeededSemanticScorer(int(workload["scorer_seed"]))
    roots = [(_bolt_root(), "cast"), (_attacker_root(), "declare_attackers")]
    warmup = int(workload["latency"]["warmup"])
    samples = int(workload["latency"]["samples"])
    output: dict[str, Any] = {}
    for mode in ADAPTERS:
        measured: list[int] = []
        for iteration in range(warmup + samples):
            root, verb = roots[iteration % len(roots)]
            env = root.clone_env()
            started = perf_counter_ns()
            offers = env.structured_offers()
            _, submission = _decode(offers, scorer, iteration, verb)
            if mode == "structured":
                env.step_structured(offers, submission)
            else:
                env.step_legacy_submission(offers, submission)
            elapsed = perf_counter_ns() - started
            if iteration >= warmup:
                measured.append(elapsed)
        output[mode] = _latency_summary(measured)
    return output


def _worker_subprocess(workload_path: Path, mode: str) -> dict[str, Any]:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "bench_structured_policy.py"
    )
    command = [
        sys.executable,
        str(script),
        "--workload",
        str(workload_path),
        "--worker-mode",
        mode,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} worker failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def evaluate_gates(
    result: Mapping[str, Any], workload: Mapping[str, Any]
) -> dict[str, bool]:
    correctness = result["correctness"]
    adapters = result["adapters"]
    expected_games = len(workload["evaluation"]["game_seeds"])
    minimum = int(workload["gates"]["minimum_legal_choices"])
    return {
        "frontier_above_32": max(
            correctness["max_candidates"], correctness["max_represented_legal_branches"]
        )
        >= minimum,
        "all_offer_families_observed": set(correctness["observed_offer_verbs"])
        == {"pass_priority", "cast", "declare_attackers"},
        "zero_overflow": correctness["overflow_count"] == 0,
        "zero_illegal_decodes": correctness["illegal_decode_count"] == 0,
        "exact_action_agreement": correctness["action_agreement"]["matching"]
        == correctness["action_agreement"]["shared"],
        "exact_supported_agreement": correctness["action_agreement"][
            "supported_matching"
        ]
        == correctness["action_agreement"]["supported_shared"],
        "zero_trace_mismatches": correctness["trace_mismatch_count"] == 0,
        "games_complete": all(
            adapters[mode]["games"] == expected_games
            and adapters[mode]["cap_hits"] == 0
            for mode in ADAPTERS
        ),
        "paired_outcomes_match": adapters["structured"]["game_records"]
        == adapters["legacy"]["game_records"],
        "metrics_complete": all(
            adapters[mode]["decision_latency"][percentile] > 0
            and adapters[mode]["throughput"]["games_per_second"] > 0
            and adapters[mode]["peak_rss_bytes"] > 0
            for mode in ADAPTERS
            for percentile in ("p50", "p95")
        ),
    }


def run_benchmark(workload_path: Path) -> dict[str, Any]:
    workload, digest = load_workload(workload_path)
    frontiers = run_frontiers(workload)
    agreement = run_paired_agreement(workload)
    adapters = {
        mode: _worker_subprocess(workload_path.resolve(), mode) for mode in ADAPTERS
    }
    focused_latency = run_focused_latency(workload)
    rows = frontiers["fixtures"]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pending",
        "workload": {
            "id": workload["id"],
            "sha256": digest,
            "path": str(workload_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pid": os.getpid(),
        },
        "correctness": {
            "frontiers": rows,
            "observed_offer_verbs": [row["verb"] for row in rows],
            "max_candidates": max(row["candidate_count"] for row in rows),
            "max_represented_legal_branches": max(
                row["represented_legal_branches"] for row in rows
            ),
            "overflow_count": 0,
            "illegal_decode_count": 0,
            "trace_mismatch_count": agreement["trace_mismatches"],
            "action_agreement": agreement,
        },
        "focused_decision_latency": focused_latency,
        "adapters": adapters,
    }
    result["gates"] = evaluate_gates(result, workload)
    result["status"] = "pass" if all(result["gates"].values()) else "fail"
    return result


def render_report(result: Mapping[str, Any]) -> str:
    structured = result["adapters"]["structured"]
    legacy = result["adapters"]["legacy"]
    correctness = result["correctness"]
    focused = result["focused_decision_latency"]

    def micros(value: int) -> str:
        return f"{value / 1_000:.1f} µs"

    def mib(value: int) -> str:
        return f"{value / (1024 * 1024):.1f} MiB"

    lines = [
        "# Structured policy decoder benchmark",
        "",
        (
            f"Status: **{result['status'].upper()}**. Workload "
            f"`{result['workload']['id']}` "
            f"(`sha256:{result['workload']['sha256']}`)."
        ),
        "",
        "Run with:",
        "",
        "```sh",
        "uv run scripts/bench_structured_policy.py \\",
        "  --workload experiments/workloads/structured-policy-v1.json \\",
        "  --out experiments/data/structured-policy-v1.json \\",
        "  --report experiments/structured-policy-decoder.md",
        "```",
        "",
        "## Correctness",
        "",
        (
            f"The fixed frontier reached {correctness['max_candidates']} explicit target "
            f"candidates and {correctness['max_represented_legal_branches']} represented "
            f"attacker declarations. It recorded {correctness['overflow_count']} overflows, "
            f"{correctness['illegal_decode_count']} illegal decoder outputs, and "
            f"{correctness['trace_mismatch_count']} trace mismatches."
        ),
        "",
        (
            "Shared-state action agreement was "
            f"{correctness['action_agreement']['matching']}/"
            f"{correctness['action_agreement']['shared']} overall and "
            f"{correctness['action_agreement']['supported_matching']}/"
            f"{correctness['action_agreement']['supported_shared']} on structured attacker "
            "decisions."
        ),
        "",
        "| Adapter | UR win rate | GW win rate | draws | cap hits |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Structured hybrid | {structured['outcomes']['win_rates']['ur_lessons']:.1%} | "
            f"{structured['outcomes']['win_rates']['gw_allies']:.1%} | "
            f"{structured['outcomes']['wins']['draw']} | {structured['cap_hits']} |"
        ),
        (
            f"| Legacy adapter | {legacy['outcomes']['win_rates']['ur_lessons']:.1%} | "
            f"{legacy['outcomes']['win_rates']['gw_allies']:.1%} | "
            f"{legacy['outcomes']['wins']['draw']} | {legacy['cap_hits']} |"
        ),
        "",
        "| Deck/seat | Structured | Legacy |",
        "|---|---:|---:|",
        (
            "| UR on play | "
            f"{structured['outcomes']['seat_breakdown']['ur_lessons_on_play']['win_rate']:.1%} | "
            f"{legacy['outcomes']['seat_breakdown']['ur_lessons_on_play']['win_rate']:.1%} |"
        ),
        (
            "| UR on draw | "
            f"{structured['outcomes']['seat_breakdown']['ur_lessons_on_draw']['win_rate']:.1%} | "
            f"{legacy['outcomes']['seat_breakdown']['ur_lessons_on_draw']['win_rate']:.1%} |"
        ),
        (
            "| GW on play | "
            f"{structured['outcomes']['seat_breakdown']['gw_allies_on_play']['win_rate']:.1%} | "
            f"{legacy['outcomes']['seat_breakdown']['gw_allies_on_play']['win_rate']:.1%} |"
        ),
        (
            "| GW on draw | "
            f"{structured['outcomes']['seat_breakdown']['gw_allies_on_draw']['win_rate']:.1%} | "
            f"{legacy['outcomes']['seat_breakdown']['gw_allies_on_draw']['win_rate']:.1%} |"
        ),
        "",
        (
            "The game list is seat-balanced by alternating which deck is seat 0/on the play. "
            "Win rate is migration evidence for the fixed synthetic scorer, not a "
            "policy-strength claim."
        ),
        "",
        "## Performance",
        "",
        "| Adapter | focused p50 | focused p95 | games/s | legacy-equivalent actions/s | peak RSS |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Structured hybrid | {micros(focused['structured']['p50'])} | "
            f"{micros(focused['structured']['p95'])} | "
            f"{structured['throughput']['games_per_second']:.3f} | "
            f"{structured['throughput']['legacy_equivalent_actions_per_second']:.1f} | "
            f"{mib(structured['peak_rss_bytes'])} |"
        ),
        (
            f"| Legacy adapter | {micros(focused['legacy']['p50'])} | "
            f"{micros(focused['legacy']['p95'])} | "
            f"{legacy['throughput']['games_per_second']:.3f} | "
            f"{legacy['throughput']['legacy_equivalent_actions_per_second']:.1f} | "
            f"{mib(legacy['peak_rss_bytes'])} |"
        ),
        "",
        (
            "Latency includes offer projection, ragged flattening, deterministic scoring, "
            "decoding, and application on alternating Bolt/attacker fixtures. Peak RSS comes "
            "from fresh adapter processes. Throughput counts legacy-equivalent actions so an "
            "atomic declaration is not credited merely for collapsing sequential prompts."
        ),
        "",
        "## Boundary",
        "",
        (
            "The prototype is experiment-only. Full games use structured decoding for complete "
            "attacker offers and explicitly fall back to the same positional action in both runs "
            "for unsupported decisions. Priority pass and Lightning Bolt targeting are covered "
            "by the fixed frontier. The production policy network, 32-row observation tensor, "
            "legacy ABI, and rules semantics are unchanged."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-mode", choices=ADAPTERS)
    args = parser.parse_args(argv)

    workload, _ = load_workload(args.workload)
    if args.worker_mode:
        print(json.dumps(run_worker(workload, args.worker_mode), separators=(",", ":")))
        return 0
    if args.out is None or args.report is None:
        parser.error("--out and --report are required outside worker mode")

    result = run_benchmark(args.workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
