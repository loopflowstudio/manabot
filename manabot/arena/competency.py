"""Common-seed S1-S5 competency evidence for arena registrations."""

from __future__ import annotations

from typing import Any

import numpy as np

from manabot.verify.competency import (
    SCENARIOS,
    aggregate_scenario_results,
    run_scenario_once,
)

from .models import PlayerRegistration, canonical_sha256
from .players import build_player


def competency_seed(player_id: str, scenario: str, run_seed: int) -> int:
    return int(canonical_sha256([player_id, scenario, run_seed])[:16], 16)


def run_competencies(
    registrations: list[PlayerRegistration],
    *,
    seeds: tuple[int, ...],
    checkpoint_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint_paths = checkpoint_paths or {}
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "scenario_authority": "manabot.verify.competency.SCENARIOS",
        "scenario_seeds": list(seeds),
        "scenario_seed_set_sha256": canonical_sha256(list(seeds)),
        "players": {},
    }
    for registration in registrations:
        player_block: dict[str, Any] = {}
        for scenario_name, scenario in SCENARIOS.items():
            runs = []
            for run_seed in seeds:
                seed = competency_seed(registration.player_id, scenario_name, run_seed)
                player, obs_space = build_player(
                    registration,
                    seed=seed,
                    checkpoint_path=checkpoint_paths.get(registration.player_id),
                )
                result = run_scenario_once(scenario, player, obs_space, seed=seed)
                runs.append({"run_seed": run_seed, "player_seed": seed, **result})
            player_block[scenario_name] = {
                "correct_line": scenario.correct_line,
                "runs": runs,
                "aggregate": aggregate_scenario_results(runs),
            }
        evidence["players"][registration.player_id] = player_block
    return evidence


def competency_noninferiority(
    evidence: dict[str, Any],
    challenger: str,
    incumbent: str,
    *,
    margin: float,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(bootstrap_seed)
    result = {}
    for scenario in SCENARIOS:
        challenger_runs = evidence["players"][challenger][scenario]["runs"]
        incumbent_runs = evidence["players"][incumbent][scenario]["runs"]
        paired = list(zip(challenger_runs, incumbent_runs, strict=True))
        if any(left["run_seed"] != right["run_seed"] for left, right in paired):
            raise ValueError("competency evidence is not paired by scenario seed")
        differences = np.asarray(
            [
                float(bool(left["correct"])) - float(bool(right["correct"]))
                for left, right in paired
            ],
            dtype=np.float64,
        )
        if not len(differences):
            raise ValueError("competency noninferiority requires paired runs")
        bootstrap = np.mean(
            differences[
                rng.integers(
                    0,
                    len(differences),
                    size=(bootstrap_replicates, len(differences)),
                )
            ],
            axis=1,
        )
        point = float(np.mean(differences))
        lower = float(np.percentile(bootstrap, 5.0))
        result[scenario] = {
            "point_difference": point,
            "one_sided_95_lower": lower,
            "noninferiority_floor": -margin,
            "point_passed": point >= -margin,
            "lower_bound_passed": lower > -margin,
            "passed": point >= -margin and lower > -margin,
        }
    return {
        "available": True,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "scenarios": result,
        "passed": all(block["passed"] for block in result.values()),
    }
