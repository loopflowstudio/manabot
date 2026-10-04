"""Freeze count-based scientific workloads from completed integrated calibration."""

import hashlib
import json
from pathlib import Path
import shutil

from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256
from manabot.training.analysis import verify_saved_inputs
from manabot.training.models import CollectSearch, TrainingRegime, TrainSelfPlay


def scientific_plan(
    calibration: Path | str, prior_seconds: float, reserved_disk_bytes: int = 0
) -> ResolvedStudy:
    """Reserve both studies inside 168 hours; extrapolation is not timing proof."""
    out = Path(calibration).resolve()
    study = json.loads((out / "study.json").read_text())
    if study["study"] == "compound-decisions":
        raise ValueError(
            "compound study needs a separate allocation; the ETU-91 campaign budget cannot be reused"
        )
    verify_saved_inputs(out, study)
    if study["profile"] != "calibration" or study["status"] != "completed":
        raise ValueError("a completed integrated calibration cohort is required")
    screen = study["study"] == "ataraxos-ablations"
    checkpoints = 3 if screen else 4
    run_budget = (1 if screen else 18) * 3600
    evaluation_budget = (6 if screen else 24) * 3600
    allocation = (21 if screen else 132) * 3600
    # Reserve the other study, four hours of calibration and eleven of recovery.
    minimum_prior = (4 if screen else 25) * 3600
    prior_seconds = max(prior_seconds, minimum_prior)
    if prior_seconds + allocation + (132 * 3600 if screen else 0) > 157 * 3600:
        raise ValueError(
            "campaign has insufficient budget; amend allocations before scoring"
        )
    runs = [json.loads(Path(r["path"]).read_text()) for r in study["runs"]]
    identity_keys = (
        "engine_extension_sha256",
        "engine_source_sha256",
        "training_source_sha256",
        "content_manifest_sha256",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
    )
    identities = {k: runs[0]["identities"][k] for k in identity_keys}
    if any({k: r["identities"][k] for k in identity_keys} != identities for r in runs):
        raise ValueError("calibration mixes runtime or source identities")
    recipes = []
    predicted = []
    for run in runs:
        if run["status"] != "completed":
            raise ValueError("calibration contains an incomplete training run")
        recipe = TrainingRegime.model_validate(run["regime"])
        records = {s["id"]: s for s in run["stages"]}
        recipe.wall_seconds = run_budget
        stages = []
        segment = run_budget / checkpoints
        if isinstance(recipe.stages[0], TrainSelfPlay):
            template = recipe.stages[0]
            rate = max(records[s.id]["seconds"] / s.updates for s in recipe.stages)
            overhead = max(r["export_seconds"] for r in records.values())
            updates = int((segment * 0.75 - overhead - run["setup_seconds"]) / rate)
            if updates <= template.updates:
                raise ValueError("calibration cannot support an expanded RL workload")
            for index in range(checkpoints):
                stage = template.model_copy(deep=True)
                stage.id = f"policy-{index}"
                stage.initial = f"policy-{index - 1}" if index else None
                stage.updates = updates
                stage.execution.wall_seconds = segment
                stages.append(stage)
            predicted.append(updates * rate)
        else:
            collect = next(s for s in recipe.stages if isinstance(s, CollectSearch))
            fit = next(s for s in recipe.stages if s.operation == "train_supervised")
            collect_rate = max(
                records[s.id]["seconds"] / s.games
                for s in recipe.stages
                if isinstance(s, CollectSearch)
            )
            # Fit cost grows with the cumulative corpus, including every epoch.
            fit_rate = max(
                records[s.id]["seconds"]
                / sum(
                    next(t.games for t in recipe.stages if t.id == ref)
                    for ref in s.datasets
                )
                for s in recipe.stages
                if s.operation == "train_supervised"
            )
            games = int(
                (segment * 0.75 - run["setup_seconds"])
                / (collect_rate + checkpoints * fit_rate)
            )
            if games <= collect.games:
                raise ValueError(
                    "calibration cannot support an expanded teacher workload"
                )
            for index in range(checkpoints):
                collection = collect.model_copy(deep=True)
                collection.id = f"collect-{index}"
                collection.games = games
                collection.execution.wall_seconds = (
                    segment * collect_rate / (collect_rate + checkpoints * fit_rate)
                )
                training = fit.model_copy(deep=True)
                training.id = f"policy-{index}"
                training.initial = f"policy-{index - 1}" if index else None
                training.datasets = [f"collect-{i}" for i in range(index + 1)]
                training.execution.wall_seconds = (
                    segment - collection.execution.wall_seconds
                )
                stages.extend((collection, training))
            predicted.append(games * (collect_rate + fit_rate))
        recipe.stages = stages
        recipes.append(TrainingRegime.model_validate(recipe.model_dump()))
    expected = 5 if screen else 2
    if len(recipes) != expected or len({r.id for r in recipes}) != expected:
        raise ValueError("calibration must contain exactly one run per study arm")
    # Charge each measured anchor separately, including process startup and replay.
    costs = {}
    for cell in study["comparisons"]:
        if not cell["replay"]["passed"] or len(cell["rows"]) != cell["scheduled_games"]:
            raise ValueError("calibration arena cohort is incomplete")
        opponent = cell["b"] if cell["b"].endswith("-anchor") else "paired"
        costs[opponent] = max(
            costs.get(opponent, 0),
            cell["evaluation_seconds"] / (cell["scheduled_games"] / 4),
        )
    names = (
        "random-smoke-anchor",
        "scripted-greedy-fixed-anchor",
        "puct-64-fixed-anchor",
    )
    if any(name not in costs for name in (*names, "paired")):
        raise ValueError("calibration must measure all three anchors and paired games")
    seeds = (601, 1601, 2601) if screen else (5601, 6601, 7601)
    endpoint_cells = 12 if screen else 9
    # Scale all pre-scoring counts together; never select by measured score.
    for dev in range(16, 0, -1):
        endpoint_anchor = 2 * dev
        endpoint_pair = (2 if screen else 8) * dev
        estimate = (
            1.25
            * (
                checkpoints
                * 3
                * dev
                * (
                    (expected - 1) * costs["paired"]
                    + expected * sum(costs[n] for n in names)
                )
                + endpoint_cells * endpoint_pair * costs["paired"]
                + expected * 3 * endpoint_anchor * sum(costs[n] for n in names)
            )
            + 600
        )
        if estimate <= evaluation_budget:
            break
    else:
        raise ValueError("even the smallest evaluation cohort exceeds its allocation")
    family = 1000000 if screen else 1200000
    endpoint = 1400000 if screen else 1600000
    # Interior cutoffs are conservative predictions. Actual support is checked by
    # analysis; a slow/fast run never fabricates a point at a nominal budget.
    first = max(predicted) * 1.1
    last = min(predicted) * checkpoints * 0.9
    if last <= first:
        raise ValueError("calibration predicts no shared observed cost window")
    cutoffs = tuple(
        first + (last - first) * i / (checkpoints - 1) for i in range(checkpoints)
    )
    # Conservatively scale complete calibration bytes by workload growth. This
    # includes SQLite, diagnostics, checkpoints, shards, traces and notebooks;
    # it deliberately does not assume compression or delete retained evidence.
    factors = {}
    for old, new in zip(runs, recipes):
        old_stages = old["regime"]["stages"]
        variable = 1.0
        games_factor = 1.0
        for field in ("games", "updates"):
            old_count = sum(s.get(field, 0) for s in old_stages)
            new_count = sum(getattr(s, field, 0) for s in new.stages)
            if old_count:
                variable = max(variable, 3 * new_count / old_count)
                if field == "games":
                    games_factor = 3 * new_count / old_count
        entry = next(
            e
            for e in study["runs"]
            if json.loads(Path(e["path"]).read_text())["regime"]["id"] == new.id
        )
        factors[Path(entry["path"]).parent] = (variable, games_factor)
    scheduled_blocks = (
        checkpoints * 3 * dev * (expected - 1 + expected * 3)
        + endpoint_cells * endpoint_pair
        + expected * 3 * endpoint_anchor * 3
    )
    old_blocks = sum(c["scheduled_games"] / 4 for c in study["comparisons"])
    evaluation_growth = scheduled_blocks / old_blocks
    measured = {"shards": 0, "checkpoints": 0, "training_records": 0, "arena_report": 0}
    projected = dict.fromkeys(measured, 0.0)
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        owner = next((root for root in factors if path.is_relative_to(root)), None)
        if owner and path.suffix == ".npz":
            category, factor = "shards", factors[owner][1]
        elif owner and path.suffix == ".pt":
            category, factor = "checkpoints", 3 * checkpoints / 2
        elif owner or path.suffix == ".sqlite":
            category, factor = "training_records", max(v[0] for v in factors.values())
        else:
            category, factor = "arena_report", evaluation_growth
        size = path.stat().st_size
        measured[category] += size
        projected[category] += size * max(1, factor)
    projected_bytes = int(1.25 * sum(projected.values()))
    reserve = 4 * 1024**3 + reserved_disk_bytes
    free = shutil.disk_usage(out).free
    if projected_bytes + reserve > free:
        raise ValueError(
            f"storage infeasible: projected {projected_bytes} bytes + reserve {reserve} > free {free}; archive retained evidence elsewhere or amend workload before scoring"
        )
    protocol = EvaluationProtocol(
        study=study["study"],
        purpose="scientific",
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in recipes
        ),
        training_seeds=seeds,
        checkpoint_count=checkpoints,
        cost_cutoffs_seconds=cutoffs,
        paired_deals=tuple(range(family, family + dev)),
        anchor_deals=tuple(range(family + 100000, family + 100000 + dev)),
        endpoint_paired_deals=tuple(range(endpoint, endpoint + endpoint_pair)),
        endpoint_anchor_deals=tuple(
            range(endpoint + 100000, endpoint + 100000 + endpoint_anchor)
        ),
        endpoint_seed_pairs=tuple((s, s) for s in seeds)
        if screen
        else tuple((a, b) for a in seeds for b in seeds),
        anchors=("random", "scripted-greedy", "puct-64"),
        process_seconds=allocation,
        uncertainty="paired-seed-descriptive",
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=tuple(r.model_dump(mode="json") for r in recipes),
        runtime_identities=identities,
        projected_disk_bytes=projected_bytes,
        disk_reserve_bytes=4 * 1024**3,
        allocation_seconds=allocation,
        prior_campaign_seconds=prior_seconds,
        calibration_evidence=f"{out}; study SHA256 {hashlib.sha256((out / 'study.json').read_bytes()).hexdigest()}; projected evaluation including 25% margin and report allowance {estimate:.1f}s; count extrapolation, not a guaranteed runtime; measured storage bytes {measured}; projected bytes with 25% margin {projected_bytes}",
    )
