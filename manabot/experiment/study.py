"""Bounded arena comparisons of training regimes, from frozen protocol to report.

A regime study trains each arm (a recipe function) for every training seed,
then plays each checkpoint against the first arm and against fixed anchors on
reserved deals, and reports scores at shared observed training cost.

- `EvaluationProtocol` freezes seeds, deals, anchors, checkpoints and limits;
  `ResolvedStudy` binds it to resolved recipes and an allocation.
- `smoke_recipe`, `calibration_plan` and `scientific_plan` resolve the three
  profiles: a seconds-long smoke run, a CPU timing cohort, and a count-based
  workload frozen from a completed calibration.
- `run_study` executes a study and retains every attempt; `main` is the command
  line each experiment module exposes (`python -m experiments.<name>`).

Study outputs are files under one directory: `study.json` owns status, runs,
comparisons and measurements; `manabot.reports.analysis.report` renders them.
"""

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from manabot.env import ObservationSpace
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import runtime_fingerprints, source_bundle_sha256
from manabot.model.checkpoint import load_checkpoint
from manabot.reports.analysis import report, valid_game, verify_saved_inputs
from manabot.sim.matchup import SELECTED_SUITE
from manabot.sim.registration import PlayerRegistration, SearchSemantics
from manabot.sim.search.branch import SELECTED_BRANCH_DRIVER_ID
from manabot.training.execution import execute_regime
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import (
    TrainCompound,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]

Study = Literal["ataraxos-ablations", "compound-decisions"]
Arm = Callable[[], TrainingRegime]
# Arms per study; the protocol binds one resolved recipe digest per arm.
_ARM_COUNTS: dict[Study, int] = {
    "ataraxos-ablations": 5,
    "compound-decisions": 4,
}
_IDENTITY_KEYS = (
    "engine_extension_sha256",
    "engine_source_sha256",
    "training_source_sha256",
    "content_manifest_sha256",
    "observation_abi_sha256",
    "action_abi_sha256",
    "matchup_sha256",
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class EvaluationProtocol(BaseModel):
    """Frozen evaluation: seeds, deal families, anchors, checkpoints and limits."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[2] = 2
    study: Study
    purpose: Literal["workflow-smoke", "calibration", "scientific"] = "workflow-smoke"
    evaluation_variants: tuple[Literal["raw"], ...] = ("raw",)
    regime_digests: tuple[str, ...]
    training_seeds: tuple[int, ...] = (197,)
    paired_deals: tuple[int, ...] = (910001,)
    anchor_deals: tuple[int, ...] = (920001,)
    checkpoint_count: int = Field(default=2, ge=2)
    cost_cutoffs_seconds: tuple[float, ...] = ()
    anchors: tuple[Literal["random", "scripted-greedy", "puct-64"], ...] = ("random",)
    endpoint_paired_deals: tuple[int, ...] = ()
    endpoint_anchor_deals: tuple[int, ...] = ()
    endpoint_seed_pairs: tuple[tuple[int, int], ...] = ()
    seat_deck_legs: Literal[4] = 4
    inference: Literal["stochastic-policy-cpu-one-thread"] = (
        "stochastic-policy-cpu-one-thread"
    )
    game_seconds: float = Field(default=120, gt=0, le=120)
    max_commands: int = Field(default=10000, gt=0, le=10000)
    process_seconds: float = Field(default=900, gt=0, le=168 * 3600)
    selection: Literal["all-completed-cutoffs-raw"] = "all-completed-cutoffs-raw"
    uncertainty: Literal["cross-seed-unavailable", "paired-seed-descriptive"] = (
        "cross-seed-unavailable"
    )
    incomplete: Literal["fail-retain-all-attempts"] = "fail-retain-all-attempts"

    @model_validator(mode="after")
    def _consistent(self) -> "EvaluationProtocol":
        if (
            any(c <= 0 for c in self.cost_cutoffs_seconds)
            or tuple(sorted(set(self.cost_cutoffs_seconds)))
            != self.cost_cutoffs_seconds
        ):
            raise ValueError("cost cutoffs must be positive and strictly increasing")
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("training seeds must be nonempty and unique")
        if any(seed < 0 or seed >= 100000 for seed in self.training_seeds):
            raise ValueError("training seeds must be in [0, 100000)")
        if self.purpose == "workflow-smoke" and (
            self.process_seconds > 900 or len(self.training_seeds) != 1
        ):
            raise ValueError("smoke is limited to 900 seconds and one seed")
        if not self.anchors or len(set(self.anchors)) != len(self.anchors):
            raise ValueError("anchors must be nonempty and unique")
        if self.purpose == "scientific":
            if not self.cost_cutoffs_seconds:
                raise ValueError("scientific profiles require declared cost cutoffs")
            if (
                len(self.training_seeds) < 3
                or self.uncertainty != "paired-seed-descriptive"
            ):
                raise ValueError(
                    "scientific profiles require at least three seeds and declared uncertainty"
                )
            if not self.endpoint_paired_deals or not self.endpoint_anchor_deals:
                raise ValueError("scientific profiles require untouched endpoint deals")
            if set(self.anchors) != {"random", "scripted-greedy", "puct-64"}:
                raise ValueError(
                    "scientific profiles require all three fixed baselines"
                )
            expected_pairs = {(s, s) for s in self.training_seeds}
            if set(self.endpoint_seed_pairs) != expected_pairs or len(
                self.endpoint_seed_pairs
            ) != len(expected_pairs):
                raise ValueError(
                    "endpoint seed schedule must cover the declared cross-seed comparison"
                )
        families = (
            self.paired_deals,
            self.anchor_deals,
            self.endpoint_paired_deals,
            self.endpoint_anchor_deals,
        )
        flat = [s for family in families for s in family]
        if len(flat) != len(set(flat)):
            raise ValueError(
                "development and endpoint deal families must be disjoint and unique"
            )
        if any(s < 900000 for s in flat):
            raise ValueError("evaluation deals must use reserved family >=900000")
        if len(self.regime_digests) != _ARM_COUNTS[self.study] or any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in self.regime_digests
        ):
            raise ValueError("protocol must bind every resolved recipe digest")
        if not self.paired_deals or not self.anchor_deals:
            raise ValueError("evaluation deal families must be nonempty")
        return self


class ResolvedStudy(BaseModel):
    """Explicit recipes and allocation; the runner never scales smoke implicitly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: EvaluationProtocol
    # Serialized TrainingRegime values, validated against the protocol digests.
    recipes: tuple[dict[str, Any], ...]
    allocation_seconds: float = Field(gt=0, le=168 * 3600)
    prior_campaign_seconds: float = Field(ge=0)
    runtime_identities: dict[str, str] = Field(default_factory=dict)
    projected_disk_bytes: int = Field(default=0, ge=0)
    disk_reserve_bytes: int = Field(default=4 * 1024**3, ge=0)
    calibration_evidence: str

    @model_validator(mode="after")
    def _allocation(self) -> "ResolvedStudy":
        recipes = [TrainingRegime.model_validate(r) for r in self.recipes]
        if (
            tuple(canonical_sha256(r.model_dump(mode="json")) for r in recipes)
            != self.protocol.regime_digests
        ):
            raise ValueError("resolved plan recipe digests do not match")
        if len({r.id for r in recipes}) != len(recipes):
            raise ValueError("recipe IDs must be unique")
        if any(len(r.stages) != self.protocol.checkpoint_count for r in recipes):
            raise ValueError("every recipe must export the declared checkpoint count")
        if self.protocol.process_seconds > self.allocation_seconds:
            raise ValueError("process deadline exceeds study allocation")
        if self.prior_campaign_seconds + self.allocation_seconds > 168 * 3600:
            raise ValueError(
                "campaign exceeds 168 laptop hours including prior attempts"
            )
        if (
            sum(r.wall_seconds for r in recipes) * len(self.protocol.training_seeds)
            >= self.allocation_seconds
        ):
            raise ValueError("allocation must leave time for evaluation and reports")
        if (
            self.protocol.purpose == "scientific"
            and not self.calibration_evidence.strip()
        ):
            raise ValueError("scientific execution requires calibration evidence")
        return self


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def smoke_recipe(recipe: TrainingRegime) -> TrainingRegime:
    """Shrink a fresh recipe to a seconds-long CPU smoke run, revalidated."""
    recipe.agent.hidden_dim = 16
    recipe.agent.num_attention_heads = 2
    for stage in recipe.stages:
        if isinstance(stage, TrainCompound):
            stage.games_per_update = 1
            stage.updates = 1
            stage.learning.epochs = 1
        elif isinstance(stage, TrainSelfPlay):
            stage.transitions = 64
            stage.updates = 2
            stage.learning.epochs = 1
    return TrainingRegime.model_validate(recipe.model_dump())


def calibration_plan(study: Study, arms: Sequence[Arm]) -> ResolvedStudy:
    """Resolve a bounded CPU timing cohort without starting training."""
    if study == "compound-decisions":
        raise ValueError(
            "compound scoring needs its own measured protocol and authorized allocation"
        )
    recipes = []
    for build in arms:
        recipe = build()
        recipe.wall_seconds = 600
        for stage in recipe.stages:
            stage.execution.wall_seconds = 300
            if isinstance(stage, TrainSelfPlay):
                stage.updates = 8
                stage.transitions = 256
        recipes.append(recipe)
    protocol = EvaluationProtocol(
        study=study,
        purpose="calibration",
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in recipes
        ),
        training_seeds=(397,),
        paired_deals=(930001,),
        anchor_deals=(940001,),
        process_seconds=3600,
        anchors=("random", "scripted-greedy", "puct-64"),
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=tuple(r.model_dump(mode="json") for r in recipes),
        allocation_seconds=3600,
        prior_campaign_seconds=0,
        calibration_evidence="Initial integrated CPU throughput calibration; no strength inference",
    )


def scientific_plan(calibration: Path | str, prior_seconds: float) -> ResolvedStudy:
    """Reserve the ablation screen inside 168 hours; extrapolation is not timing proof."""
    out = Path(calibration).resolve()
    study = json.loads((out / "study.json").read_text())
    if study["study"] == "compound-decisions":
        raise ValueError(
            "compound study needs a separate allocation; the ETU-91 campaign budget cannot be reused"
        )
    verify_saved_inputs(out, study)
    if study["profile"] != "calibration" or study["status"] != "completed":
        raise ValueError("a completed integrated calibration cohort is required")
    checkpoints = 3
    run_budget = 3600
    evaluation_budget = 6 * 3600
    allocation = 21 * 3600
    # Reserve four hours of calibration and eleven of recovery.
    prior_seconds = max(prior_seconds, 4 * 3600)
    if prior_seconds + allocation > 157 * 3600:
        raise ValueError(
            "campaign has insufficient budget; amend allocations before scoring"
        )
    runs = [json.loads(Path(r["path"]).read_text()) for r in study["runs"]]
    identities = {k: runs[0]["identities"][k] for k in _IDENTITY_KEYS}
    if any({k: r["identities"][k] for k in _IDENTITY_KEYS} != identities for r in runs):
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
        if not isinstance(recipe.stages[0], TrainSelfPlay):
            raise ValueError("ablation calibration requires self-play arms")
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
        recipe.stages = stages
        recipes.append(TrainingRegime.model_validate(recipe.model_dump()))
    expected = 5
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
    seeds = (601, 1601, 2601)
    endpoint_cells = 12
    # Scale all pre-scoring counts together; never select by measured score.
    for dev in range(16, 0, -1):
        endpoint_anchor = 2 * dev
        endpoint_pair = 2 * dev
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
    family = 1000000
    endpoint = 1400000
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
    reserve = 4 * 1024**3
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
        endpoint_seed_pairs=tuple((s, s) for s in seeds),
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


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def registration(
    run: TrainingRun, stage: StageRecord, variant: str = "raw"
) -> PlayerRegistration:
    artifact = stage.artifacts[variant]
    agent, _ = load_checkpoint(artifact["path"])
    identity = run.identities
    return PlayerRegistration(
        # Arena names are separate from immutable training recipe identities.
        player_id=f"{run.regime.id}-{run.seed}-{stage.id}-{variant}".replace("_", "-"),
        display_name=run.regime.id,
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
        },
        compute_class_id="compound-cpu-one-thread-autoregressive"
        if run.regime.agent.compound_decisions
        else "policy-cpu-one-thread-one-pass",
        information_boundary="acting-viewer",
        world=run.regime.world,
        content_suite=SELECTED_SUITE,
        observation_abi_sha256=identity["observation_abi_sha256"],
        action_abi_sha256=identity["action_abi_sha256"],
        matchup_sha256=identity["matchup_sha256"],
        checkpoint_sha256=artifact["sha256"],
        checkpoint_bytes=artifact["bytes"],
        parameter_count=sum(p.numel() for p in agent.parameters()),
        training_seed=run.seed,
        artifact_id=f"{run.id}/{stage.id}/{variant}",
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )


def run_study(
    study: Study,
    arms: Sequence[Arm],
    out: Path,
    plan: ResolvedStudy | None = None,
    resume: bool = False,
    *,
    render_report: bool = True,
) -> None:
    """Train every arm per seed, play the frozen arena cohort and retain it all.

    Without a plan this is the smoke profile. A retained interrupted study
    resumes only when every training run completed; evaluation cells already
    played are skipped. Failures mark `study.json` failed and re-raise.
    """
    if plan is not None and plan.protocol.purpose == "scientific":
        free = shutil.disk_usage(out.parent).free
        if free < plan.projected_disk_bytes + plan.disk_reserve_bytes:
            raise RuntimeError(
                "insufficient free disk for frozen study and evidence reserve"
            )
    if plan is not None and plan.protocol.study != study:
        raise ValueError("study differs from resolved protocol")
    retained = None
    resumed_runs = []
    if resume:
        retained = json.loads((out / "study.json").read_text())
        verify_saved_inputs(out, retained)
        if retained["status"] == "running":
            raise ValueError(
                "unclean shutdown has no final cost receipt; use report-only and account for the lost interval before a new cohort"
            )
        if retained["status"] == "completed":
            raise ValueError("study already completed; use --report-only")
        for entry in retained["runs"]:
            run = TrainingRun.model_validate_json(Path(entry["path"]).read_text())
            if run.status != "completed":
                raise ValueError(
                    "interrupted training cannot resume; retain this attempt and freeze a new cohort"
                )
            resumed_runs.append(run)
    else:
        out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter() - (retained["seconds"] if retained else 0.0)
    if plan is None:
        recipes = [smoke_recipe(build()) for build in arms]
        protocol = EvaluationProtocol(
            study=study,
            regime_digests=tuple(
                canonical_sha256(r.model_dump(mode="json")) for r in recipes
            ),
        )
    else:
        protocol = plan.protocol
        recipes = [TrainingRegime.model_validate(r) for r in plan.recipes]
        if [r.id for r in recipes] != [build().id for build in arms]:
            raise ValueError("resolved recipes must preserve the study arm order")
        if not resume:
            atomic_json(out / "resolved-plan.json", plan.model_dump(mode="json"))
    if len({recipe.id.replace("_", "-") for recipe in recipes}) != len(recipes):
        raise ValueError("recipe IDs collide after arena normalization")
    if resume or (plan is not None and plan.protocol.purpose == "scientific"):
        current = runtime_fingerprints(
            protocol.training_seeds[0],
            match_config=recipes[0].match,
            observation_space=ObservationSpace(recipes[0].observation),
        )
        current["training_source_sha256"] = source_bundle_sha256(
            sorted((ROOT / "manabot").rglob("*.py"))
        )
        expected = (
            plan.runtime_identities
            if plan and plan.protocol.purpose == "scientific"
            else {k: resumed_runs[0].identities[k] for k in _IDENTITY_KEYS}
        )
        if not expected or any(current.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "runtime/source differs from calibrated or retained cohort"
            )
    if resume:
        if retained["protocol_sha256"] != canonical_sha256(
            protocol.model_dump(mode="json")
        ):
            raise ValueError("resume protocol differs from frozen cohort")
        if {(r.regime.id, r.seed) for r in resumed_runs} != {
            (r.id, seed) for r in recipes for seed in protocol.training_seeds
        }:
            raise ValueError(
                "resume requires every frozen training run to have completed"
            )
    else:
        atomic_json(out / "recipes.json", [r.model_dump(mode="json") for r in recipes])
        atomic_json(out / "protocol.json", protocol.model_dump(mode="json"))
    result = {
        "study": study,
        "profile": protocol.purpose,
        "status": "running",
        "runs": [],
        "measurements": [],
        "comparisons": [],
        "seeds": protocol.training_seeds,
        "protocol_sha256": canonical_sha256(protocol.model_dump(mode="json")),
        "limits": f"{protocol.process_seconds} seconds total; {len(protocol.training_seeds)} training seeds; {protocol.uncertainty}; purpose={protocol.purpose}",
    }

    if retained:
        result = retained
        result.setdefault("interruptions", []).append(
            {
                "seconds": result["seconds"],
                "error": result.pop("error", "process stopped"),
            }
        )
        result["status"] = "running"
    remaining = protocol.process_seconds - (time.perf_counter() - start)
    if remaining <= 0:
        raise TimeoutError("frozen study allocation is exhausted; cannot resume")

    def save() -> None:
        result["seconds"] = time.perf_counter() - start
        atomic_json(out / "study.json", result)

    def deadline(*_: object) -> None:
        raise TimeoutError("study exceeded its frozen process deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    save()
    try:
        runs = resumed_runs
        with VerifyStore(out / "training.sqlite") as store:
            for index, seed in enumerate(() if resume else protocol.training_seeds):
                # Alternate arm order per seed so no arm always trains first.
                ordered = recipes if index % 2 == 0 else list(reversed(recipes))
                for recipe in ordered:
                    entry = {
                        "path": str(out / f"{recipe.id}-seed-{seed}" / "run.json"),
                        "regime": recipe.id,
                    }
                    result["runs"].append(entry)
                    save()
                    run = execute_regime(
                        recipe, seed, out / f"{recipe.id}-seed-{seed}", store
                    )
                    runs.append(run)
                    entry.update(
                        id=run.id,
                        sha256=hashlib.sha256(
                            Path(entry["path"]).read_bytes()
                        ).hexdigest(),
                    )
                    save()
        checkpoints = {
            run.id: [s for s in run.stages if "raw" in s.artifacts] for run in runs
        }
        identity = runs[0].identities
        anchor = PlayerRegistration(
            player_id="random-smoke-anchor",
            display_name="Uniform random",
            role="anchor",
            runner_kind="code",
            player_spec={"kind": "random"},
            source_sha256=canonical_sha256(
                {
                    "engine": identity["engine_source_sha256"],
                    "manabot": identity["training_source_sha256"],
                }
            ),
            compute_class_id="random-cpu",
            information_boundary="acting-viewer",
            world=runs[0].regime.world,
            content_suite=SELECTED_SUITE,
            observation_abi_sha256=identity["observation_abi_sha256"],
            action_abi_sha256=identity["action_abi_sha256"],
            matchup_sha256=identity["matchup_sha256"],
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )

        anchors = []
        for name in protocol.anchors:
            spec = {"kind": "random" if name == "random" else "scripted_greedy"}
            extra = {}
            if name == "puct-64":
                spec = dict(
                    kind="determinized_puct",
                    sims=64,
                    worlds=4,
                    c_puct=1.5,
                    max_steps=2000,
                    branch_driver_id=SELECTED_BRANCH_DRIVER_ID,
                )
                extra = dict(
                    search_semantics=SearchSemantics(
                        root_prior="uniform-v1",
                        leaf_evaluator="uniform-random-terminal-v1",
                    ),
                    search_call_seed_derivation_id="mcts-mix-player-seed-decision-v1",
                )
            anchors.append(
                PlayerRegistration.model_validate(
                    {
                        **anchor.model_dump(),
                        "player_id": "random-smoke-anchor"
                        if name == "random"
                        else f"{name}-fixed-anchor",
                        "display_name": name,
                        "player_spec": spec,
                        "compute_class_id": f"{name}-cpu",
                        **extra,
                    }
                )
            )

        key = EvaluationKey(
            world=runs[0].regime.world,
            content_suite=SELECTED_SUITE,
            viewer_boundary="acting-viewer",
            arena_version="training-regime-smoke-v1",
            rating_model_version="unrated",
            rating_prior_sha256=canonical_sha256({}),
            anchor_cohort_sha256=canonical_sha256(
                [a.model_dump(mode="json") for a in anchors]
            ),
            evaluation_compute_envelope_id="compound-cpu-one-thread-autoregressive"
            if study == "compound-decisions"
            else "policy-cpu-one-thread-one-pass",
        )

        def compare_variant(
            a_run: TrainingRun,
            a_stage: StageRecord,
            b_run: TrainingRun | None,
            b_stage: StageRecord | None,
            cutoff: int,
            *,
            baseline: PlayerRegistration | None = None,
            phase: str = "development",
            variant: str = "raw",
        ) -> None:
            if b_run is None:
                assert baseline is not None
            else:
                assert b_stage is not None
            identity = (
                a_run.regime.id,
                a_run.seed,
                b_run.regime.id if b_run else baseline.player_id,
                b_run.seed if b_run else None,
                cutoff,
                phase,
                variant,
            )
            if any(
                (
                    c["a"],
                    c["training_seed"],
                    c["b"],
                    c.get("b_training_seed"),
                    c["cutoff"],
                    c.get("phase", "development"),
                    c.get("variant", "raw"),
                )
                == identity
                for c in result["comparisons"]
            ):
                return
            a = registration(a_run, a_stage, variant)
            b = registration(b_run, b_stage, variant) if b_run else baseline
            deals = (
                (protocol.paired_deals if b_run else protocol.anchor_deals)
                if phase == "development"
                else (
                    protocol.endpoint_paired_deals
                    if b_run
                    else protocol.endpoint_anchor_deals
                )
            )
            arena_out = out / f"arena-{len(result['comparisons'])}"
            arena_out.mkdir()
            paths = {a.player_id: a_stage.artifacts[variant]["path"]}
            if b_run:
                paths[b.player_id] = b_stage.artifacts[variant]["path"]
            atomic_json(
                arena_out / "registrations.json", [a.model_dump(), b.model_dump()]
            )
            comparison = dict(
                cutoff=cutoff,
                a=a_run.regime.id,
                training_seed=a_run.seed,
                b_training_seed=b_run.seed if b_run else None,
                phase=phase,
                variant=variant,
                b=b_run.regime.id if b_run else b.player_id,
                rows=[],
                scheduled_games=4 * len(deals),
                trace=None,
                replay={"passed": False},
                evaluation_seconds=0,
                path=str(arena_out),
            )
            result["comparisons"].append(comparison)
            save()
            tick = time.perf_counter()
            try:
                rows, trace, replay = play_cell(
                    key=key,
                    player_a=a,
                    player_b=b,
                    deal_seeds=deals,
                    game_seconds=protocol.game_seconds,
                    max_commands=protocol.max_commands,
                    out_dir=arena_out,
                    checkpoint_paths=paths,
                    comparison_seed_aliases={
                        a.player_id: "candidate",
                        b.player_id: "reference",
                    },
                )
                atomic_json(arena_out / "rows.json", rows)
                comparison.update(rows=rows, trace=trace, replay=replay)
            finally:
                comparison["evaluation_seconds"] = time.perf_counter() - tick
                save()
            valid = (
                replay["passed"]
                and len(rows) == 4 * len(deals)
                and all(valid_game(row) for row in rows)
            )
            for run, stage, is_a in ((a_run, a_stage, True), (b_run, b_stage, False)):
                if run is None:
                    continue
                cumulative = run.stages[: run.stages.index(stage) + 1]
                result["measurements"].append(
                    dict(
                        regime=run.regime.id,
                        seed=run.seed,
                        variant=variant,
                        phase=phase,
                        checkpoint=stage.artifacts[variant],
                        cutoff=cutoff,
                        opponent=comparison["b"] if is_a else comparison["a"],
                        training_seconds=stage.cumulative_seconds,
                        collection_seconds=sum(
                            s.collection_seconds for s in cumulative
                        ),
                        learning_seconds=sum(s.learning_seconds for s in cumulative),
                        export_seconds=sum(s.export_seconds for s in cumulative),
                        decisions=sum(s.environment_decisions for s in cumulative),
                        learner_transitions=sum(
                            s.learner_transitions for s in cumulative
                        ),
                        optimizer_exposures=sum(
                            s.optimizer_exposures for s in cumulative
                        ),
                        games=sum(s.games for s in cumulative),
                        **(
                            {"compound_accounting": [s.diagnostics for s in cumulative]}
                            if run.regime.agent.compound_decisions
                            else {}
                        ),
                        score=sum(
                            r["score_a"] if is_a else 1 - r["score_a"] for r in rows
                        )
                        / len(rows)
                        if valid
                        else None,
                        complete=valid,
                    )
                )
            save()

        def compare(
            a_run: TrainingRun,
            a_stage: StageRecord,
            b_run: TrainingRun | None,
            b_stage: StageRecord | None,
            cutoff: int,
            *,
            baseline: PlayerRegistration | None = None,
            phase: str = "development",
        ) -> None:
            for variant in protocol.evaluation_variants:
                compare_variant(
                    a_run,
                    a_stage,
                    b_run,
                    b_stage,
                    cutoff,
                    baseline=baseline,
                    phase=phase,
                    variant=variant,
                )

        if any(
            len(stages) != protocol.checkpoint_count for stages in checkpoints.values()
        ):
            raise ValueError("study checkpoint count differs from the frozen protocol")
        for cutoff in range(protocol.checkpoint_count):
            for seed in protocol.training_seeds:
                paired = [
                    next(r for r in runs if r.seed == seed and r.regime.id == recipe.id)
                    for recipe in recipes
                ]
                for candidate in paired[1:] if protocol.paired_deals else ():
                    compare(
                        paired[0],
                        checkpoints[paired[0].id][cutoff],
                        candidate,
                        checkpoints[candidate.id][cutoff],
                        cutoff,
                    )
                for run in paired:
                    for baseline in anchors:
                        compare(
                            run,
                            checkpoints[run.id][cutoff],
                            None,
                            None,
                            cutoff,
                            baseline=baseline,
                        )
        if protocol.endpoint_seed_pairs:
            for a_seed, b_seed in protocol.endpoint_seed_pairs:
                reference = next(
                    r for r in runs if r.regime.id == recipes[0].id and r.seed == a_seed
                )
                for recipe in recipes[1:]:
                    candidate = next(
                        r for r in runs if r.regime.id == recipe.id and r.seed == b_seed
                    )
                    compare(
                        reference,
                        checkpoints[reference.id][-1],
                        candidate,
                        checkpoints[candidate.id][-1],
                        protocol.checkpoint_count - 1,
                        phase="endpoint",
                    )
            for run in runs:
                for baseline in anchors:
                    compare(
                        run,
                        checkpoints[run.id][-1],
                        None,
                        None,
                        protocol.checkpoint_count - 1,
                        baseline=baseline,
                        phase="endpoint",
                    )
        incomplete = any(
            not c["replay"]["passed"]
            or len(c["rows"]) != c["scheduled_games"]
            or not all(valid_game(row) for row in c["rows"])
            for c in result["comparisons"]
        )
        result["status"] = "failed" if incomplete else "completed"
        if incomplete:
            result["error"] = (
                "incomplete arena cohort; every attempted cell retained without retry"
            )
        save()
        if render_report:
            report(out)
        if incomplete:
            raise RuntimeError(result["error"])
        save()
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main(study: Study, arms: Sequence[Arm], argv: Sequence[str] | None = None) -> None:
    """Command line for one study: smoke by default, or a frozen plan's profile."""
    parser = argparse.ArgumentParser(description=f"Run or report the {study} study.")
    parser.add_argument(
        "--profile", choices=["smoke", "calibration", "scientific"], default="smoke"
    )
    parser.add_argument("--plan", type=Path, help="Frozen ResolvedStudy JSON")
    parser.add_argument("--write-plan", type=Path, help="Resolve a plan and stop")
    parser.add_argument("--calibration", type=Path, help="Completed calibration study")
    parser.add_argument("--prior-campaign-seconds", type=float, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", type=Path, help="Re-render a retained study")
    args = parser.parse_args(argv)
    if args.write_plan:
        if (
            args.profile == "smoke"
            or args.plan
            or args.out
            or args.report_only
            or args.resume
        ):
            parser.error(
                "--write-plan requires calibration/scientific and no execution arguments"
            )
        if args.write_plan.exists():
            parser.error("refusing to overwrite an existing plan")
        if args.profile == "scientific":
            if args.calibration is None:
                parser.error("scientific plan generation requires --calibration")
            plan = scientific_plan(args.calibration, args.prior_campaign_seconds)
            if plan.protocol.study != study:
                parser.error("calibration belongs to a different study")
        else:
            plan = calibration_plan(study, arms)
            plan = ResolvedStudy.model_validate(
                {
                    **plan.model_dump(),
                    "prior_campaign_seconds": args.prior_campaign_seconds,
                }
            )
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.write_plan, plan.model_dump(mode="json"))
        return
    if args.calibration or args.prior_campaign_seconds:
        parser.error("calibration evidence/accounting arguments require --write-plan")
    if args.report_only:
        report(args.report_only.resolve())
        return
    if args.out is None:
        parser.error("--out is required")
    plan = None
    if args.profile != "smoke":
        if args.plan is None:
            parser.error(
                "calibration/scientific profiles require --plan with resolved recipes"
            )
        plan = ResolvedStudy.model_validate_json(args.plan.read_text())
        if plan.protocol.purpose != args.profile or plan.protocol.study != study:
            parser.error("profile/study differs from frozen plan")
    elif args.plan is not None:
        parser.error("smoke resolves its own recipes; --plan needs a profile")
    torch.set_num_threads(1)
    run_study(study, arms, args.out.resolve(), plan, resume=args.resume)
