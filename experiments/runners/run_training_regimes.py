"""Run bounded regime studies or regenerate their saved analysis offline."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import torch

from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.match import SELECTED_SUITE, play_cell
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    SearchSemantics,
    canonical_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.analysis import report, valid_game, verify_saved_inputs
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import (
    CollectSearch,
    StageRecord,
    TrainCompound,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {
    "model-capacity": [],
    "omitted-controls": [],
    "compound-decisions": [
        "compound-sequential-bootstrap",
        "compound-grouped-bootstrap",
        "compound-sequential-outcome",
        "compound-grouped-outcome",
    ],
    "learning-speed": ["search-distillation", "direct-self-play"],
    "ataraxos-ablations": [
        "rl-control",
        "separate-estimators",
        "advantage-filtering",
        "coordinated-decay",
        "combined",
    ],
}


def registration(
    run: TrainingRun, stage: StageRecord, variant: str = "raw"
) -> PlayerRegistration:
    artifact = stage.artifacts[variant]
    agent, _ = load_checkpoint_agent(artifact["path"])
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


def smoke_recipe(name: str) -> TrainingRegime:
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{name}.json").read_text()
    )
    recipe.agent.hidden_dim = 16
    recipe.agent.num_attention_heads = 2
    for stage in recipe.stages:
        if isinstance(stage, CollectSearch):
            stage.simulations = 4
            stage.worlds = 1
        elif isinstance(stage, TrainCompound):
            stage.games_per_update = 1
            stage.updates = 1
            stage.learning.epochs = 1
        elif isinstance(stage, TrainSelfPlay):
            stage.transitions = 64
            stage.updates = 2
            stage.learning.epochs = 1
        else:
            stage.epochs = 1
    return TrainingRegime.model_validate(recipe.model_dump())


def calibration_plan(study: str) -> ResolvedStudy:
    """Resolve a bounded CPU timing cohort without starting training."""
    if study == "compound-decisions":
        raise ValueError(
            "compound scoring needs its own measured protocol and authorized allocation"
        )
    recipes = []
    for name in STUDIES[study]:
        recipe = TrainingRegime.model_validate_json(
            (ROOT / "experiments/regimes" / f"{name}.json").read_text()
        )
        recipe.wall_seconds = 600
        for stage in recipe.stages:
            stage.execution.wall_seconds = 300
            if isinstance(stage, TrainSelfPlay):
                stage.updates = 8
                stage.transitions = 256
            elif isinstance(stage, CollectSearch):
                stage.games = 4
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


def run_study(
    study: str,
    out: Path,
    plan: ResolvedStudy | None = None,
    resume: bool = False,
    *,
    render_report: bool = True,
) -> None:
    if "cumulative_seconds" not in StageRecord.model_fields:
        raise RuntimeError(
            "ETU-89 cumulative checkpoint clock must be integrated before study execution"
        )
    if plan is not None and plan.protocol.purpose == "scientific":
        free = shutil.disk_usage(out.parent).free
        if free < plan.projected_disk_bytes + plan.disk_reserve_bytes:
            raise RuntimeError(
                "insufficient free disk for frozen study and evidence reserve"
            )
    explicit_plan = study in {
        "omitted-controls",
        "training-calibration",
        "capacity-calibration",
        "model-capacity",
        "value-models",
    }
    if explicit_plan and plan is None:
        raise ValueError(f"{study} requires an explicit separately resolved plan")
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
    start = time.perf_counter() - (retained["seconds"] if retained else 0)
    if plan is None:
        recipes = [smoke_recipe(name) for name in STUDIES[study]]
        protocol = EvaluationProtocol(
            study=study,
            regime_digests=tuple(
                canonical_sha256(r.model_dump(mode="json")) for r in recipes
            ),
        )
    else:
        protocol = plan.protocol
        recipes = [TrainingRegime.model_validate(r) for r in plan.recipes]
        if not explicit_plan and [r.id for r in recipes] != STUDIES[study]:
            raise ValueError("resolved recipes must preserve the study arm order")
        if not resume:
            atomic_json(out / "resolved-plan.json", plan.model_dump(mode="json"))
    if len({recipe.id.replace("_", "-") for recipe in recipes}) != len(recipes):
        raise ValueError("recipe IDs collide after arena normalization")
    if resume or (plan is not None and plan.protocol.purpose == "scientific"):
        from manabot.env import ObservationSpace
        from manabot.sim.teacher1_evidence import (
            runtime_fingerprints,
            source_bundle_sha256,
        )

        current = runtime_fingerprints(
            protocol.training_seeds[0],
            match_hypers=recipes[0].match,
            observation_space=ObservationSpace(recipes[0].observation),
        )
        current["training_source_sha256"] = source_bundle_sha256(
            sorted((ROOT / "manabot").rglob("*.py"))
        )
        expected = (
            plan.runtime_identities
            if plan and plan.protocol.purpose == "scientific"
            else {
                k: resumed_runs[0].identities[k]
                for k in (
                    "engine_extension_sha256",
                    "engine_source_sha256",
                    "training_source_sha256",
                    "content_manifest_sha256",
                    "observation_abi_sha256",
                    "action_abi_sha256",
                    "matchup_sha256",
                )
            }
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
                for recipe in recipes if index % 2 == 0 else list(reversed(recipes)):
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
                from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID

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

        key = ArenaKey(
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
                for candidate in paired[1:]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, default="learning-speed")
    parser.add_argument(
        "--profile", choices=["smoke", "calibration", "scientific"], default="smoke"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--companion-plan", type=Path)
    parser.add_argument("--prior-campaign-seconds", type=float, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
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
            from experiments.runners.training_plan import scientific_plan

            if args.calibration is None:
                parser.error("scientific plan generation requires --calibration")
            reserved = 0
            if args.companion_plan:
                companion = ResolvedStudy.model_validate_json(
                    args.companion_plan.read_text()
                )
                if companion.protocol.study == args.study:
                    parser.error("companion must be the other study")
                reserved = companion.projected_disk_bytes
            plan = scientific_plan(
                args.calibration, args.prior_campaign_seconds, reserved
            )
            if plan.protocol.study != args.study:
                parser.error("calibration belongs to a different study")
        else:
            plan = calibration_plan(args.study)
            plan = ResolvedStudy.model_validate(
                {
                    **plan.model_dump(),
                    "prior_campaign_seconds": args.prior_campaign_seconds,
                }
            )
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.write_plan, plan.model_dump(mode="json"))
        return
    if args.calibration or args.prior_campaign_seconds or args.companion_plan:
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
        if plan.protocol.purpose != args.profile or plan.protocol.study != args.study:
            parser.error("profile/study differs from frozen plan")
    elif args.plan is not None:
        plan = ResolvedStudy.model_validate_json(args.plan.read_text())
        if (
            args.study not in {"omitted-controls", "model-capacity"}
            or plan.protocol.study != args.study
            or plan.protocol.purpose != "workflow-smoke"
        ):
            parser.error(
                "smoke accepts only an omitted-controls or model-capacity workflow plan"
            )
    if args.study == "model-capacity" and plan is None:
        parser.error("model-capacity requires an explicit --plan")
    torch.set_num_threads(1)
    out = args.out.resolve()
    run_study(args.study, out, plan, resume=args.resume)


if __name__ == "__main__":
    main()
