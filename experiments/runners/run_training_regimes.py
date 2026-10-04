"""Run bounded regime studies or regenerate their saved analysis offline."""

import argparse
import hashlib
from pathlib import Path
import signal
import time

import torch

from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.match import SELECTED_SUITE, play_cell
from manabot.arena.models import ArenaKey, PlayerRegistration, canonical_sha256
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import CollectSearch, TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {
    "learning-speed": ["search-distillation", "direct-self-play"],
    "ataraxos-ablations": [
        "rl-control",
        "separate-estimators",
        "advantage-filtering",
        "coordinated-decay",
        "combined",
    ],
}


def registration(run, stage, variant="raw"):
    artifact = stage.artifacts[variant]
    agent, _ = load_checkpoint_agent(artifact["path"])
    identity = run.identities
    return PlayerRegistration(
        player_id=f"{run.regime.id}-{stage.id}-{variant}",
        display_name=run.regime.id,
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
        },
        compute_class_id="policy-cpu-one-thread-one-pass",
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


def smoke_recipe(name):
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes" / f"{name}.json").read_text()
    )
    recipe.agent.hidden_dim = 16
    recipe.agent.num_attention_heads = 2
    for stage in recipe.stages:
        if isinstance(stage, CollectSearch):
            stage.simulations = 4
            stage.worlds = 1
        elif isinstance(stage, TrainSelfPlay):
            stage.transitions = 64
            stage.updates = 2
            stage.learning.epochs = 1
        else:
            stage.epochs = 1
    return TrainingRegime.model_validate(recipe.model_dump())


def run_study(study, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    recipes = [smoke_recipe(name) for name in STUDIES[study]]
    protocol = EvaluationProtocol(
        study=study,
        regime_digests=tuple(
            canonical_sha256(r.model_dump(mode="json")) for r in recipes
        ),
    )
    atomic_json(out / "recipes.json", [r.model_dump(mode="json") for r in recipes])
    atomic_json(out / "protocol.json", protocol.model_dump(mode="json"))
    result = {
        "study": study,
        "profile": "smoke",
        "status": "running",
        "runs": [],
        "measurements": [],
        "comparisons": [],
        "seed": protocol.training_seed,
        "protocol_sha256": canonical_sha256(protocol.model_dump(mode="json")),
        "limits": "900 seconds total; one training seed; one four-leg deal block per comparison; no strength claim",
    }

    def save():
        result["seconds"] = time.perf_counter() - start
        atomic_json(out / "study.json", result)

    def deadline(*_):
        raise TimeoutError("study exceeded its 15-minute process deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, protocol.process_seconds)
    save()
    try:
        runs = []
        with VerifyStore(out / "training.sqlite") as store:
            for recipe in recipes:
                entry = {"path": str(out / recipe.id / "run.json"), "regime": recipe.id}
                result["runs"].append(entry)
                save()
                run = execute_regime(
                    recipe, protocol.training_seed, out / recipe.id, store
                )
                runs.append(run)
                entry.update(
                    id=run.id,
                    sha256=hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest(),
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

        key = ArenaKey(
            world=runs[0].regime.world,
            content_suite=SELECTED_SUITE,
            viewer_boundary="acting-viewer",
            arena_version="training-regime-smoke-v1",
            rating_model_version="unrated",
            rating_prior_sha256=canonical_sha256({}),
            anchor_cohort_sha256=canonical_sha256([anchor.model_dump(mode="json")]),
            evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
        )

        def compare(a_run, a_stage, b_run, b_stage, cutoff):
            a = registration(a_run, a_stage)
            b = registration(b_run, b_stage) if b_run else anchor
            deals = protocol.paired_deals if b_run else protocol.anchor_deals
            arena_out = out / f"arena-{len(result['comparisons'])}"
            arena_out.mkdir()
            paths = {a.player_id: a_stage.artifacts["raw"]["path"]}
            if b_run:
                paths[b.player_id] = b_stage.artifacts["raw"]["path"]
            atomic_json(
                arena_out / "registrations.json", [a.model_dump(), b.model_dump()]
            )
            comparison = dict(
                cutoff=cutoff,
                a=a_run.regime.id,
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
                and all(
                    r["failure"] is None
                    and r["terminated"]
                    and not r["truncated"]
                    and r["replay_passed"]
                    for r in rows
                )
            )
            for run, stage, is_a in ((a_run, a_stage, True), (b_run, b_stage, False)):
                if run is None:
                    continue
                cumulative = run.stages[: run.stages.index(stage) + 1]
                result["measurements"].append(
                    dict(
                        regime=run.regime.id,
                        seed=run.seed,
                        variant="raw",
                        checkpoint=stage.artifacts["raw"],
                        cutoff=cutoff,
                        opponent=comparison["b"] if is_a else comparison["a"],
                        training_seconds=run.setup_seconds
                        + sum(s.seconds for s in cumulative),
                        collection_seconds=sum(
                            s.collection_seconds for s in cumulative
                        ),
                        learning_seconds=sum(s.learning_seconds for s in cumulative),
                        export_seconds=sum(s.export_seconds for s in cumulative),
                        decisions=sum(s.environment_decisions for s in cumulative),
                        games=sum(s.games for s in cumulative),
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
            if not valid:
                raise RuntimeError(
                    "incomplete arena cohort; retained attempts cannot establish a score"
                )

        if any(
            len(stages) != protocol.checkpoint_count for stages in checkpoints.values()
        ):
            raise ValueError(
                "study requires exactly two completed raw checkpoints per recipe"
            )
        for cutoff in range(protocol.checkpoint_count):
            for candidate in runs[1:]:
                compare(
                    runs[0],
                    checkpoints[runs[0].id][cutoff],
                    candidate,
                    checkpoints[candidate.id][cutoff],
                    cutoff,
                )
            for run in runs:
                compare(run, checkpoints[run.id][cutoff], None, None, cutoff)
        result["status"] = "completed"
        save()
        from manabot.training.analysis import report

        report(out)
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
    parser.add_argument("--profile", choices=["smoke"], default="smoke")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    from manabot.training.analysis import report

    if args.report_only:
        report(args.report_only.resolve())
        return
    if args.out is None:
        parser.error("--out is required")
    torch.set_num_threads(1)
    out = args.out.resolve()
    run_study(args.study, out)


if __name__ == "__main__":
    main()
