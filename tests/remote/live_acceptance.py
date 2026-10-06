"""Explicit paid proof, never collected by pytest; includes all retained attempts.

Run only after the exact committed source is publicly fetchable. This helper
charges every prior deployment in its fixed evidence directory to the same $4.90
allocation and evaluates the returned raw checkpoint through the ordinary arena.
"""

import argparse
from pathlib import Path
import time

from manabot.remote.bundle import Bundle, verify_training_bundle
from manabot.remote.deploy import Receipt, current_source, deploy
from manabot.remote.plan import HardwareMix, compile_plan
from manabot.remote.provider import RunPod
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingCoordinates, TrainingRegime, TrainingRun
from manabot.training.monitor_evaluation import MonitorProtocol, evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    out = root / ".runs/remote-acceptance"
    out.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    previous = sorted(out.glob("attempt-*/deployment.json"))
    for path in previous:
        receipt = Receipt.model_validate_json(path.read_text())
        if receipt.phase != "deleted" or receipt.estimated_dollars is None:
            raise ValueError("prior cost/deletion unresolved; do not rent again")
        spent += receipt.estimated_dollars
    mix_data = HardwareMix.model_validate_json(args.mix.read_text()).model_dump()
    mix_data["dollar_cap"] = min(mix_data["dollar_cap"], 4.90 - spent)
    mix = HardwareMix.model_validate(mix_data)
    regime = TrainingRegime.model_validate_json(args.regime.read_text())
    if any(
        stage.operation != "train_self_play" or stage.learning.ema is None
        for stage in regime.stages
    ):
        raise ValueError("acceptance requires raw and EMA self-play exports")
    plan = compile_plan(args.regime.read_text(), mix, current_source(root), 197)
    destination = out / f"attempt-{len(previous):03d}"
    initial_pods = RunPod().list()
    if any(p.name.startswith("manabot-") for p in initial_pods):
        raise ValueError("an owned pod already exists; reconcile before acceptance")
    # Keep counts even if deployment fails; no account or pod identities enter
    # the acceptance summary. Unrelated rentals are observed, never removed.
    atomic_json(
        out / f"{destination.name}-inventory.json",
        {"observed_at": time.time(), "initial_inventory_pods": len(initial_pods)},
    )
    result = deploy(plan, destination, root)
    bundle = Bundle.model_validate_json((destination / "bundle.json").read_text())
    evidence = destination / "evidence"
    verify_training_bundle(evidence, bundle)
    run = TrainingRun.model_validate_json((evidence / "run/run.json").read_text())
    if not all(
        stage.actual_device.startswith("cuda") and stage.optimizer_exposures > 0
        for stage in run.stages
    ):
        raise ValueError("CUDA optimizer work was not demonstrated")
    selected = run.selected_artifact
    if selected is None:
        raise ValueError("no selected policy")
    path = bundle.resolve(evidence, selected["path"])
    # This is a local evaluation input, not a rewrite of producer evidence.
    local_artifact = {**selected, "path": str(path)}
    final_stage = run.stages[-1]
    if final_stage.cumulative_seconds is None:
        raise ValueError("final checkpoint lacks its elapsed coordinate")
    start = time.perf_counter()
    arena = evaluate_checkpoint(
        run,
        local_artifact,
        TrainingCoordinates(
            stage_id=final_stage.id,
            updates=run.updates_through(final_stage.id),
            training_seconds=final_stage.cumulative_seconds,
            environment_decisions=sum(s.environment_decisions for s in run.stages),
            learner_transitions=sum(s.learner_transitions for s in run.stages),
            optimizer_exposures=sum(s.optimizer_exposures for s in run.stages),
            games=sum(s.games for s in run.stages),
        ),
        destination / "arena",
        protocol=MonitorProtocol(deal_seeds=(1_910_114_000,), game_seconds=120),
    )
    pods = RunPod().list()
    atomic_json(
        destination / "acceptance.json",
        {
            "arena_status": arena.status,
            "arena_seconds": time.perf_counter() - start,
            "rental_estimated_dollars_including_prior_attempts": spent
            + result.estimated_dollars,
            "inventory_pods": len(pods),
            "initial_inventory_pods": len(initial_pods),
            "owned_pods": sum(p.name.startswith("manabot-") for p in pods),
            "strength_claim": False,
        },
    )
    if (
        arena.status != "completed"
        or any(p.name.startswith("manabot-") for p in pods)
        or (not initial_pods and pods)
    ):
        raise ValueError("live acceptance incomplete")
    print(
        "Returned CUDA raw/EMA exports verified; four terminal replayed arena games; no owned pods."
    )


if __name__ == "__main__":
    main()
