"""Exploratory execution comparison; run each arm on the same rented GPU.

The main arm runs this file from an archive of the reference revision, with
the branch's resolved Elemental settings supplied as JSON. No historical
regime or evidence is changed. The branch arm is a normal Etude recipe.
"""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRef
from etude.jobs import TrainingJob
from pydantic import BaseModel

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


class Config(BaseModel):
    workload: Literal["elemental", "baseline"] = "elemental"
    device: Literal["cpu", "cuda"] = "cuda"
    updates: int = 3
    stop_at: int | None = None


def settings(config: Config) -> "SelfPlay":
    from manabot.recipes.baseline import BaselineConfig, settings as baseline
    from manabot.recipes.builders import elemental_self_play

    if config.workload == "baseline":
        return baseline(BaselineConfig(updates=config.updates, device=config.device))
    return elemental_self_play().model_copy(update={"device": config.device})


def train(config: Config) -> None:
    import os

    from etude.train.tracking import track

    from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play

    inputs = SelfPlay.model_validate(settings(config))
    with track(clocks=CLOCKS, config=config) as run:
        # Time the real save calls, including calls that decide no save is due.
        save = run.save
        durations: list[float] = []

        def timed_save() -> None:
            if config.stop_at == run.at("updates"):
                Path(os.environ["ETUDE_STOP_FILE"]).touch()
            started = time.perf_counter()
            save()
            durations.append(time.perf_counter() - started)

        run.save = timed_save
        try:
            self_play(inputs, run, updates=config.updates)
        finally:
            with (run.out / "save-times.jsonl").open("a") as output:
                output.write(json.dumps(durations) + "\n")


def training_jobs() -> list["TrainingJob"]:
    return [TrainingJob(name="comparison", train=train, config=Config(), seeds=(197,))]


def main_arm(path: Path, out: Path) -> None:
    """Use the reference revision's executor with exactly the resolved inputs."""
    from manabot.backends import resolve
    from manabot.training.execution import execute_regime
    from manabot.training.regime import Execution, TrainingRegime, TrainSelfPlay
    from manabot.verify.store import VerifyStore

    values = json.loads(path.read_text())
    stage_keys = (
        "streams",
        "transitions",
        "games_per_stream",
        "learning_seats",
        "batch_closure",
        "matchup_curriculum",
        "learning",
        "behavior",
        "game_errors",
    )
    stage = TrainSelfPlay(
        id="elemental",
        operation="train_self_play",
        updates=3,
        execution=Execution(
            device="cuda",
            precision=values["precision"],
            wall_seconds=1200,
            workers=values["workers"],
            threads=values["threads"],
            bucketed_shapes=values["bucketed_shapes"],
            live_tokens=values["live_tokens"],
        ),
        **{key: values[key] for key in stage_keys},
    )
    regime = TrainingRegime(
        id="etude-execution-elemental",
        world=values["world"],
        match=values["match"],
        observation=values["observation"],
        agent=values["model"],
        stages=[stage],
        wall_seconds=1200,
        schedule_clock="iteration_fraction",
    )
    with VerifyStore(out.with_suffix(".sqlite")) as store:
        result = execute_regime(regime, 197, out, store, backend=resolve("torch"))
    out.with_suffix(".json").write_text(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise RuntimeError(f"main arm {result.status}: {result.error}")


def measured_load(
    checkpoint: "CheckpointRef",
) -> Callable[[list[object]], list[object]]:
    """Retain actual CUDA forward batch sizes in Etude's inference process."""
    import os

    import torch

    from manabot.eval.agent_loader import load

    forward = load(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    destination = Path(os.environ["EXECUTION_BATCHES"])

    def measured(rows: list[object]) -> list[object]:
        result = forward(rows)
        with destination.open("a") as output:
            output.write(json.dumps({"batch": len(rows), "device": device}) + "\n")
        return result

    return measured


def evaluate(path: Path, out: Path) -> None:
    """Replay-checked paired deals through the production managed callback."""
    import os

    from etude.checkpoints import CheckpointRef
    from etude.eval.records import (
        Benchmark,
        CheckpointSlot,
        Code,
        Composition,
        Environment,
        Seating,
    )
    from etude.eval.tournament import evaluating
    from etude.jobs import Agents
    from etude.records import ArtifactRef

    from manabot.backends import resolve
    from manabot.eval.agents import build, etude_registration
    from manabot.eval.benchmark import Conditions, play
    from manabot.eval.local import checkpoint_request
    from manabot.infra.identity import canonical_sha256, file_sha256
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.sim.registration import ETUDE_COMPUTE_CLASS

    request = checkpoint_request(
        path,
        backend=resolve("torch"),
        include_mirrors=True,
        deal_seeds=tuple(range(1_915_100_000, 1_915_100_016)),
        game_seconds=60,
    )
    ref = CheckpointRef(
        address="execution/policy",
        artifact=ArtifactRef(
            uri=path.resolve().as_uri(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        ),
    )
    candidate = etude_registration(request.candidate)
    tournament = Benchmark(
        name="execution-cuda-agents-v1",
        environment=Environment(
            name="execution-comparison",
            files=canonical_sha256(
                {
                    "harness": file_sha256(Path(__file__)),
                    "runtime": runtime_fingerprints(),
                }
            ),
        ),
        seatings=tuple(
            Seating(
                name=f"deal-{deal}",
                players=(
                    Composition(
                        type="etude:configured",
                        fields={
                            "over": CheckpointSlot().model_dump(mode="json"),
                            "config": {
                                "registration": candidate.model_dump(mode="json")
                            },
                        },
                    ),
                    Code(
                        name="heuristic",
                        config={
                            "registration": request.opponent.model_dump(mode="json")
                        },
                    ),
                ),
                config=Conditions(
                    deal=deal,
                    key=request.key.model_copy(
                        update={
                            "evaluation_compute_envelope_id": ETUDE_COMPUTE_CLASS,
                            "arena_version": "execution-cuda-agents-v1",
                        }
                    ),
                    game_seconds=60,
                    max_commands=10_000,
                ).model_dump(mode="json"),
                games=1,
            )
            for deal in request.deal_seeds
        ),
    )
    out.mkdir(parents=True)
    os.environ["ETUDE_OUTPUT_DIR"] = str(out.resolve())
    os.environ["EXECUTION_BATCHES"] = str((out / "batches.jsonl").resolve())
    started = time.perf_counter()
    with evaluating(
        tournament, experiment=out.name, checkpoint=ref, directory=out / "track"
    ) as active:
        result = active.play(play, agents=Agents(load=measured_load, build=build))
    (out / "result.json").write_text(
        json.dumps(
            {
                "elapsed": time.perf_counter() - started,
                "result": result.model_dump(mode="json"),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("settings", "main", "eval"))
    parser.add_argument("--path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "settings":
        args.out.write_text(settings(Config()).model_dump_json(indent=2))
    elif args.mode == "main":
        main_arm(args.path, args.out)
    else:
        evaluate(args.path, args.out)


if __name__ == "__main__":
    # Etude serializes loader functions by importable module, never __main__.
    from experiments.etude_execution import main

    main()
