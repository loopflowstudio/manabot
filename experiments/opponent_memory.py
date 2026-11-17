"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report (`opponent_memory.md`) owns the design, evidence and open decisions.
`training_jobs(..., calibration=True)` declares the bounded exploratory CUDA
measurement. Pilot training and evaluation stop until its results date a branch
recipe; `branch_jobs` freezes the three arms from each published trunk.
`census` and `probe` are the USD0 commands. Nothing runs on import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

from etude.jobs import ExperimentState
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.spec import Job, RunRequest
from pydantic import BaseModel, ConfigDict

from manabot.experiment.job import PretrainedInitialization, TrainingJob
from manabot.infra.source import SourceBundle
from manabot.training.regime import TrainingRegime


@dataclass(frozen=True)
class Arm:
    """One encoder/summary choice over the shared viewer decision log."""

    name: str
    encoder: Literal["none", "transformer", "gru"]
    summaries: int


# K=1 selected by the corrected supervised probe; three pilot arms.
ARMS = (
    Arm("control", "none", 0),
    Arm("transformer-k1", "transformer", 1),
    Arm("gru-k1", "gru", 1),
)
SEEDS = (39101, 39102, 39103)
DISCLOSURES = ("open", "closed")
# Inclusive of calibration, training, evaluation, startup and failed attempts.
MAX_USD = 50


def training_jobs(
    revision: str, calibration: bool = False, calibration_reference: bool = False
) -> list[RunRequest]:
    """Declare calibration only; pilot allocation requires its measured results."""
    if calibration and calibration_reference:
        raise ValueError("select one calibration workload")
    if calibration or calibration_reference:
        from etude.jobs import TrainingJob as PythonJob

        from experiments.opponent_memory import Calibration, calibrate
        from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
        from manabot.infra.source import bundle_at

        root = Path(__file__).resolve().parents[1]
        job = PythonJob(
            name="cuda-calibration-reference"
            if calibration_reference
            else "cuda-calibration-full",
            train=calibrate,
            config=Calibration(
                source_bundle=bundle_at(root, revision), reference=calibration_reference
            ),
            repo=str(root),
            revision=revision,
            setup=machine_setup(" --extra artifacts --extra dev"),
            extras=("artifacts", "dev"),
            max_hours=0.65 if calibration_reference else 3.0,
            max_usd=Decimal("0.50") if calibration_reference else Decimal("2.00"),
            max_attempts=1,
        ).plain()
        return job.model_copy(
            update={"command": f"{TOOLCHAIN_ENV}\n{job.command}"}
        ).runs(39100)
    raise RuntimeError(
        "Opponent-memory CUDA peaks, worker/update and checkpoint-reload admission "
        "have not passed admission. Calibrate a common learner-"
        "decision horizon and wire jobs before allocating training. "
        "See experiments/opponent_memory.md."
    )


def branch_jobs(
    revision: str,
    *,
    regime: TrainingRegime,
    trunks: Mapping[int, PretrainedInitialization],
    max_hours: float,
    max_usd: Decimal,
) -> list[RunRequest]:
    """Freeze matched branches from published trunk inputs without launching.

    Calibration supplies the shared branch recipe, horizon, gate scale and caps.
    Each seed maps to exactly one producer receipt and EMA export, reused by all
    three arms. The ordinary training declaration stays stopped until CUDA
    admission and calibration.
    """
    if not trunks or any(seed not in SEEDS for seed in trunks):
        raise ValueError("branches require declared pilot seed labels")
    if regime.agent.decision_memory != "none":
        raise ValueError("the shared branch recipe must describe the control")
    if any(trunk.weights != "ema" for trunk in trunks.values()):
        raise ValueError("all branches use the same EMA trunk role")
    if any(trunk.source_seed != seed for seed, trunk in trunks.items()):
        raise ValueError("each branch seed label must match its producer seed")
    jobs: list[RunRequest] = []
    for seed, trunk in sorted(trunks.items()):
        for arm in ARMS:
            configuration = regime.model_copy(deep=True)
            configuration.id = f"decision-memory-{arm.name}"
            configuration.agent.decision_memory = arm.encoder
            if arm.summaries:
                configuration.agent.decision_memory_summaries = arm.summaries
            job = TrainingJob(
                regime=configuration,
                experiment=Path(__file__).stem,
                pretrained=trunk,
            ).job(
                repo=Path(__file__).resolve().parents[1],
                revision=revision,
                max_hours=max_hours,
                max_usd=max_usd,
            )
            jobs.extend(job.runs(seed))
    return jobs


def eval_jobs(state: ExperimentState) -> list[Job]:
    """Stop rather than score unmatched updates as matched learner decisions."""
    raise RuntimeError(
        "Wire paired Elemental 10x10 evaluation to retained learner-decision "
        "checkpoint clocks after the software prerequisite; see the report."
    )


def report() -> Report:
    """Use the existing Elemental matchup panels and the learner-decision clock."""
    from manabot.training.panels import ML, PERF

    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="learner_transitions",
        evals=(
            Curve(
                title="Head-to-head against control",
                matches=Matches(tag="control"),
                split="decklists",
            ),
            Curve(
                title="Intact memory against blanked memory",
                matches=Matches(tag="reliance"),
                split="decklists",
            ),
            Matrix(
                title="Descriptive arm versus control matchups",
                rows="opponent_deck",
                columns="deck",
                matches=Matches(tag="control"),
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )


class Calibration(BaseModel):
    """Exploratory real-worker calibration, separate from pilot training."""

    model_config = ConfigDict(extra="forbid")
    reference: bool = False

    source_bundle: SourceBundle | None = None
    smoke: bool = False


def calibration_recipe(name: str, *, smoke: bool) -> TrainingRegime:
    """The ordinary Elemental workload with explicit memory execution bounds."""
    from manabot.infra.config import ModelConfig, ObservationConfig
    from manabot.recipes.builders import sampled_self_play
    from manabot.training.decks import UniformDecks
    from manabot.training.regime import IterationSchedule, TrainSelfPlay
    import managym

    incumbent = name == "incumbent"
    arm = next((arm for arm in ARMS if arm.name == name), ARMS[0])
    if name not in {"incumbent", *(arm.name for arm in ARMS)}:
        raise ValueError(f"unknown calibration arm {name}")
    observation = ObservationConfig(
        max_decision_records=0 if incumbent else 512,
        max_decision_reference_details=0 if incumbent else 16384,
        max_decision_references=0 if incumbent else 128,
        max_decision_members=0 if incumbent else 8192,
    )
    recipe = sampled_self_play(
        id=f"decision-memory-calibration-{name}",
        world=managym.WORLD_VERSION,
        agent=ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=64,
            attention_layers=2,
            num_attention_heads=4,
            value_aggregation="masked_mean",
            policy_film=True,
            decision_memory_summaries=1,
        ),
        observation=ObservationConfig(),
        deck_source=UniformDecks(
            decks=tuple(
                managym.Deck(player.decklist, player.sideboard)
                for key in managym.authored_deck_keys("elemental-world-v2")
                for player in [managym.authored_deck_setup("elemental-world-v2", key)]
            ),
            sampling="uniform-independent-v1",
            decklists="mixed",
        ),
        # A sampled deck source owns mixed disclosure and independent mirrors;
        # TrainingRegime forbids combining it with the older curriculum flag.
        wall_seconds=600 if smoke else 1800,
        seconds_per_update=60,
        workers=1 if smoke else 4,
    )
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 1 if smoke else 2
    stage.streams = 2 if smoke else 256
    stage.learning.iteration_schedule = IterationSchedule(steps=100)
    # Calibration fails on capacity; paid-pilot recovery is still undecided.
    stage.game_errors = None
    stage.execution.collection_rows = 2 if smoke else 32
    stage.execution.bootstrap_rows = 2 if smoke else 32
    stage.execution.learner_rows = 2 if smoke else 32
    stage.execution.precision = "float32"
    recipe.observation = observation
    recipe.agent.decision_memory = arm.encoder
    if smoke:
        stage.learning_seats = "one"
        stage.batch_closure = "exact-quota"
        stage.transitions = 8
        stage.execution.device = "cpu"
        stage.execution.live_tokens = False
        stage.execution.bucketed_shapes = False
    return TrainingRegime.model_validate(recipe.authored_dump())


def calibration_child(
    connection: "Connection",
    name: str,
    directory: str,
    smoke: bool,
    producer: str | None,
) -> None:
    """A fresh actual worker and allocator per repeat, including ordinary reload."""
    from datetime import datetime, timedelta, timezone
    import json
    import os
    import time

    from etude.storage import DirectoryStorage
    import numpy as np
    import torch

    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.env.memory_store import MemoryStore, Prefix, PrefixIdentity
    from manabot.experiment.job import run_worker
    from manabot.infra.host import host_facts
    from manabot.training.records import TrainingRun
    from scripts.verify_decision_memory_storage import HostPeak, residency

    out = Path(directory)
    out.mkdir(parents=True)
    recipe = calibration_recipe(name, smoke=smoke)
    spec = TrainingJob(regime=recipe, experiment="opponent_memory_calibration")
    if producer is not None:
        # Flatten only the two published inputs, as from_storage expects;
        # ordinary worker binding still owns digest/model/world admission.
        parent = TrainingRun.model_validate_json(Path(producer).read_bytes())
        storage = DirectoryStorage(out / "published")
        storage.put_file("run.json", Path(producer))
        artifact = Path(parent.stages[0].artifacts["ema"]["path"])
        storage.put_file(artifact.name, artifact)
        spec.pretrained = PretrainedInitialization.from_storage(
            storage, "run.json", source_stage=parent.stages[0].id, weights="ema"
        )
        inputs = out / "inputs"
        inputs.mkdir()
        for given in spec.pretrained.inputs():
            (inputs / given.name).write_bytes(storage.get(given.key))
    config = out / "worker.json"
    config.write_text(spec.model_dump_json(indent=2))
    # Nested workers have independent tracking identities; the launcher owns
    # the outer calibration and carries every child's receipts as output.
    for key in tuple(os.environ):
        if key.startswith("ETUDE_"):
            del os.environ[key]
    os.environ.update(
        ETUDE_RUN=out.name,
        ETUDE_TRACK_DIR=str(out / "tracking"),
        ETUDE_CONFIG=str(config),
        ETUDE_OUTPUT_DIR=str(out / "output"),
        ETUDE_STOP_FILE=str(out / "stop"),
        ETUDE_SEED="39100",
        ETUDE_UNTIL_CLOCK="updates",
        ETUDE_UNTIL_VALUE=str(spec.updates),
        ETUDE_DEADLINE=(
            datetime.now(timezone.utc) + timedelta(seconds=recipe.wall_seconds + 90)
        ).isoformat(),
    )
    if producer is not None:
        os.environ["ETUDE_INPUTS_DIR"] = str(out / "inputs")
    maxima = [0, 0, 0, 0]
    original = MemoryStore.capture

    def capture(
        store: MemoryStore,
        buffers: Mapping[str, np.ndarray],
        row: int,
        identity: PrefixIdentity,
    ) -> Prefix:
        prefix = original(store, buffers, row, identity)
        for index, count in enumerate(prefix.counts):
            maxima[index] = max(maxima[index], count)
        return prefix

    MemoryStore.capture = capture
    device = torch.device("cpu" if smoke else "cuda")
    if not smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration requires CUDA; no device fallback")
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    receipt: dict[str, object] = {"arm": name, "host": host_facts().model_dump()}
    peak = HostPeak()
    try:
        with peak:
            run = run_worker()
            if run.status != "completed" or run.updates_through() != spec.updates:
                raise RuntimeError("worker did not finish the declared updates")
            stage = run.stages[0]
            # All outputs load through ordinary world/shape admission. Move the
            # final policy to CUDA and check every parameter after the transfer.
            for role in ("raw", "ema"):
                model, _ = load_checkpoint(stage.artifacts[role]["path"])
                model.to(device)
                if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                    raise RuntimeError(f"nonfinite {role} checkpoint after reload")
                del model
            receipt.update(
                status="completed",
                worker=stage.model_dump(mode="json"),
                reload="raw and EMA admitted and finite on the selected device",
                residency=residency(device),
                gpu=None if smoke else torch.cuda.get_device_name(device),
                cuda_total_bytes=None
                if smoke
                else torch.cuda.get_device_properties(device).total_memory,
            )
        receipt["sampled_host_tree_peak_bytes"] = peak.peak
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(
            wall_seconds=time.monotonic() - started,
            table_maxima=maxima,
            sampled_host_tree_peak_bytes=peak.peak,
            residency=residency(device),
        )
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        MemoryStore.capture = original
        connection.close()


def calibrate(config: Calibration) -> None:
    """Interleave incumbent and memory workers within one bounded GPU rental."""
    import json
    import os
    import subprocess
    import time

    import psutil

    from manabot.infra.source import bind_etude_source
    from manabot.sim.child import ChildProcess

    if config.source_bundle is not None:
        bind_etude_source()
    out = Path(os.environ["ETUDE_OUTPUT_DIR"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not config.smoke:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration requires an actual CUDA device")
        # Run the directed CUDA backward regression before the real workers;
        # the subprocess releases its allocator before any timed measurement.
        with (out / "attention-proof.log").open("w") as proof:
            print("calibration: CUDA backward preflight", flush=True)
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    "tests/model/test_live_tokens.py::test_sparse_short_chunk_matches_padded_backward",
                    "-q",
                ],
                stdout=proof,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        print("calibration: CUDA backward preflight passed", flush=True)
    sequence = ["incumbent", *(arm.name for arm in ARMS)]
    if not config.smoke:
        sequence += list(reversed(sequence))
    sequence.insert(0, "control")  # one shared EMA producer before any branches
    if config.reference:
        # Complete the missing reference variability without rerunning admitted
        # memory workers. This is a separately labelled, fresh-host measurement.
        sequence = ["incumbent", "control", "incumbent"]
    receipts: list[str] = []
    try:
        for ordinal, name in enumerate(sequence):
            remaining = (2400 if config.smoke else 9000) - (time.monotonic() - started)
            worker_seconds = 690 if config.smoke else 1890
            if remaining < worker_seconds:
                raise RuntimeError("calibration wall cap reached before all repeats")
            directory = out / f"{ordinal:02d}-{name}"
            print(
                f"calibration: {ordinal + 1}/{len(sequence)} {name} starting",
                flush=True,
            )
            child = ChildProcess(
                "experiments.opponent_memory:calibration_child",
                (
                    name,
                    str(directory),
                    config.smoke,
                    str(out / "00-control/output/training/run.json")
                    if ordinal and name != "incumbent" and not config.reference
                    else None,
                ),
            )
            try:
                child.join(timeout=min(worker_seconds, remaining))
                if child.exitcode != 0:
                    raise RuntimeError(
                        f"{name} worker failed or timed out; see {directory}"
                    )
            finally:
                if child.is_alive():
                    # A killed learner cannot run its producer cleanup. Reap
                    # this measurement's descendants before closing its parent.
                    descendants = psutil.Process(child.pid).children(recursive=True)
                    for process in reversed(descendants):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    psutil.wait_procs(descendants, timeout=10)
                child.close()
            receipts.append(str(directory.relative_to(out) / "receipt.json"))
            print(f"calibration: {name} completed; {receipts[-1]}", flush=True)
    finally:
        (out / "calibration.json").write_text(
            json.dumps(
                {
                    "receipts": receipts,
                    "smoke": config.smoke,
                    "reference": config.reference,
                    "wall_seconds": time.monotonic() - started,
                    "complete": len(receipts) == len(sequence),
                },
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    census_parser = operations.add_parser("census")
    census_parser.add_argument("--out", type=Path, required=True)
    probe_parser = operations.add_parser("probe")
    probe_parser.add_argument("--out", type=Path, required=True)
    probe_parser.add_argument("--epochs", type=int, default=8)
    probe_parser.add_argument("--max-games", type=int, default=120)
    probe_parser.add_argument("--wall-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.operation == "census":
        from manabot.verify.decision_memory_capacity import run_census

        run_census(args.out)
        return
    if args.operation == "probe":
        from manabot.backends.torch.decision_memory_probe import (
            MAX_WALL_SECONDS,
            MIN_GAMES,
            run_probe,
        )

        if args.epochs < 1:
            probe_parser.error("--epochs must be at least 1")
        if args.max_games < MIN_GAMES:
            probe_parser.error(f"--max-games must be at least {MIN_GAMES}")
        if not 0 < args.wall_seconds <= MAX_WALL_SECONDS:
            probe_parser.error(f"--wall-seconds must be in (0, {MAX_WALL_SECONDS:g}]")
        if args.out.exists():
            probe_parser.error(f"--out {args.out} exists; choose a fresh directory")
        run_probe(
            args.out,
            epochs=args.epochs,
            max_games=args.max_games,
            wall_seconds=args.wall_seconds,
        )
        return


if __name__ == "__main__":
    main()
