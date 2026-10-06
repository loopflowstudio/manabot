"""Shared history/depth screen supervisor; execution is separate from software checks.

One exclusive output and checkout lock own the attempt. The parent kills whole
child groups on time/disk/progress failure. Calibration, training, evaluation and
reporting have separate ceilings inside a single monotonic allocation; no retry.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable

import numpy as np
import torch

from experiments.runners import history_input
from experiments.runners.history_input import ROOT, CalibrationArm, InputBinding
from experiments.runners.history_predecessor import Predecessor, admit_predecessor
from experiments.runners.screen_spec import ScreenSpec, specification
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_receipt
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import TrainingRegime, TrainingRun
from manabot.verify.store import VerifyStore


def history_plan(
    calibration: history_input.Calibration,
    prior_seconds: float = 0,
    *,
    spec: ScreenSpec = history_input,
) -> ResolvedStudy:
    values = spec.recipes(calibration.admitted_updates())
    return ResolvedStudy(
        protocol=EvaluationProtocol(
            study=spec.STUDY,
            purpose="screening",
            regime_digests=tuple(
                canonical_sha256(r.model_dump(mode="json")) for r in values
            ),
            training_seeds=spec.SEEDS,
            paired_deals=(),
            anchor_deals=spec.DEALS,
            anchors=("scripted-greedy",),
            process_seconds=spec.TOTAL_SECONDS - prior_seconds,
            uncertainty="paired-seed-descriptive",
        ),
        recipes=tuple(r.model_dump(mode="json") for r in values),
        allocation_seconds=spec.TOTAL_SECONDS - prior_seconds,
        prior_campaign_seconds=prior_seconds,
        runtime_identities=calibration.runtime_identities,
        input_bindings=calibration.input_bindings,
        projected_disk_bytes=calibration.projected_disk_bytes,
        calibration_evidence=calibration.model_dump_json(),
    )


def verify_runtime(plan: ResolvedStudy) -> None:
    spec = specification(plan.protocol.study)
    values = [TrainingRegime.model_validate(r) for r in plan.recipes]
    common, bindings = spec.runtime_bindings(values)
    if common != plan.runtime_identities or bindings != plan.input_bindings:
        raise ValueError("history source/native/per-arm input drift since calibration")
    receipt = spec.Calibration.model_validate_json(plan.calibration_evidence)
    _clean_source(receipt.source_commit)
    for arm in receipt.arms:
        if file_sha256(Path(arm.run_path)) != arm.run_sha256:
            raise ValueError("history calibration artifact changed")


def _child(
    arguments: list[str],
    out: Path,
    deadline: float,
    check: Callable[[], None] | None = None,
    *,
    spec: ScreenSpec = history_input,
) -> float:
    """Return full process time; preserve a failure receipt and kill descendants."""
    started = time.monotonic()
    receipt: dict[str, object] = {
        "arguments": arguments,
        "started_unix": time.time(),
        "load": os.getloadavg(),
        "status": "running",
    }
    handle = out / f"child-{time.time_ns()}.json"
    with (out / "children.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.runners.run_history_input",
                "--study",
                spec.STUDY,
                *arguments,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        receipt["pid"] = process.pid
        atomic_json(handle, receipt)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("history phase/process deadline exceeded")
                if shutil.disk_usage(out).free < 4 * 1024**3:
                    raise RuntimeError("less than 4 GiB evidence reserve")
                if check is not None:
                    check()
                try:
                    code = process.wait(timeout=min(1, remaining))
                    if code:
                        raise RuntimeError(f"history child exited {code}")
                    receipt["status"] = "completed"
                    return time.monotonic() - started
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            # Also reap descendants left by a terminated leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            receipt["seconds"] = time.monotonic() - started
            atomic_json(handle, receipt)


def remaining_feasible(
    *, deadline: float, now: float, rate: float, remaining_updates: int
) -> None:
    if now + 1.25 * rate * remaining_updates > deadline:
        raise RuntimeError("remaining training projection exceeds its envelope")


def _clean_source(source: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if actual != source or dirty:
        raise ValueError("history campaign requires exact clean delivered source")


def _check_collisions(out: Path, spec: ScreenSpec = history_input) -> None:
    """Only the explicitly admitted predecessor releases its reserved families."""
    state = json.loads((out / "supervisor.json").read_text())
    predecessor = (
        Predecessor.model_validate(state["predecessor"])
        if state.get("predecessor")
        else None
    )
    if predecessor is not None:
        if admit_predecessor(predecessor.path, predecessor.sha256) != predecessor:
            raise ValueError("predecessor receipt changed")
    collisions: list[str] = []
    for path in sorted((ROOT / "experiments/plans").glob("*.json")):
        data = json.loads(path.read_text())
        protocol = data.get("protocol", {})
        if set(protocol.get("training_seeds", ())) & set(
            (spec.CALIBRATION_SEED, *spec.SEEDS)
        ) or any(
            set(protocol.get(key, ())) & set(spec.DEALS)
            for key in (
                "anchor_deals",
                "paired_deals",
                "endpoint_anchor_deals",
                "endpoint_paired_deals",
            )
        ):
            collisions.append(str(path))
    # Existing attempts in this checkout also reserve their families.
    for path in (ROOT / ".runs").rglob("resolved-plan.json"):
        data = json.loads(path.read_text())
        protocol = data.get("protocol", {})
        if set(protocol.get("training_seeds", ())) & set(spec.SEEDS) or set(
            protocol.get("anchor_deals", ())
        ) & set(spec.DEALS):
            collisions.append(str(path))
    for path in (ROOT / ".runs").rglob("supervisor.json"):
        if path.resolve() == (out / "supervisor.json").resolve():
            continue
        if (
            predecessor is not None
            and path.resolve() == predecessor.path / "supervisor.json"
        ):
            continue
        previous = json.loads(path.read_text())
        if previous.get("study") == spec.STUDY:
            collisions.append(str(path))
    if collisions:
        raise ValueError(f"history seed/deal collision: {collisions}")


def _preflight(out: Path, spec: ScreenSpec = history_input) -> None:
    """No optimizer: native fixtures, dependencies, RNGs, resources and weights."""
    from jupyter_client.kernelspec import KernelSpecManager
    import nbclient
    import nbformat

    notebook = nbformat.read(
        ROOT / "experiments/study/training-regimes.ipynb", as_version=4
    )
    KernelSpecManager().get_kernel_spec(
        notebook.metadata.get("kernelspec", {}).get("name", "python3")
    )
    # Native fixtures run in this supervised process group and consume preflight
    # time. An incompatible installed extension fails before calibration.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/model/test_recent_events.py",
            "tests/training/test_history_arena.py",
            "-q",
        ],
        cwd=ROOT,
        check=True,
    )
    common, bindings = spec.runtime_bindings(spec.recipes())
    atomic_json(
        out / "runtime-bindings.json",
        {"common": common, "bindings": [b.model_dump(mode="json") for b in bindings]},
    )
    # Force the same package imports as offline reporting before charging training.
    assert nbclient.NotebookClient
    families = [
        seed + offset
        for seed in (spec.CALIBRATION_SEED, *spec.SEEDS)
        for offset in (0, 10000, 20000, 30000)
    ]
    if len(set(families)) != len(families) or set(families) & set(spec.DEALS):
        raise ValueError("history seed families collide")
    _check_collisions(out, spec)
    counts: list[dict[str, object]] = []
    for seed in spec.SEEDS:
        models = []
        for recipe in spec.recipes():
            torch.manual_seed(seed)
            model = Agent(ObservationSpace(recipe.observation), recipe.agent)
            models.append(model)
            counts.append(
                {
                    "seed": seed,
                    "recipe": recipe.id,
                    "architecture": architecture_receipt(model).model_dump(mode="json"),
                }
            )
        if spec.STUDY == "history-input":
            off, on = (m.state_dict() for m in models)
            if any(
                name not in on or not torch.equal(value, on[name])
                for name, value in off.items()
            ):
                raise ValueError("paired history shared initialization differs")
            if (
                sum(p.numel() for p in models[1].parameters())
                - sum(p.numel() for p in models[0].parameters())
                != 22912
            ):
                raise ValueError("history parameter increment differs from protocol")
    ram = (
        int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        if sys.platform == "darwin"
        else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    )
    atomic_json(
        out / "preflight.json",
        {
            "architecture": counts,
            "seed_families": families,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "ram_bytes": ram,
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "load": os.getloadavg(),
            "free_disk_bytes": shutil.disk_usage(out).free,
            "memory_limit": "inherited 32 GiB declaration is not enforced",
            "rng_limit": "Paired seeds do not promise shared weight bytes or training deals across architecture changes; native streams retain their own RNG.",
        },
    )


def campaign(
    out: Path,
    source: str,
    predecessor_path: Path | None = None,
    predecessor_sha256: str | None = None,
    *,
    spec: ScreenSpec = history_input,
) -> None:
    if spec.STUDY == "depth-screen" and (predecessor_path or predecessor_sha256):
        raise ValueError("depth screen does not admit predecessors or retries")
    started = time.monotonic()  # Includes lock, source and all preflight work.
    (ROOT / ".runs").mkdir(exist_ok=True)
    with (ROOT / f".runs/{spec.STUDY}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (predecessor_path is None) != (predecessor_sha256 is None):
            raise ValueError("predecessor path and hash must be selected together")
        predecessor = (
            admit_predecessor(predecessor_path, predecessor_sha256)
            if predecessor_path is not None and predecessor_sha256 is not None
            else None
        )
        prior_seconds = predecessor.seconds if predecessor is not None else 0
        total_deadline = started + spec.TOTAL_SECONDS - prior_seconds
        out.mkdir(parents=True, exist_ok=False)
        state: dict[str, object] = {
            "pid": os.getpid(),
            "study": spec.STUDY,
            "status": "preflight",
            "source_commit": source,
            "started_unix": time.time(),
            "order": spec.ORDER,
            "allocation_seconds": spec.TOTAL_SECONDS,
            "deadline_unix": time.time() + spec.TOTAL_SECONDS - prior_seconds,
        }

        if predecessor is not None:
            state["predecessor"] = predecessor.model_dump(mode="json")
            state["prior_seconds"] = prior_seconds
            state["remaining_allocation_seconds"] = spec.TOTAL_SECONDS - prior_seconds
            state["remaining_calibration_seconds"] = (
                spec.CALIBRATION_SECONDS - prior_seconds
            )

        def save() -> None:
            state["seconds"] = time.monotonic() - started
            state["cumulative_seconds"] = prior_seconds + float(state["seconds"])
            atomic_json(out / "supervisor.json", state)

        save()
        try:
            _clean_source(source)
            if shutil.disk_usage(out).free < 7 * 1024**3:
                raise RuntimeError("history requires 3 GiB evidence plus 4 GiB reserve")
            calibration_deadline = min(
                total_deadline
                - spec.TRAINING_SECONDS
                - spec.EVALUATION_SECONDS
                - spec.REPORT_SECONDS,
                started + spec.CALIBRATION_SECONDS - prior_seconds,
            )
            _child(
                ["--preflight", "--out", str(out)], out, calibration_deadline, spec=spec
            )
            runtime = json.loads((out / "runtime-bindings.json").read_text())
            common = runtime["common"]
            bindings = tuple(
                InputBinding.model_validate(b) for b in runtime["bindings"]
            )
            arms: list[CalibrationArm] = []
            for index, recipe in enumerate(spec.recipes(40, calibration=True)):
                seconds = _child(
                    ["--calibrate-arm", str(index), "--out", str(out)],
                    out,
                    min(calibration_deadline, time.monotonic() + 400),
                    spec=spec,
                )
                path = out / "calibration" / recipe.id / "run.json"
                run = TrainingRun.model_validate_json(path.read_text())
                spec.validate_run(
                    run, recipe, spec.CALIBRATION_SEED, bindings[index], common
                )
                arms.append(
                    CalibrationArm(
                        recipe_id=recipe.id,
                        run_path=str(path),
                        run_sha256=file_sha256(path),
                        process_seconds=seconds,
                        stage_seconds=tuple(s.seconds for s in run.stages),
                    )
                )
            # Include all retained prior trace sizes conservatively; no evidence deletion.
            trace_bytes_per_game: list[float] = []
            for saved in (ROOT / ".runs").rglob("study.json"):
                retained = json.loads(saved.read_text())
                for cell in retained.get("comparisons", []):
                    trace = cell.get("trace")
                    if trace and trace.get("games", 0) > 0:
                        path = Path(trace["path"])
                        if path.is_file():
                            trace_bytes_per_game.append(
                                path.stat().st_size / trace["games"]
                            )
            if not trace_bytes_per_game:
                raise ValueError(
                    "history storage admission requires retained per-game trace sizes"
                )
            checkpoint_sizes = [
                p.stat().st_size for p in (out / "calibration").rglob("*.pt")
            ]
            existing = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            projected = max(
                3 * 1024**3,
                2
                * (
                    existing
                    + 24 * max(checkpoint_sizes, default=0)
                    + int(1200 * max(trace_bytes_per_game))
                ),
            )
            if shutil.disk_usage(out).free < projected + 4 * 1024**3:
                raise RuntimeError(
                    "calibrated evidence projection exceeds available disk"
                )
            receipt = spec.Calibration(
                arms=tuple(arms),
                seconds=prior_seconds + time.monotonic() - started,
                source_commit=source,
                runtime_identities=common,
                input_bindings=bindings,
                projected_disk_bytes=projected,
            )
            atomic_json(out / "calibration.json", receipt.model_dump(mode="json"))
            plan = history_plan(receipt, prior_seconds, spec=spec)
            plan_path = out / "resolved-plan.json"
            atomic_json(plan_path, plan.model_dump(mode="json"))
            state["plan_sha256"] = file_sha256(plan_path)
            state["updates_per_run"] = receipt.admitted_updates()
            state["status"] = "training"
            save()
            training_deadline = min(
                time.monotonic() + spec.TRAINING_SECONDS,
                total_deadline - spec.EVALUATION_SECONDS - spec.REPORT_SECONDS,
            )
            count = receipt.admitted_updates()
            rate = receipt.rate()
            completed = 0
            progress_checked = False

            def progress() -> None:
                nonlocal progress_checked
                if (
                    not progress_checked
                    and prior_seconds + time.monotonic() - started >= 7200
                ):
                    # Include the active run's completed durable updates; never scores.
                    active_runs = [
                        TrainingRun.model_validate_json(p.read_text())
                        for p in (out / "training").glob("*/run.json")
                    ]
                    done = sum(
                        len(s.diagnostics) for r in active_runs for s in r.stages
                    )
                    observed_rates = [rate]
                    for run in active_runs:
                        updates = sum(len(s.diagnostics) for s in run.stages)
                        if updates:
                            observed_rates.append(run.seconds / updates)
                        observed_rates.extend(
                            s.seconds / len(s.diagnostics)
                            for s in run.stages
                            if s.diagnostics
                        )
                    remaining_feasible(
                        deadline=training_deadline,
                        now=time.monotonic(),
                        rate=max(observed_rates),
                        remaining_updates=6 * count - done,
                    )
                    progress_checked = True

            for seed_index, seed in enumerate(spec.SEEDS):
                for index in spec.ORDER[seed_index]:
                    remaining_feasible(
                        deadline=training_deadline,
                        now=time.monotonic(),
                        rate=rate,
                        remaining_updates=(6 - completed) * count,
                    )
                    seconds = _child(
                        [
                            "--train-arm",
                            str(index),
                            "--seed",
                            str(seed),
                            "--plan",
                            str(plan_path),
                            "--out",
                            str(out),
                        ],
                        out,
                        min(training_deadline, time.monotonic() + spec.RUN_SECONDS),
                        progress,
                        spec=spec,
                    )
                    rate = max(rate, seconds / count)
                    completed += 1
                    remaining_feasible(
                        deadline=training_deadline,
                        now=time.monotonic(),
                        rate=rate,
                        remaining_updates=(6 - completed) * count,
                    )
            state["status"] = "evaluation"
            save()
            _child(
                ["--evaluate", "--plan", str(plan_path), "--out", str(out)],
                out,
                min(
                    time.monotonic() + spec.EVALUATION_SECONDS,
                    total_deadline - spec.REPORT_SECONDS,
                ),
                spec=spec,
            )
            state["status"] = "reporting"
            save()
            _child(
                ["--report", "--out", str(out)],
                out,
                min(time.monotonic() + spec.REPORT_SECONDS, total_deadline),
                spec=spec,
            )
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("history-input", "depth-screen"), default="history-input"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--campaign", metavar="DELIVERED_COMMIT")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--calibrate-arm", type=int, choices=range(2))
    action.add_argument("--train-arm", type=int, choices=range(2))
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--predecessor-sha256")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = specification(args.study)
    if args.seed is not None and args.seed not in spec.SEEDS:
        parser.error("seed differs from frozen screen")
    if args.study == "depth-screen" and args.predecessor:
        parser.error("depth screen does not admit retries or predecessors")
    if not args.campaign and (args.predecessor or args.predecessor_sha256):
        parser.error("predecessor selection is only valid for --campaign")
    torch.set_num_threads(1)
    out = args.out.resolve()
    if not args.campaign:
        state = json.loads((out / "supervisor.json").read_text())
        if state["pid"] != os.getppid() or state["status"] in {"completed", "failed"}:
            raise ValueError(
                "internal phases require the active campaign supervisor; no retries"
            )
    if args.campaign:
        campaign(
            out, args.campaign, args.predecessor, args.predecessor_sha256, spec=spec
        )
    elif args.preflight:
        _preflight(out, spec)
    elif args.report:
        from experiments.runners.history_input_analysis import history_report
        from manabot.training.analysis import report

        history_report(out / "study", study=spec.STUDY)
        report(out / "study")
        atomic_json(
            out / "manifest.json",
            {
                str(p.resolve()): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                for p in sorted(out.rglob("*"))
                if p.is_file()
                and p.name not in {"manifest.json", "supervisor.json", "children.log"}
                and not p.name.startswith("child-")
            },
        )
    elif args.calibrate_arm is not None:
        recipe = spec.recipes(40, calibration=True)[args.calibrate_arm]
        with VerifyStore(out / "calibration.sqlite") as store:
            run = execute_regime(
                recipe, spec.CALIBRATION_SEED, out / "calibration" / recipe.id, store
            )
            _reload(run)
    else:
        if args.plan is None:
            parser.error("child execution requires frozen --plan")
        plan = ResolvedStudy.model_validate_json(args.plan.read_text())
        if plan.protocol.study != spec.STUDY:
            raise ValueError("requires history-input plan")
        verify_runtime(plan)
        if args.train_arm is not None:
            if args.seed is None:
                parser.error("training requires --seed")
            recipe = TrainingRegime.model_validate(plan.recipes[args.train_arm])
            with VerifyStore(out / "training.sqlite") as store:
                run = execute_regime(
                    recipe,
                    args.seed,
                    out / "training" / f"{recipe.id}-seed-{args.seed}",
                    store,
                )
                spec.validate_run(
                    run,
                    recipe,
                    args.seed,
                    plan.input_bindings[args.train_arm],
                    plan.runtime_identities,
                )
                _reload(run)
        else:
            from experiments.runners.run_training_regimes import run_study

            paths = tuple(
                out / "training" / f"{spec.ARMS[index]}-seed-{seed}" / "run.json"
                for row, seed in zip(spec.ORDER, spec.SEEDS, strict=True)
                for index in row
            )
            run_study(
                spec.STUDY,
                out / "study",
                plan,
                render_report=False,
                history_run_paths=paths,
            )


def _reload(run: TrainingRun) -> None:
    """Check every export's integrity and admit only raw/EMA as policies.

    TrainingRun also exports an Adam state dictionary under ``optimizer``.
    Its digest is checked here; optimizer restoration belongs to recovery.
    """
    for stage in run.stages:
        for name, artifact in stage.artifacts.items():
            if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("history checkpoint bytes changed")
            if name not in {"raw", "ema"}:
                continue
            agent, _ = load_checkpoint_agent(artifact["path"])
            if (
                agent.hypers != run.regime.agent
                or agent.observation_space.encoder.hypers != run.regime.observation
            ):
                raise ValueError(
                    "history checkpoint configuration differs from its run"
                )
            if any(not torch.isfinite(p).all() for p in agent.parameters()):
                raise ValueError("history checkpoint contains nonfinite parameters")


if __name__ == "__main__":
    main()
