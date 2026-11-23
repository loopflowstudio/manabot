"""Measure the baseline's collection and learning on mini-heart.

Run a width once with `run --out DIR --width 64`. Each width owns one CPU
capture and ten serial parity/live pairs. `report --out DIR [--width 64]`
reads receipts offline. Without --width, both commands visit all three widths.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
from typing import Any, Literal
from uuid import uuid4

from manabot.infra.identity import file_sha256

WIDTHS = (64, 256, 384)
SEED = 25201
THREADS = 4
EXPERIMENT = "mini-baseline-backends"


@dataclass(frozen=True)
class Arm:
    backend: Literal["torch", "mlx"]
    device: Literal["cpu", "mps"]
    precision: Literal["float32", "bf16"]

    @property
    def dtype(self) -> str:
        return "bfloat16" if self.precision == "bf16" else "float32"

    @property
    def name(self) -> str:
        device = "gpu" if self.backend == "mlx" else self.device
        return f"{self.backend}-{device}-{'f32' if self.precision == 'float32' else 'bf16'}"


ARMS = (
    Arm("torch", "cpu", "float32"),
    Arm("torch", "mps", "float32"),
    Arm("torch", "mps", "bf16"),
    Arm("mlx", "mps", "float32"),
    Arm("mlx", "mps", "bf16"),
)


def cells(width: int) -> Iterable[tuple[str, Arm]]:
    for index, layout in enumerate(("padded", "bucketed")):
        offset = (2 * WIDTHS.index(width) + index) % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            yield layout, arm


def write(path: Path, data: object) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def host_reading() -> dict[str, object]:
    import psutil

    return {
        "utc": datetime.now(UTC).isoformat(),
        "load": os.getloadavg(),
        "available_bytes": psutil.virtual_memory().available,
        "swap_used_bytes": psutil.swap_memory().used,
    }


def machine() -> dict[str, object]:
    def output(*command: str) -> str:
        return subprocess.check_output(command, text=True).strip()

    host = socket.gethostname().split(".")[0]
    if host != "mini-heart" or output("git", "status", "--porcelain", "-uno"):
        raise ValueError(
            "measurement requires the clean measured checkout on mini-heart"
        )
    return {
        "host": host,
        "revision": output("git", "rev-parse", "HEAD"),
        "chip": output("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(output("sysctl", "-n", "hw.memsize")),
        "cores": int(output("sysctl", "-n", "hw.physicalcpu")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": version("torch"),
        "mlx": version("mlx"),
        "etude": version("etude-sdk"),
    }


def worker(directory: Path, width: int, layout: str, arm: Arm, capture: Path) -> None:
    from etude.train.tracking import track

    from manabot.backends import resolve
    from manabot.recipes.baseline import BaselineConfig, settings
    from manabot.training.loops.self_play import CLOCKS, self_play

    capturing = directory == capture.parent
    baseline = settings(BaselineConfig(device=arm.device, backend=arm.backend))
    config = baseline.model_copy(
        update={
            "model": baseline.model.model_copy(update={"hidden_dim": width}),
            "threads": THREADS,
            "precision": arm.precision,
            "bucketed_shapes": layout == "bucketed",
            "pretrained": None if capturing else capture / "initial.pt",
        }
    )
    with track(clocks=CLOCKS, config=config, strict=True) as run:
        if capturing:
            from manabot.backends.torch.capture import capture as capture_update

            capture_update(capture, config, run)
        else:
            self_play(config, run, updates=4, backend=resolve(arm.backend))
    # Darwin reports ru_maxrss in bytes. Sample before any offline metric reads.
    write(
        directory / "worker.json",
        {
            "run": run.name,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def attempt(directory: Path, phase: str, args: list[str]) -> dict[str, Any]:
    """Time process start through exit, retaining the log even on timeout or refusal."""
    run_id = uuid4().hex[:16]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ETUDE_")}
    env.update(
        ETUDE_EXPERIMENT=EXPERIMENT,
        ETUDE_RUN=run_id,
        ETUDE_SEED=str(SEED),
        ETUDE_TRACK_DIR=str(directory / "track"),
        ETUDE_OUT=str(directory / "training"),
        PYTORCH_ENABLE_MPS_FALLBACK="0",
    )
    log = directory / f"{phase}.log"
    before = host_reading()
    interrupted = False
    with log.open("w") as stream:
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, *args],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
            status = (
                "completed" if result.returncode == 0 else f"exit {result.returncode}"
            )
        except subprocess.TimeoutExpired:
            status = "timeout"
        except KeyboardInterrupt:
            status = "interrupted"
            interrupted = True
        wall = time.perf_counter() - start
    receipt = dict(
        status=status, wall_seconds=wall, before=before, after=host_reading()
    )
    if status != "completed":
        lines = log.read_text(errors="replace").splitlines()
        receipt["detail"] = lines[-1] if lines else status
    write(directory / f"{phase}-attempt.json", receipt)
    if interrupted:
        raise KeyboardInterrupt
    return receipt


def finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def verdict(directory: Path, layout: str, arm: Arm, digest: str) -> tuple[str, str]:
    attempt_record = record(directory / "parity-attempt.json")
    path = directory / "parity.json"
    if attempt_record["status"] not in {"completed", "exit 1"} or not path.exists():
        return "rejected", attempt_record.get("detail", "missing parity output")
    proof = record(path)[arm.backend]
    failures = proof["failures"]
    if proof["execution_error"]:
        return "rejected", proof["execution_error"]
    if not (
        proof["checkpoint_sha256"] == digest
        and proof["backend"] == arm.backend
        and proof["actual_device"] == proof["requested_device"] == arm.device
        and proof["precision"] == proof["observed_forward_dtype"] == arm.dtype
        and proof["bucketed_shapes"] is (layout == "bucketed")
        and proof["requested_bucketed_shapes"] is proof["bucketed_shapes"]
        and proof["threads"] == THREADS
        and proof["step_checked"]
        and proof["minibatch_indices"] == list(range(8))
        and len(set(proof["collection_indices"])) == 8
        and finite(proof)
        and attempt_record["status"] == ("exit 1" if failures else "completed")
    ):
        return "rejected", "incomplete or mismatched parity evidence"
    if not failures:
        return "admitted", ""
    approximate = arm.precision == "bf16" and set(failures) <= {
        "forward logits",
        "forward values",
        "collection logits",
        "collection values",
        "learner loss",
        "full gradient",
    }
    detail = ", ".join(failures)
    if "learner loss" in failures:
        worst = max(proof["minibatches"], key=lambda batch: batch["loss_relative"])
        detail += (
            f" {worst['loss_relative']:.2%} on batch {worst['index']}"
            f" ({worst['loss_torch']:.8g} vs {worst['loss_backend']:.8g})"
        )
    return ("approximate" if approximate else "rejected"), detail


def step_rows(directory: Path, digest: str) -> dict[str, Any]:
    """Read only the four self_play.step rows, after the tracked process exits."""
    from etude.train.read import read

    worker_record = record(directory / "worker.json")
    data = read(directory / "track", EXPERIMENT, worker_record["run"])
    (setup,) = data.read_table("training_run", "updates")
    if setup.rows[0]["parent_sha256"] != digest:
        raise ValueError("Etude parent checkpoint differs from parity")
    rows = []
    for block in data.columns:
        for index in range(block.rows):
            row = block.row(index)
            if "time/collection" in row.values:
                rows.append(
                    {
                        "update": row.clocks["updates"],
                        **{
                            k: v
                            for k, v in row.values.items()
                            if k.startswith(
                                ("time/", "coordinates/", "memory/", "execution/")
                            )
                            or k
                            in {
                                "rows",
                                "retained",
                                "optimizer_exposures",
                                "numerical/optimizer_steps",
                            }
                        },
                    }
                )
    return worker_record | {"checkpoint_sha256": digest, "steps": rows}


def run_width(out: Path, width: int, host: dict[str, object]) -> None:
    root = out / f"w{width}l2"
    root.mkdir(parents=True, exist_ok=False)
    target = {"width": width, "seed": SEED, "threads": THREADS, **host}
    write(root / "machine.json", target)
    capture = root / "capture"
    script = str(Path(__file__).resolve())

    def command(directory: Path, layout: str, arm: Arm) -> list[str]:
        return [
            script,
            "_worker",
            str(directory),
            str(width),
            layout,
            arm.name,
            str(capture),
        ]

    print(root.name, "capture", flush=True)
    if (
        attempt(root, "capture", command(root, "padded", ARMS[0]))["status"]
        != "completed"
    ):
        raise RuntimeError(f"capture failed, see {root}/capture.log")
    checkpoint = capture / "initial.pt"
    digest = file_sha256(checkpoint)
    # Bind both names and bytes of every capture file, in relative path order.
    files = {
        p.relative_to(capture).as_posix(): file_sha256(p)
        for p in sorted(capture.rglob("*"))
        if p.is_file()
    }
    write(root / "capture-files.json", files)
    write(
        root / "source.json",
        {
            "capture_sha256": file_sha256(root / "capture-files.json"),
            "checkpoint_sha256": digest,
            "run": record(root / "worker.json")["run"],
        },
    )
    for order, (layout, arm) in enumerate(cells(width), 1):
        directory = root / layout / arm.name
        directory.mkdir(parents=True)
        args = [
            "-m",
            "manabot.backends.parity",
            str(checkpoint),
            "--capture",
            str(capture),
            "--device",
            arm.device,
            "--precision",
            arm.dtype,
            "--backend",
            arm.backend,
            "--threads",
            str(THREADS),
            "--json",
            str(directory / "parity.json"),
        ]
        if layout == "bucketed":
            args.append("--bucketed")
        print(root.name, order, layout, arm.name, "parity", flush=True)
        attempt(directory, "parity", args)
        gate, detail = verdict(directory, layout, arm, digest)
        write(
            directory / "cell.json",
            target
            | asdict(arm)
            | {
                "order": order,
                "layout": layout,
                "checkpoint_sha256": digest,
                "gate": gate,
                "reason": detail,
            },
        )
        print(root.name, order, gate, detail, flush=True)
        if gate == "rejected":
            continue
        live = attempt(directory, "live", command(directory, layout, arm))
        if live["status"] == "completed":
            write(directory / "live.json", step_rows(directory, digest))
        print(
            root.name,
            order,
            "live",
            live["status"],
            round(live["wall_seconds"], 1),
            flush=True,
        )


@dataclass(frozen=True)
class Rates:
    collection: float
    policy: float
    engine_share: float
    learner: float
    exposures: float
    active: float
    wall: float
    active_range: tuple[float, float]
    collection_range: tuple[float, float]
    retained_fraction: float
    optimizer_steps: int
    targets_seconds: float
    minibatch_seconds: float


def rates(steps: list[dict[str, Any]], wall: float, arm: Arm) -> Rates:
    if [s["update"] for s in steps] != [1, 2, 3, 4] or not finite(steps):
        raise ValueError("live measurement needs four finite updates")
    count = 0
    for step in steps:
        if (
            step["execution/backend"],
            step["execution/device"],
            step["execution/precision"],
        ) != (arm.backend, arm.device, arm.precision):
            raise ValueError("live execution differs from parity")
        count += step["rows"]
        if count != step["coordinates/learner_transitions"]:
            raise ValueError("update workload differs from learner transitions")
    measured = steps[1:]

    def total(key: str) -> float:
        return sum(s[key] for s in measured)

    rows = total("rows")
    collection = total("time/collection")
    learning = total("time/learning")
    policy = total("time/collection/observation_preparation") + total(
        "time/collection/policy_inference_and_action_selection"
    )
    active = [s["rows"] / (s["time/collection"] + s["time/learning"]) for s in measured]
    collected = [s["rows"] / s["time/collection"] for s in measured]
    return Rates(
        rows / collection,
        rows / policy,
        total("time/collection/environment_step_and_encoding") / collection,
        rows / learning,
        total("optimizer_exposures") / learning,
        rows / (collection + learning),
        count / wall,
        (min(active), max(active)),
        (min(collected), max(collected)),
        total("retained") / rows,
        int(total("numerical/optimizer_steps")),
        total("time/learning/targets"),
        total("time/learning/minibatches"),
    )


def report(out: Path, widths: Iterable[int]) -> str:
    lines = [
        "# Mini baseline backend throughput",
        "",
        "Rates pool updates 2–4. Wall rate includes startup, warmup and shutdown. "
        "Policy rate counts collection preparation and inference. Rates use learner decisions "
        "per second except exp/s, which counts optimizer exposures.",
    ]
    revisions = set()
    for width in widths:
        root = out / f"w{width}l2"
        host = record(root / "machine.json")
        revisions.add(host["revision"])
        source = record(root / "source.json")
        digest = source["checkpoint_sha256"]
        if (
            (host["host"], host["width"], host["seed"], host["threads"])
            != ("mini-heart", width, SEED, THREADS)
            or file_sha256(root / "capture-files.json") != source["capture_sha256"]
            or (record(root / "capture-files.json")["initial.pt"] != digest)
        ):
            raise ValueError(f"invalid capture identity at {root}")
        lines.extend(
            [
                "",
                f"## {root.name} on {host['host']}",
                "",
                f"{host['chip']}, {host['platform']}, Python {host['python']}, "
                f"Torch {host['torch']}, MLX {host['mlx']}. Four threads. "
                f"Revision `{host['revision']}`. Capture run `{source['run']}`, "
                f"checkpoint `{digest}`.",
            ]
        )
        for order, (layout, arm) in enumerate(cells(width), 1):
            if (order - 1) % len(ARMS) == 0:
                lines.extend(
                    [
                        "",
                        f"### {layout.capitalize()}",
                        "",
                        "| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |",
                        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
                    ]
                )
            directory = root / layout / arm.name
            cell = record(directory / "cell.json")
            gate, reason = verdict(directory, layout, arm, digest)
            expected = (
                host
                | asdict(arm)
                | dict(
                    order=order,
                    layout=layout,
                    checkpoint_sha256=digest,
                    gate=gate,
                    reason=reason,
                )
            )
            if cell != expected:
                raise ValueError(
                    f"cell does not match its parity evidence in {directory}"
                )
            live_path = directory / "live.json"
            prefix = f"| {arm.name} | {gate}" + (
                f" (parity FAIL) {reason}" if reason else ""
            )
            if gate == "rejected":
                if live_path.exists() or (directory / "live-attempt.json").exists():
                    raise ValueError(f"timing on rejected cell {directory}")
                lines.append(prefix + " | — | — | — | — | — | — | — | — |")
                continue
            invocation = record(directory / "live-attempt.json")
            if invocation["status"] != "completed":
                if live_path.exists():
                    raise ValueError(f"timing on failed attempt {directory}")
                lines.append(
                    prefix
                    + f" | — | — | — | — | — | — | — | {invocation['status']} {invocation.get('detail', '')} |"
                )
                continue
            live = record(live_path)
            if live["checkpoint_sha256"] != digest:
                raise ValueError(f"timing on different checkpoint {directory}")
            r = rates(live["steps"], invocation["wall_seconds"], arm)
            lines.append(
                prefix
                + f" | {r.collection:.1f} | {r.policy:.1f} | {r.engine_share:.1%} "
                f"| {r.learner:.1f} | {r.exposures:.1f} | {r.active:.1f} "
                f"({r.active_range[0]:.1f}–{r.active_range[1]:.1f}) | {r.wall:.1f} | {live['run']} |"
            )
    if len(revisions) != 1:
        raise ValueError("matrix widths use different revisions")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--width", type=int, choices=WIDTHS)
    child = commands.add_parser("_worker")
    child.add_argument("directory", type=Path)
    child.add_argument("width", type=int, choices=WIDTHS)
    child.add_argument("layout", choices=("padded", "bucketed"))
    child.add_argument("arm", choices=[a.name for a in ARMS])
    child.add_argument("capture", type=Path)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(
            args.directory,
            args.width,
            args.layout,
            next(a for a in ARMS if a.name == args.arm),
            args.capture,
        )
        return
    widths = [args.width] if args.width else WIDTHS
    if args.command == "report":
        print(report(args.out, widths))
    else:
        host = machine()
        for width in widths:
            run_width(args.out.resolve(), width, host)


if __name__ == "__main__":
    main()
