"""Step 1 of the training-throughput experiment: is collection the bottleneck? (Task b015d5)

Protocol: experiments/protocols/training-throughput-b015d5-v1.md. One Etude run per
rented machine runs `PHASES` in sequence on that host. Each phase starts its
children at once as ordinary `scripts/bench_training_update.py` subprocesses on the
control recipe, and samples the machine once a second while they run: each
child's resident and peak memory and CPU seconds, the machine's available memory,
and, where `nvidia-smi` exists, GPU utilization and per-process device memory.
Every child writes its whole `TrainingRun` receipt, so the phase tree, idle stream
steps, CPU seconds and host come from the training code itself. Nothing here
trains for strength or changes a learning rule.

A child after the first phase is skipped when the previous phase's largest child
peak, times this phase's child count, exceeds `memory_fraction` of the machine.
Results are journaled under `/dev/shm` when it exists (outside the attempt's
output, which the courier publishes) and copied to `ETUDE_OUTPUT_DIR` at each phase
end; the job's cleanup copies whatever an interrupted phase left.

    uv run etude launch experiments/training_throughput.py --rent b015d5-l4 \\
        --revision COMMIT --machines 2
    uv run etude launch experiments/training_throughput.py --rent b015d5-a4000 \\
        --revision COMMIT --usd-per-run 0.70
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from decimal import Decimal
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Literal

from etude.launch.records import Job, Run
import psutil
from pydantic import BaseModel, ConfigDict, Field

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "bench_training_update.py"
EXPERIMENT = "training_throughput"
CONTROL = "experiments.elemental_variants:elemental_recipe"


class Phase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    children: int = Field(ge=1)
    updates: int = Field(ge=1)
    # Child i uses first_seed + i.
    first_seed: int
    # Instrument the one child with CUDA launch and synchronization counts.
    cuda_counts: bool = False


PHASES: tuple[Phase, ...] = (
    Phase(name="solo", children=1, updates=12, first_seed=26001),
    Phase(name="pack-2", children=2, updates=12, first_seed=26002),
    Phase(name="pack-4", children=4, updates=8, first_seed=26004),
    # Last and alone, so its instrumentation perturbs no timed phase.
    Phase(
        name="cuda-counts", children=1, updates=3, first_seed=26008, cuda_counts=True
    ),
)


class Probe(BaseModel):
    """The probe's frozen configuration, handed back at ETUDE_CONFIG."""

    model_config = ConfigDict(extra="forbid")

    # A bench `--recipe`: regime JSON path or `module:function` declaration.
    recipe: str = CONTROL
    device: Literal["cuda", "cpu"] = "cuda"
    streams: int = Field(default=256, ge=1)
    phases: tuple[Phase, ...] = PHASES
    memory_fraction: float = Field(default=0.85, gt=0, le=1)
    sample_seconds: float = Field(default=1.0, gt=0)
    # Frozen by `runs` from the exact revision; absent for local driver runs.
    source_bundle: SourceBundle | None = None


# ---------------------------------------------------------------------------
# Samples


class ChildSample(BaseModel):
    pid: int
    rss_bytes: int
    # Linux VmHWM; None where /proc is absent.
    peak_bytes: int | None
    cpu_seconds: float


class GpuSample(BaseModel):
    utilization_percent: float
    memory_used_mib: float
    # Device memory by child process ID, from nvidia-smi's compute apps.
    process_memory_mib: dict[int, float]


class Sample(BaseModel):
    seconds: float  # since the phase started
    available_bytes: int
    children: list[ChildSample]
    gpu: list[GpuSample] | None


class ChildResult(BaseModel):
    seed: int
    returncode: int
    receipt: bool


class PhaseResult(BaseModel):
    phase: Phase
    status: Literal["completed", "failed", "skipped"]
    reason: str | None = None
    wall_seconds: float = 0
    machine_bytes: int
    children: list[ChildResult] = []
    # The largest child peak memory seen (VmHWM where readable, else sampled RSS).
    peak_child_bytes: int | None = None


def _peak_bytes(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _child_sample(process: psutil.Process) -> ChildSample | None:
    """The child's memory and CPU, including its own subprocesses' CPU."""
    try:
        with process.oneshot():
            times = process.cpu_times()
            return ChildSample(
                pid=process.pid,
                rss_bytes=process.memory_info().rss,
                peak_bytes=_peak_bytes(process.pid),
                cpu_seconds=times.user
                + times.system
                + times.children_user
                + times.children_system,
            )
    except psutil.Error:
        return None


def _gpu_sample(nvidia_smi: str) -> list[GpuSample] | None:
    def query(*arguments: str) -> list[list[str]]:
        text = subprocess.run(
            [nvidia_smi, *arguments, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return [
            [f.strip() for f in line.split(",")] for line in text.splitlines() if line
        ]

    try:
        devices = query("--query-gpu=utilization.gpu,memory.used")
        apps = query("--query-compute-apps=pid,used_memory")
    except (OSError, subprocess.SubprocessError):
        return None
    processes = {int(pid): float(used) for pid, used in apps}
    return [
        GpuSample(
            utilization_percent=float(utilization),
            memory_used_mib=float(used),
            process_memory_mib=processes,
        )
        for utilization, used in devices
    ]


class _Sampler(threading.Thread):
    """Appends one `Sample` per period to a JSON-lines file until stopped."""

    def __init__(self, path: Path, pids: Sequence[int], period: float) -> None:
        super().__init__(daemon=True)
        self._path = path
        self._processes = [psutil.Process(pid) for pid in pids]
        self._period = period
        self._halt = threading.Event()
        self._nvidia_smi = shutil.which("nvidia-smi")
        self.peak_bytes: int | None = None

    def run(self) -> None:
        started = time.monotonic()
        with self._path.open("a") as out:
            while True:
                children = [
                    sample
                    for sample in map(_child_sample, self._processes)
                    if sample is not None
                ]
                for child in children:
                    peak = max(child.rss_bytes, child.peak_bytes or 0)
                    self.peak_bytes = max(self.peak_bytes or 0, peak)
                sample = Sample(
                    seconds=time.monotonic() - started,
                    available_bytes=psutil.virtual_memory().available,
                    children=children,
                    gpu=None
                    if self._nvidia_smi is None
                    else _gpu_sample(self._nvidia_smi),
                )
                out.write(sample.model_dump_json() + "\n")
                out.flush()
                if self._halt.wait(self._period):
                    return

    def stop(self) -> None:
        self._halt.set()
        self.join()


# ---------------------------------------------------------------------------
# Driver


def _child_environment() -> dict[str, str]:
    """Children track locally: the launcher's run identity belongs to the driver."""
    return {k: v for k, v in os.environ.items() if not k.startswith("ETUDE_")}


def run_phase(
    phase: Phase, probe: Probe, root: Path, previous_peak: int | None
) -> PhaseResult:
    """Start the phase's children at once, sample until all exit, keep everything."""
    machine = psutil.virtual_memory().total
    folder = root / phase.name
    folder.mkdir(parents=True, exist_ok=True)
    if previous_peak is not None and (
        previous_peak * phase.children > probe.memory_fraction * machine
    ):
        result = PhaseResult(
            phase=phase,
            status="skipped",
            reason=f"{phase.children} x {previous_peak} B peak exceeds "
            f"{probe.memory_fraction:g} of {machine} B",
            machine_bytes=machine,
        )
        (folder / "phase.json").write_text(result.model_dump_json(indent=2))
        return result
    started = time.monotonic()
    children: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    for index in range(phase.children):
        seed = phase.first_seed + index
        child = folder / f"seed-{seed}"
        child.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(BENCH),
            "--recipe",
            probe.recipe,
            "--device",
            probe.device,
            "--streams",
            str(probe.streams),
            "--updates",
            str(phase.updates),
            "--seed",
            str(seed),
            "--json",
            str(child / "receipt.json"),
        ]
        if phase.cuda_counts:
            command += ["--cuda-counts", str(child / "cuda-counts.json")]
        with (child / "log.txt").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_child_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        children.append((seed, child, process))
    sampler = _Sampler(
        folder / "samples.jsonl", [p.pid for _, _, p in children], probe.sample_seconds
    )
    sampler.start()
    try:
        codes = [process.wait() for _, _, process in children]
    finally:
        sampler.stop()
    results = [
        ChildResult(
            seed=seed, returncode=code, receipt=(child / "receipt.json").exists()
        )
        for (seed, child, _), code in zip(children, codes, strict=True)
    ]
    failed = [r.seed for r in results if r.returncode != 0 or not r.receipt]
    result = PhaseResult(
        phase=phase,
        status="failed" if failed else "completed",
        reason=f"children failed: seeds {failed}" if failed else None,
        wall_seconds=time.monotonic() - started,
        machine_bytes=machine,
        children=results,
        peak_child_bytes=sampler.peak_bytes,
    )
    (folder / "phase.json").write_text(result.model_dump_json(indent=2))
    return result


def journal(output: Path) -> Path:
    """Where phases write while running: memory-backed when the host offers it."""
    shm = Path("/dev/shm")
    if shm.is_dir():
        return shm / f"training-throughput-{os.environ.get('ETUDE_RUN', 'local')}"
    return output.parent / "training-throughput"


def collect(output: Path) -> None:
    """Copy every journaled phase into the attempt output, completing partial ones."""
    source = journal(output)
    if source.is_dir():
        shutil.copytree(source, output / "training-throughput", dirs_exist_ok=True)


def drive(probe: Probe, output: Path) -> list[PhaseResult]:
    """Run every phase in order on this host; raise after all if any failed."""
    root = journal(output)
    root.mkdir(parents=True, exist_ok=True)
    found: list[PhaseResult] = []
    peak: int | None = None
    for phase in probe.phases:
        result = run_phase(phase, probe, root, peak)
        found.append(result)
        print(f"PHASE {phase.name}: {result.status}: {result.reason}", flush=True)
        if result.status == "completed":
            peak = result.peak_child_bytes
        collect(output)
    failed = [r.phase.name for r in found if r.status == "failed"]
    if failed:
        raise RuntimeError(f"probe phases failed: {failed}; all evidence retained")
    return found


def run(probe: Probe) -> None:
    """The launched entry (`etude.call`): bind source, require the device, drive."""
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if probe.source_bundle is not None:
        bind_etude_source()
    if probe.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback measurement")
    try:
        drive(probe, output)
    finally:
        collect(output)


# ---------------------------------------------------------------------------
# Launch


def runs(
    revision: str, machines: int = 1, hours: float = 1.5, usd_per_run: float = 1.05
) -> list[Run]:
    """One probe run per machine; the seed only names the run (phase seeds are fixed)."""
    probe = Probe(source_bundle=bundle_at(ROOT, revision))
    job = Job(
        name="training-throughput-probe",
        setup=machine_setup(""),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m etude.call "
        "--function experiments.training_throughput:run",
        cleanup="uv run --locked python -m experiments.training_throughput --collect",
        config=probe.model_dump_json(),
        repo=str(ROOT),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal(str(usd_per_run)),
        max_attempts=1,
    )
    return job.runs(*range(1, machines + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collect", action="store_true", required=True)
    parser.parse_args()
    collect(Path(os.environ["ETUDE_OUTPUT_DIR"]))


if __name__ == "__main__":
    main()
