"""The throughput probe, and step 2's launch legs and training digest."""

import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

from etude.launch.job_file import read_job_file
import pytest

from experiments import (
    training_throughput as probe,
    training_throughput_packing as packing,
)
from manabot.experiment.setup import machine_setup
from manabot.infra.host import MemoryLimit
from manabot.training.records import TrainingRun
from manabot.training.update_metrics import read_updates

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "experiments" / "regimes" / "elemental-long-183-v1.json"


@pytest.mark.training
def test_cpu_probe_writes_receipts_and_samples_for_every_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(probe, "journal", lambda _: tmp_path / "journal")
    config = probe.Probe(
        recipe=str(RECIPE),
        device="cpu",
        streams=8,
        phases=(
            probe.Phase(name="solo", children=1, updates=2, first_seed=26001),
            probe.Phase(name="pack-2", children=2, updates=2, first_seed=26002),
        ),
    )

    results = probe.drive(config, output)

    assert [r.status for r in results] == ["completed", "completed"]
    for result in results:
        folder = output / "training-throughput" / result.phase.name
        assert json.loads((folder / "phase.json").read_text())["status"] == "completed"
        samples = [
            probe.Sample.model_validate_json(line)
            for line in (folder / "samples.jsonl").read_text().splitlines()
        ]
        assert samples
        assert result.peak_child_bytes and result.peak_child_bytes > 0
        for child in result.children:
            seed_folder = folder / f"seed-{child.seed}"
            run = TrainingRun.model_validate_json(
                (seed_folder / "receipt.json").read_bytes()
            )
            assert run.status == "completed"
            assert run.stages[0].updates == 2
            assert run.stages[0].collection_timings
            # The per-update metrics run survives in the child's published folder.
            assert run.metrics_run is not None
            name = Path(run.metrics_run).name
            updates = read_updates(seed_folder / "run", [run])[name]
            assert sorted(updates) == [1, 2]
            assert all(
                any(k.startswith("time/") for k in row) for row in updates.values()
            )
        pids = {c.pid for sample in samples for c in sample.children}
        assert len(pids) == result.phase.children


def test_phase_is_skipped_when_packed_peak_would_exceed_memory(tmp_path: Path) -> None:
    config = probe.Probe(recipe=str(RECIPE), device="cpu", memory_fraction=0.5)
    phase = probe.Phase(name="pack-4", children=4, updates=1, first_seed=1)
    machine = probe.psutil.virtual_memory().total

    result = probe.run_phase(phase, config, tmp_path, previous_peak=machine // 6)

    assert result.status == "skipped"
    assert not result.children
    assert (tmp_path / "pack-4" / "phase.json").exists()


def test_gate_reads_the_pods_allowance_below_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowance = 48 * 10**9
    monkeypatch.setattr(
        probe, "memory_limit", lambda: MemoryLimit(bytes=allowance, source="cgroup-v2")
    )
    monkeypatch.setattr(
        probe.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=540 * 10**9),
    )
    config = probe.Probe(recipe=str(RECIPE), device="cpu")
    phase = probe.Phase(name="pack-4", children=4, updates=1, first_seed=1)

    result = probe.run_phase(phase, config, tmp_path, previous_peak=15 * 10**9)

    assert result.status == "skipped"
    assert result.machine_bytes == allowance
    assert result.memory_source == "cgroup-v2"


def test_gate_names_host_memory_when_no_allowance_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "memory_limit", lambda: None)
    config = probe.Probe(recipe=str(RECIPE), device="cpu", memory_fraction=0.5)
    phase = probe.Phase(name="pack-4", children=4, updates=1, first_seed=1)
    machine = probe.psutil.virtual_memory().total

    result = probe.run_phase(phase, config, tmp_path, previous_peak=machine)

    assert result.machine_bytes == machine
    assert result.memory_source == "host"


def test_children_track_locally_not_into_the_launched_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "launched")
    monkeypatch.setenv("ETUDE_TRACK_DIR", "/track")

    environment = probe._child_environment()

    assert not any(key.startswith("ETUDE_") for key in environment)


# ---------------------------------------------------------------------------
# Machine setup


def _tool(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text(f'#!/bin/sh\necho "{name} $*" >> "$HOME/calls"\n{body}\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _setup(tmp_path: Path, rustup: str, flock: bool = True) -> tuple[int, list[str]]:
    """Run the real setup script with only recording stand-ins on its PATH."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tmp_path / ".cargo/bin").mkdir(parents=True)
    if flock:
        _tool(tools, "flock", "exit 0")
    _tool(tmp_path / ".cargo/bin", "rustup", rustup)
    _tool(tools, "uv", 'test -n "$RUSTUP_TOOLCHAIN"')
    done = subprocess.run(
        ["/bin/sh", "-c", machine_setup(" --extra dev")],
        env={"HOME": str(tmp_path), "PATH": str(tools)},
    )
    return done.returncode, (tmp_path / "calls").read_text().splitlines()


def test_setup_takes_the_lock_before_the_toolchain_and_syncs_outside_it(
    tmp_path: Path,
) -> None:
    code, calls = _setup(tmp_path, "exit 0")

    assert code == 0
    assert [call.split()[0] for call in calls] == ["flock", "rustup", "uv"]
    assert calls[-1].endswith("--extra dev")
    assert (tmp_path / ".manabot-rust-setup.lock").exists()


def test_setup_runs_where_flock_is_absent(tmp_path: Path) -> None:
    code, calls = _setup(tmp_path, "exit 0", flock=False)

    assert code == 0
    assert [call.split()[0] for call in calls] == ["rustup", "uv"]


def test_failed_toolchain_install_stops_setup(tmp_path: Path) -> None:
    code, calls = _setup(tmp_path, "exit 3")

    assert code != 0
    assert not any(call.startswith("uv") for call in calls)


# ---------------------------------------------------------------------------
# Step 2 legs


def _launched(*arguments: str) -> list[packing.Run]:
    """The runs `etude launch FILE ARGUMENTS` would read from the launch file."""
    return read_job_file(
        ROOT / "experiments/training_throughput_packing.py", list(arguments)
    )


def test_launch_file_declares_every_leg() -> None:
    found = {leg: _launched("--revision", "HEAD", "--leg", leg) for leg in packing.LEGS}

    assert {leg: [run.seed for run in runs] for leg, runs in found.items()} == {
        "d1": [26201],
        "d2": [26201, 26202],
        "m1": [26211, 26212, 26213, 26214],
        "m2": [26211, 26212, 26213, 26214],
        "m3": [26211],
        "s14": [26211],
        "s22": [26211, 26212],
        "s44": [26211, 26212, 26213, 26214],
    }
    assert {r.job.name for r in found["m1"]} == {
        "elemental-throughput-pack-check-v1-m1"
    }


def test_only_cuda_legs_sample_the_gpu() -> None:
    sampled = {
        leg
        for leg in packing.LEGS
        if packing.GPU_SAMPLER in packing.runs("HEAD", leg)[0].job.command
    }
    assert sampled == set(packing.LEGS) - {"d1", "d2"}


def test_gpu_sampler_writes_rows_and_leaves_the_jobs_exit_status(
    tmp_path: Path,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    stand_in = tools / "nvidia-smi"
    stand_in.write_text(
        '#!/bin/sh\necho "$$" > "$PIDFILE"\n'
        'echo "2026/11/07 13:00:00.000, 41, 1024"\nexec sleep 60\n'
    )
    stand_in.chmod(stand_in.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "output"
    pid_file = tmp_path / "sampler-pid"

    done = subprocess.run(
        ["sh", "-c", f"{packing.GPU_SAMPLER}\nsleep 1\nexit 3"],
        env={
            "ETUDE_OUTPUT_DIR": str(output),
            "PATH": f"{tools}:/usr/bin:/bin",
            "PIDFILE": str(pid_file),
        },
        timeout=30,
    )

    assert done.returncode == 3
    assert (output / packing.GPU_SAMPLES).read_text().count("\n") == 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_one_seed_of_a_leg_can_launch_alone_and_others_are_refused() -> None:
    one = _launched("--revision", "HEAD", "--leg", "m2", "--seeds", "26212")
    assert [run.seed for run in one] == [26212]
    with pytest.raises(ValueError, match="declares no seeds"):
        packing.runs("HEAD", "m2", "99")
    with pytest.raises(ValueError, match="unknown leg"):
        packing.runs("HEAD", "m9")


# ---------------------------------------------------------------------------
# Training digest


def test_differences_name_the_path_of_a_change_or_a_missing_field() -> None:
    left = {"a": 1, "b": "x", "only-left": None}
    right = {"a": 2, "b": "x", "only-right": 0}

    assert packing.digest_differences(left, right) == ["a", "only-left", "only-right"]
