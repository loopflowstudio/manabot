"""The throughput probe, and step 2's launch legs and training digest."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

from etude.launch.job_file import read_job_file
import pytest

from experiments import (
    training_throughput as probe,
    training_throughput_packing as packing,
)
from manabot.experiment.job import TrainingJob, machine_setup
from manabot.infra.host import MemoryLimit
from manabot.infra.identity import canonical_sha256
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


def _setup(tmp_path: Path, rustup: str) -> tuple[int, list[str]]:
    """Run the real setup script against recording stand-ins for its tools."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tmp_path / ".cargo/bin").mkdir(parents=True)
    _tool(tools, "flock", "exit 0")
    _tool(tmp_path / ".cargo/bin", "rustup", rustup)
    _tool(tools, "uv", 'test -n "$RUSTUP_TOOLCHAIN"')
    done = subprocess.run(
        ["/bin/sh", "-c", machine_setup(" --extra dev")],
        env={"HOME": str(tmp_path), "PATH": f"{tools}:/usr/bin:/bin"},
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


def test_failed_toolchain_install_stops_setup(tmp_path: Path) -> None:
    code, calls = _setup(tmp_path, "exit 3")

    assert code != 0
    assert not any(call.startswith("uv") for call in calls)


# ---------------------------------------------------------------------------
# Step 2 legs


def _regime_digest(run: packing.Run) -> str:
    frozen = TrainingJob.model_validate_json(run.job.config or "")
    return canonical_sha256(frozen.regime.model_dump(mode="json"))


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
    }
    assert {r.job.name for r in found["m1"]} == {
        "elemental-throughput-pack-check-v1-m1"
    }


def test_metric_legs_share_regime_bytes_and_stay_separate_jobs() -> None:
    metric = [packing.runs("HEAD", leg)[0] for leg in ("m1", "m2", "m3")]

    assert len({_regime_digest(run) for run in metric}) == 1
    assert len({run.job.environment for run in metric}) == 3
    frozen = TrainingJob.model_validate_json(metric[0].job.config or "")
    assert frozen.updates == 20
    assert frozen.milestones == ()
    assert frozen.regime.stages[0].execution.device == "cuda"
    assert frozen.regime.stages[0].streams == 256
    assert frozen.regime.wall_seconds == 2400
    assert metric[0].job.max_hours == 1.0
    assert str(metric[0].job.max_usd) == "0.75"


def test_digest_legs_run_the_small_cpu_variant() -> None:
    solo, packed = (packing.runs("HEAD", leg)[0] for leg in ("d1", "d2"))

    assert _regime_digest(solo) == _regime_digest(packed)
    stage = TrainingJob.model_validate_json(solo.job.config or "").regime.stages[0]
    assert (stage.streams, stage.updates) == (24, 3)
    assert (stage.execution.device, stage.execution.threads) == ("cpu", 1)


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


def _worker(
    tmp_path: Path, spec: TrainingJob, name: str, seed: int
) -> tuple[subprocess.Popen[bytes], Path]:
    """Start the real training worker for one attempt; several share a track folder."""
    folder = tmp_path / name
    folder.mkdir()
    config = folder / "config.json"
    config.write_text(spec.model_dump_json())
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("ETUDE_")}
    environment |= {
        "ETUDE_RUN": name,
        "ETUDE_TRACK_DIR": str(tmp_path / "tracking"),
        "ETUDE_CONFIG": str(config),
        "ETUDE_OUTPUT_DIR": str(folder / "output"),
        "ETUDE_STOP_FILE": str(folder / "stop"),
        "ETUDE_SEED": str(seed),
        "ETUDE_UNTIL_CLOCK": "updates",
        "ETUDE_UNTIL_VALUE": str(spec.updates),
        "ETUDE_DEADLINE": deadline.isoformat(),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "manabot.experiment.job"],
        cwd=ROOT,
        env=environment,
        stdout=(folder / "log.txt").open("wb"),
        stderr=subprocess.STDOUT,
    )
    return process, folder / "output" / "training"


@pytest.fixture(scope="module")
def solo_and_packed(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    """Seed 26201 alone, then beside a neighbor: (solo, packed, neighbor) folders."""
    root = tmp_path_factory.mktemp("packing")
    regime = packing.digest_recipe(streams=8, updates=2)
    regime.wall_seconds = 600
    regime.stages[0].execution.wall_seconds = 600
    spec = TrainingJob(regime=regime, experiment=packing.EXPERIMENT)
    solo, solo_folder = _worker(root, spec, "solo", 26201)
    assert solo.wait() == 0
    started = [
        _worker(root, spec, "packed", 26201),
        _worker(root, spec, "neighbor", 26202),
    ]
    assert [process.wait() for process, _ in started] == [0, 0]
    return solo_folder, started[0][1], started[1][1]


@pytest.mark.training
def test_a_packed_seed_digests_as_it_does_solo(
    solo_and_packed: tuple[Path, Path, Path],
) -> None:
    solo, packed, neighbor = solo_and_packed

    digest = packing.training_digest(solo)

    assert packing.digest_differences(digest, packing.training_digest(packed)) == []
    # The digest does read learning: another seed differs throughout.
    different = packing.digest_differences(digest, packing.training_digest(neighbor))
    assert "self-play-optimizer.pt" in different
    assert any(
        path.startswith("self-play-raw.pt/model_state_dict") for path in different
    )
    assert any(path.startswith("game-attempts.jsonl") for path in different)


@pytest.mark.training
def test_digest_reports_an_unlisted_attempt_field_and_ignores_listed_ones(
    solo_and_packed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    solo = solo_and_packed[0]
    digest = packing.training_digest(solo)
    changed = tmp_path / "changed"
    changed.mkdir()
    for path in solo.iterdir():
        if path.is_file():
            (changed / path.name).symlink_to(path)
    attempts = changed / "game-attempts.jsonl"
    rows = [json.loads(line) for line in attempts.read_text().splitlines()]
    attempts.unlink()
    rows[0]["attempt"]["seconds"] += 1
    rows[0]["context"]["run"] = "another"
    rows[0]["context"]["coordinates"]["host_load_1m"] += 1
    rows[0]["attempt"]["valid_samples"] += 1
    rows[0]["context"]["clock"] = "new volatile field"
    attempts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert packing.digest_differences(digest, packing.training_digest(changed)) == [
        "game-attempts.jsonl/0/attempt/valid_samples",
        "game-attempts.jsonl/0/context/clock",
    ]
