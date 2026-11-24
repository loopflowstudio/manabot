"""A tracked self-play run can change numerical backend at a saved update."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from etude.train.read import read
from etude.train.state import RestoredState, read_state
from etude.train.tracking import Stopped, TrainingRun
import numpy as np
import pytest

from manabot.backends import resolve
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.policy.checkpoint import read_safetensors, write_safetensors
from manabot.policy.protocol import BackendPolicy
from manabot.training.loops import self_play as loop
from manabot.training.rollout.collection import Collector
from manabot.training.state.backend import (
    SelfPlayBackend,
    SelfPlayLearner,
    require_continuation,
)
from manabot.training.state.continuation import capture_host
from tests.fixtures.self_play import bounded_self_play, same_arrays
from tests.fixtures.training_records import recording


@pytest.fixture
def settings() -> loop.SelfPlay:
    pytest.importorskip("torch")
    recipe = bounded_self_play()
    return recipe.model_copy(
        update={
            "device": "cpu",
            "transitions": 4,
            "model": recipe.model.model_copy(
                update={"hidden_dim": 32, "attention_layers": 1}
            ),
        }
    )


def attempt(root: Path, *, resume: bool = False) -> TrainingRun:
    return recording(
        experiment="test",
        run="portable",
        seed=19,
        clocks=loop.CLOCKS,
        directory=root / "track",
        resume=resume,
    )


def latest(root: Path) -> RestoredState:
    saved = read_state(root / "out/state")
    assert saved is not None
    return saved


def train(
    root: Path,
    settings: loop.SelfPlay,
    backend: SelfPlayBackend,
    *,
    updates: int,
    resume: bool = False,
) -> Path:
    with (
        patch.dict(os.environ, ETUDE_OUT=str(root / "out")),
        attempt(root, resume=resume) as run,
    ):
        return loop.self_play(settings, run, updates=updates, backend=backend)


def stop_after(
    root: Path, settings: loop.SelfPlay, backend: SelfPlayBackend, update: int
) -> None:
    stop = root / "stop"
    with (
        patch.dict(os.environ, ETUDE_OUT=str(root / "out"), ETUDE_STOP_FILE=str(stop)),
        pytest.raises(Stopped),
        attempt(root) as run,
    ):
        steps = run.steps

        def request_stop(clock: str, *, to: int) -> Iterator[int]:
            for current in steps(clock, to=to):
                yield current
                if current == update:
                    stop.touch()

        with patch.object(run, "steps", request_stop):
            loop.self_play(settings, run, updates=update + 1, backend=backend)
    assert latest(root).record.at["updates"] == update


@contextmanager
def inspect_first_update(
    inspect: Callable[
        [
            SelfPlayLearner,
            BackendPolicy | None,
            Collector,
            loop.Counts,
            np.random.Generator,
        ],
        None,
    ],
) -> Iterator[None]:
    """Read the live state after restore and before the first resumed decision."""
    step = loop.step
    inspected = False

    def observed(
        settings: loop.SelfPlay,
        learner: SelfPlayLearner,
        ema: BackendPolicy | None,
        collector: Collector,
        counts: loop.Counts,
        rng: np.random.Generator,
        *,
        out: Path,
        backend: SelfPlayBackend,
    ) -> dict[str, object]:
        nonlocal inspected
        if not inspected:
            inspect(learner, ema, collector, counts, rng)
            inspected = True
        return step(
            settings, learner, ema, collector, counts, rng, out=out, backend=backend
        )

    with patch.object(loop, "step", observed):
        yield
    assert inspected


@pytest.mark.parametrize(("writer", "reader"), [("torch", "mlx"), ("mlx", "torch")])
def test_cross_backend_safe_stop(
    tmp_path: Path, settings: loop.SelfPlay, writer: str, reader: str
) -> None:
    pytest.importorskip("mlx.core")
    stop_after(tmp_path, settings, resolve(writer), 2)
    saved = latest(tmp_path)
    arrays, header = read_safetensors(saved.path / "learner.safetensors")
    milestone = tmp_path / "out/policies/00000002-ema.safetensors"
    digest = file_sha256(milestone)
    backend = resolve(reader)
    state = require_continuation(backend)

    def inspect(
        learner: SelfPlayLearner,
        ema: BackendPolicy | None,
        collector: Collector,
        counts: loop.Counts,
        rng: np.random.Generator,
    ) -> None:
        assert counts.model_dump(mode="json") == saved.record.extra
        assert counts.readings and counts.updates == 2
        host = capture_host(collector, rng)
        assert any(stream.recipe.actions for stream in host.collector.streams)
        assert (
            host.model_dump_json().encode()
            == (saved.path / "continuation.json").read_bytes()
        )
        assert ema is not None
        adam = state.adam_state(learner, step=counts.optimizer_updates)
        assert adam.step == int(arrays.pop("adam/step")) > 0
        assert adam.param_steps == json.loads(header["param_steps"])
        assert adam.exp_avg and any(np.any(v) for v in adam.exp_avg.values())
        actual = {
            f"{prefix}/{key}": v
            for prefix, values in (
                ("raw", state.policy_arrays(learner.agent)),
                ("ema", state.policy_arrays(ema)),
                ("adam/exp_avg", adam.exp_avg),
                ("adam/exp_avg_sq", adam.exp_avg_sq),
            )
            for key, v in values.items()
        }
        same_arrays(actual, arrays)

    with inspect_first_update(inspect):
        final = train(tmp_path, settings, backend, updates=4, resume=True)
    assert final.name == "00000004-ema.safetensors" and final.is_file()
    assert file_sha256(milestone) == digest
    assert (
        latest(tmp_path).record.extra["optimizer_updates"]
        > saved.record.extra["optimizer_updates"]
    )
    data = read(tmp_path / "track", "test", "portable")
    points = data.points("run/sampling_reseeded", "updates")
    assert list(zip(points.x, points.y, strict=True)) == [(3, 1)]
    if reader == "mlx":
        lagged = sum(
            age < saved.record.extra["optimizer_updates"]
            for age in json.loads(header["param_steps"]).values()
        )
        assert lagged > 0
        points = data.points("run/adam_steps_aligned", "updates")
        assert list(zip(points.x, points.y, strict=True)) == [(3, lagged)]


def test_torch_cuda_sampling_on_cpu(tmp_path: Path, settings: loop.SelfPlay) -> None:
    backend = resolve("torch")
    root = tmp_path
    train(root, settings, backend, updates=1)
    folder = latest(root).path
    # CUDA collector and framework generators have opaque device-specific bytes.
    # Keep the real Torch learner and host save; model only the foreign RNG file.
    sampling = folder / "sampling.safetensors"
    keys, _ = read_safetensors(sampling)
    keys["rng.collector"] = np.arange(16, dtype=np.uint8)
    keys["rng.torch.cuda"] = keys["rng.collector"].copy()
    write_safetensors(sampling, keys, {})
    learner = folder / "learner.safetensors"
    arrays, header = read_safetensors(learner)
    write_safetensors(
        learner,
        arrays,
        header
        | {
            "sampling_domain": "torch:cuda",
            "sampling_sha256": file_sha256(sampling),
        },
    )
    intact = sampling.read_bytes()
    sampling.write_bytes(intact[:-1])
    with pytest.raises(ValueError, match="sampling_sha256"):
        train(root, settings, backend, updates=2, resume=True)
    assert latest(root).path == folder
    sampling.write_bytes(intact)

    def inspect(
        learner: SelfPlayLearner,
        ema: BackendPolicy | None,
        collector: Collector,
        counts: loop.Counts,
        rng: np.random.Generator,
    ) -> None:
        import torch

        assert counts.updates == 1
        seed = 19 + counts.updates
        expected_collector = backend.collection.CollectionSession(
            device="cpu", seed=seed + loop.COLLECTION_SEED
        )
        torch.testing.assert_close(
            collector.numerics.snapshot_rng(), expected_collector.snapshot_rng()
        )
        torch.testing.assert_close(
            torch.get_rng_state(), torch.Generator().manual_seed(seed).get_state()
        )

    with inspect_first_update(inspect):
        train(root, settings, backend, updates=2, resume=True)
    points = read(root / "track", "test", "portable").points(
        "run/sampling_reseeded", "updates"
    )
    assert list(zip(points.x, points.y, strict=True)) == [(2, 1)]
    cpu, _ = read_safetensors(latest(root).path / "sampling.safetensors")
    assert cpu.keys() == {"rng.torch.cpu", "rng.collector"}


def test_tracked_mlx_continuation_without_torch(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    subprocess.run(
        [sys.executable, "-m", "tests.training.test_mlx_continuation", str(tmp_path)],
        check=True,
        timeout=120,
    )


@pytest.fixture
def legacy_save(tmp_path: Path, settings: loop.SelfPlay) -> RestoredState:
    """Write the former registered objects with real nonempty optimizer history."""
    from manabot.backends.torch.learner import Learner

    backend = resolve("torch")
    with (
        patch.dict(os.environ, ETUDE_OUT=str(tmp_path / "out")),
        loop.device_settings(settings, backend=backend),
        attempt(tmp_path) as run,
    ):
        learner, ema = loop.create(settings, seed=run.seed, backend=backend)
        assert isinstance(learner, Learner) and ema is not None
        counts = loop.Counts(model=settings.model.checkpoint_payload())
        counts.readings = loop.resume_readings(
            counts, learner.agent.readings, learner.agent.observation_space
        )
        run.resume(
            learner=learner.agent, optimizer=learner.optimizer, ema=ema, extra=counts
        )
        with loop.self_play_games(settings, run, counts, backend=backend) as collector:
            for _ in run.steps("updates", to=1):
                row = loop.step(
                    settings,
                    learner,
                    ema,
                    collector,
                    counts,
                    np.random.default_rng(1),
                    out=run.out,
                    backend=backend,
                )
                run.log(row, **loop.clocks(counts))
                run.save()
    saved = latest(tmp_path)
    assert set(saved.record.files.values()) == {"learner.pt", "ema.pt", "optimizer.pt"}
    assert saved.record.extra["optimizer_updates"] > 0
    return saved


def test_legacy_torch_save_migrates(
    tmp_path: Path, legacy_save: RestoredState, settings: loop.SelfPlay
) -> None:
    original = {p: file_sha256(p) for p in legacy_save.path.iterdir()}

    def inspect(
        learner: SelfPlayLearner,
        ema: BackendPolicy | None,
        collector: Collector,
        counts: loop.Counts,
        rng: np.random.Generator,
    ) -> None:
        import torch

        from manabot.backends.torch.learner import Learner

        assert isinstance(learner, Learner)
        assert counts.model_dump(mode="json") == legacy_save.record.extra
        for file, obj in (("learner.pt", learner.agent), ("ema.pt", ema)):
            assert isinstance(obj, torch.nn.Module)
            expected = torch.load(
                legacy_save.path / file, map_location="cpu", weights_only=True
            )
            torch.testing.assert_close(obj.state_dict(), expected, rtol=0, atol=0)
        optimizer = torch.load(
            legacy_save.path / "optimizer.pt", map_location="cpu", weights_only=True
        )
        assert optimizer["state"]
        torch.testing.assert_close(
            learner.optimizer.state_dict()["state"], optimizer["state"], rtol=0, atol=0
        )
        assert all(
            not stream.recipe.actions
            for stream in capture_host(collector, rng).collector.streams
        )

    with inspect_first_update(inspect):
        final = train(tmp_path, settings, resolve("torch"), updates=2, resume=True)
    assert final.name == "00000002-ema.safetensors" and final.is_file()
    saved = latest(tmp_path)
    assert {p.name for p in saved.path.iterdir()} == {
        "state.json",
        "learner.safetensors",
        "sampling.safetensors",
        "continuation.json",
    }
    assert (
        saved.record.extra["optimizer_updates"]
        > legacy_save.record.extra["optimizer_updates"]
    )
    assert all(file_sha256(p) == digest for p, digest in original.items())


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="requires MLX")
def test_legacy_torch_save_requires_torch(
    tmp_path: Path, legacy_save: RestoredState, settings: loop.SelfPlay
) -> None:
    with pytest.raises(
        ValueError, match="resume once on Torch to write a portable save"
    ):
        train(tmp_path, settings, resolve("mlx"), updates=2, resume=True)
    assert latest(tmp_path).path == legacy_save.path


def test_torch_cpu_safe_stop_matches_uninterrupted(
    tmp_path: Path, settings: loop.SelfPlay
) -> None:
    backend = resolve("torch")
    whole, split = tmp_path / "whole", tmp_path / "split"
    train(whole, settings, backend, updates=2)
    stop_after(split, settings, backend, 1)
    train(split, settings, backend, updates=2, resume=True)
    control, resumed = latest(whole), latest(split)
    for file in ("continuation.json", "sampling.safetensors"):
        assert (control.path / file).read_bytes() == (resumed.path / file).read_bytes()
    expected, manifest = read_safetensors(control.path / "learner.safetensors")
    actual, header = read_safetensors(resumed.path / "learner.safetensors")
    same_arrays(expected, actual)
    for metadata, saved in ((manifest, control), (header, resumed)):
        assert metadata.pop("counts_sha256") == canonical_sha256(saved.record.extra)
    assert manifest == header
    counts = loop.Counts.model_validate(control.record.extra)
    continued = loop.Counts.model_validate(resumed.record.extra)
    assert counts.updates == 2 and counts.optimizer_updates > 0
    assert counts.model_dump(exclude={"training_seconds"}) == continued.model_dump(
        exclude={"training_seconds"}
    )
