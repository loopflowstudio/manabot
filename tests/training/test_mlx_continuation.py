"""Run real Etude callback saves, restore and failure capture without Torch."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


def bounded_self_play() -> "SelfPlay":
    from manabot.recipes import baseline

    return baseline.settings(
        baseline.BaselineConfig(backend="mlx", device="mps")
    ).model_copy(
        update={
            "streams": 2,
            "transitions": 8,
            "learning_seats": "one",
            "batch_closure": "exact-quota",
            "bucketed_shapes": True,
            "checkpoint_every": 1,
        }
    )


@pytest.mark.parametrize(
    "writer,reader,message",
    [
        ("torch", "mlx", "expected an MLX callback save"),
        ("mlx", "torch", "saved state has no object"),
    ],
)
def test_tracked_state_refuses_a_different_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    reader: str,
    message: str,
) -> None:
    pytest.importorskip("mlx.core")
    from manabot.backends import resolve
    from manabot.training.loops.self_play import CLOCKS, self_play
    from tests.fixtures.training_records import recording

    out = tmp_path / "out"
    monkeypatch.setenv("ETUDE_OUT", str(out))
    settings = bounded_self_play().model_copy(update={"device": "cpu"})
    with recording(
        experiment="test", run="cross", directory=tmp_path / "track", clocks=CLOCKS
    ) as run:
        policy = self_play(settings, run, updates=1, backend=resolve(writer))
    before = (out / "state/current.json").read_bytes(), policy.read_bytes()
    with recording(
        experiment="test",
        run="cross",
        directory=tmp_path / "track",
        clocks=CLOCKS,
        resume=True,
    ) as run:
        with pytest.raises(ValueError, match=message):
            self_play(settings, run, updates=2, backend=resolve(reader))
    assert before == ((out / "state/current.json").read_bytes(), policy.read_bytes())
    assert not policy.with_name("00000002-raw.safetensors").exists()


def test_tracked_mlx_continuation_without_torch(tmp_path: Path) -> None:
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("requires MLX")
    subprocess.run(
        [sys.executable, "-m", "tests.training.test_mlx_continuation", str(tmp_path)],
        check=True,
        timeout=120,
    )


def exercise_continuation(root: Path) -> None:
    from etude.train.state import read_state
    import mlx.core as mx

    from manabot.backends import resolve
    from manabot.backends.mlx import state
    from manabot.backends.mlx.rl import NumericalError
    from manabot.training.loops import self_play as loop
    from tests.fixtures.training_records import recording

    backend = resolve("mlx")
    settings = bounded_self_play()
    out = root / "out"
    os.environ["ETUDE_OUT"] = str(out)

    def attempt(updates: int, *, resume: bool = True) -> Path:
        with recording(
            experiment="test",
            run="mlx",
            seed=19,
            clocks=loop.CLOCKS,
            directory=root / "track",
            resume=resume,
        ) as run:
            return loop.self_play(settings, run, updates=updates, backend=backend)

    milestone = attempt(1, resume=False)
    original = milestone.read_bytes()
    saved = read_state(out / "state")
    assert saved is not None and not saved.record.files
    counts = loop.Counts.model_validate(saved.record.extra)
    assert counts.updates == 1 and counts.optimizer_updates > 0 and counts.readings
    snapshot = saved.path / "training.safetensors"
    arrays, metadata = mx.load(snapshot, return_metadata=True)
    assert int(arrays["adam.step"]) == counts.optimizer_updates
    assert any(key.startswith("ema.") for key in arrays)
    moments = [
        v for k, v in arrays.items() if k.startswith("adam.") and k.endswith(".m")
    ]
    assert any(bool(mx.any(v != 0)) for v in moments)
    # Rebuild independently so the comparisons also catch an omitted restore.
    with recording(
        experiment="test",
        run="mlx",
        seed=19,
        clocks=loop.CLOCKS,
        directory=root / "track",
        resume=True,
    ) as run:
        learner, ema = loop.create(settings, seed=37, backend=backend)
        state.restore(run, learner, ema, run.resume(extra=loop.Counts()))
        live = state.training_arrays(learner, ema)
        assert live.keys() == arrays.keys()
        for name, value in live.items():
            np.testing.assert_array_equal(value, arrays[name], err_msg=name)

    moment = next(
        name for name in arrays if name.startswith("adam.") and name.endswith(".m")
    )
    for invalid, header, match in (
        (arrays, {**metadata, "backend": "torch"}, "backend"),
        (arrays, {**metadata, "schema": "0"}, "schema"),
        ({k: v for k, v in arrays.items() if k != "adam.step"}, metadata, "keys"),
        ({**arrays, "adam.step": mx.array([1])}, metadata, "shape or dtype"),
        (
            {**arrays, moment: mx.full_like(arrays[moment], float("nan"))},
            metadata,
            "nonfinite",
        ),
    ):
        mx.save_safetensors(snapshot, invalid, header)
        with pytest.raises(ValueError, match=match):
            attempt(2)
    snapshot.unlink()
    with pytest.raises(ValueError, match="MLX callback save"):
        attempt(2)
    mx.save_safetensors(snapshot, arrays, metadata)
    manifest = saved.path / "state.json"
    original_manifest = manifest.read_bytes()
    changed = json.loads(original_manifest)
    changed["extra"]["readings"]["actions"][0] = "fixture"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reads its inputs differently"):
        attempt(2)
    manifest.write_bytes(original_manifest)

    # A late failed row must survive the evidence cap and the last save must
    # remain the recovery point even though the attempt's clock advanced.
    error = NumericalError(
        "fixture_loss",
        {
            "failed": mx.array([False, True]),
            "minibatch_indexes": mx.array([0, 5]),
            "values": mx.array([0, float("nan")]),
        },
    )
    error.tensors["minibatch_observation/legal"] = np.ones((2, 4), dtype=bool)
    with (
        patch.object(loop, "learn", side_effect=error),
        patch("manabot.training.state.incidents.EVIDENCE_BYTES", 256),
        pytest.raises(NumericalError),
    ):
        attempt(2)
    assert not getattr(error, "__notes__", [])
    (receipt,) = (out / "numerical-incident").glob("*/incident.json")
    info = json.loads(receipt.read_text())
    assert info["invariant"] == "fixture_loss" and info["completed_updates"] == 1
    assert info["tensors"]["batch_layout"]["selected_flat_indexes"] == [5]
    evidence = mx.load(receipt.with_name("tensors.safetensors"))
    assert np.isnan(np.asarray(evidence["values"])).all()
    assert evidence["minibatch_observation/legal"].shape == (1, 4)
    assert any(t.get("omitted") for t in info["tensors"].values())
    assert sum(v.nbytes for v in evidence.values()) <= info["evidence_limit_bytes"]
    incident = mx.load(receipt.with_name("state.safetensors"))
    assert int(incident["adam.step"]) == counts.optimizer_updates
    assert {"rng.mlx", "rng.collector"} <= incident.keys()
    assert read_state(out / "state").path == saved.path
    final = attempt(2)
    assert milestone.read_bytes() == original
    assert final.name == "00000002-ema.safetensors"
    latest = read_state(out / "state")
    resumed = loop.Counts.model_validate(latest.record.extra)
    assert resumed.updates == 2 and resumed.optimizer_updates > counts.optimizer_updates
    assert (
        int(mx.load(latest.path / snapshot.name)["adam.step"])
        == resumed.optimizer_updates
    )


if __name__ == "__main__":
    assert "torch" not in sys.modules
    sys.modules["torch"] = None
    exercise_continuation(Path(sys.argv[1]))
