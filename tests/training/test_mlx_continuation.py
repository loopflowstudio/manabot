"""Run real Etude callback saves, restore and failure capture without Torch."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from manabot.policy.protocol import BackendPolicy
    from manabot.training.learning import Precision
    from manabot.training.loops.self_play import Counts, SelfPlay
    from manabot.training.rollout.collection import Collector
    from manabot.training.state.backend import SelfPlayBackend, SelfPlayLearner


@pytest.mark.parametrize(
    "width,precision,device",
    [
        (64, "float32", "mps"),
        (64, "bf16", "mps"),
        (256, "float32", "mps"),
        (256, "bf16", "mps"),
        (256, "float32", "cpu"),
    ],
)
def test_mlx_iteration_uses_recipe_precision(
    tmp_path: Path, width: int, precision: "Precision", device: str
) -> None:
    pytest.importorskip("mlx.core")
    from manabot.backends import resolve
    from manabot.training.loops.self_play import fresh_iteration
    from tests.fixtures.self_play import bounded_self_play

    recipe = bounded_self_play()
    settings = recipe.model_copy(
        update={
            "model": recipe.model.model_copy(update={"hidden_dim": width}),
            "precision": precision,
            "device": device,
        }
    )
    result = fresh_iteration(
        settings, backend=resolve("mlx"), seed=19, out=tmp_path / "run"
    )
    assert result.counts.learner_transitions >= settings.transitions
    assert result.counts.optimizer_updates > 0
    assert result.ema is not None
    expected = "bfloat16" if precision == "bf16" else "float32"
    assert result.raw.precision == result.ema.precision == expected
    assert result.checkpoint.is_file()


def exercise_continuation(root: Path) -> None:
    from etude.train.state import read_state
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from manabot.backends import resolve
    from manabot.backends.mlx import state
    from manabot.backends.mlx.rl import NumericalError
    from manabot.training.loops import self_play as loop
    from tests.fixtures.self_play import bounded_self_play
    from tests.fixtures.training_records import recording

    backend = resolve("mlx")
    settings = bounded_self_play().model_copy(update={"precision": "bf16"})
    out = root / "out"
    os.environ["ETUDE_OUT"] = str(out)

    def attempt(
        updates: int, *, resume: bool = True, recipe: "SelfPlay" = settings
    ) -> Path:
        with recording(
            experiment="test",
            run="mlx",
            seed=19,
            clocks=loop.CLOCKS,
            directory=root / "track",
            resume=resume,
        ) as run:
            return loop.self_play(recipe, run, updates=updates, backend=backend)

    milestone = attempt(1, resume=False)
    original = milestone.read_bytes()
    with pytest.raises(ValueError, match="learner.safetensors: saved precision"):
        attempt(2, recipe=settings.model_copy(update={"precision": "float32"}))
    # A portable policy can start a fresh learner in the recipe's precision.
    portable, _ = backend.load_policy(milestone)
    warm, _ = loop.create(
        settings.model_copy(update={"pretrained": milestone}), seed=37, backend=backend
    )
    assert portable.precision == "float32" and warm.precision == "bfloat16"
    restored_weights = warm.parameters()
    for name, value in tree_flatten(portable.parameters()):
        np.testing.assert_array_equal(restored_weights[name], value, err_msg=name)
    saved = read_state(out / "state")
    assert saved is not None
    assert {p.name for p in saved.path.iterdir()} == {
        "state.json",
        "learner.safetensors",
        "sampling.safetensors",
        "continuation.json",
    }
    counts = loop.Counts.model_validate(saved.record.extra)
    assert counts.updates == 1 and counts.optimizer_updates > 0 and counts.readings
    snapshot = saved.path / "learner.safetensors"
    arrays, metadata = mx.load(snapshot, return_metadata=True)
    assert int(arrays["adam/step"]) == counts.optimizer_updates
    assert any(key.startswith("ema/") for key in arrays)
    moments = [v for k, v in arrays.items() if k.startswith("adam/exp_avg/")]
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
        _, restored = loop.resume_training(settings, run, backend, learner, ema)
        assert restored is not None and restored.sampling is not None
        host_file = saved.path / "continuation.json"
        payload = host_file.read_bytes()
        assert restored.host.model_dump_json().encode() == payload
        assert metadata["continuation_sha256"] == hashlib.sha256(payload).hexdigest()
        assert ema is not None
        adam = state.adam_state(learner, step=counts.optimizer_updates)
        assert adam.param_steps == json.loads(metadata["param_steps"])
        for prefix, values in (
            ("raw", state.policy_arrays(learner.agent)),
            ("ema", state.policy_arrays(ema)),
            ("adam/exp_avg", adam.exp_avg),
            ("adam/exp_avg_sq", adam.exp_avg_sq),
        ):
            for name, value in values.items():
                np.testing.assert_array_equal(
                    value, arrays[f"{prefix}/{name}"], err_msg=name
                )

    moment = next(name for name in arrays if name.startswith("adam/exp_avg/"))
    for invalid, header, match in (
        (arrays, {**metadata, "precision": "float16"}, "precision"),
        (arrays, {**metadata, "schema": "0"}, "schema"),
        ({k: v for k, v in arrays.items() if k != "adam/step"}, metadata, "adam/step"),
        ({**arrays, "adam/step": mx.array([1])}, metadata, "shape or dtype"),
        (
            {**arrays, moment: mx.full_like(arrays[moment], float("nan"))},
            metadata,
            "nonfinite",
        ),
    ):
        mx.save_safetensors(snapshot, invalid, header)
        with pytest.raises(ValueError, match=match):
            attempt(2)
    current = (out / "state/current.json").read_bytes()
    for damaged, message in (
        (None, "missing"),
        (payload + b" ", "continuation_sha256"),
        (payload[:-1], "continuation_sha256"),
    ):
        mx.save_safetensors(snapshot, arrays, metadata)
        if damaged is None:
            host_file.unlink()
        else:
            host_file.write_bytes(damaged)
        with pytest.raises(ValueError, match=message):
            attempt(2)
    malformed = payload[:-1]
    host_file.write_bytes(malformed)
    mx.save_safetensors(
        snapshot,
        arrays,
        metadata | {"continuation_sha256": hashlib.sha256(malformed).hexdigest()},
    )
    with pytest.raises(ValueError, match="Invalid JSON"):
        attempt(2)
    host_file.write_bytes(payload)
    sampling_file = saved.path / "sampling.safetensors"
    original_sampling = sampling_file.read_bytes()
    sampling_file.write_bytes(original_sampling[:-1])
    mx.save_safetensors(snapshot, arrays, metadata)
    with pytest.raises(ValueError, match="sampling_sha256"):
        attempt(2)
    # Even a correctly bound sampling file must be admissible before Adam or
    # either live policy changes. Start from unrelated weights to detect copying.
    mx.save_safetensors(
        sampling_file, {"rng.collector": mx.array(0), "rng.mlx": mx.array(0)}
    )
    mx.save_safetensors(
        snapshot,
        arrays,
        metadata
        | {"sampling_sha256": hashlib.sha256(sampling_file.read_bytes()).hexdigest()},
    )
    with recording(
        experiment="test",
        run="mlx",
        seed=19,
        clocks=loop.CLOCKS,
        directory=root / "track",
        resume=True,
    ) as run:
        probe, probe_ema = loop.create(settings, seed=41, backend=backend)
        before = {
            k: np.array(v) for k, v in state.training_arrays(probe, probe_ema).items()
        }
        with pytest.raises(ValueError, match=r"sampling.safetensors: rng"):
            loop.resume_training(settings, run, backend, probe, probe_ema)
        after = state.training_arrays(probe, probe_ema)
        assert after.keys() == before.keys()
        for name, value in before.items():
            np.testing.assert_array_equal(after[name], value, err_msg=name)
    sampling_file.write_bytes(original_sampling)
    changed_host = json.loads(payload)
    changed_host["collector"]["streams"][0]["recipe"]["actions"].pop()
    changed_bytes = json.dumps(changed_host).encode()
    host_file.write_bytes(changed_bytes)
    mx.save_safetensors(
        snapshot,
        arrays,
        metadata | {"continuation_sha256": hashlib.sha256(changed_bytes).hexdigest()},
    )
    raw = milestone.with_name("00000001-raw.safetensors")
    pending = raw.rename(raw.with_suffix(".pending"))
    with pytest.raises(ValueError, match="collector root mismatch"):
        attempt(2)
    assert not raw.exists()
    pending.rename(raw)
    host_file.write_bytes(payload)
    mx.save_safetensors(snapshot, arrays, metadata)
    with pytest.raises(ValueError, match="stream geometry"):
        attempt(2, recipe=settings.model_copy(update={"streams": 3}))
    assert (out / "state/current.json").read_bytes() == current
    assert milestone.read_bytes() == original
    assert not milestone.with_name("00000002-raw.safetensors").exists()
    # The retired MLX-only writer did not provide portable optimizer state.
    old = saved.path / "training.safetensors"
    old.touch()
    with pytest.raises(ValueError, match="training.safetensors"):
        attempt(2)
    old.unlink()
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
        int(mx.load(latest.path / snapshot.name)["adam/step"])
        == resumed.optimizer_updates
    )


@pytest.mark.parametrize(
    ("stop_after", "scenario"),
    [(0, "self"), (1, "self"), (1, "random"), (1, "complete"), (1, "metal")],
)
def test_safe_stop_matches_uninterrupted(
    tmp_path: Path, stop_after: int, scenario: str
) -> None:
    pytest.importorskip("mlx.core")
    from etude.train.state import read_state
    import mlx.core as mx

    runs = [("control", "whole"), ("split", "stop"), ("split", "resume")]
    if scenario == "metal":
        runs.append(("repeat", "whole"))
    for arm, phase in runs:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.training.test_mlx_continuation",
                str(tmp_path / arm),
                phase,
                str(stop_after),
                scenario,
            ],
            check=True,
            timeout=120,
        )
    control = read_state(tmp_path / "control/out/state")
    assert control is not None
    if scenario == "complete":
        assert control.record.extra["optimizer_updates"] == 2
    expected = mx.load(control.path / "learner.safetensors")
    for arm in ["split", "repeat"] if scenario == "metal" else ["split"]:
        actual = read_state(tmp_path / arm / "out/state")
        assert actual is not None
        assert (control.path / "continuation.json").read_bytes() == (
            actual.path / "continuation.json"
        ).read_bytes()
        assert (tmp_path / "control/entry.json").read_bytes() == (
            tmp_path / arm / "entry.json"
        ).read_bytes()
        assert (actual.path / "sampling.safetensors").read_bytes() == (
            control.path / "sampling.safetensors"
        ).read_bytes()
        arrays = mx.load(actual.path / "learner.safetensors")
        assert arrays.keys() == expected.keys()
        for name, value in arrays.items():
            if scenario == "metal" and name.startswith(("raw/", "ema/", "adam/")):
                np.testing.assert_allclose(
                    value, expected[name], rtol=0, atol=1e-6, err_msg=name
                )
            else:
                np.testing.assert_array_equal(value, expected[name], err_msg=name)
        counts = control.record.extra.copy()
        resumed = actual.record.extra.copy()
        del counts["training_seconds"], resumed["training_seconds"]
        assert counts == resumed

    if scenario == "random":
        # Each attempt closes its live games. Compare the rows emitted after
        # update two, and count actual policy decisions across both attempts.
        from managym.statistics import read_statistics

        totals = []
        summaries = []
        for arm in ("control", "split"):
            out = tmp_path / arm / "out"
            summaries.append(
                sorted(
                    (row.stream, row.model_dump_json(exclude={"context", "generation"}))
                    for row in read_statistics(out / "statistics")
                    if row.context["update"] == 2
                )
            )
            totals.append(
                sum(
                    counts["decisions"]
                    for file in (out / "failures").glob("action-cutoff-*.json")
                    for counts in json.loads(file.read_bytes())[
                        "counts_by_kind"
                    ].values()
                )
            )
        assert summaries[0] and summaries[0] == summaries[1]
        assert totals[0] > 0 and totals[0] == totals[1]


def safe_stop_attempt(root: Path, phase: str, stop_after: int, scenario: str) -> None:
    from etude.train.tracking import Stopped

    from manabot.backends import resolve
    from manabot.infra.config import MatchConfig
    from manabot.infra.run_settings import RunSettings
    from manabot.training.loops import self_play as loop
    from manabot.training.rollout.collection import capture_collector
    from manabot.training.state.game_errors import GameErrorPolicy
    from tests.fixtures.self_play import bounded_self_play
    from tests.fixtures.training_records import recording

    root.mkdir(exist_ok=True)
    stop = root / "stop"
    os.environ.update(ETUDE_OUT=str(root / "out"), ETUDE_STOP_FILE=str(stop))
    settings = bounded_self_play().model_copy(
        update={
            "device": "mps" if scenario == "metal" else "cpu",
            "precision": "float32",
            "workers": 1 if phase == "resume" else 2,
        }
    )
    if scenario == "metal":
        # Compare device repeatability on one small optimizer batch.
        settings = settings.model_copy(
            update={
                "transitions": 4,
                "model": settings.model.model_copy(
                    update={"hidden_dim": 32, "attention_layers": 1}
                ),
            }
        )
    if scenario == "random":
        settings = settings.model_copy(
            update={
                "behavior": "random",
                "run_settings": RunSettings(record_games=1, statistics=True),
                "observation": settings.observation.model_copy(
                    update={"action_overflow": "first-legal"}
                ),
                "deck_source": None,
                "match": MatchConfig(),
                "game_errors": GameErrorPolicy(recover_capacity=True),
            }
        )
    elif scenario == "complete":
        settings = settings.model_copy(
            update={
                "batch_closure": "complete-games",
                "learning_seats": "both",
                "games_per_stream": 1,
                # One optimizer step per update, independent of game length.
                "learning": settings.learning.model_copy(
                    update={
                        "filter_scope": "actor_critic",
                        "filter_kind": "top_count",
                        "advantage_quantile": 1.0,
                        "min_advantage": 0.0,
                    }
                ),
                "deck_source": None,
                "match": MatchConfig(
                    hero_deck={"Mountain": 8, "Gray Ogre": 4},
                    villain_deck={"Island": 8, "Wind Drake": 4},
                ),
            }
        )
    original_step = loop.step

    def observed(
        settings: "SelfPlay",
        learner: "SelfPlayLearner",
        ema: "BackendPolicy | None",
        collector: "Collector",
        counts: "Counts",
        rng: np.random.Generator,
        *,
        out: Path,
        backend: "SelfPlayBackend",
    ) -> dict[str, object]:
        # Exercise the host RNGs even when the damped objective needs no shuffle.
        draws = [random.random(), np.random.random(), rng.permutation(8).tolist()]
        if counts.updates == stop_after:
            if scenario == "random":
                assert any(
                    any(root.annotations)
                    for root in capture_collector(collector).streams
                )
                assert collector.stats.learned_game_rows
                assert collector._game_actions == [
                    list(r.recipe.actions) for r in capture_collector(collector).streams
                ]
            (root / "entry.json").write_text(
                json.dumps(
                    {
                        "host_draws": draws,
                        "streams": capture_collector(collector).model_dump(mode="json"),
                        "key": np.asarray(collector.numerics.snapshot_rng()).tolist(),
                        "minibatch": rng.bit_generator.state,
                    },
                    sort_keys=True,
                )
            )
        row = original_step(
            settings, learner, ema, collector, counts, rng, out=out, backend=backend
        )
        if phase == "stop" and counts.updates == stop_after:
            stop.touch()
        return row

    if phase == "stop" and stop_after == 0:
        stop.touch()
    if phase == "resume":
        stop.unlink()
    with (
        patch.object(loop, "step", observed),
        pytest.raises(Stopped) if phase == "stop" else nullcontext(),
        recording(
            experiment="test",
            run="cpu",
            seed=19,
            clocks=loop.CLOCKS,
            directory=root / "track",
            resume=phase == "resume",
        ) as run,
    ):
        loop.self_play(settings, run, updates=2, backend=resolve("mlx"))


if __name__ == "__main__":
    assert "torch" not in sys.modules
    sys.modules["torch"] = None
    if len(sys.argv) == 2:
        exercise_continuation(Path(sys.argv[1]))
    else:
        safe_stop_attempt(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
