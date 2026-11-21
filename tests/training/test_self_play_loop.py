"""Self-play uses real Etude saves and preserves published policy evidence."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from etude.train.read import read
from etude.train.tracking import Stopped
import numpy as np
import pytest
import torch

from manabot.backends.torch import policy as policy_module
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import NumericalError
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.infra.run_settings import RunSettings
from manabot.policy.readings import Reading, fresh_readings, legacy_readings
from manabot.recipes.builders import gray_ogre_self_play
from manabot.reports.time_sources import load_time_sources
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.loops import self_play as loop
from manabot.training.loops.self_play import CLOCKS, Counts, SelfPlay, self_play
from manabot.training.rollout.collection import Collector, RolloutBatch
from manabot.training.updates.selection import UpdateDiagnostics
from tests.fixtures.training_records import recording

SEED = 197


def _settings(learning: Learning | DampedPolicyLearning) -> SelfPlay:
    values = gray_ogre_self_play(horizon=2).model_dump()
    values["learning"] = learning
    return SelfPlay.model_validate(values)


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"precision": "tf32"}, "precision tf32 requires CUDA"),
        ({"precision": "bf16"}, "precision bf16 requires CUDA or MPS"),
        ({"behavior": "random"}, "both learning seats"),
        ({"batch_closure": "exact-quota"}, "both learning seats"),
        (
            {
                "learning_seats": "one",
                "batch_closure": "exact-quota",
                "games_per_stream": 2,
            },
            "requires complete games",
        ),
    ],
)
def test_settings_refuse_what_the_loop_cannot_run(
    fields: dict[str, object], refusal: str
) -> None:
    values = gray_ogre_self_play(horizon=2).model_dump()
    with pytest.raises(ValueError, match=refusal):
        SelfPlay.model_validate({**values, **fields})


@pytest.mark.parametrize(
    ("averaging", "weights"), [(None, "ema"), (0.9, "ema"), (0.9, "raw")]
)
def test_resume_keeps_adam_ema_counts_and_immutable_policies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    averaging: float | None,
    weights: Literal["raw", "ema"],
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={
            "checkpoint_every": 1,
            "checkpoint_weights": weights,
            "learning": Learning(horizon=2, epochs=1, minibatches=1, ema=averaging),
        }
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test", run="policy", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        earlier = self_play(settings, run, updates=1)
    before = earlier.read_bytes()
    with recording(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        final = self_play(settings, run, updates=2)
    assert earlier.read_bytes() == before
    selected = "raw" if averaging is None else weights
    assert final.name == f"00000002-{selected}.safetensors"
    (reported,) = load_time_sources([str(tmp_path / "track")]).runs
    with recording(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        learner, ema = loop.create(settings, seed=run.seed)
        objects: dict[str, object] = {
            "learner": learner.agent,
            "optimizer": learner.optimizer,
        }
        if ema is not None:
            objects["ema"] = ema
        counts = run.resume(**objects, extra=Counts())
        assert counts.updates == 2 and counts.optimizer_updates > 0
        assert counts.training_seconds > 0 and learner.optimizer.state
        saved = load_checkpoint(final.with_name("00000002-raw.safetensors"))[
            0
        ].state_dict()
        for name, value in learner.agent.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    if ema is not None:
        averaged = load_checkpoint(final.with_name("00000002-ema.safetensors"))[
            0
        ].state_dict()
        for name, value in ema.state_dict().items():
            torch.testing.assert_close(value, averaged[name], rtol=0, atol=0)
    recorded = read(tmp_path / "track", "test", "policy")
    (population,) = recorded.read_table("training_deck_cards", "updates")
    assert population.x == 0
    assert {row["card"] for row in population.rows} == {
        "lea.gray_ogre",
        "lea.mountain",
        "lea.forest",
        "lea.llanowar_elves",
    }
    assert all(row["board"] == "main" and row["copies"] > 0 for row in population.rows)
    (settings_table,) = recorded.read_table("training_run", "updates")
    assert settings_table.rows[0]["learning_seats"] == settings.learning_seats
    (after_inspection,) = load_time_sources([str(tmp_path / "track")]).runs
    assert after_inspection.updates == reported.updates
    assert after_inspection.learner_transitions == reported.learner_transitions
    assert after_inspection.active_seconds == reported.active_seconds
    assert len(after_inspection.members) == 3
    assert any("observed attempts only" in w for w in after_inspection.warnings)
    assert list(recorded.points("coordinates/updates", "updates").x) == [1, 2]
    assert {cp.at["updates"] for cp in recorded.checkpoints} == {0, 1, 2}
    for checkpoint in recorded.checkpoints:
        update = int(checkpoint.at["updates"])
        export = final.with_name(f"{update:08d}-{selected}.safetensors")
        assert (
            checkpoint.artifact.sha256
            == hashlib.sha256(export.read_bytes()).hexdigest()
        )
    # The training-time report reads work and active time from the loop's clocks.
    assert reported.updates == 2
    assert reported.learner_transitions == counts.learner_transitions > 0
    assert reported.active_seconds is not None and reported.active_seconds > 0


@pytest.mark.parametrize(
    ("field", "legacy_state"), [("focus_film", False), ("stream_norm", True)]
)
def test_resume_refuses_changed_model_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, legacy_state: bool
) -> None:
    settings = gray_ogre_self_play(horizon=2)
    out = tmp_path / "out"
    monkeypatch.setenv("ETUDE_OUT", str(out))
    with recording(
        experiment="test", run="policy", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        self_play(settings, run, updates=1)
    (state,) = (out / "state").glob("save-*/state.json")
    saved = json.loads(state.read_text())
    assert saved["extra"]["model"] == settings.model.checkpoint_payload()
    initial = out / "policies/00000000-raw.safetensors"
    assert "model" not in loop.load_metadata(initial)["bc"]["counts"]
    if legacy_state:
        del saved["extra"]["model"]
        state.write_text(json.dumps(saved))
    else:
        initial.unlink()  # A run with recorded settings is self-contained.
    changed = settings.model_copy(
        update={"model": settings.model.model_copy(update={field: True})}
    )
    with recording(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        with pytest.raises(ValueError, match=f"{field}: False -> True"):
            self_play(changed, run, updates=2)
        if legacy_state:
            assert self_play(settings, run, updates=2).is_file()


def test_resume_refuses_a_model_that_reads_its_inputs_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={"learning": Learning(horizon=2, epochs=1, minibatches=1)}
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test", run="policy", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        self_play(settings, run, updates=1)

    # The rule changes between the start and the resume: timestamps read as stored.
    def changed(space: ObservationSpace) -> dict[str, tuple[Reading, ...]]:
        readings = fresh_readings(space)
        return {
            key: tuple(
                Reading.AS_IS if value is Reading.RANK else value for value in values
            )
            for key, values in readings.items()
        }

    monkeypatch.setattr(policy_module, "fresh_readings", changed)
    with recording(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        with pytest.raises(
            ValueError, match=r"agent_permanents\.timestamp: rank -> as_is"
        ):
            self_play(settings, run, updates=2)


def test_resume_refuses_a_state_saved_before_readings_were_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={"learning": Learning(horizon=2, epochs=1, minibatches=1)}
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test", run="policy", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        self_play(settings, run, updates=1)
    # As written before readings were recorded: the model read legacy readings.
    (state,) = (tmp_path / "out" / "state").glob("save-*/state.json")
    saved = json.loads(state.read_text())
    del saved["extra"]["readings"]
    state.write_text(json.dumps(saved))

    with recording(
        experiment="test",
        run="policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        with pytest.raises(
            ValueError,
            match=r"written before input readings were recorded.*"
            r"actions\.ability: as_is -> one_hot:15",
        ):
            self_play(settings, run, updates=2)


def test_resume_readings_records_a_new_run_and_accepts_a_legacy_state() -> None:
    space = ObservationSpace(gray_ogre_self_play(horizon=2).observation)
    fresh, legacy = fresh_readings(space), legacy_readings(space)
    assert loop.resume_readings(Counts(), fresh, space) == {
        key: [str(value) for value in values] for key, values in fresh.items()
    }
    # A run warm-started from a legacy policy was built with legacy readings.
    resumed = Counts.model_validate({"updates": 3})
    assert loop.resume_readings(resumed, legacy, space) == {
        key: [str(value) for value in values] for key, values in legacy.items()
    }


def test_resume_accepts_historical_memory_readings() -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=8,
            max_decision_references=16,
            max_decision_reference_details=32,
            max_decision_members=16,
        )
    )
    readings = fresh_readings(space)
    historical = Counts(
        updates=3,
        readings={
            key: [str(value) for value in values]
            for key, values in readings.items()
            if key not in MEMORY_FIELDS
        },
    )
    assert loop.resume_readings(historical, readings, space) == {
        key: [str(value) for value in values] for key, values in readings.items()
    }


def test_stop_keeps_registered_counts_at_the_completed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = gray_ogre_self_play(horizon=2)
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    stop = tmp_path / "stop"
    monkeypatch.setenv("ETUDE_STOP_FILE", str(stop))
    original_step = loop.step

    def stop_after_step(
        settings: SelfPlay,
        learner: Learner,
        ema: ActorCritic | None,
        collector: Collector,
        counts: Counts,
        rng: np.random.Generator,
        *,
        out: Path,
    ) -> dict[str, object]:
        row = original_step(settings, learner, ema, collector, counts, rng, out=out)
        coordinates = {k: v for k, v in row.items() if k.startswith("coordinates/")}
        assert coordinates["coordinates/updates"] == 1
        assert "coordinates/model" not in coordinates
        assert "coordinates/readings" not in coordinates
        assert all(isinstance(v, (int, float)) for v in coordinates.values())
        stop.touch()
        return row

    with monkeypatch.context() as patch:
        patch.setattr(loop, "step", stop_after_step)
        with pytest.raises(Stopped):
            with recording(
                experiment="test",
                run="stop",
                clocks=CLOCKS,
                directory=tmp_path / "track",
            ) as run:
                self_play(settings, run, updates=2)
    pointer = json.loads((tmp_path / "out/state/current.json").read_text())
    saved = json.loads(
        (tmp_path / "out/state" / pointer["save"] / "state.json").read_text()
    )
    assert saved["extra"]["updates"] == saved["at"]["updates"] == 1
    stop.unlink()
    with recording(
        experiment="test",
        run="stop",
        clocks=CLOCKS,
        directory=tmp_path / "track",
        resume=True,
    ) as run:
        assert self_play(settings, run, updates=2).name == "00000002-raw.safetensors"


@pytest.mark.parametrize("capture_failure", [False, True])
def test_numerical_failure_keeps_previous_policy_and_private_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_failure: bool
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={"learning": Learning(horizon=2, epochs=1, minibatches=1, ema=0.9)}
    )
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    with recording(
        experiment="test", run="failure", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        policy = self_play(settings, run, updates=1)
    original = policy.read_bytes()
    pointer = (tmp_path / "out/state/current.json").read_bytes()

    def fail(
        settings: SelfPlay,
        learner: Learner,
        ema: ActorCritic | None,
        counts: Counts,
        batch: RolloutBatch,
        rng: np.random.Generator,
    ) -> UpdateDiagnostics:
        raise NumericalError(
            "fixture_nonfinite", {"values": torch.tensor([float("nan")])}
        )

    monkeypatch.setattr(loop, "learn", fail)
    if capture_failure:
        (tmp_path / "out/numerical-incident").touch()
    with pytest.raises(NumericalError, match="fixture_nonfinite") as caught:
        with recording(
            experiment="test",
            run="failure",
            clocks=CLOCKS,
            directory=tmp_path / "track",
            resume=True,
        ) as run:
            self_play(settings, run, updates=2)
    assert policy.read_bytes() == original
    assert (tmp_path / "out/state/current.json").read_bytes() == pointer
    assert not (tmp_path / "out/policies/00000002-raw.safetensors").exists()
    if capture_failure:
        assert any("incident capture failed" in note for note in caught.value.__notes__)
    else:
        receipt = next((tmp_path / "out/numerical-incident").glob("*/incident.json"))
        metadata = json.loads(receipt.read_text())
        assert metadata["iteration"] == 2 and metadata["completed_updates"] == 1
        assert "packed_actions/offsets" in metadata["tensors"]
        diagnostic = torch.load(receipt.with_name("state.pt"), weights_only=False)
        assert diagnostic["optimizer"]["state"] and diagnostic["ema"]


def test_live_ppo_requires_a_horizon_but_retained_records_omit_it() -> None:
    learning = Learning()
    assert "horizon" not in learning.model_dump()
    assert "horizon" not in learning.model_dump_json()
    with pytest.raises(ValueError, match="PPO self-play requires a learning horizon"):
        _settings(learning)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Learning.model_validate({"horizon": invalid})
    fixed = Learning(horizon=2)
    assert Learning.model_validate_json(fixed.model_dump_json()) == fixed
    assert fixed.learning_rate.at(1) == fixed.learning_rate.at(2)
    assert fixed.tau.at(1) == fixed.tau.at(2)


def test_device_memory_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 12)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 45)
    cuda = loop.resource_metrics("cuda")
    assert cuda["memory/peak_allocated_cuda_bytes"] == 123
    assert cuda["memory/peak_reserved_cuda_bytes"] == 456
    mps = loop.resource_metrics("mps")
    assert mps["memory/allocated_mps_bytes"] == 12
    assert mps["memory/driver_allocated_mps_bytes"] == 45


def test_damped_forward_floor_records_two_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = DampedPolicyLearning(gradient="damped_policy")
    assert "reference_forward_kl" not in default.model_dump()
    with pytest.raises(ValueError):
        DampedPolicyLearning(gradient="damped_policy", reference_forward_kl=-1)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        reference_forward_kl=0.002,
        advantage_quantile=0,
        min_advantage=0,
    )
    assert learning.model_dump()["reference_forward_kl"] == 0.002
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("ETUDE_SEED", str(SEED))
    with recording(
        experiment="test", run="floor", clocks=CLOCKS, directory=tmp_path / "track"
    ) as run:
        self_play(_settings(learning), run, updates=2)
    data = read(tmp_path / "track", "test", "floor")
    assert list(
        data.points("reference_forward_kl_coefficient", "updates").y
    ) == pytest.approx([0.002] * 2)
    measured = data.points("reference_forward_kl", "updates")
    assert list(measured.x) == [1, 2]
    assert np.isfinite(measured.y).all() and np.any(np.asarray(measured.y) > 0)


def test_combat_probe_cadence_and_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(Learning(horizon=4, epochs=1, minibatches=1, ema=0.9))
    states = []
    for every in (0, 1, 2):
        name = f"probe-{every}"
        config = settings.model_copy(
            update={
                "run_settings": RunSettings(
                    combat_probe_every=every, combat_probe_rows=4
                )
            }
        )
        monkeypatch.setenv("ETUDE_OUT", str(tmp_path / name))
        track = tmp_path / "track"
        with recording(
            experiment="test", run=name, clocks=CLOCKS, directory=track, seed=SEED
        ) as run:
            self_play(config, run, updates=4)
        recorded = read(track, "test", name)
        expected = list(range(every, 5, every)) if every else []
        for key in (
            "combat/attack/gap",
            "combat/attack/shift/my_creatures_9_9",
            "time/combat_probe",
        ):
            assert list(recorded.points(key, "updates").x) == expected
        for sample in recorded.points("combat/attack/decisions", "updates").y:
            assert 1 <= sample <= 4
        seconds = sum(
            recorded.points(key, "updates").y
            for key in ("time/collection", "time/learning")
        )
        np.testing.assert_allclose(
            recorded.points("coordinates/training_seconds", "updates").y,
            np.cumsum(seconds),
        )
        with recording(
            experiment="test",
            run=name,
            clocks=CLOCKS,
            directory=track,
            resume=True,
            seed=SEED,
        ) as run:
            learner, ema = loop.create(config, seed=SEED)
            assert ema is not None
            run.resume(
                learner=learner.agent,
                optimizer=learner.optimizer,
                ema=ema,
                extra=Counts(),
            )
            states.append(
                (
                    learner.agent.state_dict(),
                    ema.state_dict(),
                    learner.optimizer.state_dict(),
                )
            )
    for state in states[1:]:
        torch.testing.assert_close(state, states[0], rtol=0, atol=0)


def test_combat_probe_run_settings_preserve_default_serialization() -> None:
    assert (
        not {"combat_probe_rows", "combat_probe_every"}
        & RunSettings().model_dump().keys()
    )
    custom = RunSettings(combat_probe_every=0, combat_probe_rows=2)
    assert RunSettings.model_validate_json(custom.model_dump_json()) == custom
