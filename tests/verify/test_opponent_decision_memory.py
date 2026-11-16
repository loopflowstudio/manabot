"""Live viewer memory agrees across independent encoders, replay and forks."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.env.memory_store import CompactObservation, MemoryStore, PrefixIdentity
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
import managym
from managym.decision_memory import read_records


def _root(disclosure: Literal["open", "closed"] = "open") -> managym.Env:
    env = managym.Env(seed=111)
    configs = Match().to_rust()
    for config in configs:
        config.disclosure = disclosure
    env.reset(configs)
    return env


def _observations(env: managym.Env) -> list[str]:
    return [
        env.observation_for_player(seat).decision_memory_json() for seat in range(2)
    ]


def test_decision_memory_metadata_and_legacy_admission() -> None:
    layout = managym.observation_layout(decision_memory=True)
    assert len(layout["decision_records"]["columns"]) == 74
    assert len(layout["decision_reference_details"]["columns"]) == 6
    assert "decision_records" not in managym.observation_layout()
    with pytest.raises(ValueError, match="v11"):
        managym.observation_layout(decision_memory=True, legacy=True)
    with pytest.raises(ValueError, match="v11"):
        ObservationSpace(
            ObservationConfig(
                policy_input_version=9,
                max_decision_records=1,
                max_decision_details=1,
                max_decision_references=1024,
                max_decision_members=16384,
            )
        )


@pytest.mark.parametrize("disclosure", ["open", "closed"])
def test_decision_memory_live_native_python_replay_fork_and_roundtrip(
    disclosure: Literal["open", "closed"],
) -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=32,
            max_decision_details=32768,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    env, replay = _root(disclosure), _root(disclosure)
    for _ in range(12):
        # Reproducible original positional choices; the engine's own validated
        # commit boundary also serves semantic Commands and built-in players.
        env.step(0)
        replay.step(0)
    assert _observations(env) == _observations(replay)
    assert any(read_records(source) for source in _observations(env))
    branch = env.clone_env()
    retained = _observations(env)
    branch.step(0)
    assert _observations(env) == retained
    for seat in range(2):
        obs = env.observation_for_player(seat)
        expected = space.encode(obs)
        actual = space.encoder.allocate()
        env.encode_observation_into(obs, actual)
        for key in (
            "decision_records",
            "decision_reference_details",
            "decision_references",
            "decision_members",
        ):
            np.testing.assert_array_equal(actual[key], expected[key])
        store = MemoryStore((32, 32768, 1024, 16384))
        prefix = store.capture(
            {key: value[None] for key, value in actual.items()},
            0,
            PrefixIdentity("native-replay", seat, "root"),
        )
        compact = CompactObservation({}, [prefix], (32, 32768, 1024, 16384))
        for key, width in (
            ("decision_records", 74),
            ("decision_reference_details", 6),
            ("decision_references", 14),
            ("decision_members", 2),
        ):
            rows = compact[key].shape[1]
            np.testing.assert_array_equal(compact[key][0], expected[key][:rows])
            assert compact[key].shape[-1] == width
        restored = managym.Observation.from_json(obs.toJSON())
        assert json.loads(restored.decision_memory_json()) == json.loads(
            obs.decision_memory_json()
        )
        for key in (
            "decision_records",
            "decision_reference_details",
            "decision_references",
            "decision_members",
        ):
            np.testing.assert_array_equal(space.encode(restored)[key], expected[key])
    env.reset(Match().to_rust())
    assert _observations(env) == ["[]", "[]"]


def test_decision_memory_scalar_overflow_and_companion_validation() -> None:
    env = _root()
    for _ in range(12):
        env.step(0)
    obs = next(
        env.observation_for_player(seat)
        for seat in range(2)
        if read_records(env.observation_for_player(seat).decision_memory_json())
    )
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=1,
            max_decision_details=1,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    with pytest.raises(managym.AgentError, match="Capacity"):
        space.encode(obs)
    with pytest.raises(ValueError, match="capacity exceeded"):
        env.encode_observation_into(obs, space.encoder.allocate())
    buffers = dict(ObservationSpace().encoder.allocate())
    buffers["decision_records"] = np.zeros((4, 74), np.float32)
    with pytest.raises(ValueError, match="companion"):
        env.encode_observation_into(obs, buffers)


def test_decision_memory_buffered_vector_and_builtin_choices_match_scalar() -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=32,
            max_decision_details=32768,
            max_decision_references=1024,
            max_decision_members=16384,
        )
    )
    configs = Match().to_rust()
    reference = managym.VectorEnv(2, seed=111, player_policies=("agent", "heuristic"))
    hot = managym.VectorEnv(2, seed=111, player_policies=("agent", "heuristic"))
    buffers = space.encoder.allocate(2)
    buffers.update(
        rewards=np.zeros(2, np.float64),
        terminated=np.zeros(2, np.uint8),
        truncated=np.zeros(2, np.uint8),
    )
    hot.set_buffers(buffers)
    reference.reset_all(configs)
    hot.reset_all_into_buffers(configs)
    retained = []
    for _ in range(8):
        rows = reference.step([0, 0])
        hot.step_into_buffers([0, 0])
        for index, (obs, _, _, _, _) in enumerate(rows):
            expected = space.encode(obs)
            retained.extend(read_records(obs.decision_memory_json()))
            for key in (
                "decision_records",
                "decision_reference_details",
                "decision_references",
                "decision_members",
            ):
                np.testing.assert_array_equal(buffers[key][index], expected[key])
    assert retained, "built-in opponent choices must reach the external viewer"


def test_headless_decision_memory_demo(tmp_path: Path) -> None:
    from manabot.backends.torch import verify_decision_memory
    from manabot.verify.decision_memory import run_demo

    run_demo(tmp_path, verify_decision_memory)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ordinary_cpu_update"]["updates"] == 1
    for disclosure in ("open", "closed"):
        assert len(summary["policies"][disclosure]) == 4
        for policy in summary["policies"][disclosure]:
            assert policy["checkpoint_reload_matches"]
            assert policy["legal_probability_max_delta"] > 0
            assert all(value > 0 for value in policy["temporal_gradient_l1"].values())
        native = json.loads((tmp_path / disclosure / "native.json").read_text())
        assert native["private_history_unchanged_after_public_play"]
        assert all(count > 0 for count in native["records_by_viewer"].values())
        commands = (tmp_path / disclosure / "commands.jsonl").read_text().splitlines()
        assert len(commands) == native["commands_replayed"]


@pytest.mark.parametrize(
    "disclosures",
    [("open", "open"), ("closed", "closed"), ("open", "closed"), ("closed", "open")],
)
def test_own_seat_option_is_default_off_and_rebases_opposing_view_only(
    disclosures: tuple[Literal["open", "closed"], Literal["open", "closed"]],
) -> None:
    default = managym.Env(seed=111)
    own = managym.Env(seed=111, decision_memory_include_own=True)
    configs = Match().to_rust()
    for config, disclosure in zip(configs, disclosures, strict=True):
        config.disclosure = disclosure
    default.reset(configs)
    own.reset(configs)
    for _ in range(12):
        default.step(0)
        own.step(0)
    assert default.state_digest() == own.state_digest()
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_details=65536,
            max_decision_references=1024,
            max_decision_members=16384,
            decision_memory_include_own=True,
        )
    )
    restored = ObservationConfig.model_validate_json(
        space.encoder.config.model_dump_json()
    )
    assert restored.decision_memory_include_own
    assert not ObservationConfig().decision_memory_include_own
    assert "decision_memory_include_own" not in ObservationConfig().model_dump()
    assert all(
        row.actor == 2
        for seat in range(2)
        for row in read_records(
            default.observation_for_player(seat).decision_memory_json()
        )
    )
    for seat in range(2):
        obs = own.observation_for_player(seat)
        records = read_records(obs.decision_memory_json())
        assert {record.actor for record in records} == {1, 2}
        native = space.encoder.allocate()
        own.encode_observation_into(obs, native)
        for key, expected in space.encode(obs).items():
            np.testing.assert_array_equal(native[key], expected)
        # Own public signals are the other seat's opponent records, with pairs
        # and reference roles rebased. Never recapture through private view.
        other_records = [
            record
            for record in read_records(
                own.observation_for_player(1 - seat).decision_memory_json()
            )
            if record.actor == 2
        ]
        own_records = [record for record in records if record.actor == 1]
        assert len(own_records) == len(other_records)
        for own_record, other_record in zip(own_records, other_records, strict=True):
            assert own_record.context.life == other_record.context.life[::-1]
            assert own_record.context.hand == other_record.context.hand[::-1]
            assert own_record.choice.kind == other_record.choice.kind
            for own_ref, other_ref in zip(
                own_record.choice.references,
                other_record.choice.references,
                strict=True,
            ):
                assert own_ref.program == other_ref.program
                assert own_ref.definition == other_ref.definition
                assert own_ref.role == (
                    0 if other_ref.role == 0 else 3 - other_ref.role
                )
    branch = own.clone_env()
    retained = own.observation_for_player(0).decision_memory_json()
    branch.step(0)
    assert own.observation_for_player(0).decision_memory_json() == retained
    own.reset(configs)
    assert own.observation_for_player(0).decision_memory_json() == "[]"
    for _ in range(12):
        own.step(0)
    assert any(
        record.actor == 1
        for record in read_records(own.observation_for_player(0).decision_memory_json())
    )
