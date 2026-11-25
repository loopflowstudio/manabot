"""The native batch is the only observation transport.

Policy input v11 reaches a policy as `managym.ObservationBatch` rows. The dense
dictionary transport removed in 0.3.0 must not return under its old names.
"""

import importlib.util

import pytest

from manabot.env.observation import ObservationConfig, ObservationSpace, step_buffers
from manabot.training.rollout.producers import BufferedStreams
import managym
from managym.recording import RecordedVectorEnv

REMOVED = {
    managym: (
        "encode_observation_into",
        "encode_observations_into",
        "input_v10_state",
        "legacy_card_lists",
    ),
    managym.Env: ("encode_observation", "encode_observation_into"),
    managym.VectorEnv: ("set_buffers", "write_observations_for_players"),
    managym.RolloutPool: ("set_buffers", "encode_active", "step_active"),
    managym.ObservationLayout: ("allocate", "axis_extents", "object_references"),
    RecordedVectorEnv: ("set_buffers", "write_observations_for_players"),
    BufferedStreams: ("write_observations_for_players",),
}


@pytest.mark.parametrize(
    ("owner", "name"),
    [(owner, name) for owner, names in REMOVED.items() for name in names],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_removed_dense_transport_name_is_absent(owner: object, name: str) -> None:
    assert not hasattr(owner, name)


@pytest.mark.parametrize(
    "module",
    [
        "manabot.env.legacy_input",
        "manabot.env.vector_env",
        "tests.fixtures.dense_collector",
    ],
)
def test_removed_dense_module_is_absent(module: str) -> None:
    assert importlib.util.find_spec(module) is None


def test_recorded_vector_reads_fixed_seats_as_one_native_batch() -> None:
    """Fixed-seat reads are methods, so protocol checks find them by static lookup."""
    assert "observations_for_players" in vars(RecordedVectorEnv)
    assert "observations_for_players" in vars(BufferedStreams)
    space = ObservationSpace(ObservationConfig())
    vector = RecordedVectorEnv(num_envs=2, seed=7)
    vector.set_observation_layout(step_buffers(2), layout=space.layout)
    vector.reset_all_into_buffers(
        managym.GameSetup(
            [
                managym.PlayerConfig("a", {"Mountain": 20}),
                managym.PlayerConfig("b", {"Island": 20}),
            ]
        )
    )
    reads, batch = vector.observations_for_players([0, 1])
    assert isinstance(batch, managym.ObservationBatch)
    assert [(read.slot, read.player) for read in reads] == [(0, 0), (1, 1)]
    assert batch.batch_size == 2
