"""Released compute envelopes retain their exact identity without a live server."""

from pathlib import Path

from manabot.sim.execution import EvalCompute, EvalComputeV1
from manabot.sim.registration import PlayerRegistration
from tests.eval.test_evaluation_policy_inputs import fixture_policy


def test_frozen_compute_and_player_identity_stays_readable(tmp_path: Path) -> None:
    _, player, _, _ = fixture_policy(tmp_path)
    v1 = EvalComputeV1(
        version="shared-eval-v1",
        device="cpu",
        dtype="float32",
        max_active_games=64,
        max_batch_rows=64,
        max_delay_seconds=0.005,
        resident_model_bytes=2**30,
        request_seconds=60,
        cache_catalog=True,
        resident=True,
    )
    data = {
        **player.model_dump(),
        "player_spec": {
            **player.player_spec,
            "batch_size": 64,
            "execution": v1.model_dump(),
        },
        "compute_class_id": v1.identity,
    }
    retained = PlayerRegistration.model_validate(data)
    assert retained.model_dump() == data
    assert retained.compute_class_id == v1.identity
    assert v1.identity != EvalCompute().identity
