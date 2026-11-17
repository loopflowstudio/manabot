"""Blanked-memory evaluation retains checkpoint and live-board authority."""

from pathlib import Path
import time
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import BACKEND, save_checkpoint
from manabot.env.decision_memory import blank_memory
from manabot.eval.admission import register_checkpoint
from manabot.eval.execution import bind_execution
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import SELECTED_SUITE, selected_match
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving import PolicyStore, infer_requests
from manabot.sim.serving_protocol import DecisionRequest, PolicyAdmission, PolicyKey
from managym import WORLD_SETUP
from tests.eval.arena_fixture import arena_fixture
from tests.eval.test_batched_eval import fixture_policy
from tests.model.test_decision_memory import forward, memory_input, policy


@pytest.mark.parametrize("encoder", ["transformer", "gru"])
def test_blanked_registration_serves_empty_history_without_mutating_input(
    tmp_path: Path, encoder: Literal["transformer", "gru"]
) -> None:
    space, rows = memory_input()
    model = policy(space, encoder, 1).eval()
    path = tmp_path / "memory.safetensors"
    save_checkpoint(model, space, path, player_configs=None)
    setup = selected_match().model_copy(update={"world": WORLD_SETUP})
    intact = register_checkpoint(
        path,
        backend=BACKEND,
        match=setup,
        content_suite=SELECTED_SUITE,
        player_id="intact-memory",
        display_name="Intact memory",
    ).registration
    blanked = PlayerRegistration.model_validate(
        {
            **intact.model_dump(),
            "player_id": "blanked-memory",
            "player_spec": {**intact.player_spec, "blank_decision_memory": True},
        }
    )
    assert blanked.checkpoint_sha256 == intact.checkpoint_sha256
    assert blanked.identity_sha256 != intact.identity_sha256
    execution = EvalExecution(deadline_unix=time.time() + 120)
    _, (intact, blanked) = bind_execution(
        arena_fixture().key, (intact, blanked), execution
    )
    assert PolicyKey.from_registration(
        intact, execution
    ) != PolicyKey.from_registration(blanked, execution)
    original = {key: value.copy() for key, value in rows.items()}
    store = PolicyStore(execution, EvalPerformance("reliance"), backend=BACKEND)
    observed: list[np.ndarray] = []

    def witness(logits: np.ndarray, mask: np.ndarray, actions: list[int]) -> None:
        observed.append(logits.copy())

    try:
        for player in (intact, blanked):
            store.admit(PolicyAdmission(player, str(path), (setup,)))
            request = DecisionRequest(
                player.player_id,
                PolicyKey.from_registration(player, execution),
                11,
                {key: value[0] for key, value in rows.items()},
                execution.deadline_unix,
                time.monotonic(),
            )
            infer_requests(store, [request], player.player_id, witness=witness)
        with torch.inference_mode():
            expected_intact, _ = forward(model, rows)
            expected_blank, _ = forward(model, blank_memory(rows))
        np.testing.assert_allclose(observed[0], expected_intact.numpy(), atol=1e-5)
        np.testing.assert_allclose(observed[1], expected_blank.numpy(), atol=1e-5)
        assert not np.allclose(observed[0], observed[1])
        for key, value in rows.items():
            np.testing.assert_array_equal(value, original[key])
    finally:
        store.close()


def test_blanking_registration_refuses_ambiguous_values(tmp_path: Path) -> None:
    _, player, _, _ = fixture_policy(tmp_path)
    with pytest.raises(ValueError, match="explicitly enabled"):
        PlayerRegistration.model_validate(
            {
                **player.model_dump(),
                "player_spec": {**player.player_spec, "blank_decision_memory": False},
            }
        )
