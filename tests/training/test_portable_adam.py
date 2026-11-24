"""A restored learner takes the same next Adam update, including on younger leaves."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from manabot.policy.checkpoint import read_safetensors, write_safetensors
from manabot.training.learning import DampedPolicyLearning
from manabot.training.state.optimizer import AdamState, admit_adam
from manabot.training.updates.damped_policy import update_damped_iteration
from tests.training.backend_fixtures import installed_backend, legacy_capture

if TYPE_CHECKING:
    from manabot.backends.mlx.learner import Learner as MLXLearner
    from manabot.backends.torch.learner import Learner as TorchLearner


def same_arrays(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    assert left.keys() == right.keys()
    for name, array in left.items():
        np.testing.assert_array_equal(array, right[name], strict=True, err_msg=name)


def dense_step(learner: "TorchLearner | MLXLearner", backend: str) -> None:
    """Give every parameter the same new gradient, even previously unused ones."""
    if backend == "torch":
        for parameter in learner.agent.parameters():
            parameter.grad = torch.full_like(parameter, -0.125)
        learner.optimizer.step()
    else:
        import mlx.core as mx
        from mlx.utils import tree_map

        gradients = tree_map(
            lambda p: mx.full(p.shape, -0.125), learner.agent.parameters()
        )
        learner.optimizer.update(learner.agent, gradients)


@pytest.mark.parametrize(
    ("source", "target"), [("torch", "torch"), ("torch", "mlx"), ("mlx", "torch")]
)
def test_restore_preserves_next_update(source: str, target: str) -> None:
    writer, reader = (
        installed_backend(name, precision="float32") for name in (source, target)
    )
    with (
        writer.runtime.scoped(1, "float32", "cpu"),
        reader.runtime.scoped(1, "float32", "cpu"),
    ):
        weights, batch = legacy_capture(seed=7, num_envs=2, decisions=6)
        raw, original = writer.load(weights, minibatch_rows=batch.actions.size)
        ema = deepcopy(raw)
        restored, resumed = reader.load(weights, minibatch_rows=batch.actions.size)
        restored_ema = deepcopy(restored)
        export, load = writer.state, reader.state
        empty = export.adam_state(original, step=0)
        load.load_adam(resumed, empty)
        assert load.adam_state(resumed, step=0).param_steps == {}
        learning = DampedPolicyLearning(gradient="damped_policy", min_advantage=0)
        for iteration in (1, 2):
            row = update_damped_iteration(writer, original, batch, learning, iteration)
            assert row["numerical"]["optimizer_steps"] == 1
            writer.learner.update_ema(ema, raw, 0.75)
        step = 2
        if source == "torch":
            # Exercise the first update of one previously unused encoder while the
            # other unused encoders still have no Adam entry.
            original.optimizer.zero_grad(set_to_none=True)
            unused = next(
                p for p in raw.parameters() if p not in original.optimizer.state
            )
            unused.grad = torch.ones_like(unused)
            original.optimizer.step()
            step += 1

        saved = export.adam_state(original, step=step)
        if source == "torch":
            assert {0, 1, 2} <= set(saved.param_steps.values())
        load.load_policy_arrays(restored, export.policy_arrays(raw))
        load.load_policy_arrays(restored_ema, export.policy_arrays(ema))
        aligned = load.load_adam(resumed, saved)
        assert aligned == (
            sum(t < step for t in saved.param_steps.values()) if target == "mlx" else 0
        )
        same_arrays(export.policy_arrays(raw), load.policy_arrays(restored))
        same_arrays(export.policy_arrays(ema), load.policy_arrays(restored_ema))
        same_arrays(export.policy_buffers(raw), load.policy_buffers(restored))
        actual = load.adam_state(resumed, step=step)
        assert actual.param_steps == saved.param_steps
        same_arrays(actual.exp_avg, saved.exp_avg)
        same_arrays(actual.exp_avg_sq, saved.exp_avg_sq)

        original.set_learning_rate(0.001)
        resumed.set_learning_rate(0.001)
        dense_step(original, source)
        dense_step(resumed, target)
        expected_weights = export.policy_arrays(raw)
        for name, value in load.policy_arrays(restored).items():
            if source == target:
                np.testing.assert_array_equal(
                    value, expected_weights[name], err_msg=name
                )
            else:
                np.testing.assert_allclose(
                    value, expected_weights[name], rtol=1e-6, atol=1e-7, err_msg=name
                )
        assert (
            load.adam_state(resumed, step=step + 1).param_steps
            == export.adam_state(original, step=step + 1).param_steps
        )


def test_adam_rejects_mismatched_clock_and_invalid_moments() -> None:
    moments = {"weight": np.ones(2, dtype=np.float32)}
    adam = AdamState(2, {"weight": 1}, moments, dict(moments))
    with pytest.raises(ValueError, match="adam/step"):
        admit_adam(adam, {"weight": (2,)}, step=1)
    adam.exp_avg_sq["weight"] = np.full(2, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="adam/exp_avg_sq/weight"):
        admit_adam(adam, {"weight": (2,)}, step=2)


def test_unsigned_sampling_words_round_trip(tmp_path: Path) -> None:
    words = {
        dtype.__name__: np.array([0, np.iinfo(dtype).max], dtype=dtype)
        for dtype in (np.uint16, np.uint32, np.uint64)
    }
    path = tmp_path / "sampling.safetensors"
    write_safetensors(path, words, {})
    restored, _ = read_safetensors(path)
    same_arrays(words, restored)
