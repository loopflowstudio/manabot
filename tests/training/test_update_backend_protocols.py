"""Every backend provides what the updates call (`manabot.training.updates.backend`).

No static type checker runs in CI, so this walks each Protocol's members on
the backend module and on a real learner, policy and admitted rollout: every
member exists, a property's value conforms to the Protocol it returns, and
every method accepts the Protocol's parameters (names unless positional-only,
kinds and defaults).
A backend whose framework is absent is skipped.
"""

from collections.abc import Callable
import inspect
from types import ModuleType
from typing import Any, get_type_hints

import numpy as np
import pytest
import torch

from manabot.backends import resolve
from manabot.backends.parity import BACKENDS, installed_backends
from manabot.backends.torch.arrays import array_fields
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout as TorchRollout
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.policy.checkpoint import PolicyCheckpoint, from_torch_state
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.updates.backend import (
    AdmittedRollout,
    Arrays,
    Backend,
    CategoricalRL,
    Learner,
    LearnerFile,
    PPOAdmittedRollout,
    PPOBackend,
    PPOLearner,
    categorical_rl,
)
import managym

# -- conformance ---------------------------------------------------------------


def _protocol_members(protocol: type) -> list[str]:
    attrs: frozenset[str] = getattr(protocol, "__protocol_attrs__")
    return sorted(attrs)


def _accepts(implementation: Callable[..., Any], declared: Callable[..., Any]) -> str:
    """Why `implementation` cannot be called as `declared` (without `self`), or ''."""
    want = list(inspect.signature(declared).parameters.values())[1:]
    have = inspect.signature(implementation).parameters
    positional = [
        p
        for p in have.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(p.kind is p.VAR_POSITIONAL for p in have.values())
    matched: set[str] = set()
    for index, parameter in enumerate(want):
        if parameter.kind is parameter.VAR_KEYWORD:
            if not any(p.kind is p.VAR_KEYWORD for p in have.values()):
                return f"takes no **{parameter.name}"
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            if index >= len(positional) and variadic:
                continue
            if index >= len(positional):
                return f"takes no parameter {index} ({parameter.name})"
            match = positional[index]
        elif parameter.kind is parameter.POSITIONAL_OR_KEYWORD:
            if index >= len(positional) or positional[index].name != parameter.name:
                return f"parameter {index} is not {parameter.name!r}"
            match = positional[index]
        else:
            match = have.get(parameter.name)
            if match is None or match.kind not in (
                match.KEYWORD_ONLY,
                match.POSITIONAL_OR_KEYWORD,
            ):
                return f"takes no keyword {parameter.name!r}"
        if parameter.default is not parameter.empty and match.default is match.empty:
            return f"{parameter.name!r} has no default"
        matched.add(match.name)
    for parameter in have.values():
        if (
            parameter.name not in matched
            and parameter.default is parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ):
            return f"requires {parameter.name!r}, which the protocol does not pass"
    return ""


def _violations(value: object, protocol: type, path: str) -> list[str]:
    found: list[str] = []
    for name in _protocol_members(protocol):
        where = f"{path}.{name}"
        if not hasattr(value, name):
            found.append(f"{where}: missing")
            continue
        actual = getattr(value, name)
        declared = inspect.getattr_static(protocol, name, None)
        if name == "NumericalError":
            if not (isinstance(actual, type) and issubclass(actual, Exception)):
                found.append(f"{where}: not an exception type")
        elif isinstance(declared, property):
            returned = get_type_hints(declared.fget).get("return")
            if isinstance(returned, type) and getattr(returned, "_is_protocol", False):
                found += _violations(actual, returned, where)
        elif callable(declared):
            if not callable(actual):
                found.append(f"{where}: not callable")
            elif reason := _accepts(actual, declared):
                found.append(f"{where}: {reason}")
    return found


# -- backends ------------------------------------------------------------------


@pytest.fixture(scope="module")
def captured() -> tuple[PolicyCheckpoint, RolloutBatch]:
    torch.manual_seed(3)
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
    )
    state = {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}
    collector = SeatRoutedCollector(
        space,
        Match(MatchConfig()),
        Reward(RewardConfig()),
        num_envs=1,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        device="cpu",
    )
    batch = collector.collect(agent, 2)
    return from_torch_state(
        agent.config,
        space.encoder.config,
        state,
        parameter_names=dict(agent.named_parameters()),
    ), batch


def _installed(name: str) -> ModuleType:
    if name not in installed_backends():
        pytest.skip(f"the {BACKENDS[name]} framework is not installed")
    return resolve(name)


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_backend_provides_what_the_damped_update_calls(
    name: str, captured: tuple[PolicyCheckpoint, RolloutBatch]
) -> None:
    module = _installed(name)
    weights, batch = captured
    learner = module.learner.Learner(
        module.checkpoint.assemble(weights), "cpu", minibatch_rows=4
    )
    admitted = module.rollout.AdmittedRollout(learner.agent, batch)
    assert _violations(module, Backend, name) == []
    assert _violations(learner, Learner, f"{name}.Learner") == []
    assert _violations(admitted, AdmittedRollout, f"{name}.AdmittedRollout") == []


def test_torch_provides_what_ppo_calls(
    captured: tuple[PolicyCheckpoint, RolloutBatch],
) -> None:
    module = resolve("torch")
    learner = module.learner.Learner(
        module.checkpoint.assemble(captured[0]), "cpu", minibatch_rows=4
    )
    assert _violations(module, PPOBackend, "torch") == []
    assert _violations(learner, PPOLearner, "torch.Learner") == []
    assert (
        _violations(
            module.rollout.AdmittedRollout(learner.agent, captured[1]),
            PPOAdmittedRollout,
            "torch.AdmittedRollout",
        )
        == []
    )
    assert categorical_rl(module.rl) is module.rl


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_prefetched_minibatches_preserve_order_and_policy_input(
    captured: tuple[PolicyCheckpoint, RolloutBatch], grad_enabled: bool
) -> None:
    module = resolve("torch")
    agent = module.checkpoint.assemble(captured[0])
    admitted = TorchRollout(agent, captured[1])
    order = np.array([1, 0, 1], dtype=np.int64)
    admitted.admit_rows(order, 2)
    device = torch.device("cpu")
    with torch.set_grad_enabled(grad_enabled):
        with admitted.minibatches(order, 2, device) as batches:
            for start, (indexes, prepared, rows) in zip((0, 2), batches, strict=True):
                expected_indexes = order[start : start + 2]
                np.testing.assert_array_equal(indexes, expected_indexes)
                expected, expected_rows = admitted.rows(expected_indexes, device)
                actual_fields = array_fields(prepared, torch.Tensor)
                expected_fields = array_fields(expected, torch.Tensor)
                assert actual_fields.keys() == expected_fields.keys()
                for name, value in actual_fields.items():
                    torch.testing.assert_close(
                        value, expected_fields[name], rtol=0, atol=0
                    )
                np.testing.assert_array_equal(
                    rows.log_probabilities, expected_rows.log_probabilities
                )
                for name in rows.observation:
                    np.testing.assert_array_equal(
                        rows.observation[name], expected_rows.observation[name]
                    )
                for actual, wanted in zip(
                    agent(prepared), agent(expected), strict=True
                ):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_prefetch_propagates_admission_failure_and_can_stop_early(
    captured: tuple[PolicyCheckpoint, RolloutBatch],
) -> None:
    agent = resolve("torch").checkpoint.assemble(captured[0])
    admitted = TorchRollout(agent, captured[1])
    admitted.admit_rows(np.array([0], dtype=np.int64), 1)
    device = torch.device("cpu")
    order = np.array([0, 1], dtype=np.int64)
    with admitted.minibatches(order, 1, device) as batches:
        indexes, _, _ = next(batches)
        np.testing.assert_array_equal(indexes, [0])
        with pytest.raises(ValueError, match="not admitted"):
            next(batches)
    admitted.admit_rows(order, 1)
    # Leaving before consuming the queued batch also joins the worker.
    with admitted.minibatches(order, 1, device) as batches:
        next(batches)
    with admitted.minibatches(order[:0], 1, device) as batches:
        assert list(batches) == []
    with pytest.raises(ValueError, match="must be positive"):
        with admitted.minibatches(order, 0, device):
            pass


def test_a_backend_without_a_categorical_critic_is_refused() -> None:
    module = _installed("mlx")
    assert not isinstance(module.rl, CategoricalRL)
    with pytest.raises(TypeError, match="no categorical critic"):
        categorical_rl(module.rl)


def test_the_conformance_check_sees_a_wrong_signature() -> None:
    class _Wrong:
        @staticmethod
        def from_numpy(values: object) -> object:
            return values

        @staticmethod
        def to_numpy(tensor: object, scale: float) -> object:
            return tensor

        @staticmethod
        def no_grad(enabled: bool) -> object:
            return enabled

    assert _violations(_Wrong(), Arrays, "wrong") == [
        "wrong.float32: missing",
        "wrong.from_numpy: takes no parameter 1 (device)",
        "wrong.to_numpy: requires 'scale', which the protocol does not pass",
    ]
    assert _violations(_Wrong(), LearnerFile, "wrong") == [
        "wrong.check_objective: missing",
        "wrong.no_grad: requires 'enabled', which the protocol does not pass",
    ]
