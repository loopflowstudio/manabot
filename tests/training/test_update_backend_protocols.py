"""Every backend provides what the updates call (`manabot.training.updates.backend`).

No static type checker runs in CI, so this walks each Protocol's members on
the backend module and on a real learner, policy and admitted rollout: every
member exists, a property's value conforms to the Protocol it returns, and
every method accepts the Protocol's parameters (names unless positional-only,
kinds and defaults).
A backend whose framework is absent is skipped.
"""

from collections.abc import Callable
import importlib
import inspect
from typing import Any, get_type_hints

import numpy as np
import pytest
import torch

from manabot.backends import resolve
from manabot.backends.parity import BACKENDS, ParityBackend
from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.rollout import AdmittedRollout as TorchRollout
from manabot.policy.checkpoint import PolicyCheckpoint
from manabot.policy.protocol import Backend
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.state.backend import (
    Continuation,
    SelectionBackend,
    SelfPlayBackend,
    VerificationBackend,
)
from manabot.training.updates.backend import (
    AdmittedRollout,
    Arrays,
    CategoricalRL,
    Learner,
    LearnerFile,
    UpdateBackend,
    categorical_rl,
)
from tests.training.backend_fixtures import installed_backend, legacy_capture

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
            returned = get_type_hints(
                declared.fget,
                localns={"RolloutBatch": RolloutBatch},
            ).get("return")
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
    return legacy_capture(seed=3, num_envs=1, decisions=2)


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_backend_provides_what_the_updates_call(
    name: str, captured: tuple[PolicyCheckpoint, RolloutBatch]
) -> None:
    module = installed_backend(name)
    weights, batch = captured
    _, learner = module.load(weights, minibatch_rows=4)
    admitted = module.rollout.AdmittedRollout(learner.agent, batch)
    assert _violations(module, Backend, name) == []
    assert _violations(module, UpdateBackend, name) == []
    assert (
        _violations(
            importlib.import_module(f"manabot.backends.{name}"), UpdateBackend, name
        )
        == []
    )
    assert _violations(learner, Learner, f"{name}.Learner") == []
    assert _violations(admitted, AdmittedRollout, f"{name}.AdmittedRollout") == []


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_prefetched_minibatches_preserve_order_and_policy_input(
    captured: tuple[PolicyCheckpoint, RolloutBatch], grad_enabled: bool
) -> None:
    module = resolve("torch")
    agent = module.load(captured[0])[0]
    admitted = TorchRollout(agent, captured[1])
    order = np.array([1, 0, 1], dtype=np.int64)
    admitted.admit_rows(order, 2)
    device = torch.device("cpu")
    with torch.set_grad_enabled(grad_enabled):
        bootstrap = admitted.next_observation(device)
        expected_bootstrap = on_device(
            agent.prepare(captured[1].next_obs, inference=True), device
        )
        for name, value in array_fields(bootstrap, torch.Tensor).items():
            torch.testing.assert_close(
                value,
                array_fields(expected_bootstrap, torch.Tensor)[name],
                rtol=0,
                atol=0,
            )
        with admitted.minibatches(order, 2, device) as batches:
            for start, (indexes, prepared, rows) in zip((0, 2), batches, strict=True):
                expected_indexes = order[start : start + 2]
                np.testing.assert_array_equal(indexes, expected_indexes)
                expected, expected_rows = admitted.rows(expected_indexes, device)
                learner = on_device(
                    agent.prepare(expected_rows.observation, inference=False), device
                )
                learner_fields = array_fields(learner, torch.Tensor)
                actual_fields = array_fields(prepared, torch.Tensor)
                expected_fields = array_fields(expected, torch.Tensor)
                assert (
                    actual_fields.keys()
                    == expected_fields.keys()
                    == learner_fields.keys()
                )
                for name, value in actual_fields.items():
                    torch.testing.assert_close(
                        value, expected_fields[name], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        value, learner_fields[name], rtol=0, atol=0
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
    agent = resolve("torch").load(captured[0])[0]
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
    module = installed_backend("mlx")
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


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_backends_provide_runtime_and_collection_contracts(name: str) -> None:
    backend = installed_backend(name)
    assert _violations(backend, SelfPlayBackend, name) == []
    assert _violations(backend.state, Continuation, name) == []
    assert _violations(backend, VerificationBackend, name) == []
    assert _violations(backend, ParityBackend, name) == []
    assert _violations(backend, SelectionBackend, name) == []
