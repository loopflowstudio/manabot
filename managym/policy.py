"""Framework-free policy admission and per-seat execution.

Factories own model loading and create independent sessions. The registry pins
runner, config and artifact identity before a game; sessions receive only native
viewer observations, never a live environment. Academy and other runners use
PolicyRunner to validate proposals before committing their own bound Commands.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib
from importlib.metadata import entry_points
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Protocol, TypeAlias
import uuid
import warnings

import managym
from managym.bots import BotDecision
from managym.search import (
    SearchConfig,
    SearchError,
    search_decision,
    search_descriptor,
    search_source_identity,
    search_unavailable,
)

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


def identity_digest(value: JSONValue) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PolicySpec:
    provider: str
    artifact: str | None = None
    config: dict[str, JSONValue] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> PolicySpec:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("provider"), str):
            raise ValueError("Policy specification requires a provider name.")
        if set(value) - {"provider", "artifact", "config"}:
            raise ValueError("Unknown policy specification fields.")
        if value.get("artifact") is not None and not isinstance(value["artifact"], str):
            raise ValueError("Policy artifact must be a string.")
        if not isinstance(value.get("config", {}), dict):
            raise ValueError("Policy config must be an object.")
        identity_digest(value)
        return cls(**value)


@dataclass(frozen=True)
class PolicyDetail:
    label: str
    value: str


@dataclass(frozen=True)
class PolicyDescriptor:
    name: str
    provider: str
    runner: str
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, JSONValue] = field(default_factory=dict)
    group: str | None = None
    details: tuple[PolicyDetail, ...] = ()
    alias: str | None = None
    incompatibility: str | None = None
    supports_search: bool = False
    contract_version: int = 1
    recurrent: bool = False
    search_reason: str | None = None
    base_policy: PolicyDescriptor | None = None
    recommended: bool = False
    summary: str | None = None
    trained_settings: dict[str, JSONValue] | None = None

    @property
    def id(self) -> str:
        # Display/provenance fields may improve without changing execution.
        return identity_digest(
            {
                "provider": self.provider,
                "runner": self.runner,
                "artifacts": self.artifacts,
                "config": self.config,
                "contract_version": self.contract_version,
            }
        )

    def record(self) -> dict[str, JSONValue]:
        payload = asdict(self)
        if not self.recommended:
            payload.pop("recommended", None)
        if self.summary is None:
            payload.pop("summary", None)
        if self.trained_settings is None:
            payload.pop("trained_settings", None)
        return {
            **payload,
            "details": [asdict(detail) for detail in self.details],
            "base_policy": self.base_policy.record() if self.base_policy else None,
            "id": self.id,
        }


@dataclass(frozen=True)
class PolicyView:
    observation: managym.Observation
    seat: int
    binding: str
    # Ordered references cover every native choice. A proposal must echo binding.
    choices: tuple[str, ...]
    capture_json: str | None = None


@dataclass(frozen=True)
class ValueEstimate:
    value: float
    seat: int
    semantic: str = "signed_outcome"


@dataclass(frozen=True)
class PolicyDecision:
    choice: str
    binding: str
    probabilities: tuple[float, ...] | None = None
    value: ValueEstimate | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    probabilities: tuple[float, ...]
    value: ValueEstimate


class PolicySession(Protocol):
    """Per-seat policy instance.

    ``observe(view)`` is called once after every committed state transition
    (initial reset, each committed command by either seat, and terminal
    ``game_over``), with ``view.choices`` populated only when this seat is
    acting and ``view.observation.recent_events`` carrying the last step's
    public events. ``choose(view)`` proposes an action for the already-observed
    pending decision and may be retried without another ``observe()`` call;
    recurrent policies should update state in ``observe()`` only.
    """

    def observe(self, view: PolicyView) -> None: ...
    def choose(self, view: PolicyView) -> PolicyDecision: ...
    def close(self) -> None: ...


class SearchPolicySession(PolicySession, Protocol):
    def evaluate(self, view: PolicyView) -> PolicyEvaluation: ...
    def fork(self, *, seed: int) -> SearchPolicySession: ...


class Policy(Protocol):
    def describe(self) -> PolicyDescriptor: ...
    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None: ...
    def load(self, *, seat: int, seed: int) -> PolicySession: ...


PolicyCallable: TypeAlias = Callable[
    [managym.Observation], int | Sequence[float] | PolicyDecision
]


def _is_policy_object(value: object) -> bool:
    return (
        not isinstance(value, type)
        and callable(getattr(value, "describe", None))
        and callable(getattr(value, "admit", None))
        and callable(getattr(value, "load", None))
    )


def _callable_runner_identity(
    fn: Callable[..., object], *, source_path: Path | None = None
) -> str:
    digest = hashlib.sha256()
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "callable")
    digest.update(f"{module}:{qualname}".encode())
    if source_path is not None and source_path.is_file():
        digest.update(source_path.read_bytes())
    else:
        try:
            digest.update(inspect.getsource(fn).encode())
        except (OSError, TypeError):
            code = getattr(fn, "__code__", None)
            if code is not None:
                digest.update(code.co_code)
                digest.update(repr(code.co_consts).encode())
            else:
                digest.update(f"id:{id(fn)}".encode())
    return f"python:{module}.{qualname}:{digest.hexdigest()[:16]}"


class _CallableSession:
    def __init__(self, fn: PolicyCallable, *, seed: int) -> None:
        self._fn = fn
        self._rng = random.Random(seed)

    def observe(self, view: PolicyView) -> None:
        del view

    def choose(self, view: PolicyView) -> PolicyDecision:
        result = self._fn(view.observation)
        if isinstance(result, PolicyDecision):
            return result
        n = len(view.choices)
        if isinstance(result, bool):
            raise ValueError("Policy callable must return an action index, not bool.")
        if isinstance(result, int) or hasattr(result, "__index__"):
            index = int(result.__index__())  # type: ignore[union-attr]
            if not 0 <= index < n:
                raise ValueError(
                    f"Policy callable returned action index {index} outside 0..{n - 1}."
                )
            return PolicyDecision(view.choices[index], view.binding)
        if isinstance(result, Sequence) or hasattr(result, "tolist"):
            raw = result.tolist() if hasattr(result, "tolist") else list(result)
            if not isinstance(raw, list) or len(raw) != n:
                raise ValueError(
                    f"Policy callable returned {len(raw) if isinstance(raw, list) else type(raw).__name__} values for {n} legal actions."
                )
            probs = tuple(float(x) for x in raw)
            if any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(
                sum(probs), 1.0, abs_tol=1e-4
            ):
                raise ValueError(
                    "Policy callable probabilities must be finite, nonnegative, and sum to 1."
                )
            total = sum(probs)
            norm = tuple(p / total for p in probs)
            pick = self._rng.choices(range(n), weights=norm, k=1)[0]
            return PolicyDecision(view.choices[pick], view.binding, probabilities=norm)
        raise ValueError(
            "Policy callable must return an int action index, probability sequence, or PolicyDecision."
        )

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class CallablePolicy:
    """Wrap a plain ``(obs) -> int | probs | PolicyDecision`` function as a ``Policy``."""

    fn: PolicyCallable
    name: str | None = None
    group: str = "Custom"
    alias: str | None = None
    source_path: Path | None = None

    def describe(self) -> PolicyDescriptor:
        raw_name = self.name or getattr(self.fn, "__name__", None) or "custom"
        label = "custom" if raw_name == "<lambda>" else raw_name
        runner = _callable_runner_identity(self.fn, source_path=self.source_path)
        details = (
            (PolicyDetail("Source", str(self.source_path)),)
            if self.source_path is not None
            else ()
        )
        return PolicyDescriptor(
            name=label,
            provider="python",
            runner=runner,
            group=self.group,
            alias=self.alias,
            details=details,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        del setup

    def load(self, *, seat: int, seed: int) -> PolicySession:
        del seat
        return _CallableSession(self.fn, seed=seed)


def as_policy(
    value: Policy | PolicyCallable,
    *,
    name: str | None = None,
    alias: str | None = None,
    source_path: Path | None = None,
) -> Policy:
    """Return ``value`` as a ``Policy``, wrapping a plain callable automatically."""
    if _is_policy_object(value):
        return value  # type: ignore[return-value]
    if callable(value):
        return CallablePolicy(
            value,  # type: ignore[arg-type]
            name=name,
            alias=alias,
            source_path=source_path,
        )
    raise ValueError(
        "Expected a Policy object or a callable taking an Observation and returning an action index."
    )


def is_python_policy_ref(value: str) -> bool:
    """True for ``path/to/bot.py[:attr]`` or ``pkg.mod:attr`` policy references."""
    if value.startswith("policy:"):
        return False
    target = value.split(":", 1)[0]
    if target.endswith(".py"):
        return True
    if ":" in value:
        # Exclude Windows-style drive letters or scheme prefixes; require an attr name.
        attr = value.split(":", 1)[1]
        if not attr or "/" in attr:
            return False
        return (
            "/" in target
            or os.sep in target
            or "." in target
            or Path(target).expanduser().is_file()
        )
    return False


def load_python_policy(ref: str, *, name: str | None = None) -> Policy:
    """Load a ``Policy`` or callable from ``./bot.py[:attr]`` or ``pkg.mod:attr``."""
    target, sep, attr = ref.partition(":")
    source_path: Path | None = None
    path = Path(target).expanduser()
    if target.endswith(".py") or "/" in target or os.sep in target or path.is_file():
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Policy file does not exist: {target}")
        source_path = resolved
        digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
        mod_name = f"_managym_user_policy_{resolved.stem}_{digest}"
        spec = importlib.util.spec_from_file_location(mod_name, resolved)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import Python policy file: {target}")
        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(mod_name)
        sys.modules[mod_name] = module
        parent_dir = str(resolved.parent)
        added = parent_dir not in sys.path
        if added:
            sys.path.insert(0, parent_dir)
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if previous_module is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = previous_module
            raise
        finally:
            if added and parent_dir in sys.path:
                sys.path.remove(parent_dir)
        default_name = resolved.stem
    else:
        if not sep or not attr:
            raise ValueError(
                f"Expected 'path/to/bot.py[:func]' or 'package.module:func', got {ref!r}."
            )
        module = importlib.import_module(target)
        default_name = attr

    if sep and attr:
        if not hasattr(module, attr):
            raise ValueError(f"{target!r} has no attribute {attr!r}.")
        obj = getattr(module, attr)
        label = name or (f"{default_name}:{attr}" if source_path else attr)
    else:
        for candidate in ("policy", "choose"):
            if hasattr(module, candidate):
                obj = getattr(module, candidate)
                label = name or default_name
                break
        else:
            raise ValueError(
                f"{target!r} defines neither 'policy' nor 'choose'; specify '{target}:name'."
            )

    if isinstance(obj, type) and all(
        callable(getattr(obj, m, None)) for m in ("describe", "admit", "load")
    ):
        obj = obj()
    return as_policy(obj, name=label, source_path=source_path)


class PolicyProvider(Protocol):
    def policy(self, spec: PolicySpec) -> Policy: ...


class DiscoveringPolicyProvider(PolicyProvider, Protocol):
    """Optional discovery: None reads ambient config; explicit configs are additive."""

    def discover(self, config: dict[str, JSONValue] | None) -> Sequence[Policy]: ...


@dataclass(frozen=True)
class PolicySource:
    provider: str
    config: dict[str, JSONValue] = field(default_factory=dict)


# An entry point names a provider object; a class would be called unbound.
_CLASS_ENTRY_POINT = (
    "its managym.bots entry point names a class; point it at a provider "
    "instance instead (for example `mybot:provider` where `provider = Provider()`)."
)


class PolicyRegistry:
    """Server-scoped factories; catalog refresh never substitutes pinned bytes."""

    def __init__(
        self, *, providers: Mapping[str, PolicyProvider] | None = None
    ) -> None:
        self.providers = (
            dict(providers)
            if providers is not None
            else {ep.name: ep.load() for ep in entry_points(group="managym.bots")}
        )
        self._factories: dict[str, Policy] = {}
        self._descriptors: dict[str, PolicyDescriptor] = {}
        self.aliases: dict[str, str] = {}
        self._active_sessions: set[int] = set()
        self.discovery_errors: list[str] = []
        self._discovered = False

    def register(
        self,
        policy: Policy | PolicyCallable | PolicySpec,
        *,
        name: str | None = None,
    ) -> str:
        from_spec = isinstance(policy, PolicySpec)
        if isinstance(policy, PolicySpec):
            resolved = self._provider(policy.provider).policy(policy)
        else:
            resolved = as_policy(policy, name=name)
        return self._register_factory(resolved, name=name, from_provider=from_spec)

    def _register_factory(
        self,
        policy: Policy,
        *,
        name: str | None = None,
        from_provider: bool = False,
    ) -> str:
        descriptor = deepcopy(policy.describe())
        if descriptor.contract_version != 1:
            raise ValueError("Unsupported policy adapter contract version.")
        key = descriptor.id
        existing = self._factories.get(key)
        if existing is not None:
            same_factory = existing is policy or existing == policy
            same_provider_artifact = (
                from_provider
                and type(existing) is type(policy)
                and bool(descriptor.artifacts)
                and self._descriptors[key].artifacts == descriptor.artifacts
            )
            if not (same_factory or same_provider_artifact):
                existing_name = self._descriptors[key].name
                raise ValueError(
                    f"Policy {descriptor.name!r} shares Policy ID {key[:10]} with "
                    f"already-registered {existing_name!r}; PolicyDescriptor.id hashes "
                    "(provider, runner, artifacts, config, contract_version) rather than "
                    "name or alias. Distinguish distinct factories via artifacts, config, "
                    "or runner."
                )
        name = name if name is not None else descriptor.alias
        if name is not None:
            if name in self.aliases and self.aliases[name] != key:
                raise ValueError(f"Policy alias {name!r} is already registered.")
        self._factories.setdefault(key, policy)
        self._descriptors.setdefault(key, descriptor)
        if name is not None:
            self.aliases[name] = key
        return key

    def discover(self, sources: Sequence[PolicySource] = ()) -> None:
        # Ambient discovery happens once per registry/server, even with sources.
        if not self._discovered:
            self._discovered = True
            for name, provider in self.providers.items():
                if isinstance(provider, type):
                    self.discovery_errors.append(f"{name}: {_CLASS_ENTRY_POINT}")
                    continue
                discover = getattr(provider, "discover", None)
                if discover is not None:
                    try:
                        for policy in discover(None):
                            try:
                                self._register_factory(policy, from_provider=True)
                            except (ImportError, OSError, ValueError) as exc:
                                # One unusable entry must not hide later policies.
                                self.discovery_errors.append(f"{name}: {exc}")
                    except (ImportError, OSError, ValueError) as exc:
                        self.discovery_errors.append(f"{name}: {exc}")
        for source in sources:
            provider = self._provider(source.provider)
            discover = getattr(provider, "discover", None)
            if discover is None:
                raise ValueError(
                    f"Policy provider {source.provider!r} has no discovery method."
                )
            for policy in discover(source.config):
                self._register_factory(policy, from_provider=True)

    def _provider(self, name: str) -> PolicyProvider:
        provider = self.providers.get(name)
        if provider is None:
            raise ValueError(f"Policy provider {name!r} is not installed.")
        if isinstance(provider, type):
            raise ValueError(f"Policy provider {name!r}: {_CLASS_ENTRY_POINT}")
        return provider

    def descriptors(self) -> tuple[PolicyDescriptor, ...]:
        return tuple(deepcopy(descriptor) for descriptor in self._descriptors.values())

    def descriptor(self, key: str) -> PolicyDescriptor:
        key = self.aliases.get(key, key)
        try:
            return deepcopy(self._descriptors[key])
        except KeyError as exc:
            raise ValueError("The selected policy is no longer available.") from exc

    def admit(
        self, key: str, setup: Sequence[managym.PlayerConfig]
    ) -> PolicyDescriptor:
        descriptor = self.descriptor(key)
        factory = self._factories[descriptor.id]
        # Repeat description/admission at load: files and runner may have changed.
        if factory.describe().id != descriptor.id:
            raise ValueError(
                "The selected policy changed; restore the pinned artifact and runner, or restart Academy and explicitly reselect the policy."
            )
        factory.admit(setup)
        if descriptor.incompatibility:
            raise ValueError(
                f"{descriptor.name} cannot play here: {descriptor.incompatibility}"
            )
        return descriptor

    def load(
        self,
        key: str,
        setup: Sequence[managym.PlayerConfig],
        *,
        seat: int,
        seed: int,
        search: SearchConfig | None = None,
    ) -> PolicyRunner:
        descriptor = self.admit(key, setup)
        factory = self._factories[descriptor.id]
        session = factory.load(seat=seat, seed=seed)
        if id(session) in self._active_sessions:
            raise ValueError(
                "Policy factory reused a live mutable session; each game and seat needs its own session."
            )
        try:
            if factory.describe().id != descriptor.id:
                raise ValueError(
                    "The selected policy changed while loading; restore the pinned artifact and runner, or restart Academy and explicitly reselect the policy."
                )
        except BaseException:
            session.close()
            raise
        self._active_sessions.add(id(session))
        runner = PolicyRunner(
            session,
            descriptor,
            seat,
            release=lambda: self._active_sessions.discard(id(session)),
        )
        try:
            runner.configure_search(search, seed=seed)
        except BaseException:
            runner.close()
            raise
        return runner


class PolicyRunner:
    """Bridge to existing game runners without exposing their Env to adapters."""

    def __init__(
        self,
        session: PolicySession,
        descriptor: PolicyDescriptor,
        seat: int,
        *,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.descriptor = descriptor
        self.base_descriptor = descriptor
        self.seat = seat
        self.last_decision: BotDecision | None = None
        self._closed = False
        self._release = release
        self.search: SearchConfig | None = None
        self.search_runtime: str | None = None
        self._search_seed = 0
        self._search_calls = 0
        self._started = False

    def configure_search(self, search: SearchConfig | None, *, seed: int) -> None:
        """Configure before play; preserve the base runner as a distinct identity."""
        if self._started:
            raise ValueError("Search cannot change after a decision.")
        if search is not None:
            reason = search_unavailable(self)
            if reason:
                raise SearchError("mirror_unavailable", reason)
            self.search_runtime = search_source_identity()
            self.descriptor = search_descriptor(
                self.base_descriptor, search, source=self.search_runtime
            )
        else:
            self.descriptor = self.base_descriptor
            self.search_runtime = None
        self.search = search
        self._search_seed = seed

    def observe(self, observation: managym.Observation) -> None:
        self._started = True
        self.session.observe(self.view(observation))

    def view(self, observation: managym.Observation) -> PolicyView:
        return PolicyView(
            observation,
            self.seat,
            uuid.uuid4().hex,
            tuple(str(i) for i in range(len(observation.action_space.actions))),
        )

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        if self._closed:
            raise ValueError("Policy session is closed.")
        self.last_decision = None
        self._started = True
        if int(obs.agent.player_index) != self.seat:
            raise ValueError("Policy received another chooser’s observation.")
        if self.search is not None:
            # Retry the same uncommitted decision with the same random streams.
            # Increment only once a complete searched proposal exists.
            seed = (self._search_seed * 1_000_003 + self._search_calls + 1) % 2**63
            self.last_decision = search_decision(self, env, obs, self.search, seed=seed)
            self._search_calls += 1
            return self.last_decision.action

        view = replace(
            self.view(obs), capture_json=env.capture_decision(self.seat, 0).to_json()
        )
        decision = self.session.choose(view)
        if decision.binding != view.binding:
            raise ValueError("Policy returned a stale decision binding.")
        if decision.choice not in view.choices:
            raise ValueError("Policy selected an action outside the authority offers.")
        probabilities = decision.probabilities
        if probabilities is not None and (
            len(probabilities) != len(view.choices)
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
        ):
            raise ValueError(
                "Policy probabilities must align with the complete legal set and sum to one."
            )
        value = decision.value
        if value is not None and (
            value.seat != self.seat or not math.isfinite(value.value)
        ):
            raise ValueError("Policy value must name its viewer and be finite.")
        action = view.choices.index(decision.choice)
        self.last_decision = BotDecision(
            action,
            list(probabilities or ()),
            value.value if value else None,
            value.semantic if value else None,
            unavailable=None if value else "This policy returned no value estimate.",
        )
        return action

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.session.close()
            finally:
                if self._release is not None:
                    self._release()


def __getattr__(name: str) -> object:
    if name == "PolicyFactory":
        warnings.warn(
            "managym.policy.PolicyFactory is deprecated; use managym.policy.Policy; removal in 0.3.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return Policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
