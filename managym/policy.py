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
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
from typing import Protocol, TypeAlias
import uuid

import managym
from managym.bots import BotDecision
from managym.mirror_search import (
    SearchError,
    SearchSpec,
    mirror_decision,
    search_source_identity,
    search_unavailable,
    searched_descriptor,
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
        return {
            **asdict(self),
            "details": [asdict(detail) for detail in self.details],
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
    def observe(self, view: PolicyView) -> None: ...
    def choose(self, view: PolicyView) -> PolicyDecision: ...
    def close(self) -> None: ...


class SearchPolicySession(PolicySession, Protocol):
    def evaluate(self, view: PolicyView) -> PolicyEvaluation: ...
    def fork(self, *, seed: int) -> SearchPolicySession: ...


class PolicyFactory(Protocol):
    def describe(self) -> PolicyDescriptor: ...
    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None: ...
    def load(self, *, seat: int, seed: int) -> PolicySession: ...


class PolicyProvider(Protocol):
    def policy(self, spec: PolicySpec) -> PolicyFactory: ...


class DiscoveringPolicyProvider(PolicyProvider, Protocol):
    """Optional discovery: None reads ambient config; explicit configs are additive."""

    def discover(
        self, config: dict[str, JSONValue] | None
    ) -> Sequence[PolicyFactory]: ...


@dataclass(frozen=True)
class PolicySource:
    provider: str
    config: dict[str, JSONValue] = field(default_factory=dict)


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
        self._factories: dict[str, PolicyFactory] = {}
        self._descriptors: dict[str, PolicyDescriptor] = {}
        self.aliases: dict[str, str] = {}
        self._active_sessions: set[int] = set()
        self.discovery_errors: list[str] = []
        self._discovered = False

    def register(
        self, policy: PolicyFactory | PolicySpec, *, name: str | None = None
    ) -> str:
        if isinstance(policy, PolicySpec):
            provider = self.providers.get(policy.provider)
            if provider is None:
                raise ValueError(
                    f"Policy provider {policy.provider!r} is not installed."
                )
            policy = provider.policy(policy)
        descriptor = deepcopy(policy.describe())
        if descriptor.contract_version != 1:
            raise ValueError("Unsupported policy adapter contract version.")
        key = descriptor.id
        self._factories.setdefault(key, policy)
        self._descriptors.setdefault(key, descriptor)
        name = name if name is not None else descriptor.alias
        if name is not None:
            if name in self.aliases and self.aliases[name] != key:
                raise ValueError(f"Policy alias {name!r} is already registered.")
            self.aliases[name] = key
        return key

    def discover(self, sources: Sequence[PolicySource] = ()) -> None:
        # Ambient discovery happens once per registry/server, even with sources.
        if not self._discovered:
            self._discovered = True
            for name, provider in self.providers.items():
                discover = getattr(provider, "discover", None)
                if discover is not None:
                    try:
                        for policy in discover(None):
                            self.register(policy)
                    except (ImportError, OSError, ValueError) as exc:
                        self.discovery_errors.append(f"{name}: {exc}")
        for source in sources:
            provider = self.providers.get(source.provider)
            if provider is None:
                raise ValueError(
                    f"Policy provider {source.provider!r} is not installed."
                )
            discover = getattr(provider, "discover", None)
            if discover is None:
                raise ValueError(
                    f"Policy provider {source.provider!r} has no discovery method."
                )
            for policy in discover(source.config):
                self.register(policy)

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
        search: SearchSpec | None = None,
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
        self.search: SearchSpec | None = None
        self.search_runtime: str | None = None
        self._search_seed = 0
        self._search_calls = 0
        self._started = False

    def configure_search(self, search: SearchSpec | None, *, seed: int) -> None:
        """Configure before play; preserve the base runner as a distinct identity."""
        if self._started:
            raise ValueError("Search cannot change after a decision.")
        if search is not None:
            reason = search_unavailable(self)
            if reason:
                raise SearchError("search_unavailable", reason)
            self.search_runtime = search_source_identity()
            self.descriptor = searched_descriptor(
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
            self.last_decision = mirror_decision(self, env, obs, self.search, seed=seed)
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
