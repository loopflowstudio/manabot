"""Framework-free policy admission and per-seat execution.

Factories own model loading and create independent sessions. The registry pins
runner, config and artifact identity before a game; sessions receive only native
viewer observations, never a live environment. Academy and other runners use
PolicyRunner to validate proposals before committing their own bound Commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
from typing import Protocol, TypeAlias
import uuid

import managym
from managym.bots import BotDecision

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
class PolicyDescriptor:
    name: str
    provider: str
    runner: str
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
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
        return {**asdict(self), "id": self.id}


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
    semantic: str = "signed_expected_outcome"


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
        descriptor = policy.describe()
        if descriptor.contract_version != 1:
            raise ValueError("Unsupported policy adapter contract version.")
        key = descriptor.id
        self._factories.setdefault(key, policy)
        self._descriptors.setdefault(key, descriptor)
        if name is not None:
            if name in self.aliases and self.aliases[name] != key:
                raise ValueError(f"Policy alias {name!r} is already registered.")
            self.aliases[name] = key
        return key

    def discover(self, sources: Sequence[PolicySource] = ()) -> None:
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
        return tuple(self._descriptors.values())

    def descriptor(self, key: str) -> PolicyDescriptor:
        key = self.aliases.get(key, key)
        try:
            return self._descriptors[key]
        except KeyError as exc:
            raise ValueError("The selected policy is no longer available.") from exc

    def load(
        self, key: str, setup: Sequence[managym.PlayerConfig], *, seat: int, seed: int
    ) -> PolicyRunner:
        descriptor = self.descriptor(key)
        if descriptor.incompatibility:
            raise ValueError(
                f"{descriptor.name} cannot play here: {descriptor.incompatibility}"
            )
        factory = self._factories[descriptor.id]
        # Repeat description/admission at load: files and runner may have changed.
        if factory.describe().id != descriptor.id:
            raise ValueError(
                "The selected policy changed; restore the pinned artifact and runner."
            )
        factory.admit(setup)
        session = factory.load(seat=seat, seed=seed)
        if factory.describe().id != descriptor.id:
            session.close()
            raise ValueError(
                "The selected policy changed while loading; restore the pinned artifact and runner."
            )
        return PolicyRunner(session, descriptor, seat)


class PolicyRunner:
    """Bridge to existing game runners without exposing their Env to adapters."""

    def __init__(
        self, session: PolicySession, descriptor: PolicyDescriptor, seat: int
    ) -> None:
        self.session = session
        self.descriptor = descriptor
        self.seat = seat
        self.last_decision: BotDecision | None = None
        self._closed = False

    def observe(self, observation: managym.Observation) -> None:
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
        if int(obs.agent.player_index) != self.seat:
            raise ValueError("Policy received another chooser’s observation.")
        from dataclasses import replace

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
            self.session.close()
