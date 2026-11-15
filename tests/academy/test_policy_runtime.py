"""Independent stateful adapters exercise the shared Play/Watch authority."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server
from managym.academy.opponent import opponent_catalog, set_policy_registry
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from managym.policy import (
    JSONValue,
    PolicyDecision,
    PolicyDescriptor,
    PolicyRegistry,
    PolicySpec,
    PolicyView,
)
from tests.academy.socket_messages import settled_message


@dataclass
class IndependentSession:
    seat: int
    seen: list[PolicyView] = field(default_factory=list)
    closed: bool = False
    stale: bool = False

    def observe(self, view: PolicyView) -> None:
        assert int(view.observation.agent.player_index) == self.seat
        self.seen.append(view)

    def choose(self, view: PolicyView) -> PolicyDecision:
        # No Env, hidden world or other seat's private hand is supplied.
        assert not hasattr(view, "env")
        return PolicyDecision(view.choices[0], "stale" if self.stale else view.binding)

    def close(self) -> None:
        self.closed = True


@dataclass
class IndependentFactory:
    name: str = "Independent"
    runner: str = "test-source-1"
    sessions: list[IndependentSession] = field(default_factory=list)

    def describe(self) -> PolicyDescriptor:
        return PolicyDescriptor(self.name, self.name, self.runner, recurrent=True)

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        assert len(setup) == 2

    def load(self, *, seat: int, seed: int) -> IndependentSession:
        del seed
        session = IndependentSession(seat)
        self.sessions.append(session)
        return session


@pytest.fixture
def registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PolicyRegistry]:
    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    registry = PolicyRegistry(providers={})
    set_policy_registry(registry)
    yield registry
    for key in list(server.SESSION_REGISTRY):
        server._drop_session(key, "test")
    set_policy_registry(None)


def test_independent_policies_watch_both_seats_and_cleanup(
    registry: PolicyRegistry,
) -> None:
    first, second = IndependentFactory(), IndependentFactory("Other")
    keys = [registry.register(factory) for factory in (first, second)]
    assert len(opponent_catalog()["opponents"]) == 2
    created = create_watch(
        WatchRequest(players=[f"policy:{key}" for key in keys], seed=42)
    )
    for revision in range(8):
        step_watch(created["id"], created["token"], revision)
        row = inspect_watch(created["id"], revision, created["token"])
        assert row["outputs"]["probabilities"] == []
        assert row["outputs"]["value"] is None
        assert row["observation"]["opponent"]["hand"] == []
    assert first.sessions[0] is not second.sessions[0]
    assert all(factory.sessions[0].seen for factory in (first, second))
    game = server.SESSION_REGISTRY[created["id"]].game
    before = game.revision
    first.sessions[0].stale = second.sessions[0].stale = True
    with pytest.raises(HTTPException) as refused:
        step_watch(created["id"], created["token"], before)
    assert refused.value.status_code == 422
    assert refused.value.detail["code"] == "policy_failed"
    assert (
        refused.value.detail["message"] == "Policy returned a stale decision binding."
    )
    assert game.revision == before
    calls = sum(
        len(session.seen) for factory in (first, second) for session in factory.sessions
    )
    with pytest.raises(HTTPException) as implicit:
        step_watch(created["id"], created["token"], before)
    assert implicit.value.detail["code"] == "explicit_retry_required"
    assert (
        sum(
            len(session.seen)
            for factory in (first, second)
            for session in factory.sessions
        )
        == calls
    )
    first.sessions[0].stale = second.sessions[0].stale = False
    assert (
        step_watch(created["id"], created["token"], before, retry=True)["revision"]
        == before + 1
    )
    server._drop_session(created["id"], "test")
    assert all(factory.sessions[0].closed for factory in (first, second))
    # Replay needs neither registry nor a reconstructed policy.
    set_policy_registry(PolicyRegistry(providers={}))
    from managym.academy.replay_index import load_canonical_replay
    from managym.academy.trace import load_trace

    replay = load_canonical_replay(load_trace(created["id"]))
    assert len(replay.decisions) == 9


def test_policy_identity_rejects_runner_change(registry: PolicyRegistry) -> None:
    factory = IndependentFactory()
    key = registry.register(factory)
    config = server._parse_game_config({"villain_type": "random"})
    factory.runner = "changed-source"
    with pytest.raises(ValueError, match="changed"):
        registry.load(key, config.to_rust(), seat=1, seed=2)
    assert not factory.sessions


def test_failed_opponent_policy_preserves_commitment_until_explicit_retry(
    registry: PolicyRegistry,
) -> None:
    @dataclass
    class StaleFactory(IndependentFactory):
        def load(self, *, seat: int, seed: int) -> IndependentSession:
            session = super().load(seat=seat, seed=seed)
            session.stale = True
            return session

    factory = StaleFactory()
    key = registry.register(factory)
    config = {
        "villain_type": "policy",
        "opponent_sha256": key,
        "seed": 3,
        "auto_pass": False,
    }
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json({"type": "new_game", "config": config})
            payload = settled_message(websocket)
            for attempt in range(20):
                if payload.get("mirror_retry"):
                    break
                frame = payload.get("frame") or payload["update"]["frame"]
                command = {
                    "command_id": f"human.{attempt}",
                    "match_id": frame["match_id"],
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": frame["offers"][-1]["id"],
                    "answers": [],
                }
                websocket.send_json({"type": "command", "command": command})
                payload = settled_message(websocket)
            else:
                pytest.fail("Never reached the failing bot")
            assert payload["type"] == "command_outcome"
            assert payload["status"] == "accepted"
            receipt = payload["update"]["receipt"]
            assert receipt["command_id"] == command["command_id"]
            assert receipt["resulting_revision"] == command["expected_revision"] + 1
            pending = payload["mirror_retry"]
            assert pending["message"] == "Policy returned a stale decision binding."
            game = next(
                record.game
                for record in server.SESSION_REGISTRY.values()
                if record.game.match_id == command["match_id"]
            )
            assert game.env is not None and game.trace is not None
            digest = game.env.state_digest()
            events = len(game.trace.events)
            websocket.send_json({"type": "command", "command": command})
            duplicate = settled_message(websocket)
            assert duplicate["status"] == "duplicate"
            assert duplicate["receipt"] == receipt
            assert duplicate["recovery"]["frame"]["prompt"] is None
            assert game.env.state_digest() == digest
            assert len(game.trace.events) == events
            # Resume and settings do not call choose or clear the failure.
            calls = len(factory.sessions[0].seen)
            websocket.send_json({"type": "set_stops", "auto_pass": False})
            recovered = settled_message(websocket)
            assert recovered["mirror_retry"]["code"] == "policy_failed"
            assert game.env.state_digest() == digest
            assert len(factory.sessions[0].seen) == calls
            factory.sessions[0].stale = False
            websocket.send_json(
                {
                    "type": "retry_mirror",
                    "match_id": game.match_id,
                    "expected_revision": game.revision,
                }
            )
            retried = settled_message(websocket)
            assert retried["mirror_retry"] is None
            assert len(game.trace.events) > events
            assert game.accepted_commands[command["command_id"]] == receipt
            assert len(factory.sessions) == 1


def test_class_entry_point_is_refused_with_its_fix() -> None:
    class Provider:
        def policy(self, spec: PolicySpec) -> IndependentFactory:
            return IndependentFactory()

    registry = PolicyRegistry(providers={"mine": Provider})  # type: ignore[dict-item]
    with pytest.raises(
        ValueError, match="names a class; point it at a provider instance"
    ):
        registry.register(PolicySpec("mine"))
    registry.discover()
    assert registry.discovery_errors[0].startswith("mine: its managym.bots entry point")


def test_bad_probabilities_and_bindings_leave_game_pending(
    registry: PolicyRegistry,
) -> None:
    factory = IndependentFactory()
    key = registry.register(factory)
    config = server._parse_game_config({"villain_type": "random"})
    env = managym.Env(seed=42)
    obs, _ = env.reset(config.to_rust())
    runner = registry.load(
        key, config.to_rust(), seat=int(obs.agent.player_index), seed=0
    )
    original = runner.session.choose
    runner.session.choose = lambda view: replace(original(view), probabilities=(1.0,))
    before = env.state_digest()
    with pytest.raises(ValueError, match="complete legal set"):
        runner(env, obs)
    assert env.state_digest() == before
    runner.close()


def test_independent_adapter_without_training_installation(tmp_path: Path) -> None:
    import subprocess
    import sys

    probe = """
import importlib.abc
import sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'manabot', 'torch', 'etude', 'gymnasium'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Refuse())
from pathlib import Path
from tests.academy.test_policy_runtime import IndependentFactory
from managym.policy import PolicyRegistry
from managym.academy.opponent import set_policy_registry
from managym.academy import server
from managym.academy.watch import create_watch, WatchRequest, step_watch
registry = PolicyRegistry(providers={})
key = registry.register(IndependentFactory())
set_policy_registry(registry)
server.trace_store.TRACES_DIR = Path(sys.argv[1])
watch = create_watch(WatchRequest(players=[f'policy:{key}', f'policy:{key}']))
for revision in range(6):
    step_watch(watch['id'], watch['token'], revision)
server._drop_session(watch['id'], 'test')
assert 'manabot' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", probe, str(tmp_path)], check=True, timeout=30)


def test_factory_cannot_share_a_mutable_session_between_games(
    registry: PolicyRegistry,
) -> None:
    class ReusingFactory(IndependentFactory):
        def load(self, *, seat: int, seed: int) -> IndependentSession:
            return (
                self.sessions[0]
                if self.sessions
                else super().load(seat=seat, seed=seed)
            )

    factory = ReusingFactory()
    key = registry.register(factory)
    setup = server._parse_game_config({"villain_type": "random"}).to_rust()
    first = registry.load(key, setup, seat=0, seed=0)
    try:
        with pytest.raises(ValueError, match="reused a live mutable session"):
            registry.load(key, setup, seat=1, seed=1)
        assert not factory.sessions[0].closed
    finally:
        first.close()


def test_watch_load_failure_closes_previously_loaded_seats(
    registry: PolicyRegistry,
) -> None:
    class FailingFactory(IndependentFactory):
        def load(self, *, seat: int, seed: int) -> IndependentSession:
            raise RuntimeError("model allocation failed")

    first = IndependentFactory()
    keys = [registry.register(first), registry.register(FailingFactory("Failed"))]
    with pytest.raises(RuntimeError, match="model allocation failed"):
        create_watch(WatchRequest(players=[f"policy:{key}" for key in keys]))
    assert len(first.sessions) == 1
    assert first.sessions[0].closed


def test_completed_independent_game_replays_without_a_provider(
    registry: PolicyRegistry,
) -> None:
    import random

    class RandomSession(IndependentSession):
        def __init__(self, seat: int, seed: int) -> None:
            super().__init__(seat)
            self.rng = random.Random(seed)

        def choose(self, view: PolicyView) -> PolicyDecision:
            return PolicyDecision(self.rng.choice(view.choices), view.binding)

    class RandomFactory(IndependentFactory):
        def load(self, *, seat: int, seed: int) -> IndependentSession:
            session = RandomSession(seat, seed)
            self.sessions.append(session)
            return session

    factory = RandomFactory()
    key = registry.register(factory)
    created = create_watch(
        WatchRequest(players=[f"policy:{key}", f"policy:{key}"], seed=42)
    )
    for revision in range(2000):
        if step_watch(created["id"], created["token"], revision)["finished"]:
            break
    else:
        pytest.fail("Independent game did not finish in its decision budget.")
    assert all(session.closed for session in factory.sessions)
    server._drop_session(created["id"], "test")
    set_policy_registry(PolicyRegistry(providers={}))
    retained = inspect_watch(created["id"], 0, created["token"])
    assert retained["finished"]
    assert retained["outputs"]["probabilities"] == []
    assert retained["identities"][0]["inference"]["runtime"]["id"] == key


def test_provider_presentation_alias_and_additive_discovery(
    registry: PolicyRegistry,
) -> None:
    from managym.policy import JSONValue, PolicyDetail, PolicySource, PolicySpec

    class Factory(IndependentFactory):
        def describe(self) -> PolicyDescriptor:
            return replace(
                super().describe(),
                group="Language agents",
                details=(PolicyDetail("Context", "8192 tokens"),),
                alias=self.name,
            )

    class Provider:
        ambient_seen: bool = False

        def policy(self, spec: PolicySpec) -> Factory:
            return Factory(str(spec.artifact))

        def discover(self, config: dict[str, JSONValue] | None) -> list[Factory]:
            if config is None:
                if self.ambient_seen:
                    raise ValueError("Ambient discovery repeated")
                self.ambient_seen = True
            return [Factory("Ambient" if config is None else str(config["name"]))]

    registry.providers = {"example": Provider()}
    registry.discover([PolicySource("example", {"name": "Explicit"})])
    assert {d.name for d in registry.descriptors()} == {"Ambient", "Explicit"}
    assert registry.descriptor("Ambient").alias == "Ambient"
    entry = opponent_catalog()["opponents"][0]
    assert entry["group"] == "Language agents"
    assert entry["details"] == [{"label": "Context", "value": "8192 tokens"}]
    assert "updates" not in entry and "model" not in entry
    # A later explicit discovery does not reread the provider's environment.
    registry.discover([PolicySource("example", {"name": "Another"})])
    assert registry.discovery_errors == []
    assert {d.name for d in registry.descriptors()} == {
        "Ambient",
        "Explicit",
        "Another",
    }


def test_presentation_does_not_change_execution_identity() -> None:
    from managym.policy import PolicyDetail

    original = IndependentFactory().describe()
    assert (
        replace(
            original,
            group="Other",
            alias="stable",
            details=(PolicyDetail("Note", "value"),),
        ).id
        == original.id
    )


def test_ambient_alias_collision_keeps_later_policies() -> None:
    class AliasedFactory(IndependentFactory):
        def describe(self) -> PolicyDescriptor:
            return replace(super().describe(), alias="shared")

    class Provider:
        def policy(self, spec: PolicySpec) -> IndependentFactory:
            raise NotImplementedError

        def discover(
            self, config: dict[str, JSONValue] | None
        ) -> list[IndependentFactory]:
            return [AliasedFactory("Collision"), IndependentFactory("Later")]

    registry = PolicyRegistry(providers={"example": Provider()})
    original = AliasedFactory("Original")
    original_id = registry.register(original)
    registry.discover()
    assert registry.descriptor("shared").id == original_id
    assert {d.name for d in registry.descriptors()} == {"Original", "Later"}
    assert registry.discovery_errors == [
        "example: Policy alias 'shared' is already registered."
    ]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(AliasedFactory("Explicit"))
    assert {d.name for d in registry.descriptors()} == {"Original", "Later"}


def test_historical_extension_selection_has_no_named_provider_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from managym.academy import extensions
    from managym.academy.extensions import CheckpointRegistration
    from managym.bots import VerifiedArena

    class Extension:
        name = "Independent replay tools"

        def checkpoint_registration(self, seed: int) -> CheckpointRegistration:
            raise NotImplementedError

        def verify_arena(self, record: dict[str, object]) -> VerifiedArena:
            raise NotImplementedError

    independent = Extension()
    monkeypatch.setattr(
        extensions,
        "entry_points",
        lambda **kwargs: [
            SimpleNamespace(name="manabot", load=lambda: object()),
            SimpleNamespace(name="a-independent", load=lambda: independent),
        ],
    )
    extensions.bot_extension.cache_clear()
    try:
        assert extensions.bot_extension() is independent
    finally:
        extensions.bot_extension.cache_clear()


def test_refused_launch_retains_setup_and_exact_policy(
    registry: PolicyRegistry,
) -> None:
    factory = IndependentFactory()
    key = registry.register(factory, name="chosen-config")
    launch = "test-refused-policy"
    config = {
        "villain_type": "policy",
        "opponent_sha256": key,
        "hero_deck": "gw_allies",
        "villain_deck": "ur_lessons",
        "decklists": "closed",
        "seed": 7,
    }
    server.LAUNCH_CONFIGS[launch] = config
    factory.runner = "changed"
    try:
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/play") as ws:
                ws.send_json({"type": "new_game", "config": {"launch": launch}})
                refused = settled_message(ws)
        assert refused["type"] == "error"
        assert "restart Academy and explicitly reselect" in refused["message"]
        assert refused["launch_setup"] == {
            k: v for k, v in config.items() if k != "seed"
        }
        assert not factory.sessions
    finally:
        del server.LAUNCH_CONFIGS[launch]


def test_registered_aliases_distinguish_catalog_and_watch_identities(
    registry: PolicyRegistry,
) -> None:
    class ConfiguredFactory(IndependentFactory):
        def describe(self) -> PolicyDescriptor:
            return replace(
                super().describe(),
                runner="same-runner",
                artifacts={"actor": "same-weights"},
                config={"mode": self.runner},
            )

    first, second = ConfiguredFactory(), ConfiguredFactory(runner="other-config")
    keys = [
        registry.register(first, name="uniform"),
        registry.register(second, name="action-only"),
    ]
    assert {entry["name"] for entry in opponent_catalog()["opponents"]} == {
        "Independent · uniform",
        "Independent · action-only",
    }
    created = create_watch(
        WatchRequest(players=[f"policy:{key}" for key in keys], seed=42)
    )
    step_watch(created["id"], created["token"], 0)
    recorded = inspect_watch(created["id"], 0, created["token"])
    assert {p["name"] for p in recorded["identities"]} == {
        "Independent · uniform",
        "Independent · action-only",
    }


def test_receipt_and_cached_resume_are_published_before_bot_finishes(
    registry: PolicyRegistry,
) -> None:
    from threading import Event

    entered, release = Event(), Event()

    class WaitingSession(IndependentSession):
        def choose(self, view: PolicyView) -> PolicyDecision:
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release the controlled bot")
            raise ValueError("Controlled bot failure")

    class WaitingFactory(IndependentFactory):
        def load(self, *, seat: int, seed: int) -> IndependentSession:
            session = WaitingSession(seat)
            self.sessions.append(session)
            return session

    key = registry.register(WaitingFactory())
    try:
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/play") as socket:
                socket.send_json(
                    {
                        "type": "new_game",
                        "config": {
                            "villain_type": "policy",
                            "opponent_sha256": key,
                            "seed": 3,
                            "auto_pass": False,
                        },
                    }
                )
                payload = settled_message(socket)
                credentials = {k: payload[k] for k in ("session_id", "resume_token")}
                receipt = None
                for attempt in range(20):
                    frame = payload.get("frame") or payload["update"]["frame"]
                    command = {
                        "command_id": f"controlled.{attempt}",
                        "match_id": frame["match_id"],
                        "expected_revision": frame["revision"],
                        "prompt_id": frame["prompt"]["id"],
                        "offer_id": frame["offers"][-1]["id"],
                        "answers": [],
                    }
                    socket.send_json({"type": "command", "command": command})
                    # Read the early acknowledgement itself, never a settled helper.
                    while True:
                        early = socket.receive_json()
                        if early["type"] == "command_outcome":
                            break
                    receipt = early["update"]["receipt"]
                    assert receipt["command_id"] == command["command_id"]
                    assert early["work"]["state"] == "running"
                    # Either a choose-stage follows, or this Command settles
                    # without reaching the opponent yet.
                    after = socket.receive_json()
                    if (
                        after["type"] == "work_status"
                        and after["work"]["kind"] == "choose"
                    ):
                        assert entered.wait(5)
                        with client.websocket_connect("/ws/play") as resumed:
                            resumed.send_json({"type": "resume", **credentials})
                            cached = resumed.receive_json()
                            assert cached["work"]["kind"] == "choose"
                            assert cached["work"]["state"] == "running"
                            assert receipt in cached["recovery"]["accepted_commands"]
                            assert (
                                cached["recovery"]["frame"]["revision"]
                                == receipt["resulting_revision"]
                            )
                            release.set()
                            final = settled_message(resumed)
                            assert (
                                final["mirror_retry"]["message"]
                                == "Controlled bot failure"
                            )
                            assert final["update"]["receipt"] == receipt
                        break
                    assert after["type"] == "command_outcome"
                    payload = after
                else:
                    pytest.fail("Never reached the controlled bot")
    finally:
        release.set()
