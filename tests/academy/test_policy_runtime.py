"""Independent stateful adapters exercise the shared Play/Watch authority."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import managym
from managym.academy import server
from managym.academy.opponent import opponent_catalog, set_policy_registry
from managym.academy.watch import WatchRequest, create_watch, inspect_watch, step_watch
from managym.policy import PolicyDecision, PolicyDescriptor, PolicyRegistry, PolicyView


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
    with pytest.raises(ValueError, match="stale"):
        step_watch(created["id"], created["token"], before)
    assert game.revision == before
    server._drop_session(created["id"], "test")
    assert all(factory.sessions[0].closed for factory in (first, second))
    # Replay needs neither registry nor a reconstructed policy.
    set_policy_registry(PolicyRegistry(providers={}))
    from managym.academy.replay_index import load_canonical_replay
    from managym.academy.trace import load_trace

    replay = load_canonical_replay(load_trace(created["id"]))
    assert len(replay.decisions) == 8


def test_policy_identity_rejects_runner_change(registry: PolicyRegistry) -> None:
    factory = IndependentFactory()
    key = registry.register(factory)
    config = server._parse_game_config({"villain_type": "random"})
    factory.runner = "changed-source"
    with pytest.raises(ValueError, match="changed"):
        registry.load(key, config.to_rust(), seat=1, seed=2)
    assert not factory.sessions


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
