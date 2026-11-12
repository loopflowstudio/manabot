"""World fixtures for the frozen TLA frontier tests.

The frontier protocol (`experiments/protocols/tla-frontier-c5fba4-v1.json`)
names the runtime world it was frozen on, and `frozen()` refuses any other.
Scheduling and analysis tests read synthetic outcomes that must not depend on
the evolving native catalog, so they declare the protocol's world
(`frontier_world`). Tests that play or train cannot run under that identity;
they read a temporary copy of the protocol bound to the current world
(`frontier_runtime`), never the frozen file. Rejection of the current runtime
is tested without either seam.
"""

from pathlib import Path

import pytest

from experiments import tla_frontier
from experiments.tla_frontier import Protocol, RuntimeWorld
import managym

_RUNTIME_WORLD = managym.world_identity


def _protocol() -> Protocol:
    return Protocol.model_validate_json(tla_frontier.PROTOCOL.read_bytes())


@pytest.fixture
def frontier_world(monkeypatch: pytest.MonkeyPatch) -> RuntimeWorld:
    """Declare the protocol's world, for tests that never play or train."""
    world = _protocol().runtime_world
    monkeypatch.setattr(managym, "world_identity", world.model_dump)
    return world


@pytest.fixture
def frontier_runtime(
    frontier_world: RuntimeWorld, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> RuntimeWorld:
    """The real world identity, with `frozen()` reading a protocol copy bound to it.

    Overrides `frontier_world` (which module markers may apply first). The
    copy changes only `runtime_world`; regime and shape checks stay real.
    """
    monkeypatch.setattr(managym, "world_identity", _RUNTIME_WORLD)
    current = RuntimeWorld.model_validate(_RUNTIME_WORLD())
    copy = tmp_path / "frontier-protocol.json"
    copy.write_text(
        _protocol().model_copy(update={"runtime_world": current}).model_dump_json()
    )
    monkeypatch.setattr(tla_frontier, "PROTOCOL", copy)
    return current
