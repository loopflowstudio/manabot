"""Academy lists trained checkpoints, refuses mismatches by name, and plays them."""

import hashlib
from pathlib import Path
import random

from fastapi.testclient import TestClient
import pytest
import torch

from manabot.backends import resolve
from manabot.model.world import WORLD
from manabot.sim.policy_provider import BaselineFactory
from managym.academy import server, trace as trace_store
from managym.academy.opponent import Opponent, opponent_catalog, policy_registry
from managym.academy.server import app
from managym.policy import PolicyRegistry, PolicySpec
from tests.academy.test_play_modes import _recorded_traces, _write_tiny_checkpoint
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint


@pytest.fixture()
def runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.delenv("ACADEMY_PLAY_CHECKPOINTS", raising=False)
    monkeypatch.setenv("MANABOT_RUNS_DIR", str(tmp_path / ".runs"))
    monkeypatch.setattr("managym.academy.opponent._registry", None)
    server.SESSION_REGISTRY.clear()
    # Catalog checkpoints sample their actions; fix the game's length.
    torch.manual_seed(0)
    current = tmp_path / ".runs" / "self-play" / "step_0.pt"
    _write_tiny_checkpoint(current)
    checkpoint = read_checkpoint(current, map_location="cpu", weights_only=False)
    checkpoint["world_binding"]["world"] = "w5"
    (tmp_path / ".runs" / "older").mkdir()
    write_checkpoint(checkpoint, tmp_path / ".runs" / "older" / "step_0.pt")
    return tmp_path / ".runs"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id(path: Path) -> str:
    return policy_registry().register(PolicySpec("manabot", str(path)))


def _new_game_error(client: TestClient, config: dict) -> str:
    with client.websocket_connect("/ws/play") as websocket:
        websocket.send_json({"type": "new_game", "config": config})
        payload = websocket.receive_json()
    assert payload["type"] == "error", payload
    return payload["message"]


def test_catalog_lists_admissible_and_refused_checkpoints(runs: Path) -> None:
    with TestClient(app) as client:
        catalog = client.get("/api/opponents").json()

    assert catalog["catalog_error"] is None
    entries = {entry["name"]: entry for entry in catalog["opponents"]}
    current = entries["self-play/step_0"]
    assert current["available"] and current["reason"] is None
    assert not current["supports_search"]
    assert "W/D/L value head" in current["search_reason"]
    assert current["sha256"] == _id(runs / "self-play" / "step_0.pt")
    details = {line["label"]: line["value"] for line in current["details"]}
    assert details["World"] == WORLD
    assert details["Source"] == "runs"
    assert details["Model"].startswith("width 64")
    assert set(details["Bound decks"].split(" vs ")) == {"ur_lessons", "gw_allies"}
    assert details["Updates"] == "not recorded"
    older = entries["older/step_0"]
    assert not older["available"]
    assert older["reason"] == f"trained in w5, engine is {WORLD}"


@pytest.mark.parametrize("kind", ["heuristic", "greedy"])
@pytest.mark.parametrize("extra_alias", [None, "challenger"])
def test_anchor_names_omit_case_only_aliases(
    monkeypatch: pytest.MonkeyPatch, kind: str, extra_alias: str | None
) -> None:
    factory = BaselineFactory(PolicySpec("manabot", config={"kind": kind}))
    descriptor = factory.describe()
    registry = PolicyRegistry(providers={})
    monkeypatch.setattr("managym.academy.opponent._registry", registry)
    key = registry.register(factory)
    assert descriptor.alias == kind
    assert key == descriptor.id
    assert registry.register(factory, name=kind.upper()) == key
    if extra_alias is not None:
        assert registry.register(factory, name=extra_alias) == key

    expected = kind.title() + (f" · {extra_alias}" if extra_alias else "")
    (entry,) = opponent_catalog()["opponents"]
    assert entry["name"] == expected
    assert entry["sha256"] == key
    record = Opponent(registry.descriptor(key)).record()
    assert record["name"] == expected
    assert record["sha256"] == key
    assert record["policy"]["name"] == kind.title()
    assert record["policy"] == descriptor.record()


def test_incompatible_checkpoint_is_refused_with_the_mismatch(runs: Path) -> None:
    with TestClient(app) as client:
        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _id(runs / "older" / "step_0.pt"),
            },
        )
        assert f"trained in w5, engine is {WORLD}" in message

        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _id(runs / "self-play" / "step_0.pt"),
                "hero_deck": "interactive",
                "villain_deck": "interactive",
            },
        )
        assert "was trained on ur_lessons vs gw_allies; choose those decks" in message


@pytest.mark.parametrize(
    ("hero_deck", "villain_deck"),
    [("ur_lessons", "gw_allies"), ("gw_allies", "ur_lessons")],
)
def test_chosen_checkpoint_plays_either_deck_and_is_recorded(
    runs: Path, hero_deck: str, villain_deck: str
) -> None:
    sha256 = _id(runs / "self-play" / "step_0.pt")
    rng = random.Random(4)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "checkpoint",
                        "opponent_sha256": sha256,
                        "hero_deck": hero_deck,
                        "villain_deck": villain_deck,
                        "seed": 7,
                    },
                }
            )
            payload = websocket.receive_json()
            # Enough hero decisions for the bot to act from its own seat.
            for _ in range(40):
                if payload["type"] == "game_over":
                    break
                assert payload["type"] == "observation", payload
                actions = payload["actions"]
                choice = actions[rng.randrange(len(actions))]
                websocket.send_json({"type": "action", "index": choice["index"]})
                payload = websocket.receive_json()
    assert payload["table"]["opponent"]["sha256"] == sha256

    [(_, trace)] = _recorded_traces(trace_store.TRACES_DIR)
    config = trace["config"]
    assert config["villain_type"] == "policy"
    assert config["opponent"]["sha256"] == sha256
    record = config["opponent_checkpoint"]
    assert record["schema"] == "managym-policy-1"
    assert record["policy"]["artifacts"]["actor"] == _sha(
        runs / "self-play" / "step_0.pt"
    )
    assert record["policy"]["runner"]


def test_replaced_checkpoint_cannot_replace_playable_match(runs: Path) -> None:
    checkpoint = runs / "self-play" / "step_0.pt"
    sha256 = _id(checkpoint)
    game = server.GameSession(trace_dir=trace_store.TRACES_DIR)
    game.new_game({"villain_type": "passive", "seed": 79})
    before = game.current_recovery("explicit_resync")["frame"]
    checkpoint.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="not a policy checkpoint|changed"):
        game.new_game({"villain_type": "checkpoint", "opponent_sha256": sha256})
    assert game.current_recovery("explicit_resync")["frame"] == before


def _deck_import() -> dict[str, object]:
    return {
        "hero": {"source": "10 Forest\n10 Mountain\n2 Avatar Enthusiasts"},
        "villain": {"source": "20 Island"},
    }


def test_imported_decks_play_world_checkpoints_and_refuse_setup_bindings(
    runs: Path,
) -> None:
    from manabot.backends.torch.checkpoint import save_checkpoint
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig

    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(world="world"))
    world = runs / "world" / "step_0.pt"
    save_checkpoint(agent, space, world, player_configs=None)
    rng = random.Random(0)
    with TestClient(app) as client:
        message = _new_game_error(
            client,
            {
                "villain_type": "checkpoint",
                "opponent_sha256": _id(runs / "self-play" / "step_0.pt"),
                "deck_import": _deck_import(),
            },
        )
        assert "was trained on" in message
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "villain_type": "checkpoint",
                        "opponent_sha256": _id(world),
                        "deck_import": _deck_import(),
                        "seed": 7,
                    },
                }
            )
            payload = websocket.receive_json()
            for _ in range(40):
                if payload["type"] == "game_over":
                    break
                assert payload["type"] == "observation", payload
                actions = payload["actions"]
                choice = actions[rng.randrange(len(actions))]
                websocket.send_json({"type": "action", "index": choice["index"]})
                payload = websocket.receive_json()
    [(_, trace)] = _recorded_traces(trace_store.TRACES_DIR)
    assert trace["config"]["opponent"]["sha256"] == _id(world)
    villain_turns = [
        event for event in trace["events"] if event.get("actor") == "villain"
    ]
    assert villain_turns


def test_run_facts_join_exact_bytes_and_prefer_run_file(runs: Path) -> None:
    import json
    import shutil

    from manabot.model.catalog import discover_checkpoints

    directory = runs / "self-play"
    raw = directory / "step_0.pt"
    copy = directory / "tracker" / "long-id" / "weights.pt"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(raw, copy)
    record = {
        "id": "run-123",
        "stages": [
            {
                "id": "self-play",
                "status": "completed",
                "updates": 7,
                "learner_transitions": 1234,
                "artifacts": {
                    "raw": {"path": "/another/machine/raw.pt", "sha256": _sha(raw)}
                },
            }
        ],
    }
    (directory / "run.json").write_text(json.dumps(record))
    found, _ = discover_checkpoints([], backend=resolve("torch"), runs_dir=runs)
    entry = next(entry for entry in found if entry.sha256 == _sha(raw))
    assert entry.path == raw.resolve()
    assert (
        entry.run,
        entry.run_id,
        entry.stage,
        entry.weights,
        entry.updates,
        entry.transitions,
    ) == ("self-play", "run-123", "self-play", "raw", 7, 1234)
    record["stages"][0]["artifacts"]["raw"]["sha256"] = "f" * 64
    (directory / "run.json").write_text(json.dumps(record))
    found, _ = discover_checkpoints([], backend=resolve("torch"), runs_dir=runs)
    entry = next(entry for entry in found if entry.sha256 == _sha(raw))
    assert entry.updates is None and entry.transitions is None and entry.run is None


def test_monitoring_counts_override_stage_and_unfinished_exports_stay_unknown(
    runs: Path,
) -> None:
    import json

    from manabot.model.catalog import discover_checkpoints

    directory = runs / "self-play"
    raw = directory / "step_0.pt"
    record = {
        "stages": [
            {
                "id": "self-play",
                "status": "running",
                "updates": 99,
                "learner_transitions": 999,
                "artifacts": {"raw": {"sha256": _sha(raw)}},
            }
        ]
    }
    path = directory / "run.json"
    path.write_text(json.dumps(record))
    found, _ = discover_checkpoints([directory], backend=resolve("torch"))
    assert found[0].updates is None and found[0].transitions is None
    record["monitoring_checkpoints"] = [
        {"updates": 4, "learner_transitions": 32, "artifact": {"sha256": _sha(raw)}}
    ]
    path.write_text(json.dumps(record))
    found, _ = discover_checkpoints([directory], backend=resolve("torch"))
    assert (found[0].updates, found[0].transitions) == (4, 32)


def test_portable_container_is_registered_and_plays(runs: Path) -> None:
    from manabot.backends.convert import convert
    from manabot.backends.torch.checkpoint import save_checkpoint
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig
    import managym

    config = server._parse_game_config({"villain_type": "random"})
    space = ObservationSpace()
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world="ur-lessons-vs-gw-allies",
        ),
    )
    path = runs / "portable.pt"
    save_checkpoint(model, space, path, player_configs=config.to_rust())
    converted, _ = convert(path)
    key = _id(converted)
    config = server._parse_game_config(
        {"villain_type": "policy", "opponent_sha256": key}
    )
    engine = managym.Env(seed=42)
    observation, _ = engine.reset(config.to_rust())
    policy = policy_registry().load(
        key, config.to_rust(), seat=int(observation.agent.player_index), seed=0
    )
    try:
        action = policy(engine, observation)
        assert 0 <= action < len(observation.action_space.actions)
        assert policy.last_decision is not None
        assert len(policy.last_decision.probabilities) == len(
            observation.action_space.actions
        )
    finally:
        policy.close()


def test_setup_preview_names_bound_decks_without_loading(runs: Path) -> None:
    key = _id(runs / "self-play" / "step_0.pt")
    with TestClient(app) as client:
        config = {
            "villain_type": "policy",
            "opponent_sha256": key,
            "hero_deck": "interactive",
            "villain_deck": "interactive",
        }
        refused = client.post("/api/policies/check", json=config).json()
        assert refused["compatible"] is False
        assert "was trained on ur_lessons vs gw_allies" in refused["reason"]
        config.update(hero_deck="ur_lessons", villain_deck="gw_allies")
        assert client.post("/api/policies/check", json=config).json() == {
            "compatible": True,
            "reason": None,
        }
