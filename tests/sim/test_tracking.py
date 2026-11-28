"""Simulation consumes retained Etude checkpoints and records evaluation facts."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from urllib.parse import unquote, urlsplit

from etude.checkpoints import CheckpointRecord
from etude.storage import DirectoryStorage
from etude.train.read import read, read_all
import pytest
import torch

from manabot.backends.torch.checkpoint import (
    BACKEND as TORCH_BACKEND,
    load_etude_checkpoint,
    save_checkpoint,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra import artifacts
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    RuntimeConfig,
    SimulationConfig,
)
from manabot.policy.protocol import BackendPolicy
from manabot.reports import training
from manabot.sim import sim
from manabot.sim.player import ModelPlayer, RandomPlayer
from tests.fixtures.training_records import recording


@pytest.mark.parametrize("remote", [False, True])
def test_simulation_admits_exact_etude_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
) -> None:
    space = ObservationSpace()
    original = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    policy = tmp_path / "policy.pt"
    save_checkpoint(original, space, policy, player_configs=Match().to_rust())
    directory = tmp_path / "etude"
    with recording(
        experiment="fixture",
        run="producer",
        clocks=["updates"],
        directory=directory,
        strict=True,
    ) as producer:
        reference = producer.checkpoint(policy, updates=1)
    record = tmp_path / "checkpoint.json"
    record.write_text(
        read(directory, "fixture", "producer").checkpoints[0].model_dump_json()
    )
    retained = Path(unquote(urlsplit(reference.uri).path))
    if remote:
        storage = DirectoryStorage(tmp_path / "bucket")
        storage.put_file("policy.pt", retained)
        reference = reference.model_copy(update={"uri": "s3://fixture/policy.pt"})
        stored = CheckpointRecord.model_validate_json(record.read_bytes())
        record.write_text(
            stored.model_copy(update={"artifact": reference}).model_dump_json()
        )
        monkeypatch.setattr(artifacts, "storage_from_uri", lambda _: storage)
        retained = tmp_path / "bucket/policy.pt"
    policy.unlink()  # The SDK retained its own bytes.

    class Loader:
        load_policy = staticmethod(TORCH_BACKEND.load_policy)

    player = sim.load_player(str(record), backend=Loader())
    assert isinstance(player, ModelPlayer)
    assert not player.agent.training
    for name, value in original.state_dict().items():
        torch.testing.assert_close(player.agent.state_dict()[name], value)

    retained.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="bytes differ|incomplete"):
        load_etude_checkpoint(record)


def test_simulation_records_game_results_in_etude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    games = [sim.GameRecord(sim.GameOutcome.HERO_WIN, steps=3, duration=0.25)]
    monkeypatch.setattr(sim, "simulate_models", lambda *args: games)
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "launched"))
    monkeypatch.setenv("ETUDE_RUN", "launched-evaluation")
    sim.run_simulation(
        SimulationConfig(hero="random", villain="default", num_games=1),
        RuntimeConfig(name="simulation", runs_dir=tmp_path),
        backend=TORCH_BACKEND,
    )

    runs = read_all(tmp_path / "launched")
    assert len(runs) == 1
    assert runs[0].name == "launched-evaluation"
    assert not (tmp_path / "simulation/etude").exists()
    assert list(runs[0].points("game/steps", "games").y) == [3]
    assert list(runs[0].points("evaluation/hero_win_rate", "games").y) == [1]
    assert runs[0].outcome is not None and runs[0].outcome.state == "finished"


def test_dashboard_delivery_failure_retries_without_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    dashboard = training.Dashboard(
        run_id="fixture", config={}, summary={}, rows=[{"rl/loss": 0.5}]
    )
    destination = DirectoryStorage(tmp_path / "destination")
    original = destination.put

    def fail(key: str, data: bytes) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr(destination, "put", fail)
    monkeypatch.setattr(training, "storage_from_uri", lambda _: destination)
    with pytest.raises(RuntimeError, match="delivery incomplete"):
        training.publish_dashboard(dashboard, tmp_path, destination="fixture")
    monkeypatch.setattr(destination, "put", original)
    training.publish_dashboard(dashboard, tmp_path, destination="fixture")
    copied = read(destination, "manabot-monitoring", dashboard.run_id)
    assert list(copied.points("rl/loss", "observation").y) == [0.5]
    settled = [
        table.row(index)
        for table in copied.columns
        for index in range(table.rows)
        if "projection/row_sha256" in table.row(index).values
    ]
    assert len(settled) == 1


def test_simulation_encodes_with_the_models_policy_input() -> None:
    def model(max_actions: int) -> ModelPlayer:
        space = ObservationSpace(ObservationConfig(max_actions=max_actions))
        agent = SimpleNamespace(observation_space=space, device_type="cpu")
        return ModelPlayer("model", cast(BackendPolicy, agent))

    player = model(64)
    assert sim._shared_observation_config(player, player).max_actions == 64
    assert sim._shared_observation_config(RandomPlayer("r"), RandomPlayer("r")) is None
    with pytest.raises(ValueError, match="different observation ABIs"):
        sim._shared_observation_config(player, model(128))


@pytest.mark.parametrize("failure", ["policy", "environment"])
def test_failed_game_aborts_evaluation_and_closes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class FailingEnv:
        closed = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.last_raw_obs = SimpleNamespace(agent=SimpleNamespace(player_index=0))
            instances.append(self)

        def reset(self) -> tuple[dict[str, object], dict[str, object]]:
            return {}, {}

        def step(self, action: int) -> tuple[object, ...]:
            raise RuntimeError("environment failed")

        def close(self) -> None:
            self.closed = True

    class FailingPlayer:
        name = "fixture"

        def get_action(self, obs: object) -> int:
            if failure == "policy":
                raise RuntimeError("policy failed")
            return 0

    instances: list[FailingEnv] = []
    monkeypatch.setattr(sim, "Env", FailingEnv)
    monkeypatch.setattr(sim, "load_player", lambda *args, **kwargs: FailingPlayer())
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("ETUDE_RUN", "failed-evaluation")
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        sim.run_simulation(
            SimulationConfig(
                hero="random", villain="random", num_games=1, num_threads=1
            ),
            RuntimeConfig(name="simulation", runs_dir=tmp_path),
        )
    assert len(instances) == 1 and instances[0].closed
    runs = read_all(tmp_path / "tracking")
    assert len(runs) == 1
    assert not list(runs[0].points("evaluation/hero_win_rate", "games").y)
    assert runs[0].outcome is not None and runs[0].outcome.state == "failed"


def test_game_reaching_step_limit_records_timeout() -> None:
    class LimitedEnv:
        last_raw_obs = SimpleNamespace(agent=SimpleNamespace(player_index=0))

        def reset(self) -> tuple[dict[str, object], dict[str, object]]:
            return {}, {}

        def step(
            self, action: int
        ) -> tuple[dict[str, object], int, bool, bool, dict[str, object]]:
            return {}, 0, False, False, {}

    class Player:
        def get_action(self, obs: object) -> int:
            return 0

    player = cast(sim.Player, Player())
    record = sim._simulate_game(cast(sim.Env, LimitedEnv()), player, player, 1)
    assert record.outcome == sim.GameOutcome.TIMEOUT
    assert record.steps == 1
