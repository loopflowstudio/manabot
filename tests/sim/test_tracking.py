"""Simulation consumes retained Etude checkpoints and records evaluation facts."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

from etude.run.storage import DirectoryStorage
from etude.train.read import read, read_all
from etude.train.records import CheckpointRecord, Step
from etude.train.run import track
import pytest
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RuntimeConfig, SimulationConfig
from manabot.model import checkpoint
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_etude_checkpoint, save_checkpoint
from manabot.reports import training
from manabot.sim import sim
from manabot.sim.player import ModelPlayer


@pytest.mark.parametrize("remote", [False, True])
def test_simulation_admits_exact_etude_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
) -> None:
    space = ObservationSpace()
    original = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    policy = tmp_path / "policy.pt"
    save_checkpoint(original, space, policy, player_configs=Match().to_rust())
    directory = tmp_path / "etude"
    with track(
        experiment="fixture",
        run="producer",
        clocks=[Step("updates", "updates")],
        directory=directory,
        strict=True,
    ) as producer:
        reference = producer.checkpoint(policy, updates=1)
    record = next((directory / "producer/checkpoints").glob("*.json"))
    retained = Path(unquote(urlsplit(reference.uri).path))
    if remote:
        storage = DirectoryStorage(tmp_path / "bucket")
        storage.put_file("policy.pt", retained)
        reference = reference.model_copy(update={"uri": "s3://fixture/policy.pt"})
        stored = CheckpointRecord.model_validate_json(record.read_bytes())
        record.write_text(
            stored.model_copy(update={"artifact": reference}).model_dump_json()
        )
        monkeypatch.setattr(checkpoint, "storage_from_uri", lambda _: storage)
        retained = tmp_path / "bucket/policy.pt"
    policy.unlink()  # The SDK retained its own bytes.
    player = sim.load_player(str(record))
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
    stats = sim.GameStats()
    stats.record_game(sim.GameOutcome.HERO_WIN, steps=3, duration=0.25)
    monkeypatch.setattr(sim, "simulate_models", lambda *args: stats)
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "launched"))
    monkeypatch.setenv("ETUDE_RUN", "launched-evaluation")
    sim.run_simulation(
        SimulationConfig(hero="random", villain="default", num_games=1),
        RuntimeConfig(name="simulation", runs_dir=tmp_path),
    )

    runs = read_all(tmp_path / "launched")
    assert len(runs) == 1
    assert runs[0].name == "launched-evaluation"
    assert not (tmp_path / "simulation/etude").exists()
    assert list(runs[0].points("game/steps", "games").y) == [3]
    assert list(runs[0].points("evaluation/hero_win_rate", "games").y) == [1]
    assert runs[0].state is not None and runs[0].state.state == "finished"


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
        for table in copied.tables
        for index in range(table.rows)
        if "projection/row_sha256" in table.row(index)["values"]
    ]
    assert len(settled) == 1
