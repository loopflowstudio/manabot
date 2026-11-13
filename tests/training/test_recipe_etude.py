"""Recipes resume through real Etude tracking and keep downstream policy provenance."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from etude.jobs import training
from etude.train.read import read
from etude.train.run import local_launch
import pytest
from typer.testing import CliRunner

from manabot.belief.sampling_data import read_dataset
from manabot.cli import app
from manabot.recipes.belief_sampler import BeliefConfig, train
from manabot.recipes.builders import gray_ogre_self_play
from manabot.recipes.local import LocalConfig, train as train_local
from manabot.training.loops.self_play import new_state, root_match, save_checkpoint
from manabot.training.regime import SelectionGameSpec
from manabot.training.rollout.selection_data import SelectionDataset, collect_selection


def test_cli_lists_recipes_and_refuses_saved_execution() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["train", "--list"])
    assert listed.exit_code == 0, listed.output
    assert {"local", "baseline", "elemental", "belief", "batched_baseline"} <= set(
        listed.output.splitlines()
    )
    retired = runner.invoke(app, ["train", "--regime", "saved.json"])
    assert retired.exit_code != 0
    assert "f09f5c2919" in retired.output


def test_local_cli_stop_resume_and_target_only_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "smoke"
    stop = tmp_path / "stop"
    stop.touch()
    monkeypatch.setenv("ETUDE_STOP_FILE", str(stop))
    runner = CliRunner()
    command = [
        "train",
        "--recipe",
        "local",
        "--device",
        "cpu",
        "--out",
        str(out),
        "--steps",
        "1",
    ]
    stopped = runner.invoke(app, command)
    assert stopped.exit_code == 75, stopped.output
    assert (out / "state.pt").exists()
    stop.unlink()
    finished = runner.invoke(app, command)
    assert finished.exit_code == 0, finished.output
    checkpoint = out / "checkpoints/00000001-ema.pt"
    original = checkpoint.read_bytes()
    rerun = runner.invoke(app, command)
    assert rerun.exit_code == 0, rerun.output
    extended = runner.invoke(app, command[:-1] + ["2"])
    assert extended.exit_code == 0, extended.output
    assert checkpoint.read_bytes() == original
    assert (out / "checkpoints/00000002-ema.pt").exists()
    data = read(out / "etude", "local", "smoke")
    assert data.reached["updates"] == 2
    assert data.file is not None
    assert len(data.attempts) == 4
    assert all(
        "final" in checkpoint.tags
        for checkpoint in data.checkpoints
        if checkpoint.at["updates"] in {1, 2}
    )


def test_belief_extension_has_new_policy_histories_and_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = BeliefConfig(seed=197)
    with local_launch(out=tmp_path, experiment="belief"):
        train(config)
        policy = tmp_path / "policy/checkpoints/00000001-raw.pt"
        histories = tmp_path / "histories-00000001-raw.json"
        sampler = tmp_path / "belief-00000001-raw.pt"
        original = {path: path.read_bytes() for path in (policy, histories, sampler)}
        train(config)
        monkeypatch.setenv("ETUDE_UNTIL_CLOCK", "updates")
        monkeypatch.setenv("ETUDE_UNTIL_VALUE", "2")
        train(config)
    for path, payload in original.items():
        assert path.read_bytes() == payload
    later = tmp_path / "policy/checkpoints/00000002-raw.pt"
    dataset = read_dataset(tmp_path / "histories-00000002-raw.json")
    assert dataset.policy_identity == hashlib.sha256(later.read_bytes()).hexdigest()
    assert dataset.policy_identity != read_dataset(histories).policy_identity
    assert (tmp_path / "belief-00000002-raw.pt").exists()


def test_typed_job_invokes_the_recipe_and_reports_planned_stop(tmp_path: Path) -> None:
    """Use the SDK's actual function caller, with no provider or rented machine."""

    config = LocalConfig(seed=197)
    job = training(train_local, config=config, smoke=config, max_hours=0.1)
    config_file = tmp_path / "config.json"
    assert job.config is not None
    config_file.write_text(job.config)
    stop = tmp_path / "stop"
    stop.touch()
    out = tmp_path / "out"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("ETUDE_")
    }
    env.update(
        ETUDE_CONFIG=str(config_file),
        ETUDE_OUT=str(out),
        ETUDE_EXPERIMENT="transport",
        ETUDE_RUN="transport",
        ETUDE_TRACK_DIR=str(tmp_path / "tracking"),
        ETUDE_STOP_FILE=str(stop),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "etude.call",
            "--function",
            "manabot.recipes.local:train",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert (out / "state.pt").exists()
    state = read(tmp_path / "tracking", "transport", "transport")
    assert state.state is not None and state.state.state == "stopped"


def test_selection_step_keeps_replayed_population_and_refuses_stale_output(
    tmp_path: Path,
) -> None:

    settings = gray_ogre_self_play()
    policy = save_checkpoint(tmp_path / "policy", new_state(settings, 197))
    population = tuple(
        SelectionGameSpec(
            seed=100 + i,
            action_seed=200 + i,
            assignment=i % 2,
            split="development" if i < 2 else "held_out",
        )
        for i in range(4)
    )
    target = tmp_path / "selection.json"
    collect_selection(policy, target, population, match=root_match(settings))
    dataset = SelectionDataset.model_validate_json(target.read_bytes())
    assert len(dataset.games) == 4 and all(game.rows for game in dataset.games)
    assert dataset.policy_sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert len(list(tmp_path.glob("selection-games-*/*.jsonl"))) == 4
    before = target.read_bytes()
    assert (
        collect_selection(policy, target, population, match=root_match(settings))
        == target
    )
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="other collection inputs"):
        collect_selection(
            policy, target, population, match=root_match(settings), reference="uniform"
        )
    other = save_checkpoint(tmp_path / "other", new_state(settings, 198))
    with pytest.raises(ValueError, match="another policy"):
        collect_selection(other, target, population, match=root_match(settings))
