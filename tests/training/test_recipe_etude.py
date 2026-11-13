"""Recipes resume through real Etude tracking and keep downstream policy provenance."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from etude.experiment import observe
from etude.jobs import training
from etude.train.read import read
from etude.train.records import Step, Time
from etude.train.run import local_launch, track
from pydantic import BaseModel
import pytest
from typer.testing import CliRunner

from manabot.belief.sampling_data import read_dataset
from manabot.cli import app, experiment_recipe
from manabot.recipes.belief_sampler import BeliefConfig, train
from manabot.recipes.builders import gray_ogre_self_play
from manabot.recipes.local import LocalConfig, train as train_local
from manabot.training.loops import self_play as loop
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
    help_text = runner.invoke(app, ["train", "--help"]).output
    assert "--updates" in help_text
    assert "--steps" not in help_text
    conflict = runner.invoke(app, ["train", "--updates", "1", "--steps", "2"])
    assert conflict.exit_code != 0
    assert "choose --updates or --steps" in conflict.output
    retired = runner.invoke(app, ["train", "--regime", "saved.json"])
    assert retired.exit_code != 0
    assert "f4e1822faf" in retired.output


def test_cli_function_recipes_track_under_the_module_in_distinct_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Config(BaseModel):
        seed: int
        device: str

    def train(config: Config) -> None:
        with track(
            clocks=[Step("updates", "completed updates")], seed=config.seed
        ) as run:
            run.log({"seed": config.seed}, updates=1)

    module = ModuleType("experiments.cli_tracking_fixture")
    module.train = train
    module.alternate = train
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for function in ("train", "alternate"):
        result = runner.invoke(
            app,
            [
                "train",
                "--recipe",
                f"cli_tracking_fixture:{function}",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        name = f"cli_tracking_fixture-{function}-197"
        data = read(tmp_path / ".runs" / name / "etude", "cli_tracking_fixture", name)
        assert data.file is not None
        assert data.reached["updates"] == 1
        assert data.state is not None and data.state.state == "finished"


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
        "--updates",
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
    rerun = runner.invoke(
        app, [arg if arg != "--updates" else "--steps" for arg in command]
    )
    assert rerun.exit_code == 0, rerun.output
    assert "--steps is deprecated; use --updates" in rerun.output
    assert "removed in Manabot 0.3.0" in rerun.output
    assert "leave the recipe's schedule and" in rerun.output
    before = observe(out / "etude", "local").run("smoke")
    assert before.final is not None
    assert before.final.clocks["updates"] == 1
    assert before.final.clocks["training_seconds"] > 0
    extended = runner.invoke(app, command[:-1] + ["2"])
    assert extended.exit_code == 0, extended.output
    assert checkpoint.read_bytes() == original
    assert (out / "checkpoints/00000002-ema.pt").exists()
    data = read(out / "etude", "local", "smoke")
    assert data.reached["updates"] == 2
    assert data.file is not None
    assert len(data.attempts) == 4
    after = observe(out / "etude", "local").run("smoke")
    assert after.final == after.checkpoints.last
    assert after.final is not None
    assert after.final.clocks["updates"] == 2
    assert (
        after.final.clocks["training_seconds"] > before.final.clocks["training_seconds"]
    )
    assert data.reached["training_seconds"] == after.final.clocks["training_seconds"]
    assert before.final in after.checkpoints
    assert all("final" not in checkpoint.tags for checkpoint in after.checkpoints)
    other_seed = runner.invoke(app, command + ["--seed", "5"])
    assert other_seed.exit_code == 2
    assert "belongs to another run" in other_seed.output
    assert "another seed: 197, not 5" in other_seed.output
    assert "Traceback" not in other_seed.output


@pytest.mark.parametrize("published", [False, True])
def test_resume_repairs_a_saved_milestone_before_training_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    settings = gray_ogre_self_play(horizon=3).model_copy(update={"checkpoint_every": 2})
    record_checkpoint = loop.record_checkpoint

    def lose_attempt(
        out: Path, state: loop.LoopState, run: loop.Tracker | None
    ) -> Path:
        if state.update != 2:
            return record_checkpoint(out, state, run)
        saved = loop.load_state(out / "state.pt", settings, 197)
        assert saved is not None and saved.update == 2
        if published:
            record_checkpoint(out, state, run)
        raise RuntimeError("lost attempt after saving milestone")

    def train() -> None:
        with local_launch(out=tmp_path, experiment="milestone"):
            with track(
                clocks=[
                    Step("updates", "collection/update iterations"),
                    Time("training_seconds", "active collection and learning seconds"),
                ],
                seed=197,
            ) as run:
                loop.self_play(settings, run.out, seed=197, updates=3, run=run)

    with monkeypatch.context() as patch:
        patch.setattr(loop, "record_checkpoint", lose_attempt)
        with pytest.raises(RuntimeError, match="lost attempt"):
            train()
    milestone = tmp_path / "checkpoints/00000002-raw.pt"
    assert milestone.exists() == published
    earlier = {
        path: path.read_bytes() for path in (tmp_path / "checkpoints").glob("*.pt")
    }
    saved = loop.load_state(tmp_path / "state.pt", settings, 197)
    assert saved is not None and saved.update == 2
    coordinate = loop.clocks(saved)

    train()

    assert milestone.exists()
    for path, payload in earlier.items():
        assert path.read_bytes() == payload
    recorded = observe(tmp_path / "etude", "milestone").run(tmp_path.name)
    milestones = [cp for cp in recorded.checkpoints if cp.clocks["updates"] == 2]
    assert len(milestones) == 1
    assert milestones[0].clocks == coordinate
    assert recorded.final is not None and recorded.final.clocks["updates"] == 3


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
    recorded = observe(tmp_path / "etude", "belief").run(tmp_path.name)
    samplers = [
        checkpoint for checkpoint in recorded.checkpoints if "belief" in checkpoint.tags
    ]
    assert len(samplers) == 2
    assert all(checkpoint.clocks["training_seconds"] > 0 for checkpoint in samplers)


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

    settings = gray_ogre_self_play(horizon=1)
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


def test_cli_recipe_config_needs_only_the_fields_it_receives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config(BaseModel):
        seed: int
        device: str

    def train(config: Config) -> None:
        pass

    module = ModuleType("experiments.cli_config_fixture")
    module.train = train
    monkeypatch.setitem(sys.modules, module.__name__, module)
    function, config_type = experiment_recipe("cli_config_fixture")
    config = config_type.model_validate({"seed": 197, "device": "cpu"})
    assert config.model_dump() == {"seed": 197, "device": "cpu"}
    assert function is train
