"""Recipes use Etude run inputs and keep downstream policy provenance."""

import hashlib
from pathlib import Path

from etude.train.tracking import track
import pytest

from manabot.belief.sampling_data import read_dataset
from manabot.recipes.belief_sampler import BeliefConfig, runs, train
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import (
    CLOCKS,
    Counts,
    create,
    root_match,
    self_play,
    write_policy,
)
from manabot.training.regime import SelectionGameSpec
from manabot.training.rollout.selection_data import SelectionDataset, collect_selection


def test_belief_sampler_uses_given_policy_and_restarts_its_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "policy"))
    with track(
        experiment="belief_policy",
        run="belief_policy",
        clocks=CLOCKS,
        directory=tmp_path / "track",
    ) as run:
        policy = self_play(gray_ogre_self_play(horizon=1), run, updates=1)
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    job = runs()
    assert len(job.inputs) == 1
    assert job.inputs[0].name == "start"
    assert job.inputs[0].sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert job.inputs[0].run == "belief_policy"
    # Exercise the recipe with the verified input supplied. Pinned Etude's
    # local train command does not yet deliver job inputs (see training guide).
    given = tmp_path / "inputs"
    given.mkdir()
    (given / "start").write_bytes(policy.read_bytes())
    original = policy.read_bytes()
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "sampler"))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "belief")
    monkeypatch.setenv("ETUDE_RUN", "sampler")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(given))
    train(BeliefConfig())
    dataset = read_dataset(tmp_path / "sampler/histories.json")
    assert dataset.policy_identity == hashlib.sha256(original).hexdigest()
    assert policy.read_bytes() == original
    assert (tmp_path / "sampler/belief.pt").is_file()


def test_selection_step_keeps_replayed_population_and_refuses_stale_output(
    tmp_path: Path,
) -> None:

    settings = gray_ogre_self_play(horizon=1)
    learner, _ = create(settings, seed=197)
    policy = write_policy(
        learner.agent, settings, Counts(), tmp_path / "policy.pt", "raw", seed=197
    )
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
    learner, _ = create(settings, seed=198)
    other = write_policy(
        learner.agent, settings, Counts(), tmp_path / "other.pt", "raw", seed=198
    )
    with pytest.raises(ValueError, match="another policy"):
        collect_selection(other, target, population, match=root_match(settings))
