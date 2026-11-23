"""Recipes use Etude run inputs and keep downstream policy provenance."""

from collections.abc import Callable
import hashlib
import math
from pathlib import Path
from typing import Literal

from etude.experiment import observe
import numpy as np
import pytest

from experiments import belief
from manabot.backends.torch.checkpoint import BACKEND as TORCH
from manabot.belief.sampling_data import read_dataset
from manabot.recipes import elemental
from manabot.recipes.belief_sampler import BeliefConfig, train
from manabot.recipes.builders import gray_ogre_self_play
from manabot.recipes.local import LocalConfig
from manabot.training.loops.self_play import (
    CLOCKS,
    Counts,
    create,
    root_match,
    self_play,
    write_policy,
)
from manabot.training.rollout.selection_data import (
    SelectionDataset,
    SelectionDraw,
    SelectionGameSpec,
    collect_selection,
    selection_prediction,
)
from tests.fixtures.training_records import recording


def test_elemental_records_games_for_one_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ETUDE_OUT", str(out))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "elemental")
    monkeypatch.setenv("ETUDE_RUN", "elemental@0")
    monkeypatch.setenv("ETUDE_SEED", "197")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    elemental.train(
        elemental.ElementalConfig.model_validate_json(
            '{"record_games": 1, "updates": 1, "streams": 2}'
        )
    )
    assert any(path.stat().st_size for path in (out / "games").glob("*.jsonl"))


@pytest.mark.parametrize(
    "config",
    [
        lambda: LocalConfig(backend="mlx", device="cuda"),
        lambda: elemental.ElementalConfig(backend="mlx", device="cuda"),
        lambda: elemental.ElementalConfig(backend="mlx", validate_numerics=True),
    ],
)
def test_recipe_refuses_backend_device_before_opening_a_run(
    config: Callable[[], object],
) -> None:
    # A refusal inside `track` would leave a failed run the corrected config cannot reuse.
    with pytest.raises(ValueError):
        config()


def test_belief_sampler_uses_given_policy_and_restarts_its_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ETUDE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "policy"))
    monkeypatch.setenv("ETUDE_JOB_NAME", belief.POLICY)
    with recording(
        experiment="belief",
        run="policy@0",
        clocks=CLOCKS,
        directory=tmp_path / "track",
    ) as run:
        policy = self_play(gray_ogre_self_play(horizon=1), run, updates=1)
    monkeypatch.delenv("ETUDE_JOB_NAME")
    # The experiment declares the sampler from the finished policy run it reads.
    state = observe(tmp_path / "track", "belief")
    declared = belief.training_jobs(state)
    assert [job.name for job in declared] == [belief.POLICY, belief.SAMPLER]
    job = declared[1].plain()
    assert len(job.inputs) == 1
    assert job.inputs[0].name == "start"
    assert job.inputs[0].sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert job.inputs[0].run == "belief/policy@0"
    # Exercise the recipe with the verified input supplied, as a launch delivers it.
    given = tmp_path / "inputs"
    given.mkdir()
    (given / "start").write_bytes(policy.read_bytes())
    original = policy.read_bytes()
    monkeypatch.setenv("ETUDE_OUT", str(tmp_path / "sampler"))
    monkeypatch.setenv("ETUDE_EXPERIMENT", "belief")
    monkeypatch.setenv("ETUDE_RUN", "sampler@0")
    monkeypatch.setenv("ETUDE_TRACK_DIR", str(tmp_path / "track"))
    monkeypatch.setenv("ETUDE_INPUTS_DIR", str(given))
    train(BeliefConfig())
    dataset = read_dataset(tmp_path / "sampler/histories.json")
    assert dataset.policy_identity == hashlib.sha256(original).hexdigest()
    assert policy.read_bytes() == original
    assert (tmp_path / "sampler/belief.pt").is_file()


@pytest.mark.parametrize(
    ("reference", "expected_kl"),
    [
        ("uniform", 0.5 * math.log(1.5) + 0.5 * math.log(0.75)),
        ("action_type_uniform", 0.25 * math.log(2)),
    ],
)
def test_selection_statistics_follow_the_draw_and_legal_offer_types(
    reference: Literal["uniform", "action_type_uniform"], expected_kl: float
) -> None:
    # The first two offers share a type. The final slot is padding.
    encoded = {
        "actions": np.eye(2)[[0, 0, 1, 1]],
        "actions_valid": np.array([1, 1, 1, 0]),
    }
    prediction = selection_prediction(
        SelectionDraw(2, -0.75, np.array([0.5, 0.25, 0.25, 0], dtype=np.float32)),
        encoded,
        reference,
    )
    assert (prediction.action, prediction.value, prediction.probability) == (
        2,
        -0.75,
        0.25,
    )
    assert prediction.entropy == pytest.approx(1.5 * math.log(2))
    assert prediction.reference_kl == pytest.approx(expected_kl)


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
    collect_selection(
        policy, target, population, match=root_match(settings), backend=TORCH
    )
    dataset = SelectionDataset.model_validate_json(target.read_bytes())
    assert len(dataset.games) == 4 and all(game.rows for game in dataset.games)
    assert dataset.policy_sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert len(list(tmp_path.glob("selection-games-*/*.jsonl"))) == 4
    before = target.read_bytes()
    assert (
        collect_selection(
            policy, target, population, match=root_match(settings), backend=TORCH
        )
        == target
    )
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="other collection inputs"):
        collect_selection(
            policy,
            target,
            population,
            match=root_match(settings),
            reference="uniform",
            backend=TORCH,
        )
    learner, _ = create(settings, seed=198)
    other = write_policy(
        learner.agent, settings, Counts(), tmp_path / "other.pt", "raw", seed=198
    )
    with pytest.raises(ValueError, match="another policy"):
        collect_selection(
            other, target, population, match=root_match(settings), backend=TORCH
        )
