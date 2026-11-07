"""The c5fba4 fact-contribution reading measures what facts add to their objects."""

from pathlib import Path

from etude.experiment import Checkpoint, Checkpoints, ExperimentState, TrainingRun
from etude.run.records import ArtifactRef
import numpy as np
import pytest
import torch

from experiments import tla_frontier_vocabulary as vocabulary_check
from experiments.tla_frontier import EXPERIMENT, SMALL, SizeLabel, frozen
from experiments.tla_frontier_facts import (
    Observation,
    read,
    read_checkpoints,
    read_model,
    sample_positions,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.identity import file_sha256
from manabot.infra.runtime import regime_tag
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint


@pytest.fixture(scope="module")
def positions() -> Observation:
    regimes, protocol = frozen()
    return sample_positions(regimes[SMALL], protocol)


def _fresh() -> ActorCritic:
    regime = frozen()[0][SMALL]
    torch.manual_seed(0)
    return ActorCritic(ObservationSpace(regime.observation), regime.agent)


def test_positions_are_fixed_and_carry_facts_and_effects(
    positions: Observation,
) -> None:
    regimes, protocol = frozen()
    again = sample_positions(regimes[SMALL], protocol)
    assert all(np.array_equal(positions[key], again[key]) for key in positions)
    assert positions["facts"][..., -1].sum() > 0
    assert (positions["effect_members"][..., 0] > 0).sum() > 0


def test_a_fresh_model_adds_nothing_through_facts(positions: Observation) -> None:
    reading = read_model(_fresh(), positions)
    assert reading.facts.objects > 0 and reading.facts.median == 0.0
    assert reading.by_kind and all(kind.mean == 0.0 for kind in reading.by_kind)
    # The shared-effect reducer has no zero-initialized output.
    assert reading.shared_effects.objects > 0
    assert reading.shared_effects.median is not None
    assert reading.shared_effects.median > 0


def test_weighted_facts_register_per_kind(positions: Observation) -> None:
    model = _fresh()
    assert model.facts is not None
    torch.nn.init.normal_(model.facts.output.weight, std=0.1)
    reading = read_model(model, positions)
    assert reading.facts.median is not None and reading.facts.median > 0
    assert all(kind.mean is not None and kind.mean > 0 for kind in reading.by_kind)
    assert sum(kind.objects for kind in reading.by_kind) >= reading.facts.objects


def _checkpoint(
    run: str, tag: str, update: int, artifact: ArtifactRef | None = None
) -> Checkpoint:
    return Checkpoint(
        run=run,
        experiment=EXPERIMENT,
        ordinal=update + 1,
        artifact=artifact
        or ArtifactRef(
            uri=f"file:///fixture/{run}/updates-{update}.pt",
            sha256="a" * 64,
            size_bytes=1,
        ),
        clocks={"updates": update},
        line=run,
        tags=(regime_tag(tag),),
    )


def _run(
    name: str,
    tag: str,
    updates: tuple[int, ...],
    artifact: ArtifactRef | None = None,
) -> TrainingRun:
    return TrainingRun(
        name=name,
        experiment=EXPERIMENT,
        state="finished",
        checkpoints=Checkpoints(
            _checkpoint(name, tag, update, artifact) for update in updates
        ),
    )


def test_frontier_runs_read_initial_milestone_and_endpoint_checkpoints() -> None:
    protocol = frozen()[1]
    sizes: dict[SizeLabel, int] = {"w64": 330, "w160l8": 60}
    state = ExperimentState(
        experiment=EXPERIMENT,
        runs=tuple(
            _run(
                label, f"tla-frontier-c5fba4-{label}", tuple(range(0, end, 25)) + (end,)
            )
            for label, end in sizes.items()
        ),
    )
    found = {
        (label, int(checkpoint.clocks["updates"]))
        for label, _, checkpoint in read_checkpoints(state, "tla_frontier", protocol)
    }
    assert found == {("w64", u) for u in (0, 25, 50, 75, 100, 150, 200, 300, 330)} | {
        ("w160l8", u) for u in (0, 25, 50, 60)
    }


def test_the_reader_reads_every_vocabulary_export(tmp_path: Path) -> None:
    model = _fresh()
    path = tmp_path / "policy.pt"
    save_checkpoint(
        model,
        ObservationSpace(frozen()[0][SMALL].observation),
        path,
        player_configs=None,
    )
    artifact = ArtifactRef(
        uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
    )
    state = ExperimentState(
        experiment=vocabulary_check.EXPERIMENT,
        runs=tuple(
            _run(
                arm,
                f"tla-frontier-c5fba4-vocabulary-{arm}",
                (0, 100),
                artifact=artifact,
            )
            for arm in vocabulary_check.VOCABULARIES
        ),
    )
    reading = read(state, "tla_frontier_vocabulary", tmp_path)
    assert [(c.label, c.update) for c in reading.checkpoints] == [
        ("catalog", 0),
        ("catalog", 100),
        ("hashed", 0),
        ("hashed", 100),
    ]
    assert all(c.reading.facts.median == 0.0 for c in reading.checkpoints)
