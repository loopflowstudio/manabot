"""Task c5fba4 fact-contribution reading: how much fact rows move their objects.

Neither the hashed nor the catalog fact embedding has learning evidence, and
both output layers start at zero (`manabot.model.facts`), so a trained model
whose facts still add nothing to its objects is not using them. This reading
trains nothing and plays no scored game. On one fixed sample of positions from
the frontier's 16 frozen pairs it runs each retained checkpoint's encoder and
measures, for every object a fact row or a shared-effect membership names:

- the norm of the fact contribution added to that object, in total and per
  fact kind (the sum of that kind's encoded rows linked to the object), and
- the norm of the shared-effect contribution added to it,

each divided by the norm of the object's embedding before facts. Contributions
are the encoders' own outputs, read through forward hooks, so they are exactly
what attention receives. Medians and means are over (position, object) rows.

The positions do not depend on any checkpoint: `sample_positions` plays one
seeded random-action game per frozen pair (leg 0, the pair's deal seed, native
trivial-decision skipping) and keeps every `STRIDE`th decision, at most
`PER_PAIR` per pair. A game past an observation capacity stops there, as
`recover_capacity` stops it in training.

The reading is descriptive and gates nothing; it is reported beside learning.
It reads the vocabulary check's arms (`experiments.tla_frontier_vocabulary`,
every export) and each frontier run's update 0, reached milestones and
endpoint. A reading near zero after training is a finding that neither design
is being used. Run where the analysis runs, on CPU:

    uv run --extra artifacts --extra evaluation python -m experiments.tla_frontier_facts \\
        --experiment tla_frontier --source <store> --out <facts.json>
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Literal

from etude.experiment import Checkpoint, ExperimentState, TrainingRun, observe
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from experiments import tla_frontier_vocabulary as vocabulary_check
from experiments.tla_frontier import (
    SIZES,
    SMALL,
    Protocol,
    endpoint,
    frozen,
    size_run,
    with_launcher_ends,
)
from manabot.env.match import Match
from manabot.env.observation import (
    FACT_ADDED_SUBTYPE,
    FACT_COMBAT_RESTRICTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_COPY,
    FACT_COPY_EXCEPTION,
    FACT_COUNTER,
    FACT_EFFECT,
    FACT_EFFECT_PROGRAM,
    FACT_EFFECTIVE_COLOR,
    FACT_KEYWORD_COUNTER_TIMESTAMP,
    FACT_KIND_INDEX,
    FACT_OBJECT_INDEX,
    FACT_SET_CREATURE_TYPE,
    FACT_SET_NAME,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import fetch_etude_checkpoint, load_checkpoint
from manabot.model.facts import FactEncoder
from manabot.sim.matchup import DeckMatchup
import managym

type Observation = dict[str, npt.NDArray[np.generic]]
type ReadExperiment = Literal["tla_frontier", "tla_frontier_vocabulary"]

POSITION_SEED = 21593001
STRIDE = 4
PER_PAIR = 32
MAX_DECISIONS = 2000
BATCH = 64
KIND_NAMES: dict[int, str] = {
    FACT_COUNTER: "counter",
    FACT_ADDED_SUBTYPE: "added_subtype",
    FACT_SET_NAME: "set_name",
    FACT_SET_CREATURE_TYPE: "set_creature_type",
    FACT_USED_ABILITY: "used_ability",
    FACT_EFFECT: "effect",
    FACT_COPY: "copy",
    FACT_COPY_EXCEPTION: "copy_exception",
    FACT_TRANSFORMS: "transforms",
    FACT_COMBAT_RESTRICTION: "combat_restriction",
    FACT_EFFECT_PROGRAM: "effect_program",
    FACT_EFFECTIVE_COLOR: "effective_color",
    FACT_KEYWORD_COUNTER_TIMESTAMP: "keyword_counter_timestamp",
    FACT_CONTINUATION_PROGRAM: "continuation_program",
}


# -- Positions ----------------------------------------------------------------


def sample_positions(protocol: Protocol, observation: ObservationConfig) -> Observation:
    """The fixed positions, stacked: [P, ...] per observation key.

    Every checkpoint reads these same arrays, so readings differ only by model.
    """
    regimes, _ = frozen()
    source = regimes[SMALL].deck_source
    assert source is not None and source.entry_ids is not None
    space = ObservationSpace(observation)
    rng = random.Random(POSITION_SEED)
    kept: list[Observation] = []
    for pair in protocol.pairs:
        matchup = DeckMatchup(
            population_sha256=protocol.corpus_sha256,
            entries=(source.entry_ids[pair.first], source.entry_ids[pair.second]),
            lists=(source.decks[pair.first], source.decks[pair.second]),
        )
        env = managym.Env(seed=pair.deal, skip_trivial=True)
        env.reset(Match(matchup.match(0)).to_rust())
        taken = 0
        for decision in range(MAX_DECISIONS):
            player = env.current_agent_index()
            if player is None or taken == PER_PAIR:
                break
            raw = env.observation_for_player(player)
            if decision % STRIDE == 0:
                try:
                    kept.append(space.encode(raw))
                except ValueError as error:
                    if "capacity exceeded" not in str(error):
                        raise
                    break
                taken += 1
            _, _, done, truncated, _ = env.step(
                rng.randrange(len(raw.action_space.actions))
            )
            if done or truncated:
                break
    if not kept:
        raise ValueError("the frozen pairs produced no positions")
    return {key: np.stack([row[key] for row in kept]) for key in kept[0]}


# -- Contributions ------------------------------------------------------------


class Contribution(StrictModel):
    """Contribution norm / pre-fact object norm over the objects it reaches."""

    objects: int
    median: float | None
    mean: float | None


class KindContribution(Contribution):
    kind: int
    name: str


class ModelContribution(StrictModel):
    facts: Contribution
    by_kind: list[KindContribution]
    shared_effects: Contribution


@dataclass
class _Captured:
    """One batch's encoder inputs and outputs, from forward hooks."""

    pre: torch.Tensor | None = None  # [B,N,D] objects entering the fact encoder
    rows: torch.Tensor | None = None  # [B,F,D] encoded fact rows (output layer)
    effects: torch.Tensor | None = None  # [B,N,D] shared-effect delta


@contextmanager
def _hooks(model: ActorCritic, captured: _Captured) -> Iterator[None]:
    facts = model.facts
    if not isinstance(facts, FactEncoder):
        raise ValueError("the reading needs a row fact encoder")

    def fact_input(_: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.pre = args[1].detach()

    def fact_rows(_: nn.Module, __: tuple[torch.Tensor], out: torch.Tensor) -> None:
        captured.rows = out.detach()

    def effect_delta(
        _: nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        captured.effects = (out - args[3]).detach()

    handles = [
        facts.register_forward_pre_hook(fact_input),
        facts.output.register_forward_hook(fact_rows),
    ]
    if model.shared_effects is not None:
        handles.append(model.shared_effects.register_forward_hook(effect_delta))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _ratios(contribution: torch.Tensor, pre: torch.Tensor) -> list[float]:
    """Row norms of [R,D] `contribution` over [R,D] `pre`, skipping zero `pre`."""
    base = pre.norm(dim=-1)
    keep = base > 0
    return (contribution.norm(dim=-1)[keep] / base[keep]).tolist()


def _summary(values: list[float]) -> Contribution:
    return Contribution(
        objects=len(values),
        median=median(values) if values else None,
        mean=mean(values) if values else None,
    )


def _reached(
    shape: torch.Size, batch: torch.Tensor, objects: torch.Tensor
) -> torch.Tensor:
    """[B,N] mask of the (batch, zero-based object) pairs listed."""
    reached = torch.zeros(shape, dtype=torch.bool)
    reached[batch, objects] = True
    return reached


def _fact_ratios(
    pre: torch.Tensor, rows: torch.Tensor, links: torch.Tensor, mask: torch.Tensor
) -> list[float]:
    """Ratios for each object the [B,F] `mask`ed fact rows reach, summing its rows.

    `links` holds each row's one-based object; `mask` admits only linked rows.
    """
    batch, fact = torch.nonzero(mask, as_tuple=True)
    objects = links[batch, fact] - 1
    delta = torch.zeros_like(pre).index_put(
        (batch, objects), rows[batch, fact], accumulate=True
    )
    reached = _reached(pre.shape[:2], batch, objects)
    return _ratios(delta[reached], pre[reached])


def read_model(model: ActorCritic, positions: Observation) -> ModelContribution:
    """`model`'s fact and shared-effect contributions on `positions`."""
    model.eval()
    totals: list[float] = []
    effects: list[float] = []
    by_kind: dict[int, list[float]] = {}
    captured = _Captured()
    with _hooks(model, captured), torch.no_grad():
        for start in range(0, len(positions["facts"]), BATCH):
            batch = {
                key: torch.from_numpy(value[start : start + BATCH])
                for key, value in positions.items()
            }
            model(batch)
            assert captured.pre is not None and captured.rows is not None
            pre, rows = captured.pre, captured.rows
            facts = batch["facts"]
            links = facts[..., FACT_OBJECT_INDEX].long()
            kinds = facts[..., FACT_KIND_INDEX].long()
            linked = (facts[..., FACT_VALID_INDEX] == 1) & (links > 0)
            totals += _fact_ratios(pre, rows, links, linked)
            for kind in kinds[linked].unique().tolist():
                by_kind.setdefault(kind, []).extend(
                    _fact_ratios(pre, rows, links, linked & (kinds == kind))
                )
            if captured.effects is not None:
                members = batch["effect_members"][..., 0].long()
                live, member = torch.nonzero(members > 0, as_tuple=True)
                reached = _reached(pre.shape[:2], live, members[live, member] - 1)
                effects += _ratios(captured.effects[reached], pre[reached])
    return ModelContribution(
        facts=_summary(totals),
        by_kind=[
            KindContribution(
                kind=kind,
                name=KIND_NAMES.get(kind, str(kind)),
                **_summary(values).model_dump(),
            )
            for kind, values in sorted(by_kind.items())
        ],
        shared_effects=_summary(effects),
    )


# -- Checkpoints --------------------------------------------------------------


class CheckpointContribution(StrictModel):
    label: str
    run: str
    update: int
    reading: ModelContribution


class FactContributionReading(StrictModel):
    experiment: ReadExperiment
    position_seed: int
    positions: int
    checkpoints: list[CheckpointContribution]


def _update(checkpoint: Checkpoint) -> int:
    return int(checkpoint.clocks["updates"])


def read_checkpoints(
    state: ExperimentState, experiment: ReadExperiment, protocol: Protocol
) -> list[tuple[str, TrainingRun, Checkpoint]]:
    """The checkpoints the reading covers, labelled by size or vocabulary arm.

    Frontier runs: update 0, every reached milestone and the endpoint, once
    ended. Vocabulary arms: every export, update 0 included.
    """
    found: list[tuple[str, TrainingRun, Checkpoint]] = []
    if experiment == "tla_frontier":
        for label in SIZES:
            run = size_run(state, label)
            if run is None:
                continue
            wanted = {0, *protocol.milestones}
            last = endpoint(run)
            if last is not None:
                wanted.add(last)
            found += [
                (label, run, checkpoint)
                for checkpoint in run.checkpoints
                if _update(checkpoint) in wanted
            ]
    else:
        for arm in vocabulary_check.VOCABULARIES:
            run = vocabulary_check.arm_run(state, arm)
            if run is not None:
                found += [(arm, run, checkpoint) for checkpoint in run.checkpoints]
    return sorted(found, key=lambda item: (item[0], _update(item[2])))


def read(
    state: ExperimentState, experiment: ReadExperiment, cache: Path
) -> FactContributionReading:
    """Every covered checkpoint's reading on the fixed positions."""
    regimes, protocol = frozen()
    observation = regimes[SMALL].observation
    positions = sample_positions(protocol, observation)
    readings: list[CheckpointContribution] = []
    for label, run, checkpoint in read_checkpoints(state, experiment, protocol):
        reference = checkpoint.artifact
        path = fetch_etude_checkpoint(reference, cache / f"{reference.sha256}.pt")
        model, space = load_checkpoint(str(path))
        if space.encoder.config != observation:
            raise ValueError("checkpoint observation differs from frontier protocol")
        readings.append(
            CheckpointContribution(
                label=label,
                run=run.name,
                update=_update(checkpoint),
                reading=read_model(model, positions),
            )
        )
    return FactContributionReading(
        experiment=experiment,
        position_seed=POSITION_SEED,
        positions=len(positions["facts"]),
        checkpoints=readings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("tla_frontier", "tla_frontier_vocabulary"),
        required=True,
    )
    parser.add_argument("--source", required=True, help="Etude store")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = with_launcher_ends(observe(args.source, args.experiment))
    with TemporaryDirectory() as cache:
        reading = read(state, args.experiment, Path(cache))
    args.out.write_text(reading.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
