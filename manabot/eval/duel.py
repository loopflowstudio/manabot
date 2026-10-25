"""Direct checkpoint comparisons through Etude and replay-verified simulation.

Each match contains all eight deck/seat legs of one deal."""

from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path

from etude.eval.match import EvalRun, Tables, agent
from etude.eval.records import (
    Benchmark,
    CheckpointKey,
    CheckpointRef,
    Environment,
    Match,
    Matchup,
    Model,
)
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
import numpy as np
from pydantic import BaseModel

from manabot.eval.benchmark import Conditions, evaluation_tables
from manabot.eval.endpoint import InferenceEndpoint, callback_allocation
from manabot.eval.evaluation import EvaluationRequest, evaluate as evaluate_panel
from manabot.eval.execution import bind_execution
from manabot.eval.monitor import MonitorResult
from manabot.eval.results import EvaluationRow
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.execution import EvalExecution
from manabot.sim.registration import PlayerRegistration
from manabot.training.records import TrainingCoordinates
from manabot.training.regime import ArtifactReference

EXPERIMENT = "two-seat-vs-one-seat-equal-updates"


class ScoreGroup(BaseModel):
    score: float
    interval: tuple[float, float]
    games: int
    wins: int
    draws: int
    losses: int


class DuelResult(BaseModel):
    etude: EvalRun
    groups: dict[str, ScoreGroup]
    updates: int
    two_seat: ArtifactReference
    one_seat: ArtifactReference
    two_seat_coordinates: TrainingCoordinates
    one_seat_coordinates: TrainingCoordinates


class Duel(BaseModel):
    two_seat: MonitorResult
    one_seat: MonitorResult

    def validate_inputs(self) -> None:
        for manifest in (self.two_seat, self.one_seat):
            verify_file(
                Path(manifest.artifact["path"]),
                manifest.artifact["sha256"],
                manifest.artifact["bytes"],
            )
            if (
                manifest.candidate.checkpoint_sha256 != manifest.artifact["sha256"]
                or manifest.candidate.checkpoint_bytes != manifest.artifact["bytes"]
            ):
                raise ValueError("registration differs from checkpoint receipt")
        if self.two_seat.coordinates.updates != self.one_seat.coordinates.updates:
            raise ValueError("head-to-head requires equal update counts")
        for name in (
            "world",
            "engine_source_sha256",
            "content_manifest_sha256",
            "observation_abi_sha256",
            "action_abi_sha256",
            "matchup_sha256",
        ):
            if (
                self.two_seat.evaluation_identities[name]
                != self.one_seat.evaluation_identities[name]
            ):
                raise ValueError(f"head-to-head runtime differs: {name}")
        if self.two_seat.candidate.player_spec != self.one_seat.candidate.player_spec:
            raise ValueError("head-to-head inference budgets differ")
        if (
            len(self.two_seat.protocol.deal_seeds) != 25
            or not self.two_seat.protocol.include_mirrors
        ):
            raise ValueError("head-to-head requires 25 deals and eight legs")


def registrations(duel: Duel) -> tuple[PlayerRegistration, PlayerRegistration]:
    return (
        duel.two_seat.candidate.model_copy(
            update={"player_id": "two-seat", "display_name": "Two-seat"}
        ),
        duel.one_seat.candidate.model_copy(
            update={"player_id": "one-seat", "display_name": "One-seat"}
        ),
    )


def play(match: Match, *, inference: InferenceEndpoint | None = None) -> Tables:
    duel = Duel.model_validate_json(Path(os.environ["MANABOT_DUEL"]).read_bytes())
    candidate, opponent = registrations(duel)
    seated = match.agents[0].artifact
    if seated is None or seated.sha256 != candidate.checkpoint_sha256:
        raise ValueError("Etude candidate differs from admitted checkpoint")
    conditions = Conditions.model_validate(match.conditions)
    out = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    result = evaluate_panel(
        EvaluationRequest(
            key=conditions.key,
            candidate=candidate,
            opponent=opponent,
            deal_seeds=(conditions.deal,),
            checkpoint_paths={
                candidate.player_id: duel.two_seat.artifact["path"],
                opponent.player_id: duel.one_seat.artifact["path"],
            },
            game_seconds=conditions.game_seconds,
            max_commands=conditions.max_commands,
            matchup_mode="all",
            purpose="repeated-development-comparison",
        ),
        out_dir=out,
        inference=inference,
    )
    if result.error:
        raise ValueError(result.error)
    return evaluation_tables(result.rows)


def summarize(directory: Path) -> dict[str, ScoreGroup]:
    rows = [
        EvaluationRow.model_validate(row)
        for path in sorted((directory / "arena").glob("*/rows.json"))
        for row in json.loads(path.read_bytes())
    ]
    if (
        len(rows) != 200
        or not all(row.valid for row in rows)
        or len({(r.deal_seed, r.leg) for r in rows}) != 200
    ):
        raise ValueError("expected 200 unique valid replayed games")
    labels = {
        0: "LvA",
        3: "LvA",
        1: "AvL",
        2: "AvL",
        4: "LvL",
        5: "LvL",
        6: "AvA",
        7: "AvA",
    }
    groups: dict[str, ScoreGroup] = {}
    for label in ("all", "LvA", "AvL", "LvL", "AvA"):
        chosen = [row for row in rows if label == "all" or labels[row.leg] == label]
        deals = sorted({row.deal_seed for row in chosen})
        values = np.array(
            [
                np.mean([row.score_a for row in chosen if row.deal_seed == seed])
                for seed in deals
            ]
        )
        rng = np.random.default_rng(0)
        boot = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(
            axis=1
        )
        bounds = np.quantile(boot, [0.025, 0.975])
        groups[label] = ScoreGroup(
            score=float(values.mean()),
            interval=(float(bounds[0]), float(bounds[1])),
            games=len(chosen),
            wins=sum(row.score_a == 1 for row in chosen),
            draws=sum(row.score_a == 0.5 for row in chosen),
            losses=sum(row.score_a == 0 for row in chosen),
        )
    return groups


def _checkpoint(result: MonitorResult) -> CheckpointRef:
    """A training run's export, keyed by its run and file, read from where it lies."""
    path = Path(result.artifact["path"]).resolve()
    return CheckpointRef(
        key=CheckpointKey(run=result.run_id, name=path.name),
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=result.artifact["sha256"],
            size_bytes=result.artifact["bytes"],
        ),
    )


def evaluate(path: Path, *, execution: EvalExecution | None = None) -> None:
    duel = Duel.model_validate_json(path.read_bytes())
    duel.validate_inputs()
    _, original_opponent = registrations(duel)
    # Bind the opponent cohort before changing its compute declaration: it is
    # part of the historical seed authority, just like the original envelope.
    key = duel.two_seat.key.model_copy(
        update={"anchor_cohort_sha256": original_opponent.identity_sha256}
    )
    if execution is not None:
        key, _ = bind_execution(key, (), execution)
        for manifest in (duel.two_seat, duel.one_seat):
            manifest.key, (manifest.candidate, manifest.opponent) = bind_execution(
                manifest.key, (manifest.candidate, manifest.opponent), execution
            )
        derived = path.parent / "duel-execution.json"
        atomic_json(derived, duel.model_dump(mode="json"))
        path = derived
    directory = path.parent
    candidate, opponent = registrations(duel)
    os.environ["MANABOT_DUEL"] = str(path.resolve())
    os.environ["ETUDE_OUTPUT_DIR"] = str(directory.resolve())
    os.environ["ETUDE_RUN"] = "duel-" + canonical_sha256(
        [candidate.identity_sha256, opponent.identity_sha256]
    )
    code = [
        Path(__file__),
        *sorted(Path("manabot/eval").glob("*.py")),
        *sorted(Path("manabot/model").rglob("*.py")),
        *sorted(Path("manabot/sim").glob("*.py")),
        *sorted(Path("manabot/env").glob("*.py")),
        *sorted(Path("manabot/infra").glob("*.py")),
        Path("manabot/eval/benchmark.py"),
    ]
    benchmark = Benchmark(
        name=EXPERIMENT,
        environment=Environment(
            name=f"{duel.two_seat.key.world}-direct-development",
            files=canonical_sha256(duel.two_seat.evaluation_identities),
            config={"purpose": "repeated-development-comparison"},
        ),
        matchups=(
            Matchup(
                name="two-seat vs one-seat",
                players=(
                    Model(
                        code=code,
                        config={"registration": candidate.model_dump(mode="json")},
                    ),
                    agent(
                        "one-seat",
                        checkpoint=_checkpoint(duel.one_seat),
                        code=code,
                        config={"registration": opponent.model_dump(mode="json")},
                    ),
                ),
                conditions=tuple(
                    Conditions(
                        deal=seed,
                        key=key,
                        game_seconds=600,
                        max_commands=10000,
                    ).model_dump(mode="json")
                    for seed in duel.two_seat.protocol.deal_seeds
                ),
                games=8,
            ),
        ),
    )
    with callback_allocation(execution, directory / "allocation") as inference:
        result = hold(
            benchmark,
            experiment=EXPERIMENT,
            play=partial(play, inference=inference) if inference else play,
            checkpoint=_checkpoint(duel.two_seat),
            directory=directory.parent.parent / "track",
            workers=execution.max_active_games if execution else 1,
        )

    atomic_json(
        directory / "result.json",
        DuelResult(
            etude=result,
            groups=summarize(directory),
            updates=duel.two_seat.coordinates.updates,
            two_seat=duel.two_seat.artifact,
            one_seat=duel.one_seat.artifact,
            two_seat_coordinates=duel.two_seat.coordinates,
            one_seat_coordinates=duel.one_seat.coordinates,
        ).model_dump(mode="json"),
    )
