"""Direct checkpoint comparisons through Etude and replay-verified simulation.

Each match contains all eight deck/seat legs of one deal. Checkpoint following
and process lifetime management belong to experiment.duel_follow."""

from __future__ import annotations

import json
import os
from pathlib import Path

from etude.eval.match import EvalRun, Tables, agent
from etude.eval.records import Benchmark, Environment, Match, Matchup, Model
from etude.eval.source import digest_files
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
import numpy as np
from pydantic import BaseModel

from manabot.eval.benchmark import Conditions, evaluation_tables
from manabot.eval.match import play_cell
from manabot.eval.monitor import EvaluationRow, MonitorResult
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.registration import PlayerRegistration
from manabot.training.records import TrainingCoordinates
from manabot.training.regime import ArtifactReference


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


def play(match: Match) -> Tables:
    duel = Duel.model_validate_json(Path(os.environ["MANABOT_DUEL"]).read_bytes())
    candidate, opponent = registrations(duel)
    if match.agents[0].artifact != candidate.checkpoint_sha256:
        raise ValueError("Etude candidate differs from admitted checkpoint")
    conditions = Conditions.model_validate(match.conditions)
    out = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    out.mkdir(parents=True, exist_ok=True)
    rows, trace, replay = play_cell(
        key=conditions.key,
        player_a=candidate,
        player_b=opponent,
        deal_seeds=(conditions.deal,),
        out_dir=out,
        checkpoint_paths={
            candidate.player_id: duel.two_seat.artifact["path"],
            opponent.player_id: duel.one_seat.artifact["path"],
        },
        game_seconds=conditions.game_seconds,
        max_commands=conditions.max_commands,
        matchup_mode="all",
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        atomic_json(out / f"{name}.json", value)
    return evaluation_tables([EvaluationRow.model_validate(row) for row in rows])


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


def evaluate(path: Path) -> None:
    duel = Duel.model_validate_json(path.read_bytes())
    duel.validate_inputs()
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
        name="two-seat-vs-one-seat-equal-updates",
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
                        code=digest_files(code),
                        config={"registration": candidate.model_dump(mode="json")},
                    ),
                    agent(
                        "one-seat",
                        checkpoint=ArtifactRef(
                            uri=Path(duel.one_seat.artifact["path"]).resolve().as_uri(),
                            sha256=duel.one_seat.artifact["sha256"],
                            size_bytes=duel.one_seat.artifact["bytes"],
                        ),
                        code=code,
                        config={"registration": opponent.model_dump(mode="json")},
                    ),
                ),
                conditions=tuple(
                    Conditions(
                        deal=seed,
                        key=duel.two_seat.key.model_copy(
                            update={
                                "anchor_cohort_sha256": canonical_sha256(
                                    opponent.model_dump()
                                )
                            }
                        ),
                        game_seconds=600,
                        max_commands=10000,
                    ).model_dump(mode="json")
                    for seed in duel.two_seat.protocol.deal_seeds
                ),
                games=8,
            ),
        ),
    )
    result = hold(
        benchmark,
        play=play,
        checkpoint=ArtifactRef(
            uri=Path(duel.two_seat.artifact["path"]).resolve().as_uri(),
            sha256=duel.two_seat.artifact["sha256"],
            size_bytes=duel.two_seat.artifact["bytes"],
        ),
        directory=directory.parent.parent / "track",
        workers=1,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duel", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.duel)
