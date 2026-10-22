"""Direct checkpoint comparisons through Etude and the replay-checked arena.

Each match contains all eight deck/seat legs of one deal. The follower consumes
already admitted live exports, never rents hardware, and retains failed attempts
without retry. Its fixed deadline and one-hour child limit bound CPU work.
"""

from __future__ import annotations

import fcntl
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from etude.eval.match import EvalRun, Tables, agent
from etude.eval.records import Benchmark, Environment, Match, Matchup, Model
from etude.eval.source import digest_files
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
import numpy as np
from pydantic import BaseModel, TypeAdapter

from manabot.arena.match import play_cell
from manabot.arena.models import PlayerRegistration, canonical_sha256
from manabot.infra.artifacts import verify_file
from manabot.remote.live_evaluation import Ledger
from manabot.remote.supervisor import stop_process
from manabot.training.etude_evaluation import Conditions, arena_tables
from manabot.training.execution import atomic_json
from manabot.training.models import ArtifactReference, TrainingCoordinates
from manabot.training.monitor_evaluation import ArenaRow, MonitorResult


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


class DirectAttempt(BaseModel):
    phase: Literal["running", "completed", "failed"]
    started_at: float
    two_seat: str
    one_seat: str
    pid: int | None = None
    exit_code: int | None = None
    finished_at: float | None = None


_ATTEMPTS = TypeAdapter(dict[str, DirectAttempt])


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
    return arena_tables([ArenaRow.model_validate(row) for row in rows])


def summarize(directory: Path) -> dict[str, ScoreGroup]:
    rows = [
        ArenaRow.model_validate(row)
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
        *sorted(Path("manabot/arena").glob("*.py")),
        *sorted(Path("manabot/model").rglob("*.py")),
        *sorted(Path("manabot/sim").glob("*.py")),
        *sorted(Path("manabot/env").glob("*.py")),
        *sorted(Path("manabot/infra").glob("*.py")),
        Path("manabot/training/etude_evaluation.py"),
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


def report(directory: Path, source: Path, attempts: dict[str, DirectAttempt]) -> None:
    public = source / "public"
    atomic_json(
        public / "head-to-head.json", _ATTEMPTS.dump_python(attempts, mode="json")
    )
    rows = ""
    for updates, item in sorted(attempts.items(), key=lambda pair: int(pair[0])):
        result_path = directory / "evaluations" / updates / "result.json"
        if result_path.exists():
            result = DuelResult.model_validate_json(result_path.read_bytes())
            rows += f'<tr><td colspan="5">{updates} updates: two-seat {result.two_seat_coordinates.training_seconds / 3600:.2f} training hours / {result.two_seat_coordinates.learner_transitions:,} learner decisions; one-seat {result.one_seat_coordinates.training_seconds / 3600:.2f} hours / {result.one_seat_coordinates.learner_transitions:,} decisions.</td></tr>'
            for label, group in result.groups.items():
                low, high = group.interval
                rows += f"<tr><td>{updates}</td><td>{label}</td><td>{group.score:.1%}</td><td>[{low:.1%}, {high:.1%}]</td><td>{group.wins}/{group.draws}/{group.losses}</td></tr>"
        else:
            rows += (
                f'<tr><td>{updates}</td><td colspan="4">{escape(str(item))}</td></tr>'
            )
    if not rows:
        rows = '<tr><td colspan="5">Waiting for the first completed 1,000-update admission; direct evaluations follow the heuristic evaluation.</td></tr>'
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Direct two-seat vs one-seat</title><style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem}</style><h1>Two-seat vs one-seat, equal updates</h1><p>200 games per checkpoint: 25 paired deals × eight deck/seat legs. 50 games per deck pairing, balanced play/draw. Score is from the two-seat model’s perspective. Win/draw/loss counts shown. 95% intervals resample whole deals (4000 draws, seed 0).</p><p>Repeated exploratory monitoring, one training seed, unequal training cost/data. Repeated looks and subgroup intervals do not establish method superiority. No automatic training or budget changes.</p><p><a href="/">Heuristic monitoring</a> · <a href="head-to-head.json">Attempt ledger</a></p><table><tr><th>Updates</th><th>Deck pairing</th><th>Two-seat score</th><th>95% interval</th><th>W/D/L</th></tr>'
        + rows
        + f"</table><p>Observer refreshed {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}. Above 50% favors two-seat; below 50% favors one-seat. Exact controls are retained through 12,000 updates.</p>"
    )
    temporary = public / "head-to-head.html.tmp"
    temporary.write_text(page)
    temporary.replace(public / "head-to-head.html")


def follow(source: Path, controls: Path, directory: Path, deadline: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = directory / "ledger.json"
        attempts = (
            _ATTEMPTS.validate_json(ledger_path.read_bytes())
            if ledger_path.exists()
            else {}
        )
        if any(item.phase == "running" for item in attempts.values()):
            raise RuntimeError(
                "prior direct evaluator requires explicit reconciliation"
            )
        while time.time() < deadline:
            ledger = Ledger.model_validate_json((source / "ledger.json").read_bytes())
            for item in ledger.attempts.values():
                updates = item.updates
                control_path = controls / f"{updates}.json"
                if (
                    updates <= 0
                    or updates % 1000
                    or str(updates) in attempts
                    or item.phase != "completed"
                    or not control_path.exists()
                ):
                    continue
                matches = list(
                    (source / "evaluations").glob(
                        f"{item.digest}-attempt-*/admission.json"
                    )
                )
                admission = MonitorResult.model_validate_json(
                    sorted(matches)[-1].read_bytes()
                )
                control = MonitorResult.model_validate_json(control_path.read_bytes())
                duel = Duel(two_seat=admission, one_seat=control)
                duel.validate_inputs()
                target = directory / "evaluations" / str(updates)
                target.mkdir(parents=True, exist_ok=True)
                atomic_json(target / "duel.json", duel.model_dump(mode="json"))
                entry = DirectAttempt(
                    phase="running",
                    started_at=time.time(),
                    two_seat=admission.artifact["sha256"],
                    one_seat=control.artifact["sha256"],
                )
                attempts[str(updates)] = entry
                atomic_json(ledger_path, _ATTEMPTS.dump_python(attempts, mode="json"))
                with (target / "console.log").open("ab") as console:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "manabot.training.head_to_head",
                            "--duel",
                            str(target / "duel.json"),
                        ],
                        stdout=console,
                        stderr=console,
                        start_new_session=True,
                    )
                    entry.pid = child.pid
                    atomic_json(
                        ledger_path, _ATTEMPTS.dump_python(attempts, mode="json")
                    )
                    end = min(deadline, time.time() + 3600)
                    try:
                        while child.poll() is None and time.time() < end:
                            report(directory, source, attempts)
                            try:
                                child.wait(timeout=min(30, max(0.1, end - time.time())))
                            except subprocess.TimeoutExpired:
                                pass
                    finally:
                        if child.poll() is None:
                            stop_process(child)
                        entry.phase = (
                            "completed"
                            if child.wait() == 0 and (target / "result.json").exists()
                            else "failed"
                        )
                        entry.exit_code = child.returncode
                        entry.finished_at = time.time()
                        atomic_json(
                            ledger_path, _ATTEMPTS.dump_python(attempts, mode="json")
                        )
            report(directory, source, attempts)
            if ledger.cohort_phase not in ("running", "pending") and not any(
                item.phase == "running" for item in ledger.attempts.values()
            ):
                break
            time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duel", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.duel:
        evaluate(args.duel)
    elif args.source and args.controls and args.directory and args.deadline:
        follow(args.source, args.controls, args.directory, args.deadline)
    else:
        parser.error("provide --duel or all follower arguments")
