"""Exploratory cost of all-game evaluation statistics and shared verification.

Fresh processes run a real two-seat code-bot deal, sequentially, rotating arms.
The kept and edited sources share the interpreter and dependencies. No induced
contention, policy sampling during replay, or GPU-inference claim is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from managym.statistics import Reducer

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "de8782340728027c164a77c870ad75e533c10982"
MAIN = "5af8490a1c9bd5e9e7bda1d789e04a6f6de259fb"
ARMS = (
    "main-cards",
    "before-off",
    "before-custom",
    "edited-off",
    "edited-cards",
    "edited-custom",
)


class Measurement(BaseModel):
    repeat: int
    arm: str
    revision: str
    games: int
    decisions: int
    seconds: float
    replay_seconds: float
    offline_seconds: float
    output_bytes: int
    replay_rss_bytes: int | None
    gameplay_sha256: str
    cards_sha256: str | None
    custom_sha256: str | None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _reducers() -> dict[str, Reducer[Any]]:
    # Lazy imports let the same harness execute the kept main revision, which
    # predates the reducer API. Script-local classes exercise the child boundary.
    from managym import ActionEnum
    from managym.card_capture import CapturedCards
    from managym.recording import PlayerSetup
    from managym.replay import RecordedDecision, ReplayStop
    from managym.statistics import Behaviour, Reducer

    class Sequencing(Reducer[dict[str, int]]):
        def start(self, setup: list[PlayerSetup]) -> None:
            self.choices = [[0, 0], [0, 0]]

        def decision(self, decision: RecordedDecision) -> None:
            actions = decision.observation.action_space.actions
            if any(
                a.action_type == ActionEnum.PRIORITY_PLAY_LAND for a in actions
            ) and any(a.action_type == ActionEnum.PRIORITY_CAST_SPELL for a in actions):
                row = self.choices[decision.observation.agent.player_index]
                row[0] += 1
                row[1] += (
                    actions[decision.chosen].action_type
                    == ActionEnum.PRIORITY_CAST_SPELL
                )

        def end(
            self, result: ReplayStop, capture: CapturedCards
        ) -> list[dict[str, int]]:
            return [
                {"seat": seat, "opportunities": row[0], "spells": row[1]}
                for seat, row in enumerate(self.choices)
            ]

        def finish(self, rows: list[dict[str, int]]) -> list[dict[str, int]]:
            return [
                {
                    "seat": seat,
                    **{
                        key: sum(row[key] for row in rows if row["seat"] == seat)
                        for key in ("opportunities", "spells")
                    },
                }
                for seat in (0, 1)
            ]

    class Draws(Reducer[dict[str, int]]):
        def end(
            self, result: ReplayStop, capture: CapturedCards
        ) -> list[dict[str, int]]:
            return [
                {
                    "seat": seat,
                    "draws": sum(
                        move.cause == "draw" and move.card.owner == seat
                        for move in capture.capture.moves
                    ),
                }
                for seat in (0, 1)
            ]

        def finish(self, rows: list[dict[str, int]]) -> list[dict[str, int]]:
            return [
                {
                    "seat": seat,
                    "draws": sum(row["draws"] for row in rows if row["seat"] == seat),
                }
                for seat in (0, 1)
            ]

    return {"behaviour": Behaviour(), "sequencing": Sequencing(), "draws": Draws()}


def cell(arm: str, output: Path, repeat: int) -> None:
    from pydantic import TypeAdapter

    from manabot.eval.card_metrics import reduce_cards
    from manabot.eval.match import play_cell
    from manabot.infra.source import source_commit
    from manabot.sim.card_evidence import EvidenceRef, read_evidence
    from manabot.sim.replay import read_trace
    from managym.recording import atomic_private_json
    from tests.eval.arena_fixture import arena_fixture

    expected = Path(os.environ["MANABOT_COST_SOURCE"]).resolve()
    if not Path(play_cell.__code__.co_filename).resolve().is_relative_to(expected):
        raise RuntimeError("evaluation imported the wrong revision")
    fixture = arena_fixture()
    evaluation = output / "evaluation"
    evidence_dir = (
        evaluation / "card-evidence"
        if arm not in {"before-off", "edited-off"}
        else None
    )
    questions = _reducers() if arm in {"before-custom", "edited-custom"} else {}
    arguments: dict[str, object] = {}
    if arm == "edited-custom":
        arguments["reducers"] = questions
    started = time.perf_counter()
    rows, trace, replay = play_cell(
        key=fixture.key,
        player_a=fixture.anchors[0],
        player_b=fixture.anchors[1],
        deal_seeds=(3, 4),
        legs=(0, 1),
        out_dir=evaluation,
        card_evidence_dir=evidence_dir,
        **arguments,
    )
    if not replay["passed"] or len(rows) != 4 or any(row["failure"] for row in rows):
        raise RuntimeError(f"evaluation failed: {replay}")
    card_rows: list[object] = []
    if evidence_dir is not None:
        for raw in trace["card_evidence"]:
            reference = EvidenceRef.model_validate(raw)
            evidence = read_evidence(
                (evidence_dir / reference.path).read_bytes(), reference
            )
            card_rows.extend(
                row.model_dump(mode="json")
                for row in reduce_cards(evidence, evidence.seat_players)
            )
        atomic_private_json(evaluation / "card-counts.json", {"rows": card_rows})
    offline_seconds = 0.0
    tables = None
    if arm == "before-custom":
        from managym.recording import read_records
        from managym.statistics import reduce

        offline_start = time.perf_counter()
        result = reduce(read_records(Path(trace["path"])), list(questions.values()))
        tables = TypeAdapter(dict[str, list[Any]]).dump_python(
            dict(zip(questions, result, strict=True)), mode="json"
        )
        atomic_private_json(evaluation / "statistics.json", {"tables": tables})
        offline_seconds = time.perf_counter() - offline_start
    elif arm == "edited-custom":
        tables = json.loads(Path(trace["statistics_path"]).read_text())["tables"]
        if tables is None:
            raise RuntimeError("evaluation statistics failed")
    elapsed = time.perf_counter() - started
    games = read_trace(Path(trace["path"]))
    gameplay = [
        {
            "seed": game["deal_seed"],
            "leg": game["leg"],
            "winner": game["winner"],
            "decisions": [
                {
                    key: decision[key]
                    for key in (
                        "actor",
                        "chosen_offer",
                        "pre_state_digest",
                        "post_state_digest",
                    )
                }
                for decision in game["decisions"]
            ],
        }
        for game in games
    ]
    perf = json.loads((evaluation / "performance.json").read_text())
    measurement = Measurement(
        repeat=repeat,
        arm=arm,
        revision=source_commit(),
        games=len(games),
        decisions=sum(len(game["decisions"]) for game in games),
        seconds=elapsed,
        replay_seconds=replay["seconds"],
        offline_seconds=offline_seconds,
        output_bytes=sum(
            path.stat().st_size for path in evaluation.rglob("*") if path.is_file()
        ),
        replay_rss_bytes=perf.get("counters", {}).get("rss_bytes/replay"),
        gameplay_sha256=_digest(gameplay),
        cards_sha256=_digest(card_rows) if evidence_dir else None,
        custom_sha256=_digest(tables) if tables is not None else None,
    )
    (output / "measurement.json").write_text(measurement.model_dump_json(indent=2))
    print(measurement.model_dump_json(), flush=True)


def run(before: Path, main_source: Path, output: Path, repeats: int = 5) -> None:
    output.mkdir(parents=True, exist_ok=False)
    observations: list[Measurement] = []
    for repeat in range(repeats):
        rotation = repeat % len(ARMS)
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            source = (
                main_source
                if arm == "main-cards"
                else before
                if arm.startswith("before-")
                else ROOT
            )
            destination = output / f"repeat-{repeat}" / arm
            destination.mkdir(parents=True)
            print(f"START {repeat} {arm}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "cell",
                    arm,
                    str(destination),
                    str(repeat),
                ],
                cwd=source,
                env={
                    **os.environ,
                    "PYTHONPATH": str(source),
                    "MANABOT_COST_SOURCE": str(source),
                },
                check=True,
                timeout=240,
            )
            observations.append(
                Measurement.model_validate_json(
                    (destination / "measurement.json").read_text()
                )
            )
            (output / "measurements.json").write_text(
                json.dumps([row.model_dump() for row in observations], indent=2)
            )
        group = observations[-len(ARMS) :]
        if len({row.gameplay_sha256 for row in group}) != 1:
            raise RuntimeError("gameplay differs across arms; comparison is invalid")
        if (
            len({row.cards_sha256 for row in group if row.cards_sha256 is not None})
            != 1
        ):
            raise RuntimeError("card counts differ across arms; comparison is invalid")
        if (
            len({row.custom_sha256 for row in group if row.custom_sha256 is not None})
            != 1
        ):
            raise RuntimeError("custom statistics differ; comparison is invalid")
    summary = {}
    for arm in ARMS:
        selected = [row for row in observations if row.arm == arm]
        ratios = [
            row.seconds
            / next(
                base.seconds
                for base in observations
                if base.repeat == row.repeat and base.arm == "before-off"
            )
            for row in selected
        ]
        summary[arm] = {
            "seconds_median": statistics.median(row.seconds for row in selected),
            "seconds_range": [
                min(row.seconds for row in selected),
                max(row.seconds for row in selected),
            ],
            "games_per_second_median": statistics.median(
                row.games / row.seconds for row in selected
            ),
            "time_ratio_vs_before_off_median": statistics.median(ratios),
            "time_ratio_vs_before_off_range": [min(ratios), max(ratios)],
            "output_bytes_per_game": statistics.median(
                row.output_bytes / row.games for row in selected
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("cell")
    one.add_argument("arm", choices=ARMS)
    one.add_argument("output", type=Path)
    one.add_argument("repeat", type=int)
    all_cells = sub.add_parser("run")
    all_cells.add_argument("--before", type=Path, required=True)
    all_cells.add_argument("--main", type=Path, required=True)
    all_cells.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cell":
        cell(args.arm, args.output, args.repeat)
    else:
        run(args.before.resolve(), args.main.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
