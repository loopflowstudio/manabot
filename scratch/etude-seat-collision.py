"""Reproduce the retained Elemental seat collision across SDK releases.

Write with 0.2.5, then read with 0.2.7, using the same temporary directory.
Only two synthetic scalar rows are played; no game, training or remote work.
"""

from pathlib import Path
import sys

from etude.eval.match import Tables, evaluate
from etude.eval.query import statsdb
from etude.eval.records import Environment, Match


def old_rows(match: Match) -> Tables:
    # Elemental's game_rows: Etude participant 0 wins from native seat 1.
    return {
        "games": [
            {"player": 0, "seat": 1, "score": 1.0},
            {"player": 1, "seat": 0, "score": 0.0},
        ]
    }


def main() -> None:
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "write":
        from etude.eval.match import agent

        evaluate(
            experiment="elemental-seat-collision",
            environment=Environment(name="fixture", files="a" * 64),
            matches=[Match(agents=(agent("challenger", code=["manabot/eval/benchmark.py"]), agent("heuristic", code=["manabot/eval/benchmark.py"])))],
            play=old_rows,
            run="retained",
            directory=directory,
            workers=1,
        )
        query = (
            "select a.name, g.score from games g join seats s "
            "using (run, match, player) join agents a using (agent) order by a.name"
        )
    else:
        query = (
            "select p.name, g.score from games g join seats s "
            "using (run, match, seat) join players p on p.player = s.player order by p.name"
        )
    with statsdb("elemental-seat-collision", directory) as db:
        print(mode, db.execute(query).fetchall())


if __name__ == "__main__":
    main()
