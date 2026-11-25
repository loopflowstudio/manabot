"""Exploratory optional Python-hook cost on real buffered vector games.

Run five rotating repetitions against an isolated archive of the kept source.
All streams play sixteen complete games, then close their untouched autoreset roots.
Output digests must agree before comparing elapsed and process CPU time.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


def cell(arm: str) -> dict[str, Any]:
    import numpy as np

    from manabot.env.observation import ObservationSpace
    import managym
    from managym.recording import RecordedVectorEnv
    from managym.statistics import Behaviour, CardPerformance, Reducer

    class Choices(Reducer[int]):
        def start(self, setup: Any) -> None:
            self.count = 0

        def decision(self, decision: Any) -> None:
            self.count += 1

        def end(self, result: Any, capture: Any) -> list[int]:
            return [self.count]

    count = 4
    questions = (
        [CardPerformance(), Behaviour(), Choices()] if arm == "edited-hooks" else []
    )
    vector = RecordedVectorEnv(
        num_envs=count, seed=7, **({"reducers": questions} if questions else {})
    )
    arrays = ObservationSpace().encoder.allocate(count)
    arrays.update(
        rewards=np.zeros(count),
        terminated=np.zeros(count, dtype=np.uint8),
        truncated=np.zeros(count, dtype=np.uint8),
    )
    vector.set_buffers(arrays)
    players = [managym.PlayerConfig(str(seat), {"Mountain": 12}) for seat in (0, 1)]
    completed = [0] * count
    decisions = 0
    receipts = []
    start = time.perf_counter()
    cpu_start = time.process_time()
    vector.reset_all_into_buffers(players)
    for _ in range(1000):
        active = [games < 16 for games in completed]
        before = vector.current_game_generations()
        vector.step_into_buffers([0] * count, active)
        decisions += sum(active)
        after = vector.current_game_generations()
        infos = vector.get_last_info()
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if old != new:
                completed[index] += 1
                receipts.append(
                    (index, completed[index], infos[index].get("winner_index"))
                )
        if min(completed) == 16:
            break
    else:
        raise RuntimeError("game cap exhausted")
    vector.close()
    tables = vector.rows() if questions else None
    cpu_seconds = time.process_time() - cpu_start
    seconds = time.perf_counter() - start
    if questions:
        if vector.analysis_error is not None or sum(tables[2]) != decisions:
            raise RuntimeError("incomplete analysis")
    digests = vector.native.state_digests()
    gameplay = json.dumps((receipts, digests), sort_keys=True).encode()
    return dict(
        arm=arm,
        games=sum(completed),
        decisions=decisions,
        errors=0,
        cpu_seconds=cpu_seconds,
        seconds=seconds,
        gameplay_sha256=hashlib.sha256(gameplay).hexdigest(),
    )


def run(before: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    root = script.parents[1]
    arms = ("before-off", "edited-off", "edited-hooks")
    rows = []
    for repeat in range(5):
        offset = repeat % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            source = before if arm == "before-off" else root
            result = subprocess.run(
                [sys.executable, str(script), "cell", arm],
                cwd=source,
                env={**os.environ, "PYTHONPATH": str(source)},
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            row = json.loads(result.stdout)
            row["repeat"] = repeat
            rows.append(row)
            print(json.dumps(row), flush=True)
            (output / "measurements.json").write_text(json.dumps(rows, indent=2))
    if len({(r["games"], r["decisions"], r["gameplay_sha256"]) for r in rows}) != 1:
        raise RuntimeError("behavior differs")
    summary = {}
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            key: {
                "median": statistics.median(r[key] for r in group),
                "range": [min(r[key] for r in group), max(r[key] for r in group)],
            }
            for key in ("cpu_seconds", "seconds")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("cell")
    one.add_argument("arm")
    all_cells = sub.add_parser("run")
    all_cells.add_argument("--before", required=True, type=Path)
    all_cells.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "cell":
        print(json.dumps(cell(args.arm)))
    else:
        run(args.before.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
