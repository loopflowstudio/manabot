"""Bounded reproduction of self-play metric index amplification (MTG-273).

Run with `uv run python scripts/diagnose_outcome_index.py`. Uses Etude's real
block/index writer on eight tiny synthetic updates, without games, training,
network access or resource pressure. Compare the retired scalar representation
to immutable row tables at the same history lengths. Timings include tracemalloc
overhead; they measure this fixture, not a CUDA training speedup.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import tracemalloc
from typing import Literal

from etude.run.block_file import block_key
from etude.train.block import Builder
from etude.train.manifest import write_manifest
from etude.train.run import write_block

from manabot.training.outcome_tables import OUTCOME_TABLE_METRIC, write_outcomes
from manabot.training.rollout.self_play import SelfPlayOutcome
from manabot.training.update_metrics import metric_row


@dataclass(frozen=True)
class Measurement:
    representation: Literal["scalar", "table"]
    blocks: int
    manifest_bytes: int
    refresh_peak_python_bytes: int
    refresh_seconds: float


def measure(
    root: Path, representation: Literal["scalar", "table"]
) -> list[Measurement]:
    """Refresh the current index with 1, 4 and 8 permanent blocks present."""
    writer = "abcdef012345"
    readings: list[Measurement] = []
    for update in range(1, 9):
        # Sixteen matchups, both positions, one win/loss/draw per row. Change
        # counts per update so table deduplication cannot explain the result.
        rows = [
            SelfPlayOutcome(
                deck=f"{i:064x}",
                opponent_deck=f"{i + 1:064x}",
                position=position,
                wins=update,
                losses=update,
                draws=update,
            )
            for i in range(16)
            for position in ("play", "draw")
        ]
        values = (
            metric_row({"self_play_outcomes": rows})
            if representation == "scalar"
            else {OUTCOME_TABLE_METRIC: write_outcomes(root, rows)}
        )
        builder = Builder(("updates",))
        builder.add(update, {"updates": update}, values)
        table = builder.table(writer, update)
        write_block(root, block_key(2, update, update, writer), table)
        write_manifest(root, writer)
        if update in (1, 4, 8):
            tracemalloc.start()
            try:
                started = time.perf_counter()
                write_manifest(root, writer, permanent=False)
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            readings.append(
                Measurement(
                    representation,
                    update,
                    (root / f"metrics/{writer}.permanent.json").stat().st_size,
                    peak,
                    elapsed,
                )
            )
    return readings


def main() -> None:
    with TemporaryDirectory(prefix="manabot-outcome-index-") as temporary:
        root = Path(temporary)
        readings = measure(root / "scalar", "scalar") + measure(root / "table", "table")
        print(json.dumps([asdict(reading) for reading in readings], indent=2))


if __name__ == "__main__":
    main()
