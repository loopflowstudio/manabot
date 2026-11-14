"""Bounded reproduction of self-play metric index amplification (MTG-273).

Run with `uv run python scripts/diagnose_outcome_index.py`. Uses Etude's real
block/index writer on eight tiny synthetic updates, without games, training,
network access or resource pressure. Compare the retired scalar representation
to Etude row tables, whose metric value is one content digest, at the same
history lengths. Since etude-sdk 0.2.16 a recent-index refresh no longer
redescribes permanent blocks, so both representations should stay flat; the
permanent manifest still grows with every metric name. Timings include
tracemalloc overhead; they measure this fixture, not a CUDA training speedup.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import tracemalloc
from typing import Literal

from etude.run.block_file import block_key
from etude.train.block import Builder
from etude.train.manifest import Index
from etude.train.records import RowTable
from etude.train.run import write_block

from manabot.training.rollout.self_play import SelfPlayOutcome
from manabot.training.update_metrics import OUTCOME_TABLE, metric_row


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
    root.mkdir(parents=True)
    index = Index(root, writer)
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
            # `Run.log_table` indexes only the digest of the table it stores.
            else {OUTCOME_TABLE: _digest(rows)}
        )
        builder = Builder(("updates",))
        builder.add(update, {"updates": update}, values)
        table = builder.table(writer, update)
        key = block_key(2, update, update, writer)
        write_block(root, key, table)
        index.add(key)
        index.write()
        if update in (1, 4, 8):
            tracemalloc.start()
            try:
                started = time.perf_counter()
                index.write()
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


def _digest(rows: list[SelfPlayOutcome]) -> str:
    body = RowTable.model_validate({"rows": rows}).model_dump_json().encode()
    return sha256(body).hexdigest()


def main() -> None:
    with TemporaryDirectory(prefix="manabot-outcome-index-") as temporary:
        root = Path(temporary)
        readings = measure(root / "scalar", "scalar") + measure(root / "table", "table")
        print(json.dumps([asdict(reading) for reading in readings], indent=2))


if __name__ == "__main__":
    main()
