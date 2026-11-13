"""Exploratory PPO baseline and profile using the restored MTG-325 harness.

uv run etude launch experiments/ppo_cuda.py --revision COMMIT --rent mtg338-l4 --max-usd 4
The production-size batch has five unprofiled repeats; eight streams exercise
the harness on an easier real workload. `diagnostic=True` adds a separate
profile process before timing. No candidate implementation is compared here.
"""

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job, Run

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup


def runs(revision: str, hours: float = 1, diagnostic: bool = False) -> list[Run]:
    """One bounded L4 baseline; no retries or other tasks on this allocation."""
    profile = (
        'uv run --locked python scratch/speed/bench.py "$out/diagnostic.json" '
        '--ppo-only --repeats 1 --profile "$out/profile"\n'
        if diagnostic
        else ""
    )
    evidence = (
        " diagnostic.json profile.ppo profile.ppo.txt profile.ppo.trace.json"
        if diagnostic
        else ""
    )
    command = f"""{TOOLCHAIN_ENV}
set -eu
out="$ETUDE_OUTPUT_DIR"
{profile}
uv run --locked python scratch/speed/bench.py "$out/baseline.json" --ppo-only --repeats 5
uv run --locked python scratch/speed/bench.py "$out/easier.json" --ppo-only --streams 8 --transitions 8 --repeats 1
tar -czf "$out/measurement.tar.gz" -C "$out" baseline.json easier.json{evidence}
"""
    return Job(
        name="ppo-cuda-baseline",
        setup=machine_setup(""),
        command=command,
        repo=str(Path(__file__).resolve().parents[1]),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal("4"),
        max_attempts=1,
    ).runs(0)
