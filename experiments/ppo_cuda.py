"""Exploratory PPO baseline and profile using the restored MTG-325 harness.

uv run etude launch experiments/ppo_cuda.py --revision COMMIT --rent mtg338-l4 --max-usd 4
The production-size batch has five unprofiled repeats and a separate profile;
eight streams exercise the harness on an easier real workload. No candidate
implementation is compared in this initial measurement.
"""

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job, Run

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup


def runs(revision: str) -> list[Run]:
    """One bounded L4 baseline; no retries or other tasks on this allocation."""
    command = f"""{TOOLCHAIN_ENV}
set -eu
out="$ETUDE_OUTPUT_DIR"
uv run --locked python scratch/speed/bench.py "$out/diagnostic.json" --ppo-only --repeats 1 --profile "$out/profile"
uv run --locked python scratch/speed/bench.py "$out/baseline.json" --ppo-only --repeats 5
uv run --locked python scratch/speed/bench.py "$out/easier.json" --ppo-only --streams 8 --transitions 8 --repeats 1
tar -czf "$out/measurement.tar.gz" -C "$out" baseline.json diagnostic.json easier.json profile.ppo profile.ppo.txt profile.ppo.trace.json
"""
    return Job(
        name="ppo-cuda-baseline",
        setup=machine_setup(""),
        command=command,
        repo=str(Path(__file__).resolve().parents[1]),
        revision=revision,
        max_hours=1,
        max_usd=Decimal("4"),
        max_attempts=1,
    ).runs(0)
