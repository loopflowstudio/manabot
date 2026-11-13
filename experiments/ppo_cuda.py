"""PPO CUDA update measurements for MTG-338 (report: experiments/ppo_cuda.md).

uv run etude launch experiments/ppo_cuda.py --revision 9268966e1 --paired --rent mtg338-l4
The harness lives in the launch revision's scratch/ (speed/, equivalence/), so
launch at 9268966e1. The default mode times five unprofiled baseline repeats
and an easier eight-stream workload; `diagnostic=True` adds a profile process.
`paired=True` runs the CUDA tests, the CPU equivalence harness, five
alternating kept/changed lookup pairs and a candidate profile.
"""

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job, Run

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup


def runs(
    revision: str, hours: float = 1, diagnostic: bool = False, paired: bool = False
) -> list[Run]:
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
    if paired:
        command = f"""{TOOLCHAIN_ENV}
set -eu
out="$ETUDE_OUTPUT_DIR"
export MANABOT_REQUIRE_CUDA=1
uv run --locked --extra dev --extra artifacts --extra notebook --extra reports pytest tests/model/test_semantic_cards.py tests/model/test_semantic_card_precision.py tests/model/test_bucketed_shapes.py tests/model/test_live_tokens.py tests/model/test_recent_events.py -q > "$out/tests.txt" 2>&1
for arm in baseline candidate; do
  LOOKUP_ARM=$arm uv run --locked python scratch/equivalence/run.py "$out/cpu-$arm.pt" > "$out/cpu-$arm.log" 2>&1
  LOOKUP_ARM=$arm uv run --locked python scratch/equivalence/updates.py "$out/updates-$arm.json" > "$out/updates-$arm.log" 2>&1
done
uv run --locked python scratch/equivalence/compare.py "$out/cpu-baseline.pt" "$out/cpu-candidate.pt" > "$out/cpu-comparison.txt"
cmp "$out/updates-baseline.json" "$out/updates-candidate.json"
uv run --locked python scratch/speed/paired.py "$out/paired"
uv run --locked python scratch/speed/bench.py "$out/candidate-profile.json" --ppo-only --repeats 1 --profile "$out/candidate-profile"
tar -czf "$out/comparison.tar.gz" -C "$out" tests.txt cpu-comparison.txt updates-baseline.json updates-candidate.json paired candidate-profile.json candidate-profile.ppo.txt
"""
    return Job(
        name="ppo-cuda-paired" if paired else "ppo-cuda-baseline",
        setup=machine_setup(
            " --extra dev --extra artifacts --extra notebook --extra reports"
            if paired
            else ""
        ),
        command=command,
        repo=str(Path(__file__).resolve().parents[1]),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal("4"),
        max_attempts=1,
    ).runs(0)
