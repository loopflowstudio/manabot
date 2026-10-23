"""MTG-181: two-seat self-play on the ten-deck Elemental world, three seeds for $10.

Jack Heart's request (2026-10-22/23): a basic trial of whether the two-seat
recipe learns on all ten Elemental decks. The protocol is
experiments/protocols/elemental-181.md; the recipe is
experiments/regimes/elemental-two-seat-181-v1.json, MTG-174's frozen recipe
with both seats learning.

This module freezes the allocation and evaluation constants. Jobs are compiled
and submitted with the existing `manabot deploy compile` / `deploy submit`
lifecycle, one independent L4 job per seed; `experiments.elemental_two_seat_follow`
evaluates their exports off the rental with `experiments.elemental_grid`.
"""

from pathlib import Path

from manabot.remote.plan import JobSpec, Machine

ROOT = Path(__file__).resolve().parents[1]
REGIME = ROOT / "experiments/regimes/elemental-two-seat-181-v1.json"
SPEC = ROOT / "experiments/protocols/elemental-181-job.json"

SEEDS = (18101, 18102, 18103)
JOBS = {seed: f"mtg181-elemental-two-seat-{seed}" for seed in SEEDS}
# Seed 18101's first job never rented: its guardian-probe create returned
# RunPod HTTP 500 and stays an unresolved ambiguous claim (protocol amendment).
FAILED_JOBS = {18101: "mtg181-elemental-two-seat-18101"}
JOBS[18101] = "mtg181-elemental-two-seat-18101-r2"
# Raw exports and grid evaluations every 200 absolute updates, plus the
# initialization and the actual endpoint.
CADENCE = 200
# Inclusive authorization across every seed, guardian proof and failure.
CAP_DOLLARS = 10.0

L4 = Machine(
    gpu_types=("NVIDIA L4",),
    vcpus=6,
    memory_gb=48,
    image="runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853",
    hourly_ceiling=0.60,
)
# Five hours at $0.62/hour (rental ceiling plus storage) projects $3.10 per seed;
# three seeds project $9.30 before guardian proofs.
JOB_SPEC = JobSpec(
    machine=L4,
    lifetime_hours=5.0,
    spending_limit=3.15,
    setup_seconds=900,
    checkpoint_seconds=120,
    upload_seconds=300,
    cleanup_seconds=120,
)


def main() -> None:
    SPEC.write_text(JOB_SPEC.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
