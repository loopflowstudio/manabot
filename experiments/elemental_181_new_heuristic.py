"""MTG-181 addendum: the three endpoints against the strengthened heuristic.

The frozen MTG-181 protocol scored every checkpoint against the v1 heuristic,
which the endpoints beat 0.84–0.86 of the time. Main has since replaced that
anchor with a much stronger one (639ab7770). This addendum plays each seed's
endpoint checkpoint through the same frozen 5,000-game grid
(`experiments.elemental_grid`, same schedule digest, deals and seats) against the
current `HeuristicPlayer`. It is reported beside, never instead of, the
protocol's v1 result, and selects nothing: the endpoints were fixed by the
protocol before this anchor existed.

Checkpoints come from the follower's verified byte cache; each worker re-checks
the SHA-256 before loading. Local CPU only.

    uv run python -m experiments.elemental_181_new_heuristic \
        --follow .runs/mtg181/follow --directory .runs/mtg181/new-heuristic \
        --out experiments/data/mtg181/new-heuristic
"""

import argparse
from pathlib import Path

from pydantic import BaseModel

from experiments.elemental_grid import SCHEDULE_SHA256, Summary, evaluate, summarize


class Endpoint(BaseModel):
    seed: int
    job: str
    updates: int
    sha256: str


ENDPOINTS = (
    Endpoint(
        seed=18101,
        job="mtg181-elemental-two-seat-18101-r2",
        updates=616,
        sha256="cc78fe1d44fcc65e93fe5673d8d78b6e44b85bf7433b5624a52a0be33410915e",
    ),
    Endpoint(
        seed=18102,
        job="mtg181-elemental-two-seat-18102",
        updates=474,
        sha256="1c203acb14e18bfbbf95facc368efcfa9588a3c6e17b37206dd83d0d769e13ee",
    ),
    Endpoint(
        seed=18103,
        job="mtg181-elemental-two-seat-18103",
        updates=568,
        sha256="feea8767b288916cbfdccf2296f54dd8d37dbc5c52cb083ac57d2a64db84e248",
    ),
)


class Result(BaseModel):
    endpoint: Endpoint
    schedule_sha256: str
    summary: Summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for point in ENDPOINTS:
        name = f"{point.seed}-{point.updates:06d}-{point.sha256[:12]}"
        rows = evaluate(
            args.follow.resolve() / "cache" / point.job / point.sha256,
            point.sha256,
            point.seed,
            args.directory.resolve() / name / "groups",
        )
        result = Result(
            endpoint=point, schedule_sha256=SCHEDULE_SHA256, summary=summarize(rows)
        )
        (args.out / f"{name}.json").write_text(result.model_dump_json(indent=2))
        overall = result.summary.overall
        print(f"{name}: {overall.score} [{overall.low}, {overall.high}]", flush=True)


if __name__ == "__main__":
    main()
