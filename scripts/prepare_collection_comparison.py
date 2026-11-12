"""Prepare the two pinned MTG-281 TrainingJobs, without submitting either.

Each output run contains its full source hash bundle. Compressed git archives
retain those source bytes independently of later checkout changes. A dedicated
OnDemand declaration binds startup, delivery and rental lifetime; it must be
registered and freshly quoted before spending. No existing supply is modified.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from etude.launch.records import STOCK_IMAGE, Job, MachineTemplate, OnDemand

from manabot.data import file_entry
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime
import managym
from scripts.collection_comparison import (
    DELIVERY_SECONDS,
    FINALIZATION_SECONDS,
    RECIPE,
    RECIPE_SHA256,
    RENTAL_SECONDS,
    SEED,
    SOURCES,
    STARTUP_SECONDS,
    TRAINING_SECONDS,
    UPDATES,
    Analysis,
    Preparation,
)


def _bound_rental(job: Job) -> Job:
    """Keep bootstrap, admission and training inside the original rental deadline."""
    assert job.setup is not None
    # The worker deadline is rental deadline minus delivery. Recover the
    # original startup cutoff from it, rather than granting another 900 s
    # after bootstrap. GNU date/timeout are present in the pinned Linux image.
    offset = RENTAL_SECONDS - DELIVERY_SECONDS - STARTUP_SECONDS
    setup = (
        "set -eu\n"
        f'left=$(( $(date -d "$ETUDE_DEADLINE" +%s) - {offset} - $(date +%s) ))\n'
        'test "$left" -gt 10\n'
        'timeout --signal=TERM --kill-after=10 "$((left - 10))s" sh -c '
        + shlex.quote(job.setup)
        + "\n"
    )
    # Job's own admission checks watchdog + finalization, before numerics.
    # This extra guard reserves numerics (300) and admission/slack (600),
    # without editing either pinned worker or silently shortening training.
    required = TRAINING_SECONDS + FINALIZATION_SECONDS + 900
    command = (
        "set -eu\n"
        f'test "$(( $(date -d "$ETUDE_DEADLINE" +%s) - $(date +%s) ))" -ge {required}\n'
        + job.command
    )
    return Job.model_validate(
        {**job.model_dump(), "setup": setup, "command": command, "stop_seconds": 30}
    )


def prepare(repo: Path, out: Path) -> Preparation:
    """Freeze original recipe bytes with only identity and stopping bounds edited."""
    out.mkdir(parents=True, exist_ok=False)
    bundles: dict[str, dict[str, str]] = {}
    for arm, revision in SOURCES.items():
        body = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{RECIPE}"]
        )
        if hashlib.sha256(body).hexdigest() != RECIPE_SHA256:
            raise ValueError(f"original recipe differs at {revision}")
        raw = json.loads(body)
        raw["id"] = f"collection-growth-281-{arm}"
        raw["wall_seconds"] = TRAINING_SECONDS
        raw["stages"][0]["updates"] = UPDATES
        raw["stages"][0]["execution"]["wall_seconds"] = TRAINING_SECONDS
        regime = validate_regime(TrainingRegime.model_validate(raw))
        protocol = json.loads(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repo),
                    "show",
                    f"{revision}:experiments/protocols/tla-frontier-c5fba4-v1.json",
                ]
            )
        )
        shapes = {
            key: list(shape)
            for key, shape in ObservationSpace(regime.observation).shapes.items()
        }
        if (
            shapes != protocol["observation_shapes"]
            or managym.world_identity() != protocol["runtime_world"]
        ):
            raise ValueError(
                "local native world or declared capacities differ from the frozen recipe"
            )
        job = TrainingJob(
            regime=regime,
            experiment="collection-growth-281",
            milestones=(100, 200, 300, 400),
            finalization_seconds=FINALIZATION_SECONDS,
            stop_reserve_seconds=600,
            validate_numerics=True,
        ).job(
            repo=repo,
            revision=revision,
            max_hours=RENTAL_SECONDS / 3600,
            max_usd=Decimal("7.20"),
        )
        spec = TrainingJob.model_validate_json(job.config or "")
        assert spec.source_bundle is not None
        bundles[arm] = spec.source_bundle.files
        job = _bound_rental(job)
        (out / f"{arm}.json").write_text(
            job.runs(SEED)[0].model_dump_json(indent=2) + "\n"
        )
        archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision])
        (out / f"{arm}-source.tar.gz").write_bytes(gzip.compress(archive, mtime=0))
    # Native/content, dependencies and the prepared worker must be identical.
    # Other differences are the intended MTG-273 repair and its docs/tests.
    protected = sorted(
        name
        for name in bundles["before"].keys() | bundles["after"].keys()
        if name.startswith(("managym/", "content/", "protocol/"))
        or name in {"uv.lock", "pyproject.toml", RECIPE, "manabot/experiment/job.py"}
    )
    if any(
        bundles["before"].get(name) != bundles["after"].get(name) for name in protected
    ):
        raise ValueError("comparison changes native/content/dependencies or the worker")
    for name in protected:
        if file_entry(repo / name).sha256 != bundles["before"][name]:
            raise ValueError(
                f"local native/recipe admission uses different source: {name}"
            )
    (out / "local-admission.json").write_text(
        json.dumps(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "world": managym.world_identity(),
                "observation_shapes": shapes,
                "extension": file_entry(Path(managym._managym.__file__)).model_dump(),
                "scope": "Local native world, deck source and capacity admission; CUDA numerics remains a per-arm remote gate.",
            },
            indent=2,
        )
        + "\n"
    )
    # Retain the pre-execution analysis itself as well as its thresholds.
    for name in (
        "collection_comparison.py",
        "prepare_collection_comparison.py",
        "report_collection_comparison.py",
        "measure_collection_growth.py",
        "report_collection_growth.py",
    ):
        (out / name).write_bytes((repo / "scripts" / name).read_bytes())
    prepared = Preparation(
        analysis=Analysis(),
        supply=OnDemand(
            name="mtg281-l4",
            template=MachineTemplate(
                provider="runpod",
                gpu="NVIDIA L4",
                vcpus=6,
                memory_gb=48,
                disk_gb=40,
                image=STOCK_IMAGE,
            ),
            max_machines=1,
            slots=1,
            max_usd_per_hour=Decimal("0.60"),
            startup_seconds=STARTUP_SECONDS,
            delivery_seconds=DELIVERY_SECONDS,
            setup_seconds=0,
            storage="r2",
        ),
        files={path.name: file_entry(path) for path in sorted(out.iterdir())},
        source_files_equal=protected,
    )
    (out / "preparation.json").write_text(prepared.model_dump_json(indent=2) + "\n")
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.out)
    print(f"Prepared both 500-update jobs in {args.out}; no runs submitted.")


if __name__ == "__main__":
    main()
