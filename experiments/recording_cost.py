"""Exploratory CUDA comparison of kept and edited Elemental training.

One job alternates the two revisions through five pairs on one host. A separate
option comparison rotates seven arms. Each uses the same seed and four updates;
measurements include the first update and all capture writes inside collection.
No artificial load, concurrent training or playing-strength claim is involved.
The driver retains receipts and streams cell progress to its launcher log.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig
import tarfile
from typing import TYPE_CHECKING
import zipfile

from etude.launch.records import Input, Job, Run
from pydantic import BaseModel
import torch

from experiments.elemental_variants import elemental_recipe, self_play_stage
from manabot.backends import resolve
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.training.execution import execute_regime
from manabot.verify.store import VerifyStore
import managym

if TYPE_CHECKING:
    from manabot.infra.run_settings import RunSettings

ROOT = Path(__file__).resolve().parents[1]
SEED = 80161
UPDATES = 4
REPEATS = 5
EXPERIMENT = "recording_cost"
BASELINE = "5af8490a1c9bd5e9e7bda1d789e04a6f6de259fb"
BASELINE_ARCHIVE_SHA256 = (
    "2d3268ad0bb109860d8a27f70893cc64ec876ddf65bc46169f14d757b4d394b2"
)
OPTION_ARMS = (
    "off",
    "statistics",
    "record-01",
    "record-10",
    "record-all",
    "journals",
    "marker",
)


class Probe(BaseModel):
    source_bundle: SourceBundle
    baseline_bundle: SourceBundle
    comparison: str = "revisions"


def _settings(arm: str) -> RunSettings:
    # Imported only for edited cells: the unchanged baseline predates this API.
    from manabot.infra.run_settings import RunSettings

    if arm == "marker":
        return RunSettings()
    if arm == "journals":
        return RunSettings(diagnostics="detailed")
    return RunSettings(
        diagnostics="off",
        statistics=arm == "statistics",
        record_games={"record-01": 0.01, "record-10": 0.1, "record-all": 1}.get(arm, 0),
    )


def _cell_environment(repeat: int, arm: str, destination: Path) -> dict[str, str]:
    """Keep each cell's tracking identity distinct within the one rental."""
    environment = dict(os.environ)
    name = f"{environment.get('ETUDE_RUN', 'recording-cost')}-r{repeat}-{arm}"
    environment.update(
        ETUDE_RUN=name,
        ETUDE_RUN_ADDRESS=f"{EXPERIMENT}/{name}",
        ETUDE_OUTPUT_DIR=str(destination),
    )
    return environment


def cell(arm: str, output: Path) -> None:
    """Run one ordinary training cell; refuse a CPU fallback or failed stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from manabot.training.regime import TrainingRegime

    expected = Path(os.environ.get("MANABOT_COST_SOURCE", ROOT)).resolve()
    if not Path(execute_regime.__code__.co_filename).resolve().is_relative_to(expected):
        raise RuntimeError("cell imported the other revision's training implementation")
    if not Path(managym.__file__).resolve().is_relative_to(expected):
        raise RuntimeError("cell imported the other revision's engine package")
    regime = TrainingRegime.model_validate_json(
        Path(os.environ["MANABOT_COST_RECIPE"]).read_bytes()
    )
    # Prospective input v11 uses the current shared representation. Preserve the
    # historical recipe bytes; all arms use this identical fresh-world regime.
    regime.observation.policy_input_version = 11
    stage = self_play_stage(regime)
    stage.execution.device = "cuda"
    regime.wall_seconds = stage.execution.wall_seconds = 1200
    output.mkdir(parents=True)
    (output / "regime.json").write_text(regime.model_dump_json(indent=2))
    with VerifyStore(output / "runs.sqlite") as store:
        if arm == "baseline":
            # No instrumentation flags: unchanged production defaults before this Task.
            result = execute_regime(
                regime,
                SEED,
                output / "training",
                store,
                backend=resolve("torch"),
                experiment=EXPERIMENT,
            )
        else:
            result = execute_regime(
                regime,
                SEED,
                output / "training",
                store,
                backend=resolve("torch"),
                settings=_settings(arm),
                experiment=EXPERIMENT,
            )
    (output / "receipt.json").write_text(result.model_dump_json(indent=2))
    if (
        result.status != "completed"
        or torch.device(result.stages[0].actual_device).type != "cuda"
    ):
        raise RuntimeError(f"cell {arm}: {result.status}: {result.error}")
    from safetensors.numpy import load_file

    weights = hashlib.sha256()
    for key, array in sorted(
        load_file(output / "training" / "self-play-raw.safetensors").items()
    ):
        weights.update(key.encode())
        weights.update(str((array.dtype, array.shape)).encode())
        weights.update(array.tobytes())
    (output / "weights.sha256").write_text(weights.hexdigest())


def _baseline_source(probe: Probe, output: Path) -> tuple[Path, Path]:
    """Build the kept source with the same dependencies and release toolchain."""
    # Build products are runtime inputs, not experiment outputs to upload.
    baseline = Path(os.environ["ETUDE_INPUTS_DIR"]).parent / "baseline-source"
    baseline.mkdir()
    with tarfile.open(
        Path(os.environ["ETUDE_INPUTS_DIR"]) / "baseline.tar.gz"
    ) as archive:
        archive.extractall(baseline, filter="data")
    wheels = baseline / "managym" / "target" / "wheels"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "maturin",
            "build",
            "--release",
            "-i",
            sys.executable,
            "-m",
            str(baseline / "managym" / "Cargo.toml"),
            "-o",
            str(wheels),
        ],
        cwd=ROOT,
        check=True,
    )
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX"))
    with zipfile.ZipFile(next(wheels.glob("*.whl"))) as wheel:
        extension = next(name for name in wheel.namelist() if name.endswith(suffix))
        destination = baseline / "managym" / Path(extension).name
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.write_bytes(wheel.read(extension))
        partial.replace(destination)
    config = output / "baseline-config.json"
    config.write_text(json.dumps({"source_bundle": probe.baseline_bundle.model_dump()}))
    environment = dict(os.environ, PYTHONPATH=str(baseline), ETUDE_CONFIG=str(config))
    subprocess.run(
        [sys.executable, "-m", "manabot.infra.source"],
        cwd=baseline,
        env=environment,
        check=True,
    )
    return baseline, config


def run(probe: Probe) -> None:
    bind_etude_source()
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    baseline, baseline_config = _baseline_source(probe, output)
    names = (
        ["baseline", "marker"] if probe.comparison == "revisions" else list(OPTION_ARMS)
    )
    reference = "baseline" if probe.comparison == "revisions" else "off"
    recipe = output / "comparison-recipe.json"
    regime = elemental_recipe("recording-cost-elemental", UPDATES)
    regime.observation.policy_input_version = 11
    recipe.write_text(regime.model_dump_json(indent=2))
    rows: list[dict[str, object]] = []
    for repeat in range(REPEATS):
        rotation = repeat % len(names)
        order = names[rotation:] + names[:rotation]
        for arm in order:
            destination = output / f"repeat-{repeat}" / arm
            source = baseline if arm == "baseline" else ROOT
            environment = _cell_environment(repeat, arm, destination)
            environment.update(
                PYTHONPATH=str(source),
                MANABOT_COST_SOURCE=str(source),
                MANABOT_COST_RECIPE=str(recipe),
            )
            if arm == "baseline":
                environment["ETUDE_CONFIG"] = str(baseline_config)
            print(f"START {repeat} {arm}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "recording_cost.py"),
                    arm,
                    str(destination),
                ],
                check=True,
                timeout=1500,
                cwd=source,
                env=environment,
            )
            receipt = json.loads((destination / "receipt.json").read_text())
            stage = receipt["stages"][0]
            if stage["updates"] != UPDATES:
                raise RuntimeError(f"cell {arm}: expected {UPDATES} updates")
            errors = receipt.get("game_errors") or {}
            if errors.get("crashed", 0) or errors.get("abandoned", 0):
                raise RuntimeError(f"cell {arm}: game errors invalidate the comparison")
            row = {
                "repeat": repeat,
                "arm": arm,
                "decisions": stage["environment_decisions"],
                "updates": stage["updates"],
                "collection_dps": stage["environment_decisions"]
                / stage["collection_seconds"],
                "seconds_per_update": (
                    stage["collection_seconds"] + stage["learning_seconds"]
                )
                / stage["updates"],
                "host": receipt["host"],
                "game_errors": {
                    key: errors.get(key, 0)
                    for key in (
                        "attempted",
                        "completed",
                        "crashed",
                        "abandoned",
                        "discarded_samples",
                    )
                },
                "weights_sha256": (destination / "weights.sha256").read_text(),
                "source_commit": receipt["identities"]["source_commit"],
            }
            rows.append(row)
            (output / "measurements.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        pair = [row for row in rows if row["repeat"] == repeat]
        if len({(row["decisions"], row["weights_sha256"]) for row in pair}) != 1:
            raise RuntimeError(
                "comparison changed workload or final policy; no speed claim"
            )
    summaries = {}
    for arm in names:
        ratios = [
            float(row["collection_dps"])
            / float(
                next(
                    baseline["collection_dps"]
                    for baseline in rows
                    if baseline["repeat"] == row["repeat"]
                    and baseline["arm"] == reference
                )
            )
            for row in rows
            if row["arm"] == arm
        ]
        summaries[arm] = {
            "rate_ratio_median": statistics.median(ratios),
            "rate_ratio_range": [min(ratios), max(ratios)],
        }
        update_ratios = [
            float(row["seconds_per_update"])
            / float(
                next(
                    base["seconds_per_update"]
                    for base in rows
                    if base["repeat"] == row["repeat"] and base["arm"] == reference
                )
            )
            for row in rows
            if row["arm"] == arm
        ]
        summaries[arm].update(
            update_ratio_median=statistics.median(update_ratios),
            update_ratio_range=[min(update_ratios), max(update_ratios)],
        )
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))


def training_jobs(revision: str, comparison: str = "revisions") -> list[Run]:
    if comparison not in {"revisions", "options"}:
        raise ValueError("comparison must be revisions or options")
    job = Job(
        name="recording-cost",
        setup=machine_setup(" --extra play --extra artifacts --extra reports"),
        command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m etude.call --function experiments.recording_cost:run",
        config=Probe(
            source_bundle=bundle_at(ROOT, revision),
            baseline_bundle=bundle_at(ROOT, BASELINE),
            comparison=comparison,
        ).model_dump_json(),
        inputs=(
            Input(
                name="baseline.tar.gz",
                key=f"sha256/{BASELINE_ARCHIVE_SHA256}",
                uri=f"r2://manabot/sha256/{BASELINE_ARCHIVE_SHA256}",
                sha256=BASELINE_ARCHIVE_SHA256,
                size_bytes=13110664,
            ),
        ),
        repo=str(ROOT),
        revision=revision,
        experiment=EXPERIMENT,
        max_hours=3,
        max_usd=Decimal("5"),
        max_attempts=1,
    )
    return job.runs(SEED)


if __name__ == "__main__":
    cell(sys.argv[1], Path(sys.argv[2]))
