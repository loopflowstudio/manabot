"""L4 model-size ceiling study on the complete TLA trophy split.

Compares four BF16 model sizes (`w384l8` 23.9M, `w512l8` 41.5M, `w512l12`
54.1M, `w768l8` 91.2M) trained on up to 4 solo Colab L4 runtimes (`colab-l4-4x`)
against the completed `tla_trophies_colab_500k_v2` `w160l8` (6.0M) anchor to
find the largest model that clearly trains to a higher ceiling on an L4.

Every size uses the trophy default recipe (`tau_scale=0.1`, `focus_film=True`,
`stream_norm=True`, `bucketed_shapes=True`, `live_tokens=True`, `streams=256`,
`transitions=64`, `rows=512`, `workers=4`, `threads=2`), `head_dim=64`,
explicit 4x FFN (`attention_feedforward_dim = 4 * hidden_dim`), and `bf16` on
CUDA.

6-update VRAM & throughput probe (~1.5 CU across 4 L4s):
    uv run etude launch experiments/l4_ceiling.py --rent colab-l4-4x \\
        --experiment l4_ceiling_probe --max-usd 4 \\
        -- --updates 6 --hours 1.0 --usd-per-run 1.0 --checkpoint-every 100

Full ceiling study (~24-30 wall hours across 4 L4s):
    uv run etude launch experiments/l4_ceiling.py --rent colab-l4-4x \\
        --max-usd 30
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
import json
import os
import platform
from typing import TYPE_CHECKING, Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.records import Aligned, Fraction
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Report, Score
from etude.spec import Variable
from etude.train.tracking import track
import psutil
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_frontier import learning, model
from experiments.tla_trophies_floor import final_environment
from manabot.eval.card_metrics import CARD_ROLLUPS
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig
from manabot.training.learning import IterationSchedule, Precision
from manabot.training.panels import ML, PERF
from manabot.training.report import behaviour_curves, head_to_head

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


type SizeLabel = Literal["w384l8", "w512l8", "w512l12", "w768l8"]


@dataclass(frozen=True)
class SizeSpec:
    """One model point on the L4 ceiling ladder."""

    hidden_dim: int
    attention_layers: int
    num_attention_heads: int
    attention_feedforward_dim: int
    label: str


SIZE_SPECS: dict[SizeLabel, SizeSpec] = {
    "w384l8": SizeSpec(
        hidden_dim=384,
        attention_layers=8,
        num_attention_heads=6,
        attention_feedforward_dim=1536,
        label="w384l8 (23.9M, bf16)",
    ),
    "w512l8": SizeSpec(
        hidden_dim=512,
        attention_layers=8,
        num_attention_heads=8,
        attention_feedforward_dim=2048,
        label="w512l8 (41.5M, bf16)",
    ),
    "w512l12": SizeSpec(
        hidden_dim=512,
        attention_layers=12,
        num_attention_heads=8,
        attention_feedforward_dim=2048,
        label="w512l12 (54.1M, bf16)",
    ),
    "w768l8": SizeSpec(
        hidden_dim=768,
        attention_layers=8,
        num_attention_heads=12,
        attention_feedforward_dim=3072,
        label="w768l8 (91.2M, bf16)",
    ),
}
SIZES: tuple[SizeLabel, ...] = tuple(SIZE_SPECS)

SEED = 40902
TARGET_UPDATES = 2_000
HELD_OUT = "l4-ceiling-held-out-32"
HALF_TRAINING = "l4-ceiling-half-training-32"
HALF_TRAINING_SEATINGS = tuple(
    f"L4 ceiling half-training deal {i + 1}" for i in range(32)
)
VS_ANCHOR_PREFIX = "l4-ceiling-vs-w384l8-32"


def vs_anchor_seatings(size: SizeLabel) -> tuple[str, ...]:
    return tuple(f"L4 ceiling {size} vs w384l8 deal {i + 1}" for i in range(32))


class Config(trophies.Config):
    size: SizeLabel
    precision: Precision = "bf16"
    schedule_steps: int = TARGET_UPDATES
    tau_scale: float = 0.1
    workers: int = Field(default=4, ge=1)


def model_for(size: SizeLabel) -> ModelConfig:
    spec = SIZE_SPECS[size]
    return model("w64").model_copy(
        update={
            "focus_film": True,
            "stream_norm": True,
            "hidden_dim": spec.hidden_dim,
            "attention_layers": spec.attention_layers,
            "num_attention_heads": spec.num_attention_heads,
            "attention_feedforward_dim": spec.attention_feedforward_dim,
        }
    )


def resolved_precision(config: Config) -> Precision:
    return "float32" if config.device == "cpu" else config.precision


def settings(config: Config) -> "SelfPlay":
    base = trophies.settings(config)
    return base.model_copy(
        update={
            "workers": config.workers,
            "precision": resolved_precision(config),
            "model": model_for(config.size),
            "learning": learning("w64").model_copy(
                update={
                    "tau_scale": config.tau_scale,
                    "iteration_schedule": IterationSchedule(
                        steps=config.schedule_steps,
                        reference_steps=10_000,
                    ),
                }
            ),
        }
    )


def _runtime_summary(config: Config) -> dict[str, object]:
    summary: dict[str, object] = {
        "size": config.size,
        "precision": resolved_precision(config),
        "streams": config.streams,
        "rows": config.rows,
        "workers": config.workers,
        "threads": config.threads,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    if config.device == "cuda":
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            summary["gpu_name"] = props.name
            summary["gpu_vram_total_bytes"] = props.total_memory
            summary["cuda_max_memory_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            summary["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return summary


def train(config: Config) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    if config.device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)
        summary = _runtime_summary(config)
        (run.out / "l4_runtime.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"L4_RUNTIME {json.dumps(summary)}", flush=True)


def _parse_sizes(sizes: str) -> tuple[SizeLabel, ...]:
    if sizes == "all":
        return SIZES
    chosen: list[SizeLabel] = []
    for raw in sizes.split(","):
        name = raw.strip()
        if name not in SIZE_SPECS:
            raise ValueError(
                f"unknown size {name!r}; expected 'all' or subset of {', '.join(SIZES)}"
            )
        chosen.append(name)  # type: ignore[arg-type]
    if not chosen:
        raise ValueError("sizes must select at least one model")
    return tuple(chosen)


def training_jobs(
    revision: str | None = None,
    sizes: str = "all",
    updates: int = TARGET_UPDATES,
    schedule_steps: int = TARGET_UPDATES,
    hours: float = 28.0,
    usd_per_run: float = 7.0,
    attempts: int = 8,
    threads: int = 2,
    workers: int = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = 100,
) -> list[TrainingJob]:
    selected = _parse_sizes(sizes)
    return [
        TrainingJob(
            name=size,
            train=train,
            config=Config(
                size=size,
                updates=updates,
                schedule_steps=schedule_steps,
                threads=threads,
                workers=workers,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                size=size,
                updates=1,
                schedule_steps=schedule_steps,
                device="cpu",
                threads=1,
                workers=1,
                streams=2,
                rows=64,
                checkpoint_every=1,
            ),
            environment=tuple(
                Variable(name=name, value=str(threads))
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            ),
            seeds=(SEED,),
            repo=str(trophies.ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(trophies.corpus_input(),),
            extras=("artifacts",),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=attempts,
        )
        for size in selected
    ]


def size_evaluations(
    runs: Sequence[TrainingRunSummary], size: SizeLabel
) -> list[EvalJob]:
    held_out = trophies.held_out_jobs(runs)
    if not held_out:
        return []
    (shared,) = held_out
    benchmark = shared.tournament
    env = final_environment()

    def against(
        name: str,
        checkpoints: Sequence[CheckpointRecord],
        seating_names: tuple[str, ...] | None = None,
        opponent: Fraction | Aligned | None = None,
    ) -> EvalJob:
        names = seating_names or tuple(s.name for s in benchmark.seatings)
        seatings = tuple(
            seating.model_copy(
                update={
                    "name": label,
                    "games": 1,
                    "seeds": "one",
                    **(
                        {}
                        if opponent is None
                        else {"players": (seating.players[0], opponent)}
                    ),
                }
            )
            for seating, label in zip(benchmark.seatings, names, strict=True)
        )
        return shared.model_copy(
            update={
                "name": name,
                "tournament": benchmark.model_copy(
                    update={
                        "name": name,
                        "environment": env,
                        "seatings": seatings,
                        "rollups": CARD_ROLLUPS,
                    }
                ),
                "checkpoints": checkpoints,
            }
        )

    half = Fraction(clock="updates", value=0.5, trained=True)
    jobs = [
        against(f"{HELD_OUT}-{size}", shared.checkpoints),
        against(
            f"{HALF_TRAINING}-{size}", shared.checkpoints, HALF_TRAINING_SEATINGS, half
        ),
    ]
    if size != "w384l8":
        trained = [c for c in shared.checkpoints if c.at.get("updates", 0) > 0]
        if trained:
            jobs.append(
                against(
                    f"{VS_ANCHOR_PREFIX}-{size}",
                    trained,
                    vs_anchor_seatings(size),
                    Aligned(clock="updates", job="w384l8"),
                )
            )
    return jobs


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    return [
        job
        for size in SIZES
        for job in size_evaluations(
            [run for run in state.runs if run.launch == size], size
        )
    ]


def report() -> Report:
    held_out = tuple(Matches(tournament=f"{HELD_OUT}-{size}") for size in SIZES)
    return Report(
        clock="updates",
        score=Score(table="plays", value="score", reference=0.5, range=(0.0, 1.0)),
        evals=(
            Curve(title="Score against the heuristic", matches=held_out),
            head_to_head("Against itself at half its updates", HALF_TRAINING_SEATINGS),
            *(
                head_to_head(
                    f"{size} against w384l8 at the same update",
                    vs_anchor_seatings(size),
                )
                for size in SIZES
                if size != "w384l8"
            ),
            *behaviour_curves(matches=held_out),
        ),
        labels={name: spec.label for name, spec in SIZE_SPECS.items()},
        perf=PERF,
        ml=ML,
    )
