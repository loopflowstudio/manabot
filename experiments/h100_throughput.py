"""Exploratory H100 throughput, bottleneck, and packing benchmark on TLA trophies.

Measures per-phase update timing (`native_stepping`, `observation_encoding`,
`observation_preparation`, `policy_inference_and_action_selection`,
`stream_bookkeeping`, `batch_assembly`, `learning`), host RSS, peak CUDA VRAM,
and aggregate decisions/CU across an H100-appropriate model size ladder on the
complete 7,859-pool TLA trophy split (`experiments/tla_trophies.py`).

All scaled sizes (`w384l8` through `w1024l16`) use `bf16` (activating CUDA
FlashAttention and Hopper BF16 Tensor Cores), 64-dim attention heads, and an
explicit 4x FFN (`attention_feedforward_dim = 4 * hidden_dim`) rather than
`ModelConfig`'s default `hidden_dim * num_attention_heads`. `w160l8` stays in
`float32` with its frontier dimensions as the direct Colab L4 anchor.

Run locally:
    uv run etude smoke experiments/h100_throughput.py

Solo size ladder on a 1-slot H100:
    etude launch experiments/h100_throughput.py --rent colab-h100-s1 --max-usd 10

Packing sweep at N concurrent runs on an N-slot H100:
    etude launch experiments/h100_throughput.py --rent colab-h100-s4 \\
        --experiment h100_pack_w512l12_s4 --max-usd 10 \\
        -- --sizes w512l12 --seeds 4
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import os
import platform
from typing import TYPE_CHECKING, Literal

from etude.jobs import TrainingJob
from etude.report import Report
from etude.spec import Variable
from etude.train.tracking import track
import psutil
from pydantic import Field

from experiments import tla_trophies as trophies
from experiments.tla_frontier import learning, model
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig
from manabot.training.learning import IterationSchedule, Precision
from manabot.training.panels import self_play_report

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay


type SizeLabel = Literal["w160l8", "w384l8", "w512l12", "w768l12", "w1024l16"]


@dataclass(frozen=True)
class SizeSpec:
    """One model point on the H100 throughput ladder."""

    hidden_dim: int
    attention_layers: int
    num_attention_heads: int
    attention_feedforward_dim: int | None
    precision: Precision
    label: str


# Span ~6M -> ~259M parameters (~43x):
# - w160l8 matches `experiments/tla_trophies_colab.py` (L4 anchor, fp32).
# - Larger sizes use head_dim=64, 4x FFN, and bf16 for Hopper FlashAttention
#   and BF16 Tensor Cores.
SIZE_SPECS: dict[SizeLabel, SizeSpec] = {
    "w160l8": SizeSpec(
        hidden_dim=160,
        attention_layers=8,
        num_attention_heads=8,
        attention_feedforward_dim=None,
        precision="float32",
        label="Anchor (w160l8, 6.0M, fp32)",
    ),
    "w384l8": SizeSpec(
        hidden_dim=384,
        attention_layers=8,
        num_attention_heads=6,
        attention_feedforward_dim=1536,
        precision="bf16",
        label="Bridge (w384l8, 23.2M, bf16)",
    ),
    "w512l12": SizeSpec(
        hidden_dim=512,
        attention_layers=12,
        num_attention_heads=8,
        attention_feedforward_dim=2048,
        precision="bf16",
        label="Mid (w512l12, 53.2M, bf16)",
    ),
    "w768l12": SizeSpec(
        hidden_dim=768,
        attention_layers=12,
        num_attention_heads=12,
        attention_feedforward_dim=3072,
        precision="bf16",
        label="Large (w768l12, 118.1M, bf16)",
    ),
    "w1024l16": SizeSpec(
        hidden_dim=1024,
        attention_layers=16,
        num_attention_heads=16,
        attention_feedforward_dim=4096,
        precision="bf16",
        label="XL (w1024l16, 258.9M, bf16)",
    ),
}
SIZES: tuple[SizeLabel, ...] = tuple(SIZE_SPECS)

BASE_SEED = 40902
# Update 1 carries compilation/allocator warmup; updates 2-6 supply 5 steady rows.
DEFAULT_UPDATES = 6
SCHEDULE_STEPS = 2_000


class Config(trophies.Config):
    size: SizeLabel
    precision: Precision | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    schedule_steps: int = SCHEDULE_STEPS
    tau_scale: float = 0.1
    workers: int = Field(default=4, ge=1)


def model_for(size: SizeLabel) -> ModelConfig:
    """Model config for one H100 ladder point."""
    spec = SIZE_SPECS[size]
    if size == "w160l8":
        return model("w160l8").model_copy(
            update={"focus_film": True, "stream_norm": True}
        )
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
    if config.device == "cpu":
        return "float32"
    return config.precision or SIZE_SPECS[config.size].precision


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
    """Capture host CPU/RAM and peak CUDA memory once training finishes."""
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
        (run.out / "h100_runtime.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"H100_RUNTIME {json.dumps(summary)}", flush=True)


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
    precision: Precision | None = None,
    updates: int = DEFAULT_UPDATES,
    seeds: int = 1,
    hours: float = 2.0,
    usd_per_run: float = 5.0,
    attempts: int = 6,
    threads: int = 2,
    workers: int = 4,
    streams: int = 256,
    rows: int = 512,
    checkpoint_every: int = 100,
) -> list[TrainingJob]:
    """One job per selected size; set `--seeds N` on an N-slot H100 to test packing."""
    selected = _parse_sizes(sizes)
    run_seeds = tuple(range(BASE_SEED, BASE_SEED + seeds))
    return [
        TrainingJob(
            name=f"{size}-{precision}" if precision is not None else size,
            train=train,
            config=Config(
                size=size,
                precision=precision,
                updates=updates,
                schedule_steps=SCHEDULE_STEPS,
                threads=threads,
                workers=workers,
                streams=streams,
                rows=rows,
                checkpoint_every=checkpoint_every,
            ),
            smoke=Config(
                size=size,
                updates=1,
                schedule_steps=SCHEDULE_STEPS,
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
            seeds=run_seeds,
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


def report() -> Report:
    base = self_play_report()
    return Report(
        clock="updates",
        labels={name: spec.label for name, spec in SIZE_SPECS.items()},
        perf=base.perf,
        ml=base.ml,
    )
