"""MTG-306: one pooled typed-token stream against the v9 object tables. Exploratory.

The arm is `ModelConfig.token_stream="pooled"` (`manabot.model.token_stream`): the
same embeddings and joins as the table model, then one attention sequence of each
observation's valid players, cards, permanents, continuation relations and legal
actions, padded per batch to a 32-token bucket; logits read the action tokens.
Everything else of the Elemental recipe stays.

Three stages, in order (report: experiments/token_stream.md):

1. `fill`: per-observation entity row counts on captured observation batches, and
   the padded elements of (a) the declared capacities, (b) per-axis power-of-two
   bucketing (MTG-302's scheme: minimum 8, capped) and (c) one pooled budget per
   batch at a common row width, with the attention lengths each model sees. Counts
   only; runs anywhere, on any directory of `.npz` batches or learner `.pt` files.
2. `--stage timing` (one rented L4): two in-situ CUDA updates of each arm from a
   fresh initialization (end-to-end seconds per update); the tables arm keeps a
   spread of collection observations and learner minibatches. Those inputs then
   replay through both arms at w64l2 and w160l8, interleaved per pass: collection
   `PolicyInference.prepare` + forward, and one learner step (forward, a stand-in
   loss of legal-action log-likelihood plus value MSE, backward, Adam). Each records
   CUDA-event wall time, profiled kernel time and launches, synchronizing calls
   and peak memory. Fill runs on the same captures.
3. `--stage learning`: both arms on the Elemental recipe, shared seeds, matched
   updates; `eval(state)` scores checkpoints against the heuristic on the
   closure-lambda Elemental pairings and pooled against tables at matched updates.

    uv run python -m experiments.token_stream fill DIR [DIR ...] --out fill.json
    uv run etude launch experiments/token_stream.py --rent l4 --stage timing \\
        --revision COMMIT
    uv run etude launch experiments/token_stream.py --rent l4 --stage learning \\
        --revision COMMIT
    uv run etude eval experiments/token_stream.py --rent cpu32 --source r2://manabot/jobs
    uv run etude report experiments/token_stream.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Literal
import warnings

from etude.eval.records import Code, Inference, Matchup
from etude.experiment import ExperimentState, TrainingRun
from etude.launch.records import Job, Run
from etude.report import Curve, HeadToHead, Matches, Matrix, Ranking, Report, Score
import numpy as np
from pydantic import BaseModel, ConfigDict

from experiments.elemental_variants import elemental_recipe
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    Conditions,
    _source,
    benchmark_jobs,
)
from manabot.env.observation import FACT_VALID_INDEX, RELATION_VALID_INDEX
from manabot.env.shared_effects import EFFECT_DIM
from manabot.experiment.job import TOOLCHAIN_ENV, TrainingJob, machine_setup
from manabot.infra.source import SourceBundle, bind_etude_source, bundle_at
from manabot.model.token_stream import bucket_length
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
type Arm = Literal["tables", "pooled"]
ARMS: tuple[Arm, Arm] = ("tables", "pooled")
SEEDS = "30601,30602"
UPDATES = 400
# Checkpoints scored against the heuristic; the last two also play head-to-head.
SCORED_UPDATES = (50, 100, 200, 400)
HEAD_TO_HEAD_UPDATES = (200, 400)
# Window hours and reservation per run, by stage. Learning: MTG-181's measured
# pace is 28-36 s per update on an L4, so 400 updates at 40 s plus setup and
# finalization, reserved at the $.60 ceiling. Timing: four in-situ updates and
# the replays.
WINDOWS: dict[str, tuple[float, float]] = {
    "learning": (5.5, 3.45),
    "timing": (1.0, 0.75),
}


def recipe(arm: Arm, updates: int) -> TrainingRegime:
    """The Elemental recipe (w64l2, both seats, 256 streams) with one stream choice."""
    regime = elemental_recipe(f"token-stream-{arm}-v1", updates)
    regime.agent.token_stream = arm
    return TrainingRegime.model_validate(regime.model_dump(mode="json"))


# --------------------------------------------------------------------------- fill

# Effect header columns: detail offset/count, member offset/count, validity.
_DETAIL_SPAN, _MEMBER_SPAN, _EFFECT_VALID = (13, 14), (15, 16), EFFECT_DIM - 1


@dataclass(frozen=True)
class RowTable:
    """One entity table of a batch: its valid rows and how far they reach.

    `tables` is the seat count of a [B,2,cap,...] buffer; `width` holds a row's
    floats with its one-to-one companions (a card's catalog id, an action's focus
    and name-candidate columns). `extent` is the last valid row + 1 over seats,
    which bounds how short a table can be trimmed without renumbering rows.
    """

    name: str
    tables: int
    capacity: int
    width: int
    counts: np.ndarray  # [B] valid rows, all seats
    extents: np.ndarray  # [B]


def _rows(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts and extents [B] of a validity array [B,cap] or [B,seats,cap]."""
    flat = valid.reshape(valid.shape[0], -1, valid.shape[-1]).astype(bool)
    counts = flat.sum((1, 2))
    reach = np.where(flat.any(1), np.arange(1, flat.shape[-1] + 1), 0)
    return counts, reach.max(1)


def _spans(headers: np.ndarray, span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows and extent of the CSR slices live effect headers own."""
    live = headers[..., _EFFECT_VALID] == 1
    offset, count = headers[..., span[0]], headers[..., span[1]]
    counts = np.where(live, count, 0).sum(1).astype(int)
    return counts, np.where(live, offset + count, 0).max(1).astype(int)


def row_tables(obs: Mapping[str, np.ndarray]) -> list[RowTable]:
    """Every entity table of one v9 observation batch."""
    found: list[RowTable] = []

    def table(name: str, valid: np.ndarray, width: int) -> None:
        counts, extents = _rows(valid)
        seats = valid.shape[1] if valid.ndim == 3 else 1
        found.append(RowTable(name, seats, valid.shape[-1], width, counts, extents))

    for seat in ("agent", "opponent"):
        table(
            f"{seat}_player",
            obs[f"{seat}_player_valid"],
            obs[f"{seat}_player"].shape[-1],
        )
        # The catalog id of each card row rides in `semantic_cards`.
        table(
            f"{seat}_cards",
            obs[f"{seat}_cards_valid"],
            obs[f"{seat}_cards"].shape[-1] + 1,
        )
        table(
            f"{seat}_permanents",
            obs[f"{seat}_permanents_valid"],
            obs[f"{seat}_permanents"].shape[-1],
        )
    action_width = sum(
        obs[k].shape[-1] for k in ("actions", "action_focus", "name_candidates")
    )
    table("actions", obs["actions_valid"], action_width)
    table("events", obs["events_valid"], obs["events"].shape[-1])
    table(
        "relations",
        obs["relations"][..., RELATION_VALID_INDEX] == 1,
        obs["relations"].shape[-1],
    )
    table("facts", obs["facts"][..., FACT_VALID_INDEX] == 1, obs["facts"].shape[-1])
    table("effects", obs["effects"][..., _EFFECT_VALID] == 1, obs["effects"].shape[-1])
    for name, span in (
        ("effect_details", _DETAIL_SPAN),
        ("effect_members", _MEMBER_SPAN),
    ):
        buffer = obs[name]
        counts, extents = _spans(obs["effects"], span)
        found.append(
            RowTable(name, 1, buffer.shape[1], buffer.shape[-1], counts, extents)
        )
    table("known_hand", obs["known_hand"][..., 0] != 0, obs["known_hand"].shape[-1])
    table("decklists", obs["decklists"][..., 0] != 0, obs["decklists"].shape[-1])
    return found


def _pow2(rows: int, capacity: int) -> int:
    return min(capacity, max(8, 1 << max(rows - 1, 0).bit_length()))


# Tables the policy attends over, and the extra token types the pooled stream adds.
_OBJECTS = tuple(
    f"{s}_{k}" for s in ("agent", "opponent") for k in ("player", "cards", "permanents")
)
_POOLED_EXTRA = ("relations", "actions")


def batch_fill(obs: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Padded elements of one batch under each layout, and attention lengths."""
    tables = row_tables(obs)
    batch = len(tables[0].counts)
    totals = sum(t.counts for t in tables)
    slots = sum(t.tables * t.capacity for t in tables)
    common = max(t.width for t in tables) + 1  # the widest row and a type id
    capacity = sum(batch * t.tables * t.capacity * t.width for t in tables)
    bucketed = sum(
        batch * t.tables * _pow2(int(t.extents.max()), t.capacity) * t.width
        for t in tables
    )
    pooled = batch * bucket_length(int(totals.max()), slots) * common
    ragged = int(sum((t.counts * t.width).sum() for t in tables))
    by_name = {t.name: t for t in tables}
    attended = sum(by_name[k].counts for k in (*_OBJECTS, *_POOLED_EXTRA))
    object_slots = sum(by_name[k].tables * by_name[k].capacity for k in _OBJECTS)
    # MTG-302 trims each object table to its power-of-two bucket.
    bucketed_slots = sum(
        by_name[k].tables * _pow2(int(by_name[k].extents.max()), by_name[k].capacity)
        for k in _OBJECTS
    )
    stream_slots = object_slots + sum(by_name[k].capacity for k in _POOLED_EXTRA)
    stream_length = bucket_length(int(attended.max()), stream_slots)
    return {
        "batch": batch,
        "tokens_per_observation": totals.tolist(),
        "rows": {t.name: t.counts.tolist() for t in tables},
        "elements": {
            "capacity": capacity,
            "pow2_buckets": bucketed,
            "pooled": pooled,
            "ragged": ragged,
        },
        "attention": {
            "tables_slots": object_slots,
            "pooled_length": stream_length,
            "pow2_slots": bucketed_slots,
            "pooled_tokens_per_observation": attended.tolist(),
            # Query-key pairs per batch: B·N² against B·L².
            "pair_ratio": object_slots**2 / stream_length**2,
            "pow2_pair_ratio": bucketed_slots**2 / stream_length**2,
        },
    }


def _batches(directory: Path) -> Iterator[dict[str, np.ndarray]]:
    """Captured batches: collection `.npz` files and learner `.pt` minibatches."""
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path) as data:
            yield {key: data[key] for key in data.files}
    learner = sorted(directory.rglob("*.pt"))
    if learner:
        import torch

        for path in learner:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(saved, dict) and isinstance(saved.get("obs"), dict):
                yield {k: v.numpy() for k, v in saved["obs"].items()}


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "median": float(statistics.median(ordered)),
        "p90": float(p90),
        "max": float(ordered[-1]),
    }


def fill(directory: Path) -> dict[str, object]:
    """Fill summary over every captured batch under `directory`."""
    batches = [batch_fill(obs) for obs in _batches(directory)]
    if not batches:
        raise ValueError(f"no captured batches under {directory}")
    tokens = [n for b in batches for n in b["tokens_per_observation"]]  # type: ignore[union-attr]
    rows: dict[str, list[float]] = {}
    for b in batches:
        for name, counts in b["rows"].items():  # type: ignore[union-attr]
            rows.setdefault(name, []).extend(counts)
    elements = {
        key: sum(b["elements"][key] for b in batches)  # type: ignore[index]
        for key in ("capacity", "pow2_buckets", "pooled", "ragged")
    }
    attention = [b["attention"] for b in batches]
    histogram, edges = np.histogram(tokens, bins=np.arange(0, max(tokens) + 17, 16))
    return {
        "directory": str(directory),
        "batches": len(batches),
        "observations": len(tokens),
        "batch_sizes": _quantiles([b["batch"] for b in batches]),  # type: ignore[misc]
        "tokens_per_observation": _quantiles(tokens),
        "token_histogram": {"edges": edges.tolist(), "counts": histogram.tolist()},
        "rows_per_observation": {name: _quantiles(v) for name, v in rows.items()},
        "padded_elements": elements,
        "ratios": {
            "capacity_over_pooled": elements["capacity"] / elements["pooled"],
            "pow2_over_pooled": elements["pow2_buckets"] / elements["pooled"],
            "pooled_over_ragged": elements["pooled"] / elements["ragged"],
        },
        "attention": {
            "tables_slots": attention[0]["tables_slots"],  # type: ignore[index]
            "pooled_length": _quantiles([a["pooled_length"] for a in attention]),  # type: ignore[index]
            "pooled_tokens_per_observation": _quantiles(
                [n for a in attention for n in a["pooled_tokens_per_observation"]]  # type: ignore[index]
            ),
            "pow2_slots": _quantiles([a["pow2_slots"] for a in attention]),  # type: ignore[index]
            "pair_ratio": _quantiles([a["pair_ratio"] for a in attention]),  # type: ignore[index]
            "pow2_pair_ratio": _quantiles(
                [a["pow2_pair_ratio"] for a in attention]  # type: ignore[index]
            ),
        },
    }


# --------------------------------------------------------------------------- timing


class Probe(BaseModel):
    """The timing job's configuration."""

    model_config = ConfigDict(extra="forbid")
    source_bundle: SourceBundle | None = None
    seed: int = 30600
    insitu_updates: int = 2
    # Every `collect_every`-th collection forward is kept, up to `collect_keep`.
    collect_every: int = 6
    collect_keep: int = 48
    learn_keep: int = 24
    warmup_passes: int = 2
    timed_passes: int = 5
    # `smoke` (local) only: a CPU run of every stage on a few streams. Its
    # timings mean nothing; it proves the job runs before renting.
    device: Literal["cuda", "cpu"] = "cuda"
    streams: int | None = None


SMOKE = Probe(
    device="cpu",
    streams=4,
    insitu_updates=1,
    collect_every=1,
    collect_keep=3,
    learn_keep=2,
    warmup_passes=0,
    timed_passes=1,
)


# Replay sizes: the recipe's own, and the w160l8 class of MTG-267/301.
SIZES: dict[str, dict[str, int]] = {
    "w64l2": {},
    "w160l8": {"hidden_dim": 160, "attention_layers": 8, "num_attention_heads": 8},
}


@contextmanager
def _capturing(out: Path, probe: Probe) -> Iterator[None]:
    """Keep collection inputs (no grad) and learner minibatches (grad) of the forward."""
    import torch

    from manabot.backends.torch.policy import ActorCritic
    from manabot.model.inference import PreparedObservation

    original = ActorCritic.forward_distribution
    calls = {"collect": 0, "kept": 0, "learn": 0}

    def hooked(
        self: ActorCritic, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            if calls["learn"] < probe.learn_keep:
                torch.save(
                    {"obs": {k: v.detach().cpu() for k, v in obs.items()}},
                    out / "learner" / f"{calls['learn']:03d}.pt",
                )
            calls["learn"] += 1
        elif isinstance(obs, PreparedObservation):
            # Collection's own forwards; other no-grad reads are not collection.
            if (
                calls["collect"] % probe.collect_every == 0
                and calls["kept"] < probe.collect_keep
            ):
                np.savez_compressed(
                    out / "collection" / f"{calls['kept']:03d}.npz",
                    **{k: v.detach().cpu().numpy() for k, v in obs.items()},
                )
                calls["kept"] += 1
            calls["collect"] += 1
        return original(self, obs)

    (out / "learner").mkdir(parents=True, exist_ok=True)
    (out / "collection").mkdir(parents=True, exist_ok=True)
    ActorCritic.forward_distribution = hooked  # type: ignore[method-assign]
    try:
        yield
    finally:
        ActorCritic.forward_distribution = original  # type: ignore[method-assign]


@contextmanager
def _untracked() -> Iterator[None]:
    """Hide the launcher's identity: in-situ updates are local runs, never this
    job's Etude run, which would otherwise receive both arms' checkpoints."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("ETUDE_")}
    for key in saved:
        del os.environ[key]
    try:
        yield
    finally:
        os.environ.update(saved)


def _insitu(
    arm: Arm, probe: Probe, out: Path, capture: Path | None
) -> dict[str, object]:
    """`insitu_updates` real CUDA updates of one arm; its receipt's phase seconds."""
    from manabot.training.execution import execute_regime, validate_regime
    from manabot.verify.store import VerifyStore

    regime = recipe(arm, probe.insitu_updates)
    regime.wall_seconds = 3600.0
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.execution.wall_seconds = 3600.0
    stage.execution.device = probe.device
    if probe.streams is not None:
        stage.streams = probe.streams
    regime = validate_regime(regime)
    folder = out / f"insitu-{arm}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with _untracked(), VerifyStore(folder / "runs.sqlite") as store:
        if capture is None:
            run = execute_regime(regime, probe.seed, folder / "run", store)
        else:
            with _capturing(capture, probe):
                run = execute_regime(regime, probe.seed, folder / "run", store)
    (folder / "receipt.json").write_text(run.model_dump_json())
    if run.status != "completed":
        raise RuntimeError(f"in-situ {arm} run {run.status}: {run.error}")
    record = run.stages[0]
    return {
        "process_seconds": time.perf_counter() - started,
        "updates": record.updates,
        "collection_seconds": record.collection_seconds,
        "learning_seconds": record.learning_seconds,
        "environment_decisions": record.environment_decisions,
        "games": record.games,
    }


@dataclass
class _Sample:
    wall_ms: list[float]
    kernel_ms: float = 0.0
    launches: int = 0
    syncs: int = 0
    peak_bytes: int = 0


def _timed(call: Callable[[], object], device: str) -> float:
    import torch

    if device != "cuda":
        started = time.perf_counter()
        call()
        return (time.perf_counter() - started) * 1000
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _profile(calls: list[Callable[[], object]], sample: _Sample) -> None:
    """Kernel time and launches over one pass, then synchronizing calls."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for call in calls:
            call()
        torch.cuda.synchronize()
    events = prof.key_averages()
    sample.kernel_ms = (
        sum(
            getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
            for e in events
        )
        / 1000
        / len(calls)
    )
    sample.launches = sum(
        e.count
        for e in events
        if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")
    ) // len(calls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for call in calls:
                call()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    sample.syncs = sum("synchroniz" in str(w.message) for w in caught) // len(calls)


def _replay(capture: Path, probe: Probe) -> dict[str, object]:
    """Both arms at each size on the captured inputs, interleaved pass by pass."""
    import torch

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.model.inference import PolicyInference

    collection = [
        {k: torch.from_numpy(v) for k, v in obs.items()}
        for obs in _batches(capture / "collection")
    ]
    learner = [
        {k: torch.from_numpy(v) for k, v in obs.items()}
        for obs in _batches(capture / "learner")
    ]
    device = torch.device(probe.device)
    cuda = probe.device == "cuda"
    results: dict[str, object] = {}
    for size, overrides in SIZES.items():
        agents: dict[Arm, ActorCritic] = {}
        optimizers: dict[Arm, torch.optim.Optimizer] = {}
        for arm in ARMS:
            regime = recipe(arm, 1)
            config = regime.agent.model_copy(update=overrides)
            torch.manual_seed(probe.seed)
            agent = ActorCritic(ObservationSpace(regime.observation), config).to(device)
            agents[arm] = agent
            optimizers[arm] = torch.optim.Adam(agent.parameters(), lr=1e-4)
        learner_device = [{k: v.to(device) for k, v in obs.items()} for obs in learner]

        def collect_calls(arm: Arm) -> list[Callable[[], object]]:
            agent = agents[arm].eval()
            # Collection holds one scope per update, so its catalog encodes once.
            scope = PolicyInference(agent)

            def forward(obs: dict[str, torch.Tensor]) -> Callable[[], object]:
                def call() -> object:
                    with torch.no_grad():
                        return agent(scope.prepare(obs))

                return call

            return [forward(obs) for obs in collection]

        def learn_calls(arm: Arm) -> list[Callable[[], object]]:
            agent, optimizer = agents[arm], optimizers[arm]

            def step(obs: dict[str, torch.Tensor]) -> Callable[[], object]:
                def call() -> object:
                    agent.train()
                    logits, value = agent.forward_distribution(obs)
                    legal = obs["actions_valid"].bool()
                    # Stand-in loss with the damped loss's gradient support:
                    # every legal logit and the value head.
                    logs = torch.log_softmax(logits.masked_fill(~legal, -1e8), -1)
                    loss = -(logs * legal).sum(-1).mean() + value.square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    return loss

                return call

            return [step(obs) for obs in learner_device]

        for phase, calls_for in (
            ("collection", collect_calls),
            ("learner", learn_calls),
        ):
            samples = {arm: _Sample(wall_ms=[]) for arm in ARMS}
            calls = {arm: calls_for(arm) for arm in ARMS}
            for arm in ARMS:
                for _ in range(probe.warmup_passes):
                    for call in calls[arm]:
                        call()
            for _ in range(probe.timed_passes):
                for arm in ARMS:
                    samples[arm].wall_ms.append(
                        sum(_timed(call, probe.device) for call in calls[arm])
                        / len(calls[arm])
                    )
            for arm in cuda * ARMS:
                torch.cuda.reset_peak_memory_stats()
                for call in calls[arm]:
                    call()
                samples[arm].peak_bytes = torch.cuda.max_memory_allocated()
                _profile(calls[arm], samples[arm])
            results[f"{size}/{phase}"] = {
                arm: {
                    "wall_ms_per_call": _quantiles(s.wall_ms),
                    "kernel_ms_per_call": s.kernel_ms,
                    "launches_per_call": s.launches,
                    "syncs_per_call": s.syncs,
                    "peak_bytes": s.peak_bytes,
                    "parameters": sum(p.numel() for p in agents[arm].parameters()),
                    "calls": len(calls[arm]),
                }
                for arm, s in samples.items()
            }
            print(
                f"TIMING {size}/{phase}: {json.dumps(results[f'{size}/{phase}'])}",
                flush=True,
            )
        agents.clear()
        optimizers.clear()
        learner_device.clear()
        if cuda:
            torch.cuda.empty_cache()
    return results


def measure(probe: Probe) -> None:
    """The timing job (`etude.call`): in-situ updates, fill, then replay."""
    import torch

    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if probe.source_bundle is not None:
        bind_etude_source()
    if probe.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback measurement")
    summary: dict[str, object] = {
        "host": {
            "device": probe.device,
            "gpu": torch.cuda.get_device_name() if probe.device == "cuda" else None,
            "torch": torch.__version__,
        },
    }
    capture = output / "captures"

    def write() -> None:
        (output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    summary["insitu"] = {
        arm: _insitu(arm, probe, output, capture if arm == "tables" else None)
        for arm in ARMS
    }
    write()
    summary["fill"] = {
        "collection": fill(capture / "collection"),
        "learner": fill(capture / "learner"),
    }
    write()
    summary["replay"] = _replay(capture, probe)
    write()


# --------------------------------------------------------------------------- launch


def runs(
    revision: str,
    stage: str = "timing",
    updates: int = UPDATES,
    seeds: str = SEEDS,
    hours: float | None = None,
    usd_per_run: float | None = None,
) -> list[Run]:
    """`--stage timing` is one L4 probe; `--stage learning` trains both arms per seed.

    `hours` and `usd_per_run` default to the stage's `WINDOWS` entry."""
    if stage not in WINDOWS:
        raise ValueError("stage is timing or learning")
    hours = WINDOWS[stage][0] if hours is None else hours
    usd_per_run = WINDOWS[stage][1] if usd_per_run is None else usd_per_run
    if stage == "timing":
        probe = Probe(source_bundle=bundle_at(ROOT, revision))
        return Job(
            name="token-stream-timing",
            setup=machine_setup(""),
            command=f"{TOOLCHAIN_ENV}\nuv run --locked python -m etude.call "
            "--function experiments.token_stream:measure",
            config=probe.model_dump_json(),
            repo=str(ROOT),
            revision=revision,
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
            max_attempts=1,
        ).runs(probe.seed)
    wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    found: list[Run] = []
    for arm in ARMS:
        regime = recipe(arm, updates)
        regime.wall_seconds = wall_seconds
        stage_record = regime.stages[0]
        assert isinstance(stage_record, TrainSelfPlay)
        stage_record.execution.wall_seconds = wall_seconds
        found += (
            TrainingJob(
                regime=TrainingRegime.model_validate(regime.model_dump(mode="json")),
                experiment=EXPERIMENT,
                milestones=tuple(u for u in SCORED_UPDATES if u < updates),
                finalization_seconds=FINALIZATION_SECONDS,
            )
            .job(
                repo=ROOT,
                revision=revision,
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
            )
            .runs(*(int(seed) for seed in seeds.split(",")))
        )
    return found


# --------------------------------------------------------------------------- evaluation


def _arm(run: TrainingRun) -> Arm | None:
    """A learning run's arm from its job name; the timing probe has none."""
    for arm in ARMS:
        if run.name.startswith(f"token-stream-{arm}-"):
            return arm
    return None


def _at(run: TrainingRun) -> dict[int, object]:
    return {int(c.clocks["updates"]): c for c in run.checkpoints}


def eval(state: ExperimentState) -> list[Job]:
    """Each scored checkpoint against the heuristic; pooled against tables at matched
    updates, every pooled run against every tables run. Etude plays only new matches."""
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source)
    conditions = tuple(
        Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
        for pairing in ELEMENTAL_PAIRS.pairings
    )
    matchups: list[Matchup] = []
    for run in state.runs:
        if _arm(run) is None:
            continue
        for update, checkpoint in sorted(_at(run).items()):
            if update in SCORED_UPDATES:
                matchups.append(
                    Matchup(
                        name=f"{run.name} u{update} vs heuristic",
                        players=(checkpoint.model(inference), heuristic),  # type: ignore[attr-defined]
                        conditions=conditions,
                        games=ELEMENTAL_PAIRS.legs,
                        tags=("heuristic",),
                    )
                )
    pooled = [run for run in state.runs if _arm(run) == "pooled"]
    tables = [run for run in state.runs if _arm(run) == "tables"]
    for a in pooled:
        for b in tables:
            ours, theirs = _at(a), _at(b)
            for update in HEAD_TO_HEAD_UPDATES:
                if update in ours and update in theirs:
                    matchups.append(
                        Matchup(
                            name=f"{a.name} vs {b.name} u{update}",
                            players=(
                                ours[update].model(inference),  # type: ignore[attr-defined]
                                theirs[update].model(inference),  # type: ignore[attr-defined]
                            ),
                            conditions=conditions,
                            games=ELEMENTAL_PAIRS.legs,
                            tags=("head-to-head",),
                        )
                    )
    return benchmark_jobs(
        tuple(matchups), "token-stream-v1", ELEMENTAL_PAIRS, hours=2.5, usd=3.0
    )


def report() -> Report:
    """Strength at matched updates; the perf panel holds seconds per update."""
    heuristic = Matches(tag="heuristic")
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
        baseline="token-stream-tables-v1",
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            Curve(title="Pooled against tables", matches=Matches(tag="head-to-head")),
            HeadToHead(matches=Matches(tag="head-to-head")),
            Matrix(
                title="Model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MTG-306 fill counts (no timing)")
    commands = parser.add_subparsers(dest="command", required=True)
    counting = commands.add_parser("fill")
    counting.add_argument("directories", nargs="+", type=Path)
    counting.add_argument("--out", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="every timing stage on CPU, tiny")
    smoke.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        os.environ["ETUDE_OUTPUT_DIR"] = str(args.out)
        measure(SMOKE)
        return
    result = {str(d): fill(d) for d in args.directories}
    args.out.write_text(json.dumps(result, indent=1))
    for directory, summary in result.items():
        print(
            directory, json.dumps({k: summary[k] for k in ("observations", "ratios")})
        )


if __name__ == "__main__":
    sys.exit(main())
