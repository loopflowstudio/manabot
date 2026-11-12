"""MTG-328: what the forward layout costs on CPU and saves in CUDA evaluation.

Two exploratory measurements behind the `auto` forward-layout defaults
(`Execution` and `EvalAllocation`: bucketed padding and live-token attention on
for CUDA, off elsewhere, `live_tokens=True` refused off CUDA).

`cpu` times the live-token fallback off CUDA. It replays MTG-319's retained
w160l8 inputs (`experiments/data/live_token_attention/inputs`: collection
observations and learner minibatches) through the padded path, live tokens and
live tokens over buckets on the local CPU with a fixed thread count:

    uv run python -m experiments.eval_forward_layout cpu \\
        experiments/data/live_token_attention/inputs \\
        experiments/data/eval_forward_layout

`job` runs on one rented L4. It saves a seeded random w384l8 policy and plays the
same served evaluation pass (the policy against itself, same deals) through
`manabot.eval.evaluation.evaluate` with the padded layout and with the layout a
CUDA allocation resolves to, interleaved, in separate processes:

    uv run etude launch experiments/eval_forward_layout.py --rent l4-etude \\
        --revision COMMIT

Report: experiments/eval_forward_layout.md. Its figures come from the evidence:

    uv run --extra reports python -m experiments.eval_forward_layout charts \\
        experiments/data/eval_forward_layout
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, TypedDict

import torch

from experiments import l4_step_profile as profile

ROOT = Path(__file__).resolve().parents[1]
SEED = 32801
SIZE = "w384l8"
# One evaluation pass: every deal is played in both seatings.
DEALS = tuple(range(328_000, 328_032))
ACTIVE_GAMES = 32
# Padded first and last, so host drift shows as a difference between repeats.
ORDER: tuple[tuple[str, str], ...] = (
    ("padded", "padded-1"),
    ("layout", "layout-1"),
    ("padded", "padded-2"),
    ("layout", "layout-2"),
)
# CPU fixture: threads, untimed warm-up passes and timed passes per layout.
CPU_THREADS = 4
CPU_WARMUP = 1
CPU_PASSES = 3
CPU_ARMS: tuple[tuple[str, str], ...] = (
    ("baseline", "Padded"),
    ("live", "Live tokens, full tables"),
    ("bucketed-live", "Live tokens over buckets"),
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CPU cost of the live-token fallback


class CpuArm(TypedDict):
    learner_seconds: list[list[float]]
    inference_seconds: list[list[float]]
    learner_ms: float
    inference_ms: float


def cpu(captures: Path, out: Path) -> dict[str, Any]:
    """Time each layout's learner forward/backward and collection forward on CPU.

    The inputs are the retained minibatches and observations, replayed in the
    same order for every layout; nothing here creates work beyond those passes.
    A learner pass is the minibatch's forward, loss and backward without an
    optimizer step, so every layout sees identical weights.
    """
    from manabot.model.inference import PolicyInference

    torch.set_num_threads(CPU_THREADS)
    inputs = profile._inputs(captures)
    arms: dict[str, CpuArm] = {}
    for arm, _ in CPU_ARMS:
        agent = profile._agent(captures, "w160l8", "cpu", arm)
        learner, observer = profile._learner(agent, "cpu")
        generator = torch.Generator().manual_seed(SEED)
        learner_seconds: list[list[float]] = []
        inference_seconds: list[list[float]] = []
        with PolicyInference(agent) as scope:
            for index in range(CPU_WARMUP + CPU_PASSES):
                learn: list[float] = []
                for payload in inputs["learner"]:
                    started = time.perf_counter()
                    profile.learn_once(learner, observer, payload, step=False)
                    learn.append(time.perf_counter() - started)
                infer: list[float] = []
                for observation in inputs["inference"]:
                    started = time.perf_counter()
                    profile.infer_once(agent, scope, observation, generator)
                    infer.append(time.perf_counter() - started)
                if index >= CPU_WARMUP:
                    learner_seconds.append(learn)
                    inference_seconds.append(infer)
        arms[arm] = {
            "learner_seconds": learner_seconds,
            "inference_seconds": inference_seconds,
            "learner_ms": 1000
            * statistics.median(t for row in learner_seconds for t in row),
            "inference_ms": 1000
            * statistics.median(t for row in inference_seconds for t in row),
        }
        print(
            f"{arm}: learner {arms[arm]['learner_ms']:.0f} ms, "
            f"collection forward {arms[arm]['inference_ms']:.0f} ms",
            flush=True,
        )
    result = {
        "size": "w160l8",
        "threads": CPU_THREADS,
        "warmup_passes": CPU_WARMUP,
        "timed_passes": CPU_PASSES,
        "learner_rows": [len(p["actions"]) for p in inputs["learner"]],
        "inference_rows": [len(o["actions_valid"]) for o in inputs["inference"]],
        "host": profile._host(),
        "arms": arms,
    }
    _write(out / "cpu.json", result)
    return result


# ---------------------------------------------------------------------------
# One served evaluation pass per layout


def checkpoint(path: Path, size: str = SIZE) -> None:
    """Save a seeded random policy of `size`, bound to the selected matchup."""
    from manabot.env.match import Match
    from manabot.env.observation import ObservationSpace
    from manabot.infra.config import ModelConfig, ObservationConfig
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.checkpoint import save_checkpoint
    from manabot.recipes.builders import PACK
    from manabot.sim.matchup import selected_match

    space = ObservationSpace(ObservationConfig())
    torch.manual_seed(SEED)
    agent = ActorCritic(space, ModelConfig(world=PACK, **profile.SIZES[size]))
    save_checkpoint(
        agent, space, path, player_configs=Match(selected_match()).to_rust()
    )


def evaluation_pass(
    policy: Path, out: Path, arm: str, device: str, deals: int
) -> dict[str, Any]:
    """Play one pass of the policy against itself and write `receipt.json`.

    `padded` writes both layout switches off; `layout` leaves them to the
    allocation, which is what an ordinary evaluation does. The wall time covers
    `evaluate`: admission, the serving child's cold load, every game and replay.
    """
    from manabot.eval.evaluation import evaluate
    from manabot.eval.local import checkpoint_request
    from manabot.sim.execution import EvalExecution

    request = checkpoint_request(
        policy, opponent=policy, deal_seeds=DEALS[:deals], game_seconds=600
    )
    layout: dict[str, bool] = (
        {"bucketed_shapes": False, "live_tokens": False} if arm == "padded" else {}
    )
    execution = EvalExecution(
        device=device,
        max_active_games=ACTIVE_GAMES,
        max_batch_rows=ACTIVE_GAMES,
        resident_model_bytes=2**31,
        deadline_unix=time.time() + 1500,
        **layout,
    )
    started = time.perf_counter()
    result = evaluate(request, out_dir=out / "evaluation", execution=execution)
    seconds = time.perf_counter() - started
    receipt = {
        "arm": arm,
        "device": device,
        "status": result.status,
        "error": result.error,
        "seconds": seconds,
        "evaluation_seconds": result.evaluation_seconds,
        "games": len(result.rows),
        "bucketed_shapes": execution.bucketed_shapes,
        "live_tokens": execution.live_tokens,
        "execution": execution.model_dump(mode="json"),
        "decisions": sum(row.decisions for row in result.rows),
        "score": result.score.model_dump() if result.score else None,
    }
    _write(out / "receipt.json", receipt)
    return receipt


def job(smoke: bool = False) -> None:
    """Checkpoint, then each pass in its own process; `smoke` is a toy CPU run."""
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    policy = out / "inputs" / "policy.pt"
    device, size, deals = ("cpu", "w160l8", 1) if smoke else ("cuda", SIZE, len(DEALS))
    checkpoint(policy, size)
    stages: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for arm, name in ORDER:
            if smoke and arm == "layout":
                continue
            log = out / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            began = time.perf_counter()
            with log.open("w") as handle:
                code = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.eval_forward_layout",
                        "pass",
                        str(policy),
                        str(out / "passes" / name),
                        arm,
                        "--device",
                        device,
                        "--deals",
                        str(deals),
                    ],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                ).returncode
            stages.append(
                {
                    "name": name,
                    "returncode": code,
                    "seconds": time.perf_counter() - began,
                }
            )
            print(f"{name}: exit {code}, {stages[-1]['seconds']:.0f} s", flush=True)
    finally:
        _write(
            out / "job.json",
            {
                "stages": stages,
                "seconds": time.perf_counter() - started,
                "host": profile._host(),
            },
        )
        _write(out / "summary.json", summarize(out))
        policy.unlink(missing_ok=True)


def summarize(out: Path) -> dict[str, Any]:
    """Each pass's receipt and serving measurements, and the layout's speedup."""
    passes: dict[str, Any] = {}
    for _, name in ORDER:
        receipt = out / "passes" / name / "receipt.json"
        if not receipt.exists():
            continue
        row = json.loads(receipt.read_text())
        row["serving"] = _serving(out / "passes" / name / "evaluation")
        service = row["serving"].get("allocation/serving.json")
        if service:
            # A served forward is the model call plus the transfer that waits
            # for its outputs; CUDA work is asynchronous until that transfer.
            measured = service["measurements"]
            calls = service["counters"]["forward_calls"]
            forward = sum(
                measured[name]["seconds"]
                for name in ("forward_host", "output_transfer_host")
            )
            row["forward_calls"] = calls
            row["forward_ms"] = 1000 * forward / calls
            row["rows_per_forward"] = service["counters"]["decisions"] / calls
            row["decisions_per_second"] = row["decisions"] / row["seconds"]
        passes[name] = row

    def seconds(arm: str) -> list[float]:
        return [
            p["seconds"]
            for p in passes.values()
            if p["arm"] == arm and p["status"] == "completed"
        ]

    padded, layout = seconds("padded"), seconds("layout")
    summary: dict[str, Any] = {"passes": passes}
    if padded and layout:
        summary["padded_seconds"] = statistics.mean(padded)
        summary["layout_seconds"] = statistics.mean(layout)
        summary["speedup"] = summary["padded_seconds"] / summary["layout_seconds"]

    def forward(arm: str) -> list[float]:
        return [
            p["forward_ms"]
            for p in passes.values()
            if p["arm"] == arm and p["status"] == "completed" and "forward_ms" in p
        ]

    if forward("padded") and forward("layout"):
        summary["padded_forward_ms"] = statistics.mean(forward("padded"))
        summary["layout_forward_ms"] = statistics.mean(forward("layout"))
        summary["forward_speedup"] = (
            summary["padded_forward_ms"] / summary["layout_forward_ms"]
        )
    return summary


def _serving(evaluation: Path) -> dict[str, Any]:
    """The serving child's retained counters and timed measurements, as written."""
    found: dict[str, Any] = {}
    for path in sorted(evaluation.rglob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and "measurements" in body and "counters" in body:
            found[str(path.relative_to(evaluation))] = {
                "counters": body["counters"],
                "measurements": body["measurements"],
            }
    return found


def runs(revision: str, hours: str = "1.0", usd: str = "3.00") -> Any:
    """One L4 job at `revision`."""
    from etude.launch.records import Job

    from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

    return Job(
        name="mtg328-eval-forward-layout",
        setup=machine_setup(" --extra evaluation"),
        command=(
            'mkdir -p "$ETUDE_OUTPUT_DIR"\n'
            "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used "
            '--format=csv,noheader,nounits -l 1 > "$ETUDE_OUTPUT_DIR/gpu-samples.csv" &\n'
            "sampler=$!\n"
            "trap 'kill \"$sampler\"' EXIT\n"
            f"{TOOLCHAIN_ENV}\n"
            "uv run --locked --extra evaluation python -m experiments.eval_forward_layout job"
        ),
        repo=str(ROOT),
        revision=revision,
        max_hours=float(hours),
        max_usd=Decimal(usd),
        max_attempts=1,
    )


# ---------------------------------------------------------------------------
# Report figures

_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
_PADDED, _LAYOUT = "#898781", "#eb6834"


def charts(folder: Path) -> None:
    """Write `evaluation_pass.svg` and `cpu_cost.svg` into `folder` (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": _INK,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "xtick.color": _MUTED,
            "ytick.color": _INK,
            "svg.fonttype": "none",
            # Fixed ids and no date: redrawn figures keep their manifest digests.
            "svg.hashsalt": "eval_forward_layout",
        }
    )

    def frame(axis: Any) -> None:
        axis.set_facecolor(_SURFACE)
        axis.grid(axis="x", color=_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(axis="y", length=0)

    def bars(
        axis: Any, labels: list[str], values: list[float], colors: list[str], unit: str
    ) -> None:
        frame(axis)
        rows = list(range(len(values)))
        axis.barh(rows, values, height=0.6, color=colors)
        limit = max(values) * 1.2
        for row, value in zip(rows, values, strict=True):
            axis.text(
                value + limit * 0.015,
                row,
                f"{value:.0f} {unit}",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(0, limit)
        axis.set_yticks(rows, labels)
        axis.set_ylim(len(values) - 0.5, -0.5)

    summary_path = folder / "summary.json"
    if summary_path.exists():
        passes = json.loads(summary_path.read_text())["passes"]
        names = [name for _, name in ORDER if name in passes]
        figure, axis = plt.subplots(figsize=(7, 2.2))
        figure.patch.set_facecolor(_SURFACE)
        bars(
            axis,
            [
                ("Padded" if passes[n]["arm"] == "padded" else "Bucketed, live tokens")
                + f", pass {n[-1]}"
                for n in names
            ],
            [passes[n]["seconds"] for n in names],
            [_PADDED if passes[n]["arm"] == "padded" else _LAYOUT for n in names],
            "s",
        )
        axis.set_xlabel(f"wall time of one evaluation pass, {SIZE} on an L4 (s)")
        figure.tight_layout()
        figure.savefig(folder / "evaluation_pass.svg", metadata={"Date": None})
        plt.close(figure)

    cpu_path = folder / "cpu.json"
    if cpu_path.exists():
        arms = json.loads(cpu_path.read_text())["arms"]
        figure, axes = plt.subplots(1, 2, figsize=(8, 2.0), sharey=True)
        figure.patch.set_facecolor(_SURFACE)
        for axis, key, title in (
            (axes[0], "learner_ms", "learner forward and backward (ms)"),
            (axes[1], "inference_ms", "collection forward (ms)"),
        ):
            bars(
                axis,
                [label for _, label in CPU_ARMS],
                [arms[arm][key] for arm, _ in CPU_ARMS],
                [_PADDED, _LAYOUT, _LAYOUT],
                "ms",
            )
            axis.set_xlabel(f"median {title}")
        figure.tight_layout()
        figure.savefig(folder / "cpu_cost.svg", metadata={"Date": None})
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cpu")
    p.add_argument("captures", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("job")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("pass")
    p.add_argument("policy", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=("padded", "layout"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--deals", type=int, default=len(DEALS))
    p = sub.add_parser("summarize")
    p.add_argument("out", type=Path)
    p = sub.add_parser("charts")
    p.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "cpu":
        cpu(args.captures, args.out)
    elif args.command == "job":
        job(args.smoke)
    elif args.command == "pass":
        receipt = evaluation_pass(
            args.policy, args.out, args.arm, args.device, args.deals
        )
        print(json.dumps({k: receipt[k] for k in ("status", "seconds", "games")}))
        if receipt["status"] != "completed":
            raise SystemExit(1)
    elif args.command == "summarize":
        _write(args.out / "summary.json", summarize(args.out))
    else:
        charts(args.folder)


if __name__ == "__main__":
    main()
