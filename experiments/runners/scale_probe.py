"""Isolated laptop model timings and ordinary recipe train/export/reload proofs.

The parent bounds all children under one 900-second attempt. TrainingRun owns
training evidence; this report derives timings and retains every failed child.
Diagnostic Adam throughput is not collector or Ataraxos learner throughput.
"""

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
from pydantic import BaseModel, Field
import torch

from experiments.runners.model_capacity import regimes
from experiments.runners.run_training_regimes import smoke_recipe
from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.hypers import RewardHypers
from manabot.model.agent import Agent
from manabot.model.architecture import ArchitectureReceipt, architecture_receipt
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.clock import watchdog_seconds
from manabot.training.execution import atomic_json, execute_regime
from manabot.training.models import (
    AtaraxosMoveLearning,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore


class Window(BaseModel):
    samples: int
    seconds: float


class Probe(BaseModel):
    architecture: ArchitectureReceipt
    attention_slots: int
    input_bytes: int
    parameter_bytes: int
    buffer_bytes: int
    gradient_adam_estimate_bytes: int
    construction_seconds: list[float] = Field(default_factory=list)
    first_forward_seconds: list[float] = Field(default_factory=list)
    forward: list[Window] = Field(default_factory=list)
    update: list[Window] = Field(default_factory=list)
    sampled_rss_bytes: int = 0
    mps_allocated_bytes: int | None = None
    mps_driver_bytes: int | None = None


class Attempt(BaseModel):
    case: str
    mode: str
    status: str
    seconds: float
    error: str | None = None
    result: str | None = None


class ScaleReport(BaseModel):
    status: str = "running"
    error: str | None = None
    platform_name: str = Field(default_factory=platform.platform)
    machine: str = Field(default_factory=platform.machine)
    torch_version: str = torch.__version__
    python_version: str = platform.python_version()
    batch: int = 4
    threads: int = 1
    seed: int = 115
    dtype: str = "float32"
    load_before: tuple[float, float, float] = Field(default_factory=os.getloadavg)
    load_after: tuple[float, float, float] | None = None
    elapsed_seconds: float = 0
    input_sha256: str = ""
    source: dict[str, str] = Field(default_factory=dict)
    attempts: list[Attempt] = Field(default_factory=list)


def scale_recipes() -> dict[str, TrainingRegime]:
    base = smoke_recipe("direct-self-play")
    base.agent.value_aggregation = "value_token"
    base.stages = base.stages[:1]
    stage = base.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates, stage.streams, stage.transitions = 1, 2, 8
    stage.learning = AtaraxosMoveLearning(gradient="ataraxos_move", min_advantage=0)
    stage.execution.wall_seconds = 80
    base.wall_seconds = 85
    return regimes(
        TrainingRegime.model_validate(base.model_dump()), include_ataraxos=True
    )


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _probe(recipe: TrainingRegime, obs: dict[str, torch.Tensor], device: str) -> Probe:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    space = ObservationSpace(recipe.observation)
    obs = {k: v.to(device) for k, v in obs.items()}
    report: Probe | None = None
    for phase in ("forward", "update"):
        torch.manual_seed(115)
        tick = time.perf_counter()
        agent = Agent(space, recipe.agent).to(device)
        _sync(device)
        construction = time.perf_counter() - tick
        agent.train(phase == "update")
        slots: list[int] = []

        def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            slots.append(args[0].shape[1])

        hook = agent.attention.register_forward_pre_hook(capture)
        if report is None:
            params = sum(p.numel() * p.element_size() for p in agent.parameters())
            report = Probe(
                architecture=architecture_receipt(agent),
                attention_slots=0,
                input_bytes=sum(v.numel() * v.element_size() for v in obs.values()),
                parameter_bytes=params,
                gradient_adam_estimate_bytes=3 * params,
                buffer_bytes=sum(v.numel() * v.element_size() for v in agent.buffers()),
            )
        report.construction_seconds.append(construction)
        optimizer = (
            torch.optim.Adam(agent.parameters(), lr=1e-4) if phase == "update" else None
        )

        def step() -> None:
            if optimizer is None:
                with torch.inference_mode():
                    agent(obs)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits, value = agent(obs)
                # Uniform legal-action cross entropy plus scalar value square.
                # All samples contribute; this is a diagnostic, not an RL loss.
                valid = obs["actions_valid"].bool()
                logp = logits.log_softmax(-1)
                loss = (
                    -(logp.masked_fill(~valid, 0).sum(-1) / valid.sum(-1)).mean()
                    + value.square().mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite diagnostic loss")
                loss.backward()
                optimizer.step()

        _sync(device)
        tick = time.perf_counter()
        step()
        _sync(device)
        report.first_forward_seconds.append(time.perf_counter() - tick)
        report.attention_slots = slots[0]
        expected = (
            2
            + 2 * space.encoder.cards_per_player
            + 2 * space.encoder.perms_per_player
            + 1
        )
        assert set(slots) == {expected}
        hook.remove()
        for _ in range(2):
            step()
        windows = report.forward if phase == "forward" else report.update
        for _ in range(3):
            _sync(device)
            tick = time.perf_counter()
            count = 0
            while count < 1000:
                step()
                _sync(device)
                count += 1
                if time.perf_counter() - tick >= 2:
                    break
            windows.append(
                Window(samples=count * 4, seconds=time.perf_counter() - tick)
            )
            report.sampled_rss_bytes = max(
                report.sampled_rss_bytes, psutil.Process().memory_info().rss
            )
        if device == "mps":
            report.mps_allocated_bytes = torch.mps.current_allocated_memory()
            report.mps_driver_bytes = torch.mps.driver_allocated_memory()
    assert report is not None
    return report


def _fixture(recipe: TrainingRegime, obs: dict[str, torch.Tensor], out: Path) -> None:
    with VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(recipe, 115, out / "run", store)
    verify_fixture(run, obs, out)


def verify_fixture(run: TrainingRun, obs: dict[str, torch.Tensor], out: Path) -> None:
    """Verify retained exports without training again; keep the failed attempt."""
    assert run.status == "completed"
    exposures = sum(stage.optimizer_exposures for stage in run.stages)
    assert exposures > 0
    checkpoint = Path(run.stages[-1].artifacts["raw"]["path"])
    agent, space = load_checkpoint_agent(str(checkpoint))
    torch.manual_seed(run.seed_streams["initialization"])
    initial = Agent(space, run.regime.agent).eval()
    assert any(
        not torch.equal(p, q)
        for p, q in zip(agent.parameters(), initial.parameters(), strict=True)
    )
    saved = torch.load(checkpoint, weights_only=False)
    initial.load_state_dict(saved["model_state_dict"])
    with torch.inference_mode():
        for a, b in zip(agent(obs), initial(obs), strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    atomic_json(
        out / "result.json",
        {
            "run_id": run.id,
            "checkpoint": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "optimizer_exposures": exposures,
            "architecture": architecture_receipt(agent).model_dump(mode="json"),
            "weight_change": True,
            "reload_equal": True,
        },
    )


def _child(root: Path, case: str, mode: str) -> None:
    torch.set_num_threads(1)
    recipe = TrainingRegime.model_validate_json((root / f"{case}.json").read_text())
    obs = torch.load(root / "input.pt", weights_only=True)
    out = root / f"{case}-{mode}"
    out.mkdir()
    if mode == "fixture":
        _fixture(recipe, obs, out)
    else:
        atomic_json(
            out / "result.json", _probe(recipe, obs, mode).model_dump(mode="json")
        )


def scale(out: Path) -> ScaleReport:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = watchdog_seconds()
    report = ScaleReport()
    try:
        torch.set_num_threads(1)
        recipes = scale_recipes()
        first = next(iter(recipes.values()))
        torch.manual_seed(115)
        agent = Agent(ObservationSpace(first.observation), first.agent)
        collector = SeatRoutedCollector(
            agent.observation_space,
            Match(first.match),
            Reward(RewardHypers()),
            num_envs=4,
            seed=115,
        )
        batch = collector.collect(agent, 1)
        obs = {k: torch.from_numpy(v[0]) for k, v in batch.obs.items()}
        torch.save(obs, out / "input.pt")
        report.input_sha256 = file_sha256(out / "input.pt")
        for folder in ("manabot", "experiments/runners"):
            for path in sorted(Path(folder).rglob("*.py")):
                report.source[str(path)] = file_sha256(path)
        for case, recipe in recipes.items():
            atomic_json(out / f"{case}.json", recipe.model_dump(mode="json"))
            for mode in ("fixture", "cpu", "mps"):
                tick = watchdog_seconds()
                remaining = 900 - (tick - start)
                error = None
                status = "completed"
                result = out / f"{case}-{mode}" / "result.json"
                try:
                    if remaining <= 0:
                        raise TimeoutError("total attempt budget exhausted")
                    with (out / f"{case}-{mode}.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "experiments.runners.scale_probe",
                                "--out",
                                str(out),
                                "--case",
                                case,
                                "--mode",
                                mode,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=min(90, remaining),
                            check=True,
                            env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                        )
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    status, error = "failed", str(exc)
                report.attempts.append(
                    Attempt(
                        case=case,
                        mode=mode,
                        status=status,
                        error=error,
                        seconds=watchdog_seconds() - tick,
                        result=str(result) if result.exists() else None,
                    )
                )
                report.elapsed_seconds = watchdog_seconds() - start
                report.load_after = os.getloadavg()
                atomic_json(out / "scale.json", report.model_dump(mode="json"))
        report.status = (
            "completed"
            if all(a.status == "completed" for a in report.attempts)
            else "failed"
        )
        return report
    except BaseException as error:
        report.status = "failed"
        report.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.elapsed_seconds = watchdog_seconds() - start
        report.load_after = os.getloadavg()
        atomic_json(out / "scale.json", report.model_dump(mode="json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--mode", choices=("fixture", "cpu", "mps"))
    args = parser.parse_args()
    if args.case:
        _child(args.out, args.case, args.mode)
    else:
        scale(args.out)


if __name__ == "__main__":
    main()
