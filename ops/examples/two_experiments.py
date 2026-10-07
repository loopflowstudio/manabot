"""Worked example: two seeds at one pinned commit on one bounded RunPod rental.

This deliberately uses the existing deployment internals rather than adding a
reusable-machine API. Run from the repository root after committing and pushing.
The fixed acceptance ledger retains every attempt; the smaller $4.90 ceiling
remains in force. The startup guardian deletes the pod at the original deadline.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import signal
import subprocess
import time

from manabot.remote.bundle import Bundle, verify_training_bundle
from manabot.remote.deploy import (
    Receipt,
    _create,
    _estimate_cost,
    _ready,
    _save_cleanup,
    _terminate_signal,
    confirm_delete,
    current_source,
    deployment_lock,
    save,
    verify_public_source,
)
from manabot.remote.plan import DeploymentPlan, HardwareMix, compile_plan, digest
from manabot.remote.provider import RunPod
from manabot.remote.transport import REPO_DIR, Transport, bootstrap
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun


@dataclass
class ExperimentTiming:
    seed: int
    source_commit: str
    setup_seconds: float
    idle_before_seconds: float
    command_seconds: float
    transfer_seconds: float
    training_seconds: float


def _experiment(
    transport: Transport, plan: DeploymentPlan, out: Path, index: int
) -> ExperimentTiming:
    """A fresh TrainingRun and database per seed; no sync/build in either command."""
    seed = plan.seed + index
    directory = out / f"experiment-{index}"
    directory.mkdir()
    remote = f"/workspace/experiment-{index}"
    began = time.perf_counter()
    transport.shell(f"""set -eu
export PATH=/root/.local/bin:/root/.cargo/bin:$PATH
cd {REPO_DIR}
test "$(git rev-parse HEAD)" = {plan.source.commit}
test "$(git rev-parse 'HEAD^{{tree}}')" = {plan.source.tree}
mkdir -p {remote}/evidence
set +e
uv run --no-sync manabot train --regime /workspace/regime.json --seed {seed} --out {remote}/evidence/run > {remote}/evidence/training.log 2>&1
status=$?
set -e
printf '%s\\n' "$status" > {remote}/evidence/training-exit.txt
uv run --no-sync python -m manabot.remote.bundle {remote}/evidence
""")
    command_seconds = time.perf_counter() - began
    began = time.perf_counter()
    transport.get(f"{remote}/bundle.json", directory / "bundle.json")
    bundle = Bundle.model_validate_json((directory / "bundle.json").read_text())
    evidence = directory / "evidence"
    evidence.mkdir()
    for item in bundle.files:
        target = item.destination(evidence)
        target.parent.mkdir(parents=True, exist_ok=True)
        transport.get(f"{remote}/evidence/{item.relative_path}", target)
    verify_training_bundle(evidence, bundle)
    run = TrainingRun.model_validate_json((evidence / "run/run.json").read_text())
    if not all(
        s.actual_device.startswith("cuda") and s.optimizer_exposures for s in run.stages
    ):
        raise ValueError("example did not demonstrate CUDA optimizer work")
    return ExperimentTiming(
        seed,
        plan.source.commit,
        0,
        0,
        command_seconds,
        time.perf_counter() - began,
        run.seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    ledger = root / ".runs/remote-acceptance"
    ledger.mkdir(parents=True, exist_ok=True)
    previous = sorted(ledger.glob("attempt-*/deployment.json"))
    spent = 0.0
    for path in previous:
        receipt = Receipt.model_validate_json(path.read_text())
        if receipt.phase != "deleted" or receipt.estimated_dollars is None:
            raise ValueError("prior deletion/cost unresolved")
        spent += receipt.estimated_dollars
    data = HardwareMix.model_validate_json(args.mix.read_text()).model_dump()
    data["dollar_cap"] = min(data["dollar_cap"], 4.90 - spent)
    plan = compile_plan(
        args.regime.read_text(), HardwareMix(**data), current_source(root), 197
    )
    # Both runs and transfers must fit the original hard deadline; never renew it.
    required = (
        plan.mix.setup_seconds
        + 2 * (plan.regime.wall_seconds + plan.mix.transfer_seconds)
        + plan.mix.cleanup_seconds
    )
    if required > plan.mix.wall_seconds:
        raise ValueError("two experiments and reserves exceed the rental allowance")
    verify_public_source(plan.source)
    provider = RunPod()
    with deployment_lock(Path.home() / ".cache/manabot"):
        if provider.list():
            raise ValueError("example requires an initially empty inventory")
        out = ledger / f"attempt-{len(previous):03d}"
        out.mkdir(mode=0o700)
        path = out / "deployment.json"
        start = time.time()
        receipt = Receipt(
            plan_sha256=digest(plan.model_dump_json().encode()),
            started=start,
            deadline=start + plan.mix.wall_seconds,
        )
        save(path, receipt)
        (out / "plan.json").write_text(plan.model_dump_json(indent=2))
        recipe = out / "regime.json"
        recipe.write_text(plan.regime.model_dump_json(indent=2))
        identity = out / "ssh_key"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "manabot-example",
                "-f",
                str(identity),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        previous_signal = signal.signal(signal.SIGTERM, _terminate_signal)
        timings: list[ExperimentTiming] = []
        try:
            deadline = receipt.deadline - plan.mix.cleanup_seconds
            pod = _create(
                provider,
                plan,
                receipt,
                path,
                "training",
                deadline,
                identity.with_suffix(".pub").read_text().strip(),
            )
            transport = _ready(
                provider, pod, out, start + plan.mix.setup_seconds, identity
            )
            receipt.phase = "bootstrap"
            save(path, receipt)
            transport.shell(bootstrap(plan))
            transport.put(recipe, "/workspace/regime.json")
            transport.get(
                "/workspace/evidence/bootstrap-timing.json",
                out / "bootstrap-timing.json",
            )
            setup_seconds = time.time() - start
            last_finished = time.time()
            for index in range(2):
                receipt.phase = f"experiment-{index}"
                save(path, receipt)
                idle = time.time() - last_finished
                # Reserve the second run, its retrieval and cleanup while first runs.
                transport.deadline = deadline - (1 - index) * (
                    plan.regime.wall_seconds + plan.mix.transfer_seconds
                )
                timing = _experiment(transport, plan, out, index)
                timing.setup_seconds = setup_seconds if index == 0 else 0
                timing.idle_before_seconds = idle
                timings.append(timing)
                atomic_json(out / "experiments.json", [asdict(t) for t in timings])
                last_finished = time.time()
            receipt.complete = True
        except BaseException as error:
            receipt.error = type(error).__name__
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            until = time.time() + plan.mix.cleanup_seconds
            settled = all(
                confirm_delete(provider, a, until)
                for a in receipt.attempts
                if a.deleted_time is None
            )
            receipt.phase = "deleted" if settled else "cleanup-unconfirmed"
            receipt.complete = receipt.complete and settled
            receipt.estimated_dollars = _estimate_cost(receipt, plan)
            _save_cleanup(path, receipt)
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_signal)
            if not settled:
                raise RuntimeError(
                    f"CLEANUP UNCONFIRMED; run uv run manabot deploy cleanup --deployment {path}"
                )
        if provider.list():
            raise ValueError("final inventory is not empty")
        atomic_json(
            out / "example.json",
            {
                "inventory_pods": 0,
                "estimated_dollars": receipt.estimated_dollars,
                "prior_estimated_dollars": spent,
                "source_commit": plan.source.commit,
                "experiments": len(timings),
            },
        )
        print(
            f"Two experiments returned; no pods; estimated rental ${receipt.estimated_dollars:.4f}"
        )


if __name__ == "__main__":
    main()
