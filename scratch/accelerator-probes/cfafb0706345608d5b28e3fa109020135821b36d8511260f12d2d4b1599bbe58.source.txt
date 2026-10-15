"""Bounded admission/regression checks and complete-service learner comparisons."""

from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys
import tempfile

from experiments.runners.pipeline_benchmark import Replay, Tape
from experiments.runners.serving_cost import ServiceMeter, load_serving


def rejects(action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("mismatched serving evidence accepted")


def cli(path: str, args: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [path, *args]
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = previous


def main() -> None:
    root = Path("scratch/accelerator-probes/serving-integration")
    root.mkdir(exist_ok=False)
    tape_path = Path("scratch/accelerator-probes/cohort-t8-s64/tape.json")
    tape = Tape.model_validate_json(tape_path.read_text())
    replay = Replay.model_validate(
        json.loads((tape_path.parent / "replay.json").read_text())["samples"][1]
    )
    bundle = Path("scratch/gpu-probes/serving-007")
    model_path = bundle / "service-model.json"
    hardware = bundle / "capacity-report.json"
    payload = (
        replay.demand.input_bytes + replay.demand.output_bytes
    ) / replay.demand.decisions
    serving = load_serving(model_path, hardware, "h100", tape.recipe, payload)
    rejects(lambda: load_serving(model_path, hardware, "l4", tape.recipe, payload))
    rejects(
        lambda: load_serving(model_path, hardware, "h100", tape.recipe, payload + 1)
    )
    rejects(
        lambda: load_serving(
            model_path,
            Path("scratch/gpu-probes/h100-005/capacity-report.json"),
            "h100",
            tape.recipe,
            payload,
        )
    )
    changed = tape.recipe.model_copy(deep=True)
    changed.agent.hidden_dim += 1
    rejects(lambda: load_serving(model_path, hardware, "h100", changed, payload))
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "model.json"
        record = json.loads(model_path.read_text())
        record["overhead_source_sha256"] = "0" * 64
        bad.write_text(json.dumps(record))
        rejects(lambda: load_serving(bad, hardware, "h100", tape.recipe, payload))
    queue = runpy.run_path("scratch/accelerator-queue-probe.py")
    queue["check"]()
    summary = json.loads((bundle / "summary.json").read_text())
    for case in summary["cells"]:
        meter = ServiceMeter(serving.model)
        outcome = queue["simulate"](
            [f.rows for f in tape.frames],
            0,
            serving.model.forward_seconds,
            payload,
            queue["Settings"](
                actors=case["actors"],
                cpu_workers=min(case["actors"], 8),
                gpus=1,
                max_batch=case["max_batch"],
                wait_seconds=0.00025,
            ),
            request_host_seconds=replay.request_host_seconds,
            tail_host_seconds=replay.tail_host_seconds,
            batch_service=meter,
        )
        assert outcome.seconds == case["predicted_seconds"]
        assert meter.usage.calls == sum(outcome.padded_batch_counts.values())
    # Explicit optimizer reserve is additional service, without copying the
    # inference cost or the calibrated links a second time.
    demand = queue["BatchDemand"](64, 40, 2)
    reserved = ServiceMeter(serving.model, {64: 0.1})
    assert abs(reserved(demand) - serving.model.duration(demand) - 0.1) < 1e-12
    one = ServiceMeter(serving.model)
    one(queue["BatchDemand"](1, 1, 9))
    assert asdict(one.usage) == dict(
        calls=1, outside_padded_row_range=1, outside_request_range=1, maximum_requests=9
    )
    script = "scratch/cohort-flow-probe.py"
    cli(script, ["--out", str(root / "l4-control.json")])
    original = json.loads(
        Path("scratch/accelerator-probes/cohort-flow-detail.json").read_text()
    )
    assert (
        json.loads((root / "l4-control.json").read_text())["outcomes"]
        == original["outcomes"]
    )
    for streams in (512, 4096):
        common = [
            "--card",
            "h100",
            "--hardware",
            str(hardware),
            "--learner-service",
            "scratch/gpu-probes/h100-005/learner-service.json",
            "--global-streams",
            str(streams),
        ]
        cli(script, [*common, "--out", str(root / f"h100-s{streams}-control.json")])
        cli(
            script,
            [
                *common,
                "--serving-costs",
                str(model_path),
                "--out",
                str(root / f"h100-s{streams}-complete.json"),
            ],
        )
    (root / "checks.json").write_text(
        json.dumps(
            dict(
                admission_mismatch_checks=5,
                prospective_predictions_exact=9,
                l4_outcomes_exact=27,
                complete_and_control_cohort_cases=108,
                extra_optimizer_cost_and_extrapolation=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Admission guards, nine prospective predictions,27L4 outcomes,108paired cohort cases passed"
    )


if __name__ == "__main__":
    main()
