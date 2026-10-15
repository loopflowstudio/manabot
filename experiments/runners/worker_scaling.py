"""Measure fixed resident replay work across persistent CPU worker processes.

The supervisor owns a process group and bounds its complete lifetime and summed
RSS. The controller coordinates prepare/collect/verify barriers; workers retain
all actors and rollouts until verification finishes. Collection wall time includes
IPC dispatch and the slowest worker. All other phases remain in the receipt.

Modes use recorded outputs, a shared CUDA service, or worker-local CUDA models.
No optimizer runs inside this curve. Repeated actors share one frozen trajectory
intentionally. Run through uv; see experiments/protocols/.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import resource
import selectors
import signal
import subprocess
import sys
import time
from typing import IO, TypeAlias

import psutil

Json: TypeAlias = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
Payload: TypeAlias = dict[str, Json]
RequestHandler: TypeAlias = Callable[[int, int, int], None]


def write_json(path: Path, value: Payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_payload(line: str) -> Payload:
    # Messages are produced exclusively by this runner's owned local workers.
    value: object = json.loads(line)
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("worker message must be a JSON object")
    return value


def worker(args: argparse.Namespace) -> None:
    import torch

    from experiments.runners.pipeline_benchmark import PreparedReplay, Tape
    from experiments.runners.worker_inference import CudaPolicy

    torch.set_num_threads(1)
    forward = None
    policy: CudaPolicy | None = None
    if args.device == "cuda-local":
        tape = Tape.model_validate_json((args.tape / "tape.json").read_text())
        policy = CudaPolicy(tape)
        forward = policy.forward
    if args.exchange is not None:
        from experiments.runners.worker_inference import Exchange

        forward = Exchange(args.exchange / str(args.worker)).client(args.worker)
    actors: list[PreparedReplay] = []
    for line in sys.stdin:
        command = read_payload(line)
        stage = command["stage"]
        started = time.perf_counter()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        inference_calls = policy.calls if policy is not None else 0
        inference_seconds = policy.seconds if policy is not None else 0.0
        result: Payload = {"worker": args.worker, "stage": stage}
        if stage == "prepare":
            actors.clear()
            count = command["count"]
            if not isinstance(count, int) or not 0 <= count <= 64:
                raise ValueError("invalid actor count")
            if count:
                audited = PreparedReplay(args.tape, verify_inputs=True)
                audited.collect(forward)
                result["audit"] = audited.verify().model_dump(mode="json")
                del audited
            actors = [
                PreparedReplay(args.tape, verify_inputs=False) for _ in range(count)
            ]
            result["count"] = count
        elif stage == "collect":
            for actor in actors:
                actor.collect(forward)
            result["count"] = len(actors)
        elif stage == "verify":
            result["actors"] = [a.verify().model_dump(mode="json") for a in actors]
        elif stage == "release":
            actors.clear()
        elif stage == "stop":
            return
        else:
            raise ValueError(f"unknown worker stage {stage}")
        finished = resource.getrusage(resource.RUSAGE_SELF)
        result.update(
            seconds=time.perf_counter() - started,
            user_seconds=finished.ru_utime - cpu.ru_utime,
            system_seconds=finished.ru_stime - cpu.ru_stime,
            rss_bytes=psutil.Process().memory_info().rss,
        )
        if policy is not None:
            result.update(
                inference_calls=policy.calls - inference_calls,
                inference_seconds=policy.seconds - inference_seconds,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        print(json.dumps(result), flush=True)


@dataclass
class Worker:
    process: subprocess.Popen[str]
    stdin: IO[str]
    stdout: IO[str]


def barrier(
    workers: list[Worker],
    commands: list[Payload],
    serve: RequestHandler | None = None,
) -> Payload:
    """Dispatch every worker before waiting; reject deaths and stage mismatch."""
    started = time.perf_counter()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    results: list[Json] = []
    requests = 0
    service_seconds = 0.0
    with selectors.DefaultSelector() as selector:
        for index, (worker, command) in enumerate(zip(workers, commands, strict=True)):
            worker.stdin.write(json.dumps(command) + "\n")
            worker.stdin.flush()
            selector.register(worker.stdout, selectors.EVENT_READ, index)
        while selector.get_map():
            for key, _ in selector.select(timeout=0.1):
                index = key.data
                line = workers[index].stdout.readline()
                if not line:
                    raise RuntimeError(f"worker {index} exited before barrier")
                result = read_payload(line)
                if result.get("kind") == "inference":
                    rows, sequence = result.get("rows"), result.get("sequence")
                    if (
                        serve is None
                        or result.get("worker") != index
                        or not isinstance(rows, int)
                        or not isinstance(sequence, int)
                    ):
                        raise ValueError("invalid inference request")
                    before = time.perf_counter()
                    serve(index, rows, sequence)
                    service_seconds += time.perf_counter() - before
                    requests += 1
                    workers[index].stdin.write(
                        json.dumps(
                            {
                                "kind": "inference_done",
                                "sequence": sequence,
                            }
                        )
                        + "\n"
                    )
                    workers[index].stdin.flush()
                    continue
                if (
                    result.get("worker") != index
                    or result.get("stage") != commands[index]["stage"]
                ):
                    raise ValueError("worker response does not match barrier")
                results.append(result)
                selector.unregister(key.fileobj)
    finished = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "seconds": time.perf_counter() - started,
        "controller_user_seconds": finished.ru_utime - cpu.ru_utime,
        "controller_system_seconds": finished.ru_stime - cpu.ru_stime,
        "workers": results,
        "requests": requests,
        "service_seconds": service_seconds,
    }


def controller(args: argparse.Namespace) -> None:
    import torch

    from experiments.runners.pipeline_benchmark import (
        PreparedReplay,
        Tape,
        _source,
        capture,
        file_sha256,
    )
    import managym

    torch.set_num_threads(1)
    tape = Tape.model_validate_json((args.tape / "tape.json").read_text())
    serve: RequestHandler | None = None
    hardware: Payload = {"mode": "recorded-outputs"}
    if args.capture_replay:
        # Portability is a separate check, never a relaxation of worker parity.
        # Every worker below must match a fresh capture on this exact host.
        portability: Payload = {
            "original_tape_sha256": file_sha256(args.tape / "tape.json")
        }
        try:
            audited = PreparedReplay(args.tape, verify_inputs=True)
            audited.collect()
            portability["result"] = audited.verify().model_dump(mode="json")
            portability["status"] = "matched"
        except ValueError as error:
            portability.update(status="failed", error=str(error))
        write_json(args.out / "portability.json", portability)
        args.tape = args.out / "host-tape"
        args.tape.mkdir()
        tape = capture(
            tape.recipe, args.tape, tape.streams, tape.transitions, tape.seed
        )
        hardware["capture"] = (
            "fresh CPU policy on this host; original portability audited separately"
        )
    if args.device in {"cuda", "cuda-local"}:
        from experiments.runners.worker_inference import CudaService

        service = CudaService(args.tape, args.out)
        args.tape, args.exchange = service.tape_dir, service.exchange_dir
        tape = Tape.model_validate_json((args.tape / "tape.json").read_text())
        serve = service.handle
        hardware = dict(service.hardware)
        if args.profile:
            from experiments.runners.worker_service_profile import profile_service

            profile_service(service, args.out)
            return
        if args.device == "cuda-local":
            args.exchange = None
            serve = None
            hardware.update(
                mode="worker-local-cuda",
                ownership="one model per persistent worker on the same GPU; controller reference model remains resident",
            )
    write_json(
        args.out / "identity.json",
        {
            "tape_sha256": file_sha256(args.tape / "tape.json"),
            "outputs_sha256": tape.outputs_sha256,
            "native_sha256": file_sha256(Path(managym._managym.__file__)),
            "source": _source(),
            "runner_sha256": file_sha256(Path(__file__)),
            "actors": args.actors,
            "streams": tape.streams * args.actors,
            "learner_rows": tape.streams * tape.transitions * args.actors,
            "decisions": sum(f.rows for f in tape.frames) * args.actors,
            "initialization": tape.initialization,
            "hardware": hardware,
            "replication": "All actors repeat the same frozen tape; not independent games",
        },
    )
    workers: list[Worker] = []
    for index in range(4):
        exchange_args = (
            [] if args.exchange is None else ["--exchange", str(args.exchange)]
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.runners.worker_scaling",
                "--worker",
                str(index),
                "--tape",
                str(args.tape),
                "--out",
                str(args.out),
                "--device",
                args.device,
                *exchange_args,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        workers.append(Worker(process, process.stdin, process.stdout))
    try:
        for repeat in range(args.repeats):
            order = (1, 2, 4) if repeat % 2 == 0 else (4, 2, 1)
            for active in order:
                stem = args.out / f"r{repeat}-w{active}"
                preparation = barrier(
                    workers,
                    [
                        {
                            "stage": "prepare",
                            "count": args.actors // active if i < active else 0,
                        }
                        for i in range(4)
                    ],
                    serve,
                )
                write_json(stem.with_suffix(".prepare.json"), preparation)
                collected = barrier(
                    workers, [{"stage": "collect"} for _ in workers], serve
                )
                if args.device in {"cuda", "cuda-local"}:
                    collected.update(
                        controller_gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        controller_gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                write_json(stem.with_suffix(".collect.json"), collected)
                verified = barrier(workers, [{"stage": "verify"} for _ in workers])
                write_json(stem.with_suffix(".verify.json"), verified)
                released = barrier(workers, [{"stage": "release"} for _ in workers])
                write_json(stem.with_suffix(".release.json"), released)
    finally:
        for worker in workers:
            try:
                worker.stdin.write('{"stage":"stop"}\n')
                worker.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        for worker in workers:
            worker.process.wait()


def supervise(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    receipt: Payload = {
        "status": "running",
        "platform": platform.platform(),
        "seconds_limit": args.seconds,
        "rss_limit_bytes": args.rss_gib * 1024**3,
        "load_start": list(os.getloadavg()),
        "threads_per_worker": 1,
        "sampling": "sum of process-tree RSS; shared pages may be counted repeatedly",
    }
    write_json(args.out / "attempt.json", receipt)
    samples: list[Json] = []
    with (
        (args.out / "stdout.log").open("w") as stdout,
        (args.out / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.runners.worker_scaling",
                *sys.argv[1:],
                "--controller",
            ],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                WANDB_MODE="disabled",
            ),
        )
        status = "completed"
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                used = 0
                try:
                    root = psutil.Process(process.pid)
                    for child in [root, *root.children(recursive=True)]:
                        try:
                            used += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                samples.append({"seconds": elapsed, "rss_bytes": used})
                if elapsed >= args.seconds or used > args.rss_gib * 1024**3:
                    status = "timeout" if elapsed >= args.seconds else "memory_limit"
                    break
                time.sleep(0.02)
        finally:
            # Kill the owned group even if its controller died first: descendants
            # must not survive either a failed controller or a deadline.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
            receipt.update(
                status=status if status != "completed" or code == 0 else "failed",
                exit_code=code,
                seconds=time.monotonic() - started,
                load_end=list(os.getloadavg()),
                samples=samples,
            )
            write_json(args.out / "attempt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "samples"}))
    if code != 0 or status != "completed":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actors", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=14.5)
    parser.add_argument("--rss-gib", type=float, default=6)
    parser.add_argument(
        "--device", choices=["replay", "cuda", "cuda-local"], default="replay"
    )
    parser.add_argument(
        "--capture-replay",
        action="store_true",
        help="audit the supplied tape, then capture a fresh host CPU tape",
    )
    parser.add_argument("--exchange", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="attribute one shared service separately from worker timing",
    )
    parser.add_argument("--controller", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture_replay and args.device != "replay":
        parser.error("--capture-replay requires --device replay")
    if args.profile and args.device != "cuda":
        parser.error("--profile requires --device cuda")
    if (
        args.actors < 4
        or args.actors > 64
        or args.actors % 4
        or not 1 <= args.repeats <= 3
    ):
        parser.error("actors must be divisible by4 in4..64; repeats1..3")
    if not 0 < args.seconds <= 600 or not 0 < args.rss_gib <= 32:
        parser.error("seconds must be in(0,600]; RSS in(0,32]GiB")
    if args.worker is not None:
        worker(args)
    elif args.controller:
        controller(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
