"""Run independent evaluation CLIs under one deadline and failure boundary.

Threads only supervise subprocesses; policies and game RNGs remain isolated in
their ordinary CLI processes. Each command owns its logs and Worker receipt.
The caller supplies the CPU concurrency allowance and the outer rental deadline.
"""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
import re

from experiments.runners.gpu_job import Worker


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]
    seconds: float


def run_parallel(worker: Worker, commands: tuple[Command, ...], workers: int) -> int:
    """Stop active children and admit no more commands after a failure or stop.

    Worker.command owns process-group termination and the retention reserve.
    Waiting for its supervisors on exit ensures no child is left executing after
    this function returns, including when subprocess admission raises.
    """
    if isinstance(workers, bool) or not 1 <= workers <= 4:
        raise ValueError("Parallel command allowance must be between one and four")
    if not commands or len({c.name for c in commands}) != len(commands):
        raise ValueError("Commands must be nonempty and have unique names")
    for command in commands:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", command.name):
            raise ValueError("Command names must be safe directory names")
        if (
            not command.argv
            or not math.isfinite(command.seconds)
            or command.seconds <= 0
        ):
            raise ValueError("Each command requires argv and a positive finite limit")
    pending = deque(commands)
    active: dict[Future[int], tuple[str, Worker]] = {}
    result = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while pending or active:
                if worker.stopping and result == 0:
                    result = 124
                if result:
                    pending.clear()
                    for _, child in active.values():
                        child.stopping = True
                while pending and len(active) < workers and not result:
                    command = pending.popleft()
                    directory = worker.output / "commands" / command.name
                    directory.mkdir(parents=True, exist_ok=False)
                    child = Worker(directory, worker.deadlines)
                    future = pool.submit(
                        child.command, "run", list(command.argv), command.seconds
                    )
                    active[future] = (command.name, child)
                    worker.phase(
                        "parallel_commands",
                        active=sorted(name for name, _ in active.values()),
                        pending=len(pending),
                    )
                if not active:
                    break
                done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    name, _ = active.pop(future)
                    code = future.result()
                    if code and not result:
                        result = code
                    worker.phase("command_finished", command=name, exit_code=code)
        finally:
            for _, child in active.values():
                child.stopping = True
    return result
