"""Attribute ordinary rollout assembly under default/process-disabled huge pages.

The parent owns existing hard bounds and samples Linux counters. The child sets
only its own THP policy before ML imports, then runs the shared phase worker.
AST wrappers preserve the production assembly statements and add diagnostic
timers; no independently maintained assembly algorithm is used.
"""

import argparse
import ast
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import textwrap
import time
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, "scratch/" + filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def optional(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def telemetry(pid: int) -> dict[str, object]:
    raw = optional(f"/proc/{pid}/stat")
    process = None
    if raw is not None:
        values = raw.rsplit(")", 1)[1].split()
        process = {
            "state": values[0],
            "minor_faults": int(values[7]),
            "major_faults": int(values[9]),
            "user_ticks": int(values[11]),
            "system_ticks": int(values[12]),
            "ticks_per_second": os.sysconf("SC_CLK_TCK"),
        }
    vm = optional("/proc/vmstat")
    return {
        "process": process,
        "cgroup_memory_pressure": optional("/sys/fs/cgroup/memory.pressure"),
        "cgroup_cpu_stat": optional("/sys/fs/cgroup/cpu.stat"),
        "global_vm": {
            key: int(value)
            for key, value in (line.split() for line in (vm or "").splitlines())
            if key.startswith(("compact_", "thp_", "pgscan_", "pgsteal_", "allocstall"))
        },
    }


def process_policy(policy: str) -> dict[str, object]:
    # Linux uapi PR_SET/GET_THP_DISABLE. No machine-global state is changed.
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    before = prctl(42, 0, 0, 0, 0)
    if before < 0:
        raise OSError(ctypes.get_errno(), "PR_GET_THP_DISABLE failed")
    if policy == "disabled" and prctl(41, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
    after = prctl(42, 0, 0, 0, 0)
    if after < 0 or (policy == "disabled" and after != 1):
        raise RuntimeError("process THP policy readback failed")
    return {
        "requested": policy,
        "before": before,
        "after": after,
        "sysfs": {
            name: optional("/sys/kernel/mm/transparent_hugepage/" + name)
            for name in ("enabled", "defrag", "hpage_pmd_size", "use_zero_page")
        },
        "process_status": optional("/proc/self/status"),
        "meaning": "Inherited control or process-only disable before ML imports.",
    }


@dataclass
class Cost:
    calls: int = 0
    seconds: float = 0.0
    user_cpu: float = 0.0
    system_cpu: float = 0.0
    minor_faults: int = 0
    major_faults: int = 0


class AssemblyMeter:
    def __init__(self, trace: Any, ordinal: int) -> None:
        # Shared phase-worker Trace is loaded dynamically at this boundary.
        self.trace = trace
        self.path = trace.out / f"assembly-{ordinal:02d}.json"
        self.stack = list(trace.stack)
        self.costs = {name: Cost() for name in ("copy", "release", "bootstrap")}
        self.rows = 0
        self.save(False)

    def save(self, complete: bool) -> None:
        value = {
            "stack": self.stack,
            "completed_rows": self.rows,
            "complete": complete,
            "costs": {key: asdict(cost) for key, cost in self.costs.items()},
            "meaning": "Aggregates exclude timer/receipt overhead; partial rows may be missing.",
        }
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.path)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_SELF)
        try:
            yield
        finally:
            after = resource.getrusage(resource.RUSAGE_SELF)
            elapsed = time.monotonic() - start
            cost = self.costs[name]
            cost.calls += 1
            cost.seconds += elapsed
            cost.user_cpu += after.ru_utime - before.ru_utime
            cost.system_cpu += after.ru_stime - before.ru_stime
            cost.minor_faults += after.ru_minflt - before.ru_minflt
            cost.major_faults += after.ru_majflt - before.ru_majflt
            if name == "bootstrap":
                self.rows += 1
                if self.rows % 128 == 0:
                    self.save(False)


def wrapped(expression: str, body: list[ast.stmt]) -> ast.With:
    context = ast.parse(expression, mode="eval").body
    return ast.With(items=[ast.withitem(context_expr=context)], body=body)


def assembly_builder(
    original: FunctionType, trace_getter: Callable[[], Any], out: Path
) -> FunctionType:
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    loops = [i for i, stmt in enumerate(function.body) if isinstance(stmt, ast.For)]
    assert len(loops) == 1 and loops[0] == len(function.body) - 2
    loop_index = loops[0]
    loop = function.body[loop_index]
    assert isinstance(loop, ast.For) and isinstance(function.body[-1], ast.Return)
    assert ast.unparse(loop.target) == "env_index"
    assert ast.unparse(loop.iter) == "range(num_envs)"
    deletion = [i for i, stmt in enumerate(loop.body) if isinstance(stmt, ast.Delete)]
    assert deletion == [2] and len(loop.body) == 6
    assert ast.unparse(loop.body[2]) == "del stream[:num_steps]"
    assert isinstance(loop.body[1], ast.For) and isinstance(loop.body[3], ast.Assert)
    loop.body = [
        wrapped("__assembly.stage('copy')", loop.body[:2]),
        wrapped("__assembly.stage('release')", loop.body[2:4]),
        wrapped("__assembly.stage('bootstrap')", loop.body[4:]),
    ]
    function.body = [
        *ast.parse("__assembly = __assembly_factory()").body,
        wrapped(
            "__assembly.trace.span('assembly_allocate')", function.body[:loop_index]
        ),
        wrapped("__assembly.trace.span('assembly_rows')", [loop]),
        *ast.parse("__assembly.save(True)").body,
        function.body[-1],
    ]
    ast.fix_missing_locations(tree)
    ordinal = 0

    def factory() -> AssemblyMeter:
        nonlocal ordinal
        ordinal += 1
        return AssemblyMeter(trace_getter(), ordinal)

    namespace = dict(original.__globals__)
    namespace["__assembly_factory"] = factory
    # Compilation preserves production annotations/imports without evaluating them.
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(tree),
            "<instrumented-ordinary-assembly>",
            "exec",
        ),
        namespace,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "assembly-method.json").write_text(
        json.dumps(
            {
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "original": source,
                "instrumented": ast.unparse(tree),
            },
            indent=2,
        )
        + "\n"
    )
    return namespace[original.__name__]


def child(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    policy = process_policy(args.policy)
    (args.out / "thp-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    worker = load("assembly_phase_worker", "collector-phase-worker.py")
    trace_instance: Any = None
    base_trace = worker.Trace

    class Trace(base_trace):
        def __init__(self, out: Path) -> None:
            nonlocal trace_instance
            super().__init__(out)
            trace_instance = self

    worker.Trace = Trace
    original = worker.SeatRoutedCollector._build_batch
    worker.SeatRoutedCollector._build_batch = assembly_builder(
        original, lambda: trace_instance, args.out
    )
    sys.argv = [
        "assembly-child",
        "--root",
        str(args.root),
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
        "--variant",
        "reference",
    ]
    worker.main()


def check(out: Path) -> None:
    from copy import deepcopy

    import numpy as np

    from manabot.sim.net_opponent import SeatRoutedCollector, _Pending, _Transition
    from manabot.sim.rollout import OBS_KEYS

    worker = load("assembly_check_worker", "collector-phase-worker.py")
    out.mkdir(parents=True, exist_ok=True)
    trace = worker.Trace(out)
    original = SeatRoutedCollector._build_batch
    instrumented = assembly_builder(original, lambda: trace, out)
    for categorical in (False, True):
        collector = object.__new__(SeatRoutedCollector)
        collector.num_envs = 2
        shapes = {key: np.zeros((1, 3), dtype=np.float32) for key in OBS_KEYS}
        collector.observation_space = SimpleNamespace(
            encoder=SimpleNamespace(allocate=lambda count: shapes)
        )
        collector._buffers = {
            key: np.full((2, 3), 7, dtype=value.dtype) for key, value in shapes.items()
        }
        collector._pending = [None, None]
        collector._streams = [
            [
                _Transition(
                    _Pending(
                        obs={
                            key: np.full(3, row + step, dtype=value.dtype)
                            for key, value in shapes.items()
                        },
                        action=step,
                        logprob=-1.0,
                        value=0.25,
                        probabilities=np.array([0.2, 0.3, 0.5], dtype=np.float32),
                        log_probabilities=np.array(
                            [-1.6, -1.2, -0.7], dtype=np.float32
                        ),
                        outcome_probabilities=np.array(
                            [0.2, 0.3, 0.5], dtype=np.float32
                        )
                        if categorical
                        else None,
                    ),
                    reward=float(row),
                    done=bool(step),
                )
                for step in range(2)
            ]
            for row in range(2)
        ]
        candidate = deepcopy(collector)
        expected, actual = original(collector, 2), instrumented(candidate, 2)
        for field in fields(expected):
            first, second = getattr(expected, field.name), getattr(actual, field.name)
            if isinstance(first, dict):
                assert first.keys() == second.keys()
                for key in first:
                    np.testing.assert_array_equal(first[key], second[key])
            elif first is None:
                assert second is None
            else:
                np.testing.assert_array_equal(first, second)
        assert candidate._streams == collector._streams == [[], []]
        for key in actual.obs:
            assert not np.shares_memory(actual.obs[key], candidate._buffers[key])
    trace.events.close()
    print("Instrumented ordinary assembly matches scalar and categorical fixtures.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(10349, 10350))
    parser.add_argument("--policy", choices=("default", "disabled"))
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(args.out)
        return
    assert args.root is not None and args.seed is not None and args.policy is not None
    if args.child:
        child(args)
        return
    supervisor = load("assembly_supervisor", "collector-phase-supervisor.py")
    report = supervisor.supervise(
        [
            sys.executable,
            __file__,
            "--child",
            "--root",
            str(args.root),
            "--out",
            str(args.out),
            "--seed",
            str(args.seed),
            "--policy",
            args.policy,
        ],
        args.out,
        sample=telemetry,
    )
    print(
        json.dumps({key: report[key] for key in ("status", "seconds", "expired_call")}),
        flush=True,
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
