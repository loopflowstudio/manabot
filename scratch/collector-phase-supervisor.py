"""Externally bound diagnostic CUDA children, including assembly and synchronization.

This parent imports only the standard library. The worker publishes cumulative
atomic call boundaries; the parent kills its process group if any active call
exceeds its allowance. Receipts retain RSS samples and the last trace event.
"""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


@dataclass
class Boundary:
    name: str
    start: float
    end: float | None = None


@dataclass
class MemorySample:
    monotonic: float
    rss_bytes: int | None


def atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def boundaries(path: Path) -> list[Boundary]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("invalid call boundaries")
    result = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("invalid call boundary")
        start = float(item["start"])
        end = float(item["end"]) if item["end"] is not None else None
        if end is not None and end < start:
            raise ValueError("call end precedes start")
        result.append(Boundary(item["name"], start, end))
    return result


def rss(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def supervise(
    command: list[str],
    out: Path,
    *,
    seconds: float = 85,
    call_seconds: float = 25,
    rss_limit: int = 12 * 1024**3,
    poll_seconds: float = 0.05,
    sample: Callable[[int], dict[str, object]] | None = None,
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    memory: list[MemorySample] = []
    telemetry: list[dict[str, object]] = []
    reason: str | None = None
    expired: str | None = None
    observed_overshoot: float | None = None
    with (out / "child.log").open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                now = time.monotonic()
                value = rss(child.pid)
                memory.append(MemorySample(now, value))
                if sample is not None:
                    telemetry.append({"monotonic": now, **sample(child.pid)})
                calls = boundaries(out / "bounds.json")
                violation = next(
                    (
                        b
                        for b in calls
                        if (now if b.end is None else b.end) - b.start > call_seconds
                    ),
                    None,
                )
                if violation is not None:
                    reason = "call_deadline"
                    expired = violation.name
                    observed_overshoot = (
                        (now if violation.end is None else violation.end)
                        - violation.start
                        - call_seconds
                    )
                elif now - started >= seconds:
                    reason = "child_deadline"
                    observed_overshoot = now - started - seconds
                elif value is not None and value > rss_limit:
                    reason = "rss_limit"
                if reason is not None or child.poll() is not None:
                    break
                time.sleep(poll_seconds)
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=2)
    last_event = None
    incomplete_tail_lines = 0
    trace = out / "events.jsonl"
    if trace.exists():
        lines = trace.read_text().splitlines()
        for line in reversed(lines):
            try:
                last_event = json.loads(line)
                break
            except json.JSONDecodeError:
                incomplete_tail_lines += 1
    report: dict[str, object] = {
        "status": reason or ("completed" if child.returncode == 0 else "failed"),
        "exit_code": child.returncode,
        "seconds": time.monotonic() - started,
        "call_limit_seconds": call_seconds,
        "child_limit_seconds": seconds,
        "expired_call": expired,
        "observed_overshoot_seconds": observed_overshoot,
        "last_event": last_event,
        "incomplete_trace_tail_lines": incomplete_tail_lines,
        "memory": [asdict(s) for s in memory],
        "telemetry": telemetry,
        "boundaries": [asdict(b) for b in boundaries(out / "bounds.json")],
    }
    atomic(out / "supervision.json", report)
    return report


def check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, body, expected in (
            (
                "healthy",
                "bounds[0]['end']=time.monotonic(); t.write_text(json.dumps(bounds)); t.replace(p)",
                "completed",
            ),
            ("assembly", "time.sleep(10)", "call_deadline"),
            ("no_boundary", "time.sleep(10)", "child_deadline"),
        ):
            out = root / name
            out.mkdir()
            code = (
                "import json,time;from pathlib import Path;p=Path('"
                + str(out / "bounds.json")
                + "');t=p.with_suffix('.partial');bounds=[{'name':'fake-collect','start':time.monotonic(),'end':None}];"
            )
            if name != "no_boundary":
                code += "t.write_text(json.dumps(bounds));t.replace(p);"
            code += body
            result = supervise(
                [sys.executable, "-c", code],
                out,
                seconds=0.25,
                call_seconds=0.1,
                poll_seconds=0.01,
                sample=lambda pid: {"pid": pid},
            )
            assert result["status"] == expected, result
            assert result["telemetry"], result
            if expected != "completed":
                assert result["exit_code"] == -signal.SIGKILL
        print(
            "Healthy exit, stalled assembly and missing-boundary process checks passed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, choices=(10349, 10350))
    parser.add_argument("--variant", choices=("reference", "staged"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    assert (
        args.root is not None
        and args.out is not None
        and args.seed is not None
        and args.variant is not None
    )
    report = supervise(
        [
            sys.executable,
            "scratch/collector-phase-worker.py",
            "--root",
            str(args.root),
            "--out",
            str(args.out),
            "--seed",
            str(args.seed),
            "--variant",
            args.variant,
        ],
        args.out,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"memory", "boundaries", "last_event"}
            }
        ),
        flush=True,
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
