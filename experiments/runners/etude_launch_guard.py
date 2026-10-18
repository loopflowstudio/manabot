"""Protect a laptop's bounded Etude launcher and refuse stale rental creation.

Etude 0.1.2 starts its five-minute pre-start clock before publishing the session's
files. A sleeping controller can finish that publication after the clock expires.
This process-local adapter checks the recorded claim immediately before the
provider create call; it never resets deadlines or retries creation. On macOS it
also holds an idle-sleep assertion for this context's bounded lifetime. Explicit
sleep, lid closure, power loss and network outages still require reconciliation.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import NotRequired, TypedDict, Unpack

from etude.launch.executor import Refused
from etude.launch.providers.runpod.api import PodFact, RunPodAPI
from etude.launch.records import Session


class _CreateOptions(TypedDict):
    name: str
    gpu: str
    image: str
    registry: str
    env: dict[str, str]
    vcpus: int
    memory_gb: int
    disk_gb: int
    data_center: NotRequired[str | None]
    entrypoint: tuple[str, ...]


def _inhibit_idle_sleep(seconds: int) -> subprocess.Popen[bytes] | None:
    if sys.platform != "darwin":
        return None
    process = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-t", str(seconds), "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Check the assertion, not merely whether its helper was spawned.
        for _ in range(10):
            if process.poll() is not None:
                break
            report = subprocess.run(
                ["/usr/bin/pmset", "-g", "assertions"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            if re.search(
                rf"pid {process.pid}\(caffeinate\):[^\n]*PreventUserIdleSystemSleep",
                report,
            ):
                return process
            time.sleep(0.05)
        raise RuntimeError("The launcher's idle-sleep assertion was not established")
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise


@contextmanager
def protect_launcher(
    session: Callable[[str], Session],
    evidence: Path,
    *,
    seconds: int,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Iterator[None]:
    """Guard this process's rentals; retain decisions without requests or secrets.

    The caller supplies the owning SDK database's session lookup. Refusals occur
    before sending a provider create request, so the SDK can safely settle them
    as capacity_refused. The retained reason distinguishes a stale local launch
    from unavailable GPU stock. Existing rentals and cleanup remain untouched.
    """
    if seconds <= 0:
        raise ValueError("Launcher lifetime must be positive")
    evidence.parent.mkdir(parents=True, exist_ok=True)
    inhibitor = _inhibit_idle_sleep(seconds)
    began = clock()
    original = RunPodAPI.create

    def record(phase: str, **fields: str | int | float | None) -> None:
        with evidence.open("a") as output:
            output.write(
                json.dumps({"phase": phase, "at": clock().isoformat(), **fields}) + "\n"
            )

    @wraps(original)
    def create(api: RunPodAPI, **options: Unpack[_CreateOptions]) -> PodFact:
        name = options["name"]
        reason: str | None = None
        age: float | None = None
        if not name.startswith("etude-"):
            reason = "unrecognized_session"
        else:
            current = session(name.removeprefix("etude-"))
            if current.claimed_at is None:
                reason = "unclaimed_session"
            else:
                age = (clock() - current.claimed_at).total_seconds()
                # Preserve >=180 of the SDK's unchanged 300 seconds for boot,
                # its 120-second installer, source delivery and startup receipt.
                if not 0 <= age <= 120:
                    reason = "startup_budget_spent_before_provider_creation"
            if (
                current.receipt is not None
                or current.ended_at is not None
                or current.refused_at is not None
            ):
                reason = "session_already_settled_or_created"
        elapsed = (clock() - began).total_seconds()
        if not 0 <= elapsed < seconds:
            reason = "launcher_lifetime_expired_or_clock_reversed"
        if inhibitor is not None and inhibitor.poll() is not None:
            reason = "idle_sleep_assertion_lost"
        record(
            "create_refused" if reason else "create_allowed",
            name=name,
            claim_age_seconds=age,
            reason=reason,
        )
        if reason is not None:
            raise Refused(reason)
        # The SDK retains the create outcome, including unknown replies, in its
        # sanitized launch diagnostics (`etude logs --machine`).
        return original(api, **options)

    try:
        record(
            "guard_started",
            pid=os.getpid(),
            inhibitor_pid=inhibitor.pid if inhibitor else None,
            seconds=seconds,
        )
        RunPodAPI.create = create
        yield
    finally:
        RunPodAPI.create = original
        if inhibitor is not None:
            if inhibitor.poll() is None:
                inhibitor.terminate()
            inhibitor.wait(timeout=5)
        record("guard_stopped")
