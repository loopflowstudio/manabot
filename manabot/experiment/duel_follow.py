"""Follow admitted checkpoint exports and retain bounded comparison attempts."""

from __future__ import annotations

import fcntl
from html import escape
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from manabot.eval.duel import Duel, DuelResult
from manabot.eval.monitor import MonitorResult
from manabot.infra.files import atomic_json
from manabot.remote.live_evaluation import Ledger
from manabot.remote.supervisor import stop_process


class DirectAttempt(BaseModel):
    phase: Literal["running", "completed", "failed"]
    started_at: float
    two_seat: str
    one_seat: str
    pid: int | None = None
    exit_code: int | None = None
    finished_at: float | None = None


_ATTEMPTS = TypeAdapter(dict[str, DirectAttempt])


def report(directory: Path, source: Path, attempts: dict[str, DirectAttempt]) -> None:
    public = source / "public"
    atomic_json(
        public / "head-to-head.json", _ATTEMPTS.dump_python(attempts, mode="json")
    )
    rows = ""
    for updates, item in sorted(attempts.items(), key=lambda pair: int(pair[0])):
        result_path = directory / "evaluations" / updates / "result.json"
        if result_path.exists():
            result = DuelResult.model_validate_json(result_path.read_bytes())
            rows += f'<tr><td colspan="5">{updates} updates: two-seat {result.two_seat_coordinates.training_seconds / 3600:.2f} training hours / {result.two_seat_coordinates.learner_transitions:,} learner decisions; one-seat {result.one_seat_coordinates.training_seconds / 3600:.2f} hours / {result.one_seat_coordinates.learner_transitions:,} decisions.</td></tr>'
            for label, group in result.groups.items():
                low, high = group.interval
                rows += f"<tr><td>{updates}</td><td>{label}</td><td>{group.score:.1%}</td><td>[{low:.1%}, {high:.1%}]</td><td>{group.wins}/{group.draws}/{group.losses}</td></tr>"
        else:
            rows += (
                f'<tr><td>{updates}</td><td colspan="4">{escape(str(item))}</td></tr>'
            )
    if not rows:
        rows = '<tr><td colspan="5">Waiting for the first completed 1,000-update admission; direct evaluations follow the heuristic evaluation.</td></tr>'
    page = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Direct two-seat vs one-seat</title><style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem}</style><h1>Two-seat vs one-seat, equal updates</h1><p>200 games per checkpoint: 25 paired deals × eight deck/seat legs. 50 games per deck pairing, balanced play/draw. Score is from the two-seat model’s perspective. Win/draw/loss counts shown. 95% intervals resample whole deals (4000 draws, seed 0).</p><p>Repeated exploratory monitoring, one training seed, unequal training cost/data. Repeated looks and subgroup intervals do not establish method superiority. No automatic training or budget changes.</p><p><a href="/">Heuristic monitoring</a> · <a href="head-to-head.json">Attempt ledger</a></p><table><tr><th>Updates</th><th>Deck pairing</th><th>Two-seat score</th><th>95% interval</th><th>W/D/L</th></tr>'
        + rows
        + f"</table><p>Observer refreshed {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}. Above 50% favors two-seat; below 50% favors one-seat. Exact controls are retained through 12,000 updates.</p>"
    )
    temporary = public / "head-to-head.html.tmp"
    temporary.write_text(page)
    temporary.replace(public / "head-to-head.html")


def follow(source: Path, controls: Path, directory: Path, deadline: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = directory / "ledger.json"
        attempts = (
            _ATTEMPTS.validate_json(ledger_path.read_bytes())
            if ledger_path.exists()
            else {}
        )
        if any(item.phase == "running" for item in attempts.values()):
            raise RuntimeError(
                "prior direct evaluator requires explicit reconciliation"
            )
        while time.time() < deadline:
            ledger = Ledger.model_validate_json((source / "ledger.json").read_bytes())
            for item in ledger.attempts.values():
                updates = item.updates
                control_path = controls / f"{updates}.json"
                if (
                    updates <= 0
                    or updates % 1000
                    or str(updates) in attempts
                    or item.phase != "completed"
                    or not control_path.exists()
                ):
                    continue
                matches = list(
                    (source / "evaluations").glob(
                        f"{item.digest}-attempt-*/admission.json"
                    )
                )
                admission = MonitorResult.model_validate_json(
                    sorted(matches)[-1].read_bytes()
                )
                control = MonitorResult.model_validate_json(control_path.read_bytes())
                duel = Duel(two_seat=admission, one_seat=control)
                duel.validate_inputs()
                target = directory / "evaluations" / str(updates)
                target.mkdir(parents=True, exist_ok=True)
                atomic_json(target / "duel.json", duel.model_dump(mode="json"))
                entry = DirectAttempt(
                    phase="running",
                    started_at=time.time(),
                    two_seat=admission.artifact["sha256"],
                    one_seat=control.artifact["sha256"],
                )
                attempts[str(updates)] = entry
                atomic_json(ledger_path, _ATTEMPTS.dump_python(attempts, mode="json"))
                with (target / "console.log").open("ab") as console:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "manabot.eval.duel",
                            "--duel",
                            str(target / "duel.json"),
                        ],
                        stdout=console,
                        stderr=console,
                        start_new_session=True,
                    )
                    entry.pid = child.pid
                    atomic_json(
                        ledger_path, _ATTEMPTS.dump_python(attempts, mode="json")
                    )
                    end = min(deadline, time.time() + 3600)
                    try:
                        while child.poll() is None and time.time() < end:
                            report(directory, source, attempts)
                            try:
                                child.wait(timeout=min(30, max(0.1, end - time.time())))
                            except subprocess.TimeoutExpired:
                                pass
                    finally:
                        if child.poll() is None:
                            stop_process(child)
                        entry.phase = (
                            "completed"
                            if child.wait() == 0 and (target / "result.json").exists()
                            else "failed"
                        )
                        entry.exit_code = child.returncode
                        entry.finished_at = time.time()
                        atomic_json(
                            ledger_path, _ATTEMPTS.dump_python(attempts, mode="json")
                        )
            report(directory, source, attempts)
            if ledger.cohort_phase not in ("running", "pending") and not any(
                item.phase == "running" for item in ledger.attempts.values()
            ):
                break
            time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    follow(args.source, args.controls, args.directory, args.deadline)
