"""MTG-174's bounded, restartable local Etude evaluator and report owner.

This owner never rents or starts training. It polls immutable Manabot snapshots,
launches real typed Etude evaluation jobs, and cancels its named training job on
evaluation failure. Its frozen config and create-before-launch ledger survive a
conversation or process exit. A launchd service owns the process, not a chat.
"""

import argparse
from datetime import UTC, datetime
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import time

from etude.eval.query import connect
from etude.eval.report import write
from etude.launch.client import status
from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.elemental_eval import launch_evaluation


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Path
    directory: Path
    job_id: str
    revision: str
    deadline: float
    cadence: int = 500
    monitoring: tuple[dict[str, str | int | bool], ...]
    scoring: tuple[dict[str, str | int | bool], ...]


class Point(BaseModel):
    key: str
    update: int
    checkpoint: Path
    sha256: str
    cohort: str
    opponent: Path | None = None
    eval_id: str
    phase: str = "pending"
    outcome: str | None = None
    score: float | None = None
    games: int = 0
    valid: int = 0
    started: float | None = None
    finished: float | None = None


class Ledger(BaseModel):
    config_sha256: str
    points: list[Point] = Field(default_factory=list)
    observed_at: str | None = None
    remote_phase: str | None = None
    remote_updates: int = 0
    cleanup_confirmed: bool = False
    inclusive_cost: float | None = None
    error: str | None = None
    settled: bool = False


def _save(path: Path, value: BaseModel) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(value.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def _remote(config: Config, *, cancel: bool = False) -> dict:
    command = [
        "doppler",
        "run",
        "--project",
        "etude",
        "--config",
        "prd",
        "--",
        "uv",
        "run",
        "--no-sync",
        "--extra",
        "artifacts",
        "python",
        "-m",
        "experiments.runners.elemental_remote",
        "--job-id",
        config.job_id,
        "--cache",
        str(config.directory / "cache"),
    ]
    if cancel:
        command.append("--cancel")
    result = subprocess.run(
        command, cwd=config.root, capture_output=True, text=True, timeout=180
    )
    if result.returncode:
        # Credentials and network responses must never enter report errors.
        raise RuntimeError(f"remote reader exited {result.returncode}")
    return json.loads(result.stdout)


def _report(config: Config, ledger: Ledger) -> None:
    report = config.directory / "reports"
    report.mkdir(exist_ok=True)
    track = Path(os.environ["ETUDE_HOME"]) / "track"
    if track.exists():
        write(track, report, refresh=30)
    rows = "".join(
        f"<tr><td>{p.update}</td><td>{escape(p.cohort)}</td>"
        f"<td>{escape(p.eval_id)}</td><td>{escape(p.phase)}/{escape(p.outcome or '')}</td>"
        f"<td>{p.score if p.score is not None else 'missing'}</td>"
        f"<td>{p.valid}/{p.games}</td><td>{p.sha256}</td></tr>"
        for p in ledger.points
    )
    complete = [
        p for p in ledger.points if p.cohort == "monitor" and p.score is not None
    ]
    xmax = max([p.update for p in complete] + [config.cadence])
    points = " ".join(
        f"{40 + 700 * p.update / xmax:.2f},{220 - 180 * p.score:.2f}" for p in complete
    )
    body = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        "<title>MTG-174 Elemental baseline live report</title>"
        "<h1>MTG-174 Elemental baseline</h1>"
        f"<p>Report generated {datetime.now(UTC).isoformat()}; last remote observation "
        f"{ledger.observed_at}. Training {ledger.remote_phase}, update {ledger.remote_updates}. "
        f"Confirmed cleanup: {ledger.cleanup_confirmed}. Inclusive estimated cost: {ledger.inclusive_cost}.</p>"
        "<p>Freshness: observations older than 180 seconds are stale. Missing evaluations are "
        "pending or failed, never zero scores. Monitoring is a fixed cohort against scripted greedy; "
        "final scoring is disjoint. One training seed; game noise is not method evidence.</p>"
        f"<p>Source {escape(config.revision)}; error: {escape(ledger.error or 'none')}</p>"
        "<h2>Learning curve</h2><p>Raw policy score vs scripted greedy; x = absolute updates; "
        "y = score, 0 at bottom and 1 at top. Fixed pair/deal/seat cohort.</p>"
        f'<svg viewBox="0 0 800 250"><path d="M40 40 V220 H760" fill="none" stroke="black"/>'
        f'<polyline points="{points}" fill="none" stroke="#964d35" stroke-width="3"/></svg>'
        '<p><a href="eval/index.html">Etude evaluation tables and uncertainty</a></p>'
        '<table border="1"><tr><th>update</th><th>cohort</th><th>Etude run</th>'
        "<th>state</th><th>score</th><th>valid/attempted</th><th>checkpoint sha256</th></tr>"
        f"{rows}</table>"
    )
    temporary = report / "index.partial"
    temporary.write_text(body)
    temporary.replace(report / "index.html")


def _point(
    config: Config, checkpoint: dict, cohort: str, opponent: Path | None = None
) -> Point:
    update = checkpoint["coordinates"]["updates"]
    key = f"{checkpoint['sha256']}/{cohort}"
    return Point(
        key=key,
        update=update,
        checkpoint=Path(checkpoint["path"]),
        sha256=checkpoint["sha256"],
        cohort=cohort,
        opponent=opponent,
        eval_id=hashlib.sha256(key.encode()).hexdigest()[:12],
    )


def _refresh(config: Config, point: Point) -> None:
    state = status(point.eval_id)
    point.phase = state.phase
    point.outcome = state.outcome
    if state.done:
        point.finished = time.time()
        track = Path(os.environ["ETUDE_HOME"]) / "track"
        db = connect(track)
        try:
            tables = {r[0] for r in db.sql("show tables").fetchall()}
            if "plays" in tables:
                row = db.execute(
                    "select avg(score), count(*), count(*) filter (where valid) "
                    "from plays where run = ? and player = 0",
                    [point.eval_id],
                ).fetchone()
                if row is not None:
                    point.score, point.games, point.valid = row
        finally:
            db.close()


def operate(config: Config, ledger: Ledger, path: Path) -> None:
    if ledger.error is not None:
        # A service restart cannot turn a retained failed evaluation into a
        # successful campaign by skipping it and launching later checkpoints.
        observation = _remote(config, cancel=True)
        remote = observation["status"]
        ledger.observed_at = observation["observed_at"]
        ledger.cleanup_confirmed = remote["cleanup"] is not None
        if remote["cleanup"]:
            ledger.inclusive_cost = remote["cleanup"]["estimated_dollars"]
        _save(path, ledger)
        _report(config, ledger)
        if not ledger.cleanup_confirmed:
            raise RuntimeError("failed campaign cancellation awaits confirmed cleanup")
        return
    while time.time() < config.deadline and not ledger.settled:
        try:
            observation = _remote(config)
            observed = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            (config.directory / f"remote-{observed}.json").write_text(
                json.dumps(observation)
            )
            if observation["source"]["commit"] != config.revision:
                raise ValueError("remote source differs from frozen source")
            remote = observation["status"]
            record = remote.get("record")
            ledger.observed_at = observation["observed_at"]
            ledger.remote_phase = record["phase"] if record else "not-accepted"
            ledger.remote_updates = record["updates"] if record else 0
            ledger.cleanup_confirmed = remote["cleanup"] is not None
            ledger.inclusive_cost = observation.get("estimated_cumulative_dollars")
            if remote["cleanup"]:
                ledger.inclusive_cost = remote["cleanup"]["estimated_dollars"]
            available = observation["checkpoints"]
            initial = next(
                (c for c in available if c["coordinates"]["updates"] == 0), None
            )
            terminal = bool(
                record
                and record["phase"]
                in ("completed", "paused", "failed", "cancelled", "deadline")
            )
            additions = [
                _point(config, c, "monitor")
                for c in available
                if c["coordinates"]["updates"] % config.cadence == 0 or terminal
            ]
            if terminal and initial is not None and available:
                endpoint = max(available, key=lambda c: c["coordinates"]["updates"])
                additions.extend(
                    (
                        _point(
                            config, endpoint, "final-initial", Path(initial["path"])
                        ),
                        _point(config, endpoint, "final-greedy"),
                        _point(config, initial, "initial-greedy"),
                    )
                )
            known = {p.key for p in ledger.points}
            ledger.points.extend(p for p in additions if p.key not in known)
            _save(path, ledger)
            active = next((p for p in ledger.points if p.phase not in ("done",)), None)
            if active is not None:
                if active.started is None:
                    active.started = time.time()
                    _save(path, ledger)
                    launch_evaluation(
                        config.root,
                        active.checkpoint,
                        active.opponent,
                        name=f"mtg174-{active.cohort}-updates-{active.update}",
                        conditions=config.monitoring
                        if active.cohort == "monitor"
                        else config.scoring,
                        revision=config.revision,
                        pool="controller",
                        run_id=active.eval_id,
                    )
                _refresh(config, active)
                if active.phase == "done" and (
                    active.outcome != "succeeded"
                    or active.games == 0
                    or active.valid != active.games
                ):
                    raise RuntimeError(
                        f"evaluation {active.eval_id} failed or has invalid games"
                    )
            ledger.settled = (
                terminal
                and ledger.cleanup_confirmed
                and initial is not None
                and bool(ledger.points)
                and all(p.phase == "done" for p in ledger.points)
                and {"final-initial", "final-greedy", "initial-greedy"}.issubset(
                    {p.cohort for p in ledger.points}
                )
            )
            ledger.error = None
        except Exception as error:
            ledger.error = type(error).__name__ + ": " + str(error)
            _save(path, ledger)
            _report(config, ledger)
            # A failed instrument must stop paid learning. A failed cancellation
            # remains visible and is retried under the original deadline.
            _remote(config, cancel=True)
            raise
        _save(path, ledger)
        _report(config, ledger)
        if not ledger.settled:
            time.sleep(30)
    if not ledger.settled:
        _remote(config, cancel=True)
        raise RuntimeError("bounded evaluator owner deadline exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = Config.model_validate_json(args.config.read_bytes())
    config.directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(args.config.read_bytes()).hexdigest()
    path = config.directory / "ledger.json"
    with (config.directory / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            Ledger.model_validate_json(path.read_bytes())
            if path.exists()
            else Ledger(config_sha256=identity)
        )
        if ledger.config_sha256 != identity:
            raise ValueError("owner configuration changed")
        operate(config, ledger, path)


if __name__ == "__main__":
    main()
