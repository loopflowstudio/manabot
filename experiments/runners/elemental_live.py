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
import json
import os
from pathlib import Path
import subprocess
import time

from etude.eval.query import connect
from etude.launch.client import status
from etude.launch.errors import LauncherError
from etude.reports import Evaluation
from etude.train.report import pages, write
from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.elemental_eval import launch_evaluation, prepare_evaluation


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
    training_run: str
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
    for page in pages(track).values():
        write(
            page
            | {
                "owner": ledger.model_dump(mode="json"),
                "owner_deadline": config.deadline,
            },
            report,
            refresh=30,
        )
    # Readiness and remote lifecycle are machine-readable beside the native report.
    _save(report / "readiness.json", ledger)


def _point(
    config: Config,
    checkpoint: dict,
    cohort: str,
    opponent: Path | None = None,
    *,
    training_run: str,
) -> Point:
    update = checkpoint["coordinates"]["updates"]
    key = hashlib.sha256(
        json.dumps(
            {
                "job": config.job_id,
                "training_run": training_run,
                "update": update,
                "checkpoint": checkpoint["sha256"],
                "cohort": cohort,
                "opponent": hashlib.file_digest(
                    opponent.open("rb"), "sha256"
                ).hexdigest()
                if opponent
                else None,
                "conditions": config.monitoring
                if cohort == "monitor"
                else config.scoring,
                "revision": config.revision,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return Point(
        key=key,
        training_run=training_run,
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
                binding = Evaluation.model_validate_json(
                    (
                        track
                        / point.training_run
                        / "evaluations"
                        / f"{point.eval_id}.json"
                    ).read_bytes()
                )
                row = db.execute(
                    "select avg(p.score), count(*), count(*) filter (where p.valid) "
                    "from plays p join seats s using (run, match, player) "
                    "where p.run = ? and s.agent = ?",
                    [point.eval_id, binding.subject],
                ).fetchone()
                if row is not None:
                    point.score, point.games, point.valid = row
        finally:
            db.close()


def _ensure_launched(config: Config, point: Point, ledger: Ledger, path: Path) -> None:
    """Reconcile a stable ID after interruption on either side of submission.

    Only a definite missing run permits submission. Transport errors and other
    refusals stop the owner; they cannot establish that a launch was absent.
    """
    try:
        status(point.eval_id)
    except LauncherError as error:
        if str(error) != f"no run {point.eval_id!r}":
            raise
        point.started = time.time()
        _save(path, ledger)
        launch_evaluation(
            config.root,
            point.checkpoint,
            point.opponent,
            name=f"mtg174-{point.cohort}-updates-{point.update}",
            conditions=config.monitoring
            if point.cohort == "monitor"
            else config.scoring,
            revision=config.revision,
            pool="controller",
            run_id=point.eval_id,
            training_run=point.training_run,
            update=point.update,
            cohort=point.cohort,
        )


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
            final_observation = config.directory / "remote-final.json"
            observation = (
                json.loads(final_observation.read_bytes())
                if final_observation.exists()
                else _remote(config)
            )
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
            if terminal and ledger.cleanup_confirmed and not final_observation.exists():
                # After verified provider absence, final scoring needs only the
                # retained immutable generation, not renewable cloud credentials.
                with final_observation.open("x") as output:
                    json.dump(observation, output)
            training_run = observation["run"]["id"] if available else ""
            endpoint_update = max(
                (c["coordinates"]["updates"] for c in available), default=0
            )
            additions = [
                _point(config, c, "monitor", training_run=training_run)
                for c in available
                if c["coordinates"]["updates"] % config.cadence == 0
                or (terminal and c["coordinates"]["updates"] == endpoint_update)
            ]
            if terminal and initial is not None and available:
                endpoint = max(available, key=lambda c: c["coordinates"]["updates"])
                additions.extend(
                    (
                        _point(
                            config,
                            endpoint,
                            "final-initial",
                            Path(initial["path"]),
                            training_run=training_run,
                        ),
                        _point(
                            config, endpoint, "final-greedy", training_run=training_run
                        ),
                        _point(
                            config, initial, "initial-greedy", training_run=training_run
                        ),
                    )
                )
            known = {p.key for p in ledger.points}
            fresh = [p for p in additions if p.key not in known]
            ledger.points.extend(fresh)
            _save(path, ledger)
            for point in ledger.points:
                binding = (
                    Path(os.environ["ETUDE_HOME"])
                    / "track"
                    / point.training_run
                    / "evaluations"
                    / f"{point.eval_id}.json"
                )
                if binding.exists():
                    continue
                prepare_evaluation(
                    config.root,
                    point.checkpoint,
                    point.opponent,
                    name=f"mtg174-{point.cohort}-updates-{point.update}",
                    conditions=config.monitoring
                    if point.cohort == "monitor"
                    else config.scoring,
                    revision=config.revision,
                    run_id=point.eval_id,
                    training_run=point.training_run,
                    update=point.update,
                    cohort=point.cohort,
                )
            active = next((p for p in ledger.points if p.phase not in ("done",)), None)
            if active is not None:
                _ensure_launched(config, active, ledger, path)
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
