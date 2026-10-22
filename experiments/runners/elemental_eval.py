"""Etude tournament adapter with an isolated released-SDK Manabot arena worker.

Etude schedules real games, stores tables and renders its report. No legacy
MonitorResult upload stands in for an evaluation. Game subprocesses use the
frozen Manabot environment; this module uses the exact newer Etude source.
"""

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

from etude.eval.match import Tables, agent, environment
from etude.eval.records import Agent, Match, RoundRobin, digest
from etude.jobs import evaluation
from etude.launch.client import launch
from etude.launch.records import Runtime
from etude.run.declaration import declare
from etude.run.records import ArtifactRef, Clock
from etude.run.storage import DirectoryStorage
from etude.reports import Evaluation, record
from etude.train.records import CheckpointRecord


def play(match: Match) -> Tables:
    root = Path(str(match.agents[0].config["root"]))
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    output.parent.mkdir(parents=True, exist_ok=True)
    players = []
    for member in match.agents:
        if member.kind == "checkpoint":
            parsed = urlparse(member.uri or "")
            if parsed.scheme != "file":
                raise ValueError(
                    "this local CPU evaluator requires retained local bytes"
                )
            players.append(
                {
                    "kind": "checkpoint",
                    "path": unquote(parsed.path),
                    "sha256": member.artifact,
                }
            )
        else:
            players.append({"kind": "scripted_greedy"})
    request = dict(
        players=players,
        role_a=match.conditions["a"],
        role_b=match.conditions["b"],
        deal=match.conditions["deal"],
        mirrors=match.conditions["mirrors"],
        output=str(output),
    )
    result = subprocess.run(
        [str(root / ".venv/bin/python"), "-m", "experiments.runners.elemental_games"],
        cwd=root,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=600,
    )
    if result.returncode:
        (output.parent / f"{match.key}.error.txt").write_text(result.stderr)
        raise RuntimeError(f"arena subprocess failed; retained {match.key}.error.txt")
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    return {
        "plays": [
            dict(
                player=seat,
                game=i,
                score=row["score_a"]
                if seat == 0
                else (None if row["score_a"] is None else 1 - row["score_a"]),
                valid=row["terminated"]
                and row["failure"] is None
                and row["replay_passed"],
                role_a=match.conditions["a"],
                role_b=match.conditions["b"],
                engine_seat=row["player_a_seat"]
                if seat == 0
                else 1 - row["player_a_seat"],
                decisions=row["decisions"],
                seconds=row["game_seconds"],
            )
            for i, row in enumerate(rows)
            for seat in (0, 1)
        ]
    }


def launch_evaluation(
    root: Path,
    checkpoint: Path,
    opponent: Path | None,
    *,
    name: str,
    conditions: tuple[dict[str, str | int | bool], ...],
    revision: str,
    pool: str = "local",
    run_id: str | None = None,
    training_run: str | None = None,
    update: int | None = None,
    cohort: str | None = None,
) -> str:
    def checkpoint_agent(path: Path, label: str) -> Agent:
        return agent(
            label,
            checkpoint=ArtifactRef(
                uri=path.resolve().as_uri(),
                sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                size_bytes=path.stat().st_size,
            ),
            code=[
                root / "manabot/arena",
                root / "experiments/runners/elemental_games.py",
            ],
            config={"root": str(root)},
        )

    candidate = checkpoint_agent(checkpoint, name)
    anchor = (
        checkpoint_agent(opponent, "initialization")
        if opponent
        else agent(
            "scripted-greedy",
            code=[root / "manabot/arena/players.py"],
            config={"root": str(root)},
        )
    )
    world = environment(
        "elemental-world-v1",
        files=[root / "managym/src", root / "content"],
        config={"world": "w10"},
    )
    tournament = RoundRobin(
        name=name, environment=world, conditions=conditions, games=4
    )
    job = evaluation(
        tournament,
        play=play,
        enter=(candidate, anchor),
        workers=2,
        repo=str(root),
        revision=revision,
        max_hours=1,
        max_usd=Decimal("0.01"),
        max_attempts=1,
        runtime=Runtime(
            python=str(root / ".runs/mtg174-eval-worker/bin/python"),
            python_version="3.12",
            packages={"etude-sdk": "0.1.2"},
        ),
    )
    run = job.runs(17402)[0]
    if run_id is not None:
        run = run.model_copy(update={"id": run_id})
    if training_run is not None:
        if update is None or cohort is None:
            raise ValueError("training link requires absolute update and frozen cohort")
        track = Path(os.environ["ETUDE_HOME"]) / "track"
        declared = declare(
            experiment="mtg174-elemental",
            run=training_run,
            namespace="manabot",
            clocks=[
                Clock(name="updates", kind="step", meaning="absolute optimizer updates")
            ],
            directory=track,
            kind="training",
            config={"revision": revision},
        )
        # A projection of the verified Manabot checkpoint, not a second trainer.
        saved = CheckpointRecord(
            ordinal=update + 1,
            artifact=ArtifactRef(
                uri=candidate.uri or "",
                sha256=candidate.artifact,
                size_bytes=checkpoint.stat().st_size,
                state="verified",
            ),
            at={"updates": update},
            wall_ms=0,
        )
        storage = DirectoryStorage(track)
        checkpoint_key = f"{training_run}/checkpoints/updates-{update}.json"
        try:
            prior = CheckpointRecord.model_validate_json(storage.get(checkpoint_key))
        except FileNotFoundError:
            storage.put(checkpoint_key, saved.model_dump_json().encode())
        else:
            if prior != saved:
                raise ValueError("training checkpoint binding changed")
        protocol = digest(
            {
                "environment": world.identity,
                "conditions": conditions,
                "games": tournament.games,
                "candidate_code": candidate.code,
                "candidate_config": candidate.config,
                "opponent": anchor.identity,
                "cohort": cohort,
                "revision": revision,
            }
        )
        record(
            track,
            training_run,
            Evaluation(
                training=declared.declaration.source_key,
                checkpoint=saved,
                run=run.id,
                benchmark=protocol,
                label=cohort,
                subject=candidate.identity,
            ),
        )
    return launch(run, pool=pool)
