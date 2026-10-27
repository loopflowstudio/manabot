"""Bounded, reasoning-enabled Qwen versus MTG-183's final raw checkpoint.

The fixed 100-game deck/seat schedule is retained before scoring. One resident
LLM worker and one CPU policy share sequential games. Etude owns the rental
ceiling; a process-group watchdog independently bounds executable game work.
Per-game archives and cumulative receipts survive interruption and cleanup.
"""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tarfile
import time

from etude.launch.records import Job
from pydantic import BaseModel, Field

from experiments.elemental_grid import DECKS, PACK
from manabot.infra.artifacts import StoredArtifact, split_bucket_uri
from manabot.infra.config import MatchConfig
from manabot.verify.llm_game import GameConfig, GameSummary, _write

CHECKPOINT_SHA256 = "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4"
CHECKPOINT_BYTES = 1071166
INFERENCE = Path("configs/llm/qwen3.5-9b-reasoning.json")


class Campaign(BaseModel):
    # Leaves setup, publication and shutdown inside the 11-hour rental bound.
    seconds: float = Field(default=10 * 3600, gt=0, le=10 * 3600)
    game_seconds: float = Field(default=3600, gt=0, le=3600)


class ScheduledGame(BaseModel):
    index: int
    qwen_deck: str
    policy_deck: str
    config: GameConfig


class GameResult(BaseModel):
    game: ScheduledGame
    summary: GameSummary


def schedule(config: Campaign) -> list[ScheduledGame]:
    games: list[ScheduledGame] = []
    for x, qwen in enumerate(DECKS):
        for y, policy in enumerate(DECKS):
            index = x * len(DECKS) + y
            seat = (x + y) % 2
            decks = (qwen, policy) if seat == 0 else (policy, qwen)
            games.append(
                ScheduledGame(
                    index=index,
                    qwen_deck=qwen,
                    policy_deck=policy,
                    config=GameConfig(
                        inference=INFERENCE,
                        seed=188_183_000 + index,
                        opponent_seed=188_595_000 + index,
                        model_seat=seat,
                        match=MatchConfig.authored(PACK, *decks),
                        skip_trivial=True,
                        max_decisions=2000,
                        max_seconds=config.game_seconds,
                    ),
                )
            )
    random.Random(188_183).shuffle(games)
    return games


def collect(out: Path) -> None:
    """Archive only new/current games; finalized archives are never rewritten."""
    records = out.parent / "game-records"
    if not records.exists():
        return
    for game in sorted(records.glob("game-*")):
        archive = out / f"{game.name}.tar.gz"
        if (
            archive.exists()
            and (game / "summary.json").exists()
            and archive.stat().st_mtime_ns >= (game / "summary.json").stat().st_mtime_ns
        ):
            continue
        temporary = archive.with_suffix(".pending")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
            bundle.add(game, arcname=game.name)
        temporary.replace(archive)


def worker(config: Campaign, out: Path) -> None:
    import torch

    from manabot.llm.schema import InferenceConfig
    from manabot.llm.session import InferenceSession
    from manabot.verify.llm_game import CheckpointOpponent, play

    torch.set_num_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.monotonic()
    deadline = started + config.seconds
    games = schedule(config)
    _write(
        out / "schedule.json",
        json.dumps([g.model_dump(mode="json") for g in games], indent=2),
    )
    _write(
        out / "games.json",
        json.dumps(
            {
                "version": "llm-game-sequence-v1",
                "games": [f"game-{g.index:03d}" for g in games],
            },
            indent=2,
        ),
    )
    opponent = CheckpointOpponent(
        Path(os.environ["ETUDE_INPUTS_DIR"]) / "checkpoint.pt",
        CHECKPOINT_SHA256,
        CHECKPOINT_BYTES,
    )
    inference = InferenceConfig.model_validate_json(INFERENCE.read_text())
    results: list[GameResult] = []
    session = InferenceSession(inference, load_seconds=300)
    _write(
        out / "readiness.json",
        json.dumps(
            {
                "policy_cpu_seconds": opponent.load_seconds,
                "llm_session_seconds": session.load_seconds,
                "seconds": session.load_seconds,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "runtime": session.identity.model_dump(mode="json"),
                "warmup": None,
            },
            indent=2,
        ),
    )
    stop_reason: str | None = None
    try:
        for game in games:
            remaining = deadline - time.monotonic() - 120
            if remaining < inference.deadline_seconds:
                stop_reason = "campaign time reserve reached"
                break
            bounded = game.config.model_copy(
                update={"max_seconds": min(config.game_seconds, remaining)}
            )
            record = out.parent / "game-records" / f"game-{game.index:03d}"
            try:
                summary = play(
                    bounded,
                    record,
                    session,
                    opponent=opponent,
                    progress_path=out / "progress.json",
                )
            except Exception as error:
                summary = GameSummary(
                    mode="qwen-v-checkpoint", failure=f"{type(error).__name__}: {error}"
                )
                record.mkdir(parents=True, exist_ok=True)
                _write(record / "summary.json", summary.model_dump_json(indent=2))
            results.append(GameResult(game=game, summary=summary))
            _write(
                out / "results.json",
                json.dumps([r.model_dump(mode="json") for r in results], indent=2),
            )
            collect(out)
            print(
                f"GAME {game.index}: completed={summary.completed} winner={summary.winner} seconds={summary.game_seconds:.3f}",
                flush=True,
            )
            # Failed worker/native admission is not grounds for a paid retry.
            if not summary.completed:
                stop_reason = summary.failure or "game incomplete"
                break
    finally:
        session.close()
        collect(out)
        _write(
            out / "campaign.json",
            json.dumps(
                {
                    "planned": len(games),
                    "attempted": len(results),
                    "completed": sum(r.summary.completed for r in results),
                    "seconds": time.monotonic() - started,
                    "stop_reason": stop_reason,
                    "session_closed": True,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                },
                indent=2,
            ),
        )


def run(config: Campaign) -> None:
    """Etude entry point: supervise the whole process tree, then collect."""
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "campaign-config.json"
    _write(config_path, config.model_dump_json(indent=2))
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv",
            "--loop-ms=1000",
            f"--filename={out / 'gpu-memory.csv'}",
        ]
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "experiments.qwen_mtg183",
            "--worker",
            "--config",
            str(config_path),
            "--out",
            str(out),
        ],
        start_new_session=True,
    )
    timed_out = False
    try:
        child.wait(timeout=config.seconds + 60)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # Also reap a spawned inference worker if the game owner died abruptly.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
        sampler.terminate()
        sampler.wait(timeout=10)
        collect(out)
        _write(
            out / "supervision.json",
            json.dumps({"timed_out": timed_out, "returncode": child.returncode}),
        )
    if timed_out or child.returncode:
        raise RuntimeError("campaign worker interrupted; partial evidence retained")


def runs(native_artifact: str, native_source: str) -> Job:
    """One rental, one attempt; caller's $10 cap includes setup and delivery."""
    from etude.jobs import training
    from etude.launch.records import Input, Variable

    from experiments.qwen_mtg183 import Campaign as CampaignConfig, run as campaign_run

    artifact = StoredArtifact.model_validate_json(Path(native_artifact).read_text())
    provenance = json.loads(Path(native_source).read_text())
    if (artifact.sha256, artifact.bytes) != (provenance["sha256"], provenance["bytes"]):
        raise ValueError("native artifact provenance differs")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            provenance["source_commit"],
            "HEAD",
            "--",
            "managym/src",
            "managym/Cargo.toml",
            "managym/Cargo.lock",
            "managym/build.rs",
            "content",
        ],
        check=True,
    )
    scheme, bucket, key = split_bucket_uri(artifact.uri)
    if (scheme, bucket) != ("r2", "manabot"):
        raise ValueError("expected manabot R2 artifact")
    return training(
        campaign_run,
        config=CampaignConfig(),
        repo=".",
        revision="HEAD",
        name="mtg188-qwen-thinking-v-mtg183-u5950",
        max_hours=11,
        max_usd=Decimal("9.50"),
        max_attempts=1,
        setup="sh scripts/qwen-mtg183-setup.sh",
        cleanup="uv run --no-sync python -m experiments.qwen_mtg183 --collect",
        cleanup_seconds=180,
        inputs=(
            Input(
                name="native.so",
                key=key,
                sha256=artifact.sha256,
                size_bytes=artifact.bytes,
            ),
            Input(
                name="checkpoint.pt",
                key="jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt",
                sha256=CHECKPOINT_SHA256,
                size_bytes=CHECKPOINT_BYTES,
            ),
        ),
        environment=tuple(
            Variable(name=k, value=v)
            for k, v in {
                "HF_HOME": "/workspace/manabot-hf",
                "UV_CACHE_DIR": "/workspace/manabot-uv",
                "UV_NO_SYNC": "1",
                "OMP_NUM_THREADS": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            }.items()
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    out = args.out or Path(os.environ["ETUDE_OUTPUT_DIR"])
    if args.collect:
        collect(out)
    elif args.worker and args.config:
        worker(Campaign.model_validate_json(args.config.read_text()), out)
    else:
        parser.error("choose --collect or --worker --config")


if __name__ == "__main__":
    main()
