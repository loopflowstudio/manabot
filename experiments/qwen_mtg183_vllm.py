"""Resume the accepted matchup with four isolated games sharing one vLLM server.

Each game process owns its CPU policy RNG and native state. The coordinator
reserves the unchanged schedule first and submits at most four games at once.
A failed game stops new admissions; already admitted games retain their outcomes.
The existing campaign watchdog kills the whole game process group on timeout.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from decimal import Decimal
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import time

from etude.launch.records import Job
from pydantic import Field

from experiments import qwen_mtg183 as original
from manabot.llm.schema import InferenceConfig
from manabot.verify.llm_game import GameSummary, _write


class Campaign(original.Campaign):
    concurrency: int = Field(default=4, ge=1, le=4)


def play_game(
    game: original.ScheduledGame, out: Path, deadline: float
) -> original.GameResult:
    """One process per game prevents Torch's process-global RNG from crossing games."""
    import torch

    from manabot.llm.session import InferenceSession
    from manabot.llm.vllm_runtime import VLLMRuntime
    from manabot.verify.llm_game import CheckpointOpponent, play

    torch.set_num_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    record = out.parent / "game-records" / f"game-{game.index:03d}"
    started = time.monotonic()
    try:
        inference = InferenceConfig.model_validate_json(original.INFERENCE.read_text())
        opponent = CheckpointOpponent(
            Path(os.environ["ETUDE_INPUTS_DIR"]) / "checkpoint.pt",
            original.CHECKPOINT_SHA256,
            original.CHECKPOINT_BYTES,
        )
        factory = partial(VLLMRuntime, identity_path=out / "vllm-identity.json")
        with InferenceSession(inference, load_seconds=60, factory=factory) as session:
            startup = time.monotonic() - started
            remaining = deadline - time.time() - 120
            if remaining < inference.deadline_seconds:
                raise TimeoutError("campaign reserve reached before game admission")
            summary = play(
                game.config.model_copy(
                    update={"max_seconds": min(game.config.max_seconds, remaining)}
                ),
                record,
                session,
                opponent=opponent,
            )
            _write(
                record / "actor-startup.json",
                json.dumps(
                    {
                        "seconds": startup,
                        "policy_cpu_seconds": opponent.load_seconds,
                        "client_session_seconds": session.load_seconds,
                        "boundary": "per-game policy/client preparation, outside game wall; shared GPU startup recorded once at campaign root",
                    }
                ),
            )
    except Exception as error:
        summary = GameSummary(
            mode="qwen-v-checkpoint", failure=f"{type(error).__name__}: {error}"
        )
        record.mkdir(parents=True, exist_ok=True)
        _write(record / "config.json", game.config.model_dump_json(indent=2))
        _write(record / "summary.json", summary.model_dump_json(indent=2))
    return original.GameResult(game=game, summary=summary)


def worker(config: Campaign, out: Path) -> None:
    started = time.monotonic()
    deadline = time.time() + config.seconds
    games = original.schedule(config)
    _write(
        out / "schedule.json",
        json.dumps([g.model_dump(mode="json") for g in games], indent=2),
    )
    _write(
        out / "games.json",
        json.dumps(
            {
                "version": "llm-game-sequence-v1",
                "mode": "concurrent",
                "games": [f"game-{g.index:03d}" for g in games],
            }
        ),
    )
    results: list[original.GameResult] = []
    pending: dict[Future[original.GameResult], original.ScheduledGame] = {}
    queued = iter(games)
    stop_reason: str | None = None

    def retain(*, closed: bool = False) -> None:
        # Persist admissions before waiting: a killed worker is still attempted.
        _write(
            out / "results.json",
            json.dumps([r.model_dump(mode="json") for r in results], indent=2),
        )
        _write(
            out / "campaign.json",
            json.dumps(
                {
                    "planned": len(games),
                    "attempted": len(results) + len(pending),
                    "settled": len(results),
                    "completed": sum(r.summary.completed for r in results),
                    "pending_games": [g.index for g in pending.values()],
                    "seconds": time.monotonic() - started,
                    "stop_reason": stop_reason,
                    "concurrency": config.concurrency,
                    "session_closed": closed,
                    "checkpoint_sha256": original.CHECKPOINT_SHA256,
                },
                indent=2,
            ),
        )

    with ProcessPoolExecutor(
        max_workers=config.concurrency, mp_context=multiprocessing.get_context("spawn")
    ) as pool:

        def admit() -> None:
            nonlocal stop_reason
            if deadline - time.time() < 240:
                stop_reason = "campaign time reserve reached"
                return
            game = next(queued, None)
            if game is not None:
                future: Future[original.GameResult] = Future()
                pending[future] = game
                retain()
                try:
                    submitted = pool.submit(play_game, game, out, deadline)
                except Exception as error:
                    future.set_exception(error)
                else:
                    del pending[future]
                    pending[submitted] = game

        for _ in range(config.concurrency):
            admit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Observe every finished failure before admitting another game.
            for future in done:
                game = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = original.GameResult(
                        game=game,
                        summary=GameSummary(
                            mode="qwen-v-checkpoint",
                            failure=f"{type(error).__name__}: {error}",
                        ),
                    )
                results.append(result)
                if not result.summary.completed:
                    stop_reason = result.summary.failure or "game incomplete"
                print(
                    f"GAME {game.index}: completed={result.summary.completed} winner={result.summary.winner} seconds={result.summary.game_seconds:.3f}",
                    flush=True,
                )
            retain()
            # A summary can exist before its process finishes writing startup
            # evidence. Only archive settled futures while other games run.
            original.collect(
                out, settled_games={f"game-{r.game.index:03d}" for r in results}
            )
            if stop_reason is None:
                for _ in done:
                    admit()
    # Every child has exited; preserve the final parent denominator and cleanup.
    retain(closed=True)


def run(config: Campaign) -> None:
    from manabot.llm.vllm_server import VLLMServer

    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    games = original.schedule(config)
    _write(
        out / "schedule.json",
        json.dumps([g.model_dump(mode="json") for g in games], indent=2),
    )
    _write(
        out / "games.json",
        json.dumps(
            {
                "version": "llm-game-sequence-v1",
                "mode": "concurrent",
                "games": [f"game-{g.index:03d}" for g in games],
            }
        ),
    )
    _write(out / "campaign-config.json", config.model_dump_json(indent=2))
    _write(
        out / "campaign.json",
        json.dumps(
            {
                "planned": len(games),
                "attempted": 0,
                "completed": 0,
                "stop_reason": "preparing shared server",
                "session_closed": False,
                "concurrency": config.concurrency,
            }
        ),
    )
    inference = InferenceConfig.model_validate_json(original.INFERENCE.read_text())
    started = time.monotonic()
    try:
        with VLLMServer(inference, out, config.concurrency) as server:
            _write(
                out / "readiness.json",
                json.dumps(
                    {
                        "seconds": server.seconds,
                        "boundary": "shared vLLM spawn/load/compile to healthy; charged once outside game windows",
                    }
                ),
            )
            original.supervise(config, worker_module="experiments.qwen_mtg183_vllm")
    except BaseException as error:
        receipt = json.loads((out / "campaign.json").read_text())
        receipt.update(
            stop_reason=f"{type(error).__name__}: {error}", session_closed=True
        )
        _write(out / "campaign.json", json.dumps(receipt, indent=2))
        raise
    finally:
        _write(
            out / "campaign-wall.json",
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "boundary": "server preparation through game execution, collection and server cleanup; excludes Etude setup/publication and this final write",
                }
            ),
        )


def runs(native_artifact: str, native_source: str, attempt_usd: str = "9.40") -> Job:
    """The operator deducts every earlier allocation before setting this cap."""
    ceiling = Decimal(attempt_usd)
    if not ceiling.is_finite() or not Decimal("0") < ceiling <= Decimal("9.40"):
        raise ValueError("attempt_usd must be positive and at most 9.40")
    job = original.runs(native_artifact, native_source)
    return job.model_copy(
        update={
            "name": "mtg188-qwen-vllm-thinking-v-mtg183-u5950",
            "command": "uv run python -m etude.call --function experiments.qwen_mtg183_vllm:run",
            "config": Campaign().model_dump_json(),
            "setup": "sh scripts/qwen-mtg183-vllm-setup.sh",
            "max_usd": ceiling,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    worker(Campaign.model_validate_json(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
