"""Sequential LLM baselines using the existing cache, session and game owners."""

from __future__ import annotations

from pathlib import Path
import time
from typing import TYPE_CHECKING

from pydantic import JsonValue

from manabot.llm.cache import CacheManifest, prepare
from manabot.llm.schema import InferenceConfig, SelectionResult
from manabot.llm.selection import Runtime
from manabot.llm.session import InferenceSession

if TYPE_CHECKING:
    from experiments.inference_baseline import Cell, LLMWorkload, Protocol


def _runtime(config: InferenceConfig) -> Runtime:
    # Apply the frozen experiment settings in the spawned runtime itself.
    import torch

    from manabot.llm.transformers_runtime import TransformersRuntime

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return TransformersRuntime(config)


def execute(
    protocol: Protocol,
    workload: LLMWorkload,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> dict[str, JsonValue]:
    from experiments.inference_baseline import _write

    if cell.execution is not None:
        raise ValueError(
            "LLM cells use their pinned InferenceConfig, not policy execution settings"
        )
    config = InferenceConfig.model_validate_json(workload.config.read_bytes())
    if config.deadline_seconds > 120:
        raise ValueError("LLM decision deadline exceeds 120 seconds")
    _write(
        out / "games.json",
        {
            "version": "llm-game-sequence-v1",
            "games": [f"game-{i:03d}" for i in range(len(workload.games))],
        },
    )
    receipt = prepare(
        config,
        CacheManifest.model_validate_json(workload.cache_manifest.read_bytes()),
        offline=True,
    )
    _write(out / "cache.json", receipt)
    ready_start = time.monotonic()
    session = None
    try:
        session = InferenceSession(
            config,
            factory=_runtime,
            load_seconds=min(workload.load_seconds, max(0, deadline - time.time())),
        )
        _write(
            out / "readiness.json",
            {
                "seconds": session.load_seconds,
                "ready": True,
                "boundary": "spawn through runtime construction and ready IPC",
            },
        )
        if replay:
            return _retained(protocol, workload, session, out)
        from manabot.verify.llm_game import GameConfig, play

        completed = decisions = 0
        for index, path in enumerate(workload.games):
            game = GameConfig.model_validate_json(path.read_bytes())
            if game.max_decisions > 2000:
                raise ValueError("LLM game Command ceiling exceeds 2000")
            if time.time() >= deadline:
                raise TimeoutError("LLM sequence deadline exhausted")
            game = game.model_copy(
                update={"max_seconds": min(game.max_seconds, deadline - time.time())}
            )
            summary = play(game, out / f"game-{index:03d}", session)
            decisions += summary.decisions
            if not summary.completed or not summary.replay_matches:
                return {
                    "status": "failed",
                    "reason": summary.failure or "replay mismatch",
                    "completed_games": completed,
                    "committed_decisions": decisions,
                }
            completed += 1
        return {
            "status": "completed",
            "reason": "complete replay-verified sequential games",
            "completed_games": completed,
            "committed_decisions": decisions,
        }

    finally:
        if session is None:
            _write(
                out / "readiness.json",
                {
                    "seconds": time.monotonic() - ready_start,
                    "ready": False,
                    "boundary": "failed session construction including its cleanup",
                },
            )
        else:
            started = time.monotonic()
            session.close()
            _write(
                out / "session-cleanup.json",
                {"seconds": time.monotonic() - started, "closed": session.closed},
            )


def _retained(
    protocol: Protocol, workload: LLMWorkload, session: InferenceSession, out: Path
) -> dict[str, JsonValue]:
    from experiments.inference_baseline import _write

    if not workload.corpus:
        raise ValueError("LLM retained-input execution requires an ordered corpus")
    for pass_index in range(protocol.recipe.warmup_passes + 1):
        started = time.monotonic()
        for ordinal, source in enumerate(workload.corpus):
            old = SelectionResult.model_validate_json(source.read_bytes())
            if (
                old.config != session.config
                or not old.attempts
                or old.attempts[0].prompt is None
            ):
                raise ValueError(
                    "retained LLM configuration/prompt differs or is unavailable"
                )
            step = out / f"pass-{pass_index}" / f"decision-{ordinal:05d}"
            step.mkdir(parents=True)

            def progress(result: SelectionResult) -> None:
                _write(step / "selection.json", result)

            result = session.decide(
                old.decision, expected_prompt=old.attempts[0].prompt, progress=progress
            )
            progress(result)
            if result.failure:
                raise RuntimeError(result.failure.model_dump_json())
        _write(
            out / f"pass-{pass_index}.json",
            {
                "population": "warmup"
                if pass_index < protocol.recipe.warmup_passes
                else "measured",
                "requests": len(workload.corpus),
                "seconds": time.monotonic() - started,
            },
        )
    return {
        "status": "completed",
        "reason": "exact retained prompts; not game throughput",
        "roots": len(workload.corpus),
    }
