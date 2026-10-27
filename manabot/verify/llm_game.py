"""A bounded full-game proof with native text, a warm model and command replay.

One seat uses the ordinary LLM protocol; the other uses a frozen checkpoint or
samples legal actions uniformly. Every capture, selection and native transition
is retained before the next decision. One game cannot establish playing strength.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import TYPE_CHECKING

from managym.decision import apply_semantic_command
from pydantic import BaseModel, Field

from manabot.infra.config import MatchConfig
from manabot.llm.schema import DecisionInput, InferenceConfig, SelectionResult
from manabot.llm.session import InferenceSession
from manabot.semantic.text_observation import (
    capture_text_decision,
    resolve_action,
    submit_response,
)
import managym

if TYPE_CHECKING:
    from manabot.llm.openrouter import OpenRouterResult, OpenRouterSession


class GameConfig(BaseModel):
    inference: Path = Path("configs/llm/qwen3.5-9b.json")
    seed: int = 23
    opponent_seed: int = 24
    model_seat: int = Field(default=0, ge=0, le=1)
    history_capacity: int = Field(default=32, ge=0, le=32)
    max_decisions: int = Field(default=2000, gt=0)
    max_seconds: float = Field(default=3600, gt=0)
    deck: dict[str, int] = {"Mountain": 24, "Gray Ogre": 24, "Lightning Bolt": 12}
    match: MatchConfig | None = None
    opponent_checkpoint: Path | None = None
    # Stop before another decision after this reported spend. An in-flight
    # decision (including its repair) can exceed the threshold; this is no hard cap.
    max_reported_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class GameSummary(BaseModel):
    completed: bool = False
    failure: str | None = None
    winner: int | None = None
    decisions: int = 0
    model_decisions: int = 0
    origins: dict[str, int] = {}
    max_prompt_tokens: int = 0
    decision_seconds_median: float | None = None
    decision_seconds_max: float | None = None
    game_seconds: float = 0
    model_load_seconds: float | None = None
    peak_gpu_bytes: int | None = None
    replay_matches: bool = True
    final_digest: str | None = None
    mode: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    reported_cost_usd: float = 0
    # Totals above include only reported values, including repairs and failures.
    unpriced_attempts: int = 0


def _write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(payload + "\n")
    temporary.replace(path)


def _game_mode(
    session: "InferenceSession | OpenRouterSession | None", checkpoint: bool
) -> str:
    if session is None:
        return "random-v-checkpoint-control" if checkpoint else "random-control"
    model = "openrouter" if session.identity.runtime == "openrouter-chat-v1" else "qwen"
    return f"{model}-v-{'checkpoint' if checkpoint else 'random'}"


def _record_hosted_usage(summary: GameSummary, result: "OpenRouterResult") -> None:
    """Account for every attempt, including repairs and failed requests.

    Missing billing information is counted separately; zero reported spend
    never implies that a request without usage was free.
    """
    for attempt in result.attempts:
        usage = attempt.completion.usage if attempt.completion else None
        if usage is None or usage.cost is None:
            summary.unpriced_attempts += 1
        if usage is None:
            continue
        summary.prompt_tokens += usage.prompt_tokens
        summary.completion_tokens += usage.completion_tokens
        summary.reported_cost_usd += usage.cost or 0
        if usage.prompt_tokens_details:
            summary.cached_tokens += usage.prompt_tokens_details.cached_tokens or 0
            summary.cache_write_tokens += (
                usage.prompt_tokens_details.cache_write_tokens or 0
            )
        if usage.completion_tokens_details:
            summary.reasoning_tokens += (
                usage.completion_tokens_details.reasoning_tokens or 0
            )


def play(
    config: GameConfig,
    out: Path,
    session: "InferenceSession | OpenRouterSession | None",
    *,
    progress_path: Path | None = None,
) -> GameSummary:
    """Run once; failures never reset the game or retry uncertain execution.

    A missing session selects random actions for the model seat.
    Native trivial-decision skipping is disabled for both seats.
    """
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "config.json", config.model_dump_json(indent=2))
    started = time.monotonic()
    env: managym.Env | None = None
    replay: managym.Env | None = None
    rng = random.Random(config.opponent_seed)
    summary = GameSummary(
        mode=_game_mode(session, config.opponent_checkpoint is not None)
    )
    if session:
        summary.model_load_seconds = session.load_seconds
        _write(out / "runtime.json", session.identity.model_dump_json(indent=2))
    origins: Counter[str] = Counter()
    latencies: list[float] = []
    try:
        env = managym.Env(seed=config.seed, skip_trivial=False)
        if config.match is not None:
            from manabot.env.match import Match

            players = Match(config.match).to_rust()
        else:
            players = [
                managym.PlayerConfig(f"seat-{seat}", config.deck) for seat in (0, 1)
            ]
        opponent = None
        if config.opponent_checkpoint is not None:
            from manabot.verify.llm_opponent import CheckpointOpponent

            opponent = CheckpointOpponent(
                config.opponent_checkpoint, players, config.opponent_seed
            )
            _write(out / "opponent.json", opponent.identity.model_dump_json(indent=2))
        env.reset(players)
        replay = env.clone_env()
        while not env.is_game_over():
            if summary.decisions >= config.max_decisions:
                raise TimeoutError("game decision limit reached")
            if time.monotonic() - started >= config.max_seconds:
                raise TimeoutError("game wall limit reached")
            step = out / f"decision-{summary.decisions:05d}"
            step.mkdir()
            decision = capture_text_decision(
                env, history_capacity=config.history_capacity
            )
            _write(step / "capture.json", decision.capture_json)
            _write(step / "prompt.txt", decision.text)
            actor = env.current_agent_index()
            if session and actor == config.model_seat:
                if config.max_reported_cost_usd is not None:
                    if summary.unpriced_attempts:
                        raise RuntimeError(
                            "cannot enforce spend threshold with missing costs"
                        )
                    if summary.reported_cost_usd >= config.max_reported_cost_usd:
                        raise RuntimeError("reported spend threshold reached")
                supplied = DecisionInput(
                    text=decision.text,
                    legal_references=tuple(a.reference for a in decision.actions),
                    representation_id=decision.representation_version,
                    source_identity={"capture_digest": decision.capture_digest},
                )

                def progress(result: "SelectionResult | OpenRouterResult") -> None:
                    _write(step / "selection.json", result.model_dump_json(indent=2))

                result = session.decide(supplied, progress=progress)
                progress(result)
                summary.model_decisions += 1
                if not isinstance(result, SelectionResult):
                    _record_hosted_usage(summary, result)
                if result.failure is not None:
                    raise RuntimeError(result.failure.model_dump_json())
                if result.selected_reference is None or result.origin is None:
                    raise RuntimeError("model returned no selection")
                reference, producer = result.selected_reference, result.origin
                latencies.append(result.seconds)
                summary.max_prompt_tokens = max(
                    summary.max_prompt_tokens,
                    *(a.prompt_tokens or 0 for a in result.attempts),
                )
                summary.peak_gpu_bytes = result.peak_gpu_bytes
            elif opponent is not None and actor != config.model_seat:
                reference = opponent.select(env, decision)
                producer = "checkpoint-opponent"
            else:
                reference = rng.choice(decision.actions).reference
                producer = "seeded-uniform-opponent" if session else "random-control"
            origins[producer] += 1
            command = resolve_action(decision, reference)
            # Retain intent before the potentially mutating native call.
            _write(step / "command.json", command.to_json())
            _write(
                step / "choice.json",
                json.dumps(
                    {
                        "actor": actor,
                        "reference": reference,
                        "producer": producer,
                    }
                ),
            )
            transition = submit_response(env, decision, reference)
            _write(step / "transition.json", json.dumps(asdict(transition)))
            replayed = apply_semantic_command(replay, command)
            summary.replay_matches = (
                replayed == transition and replay.state_digest() == env.state_digest()
            )
            if not summary.replay_matches:
                raise RuntimeError("native command replay diverged")
            summary.decisions += 1
            summary.origins = dict(origins)
            summary.game_seconds = time.monotonic() - started
            _write(out / "progress.json", summary.model_dump_json(indent=2))
            if progress_path is not None:
                _write(progress_path, summary.model_dump_json(indent=2))
            if summary.decisions % 10 == 0:
                print(
                    f"decisions={summary.decisions} model={summary.model_decisions} seconds={summary.game_seconds:.1f}",
                    flush=True,
                )
        summary.completed = True
        summary.winner = env.winner_index()
    except Exception as error:
        summary.failure = f"{type(error).__name__}: {error}"
    finally:
        summary.game_seconds = time.monotonic() - started
        if env is not None and replay is not None:
            summary.final_digest = env.state_digest()
        summary.origins = dict(origins)
        if latencies:
            summary.decision_seconds_median = statistics.median(latencies)
            summary.decision_seconds_max = max(latencies)
        _write(out / "summary.json", summary.model_dump_json(indent=2))
    return summary


def run(config: GameConfig) -> None:
    """Etude entry point; setup has acquired bytes, so inference is offline."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    inference = InferenceConfig.model_validate_json(config.inference.read_text())
    out = Path(os.environ.get("ETUDE_OUTPUT_DIR", ".runs/llm-game"))
    out.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv",
        ],
        text=True,
    )
    if "NVIDIA L4" not in gpu:
        raise RuntimeError(f"L4 proof requires an L4; observed {gpu}")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu",
            "--format=csv",
            "--loop-ms=1000",
            f"--filename={out / 'gpu-memory.csv'}",
        ]
    )
    try:
        with InferenceSession(inference, load_seconds=180) as session:
            summary = play(
                config,
                out.parent / "game-record",
                session,
                progress_path=out / "progress.json",
            )
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        subprocess.run(
            ["sh", "scripts/llm-game-collect.sh"],
            check=True,
            env=dict(os.environ, ETUDE_OUTPUT_DIR=str(out)),
        )
    print(summary.model_dump_json(indent=2), flush=True)
    if not summary.completed:
        raise RuntimeError(summary.failure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument(
        "--openrouter",
        type=Path,
        help="OpenRouter inference config; runs locally against the hosted API",
    )
    args = parser.parse_args()
    if args.random_control and args.openrouter:
        parser.error("--random-control and --openrouter are mutually exclusive")
    config = (
        GameConfig.model_validate_json(args.config.read_text())
        if args.config
        else GameConfig()
    )
    if args.random_control or args.openrouter:
        if args.openrouter:
            from manabot.llm.openrouter import OpenRouterConfig, OpenRouterSession

            inference = OpenRouterConfig.model_validate_json(
                args.openrouter.read_text()
            )
            with OpenRouterSession(inference) as session:
                summary = play(config, args.out, session)
        else:
            summary = play(config, args.out, None)
        print(summary.model_dump_json(indent=2))
        if not summary.completed:
            raise SystemExit(1)
    else:
        os.environ["ETUDE_OUTPUT_DIR"] = str(args.out)
        run(config)


if __name__ == "__main__":
    main()
