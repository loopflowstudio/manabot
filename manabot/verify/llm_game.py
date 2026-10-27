"""A bounded full-game proof with native text, a warm model and command replay.

One seat uses the ordinary LLM protocol; the other uses an admitted checkpoint
or samples the complete legal action set uniformly. Every capture and transition is retained
before the next decision. This is a functional test, not a strength evaluation.
"""

import argparse
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from managym.decision import DecisionFrame, apply_semantic_command
from pydantic import BaseModel, Field
import torch

from manabot.env.match import Match
from manabot.infra.artifacts import verify_file
from manabot.infra.config import MatchConfig
from manabot.llm.schema import DecisionInput, InferenceConfig, SelectionResult
from manabot.llm.session import InferenceSession
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.semantic.text_observation import (
    TextDecision,
    capture_text_decision,
    resolve_action,
    submit_response,
)
import managym


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
    skip_trivial: bool = False


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
    # Monotonic host intervals; native execution includes capture validation.
    observations: dict[str, float] = Field(default_factory=dict)
    model_latencies: list[float] = Field(default_factory=list)
    opponent_latencies: list[float] = Field(default_factory=list)
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    output_tokens: int = 0
    token_split_attempts: int = 0
    generation_attempts: int = 0
    llm_phases: dict[str, float] = Field(default_factory=dict)


@dataclass(frozen=True)
class PolicyChoice:
    reference: str
    encoding_seconds: float
    inference_seconds: float


class CheckpointOpponent:
    """One resident CPU policy; tensor offers bind back to native Commands.

    Admission and loading happen once outside the game window. The complete
    native offer space must fit the checkpoint encoder; truncation fails closed.
    """

    def __init__(self, path: Path, sha256: str, size: int) -> None:
        started = time.monotonic()
        verify_file(path, sha256, size)
        self.agent, self.space = load_checkpoint(str(path))
        if self.agent.config.compound_decisions or self.agent.belief_count_buckets:
            raise ValueError("this comparison requires the flat stateless checkpoint")
        self.agent.eval()
        self.load_seconds = time.monotonic() - started
        self.sha256 = sha256

    def admit(self, players: list[managym.PlayerConfig], seed: int) -> None:
        validate_agent_setup(self.agent, players, allow_deck_repetition=True)
        torch.manual_seed(seed)

    def choose(self, env: managym.Env, decision: TextDecision) -> PolicyChoice:
        started = time.monotonic()
        actor = env.current_agent_index()
        raw = env.observation_for_player(actor)
        encoder = self.space.encoder
        if (
            len(raw.action_space.actions) > encoder.max_actions
            or max(len(raw.agent_cards), len(raw.opponent_cards))
            > encoder.cards_per_player
            or max(len(raw.agent_permanents), len(raw.opponent_permanents))
            > encoder.perms_per_player
        ):
            raise ValueError(
                "checkpoint encoding would truncate native observations or offers"
            )
        encoded = self.space.encode(raw)
        encoding_seconds = time.monotonic() - started
        started = time.monotonic()
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for name, value in encoded.items()
        }
        with torch.inference_mode():
            action, _, _, _ = self.agent.get_action_and_value(tensors)
        index = int(action.item())
        inference_seconds = time.monotonic() - started
        if index < 0 or index >= len(encoded["actions_valid"]):
            raise ValueError("policy returned an invalid action index")
        if not encoded["actions_valid"][index]:
            raise ValueError("policy selected a masked action")
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        offer_id = int(frame.offers[index]["id"])
        choices = [a for a in decision.actions if a.command.offer_id == offer_id]
        if len(choices) != 1:
            raise ValueError("tensor action has no unique captured semantic reference")
        return PolicyChoice(choices[0].reference, encoding_seconds, inference_seconds)


def _write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(payload + "\n")
    temporary.replace(path)


def play(
    config: GameConfig,
    out: Path,
    session: InferenceSession | None,
    *,
    progress_path: Path | None = None,
    opponent: CheckpointOpponent | None = None,
) -> GameSummary:
    """Run once; failures never reset the game or retry uncertain execution.

    A missing session selects the explicit random-v-random plumbing control.
    Native trivial-decision skipping is declared and identical for both seats.
    """
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "config.json", config.model_dump_json(indent=2))
    started = time.monotonic()
    observations: dict[str, float] = {}

    @contextmanager
    def observe(name: str) -> Iterator[None]:
        phase_start = time.monotonic()
        try:
            yield
        finally:
            observations[name] = (
                observations.get(name, 0) + time.monotonic() - phase_start
            )

    def retain(path: Path, payload: str) -> None:
        with observe("evidence_write"):
            _write(path, payload)

    players = (
        Match(config.match).to_rust()
        if config.match is not None
        else [managym.PlayerConfig(f"seat-{seat}", config.deck) for seat in (0, 1)]
    )
    if opponent is not None:
        opponent.admit(players, config.opponent_seed)
    with observe("native_reset"):
        env = managym.Env(seed=config.seed, skip_trivial=config.skip_trivial)
        env.reset(players)
    with observe("clone_replay"):
        replay = env.clone_env()
    rng = random.Random(config.opponent_seed)
    summary = GameSummary(
        mode="qwen-v-checkpoint"
        if opponent
        else "qwen-v-random"
        if session
        else "random-control"
    )
    if session:
        summary.model_load_seconds = session.load_seconds
        retain(out / "runtime.json", session.identity.model_dump_json(indent=2))
    origins: Counter[str] = Counter()
    latencies: list[float] = []
    try:
        while not env.is_game_over():
            if summary.decisions >= config.max_decisions:
                raise TimeoutError("game decision limit reached")
            if time.monotonic() - started >= config.max_seconds:
                raise TimeoutError("game wall limit reached")
            step = out / f"decision-{summary.decisions:05d}"
            step.mkdir()
            with observe("render_capture"):
                decision = capture_text_decision(
                    env, history_capacity=config.history_capacity
                )
            retain(step / "capture.json", decision.capture_json)
            retain(step / "prompt.txt", decision.text)
            actor = env.current_agent_index()
            if session and actor == config.model_seat:
                supplied = DecisionInput(
                    text=decision.text,
                    legal_references=tuple(a.reference for a in decision.actions),
                    representation_id=decision.representation_version,
                    source_identity={"capture_digest": decision.capture_digest},
                )

                def progress(result: SelectionResult) -> None:
                    retain(step / "selection.json", result.model_dump_json(indent=2))

                write_before = observations.get("evidence_write", 0)
                selection_start = time.monotonic()
                try:
                    result = session.decide(supplied, progress=progress)
                finally:
                    selection_wall = time.monotonic() - selection_start
                    # Progress writes are nested synchronous host work.
                    observations["llm_wait_and_dispatch"] = (
                        observations.get("llm_wait_and_dispatch", 0)
                        + selection_wall
                        - (observations.get("evidence_write", 0) - write_before)
                    )
                progress(result)
                summary.model_decisions += 1
                summary.model_latencies.append(result.seconds)
                for attempt in result.attempts:
                    summary.generation_attempts += 1
                    summary.output_tokens += attempt.output_tokens or 0
                    for name in (
                        "encode_seconds",
                        "generation_seconds",
                        "validation_seconds",
                    ):
                        seconds = getattr(attempt, name)
                        if seconds is not None:
                            summary.llm_phases[name] = (
                                summary.llm_phases.get(name, 0) + seconds
                            )
                    if attempt.response is not None:
                        if attempt.response.reasoning_tokens is not None:
                            summary.token_split_attempts += 1
                        summary.reasoning_tokens += (
                            attempt.response.reasoning_tokens or 0
                        )
                        summary.answer_tokens += attempt.response.answer_tokens or 0
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
            elif opponent is not None:
                choice = opponent.choose(env, decision)
                reference, producer = choice.reference, "checkpoint-stochastic-cpu"
                summary.opponent_latencies.append(choice.inference_seconds)
                for name, seconds in (
                    ("policy_encoding", choice.encoding_seconds),
                    ("policy_inference", choice.inference_seconds),
                ):
                    observations[name] = observations.get(name, 0) + seconds
                retain(step / "policy.json", json.dumps(asdict(choice)))
            else:
                reference = rng.choice(decision.actions).reference
                producer = "seeded-uniform-opponent" if session else "random-control"
            origins[producer] += 1
            command = resolve_action(decision, reference)
            # Retain intent before the potentially mutating native call.
            retain(step / "command.json", command.to_json())
            retain(
                step / "choice.json",
                json.dumps(
                    {
                        "actor": actor,
                        "reference": reference,
                        "producer": producer,
                    }
                ),
            )
            with observe("native_command"):
                transition = submit_response(env, decision, reference)
            retain(step / "transition.json", json.dumps(asdict(transition)))
            with observe("clone_replay"):
                replayed = apply_semantic_command(replay, command)
                summary.replay_matches = (
                    replayed == transition
                    and replay.state_digest() == env.state_digest()
                )
            if not summary.replay_matches:
                raise RuntimeError("native command replay diverged")
            summary.decisions += 1
            summary.origins = dict(origins)
            summary.game_seconds = time.monotonic() - started
            retain(out / "progress.json", summary.model_dump_json(indent=2))
            if progress_path is not None:
                retain(progress_path, summary.model_dump_json(indent=2))
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
        summary.final_digest = env.state_digest()
        summary.origins = dict(origins)
        if latencies:
            summary.decision_seconds_median = statistics.median(latencies)
            summary.decision_seconds_max = max(latencies)
        summary.observations = dict(observations)
        summary.observations["unattributed"] = max(
            0, summary.game_seconds - sum(observations.values())
        )
        retain(out / "summary.json", summary.model_dump_json(indent=2))
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
    args = parser.parse_args()
    config = (
        GameConfig.model_validate_json(args.config.read_text())
        if args.config
        else GameConfig()
    )
    if args.random_control:
        summary = play(config, args.out, None)
        print(summary.model_dump_json(indent=2))
        if not summary.completed:
            raise SystemExit(1)
    else:
        os.environ["ETUDE_OUTPUT_DIR"] = str(args.out)
        run(config)


if __name__ == "__main__":
    main()
