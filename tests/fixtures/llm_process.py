"""Controlled child targets for LLM supervision; no model or native imports."""

from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Literal

from manabot.llm import session as session_module
from manabot.llm.schema import (
    Attempt,
    DecisionInput,
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
    SelectionResult,
)
from manabot.llm.selection import Runtime

if TYPE_CHECKING:
    from experiments.qwen_mtg183 import ScheduledGame


class CountingRuntime:
    def __init__(self, config: InferenceConfig) -> None:
        self.calls = 0
        self.identity = RuntimeIdentity(
            runtime="test",
            runtime_version="1",
            torch_version="unused",
            python_version="test",
            device_name="cpu",
            chat_template="test",
            generation_config={},
        )

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        return EncodedPrompt(text=messages[0].content, token_ids=(1, 2))

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        if "exit" in prompt.text:
            os._exit(7)
        if "stall" in prompt.text:
            time.sleep(20)
        response = GeneratedResponse(
            text=f"A{self.calls}", token_ids=(3,), finish_reason="eos"
        )
        self.calls += 1
        return response

    def peak_gpu_bytes(self) -> int | None:
        return None


class FirstReferenceRuntime(CountingRuntime):
    """Always answer A0; the native game owns what that reference means."""

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        return GeneratedResponse(text="A0", token_ids=(3,), finish_reason="eos")


ReplyMismatch = Literal["request", "version", "decision", "configuration", "runtime"]


def wrong_reply(
    connection: Connection,
    mismatch: ReplyMismatch,
    config: InferenceConfig,
    factory: Callable[[InferenceConfig], Runtime],
) -> None:
    """Corrupt one binding while keeping the reply otherwise well formed."""
    with connection:
        identity = factory(config).identity
        version = identity.behavior_version(config)
        connection.send(identity)
        request = connection.recv()
        result = SelectionResult(
            decision=request.decision.model_copy(update={"text": "previous"})
            if mismatch == "decision"
            else request.decision,
            config=config.model_copy(update={"max_new_tokens": 5})
            if mismatch == "configuration"
            else config,
            runtime=identity.model_copy(update={"runtime_version": "other"})
            if mismatch == "runtime"
            else identity,
            behavior_version="0" * 64 if mismatch == "version" else version,
            selected_reference="A0",
            origin="initial",
        )
        request_id = "previous-request" if mismatch == "request" else request.request_id
        connection.send(session_module._Progress(request_id, result))


def _stall(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    connection.send(
        SelectionResult(
            decision=decision,
            config=config,
            attempts=(
                Attempt(
                    messages=(Message(role="user", content=decision.text),),
                    response=GeneratedResponse(
                        text="bad", token_ids=(42,), finish_reason="eos"
                    ),
                    validation_error="expected a legal reference",
                ),
            ),
        )
    )
    time.sleep(30)


def _exit(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    os._exit(1)


def _complete(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    connection.send(
        SelectionResult(
            decision=decision,
            config=config,
            selected_reference=decision.legal_references[0],
            origin="initial",
        )
    )
    connection.close()


def load_failure(config: InferenceConfig) -> Runtime:
    raise RuntimeError("injected load failure")


def load_stall(config: InferenceConfig) -> Runtime:
    time.sleep(30)
    return CountingRuntime(config)


def load_exit(config: InferenceConfig) -> Runtime:
    os._exit(7)


def game_child(
    connection: Connection, mode: str, game: "ScheduledGame", out: Path, deadline: float
) -> None:
    """Bounded game IPC fixture, with no inference or native play."""
    from experiments.qwen_mtg183 import GameResult
    from manabot.verify.llm_game import GameSummary

    with connection:
        connection.send("bad" if mode == "admission" else "ready")
        if connection.recv() != "play":
            return
        if mode == "exit":
            os._exit(7)
        if mode == "deadline":
            time.sleep(30)
            return
        result = GameResult(
            game=game, summary=GameSummary(mode="fixture", completed=True)
        )
        if mode == "mismatch":
            result.game = game.model_copy(update={"index": game.index + 1})
        connection.send(result)
