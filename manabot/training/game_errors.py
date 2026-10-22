"""Settled game accounting and immutable private gameplay failure inputs.

GameErrorPolicy declares prospective tolerance. GameErrorLedger fsyncs every
settled attempt independently of optimizer snapshots, so an interrupted update
cannot erase its error allowance. The replay CLI executes only retained actions;
it never samples a policy, trains, submits a job or overwrites failure evidence.
"""

import argparse
from collections.abc import Mapping
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from manabot.arena.models import file_sha256
from manabot.env import Match
from manabot.env.failures import read_failure, replay_failure_inputs, runtime_identities
from manabot.infra.hypers import MatchHypers
from manabot.training.self_play import deck_identity
import managym

GameOutcome = Literal[
    "completed", "engine_error", "truncated", "action_limit", "deadline"
]


class GameErrorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    threshold: float = Field(default=0.1, ge=0, lt=1)
    window: int = Field(default=100, ge=1, le=100_000)
    warmup: int = Field(default=100, ge=1)
    max_game_actions: int = Field(default=40_000, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def bounded_warmup(self) -> "GameErrorPolicy":
        if self.warmup > self.window:
            raise ValueError("game error warmup must not exceed window")
        return self


class GameAttempt(BaseModel):
    """One native game, never one seat or one vector call. Inputs are private."""

    model_config = ConfigDict(extra="forbid")
    stream: int
    seed: int
    seat: int
    matchup: MatchHypers
    outcome: GameOutcome
    actions: list[int] = []
    error_class: str | None = None
    error: str | None = None
    frame_json: str | None = None
    native_trace: str | None = None
    discarded_samples: int = 0
    valid_samples: int = 0
    seconds: float = 0


class GameErrorState(BaseModel):
    """Lifetime attempts plus the ordered rolling window of settled attempts."""

    attempted: int = 0
    completed: int = 0
    crashed: int = 0
    discarded_samples: int = 0
    replacement_samples: int = 0
    window: list[bool] = []
    by_cell: dict[str, dict[str, int]] = {}
    stop_reason: str | None = None

    def settle(self, attempt: GameAttempt, policy: GameErrorPolicy) -> None:
        failed = attempt.outcome != "completed"
        self.attempted += 1
        self.completed += int(not failed)
        self.crashed += int(failed)
        self.discarded_samples += attempt.discarded_samples
        self.replacement_samples += min(self.replacement_debt, attempt.valid_samples)
        self.window = [*self.window, failed][-policy.window :]
        match = attempt.matchup
        cell = ":".join(
            (
                deck_identity(match.hero_deck, match.hero_sideboard),
                deck_identity(match.villain_deck, match.villain_sideboard),
                str(attempt.seat),
            )
        )
        counts = self.by_cell.setdefault(cell, {})
        counts[attempt.outcome] = counts.get(attempt.outcome, 0) + 1
        if len(self.window) >= policy.warmup and self.error_rate > policy.threshold:
            self.stop_reason = "game_error_rate_exceeded"

    @computed_field
    @property
    def replacement_debt(self) -> int:
        return self.discarded_samples - self.replacement_samples

    @computed_field
    @property
    def error_rate(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0


class GameErrorLimit(RuntimeError):
    """A retained rolling game error allowance has been exhausted."""


class GameErrorLedger:
    """Append-only local receipts. Parent ledgers remain immutable on resume.

    A malformed/torn ledger fails closed: no guessed counter can grant more
    recovery allowance. These paths are private trusted TrainingRun artifacts.
    """

    def __init__(
        self,
        path: Path,
        policy: GameErrorPolicy,
        *,
        parents: list[str] | None = None,
    ) -> None:
        self.path = path
        self.policy = policy
        self.state = GameErrorState()
        expected_policy = policy.model_dump(mode="json")
        for parent in parents or []:
            with Path(parent).open() as stream:
                for line in stream:
                    entry = json.loads(line)
                    if entry["policy"] != expected_policy:
                        raise ValueError("game error ledger policy mismatch")
                    self.state.settle(
                        GameAttempt.model_validate(entry["attempt"]), policy
                    )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=False)

    def record(
        self, attempt: GameAttempt, context: Mapping[str, object]
    ) -> GameErrorState:
        entry = {
            "schema": 1,
            "policy": self.policy.model_dump(mode="json"),
            "attempt": attempt.model_dump(mode="json"),
            "context": dict(context),
        }
        # Retain the original prefix before updating counters or resetting native
        # state. A crash between this fsync and a snapshot still counts on restart.
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.state.settle(attempt, self.policy)
        return self.state


def replay_attempt(attempt: GameAttempt) -> dict[str, object]:
    """Bounded positional replay; the final pre-command frame exposes the offer.

    AgentError and PyO3 panic are recorded here only around native gameplay.
    Process interrupts escape. Divergence before the retained failing action is
    distinct from reproducing the original failure.
    """
    env = managym.Env(seed=attempt.seed, skip_trivial=True)
    env.reset(Match(attempt.matchup).to_rust())
    frame: str | None = None
    for index, action in enumerate(attempt.actions):
        if index == len(attempt.actions) - 1:
            frame = env.semantic_decision_frame_json()
        try:
            _, _, terminated, truncated, _ = env.step(action)
        except BaseException as error:
            if (
                not isinstance(error, managym.AgentError)
                and type(error).__name__ != "PanicException"
            ):
                raise
            return {
                "status": "failed",
                "action_index": index,
                "frame": frame,
                "error_class": type(error).__name__,
                "error": str(error),
                "at_retained_failure": index == len(attempt.actions) - 1,
            }
        if terminated or truncated:
            return {
                "status": "completed" if terminated else "truncated",
                "action_index": index,
                "frame": frame,
            }
    return {"status": "advanced", "actions": len(attempt.actions), "frame": frame}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--attempt", type=int, required=True, help="One-based receipt line"
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--allow-changed-runtime", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.attempt < 1 or not 0 < args.seconds <= 300:
        parser.error("attempt must be positive and seconds in (0, 300]")
    if not args.worker:
        # A native hang cannot defeat the parent wall-time bound.
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "manabot.training.game_errors",
                    *sys.argv[1:],
                    "--worker",
                ],
                timeout=args.seconds,
            )
        except subprocess.TimeoutExpired:
            print(json.dumps({"status": "replay_timeout", "seconds": args.seconds}))
            raise SystemExit(2)
        raise SystemExit(result.returncode)
    with args.ledger.open() as stream:
        line = next(islice(stream, args.attempt - 1, args.attempt), None)
    if line is None:
        parser.error("attempt is beyond the end of the ledger")
    entry = json.loads(line)
    if entry.get("schema") == "native-failure-v1":
        evidence = read_failure(args.ledger)
        native = file_sha256(Path(managym._managym.__file__))
        if evidence.identities["world"] != managym.WORLD_VERSION:
            parser.error("world differs from retained failure")
        if (
            native != evidence.identities["engine_extension_sha256"]
            and not args.allow_changed_runtime
        ):
            parser.error(
                "native bytes differ; explicit --allow-changed-runtime required"
            )
        if (
            not args.allow_changed_runtime
            and evidence.identities["source_sha256"]
            != runtime_identities()["source_sha256"]
        ):
            parser.error(
                "source bytes differ; explicit --allow-changed-runtime required"
            )
        print(
            json.dumps(
                {
                    "status": "replayed",
                    "results": replay_failure_inputs(evidence),
                    "original_outcome": evidence.execution_outcome,
                }
            )
        )
        return
    attempt = GameAttempt.model_validate(entry["attempt"])
    if attempt.outcome == "completed":
        parser.error("select a failed attempt")
    identities = entry["context"]["identities"]
    native = file_sha256(Path(managym._managym.__file__))
    if identities["world"] != managym.WORLD_VERSION:
        parser.error("world differs from retained failure")
    if (
        native != identities["engine_extension_sha256"]
        and not args.allow_changed_runtime
    ):
        parser.error("native bytes differ; explicit --allow-changed-runtime required")
    start = time.perf_counter()
    result = replay_attempt(attempt)
    print(
        json.dumps(
            {**result, "seconds": time.perf_counter() - start, "native_sha256": native}
        )
    )


if __name__ == "__main__":
    main()
