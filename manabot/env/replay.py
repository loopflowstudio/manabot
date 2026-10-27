"""Policy-free replay of private native inputs and shared CLI admission.

Capture owns immutable evidence; this module only reconstructs it. Both replay
CLIs use the same subprocess deadline and runtime checks, while their distinct
receipt formats retain their own validation and result semantics.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Never

import numpy as np

from manabot.env.failures import FailureEvidence, GameInputs, runtime_identities
import managym


@dataclass
class ReplayAgreement:
    """Independent diagnostic checks, never a revision of the retained outcome.

    None means unavailable or not evaluated. Reaching the boundary means its
    execution was attempted, including advancement after repair; it does not
    establish binding or error agreement.
    """

    original_outcome: str
    matches_binding: bool | None = None
    at_retained_failure: bool = False
    matches_error: bool | None = None


def matches_retained_error(
    error: BaseException, error_class: str | None, message: str | None
) -> bool | None:
    """Compare class/message without the vector wrapper's native diagnostics.

    Only the comparison drops the appended payload; capture and returned error
    text retain it verbatim. Missing historical metadata cannot imply agreement.
    """
    if error_class is None or message is None:
        return None
    # Isolated Rust settlement names its variants; PyO3 exports the same errors
    # as Python exception classes. Preserve both spellings in retained evidence.
    replay_class = {
        "AgentError": managym.AgentError.__name__,
        "NativePanic": "PanicException",
    }.get(error_class, error_class)
    return (
        type(error).__name__ == replay_class
        and str(error).partition("; native_failure=")[0]
        == message.partition("; native_failure=")[0]
    )


class _FrameMismatch(ValueError):
    """Reconstruction diverged before the retained native call was attempted."""


def run_bounded_replay(module: str, seconds: float) -> Never:
    """Run the CLI worker with its original arguments under a hard time limit."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module, *sys.argv[1:], "--worker"],
            timeout=seconds,
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "replay_timeout", "seconds": seconds}))
        raise SystemExit(2)
    raise SystemExit(result.returncode)


def require_replay_runtime(
    identities: Mapping[str, object],
    *,
    allow_changed_runtime: bool,
    require_source: bool = False,
) -> str:
    """Admit native bytes and require automatic capture's source digest on request.

    Changed-runtime admission never permits another world. Historical arena and
    tolerance receipts retain other source identities; their replay contract only
    checks native bytes. Return the current native digest for diagnostic output.
    """
    if identities["world"] != managym.WORLD_VERSION:
        raise ValueError("world differs from retained failure")
    native = hashlib.sha256(Path(managym._managym.__file__).read_bytes()).hexdigest()
    if not allow_changed_runtime:
        if native != identities["engine_extension_sha256"]:
            raise ValueError(
                "native bytes differ; explicit --allow-changed-runtime required"
            )
        if (
            require_source
            and identities["source_sha256"] != runtime_identities()["source_sha256"]
        ):
            raise ValueError(
                "source bytes differ; explicit --allow-changed-runtime required"
            )
    return native


def replay_failure_inputs(evidence: FailureEvidence) -> list[dict[str, object]]:
    """Replay only retained inputs. The CLI supplies the process wall-time bound."""
    return [_replay_game_inputs(evidence, game) for game in evidence.games]


def _replay_game_inputs(
    evidence: FailureEvidence, game: GameInputs
) -> dict[str, object]:
    """Reconstruct one stream; keep binding divergence out of native failures."""
    # A strict vector exception records the batch outcome, without proving
    # which individual streams advanced. Replay cannot fill in that history.
    agreement = ReplayAgreement(
        "execution_outcome_unknown"
        if evidence.mode == "vector"
        else evidence.execution_outcome
    )
    # A reset/search failure has a different boundary from a rejected game
    # input. Compare both phase and index, including empty prefixes (-1).
    if evidence.operation == "reset":
        retained_phase, retained_index = "reset", -1
    elif evidence.search is not None:
        retained_phase = "search"
        retained_index = len(evidence.search.operations) - 1
    else:
        retained_phase, retained_index = "prefix", len(game.actions) - 1
    env = (
        managym.VectorEnv(
            num_envs=1,
            seed=game.seed,
            seed_stride=evidence.seed_stride,
            skip_trivial=evidence.skip_trivial,
            opponent_policy=evidence.opponent_policy,
        )
        if evidence.mode == "vector"
        else managym.Env(seed=game.seed, skip_trivial=evidence.skip_trivial)
    )
    index = -1
    phase = "reset"
    try:
        agreement.at_retained_failure = retained_phase == "reset"
        if evidence.mode == "vector":
            env.reset_all([p.native() for p in game.players])
        else:
            env.reset([p.native() for p in game.players])
        phase = "prefix"
        for index, action in enumerate(game.actions):
            if (
                index == len(game.actions) - 1
                and game.frame is not None
                and evidence.mode != "search"
            ):
                frame = (
                    env.semantic_decision_frames_json()[0]
                    if evidence.mode == "vector"
                    else env.semantic_decision_frame_json()
                )
                agreement.matches_binding = frame == game.frame
                if not agreement.matches_binding:
                    raise _FrameMismatch("retained pre-command frame differs")
            agreement.at_retained_failure = (
                phase == retained_phase and index == retained_index
            )
            if evidence.mode == "vector":
                env.step([action])
            elif isinstance(action, str):
                env.execute_semantic_command_json(action)
            else:
                env.step(action)
        if evidence.search is not None:
            if game.frame is not None:
                agreement.matches_binding = (
                    env.semantic_decision_frame_json() == game.frame
                )
                if not agreement.matches_binding:
                    raise _FrameMismatch("retained search root frame differs")
            search = evidence.search
            phase = "search"
            index = -1
            agreement.at_retained_failure = retained_index == index
            pool = getattr(env, search.method)(*search.arguments)
            if search.method == "rollout_pool" and search.buffers:
                buffers = {
                    key: np.zeros(spec.shape, dtype=spec.dtype)
                    for key, spec in search.buffers.items()
                }
                pool.set_buffers(buffers, search.capacity)
                pool.encode_active()
                for index, operation in enumerate(search.operations):
                    agreement.at_retained_failure = retained_index == index
                    if operation == "finish_random":
                        pool.finish_random()
                    else:
                        pool.step_active(operation)
    except _FrameMismatch as error:
        result: dict[str, object] = {
            "status": "binding_mismatch",
            "action_index": index,
            "phase": phase,
            "diagnostic": str(error),
        }
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        agreement.matches_error = matches_retained_error(
            error, evidence.error_class, evidence.error
        )
        result = {
            "status": "failed",
            "action_index": index,
            "error_class": type(error).__name__,
            "error": str(error),
            "phase": phase,
        }
    else:
        result = {"status": "advanced", "actions": len(game.actions)}
    return {**asdict(agreement), "stream": game.stream, **result}
