"""Two policy-free reconstructions of private native inputs.

RecordWalker verifies what happened, never attempts pending calls, and admits
any build of the same world and cards. Failure reproduction reconstructs the
original execution mode and attempts the retained call, requiring identical
native bytes and source unless changed-runtime admission is explicit. Its
prefix loop remains separate because it measures failure agreement, not history.
"""

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from itertools import chain
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal, Never

import numpy as np

import managym
from managym.card_capture import CapturedCards, captured_cards
from managym.decision import Command, DecisionFrame, SemanticContractError
from managym.recording import (
    BufferSpec,
    DecisionAnnotation,
    GameInputs,
    GameRecord,
    Intervention,
    read_record,
    read_records,
    runtime_identities,
    semantic_frame_sha256,
)
from managym.statistics import reduce_cards


@dataclass(frozen=True)
class ReplayStop:
    status: Literal["complete", "interrupted", "diverged"]
    index: int
    reason: str | None = None
    winner: int | None = None


@dataclass(frozen=True)
class RecordedDecision:
    index: int
    observation: managym.Observation
    commitment: int | Command
    annotation: DecisionAnnotation | None
    frame: DecisionFrame | None = None

    @property
    def chosen(self) -> int:
        """Index of the one legal offer the commitment names."""
        if isinstance(self.commitment, int):
            return self.commitment
        if self.frame is None:
            raise ValueError("a semantic commitment needs its legal frame")
        return next(
            i
            for i, offer in enumerate(self.frame.offers)
            if offer["id"] == self.commitment.offer_id
        )

    def apply(self, env: managym.Env) -> managym.Observation:
        if isinstance(self.commitment, int):
            return env.step(self.commitment)[0]
        env.execute_semantic_command_json(self.commitment.to_json())
        return env.observation_for_player(env.current_agent_index() or 0)


def _build_name(identities: Mapping[str, object]) -> str:
    return (
        f"managym {identities.get('managym_version', 'unknown')} "
        f"cards {identities.get('card_digest', 'unknown')} "
        f"engine {identities.get('engine_recording_source_sha256', 'unknown')}"
    )


def interventions_by_offset(game: GameInputs) -> dict[int, list[Intervention]]:
    boundaries: dict[int, list[Intervention]] = defaultdict(list)
    for op in game.interventions:
        boundaries[op.at_action].append(op)
    return boundaries


def apply_boundary(
    env: managym.Env, operations: Sequence[Intervention], remaining_frames: set[str]
) -> None:
    """Consume issued fingerprints on both sides of each live intervention."""
    for op in operations:
        if remaining_frames:
            env._restore_frame_candidates(remaining_frames)
        env._apply_intervention_json(op.model_dump_json(exclude={"at_action"}))
    if remaining_frames:
        env._restore_frame_candidates(remaining_frames)


class RecordWalker:
    """One verified traversal, optionally applying through a consumer authority.

    Academy's apply callback executes its bound Command on this same Env. Every
    other consumer uses the native step. No pending journal call is attempted.
    """

    def __init__(
        self,
        record: GameRecord,
        *,
        capture_cards: bool = False,
    ) -> None:
        if len(record.games) != 1 or record.mode == "search":
            raise ValueError("walk requires one game, not a batch or a search")
        current = runtime_identities()
        self.builds = (
            f"recorded {_build_name(record.identities)}; running {_build_name(current)}"
        )
        if (
            record.identities.get("card_digest") is not None
            and record.identities["card_digest"] != current["card_digest"]
        ):
            raise ValueError(
                f"record cards differ: {self.builds}. Open it with managym "
                f"{record.identities.get('managym_version', 'unknown')}."
            )
        self.build_differs = any(
            record.identities.get(key) != current[key]
            for key in (
                "managym_version",
                "engine_recording_source_sha256",
                "engine_extension_sha256",
            )
            if key in record.identities
        )
        self.record = record
        self.game = record.games[0]
        if not self.game.players:
            raise ValueError("record setup unavailable")
        self.env = managym.Env(
            seed=self.game.seed,
            skip_trivial=record.skip_trivial,
            player_policies=record.resolved_player_policies,
            decision_memory=self.game.decision_memory,
            decision_memory_include_own=self.game.decision_memory_include_own,
        )
        self._capture_cards = capture_cards
        if capture_cards:
            self.env.enable_exposure_capture()
        self.observation = self.env.reset(self.game.setup())[0]
        self.stop: ReplayStop | None = None
        self._used = False

    def cards(self) -> CapturedCards:
        """Private facts strictly through the last verified commitment."""
        if not self._capture_cards:
            raise ValueError("walk requires capture_cards=True to collect card facts")
        if self.stop is None:
            raise ValueError("card facts require a finished verified traversal")
        return captured_cards(self.env)

    def __iter__(self) -> Iterator[RecordedDecision]:
        return self.run()

    def run(
        self,
        apply: Callable[[RecordedDecision], None] | None = None,
    ) -> Iterator[RecordedDecision]:
        if self._used:
            raise ValueError("a record walker can only be consumed once")
        self._used = True
        game = self.game
        count = len(game.actions) if game.committed is None else game.committed
        boundary = self.record.boundary
        if boundary is not None and boundary.phase != "ready":
            count = min(
                count, boundary.committed_counts[0] if boundary.committed_counts else 0
            )
        if (
            game.committed is None
            and boundary is None
            and game.frame is not None
            and self.record.execution_outcome != "last_command_succeeded"
        ):
            count = max(0, count - 1)
        boundaries = interventions_by_offset(game)
        remaining_frames = set(game.object_candidate_frames)

        index = 0
        try:
            # The final boundary can contain interventions after the last action.
            for index, action in enumerate(chain(game.actions[:count], (None,))):
                operations = boundaries.get(index, ())
                apply_boundary(self.env, operations, remaining_frames)
                if operations:
                    self.observation = self.env.observation_for_player(
                        self.env.current_agent_index() or 0
                    )
                if action is None:
                    break
                if (
                    index < len(game.digests)
                    and self.env.state_digest() != game.digests[index]
                ):
                    raise ValueError("pre-decision digest differs")
                observation = self.observation
                frame = None
                commitment: int | Command = (
                    action if isinstance(action, int) else Command(**json.loads(action))
                )
                if isinstance(commitment, Command):
                    live_frame = self.env.semantic_decision_frame_json()
                    retained_frame = (
                        game.command_frame_hashes[index]
                        if index < len(game.command_frame_hashes)
                        else None
                    )
                    if (
                        retained_frame is not None
                        and retained_frame != semantic_frame_sha256(live_frame)
                    ):
                        raise ValueError("recorded legal semantic frame differs")
                    frame = DecisionFrame.from_json(live_frame)
                    if commitment.expected_revision != frame.revision:
                        raise ValueError("stale recorded Command")
                    frame.offer(commitment.offer_id)
                elif action != -1 and not 0 <= action < len(
                    observation.action_space.actions
                ):
                    raise ValueError("recorded action is not offered")
                decision = RecordedDecision(
                    index,
                    observation,
                    commitment,
                    game.annotations[index] if index < len(game.annotations) else None,
                    frame,
                )
                # Check the transition on a search fork before committing it to
                # the Env owning the retained card facts or consumer presentation.
                # A divergent action must contribute to neither history nor rows.
                target = self.env.clone_env()
                decision.apply(target)
                after = (
                    game.digests[index + 1]
                    if index + 1 < count and index + 1 < len(game.digests)
                    else game.end.digest
                    if index + 1 == count and game.end is not None
                    else None
                )
                # Compare the consumer immediately after its action. The retained
                # next digest includes any interventions before the next decision.
                expected = target.state_digest() if apply is not None else None
                if after is not None:
                    # Only rules interventions affect the digest. Bind issued
                    # frames on the live timeline when it reaches that boundary.
                    if operations := boundaries.get(index + 1):
                        apply_boundary(target, operations, set())
                    if target.state_digest() != after:
                        raise ValueError("transition digest differs")
                if apply is None:
                    self.observation = decision.apply(self.env)
                else:
                    apply(decision)
                    if self.env.state_digest() != expected:
                        raise RuntimeError(
                            "consumer transition differs from verified native commitment"
                        )
                    self.observation = self.env.observation_for_player(
                        self.env.current_agent_index() or 0
                    )
                yield decision
            if count == len(game.actions) and remaining_frames:
                raise ValueError("snapshot object candidate frame missing")
            end = game.end
            if (
                end is not None
                and end.digest is not None
                and self.env.state_digest() != end.digest
            ):
                raise ValueError("final digest differs")
            if end is not None and end.status == "complete":
                if not self.env.is_game_over() or (
                    end.winner_retained and self.env.winner_index() != end.winner
                ):
                    raise ValueError("recorded result differs")
                self.stop = ReplayStop(
                    "complete", count, winner=self.env.winner_index()
                )
            else:
                self.stop = ReplayStop(
                    "interrupted", count, end.reason if end else "saved prefix only"
                )
        except (
            ValueError,
            TypeError,
            SemanticContractError,
            managym.AgentError,
        ) as error:
            self.stop = ReplayStop("diverged", index, f"{error}; {self.builds}")


def walk(record: GameRecord, *, capture_cards: bool = False) -> RecordWalker:
    return RecordWalker(record, capture_cards=capture_cards)


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


def _position(observation: managym.Observation) -> dict[str, object]:
    """Name the reached decision's turn so an operator can locate it in the game."""
    turn = observation.turn
    return {
        "turn": int(turn.turn_number),
        "phase": str(turn.phase).rsplit(".", 1)[-1],
        "step": str(turn.step).rsplit(".", 1)[-1],
        "active_player": int(turn.active_player_id),
        "game_over": bool(observation.game_over),
    }


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

    Historical arena and tolerance receipts retain other source identities;
    their replay contract only checks native bytes. Return the current native
    digest for diagnostic output.
    """
    native = hashlib.sha256(Path(managym._managym.__file__).read_bytes()).hexdigest()
    if not allow_changed_runtime:
        if native != identities.get("engine_extension_sha256"):
            raise ValueError(
                "native bytes differ; explicit --allow-changed-runtime required"
            )
        if (
            require_source
            and identities.get("engine_recording_source_sha256")
            != runtime_identities()["engine_recording_source_sha256"]
        ):
            raise ValueError(
                "engine source differs or legacy source verification requires manabot; explicit --allow-changed-runtime required"
            )
    return native


def replay_failure_inputs(
    evidence: GameRecord, *, statistics: bool = False
) -> list[dict[str, object]]:
    """Replay only retained inputs. The CLI supplies the process wall-time bound."""
    boundary = evidence.boundary
    if boundary is None:
        return [
            _replay_game_inputs(evidence, game, statistics=statistics)
            for game in evidence.games
        ]
    retained = evidence.model_copy(deep=True)
    # Journal reset inputs describe the prior game, not the proposed new root.
    # Only an active step row with its exact pre-action frame binds an attempt.
    retained.operation = "step"
    results: list[dict[str, object]] = []
    for index, game in enumerate(retained.games):
        bound_step = (
            boundary.phase == "executing"
            and boundary.operation in {"step", "step_isolated", "command"}
            and index < len(boundary.active)
            and boundary.active[index]
            and index < len(boundary.committed_counts)
            and index < len(boundary.pending_actions)
            and len(game.actions) == boundary.committed_counts[index] + 1
            and game.actions[-1] == boundary.pending_actions[index]
            and game.frame is not None
        )
        if not bound_step:
            game.frame = None
            if (
                boundary.phase == "executing"
                and boundary.operation in {"step", "step_isolated", "command"}
                and index < len(boundary.committed_counts)
            ):
                game.actions = game.actions[: boundary.committed_counts[index]]
        if not game.players:
            result: dict[str, object] = {
                "stream": game.stream,
                "status": "setup_unavailable",
            }
        else:
            result = _replay_game_inputs(retained, game, statistics=statistics)
        result.update(
            boundary_operation=boundary.operation,
            boundary_phase=boundary.phase,
            native_returned=boundary.returned,
            setup_known=boundary.setup_known,
            original_outcome=(
                "failed"
                if evidence.mode != "vector" and evidence.execution_outcome == "failed"
                else "execution_outcome_unknown"
            ),
            diagnostic="Replay cannot settle the original interrupted call's outcome",
        )
        if not bound_step:
            result.update(
                at_retained_failure=False,
                matches_binding=None,
                matches_error=None,
                diagnostic=(
                    "Reconstructed the current known prefix; no native call was pending"
                    if boundary.phase == "ready"
                    else "Reconstructed last-known prefix; pending binding or current setup unavailable"
                ),
            )
        results.append(result)
    return results


_HISTORY_EVENT_WIDTH = managym.observation_layout(policy_history=True)["events"][
    "shape"
][1]


def _search_layout(specs: Mapping[str, BufferSpec]) -> managym.ObservationLayout:
    """Resolve the native layout recorded by a search receipt's buffer shapes."""

    def rows(name: str, default: int = 0) -> int:
        return specs[name].shape[1] if name in specs else default

    version = 11 if "stack" in specs else (10 if "player_turn_state" in specs else 9)
    return managym.ObservationLayout(
        policy_input_version=version,
        policy_history=specs["events"].shape[2] == _HISTORY_EVENT_WIDTH,
        max_cards_per_player=specs["agent_cards"].shape[1],
        max_permanents_per_player=specs["agent_permanents"].shape[1],
        max_stack_objects=rows("stack", 32),
        max_actions=specs["actions"].shape[1],
        max_focus_objects=specs["action_focus"].shape[2],
        max_events=specs["events"].shape[1],
        max_facts=specs["facts"].shape[1],
        max_decision_records=rows("decision_records"),
        max_decision_references=rows("decision_references"),
        max_decision_reference_details=rows("decision_reference_details"),
        max_decision_members=rows("decision_members"),
    )


def _replay_game_inputs(
    evidence: GameRecord, game: GameInputs, *, statistics: bool = False
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
            decision_memory=game.decision_memory,
            decision_memory_include_own=game.decision_memory_include_own,
            skip_trivial=evidence.skip_trivial,
            player_policies=evidence.resolved_player_policies,
        )
        if evidence.mode == "vector"
        else managym.Env(
            player_policies=evidence.resolved_player_policies,
            seed=game.seed,
            skip_trivial=evidence.skip_trivial,
            decision_memory=game.decision_memory,
            decision_memory_include_own=game.decision_memory_include_own,
        )
    )
    if statistics:
        env.enable_statistics()
    index = -1
    phase = "reset"
    # The last decision reconstruction reached, before any failing call.
    reached: managym.Observation | None = None
    try:
        agreement.at_retained_failure = retained_phase == "reset"
        # Invalid reset setups are retained verbatim for reproduction. Normal
        # replay admission still requires a complete two-seat setup.
        setup = (
            [] if evidence.operation == "reset" and not game.players else game.setup()
        )
        if evidence.mode == "vector":
            reached = env.reset_all(setup)[0][0]
        else:
            reached = env.reset(setup)[0]
        phase = "prefix"
        boundaries = interventions_by_offset(game)
        remaining_frames = set(game.object_candidate_frames)
        for offset, action in enumerate(chain(game.actions, (None,))):
            operations = boundaries.get(offset, ())
            if operations or remaining_frames:
                index, phase = offset, "intervention"
                agreement.at_retained_failure = False
                if not isinstance(env, managym.Env):
                    raise ValueError("branch interventions require a scalar replay")
                apply_boundary(env, operations, remaining_frames)
                if operations:
                    reached = env.observation_for_player(env.current_agent_index() or 0)
            if action is None:
                if remaining_frames:
                    raise ValueError("snapshot object candidate frame missing")
                break
            index, phase = offset, "prefix"
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
                if reached is not None:
                    # Printed before the attempt, so a replay that never returns
                    # still names the decision its retained call was made at.
                    print(
                        json.dumps(
                            {
                                "replay_attempt": {
                                    "stream": game.stream,
                                    "action_index": index,
                                    "position": _position(reached),
                                }
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            agreement.at_retained_failure = (
                phase == retained_phase and index == retained_index
            )
            if evidence.mode == "vector":
                reached = env.step([action])[0][0]
            elif isinstance(action, str):
                env.execute_semantic_command_json(action)
                reached = None
            else:
                reached = env.step(action)[0]
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
                pool.set_buffers(
                    buffers, search.capacity, layout=_search_layout(search.buffers)
                )
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
    if statistics:
        summaries = env.statistics()
        summary = summaries[0] if isinstance(summaries, list) else summaries
        result["statistics"] = (
            None if summary is None else summary.model_dump(mode="json")
        )
    if reached is not None:
        result["reached"] = _position(reached)
    return {**asdict(agreement), "stream": game.stream, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--allow-changed-runtime", action="store_true")
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Count the reconstructed prefix, never the unknown original outcome",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error("seconds must be in (0, 300]")
    if not args.worker:
        run_bounded_replay("managym.replay", args.seconds)
    try:
        records = read_records(args.path)
        evidence = next(records)
    except StopIteration:
        if not args.path.is_dir():
            parser.error(f"cannot read journal {args.path}: no game records")
        parser.error(
            f"no game records in {args.path}: a directory is read for its .jsonl"
            " records; pass a snapshot replay .json file directly"
        )
    except (OSError, ValueError) as error:
        parser.error(f"cannot read journal {args.path}: {error}")
    if not evidence.needs_reproduction:
        results = []
        for record in chain((evidence,), records):
            try:
                replay = walk(record, capture_cards=args.statistics)
            except ValueError as error:
                parser.error(f"cannot replay {args.path}: {error}")
            for _ in replay:
                pass
            assert replay.stop is not None
            result = {**asdict(replay.stop), "build_differs": replay.build_differs}
            if args.statistics:
                result["statistics"] = {
                    "terminal": replay.stop.status == "complete",
                    "winner": replay.stop.winner,
                    "cards": [
                        row.model_dump(mode="json")
                        for row in reduce_cards(replay.cards())
                    ],
                }
            results.append(result)
        print(json.dumps({"status": "replayed", "results": results}))
        return
    # Reproduction needs every stream and its original batch boundary.
    evidence = read_record(args.path)
    try:
        require_replay_runtime(
            evidence.identities,
            allow_changed_runtime=args.allow_changed_runtime,
            require_source=True,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "replayed",
                "results": replay_failure_inputs(evidence, statistics=args.statistics),
                "original_outcome": evidence.execution_outcome,
            }
        )
    )


if __name__ == "__main__":
    main()
