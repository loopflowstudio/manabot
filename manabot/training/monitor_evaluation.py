"""Checkpoint monitoring on reserved deals, independent of scientific evaluation.

`evaluate_checkpoint` freezes arena registrations and saves each four-game deal
block. `import_saved_rows` projects existing arena evidence without playing games.
Training coordinates belong to the checkpoint; evaluation costs never advance them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Literal

import numpy as np
import psutil
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from manabot.arena import players
from manabot.arena.heuristic import heuristic_source_paths
from manabot.arena.match import (
    SELECTED_SUITE,
    play_cell,
    selected_decks,
    selected_match,
)
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.teacher1_evidence import runtime_fingerprints, source_bundle_sha256
from manabot.training.execution import atomic_json
from manabot.training.models import (
    ArtifactReference,
    Strict,
    TrainingCoordinates,
    TrainingRun,
)

EvaluationPurpose = Literal[
    "monitoring-not-scientific-evaluation",
    "frozen-study-evaluation",
    "supplemental-matchup-diagnostic",
]


class MonitorProtocol(Strict):
    authored_matchup: AuthoredMatchup | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    purpose: EvaluationPurpose = Field(
        default="monitoring-not-scientific-evaluation",
        exclude_if=lambda v: v == "monitoring-not-scientific-evaluation",
    )
    study_protocol_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$", exclude_if=lambda v: v is None
    )
    # Reserved monitoring namespace; callers must also exclude these from any
    # scientific cohort they construct. These deals are repeatedly inspected.
    deal_seeds: tuple[int, ...] = tuple(range(1_910_101_000, 1_910_101_025))
    include_mirrors: bool = Field(default=False, exclude_if=lambda value: not value)

    @property
    def legs(self) -> int:
        return 8 if self.include_mirrors else 4

    bootstrap_seed: int = 101
    bootstrap_replicates: int = Field(default=2000, ge=1)
    game_seconds: float = Field(default=120, gt=0)
    max_commands: int = Field(default=10_000, ge=1)
    opponent: Literal["scripted_greedy", "heuristic", "random"] = Field(
        default="scripted_greedy", exclude_if=lambda value: value == "scripted_greedy"
    )
    # A frozen external comparison owns seeds, selection and analysis. This binds
    # its exact bytes; the evaluator itself supplies no promotion decision.
    comparison_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )

    @property
    def result_purpose(
        self,
    ) -> EvaluationPurpose | Literal["predeclared-comparison-not-admission"]:
        """Preserve the external-comparison label without changing frozen bytes."""
        return (
            "predeclared-comparison-not-admission"
            if self.comparison_sha256
            else self.purpose
        )

    @model_validator(mode="after")
    def unique_deals(self) -> MonitorProtocol:
        if (self.purpose == "frozen-study-evaluation") != (
            self.study_protocol_sha256 is not None
        ):
            raise ValueError(
                "held-out evaluation requires a frozen study protocol binding"
            )
        if not self.deal_seeds or len(set(self.deal_seeds)) != len(self.deal_seeds):
            raise ValueError("monitoring deals must be nonempty and unique")
        return self


class Checkpoint(Strict):
    artifact: ArtifactReference
    coordinates: TrainingCoordinates


def stage_checkpoint(run: TrainingRun, stage_id: str) -> Checkpoint | None:
    """Return an admitted completed or paused export with original coordinates."""
    for index, stage in enumerate(run.stages):
        if stage.id != stage_id:
            continue
        if (
            stage.status not in {"completed", "paused"}
            or "raw" not in stage.artifacts
            or stage.cumulative_seconds is None
        ):
            return None
        prefix = run.stages[: index + 1]
        origins = [
            s.learning_state_origin
            for s in prefix
            if s.learning_state_origin is not None
        ]
        return Checkpoint(
            artifact=stage.artifacts["raw"],
            coordinates=TrainingCoordinates(
                stage_id=stage.id,
                updates=run.updates_through(stage.id),
                segment_updates=len(stage.diagnostics)
                if run.regime.stages[index].operation == "train_self_play"
                and stage.optimizer_updates is not None
                else None,
                optimizer_updates=run.optimizer_updates_through(stage.id),
                training_seconds=stage.cumulative_seconds
                + sum(origin.cumulative_seconds for origin in origins),
                active_training_seconds=(
                    float(
                        stage.diagnostics[-1]["coordinates"]["active_training_seconds"]
                    )
                    if stage.diagnostics
                    and "active_training_seconds"
                    in stage.diagnostics[-1].get("coordinates", {})
                    else None
                ),
                **{
                    name: sum(getattr(s, name) for s in prefix)
                    + sum(getattr(origin, name) for origin in origins)
                    for name in (
                        "environment_decisions",
                        "learner_transitions",
                        "optimizer_exposures",
                        "games",
                    )
                },
            ),
        )
    return None


class ArenaRow(BaseModel):
    """Narrow the arena JSON boundary while retaining every additional field."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    arena_key: ArenaKey
    deal_seed: int
    leg: int = Field(ge=0, le=7)
    player_a: str
    player_b: str
    player_a_registration_sha256: str
    player_b_registration_sha256: str
    score_a: float | None = Field(strict=True)
    failure: str | None
    terminated: bool
    truncated: bool
    replay_passed: bool
    trace_path: str
    game_seconds: float = Field(ge=0)
    integrity: dict[str, int]

    @field_validator("score_a")
    @classmethod
    def valid_score(cls, value: float | None) -> float | None:
        if value not in (None, 0.0, 0.5, 1.0):
            raise ValueError("arena score must be loss, draw, win or unavailable")
        return value

    @property
    def valid(self) -> bool:
        return (
            self.failure is None
            and self.terminated
            and not self.truncated
            and self.replay_passed
            and self.score_a is not None
            and not any(self.integrity.values())
        )


class RateInterval(Strict):
    mean: float
    lower: float
    upper: float


class MonitorResult(Strict):
    schema_version: Literal[1] = 1
    purpose: EvaluationPurpose | Literal["predeclared-comparison-not-admission"] = (
        "monitoring-not-scientific-evaluation"
    )
    run_id: str
    regime_digest: str
    training_seed: int
    artifact: ArtifactReference
    coordinates: TrainingCoordinates
    protocol: MonitorProtocol
    key: ArenaKey
    candidate: PlayerRegistration
    opponent: PlayerRegistration
    status: Literal["running", "completed", "incomplete"] = "running"
    imported: bool = False
    source_rows_path: str | None = None
    source_rows_sha256: str | None = None
    rows: list[ArenaRow] = []
    expected_games: int
    win: RateInterval | None = None
    draw: RateInterval | None = None
    score: RateInterval | None = None
    evaluation_seconds: float | None = None
    started_unix: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    finished_unix: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinator_cpu_seconds: float | None = None
    reaped_worker_cpu_seconds: float | None = None
    coordinator_rss_before_bytes: int | None = None
    coordinator_rss_after_bytes: int | None = None
    host_load_before: tuple[float, float, float] | None = None
    host_load_after: tuple[float, float, float] | None = None
    concurrent_activity: str = "unknown; host load is not contention attribution"
    resource_limit: str = (
        "RSS is coordinator-only, not peak; reaped worker CPU includes any other "
        "children reaped by this process during evaluation"
    )
    error: str | None = None
    evaluation_identities: dict[str, JsonValue] = {}

    def validate_cohort(self) -> None:
        """Recompute completeness and paired-deal intervals from retained rows."""
        _summarize(self)

    @model_validator(mode="after")
    def purpose_matches_protocol(self) -> "MonitorResult":
        if self.purpose != self.protocol.result_purpose:
            raise ValueError(
                "evaluation result purpose differs from its frozen protocol"
            )
        return self


def _summarize(result: MonitorResult) -> None:
    expected = {
        (seed, leg)
        for seed in result.protocol.deal_seeds
        for leg in range(result.protocol.legs)
    }
    seen: set[tuple[int, int]] = set()
    for row in result.rows:
        coordinate = (row.deal_seed, row.leg)
        if coordinate not in expected or coordinate in seen:
            raise ValueError("unexpected or duplicate monitoring deal/leg")
        seen.add(coordinate)
        if result.protocol.include_mirrors:
            expected_decks = list(
                result.protocol.authored_matchup.decks(row.leg)
                if result.protocol.authored_matchup
                else selected_decks(row.leg)
            )
            extra = row.model_extra or {}
            if (
                extra.get("seat_decks") != expected_decks
                or extra.get("player_a_seat") != row.leg % 2
            ):
                raise ValueError(
                    "mirror row differs from its actual authored deck/seat leg"
                )
        if (
            row.arena_key != result.key
            or row.player_a != result.candidate.player_id
            or row.player_b != result.opponent.player_id
            or row.player_a_registration_sha256 != result.candidate.identity_sha256
            or row.player_b_registration_sha256 != result.opponent.identity_sha256
        ):
            raise ValueError("arena row differs from frozen monitoring registrations")
    result.win = result.draw = result.score = None
    if seen != expected or result.error or not all(row.valid for row in result.rows):
        result.status = "incomplete"
        return
    result.status = "completed"
    # Resample whole deals, keeping their four correlated seat/deck legs together.
    blocks = np.array(
        [
            [row.score_a for row in result.rows if row.deal_seed == seed]
            for seed in result.protocol.deal_seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(result.protocol.bootstrap_seed)
    indices = rng.integers(
        0, len(blocks), (result.protocol.bootstrap_replicates, len(blocks))
    )
    for name, values in (
        ("win", blocks == 1),
        ("draw", blocks == 0.5),
        ("score", blocks),
    ):
        means = values.mean(axis=1)
        draws = means[indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        setattr(
            result,
            name,
            RateInterval(
                mean=float(means.mean()), lower=float(lower), upper=float(upper)
            ),
        )


OPPONENT_NAMES = {
    "scripted_greedy": "Frozen scripted greedy",
    "heuristic": "Rule-of-thumb heuristic",
    "random": "Uniform legal random",
}


def _opponent_source_sha256(opponent: str) -> str:
    """v1 and random keep their frozen `players.py` identity."""
    if opponent != "heuristic":
        return file_sha256(Path(players.__file__))
    return source_bundle_sha256([Path(path) for path in heuristic_source_paths()])


def checkpoint_manifest(
    run: TrainingRun,
    artifact: ArtifactReference,
    coordinates: TrainingCoordinates,
    protocol: MonitorProtocol,
) -> MonitorResult:
    """Admit exact checkpoint bytes and runtime identities before playing games."""
    path = Path(artifact["path"]).resolve()
    if (
        path.stat().st_size != artifact["bytes"]
        or file_sha256(path) != artifact["sha256"]
    ):
        raise ValueError("checkpoint differs from recorded artifact")
    matchup = protocol.authored_matchup
    expected_match = matchup.match() if matchup else selected_match()
    suite = matchup.content_suite if matchup else SELECTED_SUITE
    if run.identities["matchup_sha256"] != canonical_sha256(
        expected_match.model_dump()
    ):
        raise ValueError("monitoring requires its exact declared authored setup")
    agent, space = load_checkpoint_agent(str(path))
    runtime = runtime_fingerprints(
        run.seed, match_hypers=run.regime.match, observation_space=space
    )
    for name in (
        "world",
        "observation_abi_sha256",
        "action_abi_sha256",
        "matchup_sha256",
        "engine_source_sha256",
        "content_manifest_sha256",
    ):
        if (
            name == "engine_source_sha256"
            and protocol.purpose == "supplemental-matchup-diagnostic"
        ):
            # Artifact loading already admitted the exact rules/content/input binding.
            # Preserve the historical producer source separately from this instrument.
            continue
        if runtime[name] != run.identities.get(name):
            raise ValueError(f"monitoring runtime differs from training: {name}")
    runtime["monitor_source_sha256"] = source_bundle_sha256(
        [
            Path(__file__),
            *sorted(Path(players.__file__).parent.glob("*.py")),
        ]
    )
    common = dict(
        information_boundary="acting-viewer",
        world=run.regime.world,
        content_suite=suite,
        observation_abi_sha256=run.identities["observation_abi_sha256"],
        action_abi_sha256=run.identities["action_abi_sha256"],
        matchup_sha256=run.identities["matchup_sha256"],
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    candidate = PlayerRegistration(
        **common,
        player_id="monitor-checkpoint",
        display_name="Monitored checkpoint",
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
            **({"allow_deck_repetition": True} if protocol.include_mirrors else {}),
        },
        compute_class_id="policy-cpu-one-thread-one-pass",
        checkpoint_sha256=artifact["sha256"],
        checkpoint_bytes=artifact["bytes"],
        parameter_count=sum(parameter.numel() for parameter in agent.parameters()),
        training_seed=run.seed,
        artifact_id=f"monitor/{artifact['sha256']}",
    )
    opponent = PlayerRegistration(
        **common,
        player_id="monitor-" + protocol.opponent.replace("_", "-"),
        display_name=OPPONENT_NAMES[protocol.opponent],
        role="anchor",
        runner_kind="code",
        player_spec={"kind": protocol.opponent},
        compute_class_id=protocol.opponent.replace("_", "-") + "-cpu-v1",
        source_sha256=_opponent_source_sha256(protocol.opponent),
    )
    key = ArenaKey(
        world=run.regime.world,
        content_suite=suite,
        viewer_boundary="acting-viewer",
        arena_version="training-monitor-v1",
        rating_model_version="unrated-monitoring",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    return MonitorResult(
        purpose=protocol.result_purpose,
        run_id=run.id,
        regime_digest=run.regime_digest,
        training_seed=run.seed,
        artifact={**artifact, "path": str(path)},
        coordinates=coordinates,
        protocol=protocol,
        key=key,
        candidate=candidate,
        opponent=opponent,
        expected_games=protocol.legs * len(protocol.deal_seeds),
        evaluation_identities=runtime,
    )


def evaluate_checkpoint(
    run: TrainingRun,
    artifact: ArtifactReference,
    coordinates: TrainingCoordinates,
    output_dir: Path,
    *,
    protocol: MonitorProtocol | None = None,
    concurrent_activity: str = "unknown; host load is not contention attribution",
) -> MonitorResult:
    """Run bounded arena games. Use a new directory per attempt; never overwrite failures."""
    output_dir.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    started_unix = time.time()
    process = psutil.Process()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        result = checkpoint_manifest(run, artifact, coordinates, protocol or MonitorProtocol())
    except Exception as exc:
        atomic_json(
            output_dir / "admission-failure.json",
            {
                "run_id": run.id,
                "artifact": artifact,
                "coordinates": coordinates.model_dump(mode="json"),
                "error": f"{type(exc).__name__}: {exc}",
                "evaluation_seconds": time.perf_counter() - start,
            },
        )
        raise
    result.concurrent_activity = concurrent_activity
    result.started_unix = started_unix
    result.coordinator_rss_before_bytes = process.memory_info().rss
    result.host_load_before = os.getloadavg()
    atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    try:
        for block, seed in enumerate(result.protocol.deal_seeds):
            directory = output_dir / f"deal-{block:03d}"
            directory.mkdir()
            # The arena persists command traces after every attempted game.
            rows, trace, replay = play_cell(
                key=result.key,
                player_a=result.candidate,
                player_b=result.opponent,
                deal_seeds=(seed,),
                out_dir=directory,
                checkpoint_paths={result.candidate.player_id: result.artifact["path"]},
                game_seconds=result.protocol.game_seconds,
                max_commands=result.protocol.max_commands,
                matchup_mode="all" if result.protocol.include_mirrors else "cross",
                authored_matchup=result.protocol.authored_matchup,
            )
            atomic_json(directory / "rows.json", rows)
            atomic_json(directory / "trace.json", trace)
            atomic_json(directory / "replay.json", replay)
            for raw in rows:
                row = ArenaRow.model_validate(raw)
                row.trace_path = str(Path(directory.name) / row.trace_path)
                result.rows.append(row)
            atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.evaluation_seconds = time.perf_counter() - start
        result.finished_unix = time.time()
        result.coordinator_cpu_seconds = time.process_time() - cpu
        child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.reaped_worker_cpu_seconds = (
            child_end.ru_utime
            + child_end.ru_stime
            - child_start.ru_utime
            - child_start.ru_stime
        )
        result.coordinator_rss_after_bytes = process.memory_info().rss
        result.host_load_after = os.getloadavg()
        try:
            _summarize(result)
        except ValueError as error:
            result.error = f"{type(error).__name__}: {error}"
            result.status = "incomplete"
            result.win = result.draw = result.score = None
        atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    return result


def import_saved_rows(
    manifest: MonitorResult,
    rows_path: Path,
    output_dir: Path,
) -> MonitorResult:
    """Import original arena rows against their saved manifest, without model loading.

    Paths remain relative to the source rows directory. Replay flags are historical
    claims, not a fresh replay. Original costs/coordinates remain unchanged.
    """
    payload: Any = json.loads(rows_path.read_text())  # Untrusted JSON boundary.
    if not isinstance(payload, list):
        raise ValueError("saved arena rows must be a list")
    result = manifest.model_copy(deep=True)
    result.rows = [ArenaRow.model_validate(row) for row in payload]
    result.imported = True
    result.source_rows_path = str(rows_path.resolve())
    result.source_rows_sha256 = file_sha256(rows_path)
    _summarize(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    return result
