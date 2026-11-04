"""Offline reconciliation of the prospective reference league's sealed results.

The manifest supplies the denominator, including absent training runs. Etude
owns training declarations/metrics and evaluation blobs; EvaluationResult owns
registration/replay validity. Four correlated legs form one bootstrap block,
paired across checkpoints within each panel and training seed. No games run.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING

from etude.eval.query import read_blobs, statsdb
from etude.eval.records import Environment, Match, Model
from etude.experiment import ExperimentState, observe
from etude.train.read import RunData, read_all
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from experiments.growing_pool import (
    comparisons,
    deck_matchup,
    policy_source,
    policy_source_sha256,
)
from experiments.growing_pool_protocol import (
    EXPERIMENT,
    PROTOCOL,
    Anchor,
    Panel,
    Protocol,
)
from manabot.eval.evaluation import EvaluationResult
from manabot.eval.results import EvaluationRow, RateInterval, bootstrap_mean_interval
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.provenance import source_bundle_sha256
from manabot.sim.card_evidence import GameEvidence
from manabot.sim.replay import ReplayReceipt

if TYPE_CHECKING:
    import duckdb


class Counts(BaseModel):
    """Expected = valid + failed + missing; attempted excludes lost callback work."""

    model_config = ConfigDict(extra="forbid")

    expected: int = 0
    attempted: int = 0
    valid: int = 0
    failed: int = 0
    missing: int = 0
    expected_matches: int = 0
    attempted_matches: int = 0
    failed_matches: int = 0
    missing_matches: int = 0


class DealResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deal: int
    match: str | None
    counts: Counts
    scores: list[float | None]
    score: float | None = None
    issues: list[str]


class Contrast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    panel: str
    update: int
    anchor: Anchor
    counts: Counts
    deals: list[DealResult]
    score: RateInterval | None = None
    gain: RateInterval | None = None


class Health(StrictModel):
    complete: bool
    passed: bool | None
    # Digest of the declaration and of the selected metric values/coordinates.
    declaration_sha256: str | None = None
    metrics_sha256: str | None = None
    failures: list[str] = Field(default_factory=list)
    issues: list[str]


class SeedAnalysis(StrictModel):
    seed: int
    run: str | None
    state: str
    exports: dict[int, str | None]
    health: Health
    counts: Counts
    contrasts: list[Contrast]
    complete: bool
    modest_check: bool | None
    issues: list[str]


class AnalysisResult(StrictModel):
    protocol_sha256: str
    bootstrap_seed: int
    bootstrap_resamples: int
    counts: Counts
    seeds: list[SeedAnalysis]
    complete: bool
    modest_check: bool | None
    # A favorable modest check never resolves this separate authored-review boundary.
    training_exposure: str = "unavailable; acceptance unresolved"


class _TrainingConfig(BaseModel):
    regime: dict[str, JsonValue]
    identities: dict[str, JsonValue]


class _Config(BaseModel):
    training: _TrainingConfig


def _total(counts: Sequence[Counts]) -> Counts:
    return Counts(
        **{
            name: sum(getattr(count, name) for count in counts)
            for name in Counts.model_fields
        }
    )


def _health(data: RunData | None, protocol: Protocol, training_source: str) -> Health:
    if data is None or data.file is None:
        return Health(
            complete=False, passed=None, issues=["Training declaration missing."]
        )
    declaration = data.file.declaration
    config = _Config.model_validate_json(declaration.config_json or "{}")
    if (
        sha256((declaration.config_json or "").encode()).hexdigest()
        != declaration.config_sha256
    ):
        raise ValueError("training configuration digest differs")
    if (
        canonical_sha256(config.training.regime)
        != canonical_sha256(protocol.regime().model_dump(mode="json"))
        or config.training.identities.get("world") != protocol.world.world_version
        or config.training.identities.get("engine_source_sha256")
        != protocol.world.engine_source_sha256
        or config.training.identities.get("content_digest")
        != protocol.world.card_digest
        or config.training.identities.get("deck_source_sha256")
        != protocol.manifest.source().digest
        or declaration.continues is not None
        or config.training.identities.get("training_source_sha256") != training_source
    ):
        raise ValueError("training regime/source or fresh-run identity differs")
    if len(data.attempts) != 1:
        raise ValueError("reference league requires one training attempt")
    spec = protocol.analysis
    evidence: dict[str, list[list[float]]] = {}
    issues: list[str] = []
    failures: list[str] = []
    for metric in (
        *spec.finite_metrics,
        *spec.zero_metrics,
        *spec.positive_metrics,
        *spec.sampled_finite_metrics,
        spec.parameter_change,
    ):
        expected = (
            np.asarray(spec.sampled_updates)
            if metric in (*spec.sampled_finite_metrics, spec.parameter_change)
            else np.arange(1, protocol.budget.updates + 1)
        )
        points = data.points(metric, spec.health_clock)
        # NaN/Inf remain failure evidence, represented explicitly for JSON hashing.
        evidence[metric] = np.column_stack((points.x, points.y)).tolist()
        if not np.array_equal(points.x, expected):
            issues.append(f"Missing or unexpected health coordinates: {metric}")
        if not np.isfinite(points.y).all():
            failures.append(f"Nonfinite health measurement: {metric}")
        if metric in spec.zero_metrics and np.any(points.y != 0):
            failures.append(f"Nonzero health incident count: {metric}")
        if metric in spec.positive_metrics and np.any(points.y <= 0):
            failures.append(f"Nonpositive optimizer count: {metric}")
        if metric == spec.parameter_change and (
            not np.any(points.y > 0) or np.any(points.y < 0)
        ):
            failures.append(
                "No positive sampled parameter change, or negative measurement."
            )
    # Hash the exact retained float representation, including nonfinite evidence.
    metrics_digest = sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return Health(
        complete=not issues,
        passed=not failures if not issues else None,
        failures=failures,
        declaration_sha256=canonical_sha256(declaration.model_dump(mode="json")),
        metrics_sha256=metrics_digest,
        issues=issues,
    )


def _blob(
    db: duckdb.DuckDBPyConnection, source: str | Path, digest: str, run: str, match: str
) -> bytes:
    if not db.execute(
        "SELECT 1 FROM blobs WHERE run = ? AND match = ? AND sha256 = ?",
        [run, match, digest],
    ).fetchone():
        raise ValueError("evidence blob does not belong to this sealed match")
    return dict(read_blobs(db, source, [digest]))[digest]


def _validate_result(
    result: EvaluationResult,
    match: Match,
    protocol: Protocol,
    panel: Panel,
    heuristic_source: str,
) -> None:
    """Bind the retained evaluator request to the scheduled world, players and budgets."""
    request = result.request
    matrix = deck_matchup(protocol, panel)
    if (
        request.deal_seeds != (match.conditions["deal"],)
        or request.authored_matchup != matrix
        or request.matchup_mode != "cross"
        or request.key.world != protocol.world.world_version
        or request.key.arena_version != protocol.id
        or request.key.content_suite != matrix.content_suite
        or request.key.evaluation_compute_envelope_id
        != "policy-cpu-one-thread-one-pass"
        or request.game_seconds != protocol.budget.game_seconds
        or request.max_commands != protocol.budget.max_commands
        or request.bootstrap_seed != protocol.analysis.bootstrap_seed
        or request.bootstrap_replicates != protocol.analysis.bootstrap_resamples
        or result.execution is not None
    ):
        raise ValueError("sealed evaluation request differs from frozen schedule")
    for player, registration, name in zip(
        match.players,
        (request.candidate, request.opponent),
        ("candidate", "opponent"),
        strict=True,
    ):
        if (
            registration.player_id != name
            or registration.world != protocol.world.world_version
            or registration.content_suite != matrix.content_suite
            or registration.matchup_sha256
            != canonical_sha256(matrix.match().model_dump(mode="json"))
            or registration.compute_class_id != "policy-cpu-one-thread-one-pass"
            or registration.information_boundary != "acting-viewer"
        ):
            raise ValueError("sealed registration differs from protocol")
        if isinstance(player, Model):
            if (
                registration.checkpoint_sha256 != player.checkpoint.artifact.sha256
                or registration.checkpoint_bytes
                != player.checkpoint.artifact.size_bytes
                or registration.player_spec
                != {
                    "kind": "checkpoint",
                    "device": "cpu",
                    "batch_size": 1,
                    "deterministic": False,
                    "allow_deck_repetition": True,
                }
            ):
                raise ValueError("sealed checkpoint differs from export")
        elif registration.player_spec != {"kind": "heuristic"} or (
            registration.source_sha256 != heuristic_source
        ):
            raise ValueError("sealed heuristic source differs")
    if request.key.anchor_cohort_sha256 != request.opponent.identity_sha256:
        raise ValueError("sealed anchor identity differs")
    result.complete_deals()  # Shared duplicate, identity, outcome and replay guards.


def _validate_replay(
    game: GameEvidence,
    result: EvaluationResult,
    row: EvaluationRow,
    protocol: Protocol,
    training_source: str,
) -> None:
    leg = row.leg
    matrix = result.request.authored_matchup
    assert matrix is not None
    players = ("candidate", "opponent") if leg % 2 == 0 else ("opponent", "candidate")
    setup = matrix.match(leg)
    setup.hero, setup.villain = players
    receipt = TypeAdapter(ReplayReceipt).validate_python(game.replay)
    trace = game.trace.data
    runtime = trace.get("runtime_identities")
    score = 0.5 if game.winner is None else float(game.winner == leg % 2)
    if (
        game.deal != row.deal_seed
        or game.leg != leg
        or game.seat_players != players
        or game.seat_decks != matrix.decks(leg)
        or game.setup != setup
        or trace.get("observation_hypers")
        != protocol.observation.model_dump(mode="json")
        or game.trace_sha256 != (row.model_extra or {}).get("trace_sha256")
        or score != row.score_a
        or not game.replay.get("passed")
        or not receipt.passed
        or receipt.games != 1
        or receipt.decisions != len(trace.get("decisions", []))
        or game.outcome not in ("win_loss", "draw")
        or trace.get("failure")
        or not trace.get("terminated")
        or trace.get("truncated")
        or not isinstance(runtime, dict)
        or runtime.get("arena_source_sha256") != training_source
        or runtime.get("world") != protocol.world.world_version
        or runtime.get("content_digest") != protocol.world.card_digest
        or runtime.get("engine_source_sha256") != protocol.world.engine_source_sha256
        or any(
            runtime.get(field) != getattr(registration, field)
            for registration in (result.request.candidate, result.request.opponent)
            for field in ("observation_abi_sha256", "action_abi_sha256")
        )
    ):
        raise ValueError("sealed replay differs from evaluation outcome/world/decks")


def _deal(
    db: duckdb.DuckDBPyConnection,
    source: str | Path,
    protocol: Protocol,
    panel: Panel,
    deal: int,
    match: Match | None,
    tables: set[str],
    heuristic_source: str,
    training_source: str,
) -> DealResult:
    counts = Counts(expected=4, expected_matches=1, missing=4, missing_matches=1)
    report = DealResult(
        deal=deal,
        match=match.key if match else None,
        counts=counts,
        scores=[None] * 4,
        issues=[],
    )
    if match is None:
        report.issues.append("Export unavailable.")
        return report
    environment = Environment(
        name=protocol.id, files=protocol.digest, config={"status": protocol.status}
    )
    receipts = db.execute(
        "SELECT m.run, m.state, m.error, m.conditions, m.games FROM matches m "
        "JOIN runs r USING (run) WHERE m.match = ? AND r.environment = ?",
        [match.key, environment.identity],
    ).fetchall()
    if len(receipts) > 1:
        raise ValueError("repeated evaluation match; no retries permitted")
    if not receipts:
        report.issues.append("Callback missing.")
        return report
    run, state, error, conditions, games = receipts[0]
    if json.loads(conditions) != match.conditions or games != 4:
        raise ValueError("sealed match conditions differ")
    counts.missing_matches = 0
    counts.attempted_matches = 1
    if state != "done":
        counts.failed_matches = 1
        report.issues.append(f"Callback failed: {error}")
        return report
    seats = db.execute(
        "SELECT s.seat, s.player, s.source, p.artifact, s.at, s.line "
        "FROM seats s JOIN players p USING (player) WHERE s.run = ? AND s.match = ? ORDER BY s.seat",
        [run, match.key],
    ).fetchall()
    if len(seats) != 2:
        raise ValueError("sealed match must retain both participants")
    for seat, player in enumerate(match.players):
        actual_seat, key, version, artifact, at, line = seats[seat]
        if (actual_seat, key, version) != (seat, player.key, player.version.source):
            raise ValueError("evaluation player/source differs from scheduled identity")
        if isinstance(player, Model) and (
            artifact != player.checkpoint.artifact.sha256
            or json.loads(at or "null") != player.checkpoint.at
            or line != player.checkpoint.line
        ):
            raise ValueError("evaluation checkpoint identity differs")
    refs = (
        db.execute(
            "SELECT sha256 FROM evaluation_results WHERE run = ? AND match = ?",
            [run, match.key],
        ).fetchall()
        if "evaluation_results" in tables
        else []
    )
    if not refs:
        report.issues.append("Sealed evaluator receipt missing.")
        return report
    if len(refs) != 1:
        raise ValueError("duplicate evaluator receipt")
    result = EvaluationResult.model_validate_json(
        _blob(db, source, refs[0][0], run, match.key)
    )
    _validate_result(result, match, protocol, panel, heuristic_source)
    counts.attempted = len(result.rows)
    counts.missing = 4 - counts.attempted
    for row in result.rows:
        if not row.valid:
            counts.failed += 1
            report.issues.append(
                f"Leg {row.leg}: {row.failure or 'invalid outcome/replay'}"
            )
            continue
        refs = (
            db.execute(
                "SELECT DISTINCT evidence_sha256 FROM card_plays WHERE run = ? AND match = ? AND trace_sha256 = ?",
                [run, match.key, (row.model_extra or {}).get("trace_sha256")],
            ).fetchall()
            if "card_plays" in tables
            else []
        )
        if not refs:
            counts.failed += 1
            report.issues.append(f"Leg {row.leg}: replay evidence missing.")
            continue
        if len(refs) != 1:
            raise ValueError("conflicting replay blobs")
        game = GameEvidence.model_validate_json(
            gzip.decompress(_blob(db, source, refs[0][0], run, match.key))
        )
        _validate_replay(game, result, row, protocol, training_source)
        counts.valid += 1
        report.scores[row.leg] = row.score_a
    if result.error:
        report.issues.append(result.error)
    if counts.valid == 4 and not report.issues:
        report.score = sum(score for score in report.scores if score is not None) / 4
    return report


def _estimate(contrasts: list[Contrast], protocol: Protocol) -> None:
    for panel in protocol.manifest.panels:
        panel_contrasts = [c for c in contrasts if c.panel == panel.id]
        indices = np.random.default_rng(protocol.analysis.bootstrap_seed).integers(
            0,
            len(panel.deals),
            (protocol.analysis.bootstrap_resamples, len(panel.deals)),
        )
        blocks = {
            (c.update, c.anchor): np.asarray(
                [d.score for d in c.deals], dtype=np.float64
            )
            for c in panel_contrasts
            if all(d.score is not None for d in c.deals)
        }
        initial = blocks.get((0, "heuristic"))
        for contrast in panel_contrasts:
            scores = blocks.get((contrast.update, contrast.anchor))
            if scores is not None:
                contrast.score = bootstrap_mean_interval(scores, indices)
                if initial is not None and contrast.anchor == "heuristic":
                    contrast.gain = bootstrap_mean_interval(scores - initial, indices)


def analyze(
    state: ExperimentState,
    training: Sequence[RunData],
    db: duckdb.DuckDBPyConnection,
    source: str | Path,
    protocol: Protocol,
) -> AnalysisResult:
    """Reconcile declared seeds first; missing evidence never shrinks the exam.

    Identity contradictions and duplicate records are rejected. Absent runs,
    exports, callbacks, games or health metrics produce an incomplete report.
    Secondary panels remain descriptive and cannot secretly become a score gate.
    """
    if state.experiment != EXPERIMENT:
        raise ValueError("analysis requires the growing-pool experiment")
    by_seed: dict[int, RunData] = {}
    for data in training:
        if (
            data.experiment != EXPERIMENT
            or data.file is None
            or data.file.declaration.kind == "evaluation"
        ):
            continue
        seed = data.file.declaration.seed
        if seed is None or seed not in protocol.budget.seeds or seed in by_seed:
            raise ValueError("unexpected or duplicate training seed")
        by_seed[seed] = data
    if {r.name for r in state.runs} != {d.name for d in by_seed.values()}:
        raise ValueError("training exports and declarations disagree")
    tables = {row[0] for row in db.execute("SHOW TABLES").fetchall()}
    training_source = source_bundle_sha256(policy_source())
    heuristic_source = policy_source_sha256()
    primary_panels = {p.id for p in protocol.manifest.panels if p.kind == "anchor"}
    reports = []
    for seed in protocol.budget.seeds:
        data = by_seed.get(seed)
        run = state.run(data.name) if data is not None else None
        run_state = run.state if run is not None else "missing"
        if run is not None and any(
            c.run != run.name or c.line != run.name or c.experiment != EXPERIMENT
            for c in run.checkpoints
        ):
            raise ValueError("checkpoint is not from this fresh learner")
        health = _health(data, protocol, training_source)
        contrasts = []
        exports: dict[int, str | None] = {
            update: None for update in protocol.budget.export_updates
        }
        for slot in comparisons(run, protocol):
            panel = slot.panel
            matches = (
                slot.matchup.matches() if slot.matchup else [None] * len(panel.deals)
            )
            if slot.matchup:
                candidate = slot.matchup.players[0]
                assert isinstance(candidate, Model)
                exports[slot.update] = str(candidate.checkpoint.key)
            deals = [
                _deal(
                    db,
                    source,
                    protocol,
                    panel,
                    deal,
                    match,
                    tables,
                    heuristic_source,
                    training_source,
                )
                for deal, match in zip(panel.deals, matches, strict=True)
            ]
            contrasts.append(
                Contrast(
                    panel=panel.id,
                    update=slot.update,
                    anchor=slot.anchor,
                    counts=_total([d.counts for d in deals]),
                    deals=deals,
                )
            )
        _estimate(contrasts, protocol)
        issues = list(health.issues)
        if data is None:
            issues.append("Declared training seed is wholly absent.")
        if run_state != "finished":
            issues.append("Training run has not finished successfully.")
        if any(c.score is None for c in contrasts):
            issues.append(
                "Expected evaluation schedule incomplete; inspect deal issues."
            )
        primary = [c for c in contrasts if c.panel in primary_panels]
        endpoints = [
            c
            for c in primary
            if c.update == protocol.budget.updates and c.anchor == "heuristic"
        ]
        modest = None
        primary_ready = all(c.score is not None for c in primary)
        if health.complete and run_state == "finished" and primary_ready:
            modest = (
                health.passed is True
                and bool(endpoints)
                and all(c.gain is not None and c.gain.mean > 0 for c in endpoints)
            )
        reports.append(
            SeedAnalysis(
                seed=seed,
                run=data.name if data else None,
                state=run_state,
                exports=exports,
                health=health,
                counts=_total([c.counts for c in contrasts]),
                contrasts=contrasts,
                complete=not issues,
                modest_check=modest,
                issues=issues,
            )
        )
    return AnalysisResult(
        protocol_sha256=protocol.digest,
        bootstrap_seed=protocol.analysis.bootstrap_seed,
        bootstrap_resamples=protocol.analysis.bootstrap_resamples,
        counts=_total([r.counts for r in reports]),
        seeds=reports,
        complete=all(r.complete for r in reports),
        modest_check=all(r.modest_check for r in reports)
        if all(r.modest_check is not None for r in reports)
        else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = Protocol.model_validate_json(args.protocol.read_bytes())
    with statsdb(EXPERIMENT, args.source) as db:
        result = analyze(
            observe(args.source, EXPERIMENT),
            read_all(args.source),
            db,
            args.source,
            protocol,
        )
    atomic_json(args.out, result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
