"""Etude entry point for the prospective MTG-245 growing-pool pilot.

Declares bounded fresh training and paired, replay-checked evaluation. Neither
`runs` nor discovery allocates compute. The frozen manifest owns every deck,
split and expected comparison, including slots whose exports never arrived.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
from typing import Literal

from etude.eval.match import Result, Row
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Matchup,
    Model,
)
from etude.experiment import ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Report, Score
from pydantic import BaseModel, JsonValue, TypeAdapter
import torch

from experiments import growing_pool
from experiments.growing_pool_exposure import decision_rows
from experiments.growing_pool_protocol import (
    EXPERIMENT,
    PROTOCOL,
    ROOT,
    Anchor,
    EvaluationRound,
    Panel,
    Protocol,
    read_protocol,
)
from manabot.env.observation import ObservationSpace
from manabot.eval.card_evaluation import trace_card_tables
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.infra.provenance import engine_source_paths, runtime_fingerprints
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.panels import ML, PERF


class Conditions(StrictModel):
    protocol_sha256: str
    panel: str
    deal: int
    update: int
    anchor: Anchor


class _TrainingOrigin(BaseModel):
    regime_digest: str
    weights: Literal["raw", "initial_raw"]


class _CheckpointOrigin(BaseModel):
    bc: _TrainingOrigin


def validate_training_origin(path: Path, protocol: Protocol, *, initial: bool) -> None:
    """Admit population/learning provenance, beyond ordinary world admission."""
    origin = _CheckpointOrigin.model_validate(
        torch.load(path, map_location="cpu", weights_only=True)
    ).bc
    if origin.regime_digest != canonical_sha256(
        protocol.regime().model_dump(mode="json")
    ):
        raise ValueError("checkpoint trained under a different regime/population")
    if (origin.weights == "initial_raw") != initial:
        raise ValueError("checkpoint weights differ from the declared update role")


@dataclass(frozen=True)
class Comparison:
    """An expected export/panel/anchor slot, even when it cannot be executed."""

    update: int
    panel: str
    anchor: Anchor
    expected_games: int
    matchup: Matchup | None


def runs(revision: str) -> list[Run]:
    """Two one-attempt training declarations; allocation still needs authorization."""
    protocol = read_protocol()
    job = protocol.job().job(
        repo=ROOT,
        revision=revision,
        max_hours=protocol.budget.training_job_hours,
        max_usd=Decimal(str(protocol.budget.training_usd_per_seed)),
    )
    frozen = TrainingJob.model_validate_json(job.config or "")
    assert frozen.source_bundle is not None
    for path in [*_source(), *engine_source_paths(), ROOT / "uv.lock"]:
        if frozen.source_bundle.files.get(str(path.relative_to(ROOT))) != file_sha256(
            path
        ):
            raise ValueError(
                f"commit does not contain the selected protocol source: {path.relative_to(ROOT)}"
            )
    return job.runs(*protocol.budget.seeds)


def _policy_source() -> list[Path]:
    """Keep the fixed opponent independent of deck/protocol revisions."""
    return sorted((ROOT / "manabot").rglob("*.py"))


def _source() -> list[Path]:
    return [
        *_policy_source(),
        ROOT / "experiments/growing_pool.py",
        ROOT / "experiments/growing_pool_protocol.py",
        ROOT / "experiments/growing_pool_exposure.py",
        PROTOCOL,
    ]


def comparisons(run: TrainingRun, protocol: Protocol) -> tuple[Comparison, ...]:
    if run.experiment != EXPERIMENT:
        raise ValueError("growing-pool comparisons require this experiment's exports")
    checkpoints = {}
    for checkpoint in run.checkpoints:
        update = checkpoint.clocks.get("updates")
        if update is None or update < 0 or update != int(update):
            raise ValueError("exports require nonnegative integer update coordinates")
        # A wall-bounded stop can publish its final raw export at the same
        # update as a monitoring export. The later admitted export owns that
        # coordinate; it does not replace any later planned milestone.
        if update != 0 or 0 not in checkpoints:
            checkpoints[int(update)] = checkpoint
    source = _policy_source()
    inference = Inference(source=source)
    initial = checkpoints.get(0)
    anchors: dict[Anchor, Model | Code | None] = {
        "heuristic": Code(
            name="heuristic", source=source, config={"kind": "heuristic"}
        ),
        "initial": initial.model(inference) if initial else None,
    }
    protocol_digest = protocol.digest
    result = []
    for contrast in protocol.budget.evaluation_rounds:
        update, label = contrast.update, contrast.anchor
        checkpoint = checkpoints.get(update)
        candidate = checkpoint.model(inference) if checkpoint else None
        anchor = anchors[label]
        for panel in protocol.manifest.panels:
            matchup = None
            if candidate is not None and anchor is not None:
                matchup = Matchup(
                    name=f"{run.name} u{update} {panel.id} vs {label}",
                    players=(candidate, anchor),
                    games=4,
                    tags=(panel.id, label),
                    conditions=tuple(
                        Conditions(
                            protocol_sha256=protocol_digest,
                            panel=panel.id,
                            deal=deal,
                            update=update,
                            anchor=label,
                        ).model_dump(mode="json")
                        for deal in panel.deals
                    ),
                )
            result.append(
                Comparison(update, panel.id, label, 4 * len(panel.deals), matchup)
            )
    return tuple(result)


def eval(state: ExperimentState) -> list[Job]:
    if state.experiment != EXPERIMENT:
        raise ValueError("wrong experiment for growing-pool evaluation")
    protocol = read_protocol()
    matchups = tuple(
        slot.matchup
        for run in state.runs
        for slot in comparisons(run, protocol)
        if slot.matchup is not None
    )
    if not matchups:
        return []
    return [
        evaluation(
            Benchmark(
                name=protocol.id,
                environment=Environment(
                    name=protocol.id,
                    files=protocol.digest,
                    config={"status": protocol.status},
                ),
                matchups=matchups,
            ),
            # Discovery runs this file through runpy. The worker needs this canonical
            # module's callback, not a function in the <run_path> namespace.
            play=growing_pool.play,
            workers=1,
            max_hours=protocol.budget.evaluation_hours,
            max_usd=Decimal(str(protocol.budget.evaluation_usd)),
            max_attempts=1,
        )
    ]


def deck_matchup(protocol: Protocol, panel: Panel) -> DeckMatchup:
    entries = {entry.id: entry for entry in protocol.manifest.entries}
    return DeckMatchup(
        population_sha256=protocol.manifest.digest,
        entries=panel.entries,
        lists=tuple(entries[name].deck for name in panel.entries),
    )


def _registration(
    player: Model | Code,
    name: str,
    matchup: DeckMatchup,
    protocol: Protocol,
) -> tuple[PlayerRegistration, Path | None]:
    setup = matchup.match()
    fingerprints = runtime_fingerprints(
        match_config=setup,
        observation_space=ObservationSpace(protocol.observation),
    )
    common = dict(
        player_id=name,
        display_name=player.name,
        role="challenger" if name == "candidate" else "anchor",
        world=protocol.world.world_version,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(setup.model_dump(mode="json")),
        player_seed_derivation_id="arena-pair-deal-player-v1",
        compute_class_id="policy-cpu-one-thread-one-pass",
    )
    if isinstance(player, Code):
        if player.config != {"kind": "heuristic"}:
            raise ValueError("opponent must use full-support heuristic")
        return PlayerRegistration(
            **common,
            runner_kind="code",
            player_spec={"kind": "heuristic"},
            source_sha256=canonical_sha256(
                {
                    str(path.relative_to(ROOT)): file_sha256(path)
                    for path in _policy_source()
                }
            ),
        ), None
    path = local_checkpoint_file(player)
    validate_training_origin(
        path, protocol, initial=player.checkpoint.at.get("updates") == 0
    )
    agent, space = load_checkpoint(str(path))
    if space.encoder.config != protocol.observation or agent.config != protocol.agent:
        raise ValueError("checkpoint architecture/input differs from frozen pilot")
    digest = file_sha256(path)
    return PlayerRegistration(
        **common,
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": False,
            "allow_deck_repetition": True,
        },
        checkpoint_sha256=digest,
        checkpoint_bytes=path.stat().st_size,
        parameter_count=sum(p.numel() for p in agent.parameters()),
        artifact_id=f"mtg245/{digest}",
    ), path


def outcome_rows(result: EvaluationResult, conditions: Conditions) -> list[Row]:
    """Missing and failed legs remain in the denominator, never zero-score wins."""
    by_leg = {row.leg: row for row in result.rows}
    rows: list[Row] = []
    for leg in range(4):
        row = by_leg.get(leg)
        valid = row is not None and row.valid
        score = row.score_a if valid else None
        for seat in (0, 1):
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": (leg + seat) % 2,
                    "panel": conditions.panel,
                    "deal": conditions.deal,
                    "update": conditions.update,
                    "anchor": conditions.anchor,
                    "protocol_sha256": conditions.protocol_sha256,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": valid,
                    "attempted": row is not None,
                    "failure": row.failure if row else result.error or "missing game",
                    "seconds": row.game_seconds if row else None,
                }
            )
    return rows


def play(match: Match) -> Result:
    """One frozen pair/deal through all four policy/deck/seat legs and replay."""
    torch.set_num_threads(1)
    protocol = read_protocol()
    conditions = Conditions.model_validate(match.conditions)
    panel = next(
        (p for p in protocol.manifest.panels if p.id == conditions.panel), None
    )
    if (
        conditions.protocol_sha256 != protocol.digest
        or panel is None
        or conditions.deal not in panel.deals
        or match.games != 4
        or EvaluationRound(conditions.update, conditions.anchor)
        not in protocol.budget.evaluation_rounds
    ):
        raise ValueError("evaluation conditions differ from frozen schedule")
    if len(match.players) != 2 or not isinstance(match.players[0], Model):
        raise ValueError("comparison needs a checkpoint candidate and one anchor")
    if isinstance(match.players[1], Code) != (conditions.anchor == "heuristic"):
        raise ValueError("anchor differs from declared comparison")
    if match.players[0].checkpoint.at.get("updates") != conditions.update:
        raise ValueError("candidate export differs from the declared update")
    if isinstance(match.players[1], Model) and (
        match.players[1].checkpoint.at.get("updates") != 0
        or match.players[1].checkpoint.line != match.players[0].checkpoint.line
    ):
        raise ValueError("initialization anchor must belong to the candidate's learner")
    matchup = deck_matchup(protocol, panel)
    candidate, candidate_path = _registration(
        match.players[0], "candidate", matchup, protocol
    )
    opponent, opponent_path = _registration(
        match.players[1], "opponent", matchup, protocol
    )
    key = EvaluationKey(
        world=protocol.world.world_version,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=protocol.id,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=opponent.identity_sha256,
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    output = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    evidence = output.parent / f"{match.key}-cards"
    result = evaluate(
        EvaluationRequest(
            key=key,
            candidate=candidate,
            opponent=opponent,
            checkpoint_paths={
                name: str(path)
                for name, path in (
                    ("candidate", candidate_path),
                    ("opponent", opponent_path),
                )
                if path is not None
            },
            deal_seeds=(conditions.deal,),
            authored_matchup=matchup,
            game_seconds=protocol.budget.game_seconds,
            max_commands=protocol.budget.max_commands,
            bootstrap_seed=2459001,
            purpose="growing-pool-protocol-check",
        ),
        out_dir=output,
        card_evidence_dir=evidence,
    )
    delivered = Result(tables={}, blobs=[])
    trace_path = output / "trace.json"
    if trace_path.exists():
        trace = TypeAdapter(dict[str, JsonValue]).validate_json(trace_path.read_bytes())
        if "card_evidence" in trace:
            delivered = trace_card_tables(evidence, trace, ("candidate", "opponent"))
    return Result(
        tables={
            **delivered.tables,
            "games": outcome_rows(result, conditions),
            "decisions": decision_rows(delivered.blobs),
        },
        blobs=delivered.blobs,
    )


def report() -> Report:
    """Descriptive scores only; completeness and paired contrasts precede claims."""
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(
                title="Fixed heuristic by panel",
                matches=Matches(tag="heuristic"),
                split="panel",
            ),
            Curve(
                title="Endpoint vs initialization",
                matches=Matches(tag="initial"),
                split="panel",
            ),
        ),
        perf=PERF,
        ml=ML,
    )
