"""MTG-213 prospective v7 fixed-corpus TLA self-play and paired checkpoint evaluation.

17Lands supplies the CC BY 4.0 trophy source; see the frozen protocol and corpus
receipt. `runs` uses TrainingJob/execute_regime, while `eval` uses Etude and the
shared replay-checked evaluator. Neither path fetches or changes deck eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
from typing import Literal

from etude.eval.match import Row, Tables
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Matchup,
    Model,
)
from etude.experiment import Checkpoint, ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Report, Score
from pydantic import Field
import torch

from experiments import tla_pilot_v6
from manabot.env.observation import ObservationSpace
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TrainingJob
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.regime import TrainingRegime
import managym

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "tla_pilot_v6"
REGIME = ROOT / "experiments/regimes/tla-trophy-213-v6.json"
PROTOCOL = ROOT / "experiments/protocols/tla-trophy-213-v6.json"
GAMES_PER_PAIR = 4

type Anchor = Literal["heuristic", "initial"]


class Pair(StrictModel):
    first: int = Field(ge=0)
    second: int = Field(ge=0)
    deal: int = Field(ge=0)


class RuntimeWorld(StrictModel):
    world_version: str
    card_digest: str


class Protocol(StrictModel):
    schema_version: Literal[3] = 3
    observation_shapes: dict[str, tuple[int, ...]]
    runtime_world: RuntimeWorld
    source_base: str
    predecessor_protocol_sha256: str
    corpus_sha256: str
    regime_sha256: str
    training_seed: int
    deck_seed: int
    pairs: tuple[Pair, ...]
    milestones: tuple[int, ...]
    game_seconds: float = Field(default=120, gt=0)
    max_commands: int = Field(default=10000, ge=1)
    bootstrap_seed: int = 21392001
    max_paid_usd: Decimal = Decimal("10")
    allocation_usd: Decimal = Decimal("1")
    allocation_hours: float = 1.0


def frozen() -> tuple[TrainingRegime, Protocol]:
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    if file_sha256(REGIME) != protocol.regime_sha256:
        raise ValueError("pilot regime differs from frozen protocol")
    regime = TrainingRegime.model_validate_json(REGIME.read_bytes())
    if protocol.runtime_world.model_dump() != managym.world_identity():
        raise ValueError("pilot runtime world differs from the prospective protocol")
    if regime.world != protocol.runtime_world.world_version:
        raise ValueError("pilot regime world differs from its protocol")
    # The hashed regime owns the configuration; the protocol also pins derived
    # shapes so an encoder change cannot silently alter its allocation.
    if ObservationSpace(regime.observation).shapes != protocol.observation_shapes:
        raise ValueError("pilot tensor capacities differ from frozen protocol")
    if regime.deck_source is None or regime.deck_source.entry_ids is None:
        raise ValueError("pilot requires a frozen population with event IDs")
    if regime.deck_source.sampling != "uniform-independent-v1":
        raise ValueError("pilot requires independent per-seat population draws")
    if any(
        max(p.first, p.second) >= len(regime.deck_source.decks) for p in protocol.pairs
    ):
        raise ValueError("evaluation pair outside frozen eligible population")
    return regime, protocol


def runs(revision: str) -> list[Run]:
    """One capped GPU allocation; no automatic retry, rental or budget expansion."""
    regime, protocol = frozen()
    return (
        TrainingJob(
            regime=regime,
            experiment=EXPERIMENT,
            milestones=protocol.milestones,
            finalization_seconds=300,
            stop_reserve_seconds=120,
            validate_numerics=True,
        )
        .job(
            repo=ROOT,
            revision=revision,
            max_hours=protocol.allocation_hours,
            max_usd=protocol.allocation_usd,
        )
        .runs(protocol.training_seed)
    )


def _source() -> list[Path]:
    return [
        *sorted((ROOT / "manabot").rglob("*.py")),
        Path(__file__),
        REGIME,
        PROTOCOL,
    ]


def environment(protocol: Protocol) -> Environment:
    return Environment(
        name="tla-trophy-213-v6",
        files=canonical_sha256(
            {
                "world": managym.world_identity(),
                "corpus": protocol.corpus_sha256,
                "protocol": protocol.model_dump(mode="json"),
            }
        ),
        config={"purpose": "fixed-corpus-development-no-heldout-deck-claim"},
    )


@dataclass(frozen=True)
class ScheduledComparison:
    """One expected comparison; absent checkpoints leave an unplayable slot."""

    update: int | None
    anchor: Anchor
    endpoint: bool
    checkpoint: Checkpoint | None
    matchup: Matchup | None


def comparison_schedule(
    run: TrainingRun, protocol: Protocol
) -> list[ScheduledComparison]:
    """One schedule for execution and analysis, including missing comparisons.

    Frozen milestones remain expected even for an early endpoint. Until a run
    finishes, its endpoint has unknown coordinates. Only comparisons with both
    required checkpoints can be played, but all slots count in the analysis.
    """
    if run.experiment != EXPERIMENT:
        raise ValueError("v6 comparisons require a fresh v6 initialization")
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source, config={"kind": "heuristic"})
    conditions = tuple(pair.model_dump(mode="json") for pair in protocol.pairs)
    checkpoints = {int(c.clocks["updates"]): c for c in run.checkpoints}
    endpoint = int(run.final.clocks["updates"]) if run.final else None
    wanted = {0, *protocol.milestones}
    if endpoint is not None:
        wanted.add(endpoint)
    updates: list[int | None] = sorted(wanted)
    if endpoint is None:
        updates.append(None)
    initial = checkpoints.get(0)
    anchors: dict[Anchor, Model | Code | None] = {
        "heuristic": heuristic,
        "initial": initial.model(inference) if initial else None,
    }
    schedule: list[ScheduledComparison] = []
    for update in updates:
        checkpoint = checkpoints.get(update) if update is not None else None
        candidate = checkpoint.model(inference) if checkpoint else None
        for label, anchor in anchors.items():
            if update == 0 and label == "initial":
                continue
            schedule.append(
                ScheduledComparison(
                    update=update,
                    anchor=label,
                    endpoint=update == endpoint,
                    checkpoint=checkpoint,
                    matchup=Matchup(
                        name=f"{run.name} u{update} vs {label}",
                        players=(candidate, anchor),
                        conditions=conditions,
                        games=GAMES_PER_PAIR,
                        tags=(label,),
                    )
                    if candidate is not None and anchor is not None
                    else None,
                )
            )
    return schedule


def eval(state: ExperimentState) -> list[Job]:
    """Play available comparisons; Etude owns already-delivered matches."""
    if state.experiment != EXPERIMENT:
        raise ValueError("v6 evaluation requires fresh v6 training runs")
    _, protocol = frozen()
    matchups = tuple(
        comparison.matchup
        for run in state.runs
        for comparison in comparison_schedule(run, protocol)
        if comparison.matchup is not None
    )
    if not matchups:
        return []
    return [
        evaluation(
            Benchmark(
                name="tla-trophy-paired-v6",
                environment=environment(protocol),
                matchups=matchups,
            ),
            # Etude discovers this file with runpy; callbacks need an importable
            # module identity even when eval itself came from <run_path>.
            play=tla_pilot_v6.play,
            workers=1,
            max_hours=4,
        )
    ]


def _registration(
    player: Model | Code,
    name: str,
    matchup: DeckMatchup,
    observation: ObservationConfig,
) -> tuple[PlayerRegistration, Path | None]:
    setup = matchup.match()
    fingerprints = runtime_fingerprints(match_config=setup)
    common = dict(
        player_id=name,
        display_name=player.name,
        role="challenger" if name == "candidate" else "anchor",
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(setup.model_dump(mode="json")),
        player_seed_derivation_id="arena-pair-deal-player-v1",
        compute_class_id="policy-cpu-one-thread-one-pass",
    )
    if isinstance(player, Code):
        return PlayerRegistration(
            **common,
            runner_kind="code",
            player_spec={"kind": "heuristic"},
            source_sha256=canonical_sha256(
                {str(p.relative_to(ROOT)): file_sha256(p) for p in _source()}
            ),
        ), None
    path = local_checkpoint_file(player)
    agent, space = load_checkpoint(str(path))
    if space.encoder.config != observation:
        raise ValueError("checkpoint observation differs from pilot")
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
        artifact_id=f"mtg213/{digest}",
    ), path


def play(match: Match) -> Tables:
    """One sampled pair/deal, four policy/deck/seat legs, with failure denominators."""
    torch.set_num_threads(1)
    regime, protocol = frozen()
    source = regime.deck_source
    assert source is not None and source.entry_ids is not None
    pair = Pair.model_validate(match.conditions)
    if pair not in protocol.pairs or match.games != GAMES_PER_PAIR:
        raise ValueError("evaluation conditions differ from frozen schedule")
    matchup = DeckMatchup(
        population_sha256=protocol.corpus_sha256,
        entries=(source.entry_ids[pair.first], source.entry_ids[pair.second]),
        lists=(source.decks[pair.first], source.decks[pair.second]),
    )
    candidate, candidate_path = _registration(
        match.players[0], "candidate", matchup, regime.observation
    )
    opponent, opponent_path = _registration(
        match.players[1], "opponent", matchup, regime.observation
    )
    paths = {
        name: str(path)
        for name, path in (("candidate", candidate_path), ("opponent", opponent_path))
        if path is not None
    }
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="tla-trophy-213-v6",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    result = evaluate(
        EvaluationRequest(
            key=key,
            candidate=candidate,
            opponent=opponent,
            checkpoint_paths=paths,
            deal_seeds=(pair.deal,),
            authored_matchup=matchup,
            game_seconds=protocol.game_seconds,
            max_commands=protocol.max_commands,
            bootstrap_seed=protocol.bootstrap_seed,
            purpose="tla-trophy-fixed-corpus-development",
        ),
        out_dir=Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key,
    )
    by_leg = {row.leg: row for row in result.rows}
    rows: list[Row] = []
    for leg in range(GAMES_PER_PAIR):
        row = by_leg.get(leg)
        valid = row is not None and row.valid
        score = row.score_a if valid else None
        decks = matchup.decks(leg)
        for seat in (0, 1):
            game_seat = (leg + seat) % 2
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": game_seat,
                    "deck": decks[game_seat],
                    "opponent_deck": decks[1 - game_seat],
                    "deal_seed": pair.deal,
                    "corpus_sha256": protocol.corpus_sha256,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": valid,
                    "attempted": row is not None,
                    "terminated": row.terminated if row else False,
                    "truncated": row.truncated if row else False,
                    "failure": row.failure if row else result.error or "missing game",
                    "seconds": row.game_seconds if row else None,
                }
            )
    return {"games": rows}


def report() -> Report:
    """Descriptive curves; `experiments.tla_pilot_analysis` owns qualified claims."""
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(
                title="Trophy decks vs fixed heuristic",
                matches=Matches(tag="heuristic"),
            ),
            Curve(
                title="Trophy decks vs initialization", matches=Matches(tag="initial")
            ),
        ),
    )
