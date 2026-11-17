"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report (`opponent_memory.md`) owns the design, evidence and open decisions.
`training_jobs` declares three trunks, or six EMA branches with `trunks`.
Frozen calibration runs retain their original source commits and receipts.
`census` and `probe` are the USD0 commands. Nothing runs on import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
import random
from typing import Literal

from etude.eval.match import Tables
from etude.eval.records import Benchmark, Environment, Inference, Match, Seating
from etude.experiment import observe
from etude.jobs import EvalJob, ExperimentState, TrainingRunSummary
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.spec import RunRequest
from etude.storage import Storage, run_prefix, storage_from_uri
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import DECKS, PACK
from experiments.learning_signal import Design, Pairing
from manabot.experiment.job import PretrainedInitialization, TrainingJob, machine_setup
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import AuthoredMatchup
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime
import managym
from managym.disclosure import DecklistDisclosure


@dataclass(frozen=True)
class Arm:
    """One encoder/summary choice over the shared viewer decision log."""

    name: str
    encoder: Literal["none", "transformer", "gru"]
    summaries: int


# K=1 remains frozen; throughput-first direction keeps only the GRU encoder.
ARMS = (
    Arm("control", "none", 0),
    Arm("gru-k1", "gru", 1),
)
SEEDS = (39101, 39102, 39103)
DISCLOSURES: tuple[DecklistDisclosure, ...] = ("open", "closed")
# Inclusive of calibration, training, evaluation, startup and failed attempts.
MAX_USD = 50


@dataclass(frozen=True)
class PilotStage:
    """A training stage's learner horizon and whole-job resource limits."""

    decisions: int
    updates: int
    max_hours: float
    max_usd: Decimal


TRUNK = PilotStage(3_300_000, 90, 3.5, Decimal("2.40"))
BRANCH = PilotStage(4_400_000, 120, 5.0, Decimal("3.30"))
MILESTONES = (20, 40, 60)
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem


def training_jobs(revision: str, trunks: str = "") -> list[RunRequest]:
    """Declare trunks, or branch both arms from every completed published trunk."""
    if trunks:
        storage = storage_from_uri(trunks)
        state = observe(storage, EXPERIMENT)
        parents: dict[int, PretrainedInitialization] = {}
        for seed in SEEDS:
            runs = [
                run
                for run in state.runs
                if run.launch == "decision-memory-trunk" and run.seed == seed
            ]
            if len(runs) != 1:
                raise ValueError(f"seed {seed}: expected one trunk, found {len(runs)}")
            run = runs[0]
            if run.state != "finished":
                raise ValueError(f"seed {seed}: trunk is unfinished ({run.state})")
            parents[seed] = _published_trunk(storage, run.name, seed)
        return branch_jobs(
            revision,
            regime=pilot_recipe(BRANCH),
            trunks=parents,
            max_hours=BRANCH.max_hours,
            max_usd=BRANCH.max_usd,
            milestones=MILESTONES,
        )
    recipe = pilot_recipe(TRUNK)
    recipe.id = "decision-memory-trunk"
    return (
        TrainingJob(regime=recipe, experiment=EXPERIMENT)
        .job(
            repo=ROOT,
            revision=revision,
            max_hours=TRUNK.max_hours,
            max_usd=TRUNK.max_usd,
        )
        .runs(*SEEDS)
    )


def _published_trunk(storage: Storage, run: str, seed: int) -> PretrainedInitialization:
    """Resolve the sole completed attempt; failed retries cannot supply weights."""
    folder = run_prefix(run)
    records = [
        key
        for key in storage.keys(folder)
        if key.startswith(f"{folder}attempts/") and key.endswith("/training/run.json")
    ]
    try:
        completed = [
            key
            for key in records
            if TrainingRun.model_validate_json(storage.get(key)).status == "completed"
        ]
        if len(completed) != 1:
            raise ValueError(
                f"trunk has {len(completed)} completed attempt records, expected one"
            )
        trunk = PretrainedInitialization.from_storage(
            storage,
            completed[0],
            source_stage="self-play",
            weights="ema",
        )
    except ValueError as error:
        raise ValueError(f"seed {seed}: {error}") from error
    if trunk.source_seed != seed:
        raise ValueError(f"seed {seed}: trunk receipt names seed {trunk.source_seed}")
    return trunk


def branch_jobs(
    revision: str,
    *,
    regime: TrainingRegime,
    trunks: Mapping[int, PretrainedInitialization],
    max_hours: float,
    max_usd: Decimal,
    milestones: tuple[int, ...],
) -> list[RunRequest]:
    """Freeze control/GRU branches from published trunk inputs without launching.

    Each seed's two arms share one producer receipt and EMA export. Their
    optimizer and learner-decision schedule start fresh at the branch point.
    """
    if not trunks or any(seed not in SEEDS for seed in trunks):
        raise ValueError("branches require declared pilot seed labels")
    if regime.agent.decision_memory != "none":
        raise ValueError("the shared branch recipe must describe the control")
    if any(trunk.weights != "ema" for trunk in trunks.values()):
        raise ValueError("all branches use the same EMA trunk role")
    if any(trunk.source_seed != seed for seed, trunk in trunks.items()):
        raise ValueError("each branch seed label must match its producer seed")
    jobs: list[RunRequest] = []
    for seed, trunk in sorted(trunks.items()):
        for arm in ARMS:
            configuration = regime.model_copy(deep=True)
            configuration.id = f"decision-memory-{arm.name}"
            configuration.agent.decision_memory = arm.encoder
            if arm.summaries:
                configuration.agent.decision_memory_summaries = arm.summaries
                configuration.agent.decision_memory_gate_scale = 2.0
            job = TrainingJob(
                regime=configuration,
                experiment=EXPERIMENT,
                pretrained=trunk,
                milestones=milestones,
            ).job(
                repo=ROOT,
                revision=revision,
                max_hours=max_hours,
                max_usd=max_usd,
            )
            jobs.extend(job.runs(seed))
    return jobs


def pilot_design() -> Design:
    """Two hundred held-out deals; shorter readouts use the same first fifty."""
    rng = random.Random(39100)
    pairs = tuple(combinations(DECKS, 2))
    return Design(
        name="decision-memory-elemental-v1",
        pack=PACK,
        pairings=tuple(
            Pairing(role_a=a, role_b=b, deal=3_910_000_000 + i)
            for i, (a, b) in enumerate(rng.choices(pairs, k=200))
        ),
        matchup_mode="cross",
        game_seconds=600,
    )


class Conditions(BaseModel):
    """A pilot pairing and native decklist disclosure, fixed before scoring."""

    model_config = ConfigDict(extra="forbid")
    design: str
    role_a: str
    role_b: str
    deal: int
    decklists: DecklistDisclosure


def seating_name(seed: int, disclosure: str, tag: str, point: int | str) -> str:
    return f"seed-{seed}-{disclosure}-{tag}-{point}"


def _branch(state: ExperimentState, seed: int, arm: str) -> TrainingRunSummary | None:
    found = [
        run
        for run in state.runs
        if run.launch == f"decision-memory-{arm}" and run.seed == seed
    ]
    if len(found) > 1:
        raise ValueError(f"seed {seed}: multiple {arm} branches")
    return found[0] if found else None


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Equal-update interiors and each completed branch's own final checkpoint."""
    design = pilot_design()
    source = [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]
    intact = Inference(source=source)
    blanked = Inference(source=source, config={"blank_decision_memory": True})
    found: list[Seating] = []
    for seed in SEEDS:
        control, gru = (_branch(state, seed, arm.name) for arm in ARMS)
        if control is None or gru is None:
            continue
        points = [
            (
                update,
                gru.checkpoints.at("updates", update).last,
                control.checkpoints.at("updates", update).last,
                "control",
            )
            for update in MILESTONES
        ]
        points += [
            ("final", gru.final, control.final, "control"),
            ("final", gru.final, gru.final, "reliance"),
        ]
        for point, first, second, tag in points:
            if first is None or second is None:
                continue
            pairings = (
                design.pairings
                if point == "final" and tag == "control"
                else design.pairings[:50]
            )
            for disclosure in DISCLOSURES:
                conditions = tuple(
                    Conditions(
                        design=design.name,
                        decklists=disclosure,
                        **pairing.model_dump(),
                    ).model_dump()
                    for pairing in pairings
                )
                found.append(
                    Seating(
                        name=seating_name(seed, disclosure, tag, point),
                        players=(
                            first.player(intact),
                            second.player(blanked if tag == "reliance" else intact),
                        ),
                        configs=conditions,
                        games=design.legs,
                        tags=(tag,),
                        # Fixed checkpoint pair: report this seed independently
                        # of the other runs grouped under the same job.
                        seeds="one",
                    )
                )
    return tuple(found)


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """Play stored branch exports; a capped pass retains matches for the next pass."""
    seatings = evaluation_matchups(state)
    if not seatings:
        return []
    design = pilot_design()
    return [
        EvalJob(
            tournament=Benchmark(
                name=design.name,
                environment=Environment(
                    name=design.name,
                    files=canonical_sha256(
                        {
                            "world": managym.world_identity(),
                            "design": design.record(),
                            "disclosures": DISCLOSURES,
                        }
                    ),
                    config={"purpose": "held-out-paired-deals"},
                ),
                seatings=seatings,
            ),
            play=play,
            extras=("evaluation",),
            workers=28,
            max_hours=2.0,
            max_usd=Decimal("4.00"),
            setup=machine_setup(" --extra evaluation"),
        )
    ]


def play(match: Match) -> Tables:
    """Validate pilot conditions and delegate the paired deal to the shared arena."""
    import torch

    from manabot.backends.torch.checkpoint import BACKEND
    from manabot.eval.pairing import play_pairing

    torch.set_num_threads(1)
    conditions = Conditions.model_validate(match.config)
    design = pilot_design()
    pairing = Pairing(
        role_a=conditions.role_a, role_b=conditions.role_b, deal=conditions.deal
    )
    if conditions.design != design.name or pairing not in design.pairings:
        raise ValueError("evaluation conditions differ from the frozen schedule")
    return play_pairing(
        match,
        matchup=AuthoredMatchup(
            pack_key=design.pack,
            role_a=pairing.role_a,
            role_b=pairing.role_b,
            decklists=conditions.decklists,
        ),
        deal=pairing.deal,
        matchup_mode=design.matchup_mode,
        game_seconds=design.game_seconds,
        arena=design.name,
        backend=BACKEND,
    )


def report() -> Report:
    """Use the existing Elemental matchup panels and the learner-decision clock."""
    from manabot.training.panels import ML, PERF

    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="learner_transitions",
        evals=(
            *(
                Curve(
                    title=f"Head-to-head against control — seed {seed}",
                    matches=tuple(
                        Matches(
                            seating=seating_name(seed, disclosure, "control", point)
                        )
                        for disclosure in DISCLOSURES
                        for point in (*MILESTONES, "final")
                    ),
                    split="decklists",
                )
                for seed in SEEDS
            ),
            Curve(
                title="Intact memory against blanked memory",
                matches=tuple(
                    Matches(seating=seating_name(seed, disclosure, "reliance", "final"))
                    for seed in SEEDS
                    for disclosure in DISCLOSURES
                ),
                split="decklists",
            ),
            Matrix(
                title="Descriptive arm versus control matchups",
                rows="opponent_deck",
                columns="deck",
                matches=Matches(tag="control"),
            ),
            Ranking(
                title="On the play or the draw",
                by="on_play",
                matches=Matches(tag="control"),
            ),
        ),
        perf=PERF,
        ml=ML,
    )


def pilot_recipe(limits: PilotStage) -> TrainingRegime:
    """Shared control recipe, bounded by the stage's horizon and job wall."""
    from manabot.infra.config import ModelConfig, ObservationConfig
    from manabot.recipes.builders import sampled_self_play
    from manabot.training.decks import UniformDecks
    from manabot.training.regime import (
        DampedPolicyLearning,
        LearnerDecisionSchedule,
        TrainSelfPlay,
    )
    import managym

    # Reserve twenty minutes of the job for setup and publishing artifacts.
    wall_seconds = limits.max_hours * 3600 - 1200
    observation = ObservationConfig(
        max_decision_records=512,
        max_decision_reference_details=16384,
        max_decision_references=256,
        max_decision_members=8192,
    )
    recipe = sampled_self_play(
        id="decision-memory-control",
        world=managym.WORLD_VERSION,
        agent=ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=64,
            attention_layers=2,
            num_attention_heads=4,
            value_aggregation="masked_mean",
            policy_film=True,
            decision_memory_summaries=1,
        ),
        # Bind the explicit execution limits before enabling memory capture.
        observation=ObservationConfig(),
        deck_source=UniformDecks(
            decks=tuple(
                managym.Deck(player.decklist, player.sideboard)
                for key in managym.authored_deck_keys("elemental-world-v2")
                for player in [managym.authored_deck_setup("elemental-world-v2", key)]
            ),
            sampling="uniform-independent-v1",
            decklists="mixed",
        ),
        # A sampled deck source owns mixed disclosure and independent mirrors;
        # TrainingRegime forbids combining it with the older curriculum flag.
        wall_seconds=wall_seconds,
        seconds_per_update=60,
        workers=4,
    )
    stage = recipe.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = limits.updates
    stage.target_learner_transitions = limits.decisions
    assert isinstance(stage.learning, DampedPolicyLearning)
    stage.learning.iteration_schedule = None
    stage.learning.decision_schedule = LearnerDecisionSchedule(
        transitions=limits.decisions
    )
    # Capacity overflow fails the run: no truncation and no abort-and-continue.
    stage.game_errors = None
    stage.execution.collection_rows = 128
    stage.execution.bootstrap_rows = 128
    stage.execution.learner_rows = 128
    stage.execution.precision = "float32"
    recipe.observation = observation
    return TrainingRegime.model_validate(recipe.authored_dump())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    census_parser = operations.add_parser("census")
    census_parser.add_argument("--out", type=Path, required=True)
    probe_parser = operations.add_parser("probe")
    probe_parser.add_argument("--out", type=Path, required=True)
    probe_parser.add_argument("--epochs", type=int, default=8)
    probe_parser.add_argument("--max-games", type=int, default=120)
    probe_parser.add_argument("--wall-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.operation == "census":
        from manabot.verify.decision_memory_capacity import run_census

        run_census(args.out)
        return
    if args.operation == "probe":
        from manabot.backends.torch.decision_memory_probe import (
            MAX_WALL_SECONDS,
            MIN_GAMES,
            run_probe,
        )

        if args.epochs < 1:
            probe_parser.error("--epochs must be at least 1")
        if args.max_games < MIN_GAMES:
            probe_parser.error(f"--max-games must be at least {MIN_GAMES}")
        if not 0 < args.wall_seconds <= MAX_WALL_SECONDS:
            probe_parser.error(f"--wall-seconds must be in (0, {MAX_WALL_SECONDS:g}]")
        if args.out.exists():
            probe_parser.error(f"--out {args.out} exists; choose a fresh directory")
        run_probe(
            args.out,
            epochs=args.epochs,
            max_games=args.max_games,
            wall_seconds=args.wall_seconds,
        )
        return


if __name__ == "__main__":
    main()
