"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report (`opponent_memory.md`) owns the design, evidence and open decisions.
`training_jobs` declares three trunks, or six EMA branches with `trunks`;
each `PilotStage` owns its stage's horizon and caps.
Frozen calibration runs retain their original source commits and receipts.
`census` and `probe` are the USD0 commands. Nothing runs on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
import random

from etude.eval.match import Tables
from etude.eval.records import Benchmark, Environment, Inference, Match, Seating
from etude.experiment import observe
from etude.jobs import EvalJob, ExperimentState, TrainingRunSummary
from etude.launch.records import Job
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.spec import RunRequest
from etude.storage import Storage, run_prefix, storage_from_uri
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import DECKS, PACK
from experiments.learning_signal import Design, Pairing
from manabot.experiment.job import PretrainedInitialization, TrainingJob, machine_setup
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256
from manabot.recipes.builders import sampled_self_play
from manabot.sim.matchup import AuthoredMatchup
from manabot.training.decks import UniformDecks
from manabot.training.panels import ML, PERF
from manabot.training.records import TrainingRun
from manabot.training.regime import (
    DampedPolicyLearning,
    LearnerDecisionSchedule,
    TrainingRegime,
    TrainSelfPlay,
)
import managym
from managym.disclosure import DecklistDisclosure

# Arm name -> encoder over the shared viewer decision log. One summary (K=1)
# is the model default; throughput-first direction keeps only the GRU encoder.
ARMS = {"control": "none", "gru-k1": "gru"}
SEEDS = (39101, 39102, 39103)
DISCLOSURES: tuple[DecklistDisclosure, ...] = ("open", "closed")


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
# Trunk launch label. "trunk" names the 2026-11-17 runs that overflowed the
# 16,384-detail layout; their failed records stay as evidence.
TRUNK_NAME = "trunk-2"
# Seed 39103 lost its only trunk-2 attempt to a provider capacity refusal
# before any work; its trunk is the same recipe under a fresh run identity.
TRUNK_NAMES = {39101: TRUNK_NAME, 39102: TRUNK_NAME, 39103: "trunk-3"}
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem


def training_jobs(revision: str, trunks: str = "") -> list[RunRequest]:
    """Declare trunks, or branch both arms from every completed published trunk.

    Each seed's two arms share one producer receipt and EMA export. Their
    optimizer and learner-decision schedule start fresh at the branch point.
    """
    if not trunks:
        return [
            request
            for seed, name in TRUNK_NAMES.items()
            for request in _job(revision, TRUNK, name).runs(seed)
        ]
    storage = storage_from_uri(trunks)
    state = observe(storage, EXPERIMENT)
    parents: dict[int, PretrainedInitialization] = {}
    for seed in SEEDS:
        runs = _runs(state, TRUNK_NAMES[seed], seed)
        if len(runs) != 1:
            raise ValueError(
                f"seed {seed}: expected one decision-memory-{TRUNK_NAMES[seed]} run in {trunks}, "
                f"found {len(runs)}; launch and finish the trunks first"
            )
        if runs[0].state != "finished":
            raise ValueError(f"seed {seed}: trunk is unfinished ({runs[0].state})")
        parents[seed] = _published_trunk(storage, runs[0].name, seed)
    return [
        request
        for seed, trunk in parents.items()
        for arm in ARMS
        for request in _job(revision, BRANCH, arm, trunk).runs(seed)
    ]


def _job(
    revision: str,
    limits: PilotStage,
    name: str,
    trunk: PretrainedInitialization | None = None,
) -> Job:
    """One capped Etude job: a trunk, or an arm started from a trunk's EMA."""
    recipe = pilot_recipe(limits)
    recipe.id = f"decision-memory-{name}"
    if trunk is not None:
        recipe.agent.decision_memory = ARMS[name]
        if ARMS[name] != "none":
            recipe.agent.decision_memory_gate_scale = 2.0
    return (
        TrainingJob(
            regime=recipe,
            experiment=EXPERIMENT,
            pretrained=trunk,
            milestones=MILESTONES if trunk is not None else (),
        )
        .job(
            repo=ROOT,
            revision=revision,
            max_hours=limits.max_hours,
            max_usd=limits.max_usd,
            # A provider capacity refusal consumes an attempt at no cost.
        )
        .model_copy(update={"max_attempts": 3})
    )


def _runs(state: ExperimentState, name: str, seed: int) -> list[TrainingRunSummary]:
    return [
        run
        for run in state.runs
        if run.launch == f"decision-memory-{name}" and run.seed == seed
    ]


def _published_trunk(storage: Storage, run: str, seed: int) -> PretrainedInitialization:
    """Resolve the sole completed attempt; failed retries cannot supply weights."""
    folder = run_prefix(run)
    try:
        completed = [
            key
            for key in storage.keys(folder)
            if key.startswith(f"{folder}attempts/")
            and key.endswith("/training/run.json")
            and TrainingRun.model_validate_json(storage.get(key)).status == "completed"
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


# Two hundred held-out deals; shorter readouts use the same first fifty.
DESIGN = Design(
    name="decision-memory-elemental-v1",
    pack=PACK,
    pairings=tuple(
        Pairing(role_a=a, role_b=b, deal=3_910_000_000 + i)
        for i, (a, b) in enumerate(
            random.Random(39100).choices(tuple(combinations(DECKS, 2)), k=200)
        )
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


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Equal-update interiors and each completed branch's own final checkpoint."""
    source = [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]
    intact = Inference(source=source)
    blanked = Inference(source=source, config={"blank_decision_memory": True})
    found: list[Seating] = []
    for seed in SEEDS:
        branches: list[TrainingRunSummary] = []
        for arm in ARMS:
            runs = _runs(state, arm, seed)
            if len(runs) > 1:
                raise ValueError(f"seed {seed}: multiple {arm} branches")
            branches += runs
        if len(branches) != len(ARMS):
            continue
        control, gru = branches
        # (point, tag, checkpoint, opposing checkpoint, opposing inference, deals)
        lines = [
            (
                update,
                "control",
                gru.checkpoints.at("updates", update).last,
                control.checkpoints.at("updates", update).last,
                intact,
                50,
            )
            for update in MILESTONES
        ]
        lines += [
            ("final", "control", gru.final, control.final, intact, 200),
            ("final", "reliance", gru.final, gru.final, blanked, 50),
        ]
        for point, tag, first, second, inference, deals in lines:
            if first is None or second is None:
                continue
            for disclosure in DISCLOSURES:
                found.append(
                    Seating(
                        name=seating_name(seed, disclosure, tag, point),
                        players=(first.player(intact), second.player(inference)),
                        configs=tuple(
                            Conditions(
                                design=DESIGN.name,
                                decklists=disclosure,
                                **pairing.model_dump(),
                            ).model_dump()
                            for pairing in DESIGN.pairings[:deals]
                        ),
                        games=DESIGN.legs,
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
    return [
        EvalJob(
            tournament=Benchmark(
                name=DESIGN.name,
                environment=Environment(
                    name=DESIGN.name,
                    files=canonical_sha256(
                        {
                            "world": managym.world_identity(),
                            "design": DESIGN.record(),
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
    pairing = Pairing(
        role_a=conditions.role_a, role_b=conditions.role_b, deal=conditions.deal
    )
    if conditions.design != DESIGN.name or pairing not in DESIGN.pairings:
        raise ValueError("evaluation conditions differ from the frozen schedule")
    return play_pairing(
        match,
        matchup=AuthoredMatchup(
            pack_key=DESIGN.pack,
            role_a=pairing.role_a,
            role_b=pairing.role_b,
            decklists=conditions.decklists,
        ),
        deal=pairing.deal,
        matchup_mode=DESIGN.matchup_mode,
        game_seconds=DESIGN.game_seconds,
        arena=DESIGN.name,
        backend=BACKEND,
    )


def report() -> Report:
    """Use the existing Elemental matchup panels and the learner-decision clock."""
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
    # Reserve twenty minutes of the job for setup and publishing artifacts.
    wall_seconds = limits.max_hours * 3600 - 1200
    observation = ObservationConfig(
        max_decision_records=1024,
        max_decision_reference_details=65536,
        max_decision_references=512,
        max_decision_members=32768,
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
    else:
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


if __name__ == "__main__":
    main()
