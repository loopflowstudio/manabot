"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report (`opponent_memory.md`) owns the design, evidence and open decisions.
`training_jobs` declares one model size at a time: three trunks, or six EMA branches with `trunks`;
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
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Seating,
)
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
from manabot.sim.heuristic import heuristic_source_paths
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
MILESTONES = (40, 80, 120)
# Paired deals per final checkpoint against the heuristic, each disclosure.
HEURISTIC_DEALS = 50


@dataclass(frozen=True)
class PilotSize:
    hidden_dim: int
    layers: int
    prefix: str
    trunk: PilotStage
    control: PilotStage
    gru: PilotStage


def _branch(hours: float, dollars: str) -> PilotStage:
    return PilotStage(8_800_000, 240, hours, Decimal(dollars))


SIZES = {
    "small": PilotSize(64, 2, "", TRUNK, _branch(4, "2.55"), _branch(5, "3.15")),
    "wide": PilotSize(
        128,
        3,
        "wide-",
        PilotStage(3_300_000, 90, 7.0, Decimal("4.35")),
        _branch(8, "4.95"),
        _branch(10, "6.15"),
    ),
}
# Trunk launch label. "trunk" names the 2026-11-17 runs that overflowed the
# 16,384-detail layout; their failed records stay as evidence.
TRUNK_NAME = "trunk-2"
# Seed 39103 lost its only trunk-2 attempt to a provider capacity refusal
# before any work; its trunk is the same recipe under a fresh run identity.
TRUNK_NAMES = {39101: TRUNK_NAME, 39102: TRUNK_NAME, 39103: "trunk-4"}
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem


def training_jobs(
    revision: str, trunks: str = "", size: str = "small", seeds: str = ""
) -> list[RunRequest]:
    """Declare trunks, or branch both arms from every completed published trunk.

    Each seed's two arms share one producer receipt and EMA export. Their
    optimizer and learner-decision schedule start fresh at the branch point.
    `seeds` (comma-separated) branches only those seed labels, so finished
    trunks need not wait for one still looking for a machine.
    """
    chosen = tuple(int(seed) for seed in seeds.split(",")) if seeds else SEEDS
    if not set(chosen) <= set(SEEDS):
        raise ValueError(f"unknown seeds {seeds!r}; expected a subset of {SEEDS}")
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected small or wide")
    model = SIZES[size]
    names = TRUNK_NAMES if size == "small" else dict.fromkeys(SEEDS, "wide-trunk")
    if not trunks:
        return [
            request
            for seed, name in names.items()
            if seed in chosen
            for request in _job(revision, model, model.trunk, name).runs(seed)
        ]
    storage = storage_from_uri(trunks)
    state = observe(storage, EXPERIMENT)
    parents: dict[int, PretrainedInitialization] = {}
    for seed in chosen:
        runs = _runs(state, names[seed], seed)
        if len(runs) != 1:
            raise ValueError(
                f"seed {seed}: expected one decision-memory-{names[seed]} run in {trunks}, "
                f"found {len(runs)}; launch and finish the trunks first"
            )
        if runs[0].state != "finished":
            raise ValueError(f"seed {seed}: trunk is unfinished ({runs[0].state})")
        parents[seed] = _published_trunk(storage, runs[0].name, seed)
    return [
        request
        for seed, trunk in parents.items()
        for arm, limits in (("control", model.control), ("gru-k1", model.gru))
        for request in _job(revision, model, limits, arm, trunk).runs(seed)
    ]


def _job(
    revision: str,
    model: PilotSize,
    limits: PilotStage,
    name: str,
    trunk: PretrainedInitialization | None = None,
) -> Job:
    """One capped Etude job: a trunk, or an arm started from a trunk's EMA."""
    recipe = pilot_recipe(limits, model)
    recipe.id = f"decision-memory-{model.prefix if trunk is not None else ''}{name}"
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


def seating_name(
    size: str, seed: int, disclosure: str, tag: str, point: int | str
) -> str:
    return f"{size}-seed-{seed}-{disclosure}-{tag}-{point}"


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Equal-update interiors and each completed branch's own final checkpoint."""
    source = [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]
    intact = Inference(source=source)
    blanked = Inference(source=source, config={"blank_decision_memory": True})
    heuristic = Code(name="heuristic", source=heuristic_source_paths())
    found: list[Seating] = []
    for size, model in SIZES.items():
        for seed in SEEDS:
            branches: list[TrainingRunSummary] = []
            for arm in ARMS:
                runs = _runs(state, model.prefix + arm, seed)
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
            # Each arm's final checkpoint against the fixed native heuristic.
            for arm, branch in zip(("control", "gru"), branches, strict=True):
                if branch.final is None:
                    continue
                for disclosure in DISCLOSURES:
                    found.append(
                        Seating(
                            name=seating_name(
                                size, seed, disclosure, f"{arm}-heuristic", "final"
                            ),
                            players=(branch.final.player(intact), heuristic),
                            configs=tuple(
                                Conditions(
                                    design=DESIGN.name,
                                    decklists=disclosure,
                                    **pairing.model_dump(),
                                ).model_dump()
                                for pairing in DESIGN.pairings[:HEURISTIC_DEALS]
                            ),
                            games=DESIGN.legs,
                            tags=("heuristic", f"{size}-heuristic"),
                            seeds="one",
                        )
                    )
            for point, tag, first, second, inference, deals in lines:
                if first is None or second is None:
                    continue
                for disclosure in DISCLOSURES:
                    found.append(
                        Seating(
                            name=seating_name(size, seed, disclosure, tag, point),
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
                            tags=(tag, f"{size}-{tag}"),
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
            # One single-threaded game per vCPU of a 32-vCPU CPU machine; more
            # workers than cores stretched games past the 600 s wall budget.
            workers=30,
            max_hours=3.0,
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
            world=managym.WORLD_SETUP,
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
        evals=tuple(
            panel
            for size in SIZES
            for panel in (
                *(
                    Curve(
                        title=f"Head-to-head against control — {size}, seed {seed}",
                        matches=tuple(
                            Matches(
                                seating=seating_name(
                                    size, seed, disclosure, "control", point
                                )
                            )
                            for disclosure in DISCLOSURES
                            for point in (*MILESTONES, "final")
                        ),
                        split="decklists",
                    )
                    for seed in SEEDS
                ),
                Curve(
                    title=f"Intact memory against blanked memory — {size}",
                    matches=tuple(
                        Matches(
                            seating=seating_name(
                                size, seed, disclosure, "reliance", "final"
                            )
                        )
                        for seed in SEEDS
                        for disclosure in DISCLOSURES
                    ),
                    split="decklists",
                ),
                Ranking(
                    title=f"Final checkpoints against the heuristic — {size}",
                    by="decklists",
                    matches=Matches(tag=f"{size}-heuristic"),
                ),
                Matrix(
                    title=f"Descriptive arm versus control matchups — {size}",
                    rows="opponent_deck",
                    columns="deck",
                    matches=Matches(tag=f"{size}-control"),
                ),
                Ranking(
                    title=f"On the play or the draw — {size}",
                    by="on_play",
                    matches=Matches(tag=f"{size}-control"),
                ),
            )
        ),
        perf=PERF,
        ml=ML,
    )


def pilot_recipe(limits: PilotStage, model: PilotSize) -> TrainingRegime:
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
            hidden_dim=model.hidden_dim,
            attention_layers=model.layers,
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
