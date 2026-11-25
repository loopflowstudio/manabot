"""Exploratory whole-game opponent-memory pilot (Task 3dca91).

The report (`opponent_memory.md`) owns the design, evidence and open decisions.
`training_jobs(state)` declares one model size: its three trunks, then both
arms of every seed whose trunk has finished, each started from that trunk's
published EMA policy. Each `PilotStage` owns its stage's horizon and caps.
Runs recorded in the report keep their original source commits and receipts.
`census` and `probe` are the USD0 commands. Nothing runs on import.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
import random
from typing import Literal

from etude.checkpoints import CheckpointRecord
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Match,
    Player,
    Seating,
    checkpoint_player,
)
from etude.eval.result import Result
from etude.jobs import EvalJob, ExperimentState, TrainingJob, TrainingRunSummary
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from etude.train.tracking import track
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import DECKS, PACK
from manabot.eval.agents import AGENTS
from manabot.eval.pairing import play_pairing
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.run_settings import RunSettings
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup
from manabot.training.decks import UniformDecks
from manabot.training.learning import DampedPolicyLearning, LearnerDecisionSchedule
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play
from manabot.training.panels import ML, PERF
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
# Paired deals per final checkpoint against the heuristic, each disclosure.
HEURISTIC_DEALS = 50
HEURISTIC_CURVE_DEALS = 25


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
type Size = Literal["small", "wide"]
type Stage = Literal["trunk", "control", "gru-k1"]
# Checkpoints fall on this update grid; the report reads the first three.
CHECKPOINT_EVERY = 40
MILESTONES = (40, 80, 120)


# ---------------------------------------------------------------------------
# Training


class PilotConfig(RunSettings):
    """One pilot run: a trunk, or an arm started from its seed's trunk."""

    size: Size = "small"
    stage: Stage = "trunk"
    device: Literal["cpu", "cuda"] = "cuda"
    streams: int = 256
    # Smoke runs shrink the horizon; a launched run keeps its stage's.
    decisions: int | None = None


def limits(config: PilotConfig) -> PilotStage:
    model = SIZES[config.size]
    return {"trunk": model.trunk, "control": model.control, "gru-k1": model.gru}[
        config.stage
    ]


def settings(config: PilotConfig, start: Path | None = None) -> SelfPlay:
    """The shared control recipe; the GRU arm alone reads the decision log."""
    stage, model = limits(config), SIZES[config.size]
    decisions = config.decisions or stage.decisions
    return SelfPlay(
        model=ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=model.hidden_dim,
            attention_layers=model.layers,
            num_attention_heads=4,
            value_aggregation="masked_mean",
            policy_film=True,
            decision_memory=ARMS.get(config.stage, "none"),
            # Twice the default gate slope, so a zero-initialized gate opens sooner.
            **({"decision_memory_gate_scale": 2.0} if config.stage == "gru-k1" else {}),
        ),
        # Every stage captures the log, so a control's bytes can seed the GRU arm.
        observation=ObservationConfig(
            max_decision_records=1024,
            max_decision_reference_details=65536,
            max_decision_references=512,
            max_decision_members=32768,
        ),
        # A sampled deck source owns mixed disclosure and independent mirrors.
        deck_source=UniformDecks(
            decks=tuple(
                managym.Deck(player.decklist, player.sideboard)
                for key in managym.authored_deck_keys("elemental-world-v2")
                for player in [managym.authored_deck_setup("elemental-world-v2", key)]
            ),
            sampling="uniform-independent-v1",
            decklists="mixed",
        ),
        learning=DampedPolicyLearning(
            gradient="damped_policy",
            decision_schedule=LearnerDecisionSchedule(transitions=decisions),
        ),
        learner_transitions=decisions,
        streams=config.streams,
        transitions=64,
        # Capacity overflow fails the run: no truncation and no abort-and-continue.
        game_errors=None,
        collection_rows=128,
        bootstrap_rows=128,
        learner_rows=128,
        device=config.device,
        workers=4 if config.device == "cuda" else 1,
        # The packed CUDA layout; a CPU smoke keeps the padded one.
        bucketed_shapes=config.device == "cuda",
        live_tokens=config.device == "cuda",
        checkpoint_every=CHECKPOINT_EVERY,
        pretrained=start,
        run_settings=config.run_settings,
    )


def train(config: PilotConfig) -> None:
    with track(clocks=CLOCKS, config=config) as run:
        start = None if config.stage == "trunk" else Path(run.input("start").path)
        # `updates` is a safety ceiling; the learner-decision target ends the run.
        self_play(settings(config, start), run, updates=limits(config).updates)


def stored_runs(
    state: ExperimentState, name: str, seed: int
) -> list[TrainingRunSummary]:
    """A stage's runs for one seed; an arm's job is named for its seed's trunk."""
    names = {f"decision-memory-{name}", f"decision-memory-{name}-{seed}"}
    return [run for run in state.runs if run.launch in names and run.seed == seed]


def training_jobs(
    state: ExperimentState, revision: str, size: str = "small"
) -> list[TrainingJob]:
    """Every seed's trunk, and both arms of each seed whose trunk has finished.

    A seed's two arms share one published EMA trunk. Their optimizer and
    learner-decision schedule start fresh at the branch point.
    """
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected small or wide")
    model = SIZES[size]
    names = TRUNK_NAMES if size == "small" else dict.fromkeys(SEEDS, "wide-trunk")

    def job(
        name: str, stage: Stage, seeds: tuple[int, ...], start: CheckpointRecord | None
    ) -> TrainingJob:
        config = PilotConfig(size=size, stage=stage)
        cap = limits(config)
        return TrainingJob(
            name=f"decision-memory-{name}",
            train=train,
            config=config,
            smoke=PilotConfig(
                size=size, stage=stage, device="cpu", streams=2, decisions=64
            ),
            start=start,
            seeds=seeds,
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=cap.max_hours,
            max_usd=cap.max_usd,
            # A provider capacity refusal consumes an attempt at no cost.
            max_attempts=3,
        )

    jobs = [
        job(name, "trunk", tuple(s for s, n in names.items() if n == name), None)
        for name in dict.fromkeys(names.values())
    ]
    for seed, name in names.items():
        finished = [run.final for run in stored_runs(state, name, seed) if run.final]
        if len(finished) > 1:
            raise ValueError(f"seed {seed}: {len(finished)} finished {name} trunks")
        for trunk in finished:
            jobs += [
                # Each arm job holds one start checkpoint, so it names its seed.
                job(f"{model.prefix}{arm}-{seed}", arm, (seed,), trunk)  # type: ignore[arg-type]
                for arm in ARMS
            ]
    return jobs


# ---------------------------------------------------------------------------
# Evaluation


class Pairing(BaseModel):
    """One scored item: the two authored decks and their deal."""

    model_config = ConfigDict(extra="forbid")
    role_a: str
    role_b: str
    deal: int


@dataclass(frozen=True)
class Design:
    """Held-out pairings, fixed before scoring and shared by every arm."""

    # The arena version and the benchmark environment's name.
    name: str
    pack: str
    pairings: tuple[Pairing, ...]
    game_seconds: float
    # A pairing's four cross deck/seat legs.
    legs: int = 4


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


def opponent(checkpoint: CheckpointRecord, blank: bool) -> Player:
    """The opposing seat; a blanked seat plays the same bytes with an empty log."""
    player = checkpoint_player(checkpoint.ref)
    if not blank:
        return player
    return player.model_copy(update={"config": {"blank_decision_memory": True}})


def evaluation_matchups(state: ExperimentState) -> tuple[Seating, ...]:
    """Equal-update interiors and each completed branch's own final checkpoint."""
    heuristic = Code(name="heuristic", source=heuristic_source_paths())
    found: list[Seating] = []
    for size, model in SIZES.items():
        for seed in SEEDS:
            branches: list[TrainingRunSummary] = []
            for arm in ARMS:
                found_runs = stored_runs(state, model.prefix + arm, seed)
                if len(found_runs) > 1:
                    raise ValueError(f"seed {seed}: multiple {arm} branches")
                branches += found_runs
            if len(branches) != len(ARMS):
                continue
            control, gru = (
                {int(each.at["updates"]): each for each in branch.checkpoints}
                for branch in branches
            )
            control_final, gru_final = (branch.final for branch in branches)
            # (point, tag, checkpoint, opposing checkpoint, opponent blanked, deals)
            lines = [
                (update, "control", gru.get(update), control.get(update), False, 50)
                for update in MILESTONES
            ]
            lines += [
                ("final", "control", gru_final, control_final, False, 200),
                ("final", "reliance", gru_final, gru_final, True, 50),
            ]
            # Each arm against the fixed native heuristic: its final checkpoint,
            # and a shorter curve over the milestones. Update 0 is the shared
            # trunk export, played once through control.
            for arm, branch, saved in zip(
                ("control", "gru"), branches, (control, gru), strict=True
            ):
                points = [("final", branch.final, HEURISTIC_DEALS)] + [
                    (update, saved.get(update), HEURISTIC_CURVE_DEALS)
                    for update in ((0,) if arm == "control" else ()) + MILESTONES
                ]
                for point, checkpoint, deals in points:
                    if checkpoint is None:
                        continue
                    for disclosure in DISCLOSURES:
                        found.append(
                            Seating(
                                name=seating_name(
                                    size, seed, disclosure, f"{arm}-heuristic", point
                                ),
                                players=(checkpoint_player(checkpoint.ref), heuristic),
                                configs=tuple(
                                    Conditions(
                                        design=DESIGN.name,
                                        decklists=disclosure,
                                        **pairing.model_dump(),
                                    ).model_dump()
                                    for pairing in DESIGN.pairings[:deals]
                                ),
                                games=DESIGN.legs,
                                tags=("heuristic", f"{size}-heuristic"),
                                seeds="one",
                            )
                        )
            for point, tag, first, second, blank, deals in lines:
                if first is None or second is None:
                    continue
                for disclosure in DISCLOSURES:
                    found.append(
                        Seating(
                            name=seating_name(size, seed, disclosure, tag, point),
                            players=(
                                checkpoint_player(first.ref),
                                opponent(second, blank),
                            ),
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
                    # This file and pairing.py when the pilot's games were played.
                    files="b47322cf2add632ad53b5f8fa035d18c154d9759cbaa51bf69ba7026a6f8b255",
                    config={
                        "world": "w17",
                        "pack": DESIGN.pack,
                        "game_seconds": DESIGN.game_seconds,
                        "disclosures": list(DISCLOSURES),
                    },
                ),
                seatings=seatings,
            ),
            play=play,
            agents=AGENTS,
            extras=("evaluation",),
            max_hours=3.0,
            max_usd=Decimal("4.00"),
            setup=machine_setup(" --extra evaluation"),
        )
    ]


def play(match: Match, agents: Sequence[object]) -> Result:
    """Validate pilot conditions and delegate the paired deal to the shared arena."""
    conditions = Conditions.model_validate(match.config)
    pairing = Pairing(
        role_a=conditions.role_a, role_b=conditions.role_b, deal=conditions.deal
    )
    if conditions.design != DESIGN.name or pairing not in DESIGN.pairings:
        raise ValueError("evaluation conditions differ from the frozen schedule")
    return play_pairing(
        match,
        agents,
        matchup=AuthoredMatchup(
            pack_key=DESIGN.pack,
            role_a=pairing.role_a,
            role_b=pairing.role_b,
            decklists=conditions.decklists,
            world=managym.WORLD_SETUP,
        ),
        deal=pairing.deal,
        matchup_mode="cross",
        game_seconds=DESIGN.game_seconds,
        arena=DESIGN.name,
    )


# ---------------------------------------------------------------------------
# Report


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
