"""Shared launch and scoring for the batch-closure and lambda studies (Tasks fc6ffb, 07c490).

`experiments/batch_closure.py` and `experiments/lambda_sweep.py` (Allies/Lessons) and
`experiments/elemental_closure.py` and `experiments/elemental_lambda.py` (the ten-deck
Elemental recipe) are separate experiments so each conclusion stands alone; they differ
in arms, seeds and budget and share the rest here. Every scheduled checkpoint plays the
same held-out pairings against the fixed heuristic through the replay-checked arena.

A `Design` fixes those pairings before scoring, identical for every arm and seed. Each
`Pairing` is a deck pair and a deal; the arena's legs exchange the decks and the seats
within it, so the model plays both decks on the play and on the draw of that deal
(four cross legs, or all eight with the pair's two mirrors). The protocols are
experiments/protocols/closure-lambda-v1.md and closure-lambda-elemental-v1.md.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
import os
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from etude.eval.match import Tables
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
from etude.report import Curve, Matches, Matrix, Ranking, Report, Score
from pydantic import BaseModel, ConfigDict
import torch

from experiments.elemental_grid import DECKS, PACK as ELEMENTAL_PACK
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TOOLCHAIN_ENV, TrainingJob, machine_setup
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
MAX_COMMANDS = 10_000
BOOTSTRAP_SEED = 6510
SCORED_UPDATES = (20, 40, 75, 150)
SETUP_SECONDS = 900
# One single-threaded game per worker on a 32-vCPU rental, leaving room for its host.
EVALUATION_WORKERS = 28
FINALIZATION_SECONDS = 300


# --------------------------------------------------------------------------- designs


class Pairing(BaseModel):
    """One scored item: the model's and the heuristic's authored decks, and their deal."""

    model_config = ConfigDict(extra="forbid")
    role_a: str
    role_b: str
    deal: int


@dataclass(frozen=True)
class Design:
    """A protocol's held-out pairings, fixed before scoring and shared by every arm."""

    # The arena version and the benchmark environment's name.
    name: str
    pack: str
    pairings: tuple[Pairing, ...]
    # "cross": the pairing's four deck/seat legs; "all": those and its two mirrors' four.
    matchup_mode: Literal["cross", "all"]
    game_seconds: float

    @property
    def legs(self) -> int:
        return 8 if self.matchup_mode == "all" else 4

    def matchup(self, pairing: Pairing) -> AuthoredMatchup:
        return AuthoredMatchup(
            pack_key=self.pack, role_a=pairing.role_a, role_b=pairing.role_b
        )

    def record(self) -> dict[str, object]:
        """What the benchmark environment hashes: everything but the arena's name."""
        return {
            "pack": self.pack,
            "pairings": [pairing.model_dump() for pairing in self.pairings],
            "matchup_mode": self.matchup_mode,
            "game_seconds": self.game_seconds,
        }


class Conditions(BaseModel):
    """An Etude match's conditions: a design's name and one of its pairings."""

    model_config = ConfigDict(extra="forbid")
    design: str
    role_a: str
    role_b: str
    deal: int

    @property
    def pairing(self) -> Pairing:
        return Pairing(role_a=self.role_a, role_b=self.role_b, deal=self.deal)


# Allies/Lessons (fc6ffb): the one selected pair over 25 deals, mirrors included.
PACK = "ur-lessons-vs-gw-allies"
# Reserved for that protocol: disjoint from monitoring and training seeds.
DEALS = tuple(range(6_510_000_000, 6_510_000_025))
ALLIES_LESSONS = Design(
    name="closure-lambda-v1",
    pack=PACK,
    pairings=tuple(
        Pairing(role_a="ur_lessons", role_b="gw_allies", deal=deal) for deal in DEALS
    ),
    matchup_mode="all",
    game_seconds=120.0,
)

# Elemental (07c490): deck pairs drawn at random rather than the whole 10×10 grid
# (Jack Heart, 2026-11-05), each with its own deal, the model scored against the
# heuristic from both decks and both seats of each. Deals are reserved here, disjoint
# from the grid's, the pairing graph's and every other protocol's.
ELEMENTAL_PAIRING_SEED = 7_070_001
ELEMENTAL_DEAL_BASE = 7_070_000_000
ELEMENTAL_PAIRING_COUNT = 50


def elemental_pairings() -> tuple[Pairing, ...]:
    """Fifty pairs drawn with replacement from the 45 distinct Elemental deck pairs.

    Mirrors are excluded: exchanging the decks of a mirror changes nothing, so a
    mirror would score the model on two legs where a pair scores it on four.
    """
    rng = random.Random(ELEMENTAL_PAIRING_SEED)
    distinct = list(combinations(DECKS, 2))
    return tuple(
        Pairing(role_a=a, role_b=b, deal=ELEMENTAL_DEAL_BASE + index)
        for index, (a, b) in enumerate(
            rng.choice(distinct) for _ in range(ELEMENTAL_PAIRING_COUNT)
        )
    )


ELEMENTAL_PAIRS = Design(
    name="closure-lambda-elemental-v1",
    pack=ELEMENTAL_PACK,
    pairings=elemental_pairings(),
    matchup_mode="cross",
    game_seconds=600.0,
)
DESIGNS = {design.name: design for design in (ALLIES_LESSONS, ELEMENTAL_PAIRS)}


# --------------------------------------------------------------------------- launch


def _source() -> list[Path]:
    return [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]


def _environment(design: Design) -> Environment:
    return Environment(
        name=design.name,
        files=canonical_sha256(
            {"world": managym.world_identity(), "design": design.record()}
        ),
        config={"purpose": "held-out-paired-deals"},
    )


def training_runs(
    recipes: Iterable[TrainingRegime],
    *,
    updates: int,
    revision: str,
    experiment: str,
    seeds: tuple[int, ...],
    hours: float,
    usd_per_run: float,
) -> list[Run]:
    """Each arm's job, once per seed; a watchdog stop keeps its exports."""
    wall_seconds = hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    found: list[Run] = []
    for recipe in recipes:
        stage = recipe.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        recipe.wall_seconds = wall_seconds
        stage.execution.wall_seconds = wall_seconds
        found += (
            TrainingJob(
                regime=TrainingRegime.model_validate(recipe.model_dump(mode="json")),
                experiment=experiment,
                milestones=tuple(u for u in SCORED_UPDATES if u < updates),
                finalization_seconds=FINALIZATION_SECONDS,
            )
            .job(
                repo=ROOT,
                revision=revision,
                max_hours=hours,
                max_usd=Decimal(str(usd_per_run)),
            )
            .runs(*seeds)
        )
    return found


def scheduled_matchups(run: TrainingRun, design: Design) -> list[Matchup]:
    """Each checkpoint at a scheduled update, or the run's end, against the heuristic."""
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source)
    conditions = tuple(
        Conditions(design=design.name, **pairing.model_dump()).model_dump()
        for pairing in design.pairings
    )
    wanted = set(SCORED_UPDATES)
    if run.final is not None:
        wanted.add(int(run.final.clocks["updates"]))
    return [
        Matchup(
            name=f"{run.name} u{int(checkpoint.clocks['updates'])} vs heuristic",
            players=(checkpoint.model(inference), heuristic),
            conditions=conditions,
            games=design.legs,
            tags=("heuristic",),
        )
        for checkpoint in run.checkpoints
        if int(checkpoint.clocks["updates"]) in wanted
    ]


def evaluation_jobs(
    state: ExperimentState, name: str, design: Design, *, hours: float, usd: float
) -> list[Job]:
    """Play what is available; Etude skips matches it already holds.

    A job stopped at its limit keeps its finished matches, and a later pass plays
    only the rest, so `usd` bounds each pass inside the protocol's evaluation line.
    """
    matchups = tuple(
        matchup for run in state.runs for matchup in scheduled_matchups(run, design)
    )
    if not matchups:
        return []
    benchmark = Benchmark(
        name=name, environment=_environment(design), matchups=matchups
    )
    job = evaluation(
        benchmark,
        play=play_deal,
        workers=EVALUATION_WORKERS,
        max_hours=hours,
        max_usd=Decimal(str(usd)),
        setup=machine_setup(" --extra evaluation"),
    )
    # Etude's default command omits the extra holding its result writer (duckdb).
    command = job.command.replace(
        "uv run python",
        f"{TOOLCHAIN_ENV}\nuv run --locked --extra evaluation python",
        1,
    )
    return [job.model_copy(update={"command": command})]


# --------------------------------------------------------------------------- scoring


def _registration(
    name: str, player: Model | Code, matchup: AuthoredMatchup, arena: str
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
    agent, _ = load_checkpoint(str(path))
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
        artifact_id=f"{arena}/{digest}",
    ), path


def play_deal(match: Match) -> Tables:
    """One held-out pairing, all its legs; a failed pairing fails its Etude match."""
    torch.set_num_threads(1)
    conditions = Conditions.model_validate(match.conditions)
    design = DESIGNS[conditions.design]
    if conditions.pairing not in design.pairings or match.games != design.legs:
        raise ValueError("evaluation conditions differ from the frozen schedule")
    matchup = design.matchup(conditions.pairing)
    candidate, path = _registration("candidate", match.players[0], matchup, design.name)
    opponent, _ = _registration("opponent", match.players[1], matchup, design.name)
    assert path is not None
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=design.name,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    retained = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    # The arena syncs every record, which a rented container disk serves at about
    # .1 s, so a game spends its time limit there. Journal in memory and keep the
    # same files: a pairing that dies is replayed whole, so nothing needs the journal.
    with TemporaryDirectory(dir=_journal_root()) as journal:
        out_dir = Path(journal) / "deal"
        try:
            result = evaluate(
                EvaluationRequest(
                    key=key,
                    candidate=candidate,
                    opponent=opponent,
                    checkpoint_paths={"candidate": str(path)},
                    deal_seeds=(conditions.deal,),
                    matchup_mode=design.matchup_mode,
                    authored_matchup=matchup,
                    game_seconds=design.game_seconds,
                    max_commands=MAX_COMMANDS,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    purpose=design.name,
                ),
                out_dir=out_dir,
            )
        finally:
            if out_dir.is_dir():
                shutil.copytree(out_dir, retained, dirs_exist_ok=True)
    if result.error:
        raise ValueError(result.error)
    return evaluation_tables(result.rows, matchup=matchup, legs=design.legs)


def _journal_root() -> str | None:
    return "/dev/shm" if Path("/dev/shm").is_dir() else None


def learning_report(baseline: str) -> Report:
    """Score at equal learner decisions; the panels hold time and critic detail."""
    heuristic = Matches(tag="heuristic")
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="learner_transitions",
        baseline=baseline,
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            Curve(title="By matchup", split="matchup", matches=heuristic),
            Matrix(
                title="Model deck × heuristic deck",
                rows="opponent_deck",
                columns="deck",
            ),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
