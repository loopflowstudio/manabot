"""Shared launch and scoring for the batch-closure and lambda studies (Task fc6ffb).

`experiments/batch_closure.py` and `experiments/lambda_sweep.py` are separate
experiments so each conclusion stands alone; they differ in arms, seeds and budget
and share the rest here. Every scheduled checkpoint plays the same held-out deals
against the fixed heuristic, all eight deck/seat legs of each deal, through the
replay-checked arena. The protocol is experiments/protocols/closure-lambda-v1.md.
"""

from collections.abc import Iterable
from decimal import Decimal
import os
from pathlib import Path

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
from pydantic import BaseModel
import torch

from manabot.eval.benchmark import evaluation_tables
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.checkpoint import load_checkpoint, local_checkpoint_file
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import PlayerRegistration
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
ARENA = "closure-lambda-v1"
PACK = "ur-lessons-vs-gw-allies"
MATCHUP = AuthoredMatchup(pack_key=PACK, role_a="ur_lessons", role_b="gw_allies")
# Reserved for this protocol: disjoint from monitoring and training seeds.
DEALS = tuple(range(6_510_000_000, 6_510_000_025))
GAME_SECONDS = 120.0
MAX_COMMANDS = 10_000
BOOTSTRAP_SEED = 6510
UPDATES = 300
SCORED_UPDATES = (0, 20, 40, 75, 150, 300)
SETUP_SECONDS = 900
FINALIZATION_SECONDS = 300


class Conditions(BaseModel):
    deal: int


def _source() -> list[Path]:
    return [*sorted((ROOT / "manabot").rglob("*.py")), Path(__file__)]


def _environment() -> Environment:
    return Environment(
        name=ARENA,
        files=canonical_sha256(
            {"world": managym.world_identity(), "matchup": MATCHUP.model_dump()}
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
                milestones=tuple(u for u in SCORED_UPDATES if 0 < u < updates),
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


def scheduled_matchups(run: TrainingRun) -> list[Matchup]:
    """Each checkpoint at a scheduled update, or the run's end, against the heuristic."""
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source)
    conditions = tuple(Conditions(deal=deal).model_dump() for deal in DEALS)
    wanted = set(SCORED_UPDATES)
    if run.final is not None:
        wanted.add(int(run.final.clocks["updates"]))
    return [
        Matchup(
            name=f"{run.name} u{int(checkpoint.clocks['updates'])} vs heuristic",
            players=(checkpoint.model(inference), heuristic),
            conditions=conditions,
            games=8,
            tags=("heuristic",),
        )
        for checkpoint in run.checkpoints
        if int(checkpoint.clocks["updates"]) in wanted
    ]


def evaluation_jobs(state: ExperimentState, name: str) -> list[Job]:
    """Play what is available; Etude skips matches it already holds."""
    matchups = tuple(
        matchup for run in state.runs for matchup in scheduled_matchups(run)
    )
    if not matchups:
        return []
    benchmark = Benchmark(name=name, environment=_environment(), matchups=matchups)
    return [evaluation(benchmark, play=play_deal, workers=8, max_hours=3.0)]


def _registration(
    name: str, player: Model | Code
) -> tuple[PlayerRegistration, Path | None]:
    setup = MATCHUP.match()
    fingerprints = runtime_fingerprints(match_config=setup)
    common = dict(
        player_id=name,
        display_name=player.name,
        role="challenger" if name == "candidate" else "anchor",
        world=managym.WORLD_VERSION,
        content_suite=MATCHUP.content_suite,
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
        artifact_id=f"{ARENA}/{digest}",
    ), path


def play_deal(match: Match) -> Tables:
    """One held-out deal, all eight legs; a failed deal fails its Etude match."""
    torch.set_num_threads(1)
    conditions = Conditions.model_validate(match.conditions)
    if conditions.deal not in DEALS or match.games != 8:
        raise ValueError("evaluation conditions differ from the frozen schedule")
    candidate, path = _registration("candidate", match.players[0])
    opponent, _ = _registration("opponent", match.players[1])
    assert path is not None
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=MATCHUP.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=ARENA,
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
            checkpoint_paths={"candidate": str(path)},
            deal_seeds=(conditions.deal,),
            matchup_mode="all",
            authored_matchup=MATCHUP,
            game_seconds=GAME_SECONDS,
            max_commands=MAX_COMMANDS,
            bootstrap_seed=BOOTSTRAP_SEED,
            purpose=ARENA,
        ),
        out_dir=Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key,
    )
    if result.error:
        raise ValueError(result.error)
    return evaluation_tables(result.rows)


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
