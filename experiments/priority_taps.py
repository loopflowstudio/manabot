"""Do fixed-source taps at priority change TLA self-play at either frontier size?

Each size trains with taps off and on. Etude plays the four policies against
the first-64 heuristic and compares on against off at matched updates. Every
point uses the same 16 trophy pairs, crossing seats and decks within each deal.
Run locally with `uv run etude smoke experiments/priority_taps.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from functools import cache
import os
from pathlib import Path
import random
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Literal

from etude.eval.match import environment
from etude.eval.records import (
    Benchmark,
    CheckpointPlayer,
    Code,
    Match,
    Seating,
    checkpoint_player,
)
from etude.eval.result import Row, Tables
from etude.jobs import EvalJob, ExperimentState, TrainingJob
from etude.launch import client
from etude.spec import Input
from etude.report import Curve, Matches, Report, Score, Series
from etude.train.tracking import track

from experiments import priority_taps
from experiments.tla_corpus import Corpus
from experiments.tla_frontier import HEURISTIC, SIZES, SizeLabel, learning, model
from manabot.data import load_manifest
from manabot.eval.admission import register_anchor
from manabot.eval.agents import AGENTS, game_players, registration
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.setup import machine_setup
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import ETUDE_COMPUTE_CLASS
from manabot.training.decks import UniformDecks
from manabot.training.panels import ML
from manabot.training.state.game_errors import GameErrorPolicy
import managym

if TYPE_CHECKING:
    from manabot.training.loops.self_play import SelfPlay

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments/data/mtg213/corpus-v1/corpus.json"
POPULATION = "a5062cc5130a6fdbc4531f4269c610ee18f922fe81c2639b01c8bf8d1488c5ed"
MILESTONES = {0, 25, 50, 100, 150, 200, 300}
OBSERVATION = ObservationConfig(
    policy_input_version=11,
    max_facts=512,
    max_cards_per_player=128,
    max_permanents_per_player=96,
    max_actions=64,
    action_overflow="first-legal",
    max_focus_objects=2,
    max_events=32,
)


def corpus_input() -> Input:
    """Etude delivers the manifest object to rentals and verifies the local smoke copy."""
    manifest = load_manifest()
    path = CORPUS.relative_to(ROOT).as_posix()
    entry = manifest.files[path]
    return Input(
        name=CORPUS.name,
        key=f"sha256/{entry.sha256}",
        sha256=entry.sha256,
        size_bytes=entry.bytes,
        uri=CORPUS.as_uri(),
    )


@cache
def corpus() -> UniformDecks:
    """Re-admit the retained text once per process, preserving event identity and order."""
    inputs = os.environ.get("ETUDE_INPUTS_DIR")
    path = Path(inputs) / CORPUS.name if inputs else CORPUS
    trophies = Corpus.model_validate_json(path.read_bytes()).trophies
    admitted = managym.admit_decklists(
        [
            (t.source_id, managym.Decklist(t.decklist.source, t.decklist.filename))
            for t in trophies
        ]
    )
    source = UniformDecks(
        decks=tuple(a.deck for a in admitted),
        entry_ids=tuple(t.source_id for t in trophies),
        sampling="uniform-independent-v1",
    )
    if source.digest != POPULATION:
        raise ValueError(
            "re-admitted TLA population differs from the 1000 training trophies"
        )
    return source


class Arm(StrictModel):
    size: SizeLabel
    taps: bool
    device: Literal["cpu", "cuda"] = "cuda"
    updates: int = 10_000

    @property
    def name(self) -> str:
        return f"priority-taps-{self.size}-{'on' if self.taps else 'off'}"


def settings(arm: Arm) -> SelfPlay:
    from manabot.training.loops.self_play import SelfPlay

    local = arm.device == "cpu"
    return SelfPlay(
        deck_source=corpus(),
        observation=OBSERVATION.model_copy(update={"priority_mana_taps": True})
        if arm.taps
        else OBSERVATION,
        model=model(arm.size),
        learning=learning(arm.size),
        behavior="current-self",
        streams=2 if local else 256,
        transitions=1 if local else 64,
        workers=1 if local else 4,
        learning_seats="both",
        batch_closure="complete-games",
        game_errors=GameErrorPolicy(
            threshold=0.25,
            window=100,
            warmup=100,
            max_game_actions=40000,
            recover_capacity=True,
        ),
        device=arm.device,
        precision="float32" if local else "bf16",
        bucketed_shapes=True,
        checkpoint_every=25,
    )


def train(config: Arm) -> None:
    from manabot.training.loops.self_play import CLOCKS, self_play

    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings(config), run, updates=config.updates)


def training_jobs(revision: str | None = None) -> list[TrainingJob]:
    arms = [Arm(size=size, taps=taps) for size in SIZES for taps in (False, True)]
    given = corpus_input()
    return [
        TrainingJob(
            name=arm.name,
            train=train,
            config=arm,
            smoke=arm.model_copy(update={"device": "cpu", "updates": 1}),
            seeds=(21501,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            inputs=(given,),
            extras=("artifacts",),
            max_hours=3,
            max_usd=Decimal("1.95"),
            max_attempts=1,
        )
        for arm in arms
    ]


class Deals(StrictModel):
    """Each Etude match index selects one pair and its four-leg deal."""

    population: str = POPULATION
    pairs: tuple[tuple[int, int], ...]
    first_seed: int = 385000


def deals() -> Deals:
    rng = random.Random(385)
    return Deals(
        pairs=tuple((rng.randrange(1000), rng.randrange(1000)) for _ in range(16))
    )


def eval_jobs(state: ExperimentState) -> list[EvalJob]:
    """One pass once all four arms stop. The last shared export is the paired endpoint."""
    checkpoints: dict[str, dict[int, CheckpointPlayer]] = {}
    for size in SIZES:
        for taps in (False, True):
            name = Arm(size=size, taps=taps).name
            runs = [run for run in state.runs if run.launch == name]
            if len(runs) > 1:
                raise ValueError(f"multiple training runs for {name}")
            if not runs:
                return []
            run = runs[0]
            if not run.ended:
                # A lost rental cannot write its own end into the checkpoint store.
                if run.state not in {"running", "stopped"}:
                    return []
                status = client.status(run.name)
                if not status.done or status.outcome == "succeeded":
                    return []
            checkpoints[name] = {
                int(c.at["updates"]): checkpoint_player(c.ref) for c in run.checkpoints
            }
    conditions = deals()
    heuristic = Code(
        name="heuristic", source=heuristic_source_paths(), config=HEURISTIC
    )
    seatings: list[Seating] = []

    def compare(
        name: str,
        tag: str,
        candidate: CheckpointPlayer,
        opponent: CheckpointPlayer | Code,
    ) -> None:
        seatings.append(
            Seating(
                name=name,
                tags=(tag,),
                players=(candidate, opponent),
                config=conditions,
                games=len(conditions.pairs),
            )
        )

    for size in SIZES:
        off, on = (checkpoints[Arm(size=size, taps=t).name] for t in (False, True))
        for label, saved in (("off", off), ("on", on)):
            if saved:
                for u in sorted(saved.keys() & MILESTONES | {max(saved)}):
                    compare(
                        f"{size}-{label} u{u} vs heuristic",
                        "heuristic",
                        saved[u],
                        heuristic,
                    )
        shared = off.keys() & on.keys()
        if shared:
            for u in sorted(shared & MILESTONES | {max(shared)}):
                compare(f"{size} on vs off u{u}", "on-vs-off", on[u], off[u])
    if not seatings:
        return []
    return [
        EvalJob(
            play=priority_taps.play,
            agents=AGENTS,
            tournament=Benchmark(
                name="priority-taps-paired",
                environment=environment(
                    name="priority-taps",
                    files=[Path(__file__)],
                    config=conditions.model_dump(mode="json"),
                ),
                seatings=tuple(seatings),
            ),
            setup=machine_setup(" --extra artifacts --extra evaluation"),
            inputs=(corpus_input(),),
            extras=("artifacts", "evaluation"),
            max_hours=1,
            max_usd=Decimal("1.50"),
            max_attempts=1,
        )
    ]


def play(match: Match[Deals], agents: Sequence[object]) -> Tables:
    conditions = Deals.model_validate(match.config)
    if match.index is None or conditions != deals():
        raise ValueError("expected an indexed match on the fixed trophy pairs")
    source = corpus()
    assert source.entry_ids is not None
    first, second = conditions.pairs[match.index]
    deal = conditions.first_seed + match.index
    matchup = DeckMatchup(
        population_sha256=POPULATION,
        entries=(source.entry_ids[first], source.entry_ids[second]),
        lists=(source.decks[first], source.decks[second]),
    )
    a, b = match.players
    if not isinstance(a, CheckpointPlayer):
        raise ValueError("the candidate must be a checkpoint")
    candidate = registration(
        a,
        name="candidate",
        role="challenger",
        match=matchup.match(),
        content_suite=matchup.content_suite,
    )
    paths = {"candidate": str(a.checkpoint.path)}
    if isinstance(b, CheckpointPlayer):
        opponent = registration(
            b,
            name="opponent",
            role="anchor",
            match=matchup.match(),
            content_suite=matchup.content_suite,
        )
        paths["opponent"] = str(b.checkpoint.path)
    else:
        opponent = register_anchor(
            candidate, "heuristic", player_id="opponent"
        ).model_copy(update={"player_spec": dict(HEURISTIC)})
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="priority-taps-paired",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=opponent.identity_sha256,
        evaluation_compute_envelope_id=ETUDE_COMPUTE_CLASS,
    )
    root = (
        Path(os.environ.get("ETUDE_OUTPUT_DIR") or ROOT / ".runs/priority-taps")
        / "arena"
        / match.key
    )
    root.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        EvaluationRequest(
            key=key,
            candidate=candidate,
            opponent=opponent,
            checkpoint_paths=paths,
            deal_seeds=(deal,),
            authored_matchup=matchup,
            game_seconds=600,
            purpose="priority-taps-tla",
        ),
        out_dir=Path(mkdtemp(dir=root)) / "evaluation",
        agents=dict(
            zip(
                ("candidate", "opponent"),
                game_players(match.players, agents),
                strict=True,
            )
        ),
    )
    rows: list[Row] = []
    by_leg = {r.leg: r for r in result.rows}
    complete = deal in result.complete_deals()
    for leg in range(4):
        row = by_leg.get(leg)
        score = row.score_a if row and complete else None
        for seat in (0, 1):
            game_seat = (leg + seat) % 2
            rows.append(
                {
                    "seat": seat,
                    "game": leg,
                    "game_seat": game_seat,
                    "deck": matchup.decks(leg)[game_seat],
                    "deal_seed": deal,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": bool(row and row.valid),
                    "complete_deal": complete,
                    "failure": row.failure if row else result.error or "missing leg",
                }
            )
    return {"games": rows}


def report() -> Report:
    return Report(
        clock="updates",
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        evals=(
            Curve(
                title="Score against the heuristic", matches=Matches(tag="heuristic")
            ),
            Curve(
                title="Priority taps on against off", matches=Matches(tag="on-vs-off")
            ),
        ),
        perf=tuple(
            Series(title=title, keys=(key,), unit=unit)
            for title, key, unit in (
                ("Decisions per completed game", "decisions_per_game", "rows/game"),
                ("Forced decisions skipped", "skipped_share", "fraction"),
                (
                    "Priority taps per surfaced decision",
                    "priority_tap_rate",
                    "fraction",
                ),
                ("Training rows per second", "rows_per_second", "rows/s"),
                ("Updates per training hour", "updates_per_hour", "updates/h"),
                ("Games per training hour", "games_per_hour", "games/h"),
            )
        ),
        ml=ML,
    )
