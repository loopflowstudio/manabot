"""Does the shape of combat and mana decisions change how fast the small TLA model learns?

Three arms train the frontier's small model (`experiments/tla_frontier.py`,
width 64, policy input v9, damped self-play) and differ only in native decision
switches, exported in every job command as 0 or 1:

- `baseline`: neither switch;
- `combat-pick`: `MANAGYM_EXPERIMENT_COMBAT_PICK=1`, attackers and blockers
  offered as one repeated pick-then-done decision;
- `mana-taps`: `MANAGYM_EXPERIMENT_SOURCE_TAPS=1`, each useful source offered
  as a tap step beside `PayCost`.

The population is the TLA pool corpus (`manabot.pools`, 1,000 trophy drafts):
each deck is a pool's human build with the rest of its pool as sideboard, which
Learn reads. All 1,000 lists admit on this engine, so training samples the 900
`train` decks and scoring uses the 100 `held_out` decks. A rented machine has
the commit and no artifact store, so `build` derives both populations from the
corpus into tracked files:

    uv run python -m experiments.decision_shapes build

`runs` trains each arm once per seed. `eval` plays each arm's scored
checkpoints against the first-64 heuristic on fifty fixed pairs of held-out
decks, four legs a pair (both deck assignments on the play and on the draw),
in one rented job per arm that carries the arm's switches, so the heuristic
answers the same decision shape the model trained in. `report` draws score
against the heuristic by update, a line per arm.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from decimal import Decimal
from functools import cache
from itertools import combinations
import json
import os
from pathlib import Path
import random
from typing import Literal

from etude.eval.match import Row, Tables
from etude.eval.records import (
    Benchmark,
    Code,
    Environment,
    Inference,
    Match,
    Seating,
)
from etude.experiment import ExperimentState, TrainingRun
from etude.jobs import evaluation
from etude.launch.records import Job, Run
from etude.report import Report, Score
from pydantic import Field
import torch

from experiments import decision_shapes
from experiments.tla_frontier import (
    EVALUATION_ATTEMPTS,
    EVALUATION_HOURS,
    EVALUATION_PASS_USD,
    EVALUATION_WORKERS,
    GAMES_PER_PAIR,
    HEURISTIC,
    REGIME as FRONTIER_REGIME,
    Pair,
    _memory_journal,
    _registration,
    _schedule,
    _stage,
    capped_runs,
    checkpoint_update,
    shortened,
    tagged_runs,
)
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup
from manabot.infra.files import atomic_json
from manabot.infra.identity import SHA256_PATTERN, StrictModel, canonical_sha256
from manabot.pools import Pool, PoolCorpus, PoolSplit
from manabot.sim.matchup import DeckMatchup
from manabot.training.decks import UniformDecks
from manabot.training.panels import ML, PERF
from manabot.training.regime import TrainingRegime
from manabot.training.report import against_fixed, learning_schedule
import managym
from managym import Deck

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
CORPUS = ROOT / "experiments/data/mtg282/pools-v2/corpus.json"
# The frontier's w64 regime with the 900 training decks as its population.
TRAINING = ROOT / "experiments/regimes/decision-shapes-w64.json"
HELD_OUT = ROOT / "experiments/regimes/decision-shapes-held-out.json"

type Arm = Literal["baseline", "combat-pick", "mana-taps"]
ARMS: tuple[Arm, ...] = ("baseline", "combat-pick", "mana-taps")
DISPLAY: dict[Arm, str] = {
    "baseline": "Baseline",
    "combat-pick": "Combat: pick then done",
    "mana-taps": "Mana: per-source taps",
}
# Every switch an arm may turn on; each command exports all of them.
SWITCHES: dict[str, Arm] = {
    "MANAGYM_EXPERIMENT_COMBAT_PICK": "combat-pick",
    "MANAGYM_EXPERIMENT_SOURCE_TAPS": "mana-taps",
}

# Training exports every `EXPORT_EVERY` updates, which bounds what a lost rental
# costs; only the scored milestones and each run's last checkpoint are played.
EXPORT_EVERY = 25
SCORED_UPDATES = (50, 100, 200, 300, 400)
# Machine setup (toolchain, locked sync, native build) and the final exports
# both come out of the job's hours before the regime's wall limit.
SETUP_SECONDS = 1800
FINALIZATION_SECONDS = 600

PAIR_SEED = 26_590_001
PAIR_COUNT = 50
DEAL_BASE = 26_591_000
GAME_SECONDS = 600.0
MAX_COMMANDS = 10_000
BOOTSTRAP_SEED = 26_592_001
ARENA = "decision-shapes-v1"
# Set to the next attempt's number to relaunch a pass that delivered nothing:
# the launcher names a pass for its pending matches and refuses a known name.
PASS_ATTEMPT = "DECISION_SHAPES_PASS_ATTEMPT"


# -- Populations --------------------------------------------------------------


class HeldOutDecks(StrictModel):
    """The scoring population: held-out pool decks, in corpus order."""

    corpus_sha256: str = Field(pattern=SHA256_PATTERN)
    entry_ids: tuple[str, ...]
    decks: tuple[Deck, ...]


def pool_deck(pool: Pool) -> Deck:
    """The human build as the main deck; the pool's other cards as its sideboard.

    A pool holds no basic lands, so the build's basics leave the sideboard alone.
    """
    sideboard = {
        name: left
        for name, count in pool.cards.items()
        if (left := count - pool.human_build.get(name, 0)) > 0
    }
    return Deck(dict(pool.human_build), sideboard)


def playable_decks(
    corpus: PoolCorpus, split: PoolSplit
) -> tuple[tuple[str, ...], tuple[Deck, ...]]:
    """The split's pool ids and decks that native admission accepts, in corpus order."""
    pools = [pool for pool in corpus.pools if pool.split == split]
    decks = [pool_deck(pool) for pool in pools]
    kept = [
        (pool.id, deck)
        for pool, deck, issues in zip(
            pools, decks, managym.check_decks(decks), strict=True
        )
        if not issues
    ]
    return tuple(name for name, _ in kept), tuple(deck for _, deck in kept)


def build(corpus_path: Path = CORPUS) -> tuple[int, int]:
    """Write both tracked populations from the corpus; returns their deck counts."""
    corpus = PoolCorpus.read(corpus_path)
    train_ids, train = playable_decks(corpus, "train")
    held_ids, held = playable_decks(corpus, "held_out")
    regime = json.loads(FRONTIER_REGIME.read_text())
    regime["id"] = "decision-shapes-w64"
    regime["deck_source"] = UniformDecks(
        decks=train, sampling="uniform-independent-v1", entry_ids=train_ids
    ).model_dump(mode="json")
    TrainingRegime.model_validate(regime)
    atomic_json(TRAINING, regime)
    atomic_json(
        HELD_OUT,
        HeldOutDecks(
            corpus_sha256=corpus.digest, entry_ids=held_ids, decks=held
        ).model_dump(mode="json"),
    )
    return len(train), len(held)


@cache
def held_out() -> HeldOutDecks:
    return HeldOutDecks.model_validate_json(HELD_OUT.read_bytes())


@cache
def pairs() -> tuple[Pair, ...]:
    """Fifty distinct pairs of different held-out decks, each with its own deal.

    Indexes are into `HELD_OUT`'s order. Mirrors are excluded: exchanging the
    decks of a mirror changes nothing, so it would score two legs, not four.
    """
    distinct = list(combinations(range(len(held_out().decks)), 2))
    drawn = random.Random(PAIR_SEED).sample(distinct, PAIR_COUNT)
    return tuple(
        Pair(first=first, second=second, deal=DEAL_BASE + index)
        for index, (first, second) in enumerate(drawn)
    )


# -- Arms ---------------------------------------------------------------------


def _regime_id(arm: Arm) -> str:
    return f"decision-shapes-{arm}"


def switches(arm: Arm) -> dict[str, int]:
    return {name: int(owner == arm) for name, owner in SWITCHES.items()}


def in_arm(job: Job, arm: Arm) -> Job:
    """`job` with the arm's native switches exported before its command."""
    exports = "".join(
        f"export {name}={value}\n" for name, value in switches(arm).items()
    )
    return job.model_copy(update={"command": exports + job.command})


def recipe(arm: Arm, updates: int = 500, hours: float = 16.0) -> TrainingRegime:
    """The frontier's small regime on the training decks, for `updates` updates.

    The damping horizon is the update target, as the frontier set its horizon
    to the updates it expected to reach.
    """
    base = TrainingRegime.model_validate_json(TRAINING.read_bytes())
    schedule = _schedule(base).model_copy(update={"steps": updates})
    learning = _stage(base).learning.model_copy(update={"iteration_schedule": schedule})
    return shortened(
        base.model_copy(
            update={
                "world": managym.WORLD_VERSION,
                "stages": [_stage(base).model_copy(update={"learning": learning})],
            }
        ),
        id=_regime_id(arm),
        updates=updates,
        wall_seconds=hours * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS,
    )


def _arms(arms: str) -> tuple[Arm, ...]:
    chosen = tuple(arm for arm in ARMS if arm in arms.split(",")) if arms else ARMS
    unknown = set(arms.split(",")) - set(ARMS) if arms else set()
    if unknown:
        raise ValueError(f"unknown arms: {', '.join(sorted(unknown))}")
    return chosen


def runs(
    revision: str,
    updates: int = 500,
    hours: float = 16.0,
    seeds: str = "26501,26502",
    arms: str = "",
    usd: float = 9.0,
) -> list[Run]:
    """Each chosen arm once per seed: one capped GPU allocation, no retry."""
    result: list[Run] = []
    for arm in _arms(arms):
        regime = recipe(arm, updates, hours)
        for seed in seeds.split(","):
            result += [
                run.model_copy(update={"job": in_arm(run.job, arm)})
                for run in capped_runs(
                    regime,
                    # Short calibration runs stay out of the study's evaluation.
                    experiment=EXPERIMENT
                    if updates >= EXPORT_EVERY
                    else f"{EXPERIMENT}_calibration",
                    revision=revision,
                    seed=int(seed),
                    max_hours=hours,
                    max_usd=Decimal(str(usd)),
                    milestones=tuple(range(EXPORT_EVERY, updates, EXPORT_EVERY)),
                )
            ]
    return result


# -- Evaluation ---------------------------------------------------------------


def _source() -> list[Path]:
    return [
        *sorted((ROOT / "manabot").rglob("*.py")),
        Path(__file__),
        ROOT / "experiments/tla_frontier.py",
        TRAINING,
        HELD_OUT,
    ]


def environment(arm: Arm) -> Environment:
    return Environment(
        name=f"{ARENA}-{arm}",
        files=canonical_sha256(
            {
                "world": managym.world_identity(),
                "held_out": held_out().model_dump(mode="json"),
                "pairs": [pair.model_dump(mode="json") for pair in pairs()],
                "game_seconds": GAME_SECONDS,
                "max_commands": MAX_COMMANDS,
            }
        ),
        config={"purpose": "held-out-pool-decks", "switches": switches(arm)},
    )


def scheduled_matchups(run: TrainingRun) -> list[Seating]:
    """Each scored milestone the run stored, and its last checkpoint, against the heuristic."""
    source = _source()
    inference = Inference(source=source)
    heuristic = Code(name="heuristic", source=source, config=HEURISTIC)
    conditions = tuple(pair.model_dump(mode="json") for pair in pairs())
    wanted = set(SCORED_UPDATES)
    if run.final is not None:
        wanted.add(checkpoint_update(run.final))
    return [
        Seating(
            name=f"{run.name} u{checkpoint_update(checkpoint)} vs heuristic",
            players=(checkpoint.model(inference), heuristic),
            conditions=conditions,
            games=GAMES_PER_PAIR,
            tags=("heuristic",),
        )
        for checkpoint in run.checkpoints
        if checkpoint_update(checkpoint) in wanted
    ]


def _pass_name(name: str) -> str:
    attempt = os.environ.get(PASS_ATTEMPT)
    return f"{name}-attempt-{attempt}" if attempt else name


def evaluation_jobs(arm: Arm, matchups: Iterable[Seating]) -> list[Job]:
    """One rented pass of `play` over the arm's matchups, under its switches."""
    seatings = tuple(matchups)
    if not seatings:
        return []
    job = evaluation(
        Benchmark(
            name=_pass_name(_regime_id(arm)),
            environment=environment(arm),
            seatings=seatings,
        ),
        # Etude discovers this file with runpy; the callback needs an importable
        # module identity even when eval itself came from <run_path>.
        play=decision_shapes.play,
        workers=EVALUATION_WORKERS,
        max_hours=EVALUATION_HOURS,
        max_usd=EVALUATION_PASS_USD,
        # A retry resumes the same run and skips stored matches.
        max_attempts=EVALUATION_ATTEMPTS,
        setup=machine_setup(" --extra evaluation"),
    )
    # Etude's default command omits the extra holding its result writer (duckdb).
    command = job.command.replace(
        "uv run python",
        f"{TOOLCHAIN_ENV}\nuv run --locked --extra evaluation python",
        1,
    )
    return [in_arm(job.model_copy(update={"command": command}), arm)]


def eval(state: ExperimentState) -> list[Job]:
    """Heuristic scores at each arm's scored checkpoints; Etude skips held matches."""
    if state.experiment != EXPERIMENT:
        raise ValueError("decision-shape evaluation requires this experiment's runs")
    return [
        job
        for arm in ARMS
        for job in evaluation_jobs(
            arm,
            (
                matchup
                for run in tagged_runs(state, _regime_id(arm))
                for matchup in scheduled_matchups(run)
            ),
        )
    ]


def deck_matchup(pair: Pair) -> DeckMatchup:
    decks = held_out()
    return DeckMatchup(
        population_sha256=decks.corpus_sha256,
        entries=(decks.entry_ids[pair.first], decks.entry_ids[pair.second]),
        lists=(decks.decks[pair.first], decks.decks[pair.second]),
    )


def play(match: Match) -> Tables:
    """One held-out pair and deal, four policy/deck/seat legs, with failure denominators.

    The process environment decides the decision shape for both players, so the
    job's exported switches are the arm's; rows keep the frontier's columns.
    """
    torch.set_num_threads(1)
    pair = Pair.model_validate(match.conditions)
    if pair not in pairs() or match.games != GAMES_PER_PAIR:
        raise ValueError("evaluation conditions differ from the fixed schedule")
    observation = TrainingRegime.model_validate_json(TRAINING.read_bytes()).observation
    matchup = deck_matchup(pair)
    candidate, candidate_path = _registration(
        match.players[0], "candidate", matchup, observation
    )
    opponent, opponent_path = _registration(
        match.players[1], "opponent", matchup, observation
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
        arena_version=ARENA,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    retained = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    with _memory_journal(retained) as out_dir:
        result = evaluate(
            EvaluationRequest(
                key=key,
                candidate=candidate,
                opponent=opponent,
                checkpoint_paths=paths,
                deal_seeds=(pair.deal,),
                authored_matchup=matchup,
                game_seconds=GAME_SECONDS,
                max_commands=MAX_COMMANDS,
                bootstrap_seed=BOOTSTRAP_SEED,
                purpose="decision-shapes-held-out",
            ),
            out_dir=out_dir,
        )
    by_leg = {row.leg: row for row in result.rows}
    rows: list[Row] = []
    for leg in range(GAMES_PER_PAIR):
        row = by_leg.get(leg)
        valid = row is not None and row.valid
        score = row.score_a if valid else None
        actions = row.policy_action_counts if row else None
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
                    "corpus_sha256": matchup.population_sha256,
                    "score": score if seat == 0 or score is None else 1 - score,
                    "valid": valid,
                    "attempted": row is not None,
                    "terminated": row.terminated if row else False,
                    "truncated": row.truncated if row else False,
                    "failure": row.failure if row else result.error or "missing game",
                    "seconds": row.game_seconds if row else None,
                    "policy_decisions": actions.decisions if actions else None,
                    "policy_dropped_actions": actions.dropped_actions
                    if actions
                    else None,
                }
            )
    return {"games": rows}


# -- Report -------------------------------------------------------------------


def report() -> Report:
    """Score against the heuristic by update, a line per arm."""
    jobs = {_regime_id(arm): recipe(arm) for arm in ARMS}
    return Report(
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        clock="updates",
        labels={_regime_id(arm): DISPLAY[arm] for arm in ARMS},
        evals=(against_fixed("Model vs heuristic", tags=("heuristic",)),),
        perf=PERF,
        ml=(*learning_schedule(jobs), *ML),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="write both tracked populations from the corpus")
    parser.parse_args()
    train, held = build()
    print(f"{train} training decks -> {TRAINING.relative_to(ROOT)}")
    print(f"{held} held-out decks -> {HELD_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
