"""The BF16 learning gate at w384l8 (MTG-324).

`arm` builds float32 and BF16 self-play regimes with the same damping horizon;
`runs` pairs their seeds. `eval` schedules heuristic and head-to-head matches
at retained checkpoints; `report` displays scores and learning trajectories.
The numerical gate rejected BF16 before evaluation. The frozen prediction,
measurements, costs and launch history are in `experiments/precision_ladder.md`.

    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        experiments/precision_ladder.py --rent mtg324-l4-ab --parallel 4 \\
        --cohort NAME --max-usd 32 --revision COMMIT --hours 10.9
    uv run etude eval experiments/precision_ladder.py --rent cpu32 --source r2://manabot/jobs
    uv run etude report experiments/precision_ladder.py --source r2://manabot/jobs --open
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from etude.eval.records import Inference, Matchup
from etude.experiment import ExperimentState, TrainingRun
from etude.launch.records import Job, Run
from etude.report import Curve, Matches, Ranking, Report, Score

from experiments.elemental_variants import elemental_recipe
from experiments.learning_signal import (
    ELEMENTAL_PAIRS,
    FINALIZATION_SECONDS,
    SETUP_SECONDS,
    Conditions,
    benchmark_jobs,
    scheduled_matchups,
    training_runs,
)
from manabot.infra.runtime import regime_tag
from manabot.recipes.builders import sampled_self_play
from manabot.training.panels import ML, PERF
from manabot.training.regime import Precision, TrainingRegime, TrainSelfPlay
import managym

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).stem
# float32 first: the reference arm and the report's dotted baseline.
ARMS: tuple[Precision, ...] = ("float32", "bf16")
SEEDS = (32401, 32402)
# Checkpoints at learning_signal's scheduled updates (20, 40, 75 and 150) and
# at each run's end are scored.
UPDATES = 300
WIDTH, LAYERS, HEADS = 384, 8, 8
# Expected float32 seconds per update at w384l8 on an L4: the A40 measured
# 100.6 s solo at a384l8 (experiments/training-update-time.md) and the L4 is
# slower. Sizes the watchdog only; the damping horizon is `UPDATES` either way.
FLOAT32_PACE_SECONDS = 150.0
# 300 × 150 s plus setup and the final export: 12.8 h per run, which also sets
# the regime's wall clock. The scored launch shortened only the job window
# (`--hours 10.9` at a USD 0.70/h ceiling; experiments/precision_ladder.md § Status),
# keeping USD 7.85 per run, 31.40 for four.
HOURS = (UPDATES * FLOAT32_PACE_SECONDS + SETUP_SECONDS + FINALIZATION_SECONDS) / 3600
USD_PER_RUN = 7.85
EVALUATION = "precision-ladder-v1"
# 20 heuristic matchups and 4 head-to-heads × 50 pairings × 4 legs: 4,800 games,
# about half of value_readout_252's pass (9,000 games in 1.25 h on cpu32).
EVALUATION_HOURS = 1.0
EVALUATION_USD = 2.0


def regime_id(precision: Precision) -> str:
    return f"precision-ladder-w{WIDTH}l{LAYERS}-{precision}-v1"


def arm(precision: Precision) -> TrainingRegime:
    """The frontier recipe at w384l8 with bucketed shapes and `precision` written out."""
    elemental = elemental_recipe(regime_id(precision), UPDATES)
    agent = elemental.agent.model_copy(
        update={
            "hidden_dim": WIDTH,
            "attention_layers": LAYERS,
            "num_attention_heads": HEADS,
        }
    )
    wall_seconds = HOURS * 3600 - SETUP_SECONDS - FINALIZATION_SECONDS
    regime = sampled_self_play(
        id=regime_id(precision),
        world=managym.WORLD_VERSION,
        agent=agent,
        observation=elemental.observation,
        match=elemental.match,
        wall_seconds=wall_seconds,
        # The horizon is `UPDATES` exactly, whatever the arm's real pace.
        seconds_per_update=wall_seconds / UPDATES,
        elemental_uniform=True,
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = UPDATES
    stage.execution.bucketed_shapes = True
    stage.execution.live_tokens = False
    stage.execution.precision = precision
    return TrainingRegime.model_validate(regime.authored_dump())


def require_single_axis(baseline: TrainingRegime, candidate: TrainingRegime) -> None:
    """Reject any resolved difference other than the label and the precision."""
    left = baseline.model_dump(mode="json")
    right = candidate.model_dump(mode="json")
    for dump in (left, right):
        dump.pop("id")
        dump["stages"][0]["execution"].pop("precision")
    if left != right:
        raise ValueError("precision arm changes more than execution.precision")


def runs(
    revision: str,
    hours: float = HOURS,
    usd_per_run: float = USD_PER_RUN,
    seeds: str = ",".join(str(seed) for seed in SEEDS),
    arms: str = ",".join(ARMS),
) -> list[Run]:
    """Both arms under the same seeds.

    The run seed fixes initialization and every native game seed, so the two
    arms of one seed start from the same weights and meet the same deck
    pairings and deals until their policies diverge. `arms` and one seed
    re-queue a single lost run with the same regime.
    """
    chosen = tuple(int(seed) for seed in seeds.split(","))
    if len(set(chosen)) != len(chosen):
        raise ValueError("seeds repeat")
    wanted = set(arms.split(","))
    if not wanted <= set(ARMS):
        raise ValueError(f"unknown arms {sorted(wanted - set(ARMS))}")
    if wanted == set(ARMS) and len(chosen) < 2:
        raise ValueError("the comparison needs at least two distinct shared seeds")
    reference, candidate = (arm(precision) for precision in ARMS)
    require_single_axis(reference, candidate)
    return training_runs(
        tuple(
            regime
            for precision, regime in zip(ARMS, (reference, candidate), strict=True)
            if precision in wanted
        ),
        updates=UPDATES,
        revision=revision,
        experiment=EXPERIMENT,
        seeds=chosen,
        hours=hours,
        usd_per_run=usd_per_run,
    )


# --------------------------------------------------------------------------- evaluation


def arm_runs(state: ExperimentState, precision: Precision) -> list[TrainingRun]:
    """The arm's runs: their checkpoints' regime tag names them."""
    tag = regime_tag(regime_id(precision))
    return [run for run in state.runs if run.checkpoints.tagged(tag)]


def eval(state: ExperimentState) -> list[Job]:
    return benchmark_jobs(
        evaluation_matchups(state),
        EVALUATION,
        ELEMENTAL_PAIRS,
        hours=EVALUATION_HOURS,
        usd=EVALUATION_USD,
    )


def evaluation_matchups(state: ExperimentState) -> tuple[Matchup, ...]:
    """Scheduled checkpoints against the heuristic, then BF16 against float32.

    A checkpoint not yet stored is left out, so a later pass plays it. A
    head-to-head pairs two ended runs at the last update both checkpointed,
    float32 first; the four legs give each both decks on the play and the draw.
    """
    found = [
        matchup
        for run in state.runs
        for matchup in scheduled_matchups(run, ELEMENTAL_PAIRS)
    ]
    # The same source as `scheduled_matchups`, so a checkpoint is one player in both.
    source = [
        *sorted((ROOT / "manabot").rglob("*.py")),
        ROOT / "experiments/learning_signal.py",
    ]
    inference = Inference(source=source)
    conditions = tuple(
        Conditions(design=ELEMENTAL_PAIRS.name, **pairing.model_dump()).model_dump()
        for pairing in ELEMENTAL_PAIRS.pairings
    )
    reference, candidate = (arm_runs(state, precision) for precision in ARMS)
    for left, right in product(reference, candidate):
        if not (left.ended and right.ended):
            continue
        shared = {int(c.clocks["updates"]) for c in left.checkpoints} & {
            int(c.clocks["updates"]) for c in right.checkpoints
        }
        if not shared:
            continue
        update = max(shared)
        first = left.checkpoints.at("updates", update).last
        second = right.checkpoints.at("updates", update).last
        assert first is not None and second is not None
        found.append(
            Matchup(
                name=f"{ARMS[0]} {left.name} v {ARMS[1]} {right.name} u{update}",
                players=(first.model(inference), second.model(inference)),
                conditions=conditions,
                games=ELEMENTAL_PAIRS.legs,
                tags=("head-to-head",),
            )
        )
    return tuple(found)


def report() -> Report:
    """Score against the heuristic by update; the ML panels hold loss, clip and KL."""
    heuristic = Matches(tag="heuristic")
    return Report(
        score=Score(table="plays", value="score", reference=0.5),
        clock="updates",
        baseline=regime_id(ARMS[0]),
        evals=(
            Curve(title="Score against the heuristic", matches=heuristic),
            Ranking(title="On the play or the draw", by="on_play"),
        ),
        perf=PERF,
        ml=ML,
    )
