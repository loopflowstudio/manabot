"""Does BF16 learn the same as float32 at w384l8? The gate on the precision ladder (MTG-324).

`Execution.precision="auto"` trains a CUDA model of width at least
`BF16_MIN_HIDDEN_DIM` (384) in BF16. MTG-317 (`experiments/cuda_precision.md`)
measured BF16 2–2.7× faster than float32 at w384l8 on an L4, but its 200-step
replay could not tell same from different learning. This experiment trains the
frontier recipe twice from the same seeds, once in each precision, and compares
playing strength at matched updates plus the learning trajectories. The
pre-registered question, prediction and refutation open
`experiments/precision_ladder.md`.

Arms (`arm`): `sampled_self_play` over the ten-deck Elemental roster (two-seat
damped self-play, 256 streams, four producers) at w384l8 (8 heads, the model's
default feed-forward, as MTG-317 measured), bucketed shapes on, and the
precision written explicitly so the ladder does not choose it. The damping
horizon is `UPDATES` in both arms, so learning rate and τ stand at the same
point at every matched update. Only `id` and `execution.precision` differ
(`require_single_axis`).

Evaluation (`eval`): each run's checkpoints at updates 20 and 40 and at its end
against the frozen heuristic over 07c490's fifty held-out Elemental pairings
(`learning_signal.ELEMENTAL_PAIRS`, four cross legs each), and each BF16 run
against each float32 run at the last update both checkpointed.

    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        experiments/precision_ladder.py --rent l4 --parallel 4 \\
        --max-usd 10 --revision COMMIT
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
# Checkpoints at learning_signal's scheduled updates below this (20 and 40) and
# at each run's end are scored.
UPDATES = 64
WIDTH, LAYERS, HEADS = 384, 8, 8
# Expected float32 seconds per update at w384l8 on an L4: the A40 measured
# 100.6 s solo at a384l8 (experiments/training-update-time.md) and the L4 is
# slower. Sizes the watchdog only; the damping horizon is `UPDATES` either way.
FLOAT32_PACE_SECONDS = 150.0
# 64 × 150 s plus setup and the final export: 3 h per run. Etude reserves the
# window plus setup at the L4's USD 0.60/h ceiling, USD 1.95 per run, 7.80 for four.
HOURS = (UPDATES * FLOAT32_PACE_SECONDS + SETUP_SECONDS + FINALIZATION_SECONDS) / 3600
USD_PER_RUN = 1.95
EVALUATION = "precision-ladder-v1"
# 12 heuristic matchups and 4 head-to-heads × 50 pairings × 4 legs: 3,200 games,
# about a third of value_readout_252's pass (9,000 games in 1.25 h on cpu32).
EVALUATION_HOURS = 0.75
EVALUATION_USD = 1.5


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
    stage.execution.precision = precision
    # The JSON round trip drops the ladder's private mark from the stage, so
    # the written precision is authored rather than re-resolved.
    return TrainingRegime.model_validate(regime.model_dump(mode="json"))


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
) -> list[Run]:
    """Both arms under the same seeds.

    The run seed fixes initialization and every native game seed, so the two
    arms of one seed start from the same weights and meet the same deck
    pairings and deals until their policies diverge.
    """
    chosen = tuple(int(seed) for seed in seeds.split(","))
    if len(chosen) < 2 or len(set(chosen)) != len(chosen):
        raise ValueError("the comparison needs at least two distinct shared seeds")
    reference, candidate = (arm(precision) for precision in ARMS)
    require_single_axis(reference, candidate)
    return training_runs(
        (reference, candidate),
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
