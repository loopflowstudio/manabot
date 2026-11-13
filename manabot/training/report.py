"""Report panels for the comparisons a manabot training study asks.

An experiment's `report()` composes these into an Etude `Report`, beside
`panels.PERF` and `panels.ML`, and adds only what is specific to it. Each helper
returns an ordinary Etude record, so a report mixes them with its own panels:

- `job_labels`: `Report.labels`, a display name per job from its regime's model;
- `against_earlier_self`, `against_fixed`, `head_to_head`: eval `Curve`s, one
  chart per question with a line per job;
- `learning_schedule`: ML `Schedule` panels of the configured damped learning
  rate and policy damping on the update clock.

Jobs are keyed by their Etude name, which for a `TrainingJob` is its regime id.
Matches are named by the tags an experiment's `eval(state)` gives its matchups.
"""

from collections.abc import Mapping, Sequence

from etude.report import Curve, Matches, Schedule, SchedulePoint

from manabot.training.learning import DampedPolicyLearning
from manabot.training.regime import TrainingRegime, TrainSelfPlay

# Configured values are sampled at this many updates per job, geometrically
# spaced so the steep early schedule is drawn as finely as its long tail;
# Etude interpolates linearly between them.
SCHEDULE_SAMPLES = 96


# -- Names ---------------------------------------------------------------------


def job_labels(jobs: Mapping[str, tuple[str, TrainingRegime]]) -> dict[str, str]:
    """`Report.labels`: each job's size word, then its model's width and depth.

    `jobs` maps a job's Etude name to the word a reader knows it by and its
    regime. Attention layers are named only for a job whose depth differs from
    the first job's, so "Small (width 64)" and "Large (width 160, 8 layers)".
    """
    if not jobs:
        return {}
    first = next(iter(jobs.values()))[1].agent.attention_layers
    labels: dict[str, str] = {}
    for name, (word, regime) in jobs.items():
        model = regime.agent
        parts = [f"width {model.hidden_dim}"]
        if model.attention_layers != first:
            layers = model.attention_layers
            parts.append(f"{layers} layer{'' if layers == 1 else 's'}")
        labels[name] = f"{word} ({', '.join(parts)})"
    return labels


# -- Comparisons ---------------------------------------------------------------


def against_earlier_self(title: str, tags: Sequence[str]) -> Curve:
    """Each checkpoint against an earlier checkpoint of its own job, a line per job.

    `tags` name the matchups that pair a checkpoint with its earlier self,
    typically one tag per job. Each checkpoint's point counts every such match
    it sat in.
    """
    return Curve(title=title, matches=tuple(Matches(tag=tag) for tag in tags))


def against_fixed(title: str, opponent: str, tags: Sequence[str]) -> Curve:
    """Each checkpoint against one fixed player, by its Etude name, a line per job."""
    return Curve(
        title=title,
        matches=tuple(Matches(tag=tag) for tag in tags),
        opponent=opponent,
    )


def head_to_head(title: str, subject: str, tag: str) -> Curve:
    """Two jobs against each other, drawn once from `subject`'s side.

    A head-to-head score and its complement carry the same information, so
    only `subject`'s line is drawn: above the reference means `subject` is
    stronger.
    """
    return Curve(title=title, matches=Matches(tag=tag), subject=subject)


# -- Learning schedule ---------------------------------------------------------


def _damped(name: str, regime: TrainingRegime) -> tuple[DampedPolicyLearning, int]:
    """The regime's one damped self-play learning rule and the last update to draw.

    That is the damping horizon, after which both values hold at their endpoint,
    or, for a rule without one, the stage's update ceiling.
    """
    stages = [stage for stage in regime.stages if isinstance(stage, TrainSelfPlay)]
    if len(stages) != 1 or not isinstance(stages[0].learning, DampedPolicyLearning):
        raise ValueError(
            f"{name}: a learning schedule needs one damped self-play stage"
        )
    learning = stages[0].learning
    schedule = learning.iteration_schedule
    return learning, schedule.steps if schedule is not None else stages[0].updates


def _sampled(last: int) -> tuple[int, ...]:
    """Updates 1..`last`, geometrically spaced, both ends included."""
    if last <= SCHEDULE_SAMPLES:
        return tuple(range(1, last + 1))
    ratio = last ** (1 / (SCHEDULE_SAMPLES - 1))
    return tuple(sorted({1, last, *(round(ratio**i) for i in range(SCHEDULE_SAMPLES))}))


def learning_schedule(
    regimes: Mapping[str, TrainingRegime],
) -> tuple[Schedule, Schedule]:
    """The configured learning rate and policy damping of each job, by update.

    Values come from `DampedPolicyLearning.rates`, the method the learner calls,
    through each job's damping horizon. These are plans, not measurements;
    Etude marks each run's latest update, and says so for a run past the horizon.
    """
    rates: dict[str, list[tuple[int, tuple[float, float]]]] = {}
    for name, regime in regimes.items():
        learning, last = _damped(name, regime)
        rates[name] = [(u, learning.rates(u)) for u in _sampled(last)]

    def panel(title: str, unit: str, index: int) -> Schedule:
        return Schedule(
            title=title,
            clock="updates",
            unit=unit,
            jobs={
                name: tuple(SchedulePoint(at=u, value=r[index]) for u, r in found)
                for name, found in rates.items()
            },
        )

    return (
        panel("Learning rate schedule", "", 0),
        panel("Policy damping schedule", "τ", 1),
    )
