"""Read the existing scientific Study export without converting it to monitoring.

These narrow views validate the JSON boundary. Study/arena remain the evidence
owners; the existing paired seed/deal bootstrap supplies descriptive intervals.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from manabot.arena.models import canonical_sha256
from manabot.training.analysis import paired_uncertainty
from manabot.training.monitor_evaluation import ArenaRow


class _Saved(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class StudyMeasurement(_Saved):
    regime: str
    seed: int
    variant: str = "raw"
    phase: str = "development"
    cutoff: int
    opponent: str
    complete: bool
    score: float | None
    training_seconds: float
    learner_transitions: int
    decisions: int


class StudyCell(_Saved):
    a: str
    b: str
    training_seed: int
    b_training_seed: int | None = None
    cutoff: int
    variant: str = "raw"
    phase: str = "development"
    scheduled_games: int
    rows: list[ArenaRow]
    replay: dict[str, int | float | bool]

    @property
    def complete(self) -> bool:
        return (
            self.replay.get("passed") is True
            and len(self.rows) == self.scheduled_games
            and bool(self.rows)
            and all(
                {r.leg for r in self.rows if r.deal_seed == deal} == {0, 1, 2, 3}
                for deal in {r.deal_seed for r in self.rows}
            )
            and all(row.valid for row in self.rows)
            and len({(r.deal_seed, r.leg) for r in self.rows}) == len(self.rows)
        )


class StudyRun(_Saved):
    regime: str


class ScientificStudy(_Saved):
    study: str
    protocol_sha256: str | None = None
    status: str
    seeds: list[int]
    seconds: float
    accounting: str = "Evaluation time; training costs are separate."
    runs: list[StudyRun]
    measurements: list[StudyMeasurement]
    comparisons: list[StudyCell]


def load_study(path: Path) -> ScientificStudy:
    """Read one explicit retained study. No recursive cohort discovery or execution."""
    study = ScientificStudy.model_validate_json(path.read_text())
    if study.protocol_sha256 is not None:
        protocol = json.loads(path.with_name("protocol.json").read_text())
        if canonical_sha256(protocol) != study.protocol_sha256:
            raise ValueError("study protocol digest mismatch")
    return study


@dataclass(frozen=True)
class StudyPanel:
    phase: str
    variant: str
    opponent: str
    measurements: list[StudyMeasurement]
    cells: list[StudyCell]


def study_strength_panels(study: ScientificStudy) -> list[StudyPanel]:
    """Admit complete common cutoffs against retained games for every renderer."""
    if study.status != "completed":
        return []
    expected = {(r.regime, seed) for r in study.runs for seed in study.seeds}
    panels = sorted({(m.phase, m.variant, m.opponent) for m in study.measurements})
    admitted: list[StudyPanel] = []
    for phase, variant, opponent in panels:
        selected: list[StudyMeasurement] = []
        cells: list[StudyCell] = []
        for cutoff in sorted({m.cutoff for m in study.measurements}):
            points = [
                m
                for m in study.measurements
                if (m.phase, m.variant, m.opponent, m.cutoff)
                == (phase, variant, opponent, cutoff)
            ]
            cohort = [
                c
                for c in study.comparisons
                if (c.phase, c.variant, c.b, c.cutoff)
                == (phase, variant, opponent, cutoff)
            ]
            if (
                len(points) != len(expected)
                or len(cohort) != len(expected)
                or {(m.regime, m.seed) for m in points} != expected
                or {(c.a, c.training_seed) for c in cohort} != expected
                or not all(m.complete and m.score is not None for m in points)
                or not all(c.complete for c in cohort)
            ):
                continue
            identities = {
                canonical_sha256(r.arena_key.model_dump(mode="json"))
                for c in cohort
                for r in c.rows
            }
            deals = [{(r.deal_seed, r.leg) for r in c.rows} for c in cohort]
            if len(identities) != 1 or any(d != deals[0] for d in deals):
                continue
            # Measurement summaries must agree with their retained terminal games.
            for point in points:
                cell = next(
                    c
                    for c in cohort
                    if (c.a, c.training_seed) == (point.regime, point.seed)
                )
                score = sum(
                    r.score_a for r in cell.rows if r.score_a is not None
                ) / len(cell.rows)
                if point.score is None or abs(score - point.score) > 1e-9:
                    raise ValueError(
                        "study measurement differs from retained arena rows"
                    )
            selected.extend(points)
            cells.extend(cohort)
        identities = {
            (
                canonical_sha256(r.arena_key.model_dump(mode="json")),
                r.player_b_registration_sha256,
            )
            for c in cells
            for r in c.rows
        }
        if not selected or len(identities) != 1:
            continue
        admitted.append(StudyPanel(phase, variant, opponent, selected, cells))
    return admitted


def study_strength_figures(
    study: ScientificStudy,
    axis: Literal["learner_transitions", "training_seconds"] = "learner_transitions",
) -> list[Figure]:
    """Plot independent fits and descriptive training-seed/common-deal intervals."""
    figures: list[Figure] = []
    for panel in study_strength_panels(study):
        selected, cells = panel.measurements, panel.cells
        phase, variant, opponent = panel.phase, panel.variant, panel.opponent
        regimes = sorted({p.regime for p in selected})
        fig = Figure(figsize=(9, 3.2 * len(regimes)), layout="constrained")
        axes = fig.subplots(len(regimes), 1, squeeze=False)[:, 0]
        intervals = paired_uncertainty([c.model_dump(mode="json") for c in cells])
        for ax, regime in zip(axes, regimes, strict=True):
            for seed in study.seeds:
                points = sorted(
                    (p for p in selected if (p.regime, p.seed) == (regime, seed)),
                    key=lambda p: p.cutoff,
                )
                ax.plot(
                    [getattr(p, axis) for p in points],
                    [p.score for p in points],
                    marker="o",
                    label=f"seed {seed}",
                )
            # Aggregate interval has a common x only when work coordinates agree.
            for interval in intervals:
                if interval["a"] != regime or interval["status"] != "available":
                    continue
                x = {
                    getattr(p, axis)
                    for p in selected
                    if p.regime == regime and p.cutoff == interval["cutoff"]
                }
                if len(x) == 1:
                    mean = interval["score_a"]
                    lo, hi = interval["interval_95"]
                    ax.errorbar(
                        list(x),
                        [mean],
                        yerr=[[mean - lo], [hi - mean]],
                        color="black",
                        marker="D",
                        capsize=6,
                        linewidth=2,
                        label="Mean / descriptive 95% seed + deal interval"
                        if interval["cutoff"] == min(p.cutoff for p in selected)
                        else None,
                    )
            ax.set(
                title=f"{regime} · {phase} · {variant} · vs {opponent}",
                xlabel="Learner transitions (matched cutoffs)"
                if axis == "learner_transitions"
                else "Recorded training seconds (unequal cost at matched cutoffs)",
                ylabel="Score (win + half draw)",
                ylim=(-0.03, 1.03),
            )
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Scientific study: {study.study} · {len(study.seeds)} training seeds"
        )
        figures.append(fig)
    return figures
