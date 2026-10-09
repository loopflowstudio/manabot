"""Notebook plots of observed training outcomes, separate from playing strength.

Aggregate counts within update windows, never average per-update percentages.
Stratify by actual deck matchup and keep stages/seeds separate. Both seats use
the current self-play policy: these curves measure matchup and first-turn balance,
not performance against a fixed opponent.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from matplotlib.figure import Figure

from manabot.training.models import TrainingRun
from manabot.training.self_play import deck_identity
import managym

if TYPE_CHECKING:
    from manabot.training.experiment_report import ReportEvidence


@dataclass(frozen=True)
class SelfPlayPoint:
    stage: str
    update: int
    deck: str
    opponent_deck: str
    position: Literal["play", "draw"]
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws


def self_play_points(
    run: TrainingRun, window_updates: int = 100
) -> list[SelfPlayPoint]:
    """Retain joint counts; no observations are emitted for unrecorded history."""
    if window_updates < 1:
        raise ValueError("window_updates must be positive")
    points: list[SelfPlayPoint] = []
    for stage in run.stages:
        groups: dict[tuple[int, str, str, Literal["play", "draw"]], list[int]] = {}
        for diagnostic in stage.diagnostics:
            update = diagnostic.get("coordinates", {}).get("updates")
            if not isinstance(update, int):
                continue
            for row in diagnostic.get("self_play_outcomes", []):
                key = (
                    (max(update, 1) - 1) // window_updates,
                    row["deck"],
                    row["opponent_deck"],
                    row["position"],
                )
                totals = groups.setdefault(key, [0, 0, 0, update])
                totals[0] += row["wins"]
                totals[1] += row["losses"]
                totals[2] += row["draws"]
                totals[3] = max(totals[3], update)
        for (_, deck, opponent, position), (wins, losses, draws, update) in sorted(
            groups.items()
        ):
            if wins + losses + draws:
                points.append(
                    SelfPlayPoint(
                        stage.id, update, deck, opponent, position, wins, losses, draws
                    )
                )
    return points


def _deck_labels() -> dict[str, str]:
    """Recognize exact authored decks, including sideboards; otherwise show IDs."""
    labels: dict[str, str] = {}
    for key, label in (("ur_lessons", "Lessons"), ("gw_allies", "Allies")):
        setup = managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
        labels[deck_identity(dict(setup.decklist), dict(setup.sideboard))] = label
    return labels


def self_play_figures(
    evidence: "ReportEvidence", window_updates: int = 100
) -> list[Figure]:
    """Two views per matchup: play/draw and deck win rates, on an update axis."""
    figures: list[Figure] = []
    for run in evidence.runs:
        points = self_play_points(run, window_updates)
        labels = _deck_labels() if points else {}
        matchups = sorted({tuple(sorted((p.deck, p.opponent_deck))) for p in points})
        if not matchups:
            continue
        for matchup in matchups:
            selected = [
                p for p in points if tuple(sorted((p.deck, p.opponent_deck))) == matchup
            ]
            fig = Figure(figsize=(11, 4), layout="constrained")
            axes = fig.subplots(1, 2)
            names = [labels.get(deck, f"Deck {deck[:8]}") for deck in matchup]
            fig.suptitle(f"{evidence.label(run.id)}\nSelf-play: {' vs '.join(names)}")
            for ax, dimension in zip(axes, ("position", "deck"), strict=True):
                grouped: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(dict)
                for point in selected:
                    label = point.position if dimension == "position" else point.deck
                    totals = grouped[(point.stage, label)].setdefault(
                        point.update, [0, 0]
                    )
                    totals[0] += point.wins
                    totals[1] += point.games
                for (stage, label), series in sorted(grouped.items()):
                    xs = sorted(series)
                    ys = [series[x][0] / series[x][1] for x in xs]
                    name = (
                        labels.get(label, f"Deck {label[:8]}")
                        if dimension == "deck"
                        else f"On the {label}"
                    )
                    ax.plot(xs, ys, marker=".", label=f"{name} · {stage}")
                ax.set(
                    xlabel="Training updates",
                    ylabel="Wins / completed seat outcomes",
                    ylim=(0, 1),
                    title="Play / draw" if dimension == "position" else "Deck win rate",
                )
                ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
                ax.legend(fontsize="small")
            fig.text(
                0.5,
                -0.06,
                f"Up to {window_updates} updates/window; draws count as non-wins. "
                "Mirror deck rate is mechanically 50% without draws. Not a strength evaluation.",
                ha="center",
                fontsize=8,
            )
            figures.append(fig)
    if not figures:
        fig = Figure(figsize=(9, 2), layout="constrained")
        ax = fig.subplots()
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Self-play deck and play/draw outcomes unavailable: not recorded in these runs.",
            ha="center",
            va="center",
            wrap=True,
        )
        figures.append(fig)
    return figures
