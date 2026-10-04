"""Whole-game held-out associations, conditional on one frozen policy.

GAE consumes complete same-viewer sequences [T,1] with a final end marker and
zero bootstrap. Filtering uses detached raw advantages across each partition.
A terminal residual is one noisy outcome sample, never true expected-value
error. Game-cluster intervals keep both seats and all decisions together.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field
import torch

from manabot.arena.models import canonical_sha256
from manabot.sim.net_opponent import transition_gae
from manabot.training.models import AtaraxosMoveLearning, CollectSelection, Strict
from manabot.training.selection import (
    SelectionGroup,
    selection_diagnostics,
    selection_mask,
)
from manabot.training.selection_data import FrozenRecord, SelectionDataset


class AnalyzedDecision(FrozenRecord):
    game: int
    step: int
    actor: int
    split: Literal["development", "held_out"]
    action_type: int
    terminal_distance: int
    advantage: float
    lambda_residual: float
    terminal_residual: float
    discounted_terminal_residual: float
    retained: bool
    entropy: float
    reference_kl: float


class GroupSummary(Strict):
    rows: int
    games: int
    advantage_quantiles: list[float]
    lambda_residual_abs_mean: float | None
    terminal_residual_mean: float | None
    terminal_residual_abs_mean: float | None
    discounted_terminal_residual_abs_mean: float | None
    entropy_mean: float | None
    reference_kl_mean: float | None


class Association(Strict):
    split: str
    action_type: int
    terminal_distance: str
    retained: GroupSummary
    excluded: GroupSummary
    terminal_abs_difference: float | None
    terminal_abs_difference_ci95: list[float] | None
    bootstrap_defined: int


class PartitionSelectionGroup(SelectionGroup):
    split: str


class SelectionReport(Strict):
    dataset_sha256: str
    recipe_sha256: str
    status: Literal["descriptive-only"] = "descriptive-only"
    optimizer_exposures: Literal[0] = 0
    bootstrapped_tail_fraction: Literal[0] = 0
    rows: list[AnalyzedDecision]
    associations: list[Association]
    # Same API as live diagnostics; terminal residuals are additional, not replacements.
    selection_groups: list[PartitionSelectionGroup] = Field(default_factory=list)


def _bin(distance: int) -> str:
    return "terminal" if distance == 0 else "1-4" if distance <= 4 else "5+"


def _summary(rows: list[AnalyzedDecision]) -> GroupSummary:
    def mean(value: Callable[[AnalyzedDecision], float]) -> float | None:
        return float(np.mean([value(r) for r in rows])) if rows else None

    return GroupSummary(
        rows=len(rows),
        games=len({r.game for r in rows}),
        advantage_quantiles=np.quantile(
            [r.advantage for r in rows], [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if rows
        else [],
        lambda_residual_abs_mean=mean(lambda r: abs(r.lambda_residual)),
        terminal_residual_mean=mean(lambda r: r.terminal_residual),
        terminal_residual_abs_mean=mean(lambda r: abs(r.terminal_residual)),
        discounted_terminal_residual_abs_mean=mean(
            lambda r: abs(r.discounted_terminal_residual)
        ),
        entropy_mean=mean(lambda r: r.entropy),
        reference_kl_mean=mean(lambda r: r.reference_kl),
    )


@torch.no_grad()
def analyze_selection(
    dataset: SelectionDataset,
    stage: CollectSelection,
    *,
    check: Callable[[], None] = lambda: None,
) -> SelectionReport:
    """No fitting; partition-local selection with complete-game end semantics.

    Raw scalar values and categorical expected signed values share this GAE
    expectation. Categorical cross-entropy/calibration is not measured here.
    Intervals condition on the observed selection mask and frozen checkpoint;
    they do not quantify training-seed uncertainty or threshold-estimation noise.
    """
    if tuple(g.spec for g in dataset.games) != stage.population:
        raise ValueError("analysis population differs from frozen recipe")
    learning = stage.learning
    gamma = 1.0 if isinstance(learning, AtaraxosMoveLearning) else learning.gamma
    fraction = (
        1 - learning.advantage_quantile
        if isinstance(learning, AtaraxosMoveLearning)
        else learning.retained_fraction
    )
    rows: list[AnalyzedDecision] = []
    for game_index, game in enumerate(dataset.games):
        check()
        for actor in (0, 1):
            decisions = [r for r in game.rows if r.actor == actor]
            if not decisions:
                continue
            values = torch.tensor([r.value for r in decisions])[:, None]
            rewards = torch.zeros_like(values)
            outcome = (
                0.0 if game.winner is None else 1.0 if game.winner == actor else -1.0
            )
            rewards[-1] = outcome
            ends = torch.zeros_like(values, dtype=torch.bool)
            ends[-1] = True
            advantage, _ = transition_gae(
                rewards, values, ends, torch.zeros(1), gamma, learning.policy_lambda
            )
            residual, _ = transition_gae(
                rewards, values, ends, torch.zeros(1), gamma, learning.value_lambda
            )
            for index, decision in enumerate(decisions):
                distance = len(decisions) - 1 - index
                rows.append(
                    AnalyzedDecision(
                        game=game_index,
                        step=decision.step,
                        actor=actor,
                        split=game.spec.split,
                        action_type=decision.action_type,
                        terminal_distance=distance,
                        advantage=float(advantage[index, 0]),
                        lambda_residual=float(residual[index, 0]),
                        terminal_residual=outcome - decision.value,
                        discounted_terminal_residual=gamma**distance * outcome
                        - decision.value,
                        retained=False,
                        entropy=decision.entropy,
                        reference_kl=decision.reference_kl,
                    )
                )
    groups: list[PartitionSelectionGroup] = []
    for split in ("development", "held_out"):
        indices = [i for i, row in enumerate(rows) if row.split == split]
        advantages = torch.tensor([rows[i].advantage for i in indices])[:, None]
        selected = selection_mask(
            advantages, learning.filter_kind, fraction, learning.min_advantage
        )
        for i, keep in zip(indices, selected.flatten().tolist(), strict=True):
            rows[i] = rows[i].model_copy(update={"retained": keep})
        for group in selection_diagnostics(
            advantages,
            torch.tensor([rows[i].lambda_residual for i in indices])[:, None],
            selected,
            torch.tensor([rows[i].terminal_distance == 0 for i in indices])[:, None],
            torch.tensor([rows[i].action_type for i in indices])[:, None],
        ):
            groups.append(PartitionSelectionGroup(split=split, **group))
    rng = np.random.default_rng(stage.bootstrap_seed)
    associations: list[Association] = []
    for split, action_type, distance in sorted(
        {(r.split, r.action_type, _bin(r.terminal_distance)) for r in rows}
    ):
        check()
        subset = [
            r
            for r in rows
            if (r.split, r.action_type, _bin(r.terminal_distance))
            == (split, action_type, distance)
        ]
        retained = _summary([r for r in subset if r.retained])
        excluded = _summary([r for r in subset if not r.retained])
        game_ids = [
            i for i, game in enumerate(dataset.games) if game.spec.split == split
        ]
        # Sums/counts keep seats together; zero-contribution games remain in the population.
        clusters = np.array(
            [
                [
                    sum(
                        abs(r.terminal_residual)
                        for r in subset
                        if r.game == i and r.retained
                    ),
                    sum(r.game == i and r.retained for r in subset),
                    sum(
                        abs(r.terminal_residual)
                        for r in subset
                        if r.game == i and not r.retained
                    ),
                    sum(r.game == i and not r.retained for r in subset),
                ]
                for i in game_ids
            ]
        )
        samples = clusters[
            rng.integers(len(game_ids), size=(stage.bootstrap_samples, len(game_ids)))
        ].sum(axis=1)
        valid = (samples[:, 1] > 0) & (samples[:, 3] > 0)
        differences = (
            samples[valid, 0] / samples[valid, 1]
            - samples[valid, 2] / samples[valid, 3]
        )
        point = (
            None
            if retained.terminal_residual_abs_mean is None
            or excluded.terminal_residual_abs_mean is None
            else retained.terminal_residual_abs_mean
            - excluded.terminal_residual_abs_mean
        )
        # Require both groups in at least two independent games and 95% defined draws.
        interval = (
            np.quantile(differences, [0.025, 0.975]).tolist()
            if retained.games >= 2 and excluded.games >= 2 and valid.mean() >= 0.95
            else None
        )
        associations.append(
            Association(
                split=split,
                action_type=action_type,
                terminal_distance=distance,
                retained=retained,
                excluded=excluded,
                terminal_abs_difference=point,
                terminal_abs_difference_ci95=interval,
                bootstrap_defined=int(valid.sum()),
            )
        )
    return SelectionReport(
        dataset_sha256=canonical_sha256(dataset.model_dump(mode="json")),
        recipe_sha256=canonical_sha256(stage.model_dump(mode="json")),
        rows=rows,
        associations=associations,
        selection_groups=groups,
    )


def write_selection_report(report: SelectionReport, path: Path) -> None:
    """Derived Markdown; numeric evidence and uncertainty remain in the JSON artifact."""
    lines = [
        "# Frozen-policy filtering associations",
        "",
        f"Dataset `{report.dataset_sha256}`; recipe `{report.recipe_sha256}`.",
        "",
        "Descriptive only. A terminal outcome is a noisy sample, not true expected-value error. Associations do not establish causal benefit or that filtering selects mistakes.",
        "",
        "95% percentile intervals resample whole games (both seats together), conditional on this frozen policy and observed mask. Missing intervals mean insufficient support; rows and raw/EMA variants are not independent training replicates.",
        "",
        "Distances count same-viewer surfaced decisions to observed terminal (last=0). Selection uses each complete partition, not historical live minibatches. Lambda-target and terminal-outcome residuals are separate. No unfinished tails or optimizer exposures.",
        "",
        "| Split | Action type | Distance | Retained / excluded | Terminal absolute residual difference | 95% game interval |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for group in report.associations:
        lines.append(
            f"| {group.split} | {group.action_type} | {group.terminal_distance} | {group.retained.rows} / {group.excluded.rows} | {group.terminal_abs_difference} | {group.terminal_abs_difference_ci95 or 'unavailable'} |"
        )
    path.write_text("\n".join(lines) + "\n")
