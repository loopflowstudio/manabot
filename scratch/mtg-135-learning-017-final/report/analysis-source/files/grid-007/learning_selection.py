"""Apply protocol017's frozen competence and recipe-selection rules.

The selector consumes already verified checkpoint scores. It preserves training
seeds separately, requires sustained competence at retained checkpoints, and
returns no preferred recipe when none passes the declared thresholds.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from manabot.training.models import TrainingCoordinates

SEEDS = (13591, 13592)
COUNTS = (0, 131072, 262144, 524288)
ANCHORS = ('scripted_greedy', 'random')
type Anchor = Literal['scripted_greedy', 'random']


class Point(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    name: str
    cell: str
    size: str
    streams: int
    seed: int
    anchor: Anchor
    transitions: int
    mean: float = Field(ge=0, le=1)
    deal_blocks: list[float]
    roles: dict[str, float]
    coordinates: TrainingCoordinates
    checkpoint_sha256: str
    initialization_improvement: float = 0
    paired_deal_improvement_95: list[float] = Field(default_factory=list)


class CellSummary(BaseModel):
    parameters: int = Field(gt=0)
    peak_cuda_allocated: int = Field(ge=0)


class RetainedCell(BaseModel):
    name: str
    cell: str
    seed: int
    size: str
    streams: int
    summary: CellSummary


class AnchorScores(BaseModel):
    scripted_greedy: float
    random: float


class SeedShortfall(BaseModel):
    seed: int
    greedy_below_threshold: float
    random_below_threshold: float
    greedy_improvement_shortfall: float


class Recipe(BaseModel):
    cell: str
    qualified: bool
    first_qualifying_transitions: int | None
    worst_seed_seconds_to_first_qualified: float | None
    parameters: int
    worst_seed_peak_cuda_bytes: int
    endpoint_scores: dict[int, AnchorScores]
    endpoint_shortfalls: list[SeedShortfall]
    full_recipe_transitions: int


class Selection(BaseModel):
    recipes: list[Recipe]
    preferred_by_frozen_rule: Recipe | None


def select_recipes(points: Sequence[Point], cells: Sequence[RetainedCell]) -> Selection:
    """Use the complete seed/checkpoint panel; absence is never a failed score."""
    metadata = {(cell.cell, cell.seed): cell for cell in cells}
    if len(metadata) != len(cells):
        raise ValueError('Duplicate retained training cell')
    names = sorted({cell.cell for cell in cells})
    if not names or set(metadata) != {(cell, seed) for cell in names for seed in SEEDS}:
        raise ValueError('Every recipe requires both declared training seeds')
    indexed = {(p.cell, p.seed, p.anchor, p.transitions): p for p in points}
    expected = {(cell, seed, anchor, count) for cell in names for seed in SEEDS
                for anchor in ANCHORS for count in COUNTS}
    if len(indexed) != len(points) or set(indexed) != expected:
        raise ValueError('Selection requires every declared checkpoint and anchor exactly once')
    for point in points:
        origin = metadata[point.cell, point.seed]
        if (point.name, point.size, point.streams) != (origin.name, origin.size, origin.streams):
            raise ValueError('Score identity differs from the retained training cell')
        if point.coordinates.learner_transitions != point.transitions:
            raise ValueError('Score checkpoint coordinate differs from its transition label')
        initial = indexed[point.cell, point.seed, point.anchor, 0]
        if abs(point.mean - initial.mean - point.initialization_improvement) > 1e-12:
            raise ValueError('Initialization improvement does not match the paired seed')
    recipes: list[Recipe] = []
    for cell in names:
        for seed in SEEDS:
            clocks: list[float] = []
            for count in COUNTS:
                g = indexed[cell, seed, 'scripted_greedy', count]
                r = indexed[cell, seed, 'random', count]
                if g.coordinates != r.coordinates:
                    raise ValueError('Anchor panels refer to different training coordinates')
                clocks.append(g.coordinates.training_seconds)
            if any(later <= earlier for earlier, later in zip(clocks, clocks[1:])):
                raise ValueError('Retained training clocks must increase')
        qualifying: list[int] = []
        for count in COUNTS[1:]:
            passes = True
            for later in (n for n in COUNTS if n >= count):
                for seed in SEEDS:
                    greedy = indexed[cell, seed, 'scripted_greedy', later]
                    random = indexed[cell, seed, 'random', later]
                    if (greedy.mean < .5 or random.mean < .75
                            or greedy.initialization_improvement < .2 - 1e-12):
                        passes = False
            if passes:
                qualifying.append(count)
        endpoint: dict[int, AnchorScores] = {}
        shortfalls: list[SeedShortfall] = []
        for seed in SEEDS:
            g = indexed[cell, seed, 'scripted_greedy', COUNTS[-1]]
            r = indexed[cell, seed, 'random', COUNTS[-1]]
            endpoint[seed] = AnchorScores(scripted_greedy=g.mean, random=r.mean)
            shortfalls.append(SeedShortfall(seed=seed,
                greedy_below_threshold=max(0, .5 - g.mean),
                random_below_threshold=max(0, .75 - r.mean),
                greedy_improvement_shortfall=max(0, .2 - g.initialization_improvement)))
        retained = [metadata[cell, seed] for seed in SEEDS]
        parameters = {c.summary.parameters for c in retained}
        if len(parameters) != 1:
            raise ValueError('A recipe changed parameter count between training seeds')
        first = min(qualifying) if qualifying else None
        worst_time = max(indexed[cell, seed, 'scripted_greedy', first].coordinates.training_seconds
                         for seed in SEEDS) if first is not None else None
        recipes.append(Recipe(cell=cell, qualified=first is not None,
            first_qualifying_transitions=first, worst_seed_seconds_to_first_qualified=worst_time,
            parameters=parameters.pop(),
            worst_seed_peak_cuda_bytes=max(c.summary.peak_cuda_allocated for c in retained),
            endpoint_scores=endpoint, endpoint_shortfalls=shortfalls,
            full_recipe_transitions=COUNTS[-1]))
    qualified = [r for r in recipes if r.qualified]
    preferred = None
    if qualified:
        fastest = min(_qualified_time(r) for r in qualified)
        near = [r for r in qualified if _qualified_time(r) <= fastest * 1.1]
        preferred = min(near, key=lambda r: (r.parameters, r.worst_seed_peak_cuda_bytes,
                                           _qualified_time(r), r.cell))
    return Selection(recipes=recipes, preferred_by_frozen_rule=preferred)


def _qualified_time(recipe: Recipe) -> float:
    if recipe.worst_seed_seconds_to_first_qualified is None:
        raise ValueError('Unqualified recipe has no time-to-competence')
    return recipe.worst_seed_seconds_to_first_qualified
