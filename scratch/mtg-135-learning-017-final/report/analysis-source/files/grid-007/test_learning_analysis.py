"""Exercise scientific selection failures and retained-evidence integrity."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from analyze_learning import verify_manifest
from learning_figures import Tradeoff, build_tradeoffs, mark_frontier
from learning_selection import (
    COUNTS, SEEDS, Anchor, CellSummary, Point, RetainedCell, select_recipes,
)
from manabot.training.models import TrainingCoordinates


@dataclass
class Fixture:
    points: list[Point]
    cells: list[RetainedCell]


def _fixture(cell: str = 'small-s64', *, parameters: int = 188610,
             first_seconds: tuple[float, float] = (100, 100),
             peak_bytes: int = 1000) -> Fixture:
    size, streams_text = cell.rsplit('-s', 1)
    streams = int(streams_text)
    fixture = Fixture([], [])
    for seed, seconds in zip(SEEDS, first_seconds, strict=True):
        name = f'seed-{seed}-{cell}'
        fixture.cells.append(RetainedCell(name=name, cell=cell, seed=seed, size=size,
            streams=streams, summary=CellSummary(parameters=parameters, peak_cuda_allocated=peak_bytes)))
        for anchor in ('scripted_greedy', 'random'):
            for count in COUNTS:
                initial = .15 if anchor == 'scripted_greedy' else .45
                score = initial if count == 0 else (.6 if anchor == 'scripted_greedy' else .8)
                fixture.points.append(Point(name=name, cell=cell, size=size, streams=streams,
                    seed=seed, anchor=anchor, transitions=count, mean=score,
                    deal_blocks=[score]*13, roles={},
                    coordinates=TrainingCoordinates(stage_id='self-play', updates=count//(streams*64),
                        learner_transitions=count, training_seconds=count/COUNTS[1]*seconds),
                    checkpoint_sha256='0'*64, initialization_improvement=score-initial))
    return fixture


def _score(fixture: Fixture, seed: int, anchor: Anchor, count: int, score: float) -> None:
    initial = next(p.mean for p in fixture.points
                   if p.seed == seed and p.anchor == anchor and p.transitions == 0)
    for index, point in enumerate(fixture.points):
        if (point.seed, point.anchor, point.transitions) == (seed, anchor, count):
            fixture.points[index] = point.model_copy(update={'mean':score,
                'initialization_improvement':score-initial, 'deal_blocks':[score]*13})
            return
    raise AssertionError('Missing fixture point')


@pytest.mark.parametrize(('anchor', 'score'), [('scripted_greedy', .49), ('random', .74)])
def test_failure_in_one_seed_prevents_any_qualification(anchor: Anchor, score: float) -> None:
    fixture = _fixture()
    _score(fixture, SEEDS[1], anchor, COUNTS[-1], score)
    result = select_recipes(fixture.points, fixture.cells)
    assert result.preferred_by_frozen_rule is None
    assert not result.recipes[0].qualified
    shortfall = result.recipes[0].endpoint_shortfalls[1]
    observed = shortfall.greedy_below_threshold if anchor == 'scripted_greedy' else shortfall.random_below_threshold
    assert observed == pytest.approx(.01)


def test_seed_specific_initialization_is_required_even_with_good_absolute_scores() -> None:
    fixture = _fixture()
    for index, point in enumerate(fixture.points):
        if point.seed == SEEDS[1] and point.anchor == 'scripted_greedy':
            score = .45 if point.transitions == 0 else .6
            fixture.points[index] = point.model_copy(update={
                'mean':score, 'initialization_improvement':score-.45, 'deal_blocks':[score]*13})
    result = select_recipes(fixture.points, fixture.cells)
    assert result.preferred_by_frozen_rule is None
    assert result.recipes[0].endpoint_shortfalls[1].greedy_improvement_shortfall == pytest.approx(.05)


def test_regression_prevents_claiming_the_earlier_crossing() -> None:
    fixture = _fixture()
    _score(fixture, SEEDS[1], 'scripted_greedy', COUNTS[2], .4)
    preferred = select_recipes(fixture.points, fixture.cells).preferred_by_frozen_rule
    assert preferred is not None
    assert preferred.first_qualifying_transitions == 524288
    assert preferred.worst_seed_seconds_to_first_qualified == 400
    assert preferred.full_recipe_transitions == 524288


def test_choice_uses_worst_seed_time_then_size_within_ten_percent() -> None:
    a = _fixture('large-s64', parameters=200, first_seconds=(20, 100))
    b = _fixture('middle-s64', parameters=100, first_seconds=(109, 109))
    c = _fixture('small-s64', parameters=50, first_seconds=(111, 111))
    result = select_recipes(a.points+b.points+c.points, a.cells+b.cells+c.cells)
    assert result.preferred_by_frozen_rule is not None
    assert result.preferred_by_frozen_rule.cell == 'middle-s64'
    assert result.preferred_by_frozen_rule.full_recipe_transitions == 524288


def test_memory_breaks_equal_parameter_ties_in_the_time_band() -> None:
    a = _fixture('small-s64', first_seconds=(100, 100), peak_bytes=1000)
    b = _fixture('small-s256', first_seconds=(105, 105), peak_bytes=900)
    result = select_recipes(a.points+b.points, a.cells+b.cells)
    assert result.preferred_by_frozen_rule is not None
    assert result.preferred_by_frozen_rule.cell == 'small-s256'


@pytest.mark.parametrize('damage', ['missing-point', 'duplicate-point', 'missing-seed',
                                    'wrong-improvement', 'different-clock', 'parameters'])
def test_incomplete_or_mismatched_evidence_is_an_error(damage: str) -> None:
    fixture = _fixture()
    if damage == 'missing-point':
        fixture.points.pop()
    elif damage == 'duplicate-point':
        fixture.points.append(fixture.points[-1])
    elif damage == 'missing-seed':
        fixture.cells.pop()
    elif damage == 'wrong-improvement':
        fixture.points[-1] = fixture.points[-1].model_copy(update={'initialization_improvement':.9})
    elif damage == 'different-clock':
        point = fixture.points[-1]
        fixture.points[-1] = point.model_copy(update={'coordinates':point.coordinates.model_copy(
            update={'training_seconds':400.1})})
    else:
        fixture.cells[-1] = fixture.cells[-1].model_copy(update={
            'summary':CellSummary(parameters=999, peak_cuda_allocated=1000)})
    with pytest.raises(ValueError):
        select_recipes(fixture.points, fixture.cells)


@pytest.mark.parametrize('damage', ['unlisted', 'changed', 'missing', 'duplicate', 'escaped'])
def test_manifest_covers_every_evidence_file_and_rejects_damage(tmp_path: Path, damage: str) -> None:
    root = tmp_path/'evidence'
    root.mkdir()
    artifact = root/'monitor.json'
    artifact.write_text('{}\n')
    entry = {'path':'monitor.json', 'bytes':3,
             'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'files':[entry]}))
    assert verify_manifest(root) == {'monitor.json'}
    if damage == 'unlisted':
        (root/'replay.json').write_text('{}\n')
    elif damage == 'changed':
        artifact.write_text('[]\n')
    elif damage == 'missing':
        artifact.unlink()
    elif damage == 'duplicate':
        manifest.write_text(json.dumps({'files':[entry, entry]}))
    else:
        (tmp_path/'outside.json').write_text('{}\n')
        manifest.write_text(json.dumps({'files':[{'path':'../outside.json',
            'bytes':3, 'sha256':entry['sha256']}]}))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_manifest(root)


def test_frontier_preserves_real_tradeoffs_across_both_anchors() -> None:
    def point(name: str, seconds: float, greedy: float, random: float) -> Tradeoff:
        return Tradeoff(cell=name,size='small',streams=64,transitions=524288,
            seed_greedy_scores={13591:greedy},seed_random_scores={13591:random},
            seed_inclusive_seconds={13591:seconds},seed_active_seconds={13591:seconds},
            worst_seed_greedy_score=greedy,worst_seed_random_score=random,
            worst_seed_inclusive_seconds=seconds,worst_seed_active_seconds=seconds)

    records = [point('fast',10,.5,.7),point('strong',20,.6,.8),
               point('dominated',15,.4,.65),point('equal-time-worse',20,.55,.75),
               point('greedy-tradeoff',12,.7,.6),point('same-as-fast',10,.5,.7)]
    assert {record.cell for record in mark_frontier(records) if record.pareto} == {
        'fast','strong','greedy-tradeoff','same-as-fast'}


def test_tradeoff_summary_keeps_seed_values_and_uses_conservative_observed_extrema() -> None:
    fixture = _fixture(first_seconds=(100,120))
    _score(fixture,SEEDS[1],'scripted_greedy',COUNTS[-1],.55)
    points = [point.model_copy(update={'coordinates':point.coordinates.model_copy(
        update={'active_training_seconds':point.coordinates.training_seconds*.8})})
        for point in fixture.points]
    endpoint = next(record for record in build_tradeoffs(points) if record.transitions==COUNTS[-1])
    assert endpoint.seed_greedy_scores == {13591:.6,13592:.55}
    assert endpoint.worst_seed_greedy_score == .55
    assert endpoint.worst_seed_inclusive_seconds == 480
    assert endpoint.worst_seed_active_seconds == 384
    with pytest.raises(ValueError,match='both anchors'):
        build_tradeoffs(points[:-1])
    with pytest.raises(ValueError,match='Missing active training clock'):
        build_tradeoffs(fixture.points)
