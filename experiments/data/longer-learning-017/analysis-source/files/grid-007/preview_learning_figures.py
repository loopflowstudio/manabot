"""Check report rendering with real earlier screen013 data, never synthetic results."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from learning_figures import build_tradeoffs, curves, endpoint_tradeoffs
from learning_selection import Point
from manabot.training.monitor_evaluation import MonitorResult

ROOT = Path('.runs/grid-007')
OUT = ROOT/'learning-017/report-rendering-check'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    points: list[Point] = []
    sources: dict[str,str] = {}
    for cohort in ('early-learning-013','early-learning-013-added'):
        for path in sorted((ROOT/cohort/'panels').glob('*/transitions-*/monitor.json')):
            result = MonitorResult.model_validate_json(path.read_text())
            assert result.status == 'completed' and result.error is None and result.score is not None
            command = path.parent.parent
            initial = MonitorResult.model_validate_json((command/'transitions-0/monitor.json').read_text())
            assert initial.score is not None
            cell = command.name.removesuffix('-'+result.protocol.opponent)
            size,streams = cell.rsplit('-s',1)
            blocks = [sum(float(row.score_a) for row in result.rows if row.deal_seed==deal)/8
                      for deal in result.protocol.deal_seeds]
            points.append(Point(name=cell,cell=cell,size=size,streams=int(streams),
                seed=result.training_seed,anchor=result.protocol.opponent,
                transitions=result.coordinates.learner_transitions,mean=result.score.mean,
                deal_blocks=blocks,roles={},coordinates=result.coordinates,
                checkpoint_sha256=result.artifact['sha256'],
                initialization_improvement=result.score.mean-initial.score.mean))
            sources[str(path)] = digest(path)
    assert len(points)==92 and {point.seed for point in points}=={13571}
    OUT.mkdir(exist_ok=True)
    (OUT/'earlier-screen-points.json').write_text(json.dumps(
        [p.model_dump(mode='json') for p in points],indent=2)+'\n')
    records = build_tradeoffs(points)
    for clock in ('transitions','inclusive','active'):
        curves(points,clock=clock,title=f'Earlier screen013 — renderer check ({clock})',
               target=OUT/f'preview-{clock}.svg')
    endpoint_tradeoffs(records,title='Earlier screen013 — renderer check (one training seed)',
                       target=OUT/'preview-endpoints.svg')
    receipt = {'observed_at':datetime.now(UTC).isoformat(),
        'scope':'Rendering check on real earlier short-screen data; not protocol017 scores',
        'points':len(points),'training_seeds':[13571],
        'source_monitor_sha256':sources,
        'plotter_sha256':digest(Path(__file__).with_name('learning_figures.py')),
        'outputs':{path.name:digest(path) for path in OUT.iterdir() if path.suffix in ('.svg','.png')}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'rendered':str(OUT),'old_screen_points':len(points)}))


if __name__ == '__main__':
    main()
