"""Verify protocol017 game evidence and apply its frozen selection rule.

Deal resampling is paired within each training seed. The two seed estimates
remain separate; these intervals do not estimate population training variance.
"""

import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from experiments.runners.batching_evaluation import select_checkpoints
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult, _manifest, _summarize
from evaluation_runtime import RECEIPT as RUNTIME_RECEIPT, RuntimeIdentity, verify_current
from learning_figures import build_tradeoffs
from learning_selection import COUNTS, SEEDS, Point, RetainedCell, select_recipes

STUDY = Path('.runs/grid-007/learning-017')
ROOT = STUDY/'evaluation'
OUT = STUDY/'analysis'
DEALS = tuple(range(1357201000,1357201013))
SHAPES = ('small','small-deep','middle','large-shallow','large')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def interval(values: NDArray[np.float64], indices: NDArray[np.int64]) -> list[float]:
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates,(.025,.975))]


def verify_manifest(root: Path) -> set[str]:
    """Require byte integrity and complete coverage of the retained evidence tree."""
    manifest = json.loads((root/'manifest.json').read_text())
    verified: set[str] = set()
    for entry in manifest['files']:
        path = root/entry['path']
        if not path.resolve().is_relative_to(root.resolve()) or entry['path'] in verified:
            raise ValueError('Manifest path escapes the evidence directory or is duplicated')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Manifest bytes differ: {entry["path"]}')
        verified.add(entry['path'])
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
              and path not in (root/'manifest.json',root/'etude-result.json')}
    if verified != actual:
        raise ValueError('Manifest does not cover the complete retained evidence tree')
    return verified


def main() -> None:
    torch.set_num_threads(1)
    verify_current()
    status = json.loads((ROOT/'worker-status.json').read_text())
    assert status['phase'] == 'completed' and status['error'] is None
    admission = json.loads((ROOT/'admission.json').read_text())
    plan = json.loads((STUDY/'allocation.json').read_text())
    assert admission['allocation_sha256'] == digest(STUDY/'allocation.json')
    assert admission['verified_results_sha256'] == digest(STUDY/'verified-results.json')
    assert admission['evaluation_runtime_sha256'] == digest(RUNTIME_RECEIPT)
    expected_runtime = RuntimeIdentity.model_validate_json(RUNTIME_RECEIPT.read_text())
    assert admission['protocol_sha256'] == digest(ROOT/'protocol.md')
    assert admission['protocol_sha256'] == plan['inputs']['experiments/protocols/longer-learning-017.md']
    assert tuple(admission['deal_seeds']) == DEALS
    assert tuple(admission['checkpoint_transitions']) == COUNTS
    verified = verify_manifest(ROOT)
    events = [json.loads(line) for line in (ROOT/'controller.log').read_text().splitlines()
              if line.startswith('{')]
    finished = [e for e in events if e.get('phase') == 'command_finished']
    completed = {e['command']:e['exit_code'] for e in finished}
    assert len(finished) == len(completed) == admission['commands'] == 44
    assert all(code == 0 for code in completed.values())
    indices = np.random.default_rng(135917001).integers(0,len(DEALS),(10000,len(DEALS)))
    points: list[Point] = []
    games = 0
    for cell in admission['cells']:
        training = Path(cell['directory'])/'study/training/run'
        selection = select_checkpoints(training,COUNTS)
        assert selection.run.seed == cell['seed'] and cell['seed'] in SEEDS
        for anchor in ('scripted_greedy','random'):
            command = f'{cell["name"]}-{anchor}'
            assert completed[command] == 0
            panel = ROOT/'panels'/command
            saved = json.loads((panel/'selection.json').read_text())
            assert saved['run_sha256'] == digest(training/'run.json')
            protocol = MonitorProtocol.model_validate_json((ROOT/f'protocol-{anchor}.json').read_text())
            initialization: NDArray[np.float64] | None = None
            for checkpoint,count in zip(selection.points,COUNTS,strict=True):
                directory = panel/f'transitions-{count}'
                result = MonitorResult.model_validate_json((directory/'monitor.json').read_text())
                assert result.status == 'completed' and result.error is None
                assert result.protocol == protocol and tuple(protocol.deal_seeds) == DEALS
                assert protocol.study_protocol_sha256 == admission['protocol_sha256']
                assert result.training_seed == cell['seed'] and result.run_id == selection.run.id
                assert result.regime_digest == selection.run.regime_digest
                assert result.evaluation_identities == expected_runtime.evaluation_identities
                assert result.coordinates == checkpoint.coordinates and result.artifact == checkpoint.artifact
                expected = _manifest(selection.run,checkpoint.artifact,checkpoint.coordinates,protocol)
                assert result.key == expected.key and result.candidate == expected.candidate
                assert result.opponent == expected.opponent
                assert len(result.rows) == result.expected_games == 104
                rates = (result.score,result.win,result.draw)
                _summarize(result)
                assert result.status == 'completed' and rates == (result.score,result.win,result.draw)
                assert result.score is not None
                roles: dict[str,list[float]] = {}
                for row in result.rows:
                    extra = row.model_extra or {}
                    trace = directory/row.trace_path
                    assert str(trace.relative_to(ROOT)) in verified
                    assert digest(trace) == extra['trace_shard_sha256']
                    assert row.valid and row.replay_passed and row.score_a is not None
                    decks = extra['seat_decks']
                    seat = extra['player_a_seat']
                    role = f'{decks[seat]} vs {decks[1-seat]}'
                    roles.setdefault(role,[]).append(float(row.score_a))
                replays = list(directory.glob('deal-*/replay.json'))
                assert len(replays) == len(DEALS)
                for path in replays:
                    replay = json.loads(path.read_text())
                    assert replay['passed'] and replay['games'] == 8
                    assert all(value == 0 for key,value in replay.items()
                               if key.endswith('mismatches') or key in ('missing_decisions','private_exposures'))
                assert len(roles) == 4 and all(len(values) == 26 for values in roles.values())
                blocks = np.asarray([sum(float(row.score_a) for row in result.rows
                                         if row.deal_seed == deal)/8 for deal in DEALS],dtype=np.float64)
                assert abs(float(blocks.mean())-result.score.mean) < 1e-12
                if initialization is None:
                    initialization = blocks.copy()
                delta = blocks-initialization
                points.append(Point(name=cell['name'],cell=cell['cell'],size=cell['size'],
                    streams=cell['streams'],seed=cell['seed'],anchor=anchor,transitions=count,
                    mean=result.score.mean,deal_blocks=blocks.tolist(),
                    roles={key:float(np.mean(values)) for key,values in roles.items()},
                    coordinates=result.coordinates,checkpoint_sha256=result.artifact['sha256'],
                    initialization_improvement=float(delta.mean()),
                    paired_deal_improvement_95=interval(delta,indices)))
                games += len(result.rows)
    assert len(points) == 176 and games == admission['games'] == 18304
    indexed = {(p.cell,p.seed,p.anchor,p.transitions):p for p in points}
    assert len(indexed) == len(points)
    comparisons: list[dict[str,object]] = []
    pairs = [(f'{size}-s256',f'{size}-s64','batching') for size in SHAPES]
    pairs += [(f'{a}-s{streams}',f'{b}-s{streams}','capacity')
              for streams in (64,256) for a,b in combinations(SHAPES,2)]
    pairs += [('small-s64','small-s4','reference'),('small-s256','small-s4','reference')]
    for a,b,kind in pairs:
        for seed in SEEDS:
            for anchor in ('scripted_greedy','random'):
                for count in COUNTS:
                    left,right = indexed[a,seed,anchor,count],indexed[b,seed,anchor,count]
                    delta = np.asarray(left.deal_blocks)-np.asarray(right.deal_blocks)
                    left_initial = np.asarray(indexed[a,seed,anchor,0].deal_blocks)
                    right_initial = np.asarray(indexed[b,seed,anchor,0].deal_blocks)
                    improvement_delta = delta-(left_initial-right_initial)
                    comparisons.append({'kind':kind,'left':a,'right':b,'seed':seed,
                        'anchor':anchor,'transitions':count,'difference':float(delta.mean()),
                        'paired_deal_difference_95':interval(delta,indices),
                        'initialization_relative_difference':float(improvement_delta.mean()),
                        'paired_deal_improvement_difference_95':interval(improvement_delta,indices)})
    selection = select_recipes(points,[RetainedCell.model_validate(c) for c in admission['cells']])
    tradeoffs = build_tradeoffs(points)
    OUT.mkdir(exist_ok=True)
    proof = {'status':'verified','manifest_files':len(verified),'completed_commands':44,
        'completed_panels':176,'completed_replayed_games':games,'training_seeds':SEEDS,
        'protocol_sha256':admission['protocol_sha256'],'root_manifest_sha256':digest(ROOT/'manifest.json'),
        'analyzer_sha256':digest(Path(__file__)),'evaluation_wall_seconds':status['seconds'],
        'selection_sha256':digest(Path(__file__).with_name('learning_selection.py')),
        'tradeoff_analysis_sha256':digest(Path(__file__).with_name('learning_figures.py')),
        'limits':'Two training seeds; paired deal-bootstrap intervals are descriptive, not population training-variance or multiple-comparison guarantees'}
    result = {'verification':proof,'points':[p.model_dump(mode='json') for p in points],
        'comparisons':comparisons,**selection.model_dump(mode='json'),
        'score_time_tradeoffs':[record.model_dump(mode='json') for record in tradeoffs],
        'tradeoff_interpretation':'Pareto minimizes worst observed seed inclusive time and maximizes worst observed seed scores against both anchors across retained checkpoints. These cohort extrema are not population confidence bounds; they do not replace the frozen competence selection rule.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))
    print(json.dumps({'preferred_by_frozen_rule':result['preferred_by_frozen_rule']}))


if __name__ == '__main__':
    main()
