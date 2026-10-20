"""Admit the full frozen protocol017 panel on the verified Mini runtime.

This is the explicit placement recovery after Jack stopped the laptop attempt.
The scientific allocation is unchanged; a separate Mini plan binds calibrated
operational deadlines and the merged supervisor logging repair.
"""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time
import traceback

from experiments.runners.batching_evaluation import select_checkpoints
from experiments.runners.gpu_job import Deadlines, Worker, write_json, write_retained_manifest
from experiments.runners.parallel_commands import Command, run_parallel
from evaluation_runtime import RECEIPT as RUNTIME_RECEIPT, verify_current

STUDY = Path('.runs/grid-007/learning-017')
ROOT = STUDY/'evaluation'
DEALS = tuple(range(1357201000,1357201013))


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main() -> None:
    if platform.node().split('.')[0] != 'mini-heart':
        raise RuntimeError('Evaluation is restricted to mini-heart.local')
    verify_current()
    mini_path = STUDY/'mini-plan.json'
    mini = json.loads(mini_path.read_text())
    assert mini['status'] == 'frozen' and mini['games'] == 18304
    for name, expected in mini['inputs'].items():
        assert digest(Path(name)) == expected, name
    transfer = json.loads((STUDY/'mini-transfer/verification.json').read_text())
    assert transfer['scope'] == 'full' and transfer['all_files_verified']
    assert transfer['source_verified'] and transfer['files'] == 219
    calibration = json.loads((STUDY/'mini-transfer/calibration-standard-verification.json').read_text())
    assert calibration['status'] == 'verified'
    assert calibration['games'] == 32 and calibration['all_decisions_identities_and_replays_equal']
    plan = json.loads((STUDY/'allocation.json').read_text())
    evidence = json.loads((STUDY/'verified-results.json').read_text())
    assert evidence['allocation_sha256'] == digest(STUDY/'allocation.json')
    assert evidence['provider_absent'] and evidence['complete_cohort']
    cells = evidence['cells']
    assert len(cells) == len({c['name'] for c in cells}) == 22
    assert not evidence['incomplete']
    assert json.loads(Path('.runs/grid-007/local-evaluation-parallel/verification.json').read_text())['all_decisions_identities_and_replays_equal']
    assert not (ROOT/'admission.json').exists(), 'Panel already admitted; inspect retained outcomes without replay'
    ROOT.mkdir(exist_ok=True)
    protocol = Path('experiments/protocols/longer-learning-017.md')
    protocol_hash = digest(protocol)
    assert protocol_hash == plan['inputs'][str(protocol)]
    (ROOT/'protocol.md').write_bytes(protocol.read_bytes())
    for anchor in ('scripted_greedy','random'):
        write_json(ROOT/f'protocol-{anchor}.json',{'purpose':'frozen-study-evaluation',
            'study_protocol_sha256':protocol_hash,'deal_seeds':list(DEALS),
            'include_mirrors':True,'opponent':anchor,'game_seconds':120,'max_commands':10000})
    limits = mini['evaluation']
    assert limits['workers'] == 4 and limits['gpu_cost_usd'] == 0
    assert limits['retention_seconds'] == 240
    assert 0 < limits['command_seconds'] <= limits['work_seconds'] <= 172800
    points = (0,*plan['checkpoint_transitions'])
    assert points == (0,131072,262144,524288)
    interpreter = str(Path('.venv/bin/python').absolute())
    commands: list[Command] = []
    for cell in cells:
        retained = Path(cell['directory'])
        proof = json.loads((retained/'verification.json').read_text())
        assert proof == cell['verification']
        assert proof['all_manifest_files_verified'] and proof['all_raw_checkpoints_finite']
        directory = retained/'study/training/run'
        selected = select_checkpoints(directory,points)
        assert selected.run.seed == cell['seed']
        assert [p.model_dump(mode='json') for p in selected.points] == cell['checkpoints']
        for anchor in ('scripted_greedy','random'):
            name = f'{cell["name"]}-{anchor}'
            commands.append(Command(name,(interpreter,'-m','experiments.runners.batching_evaluation',
                '--run',str(directory),'--transitions',*(str(p) for p in points),
                '--protocol',str(ROOT/f'protocol-{anchor}.json'),
                '--out',str(ROOT/'panels'/name)),limits['command_seconds']))
    write_json(ROOT/'admission.json',{'observed_at':datetime.now(UTC).isoformat(),
        'host':platform.node(),'pid':os.getpid(),'mini_plan_sha256':digest(mini_path),
        'placement_recovery':'Full unchanged panel after explicit laptop cancellation; no mixed partial scores',
        'protocol_sha256':protocol_hash,'allocation_sha256':digest(STUDY/'allocation.json'),
        'verified_results_sha256':digest(STUDY/'verified-results.json'),
        'evaluation_runtime_sha256':digest(RUNTIME_RECEIPT),
        'controller_sha256':digest(Path(__file__)),'cells':cells,'commands':len(commands),
        'checkpoint_transitions':points,'deal_seeds':DEALS,'games':18304,**limits})
    worker = Worker(ROOT,Deadlines(time.time()+limits['work_seconds']+limits['retention_seconds']))
    signal.signal(signal.SIGTERM,worker.stop)
    signal.signal(signal.SIGINT,worker.stop)
    error: str | None = None
    started = time.monotonic()
    try:
        code = run_parallel(worker,tuple(commands),workers=limits['workers'])
        if code:
            raise RuntimeError(f'Panel exited {code}; partial results retained without retry')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        # Retain the traceback before the final manifest hashes controller.log.
        traceback.print_exc()
        raise SystemExit(1) from None
    finally:
        worker.phase('failed' if error else 'completed',error=error,seconds=time.monotonic()-started)
        write_retained_manifest(ROOT)


if __name__ == '__main__':
    main()
