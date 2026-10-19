"""Admit the frozen protocol017 CPU evaluation once, after full retention audit."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import signal
import time

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
    verify_current()
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
    limits = plan['evaluation']
    assert limits == {'workers':4,'command_seconds':4500,'work_seconds':25800,
                      'retention_seconds':240,'gpu_cost_usd':0}
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
        raise
    finally:
        worker.phase('failed' if error else 'completed',error=error,seconds=time.monotonic()-started)
        write_retained_manifest(ROOT)


if __name__ == '__main__':
    main()
