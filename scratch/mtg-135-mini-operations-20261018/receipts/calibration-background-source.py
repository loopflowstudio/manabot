"""Admit the frozen evaluation runtime on the Mini using retained controls.

Runs only the previously scored calibration checkpoint and deal. The full
protocol017 panel is a separate explicit admission after this proof succeeds.
"""

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import time

from experiments.runners.gpu_job import (
    Deadlines, Worker, write_json, write_retained_manifest,
)
from experiments.runners.parallel_commands import Command, run_parallel
from experiments.runners.parallel_evaluation_probe import panel_evidence
from evaluation_runtime import RECEIPT, verify_current

STUDY = Path('.runs/grid-007/learning-017')
ROOT = STUDY / 'mini-calibration'
CONTROLS = Path('.runs/grid-007/local-evaluation-parallel')
AMENDMENT = Path('scratch/mtg-135-mini-evaluation-20261018.md')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    if platform.node().split('.')[0] != 'mini-heart':
        raise RuntimeError('This admission is restricted to mini-heart.local')
    if (ROOT / 'admission.json').exists():
        raise RuntimeError('Control already admitted; inspect evidence without retry')
    ROOT.mkdir(exist_ok=True)
    verify_current()
    transfer = STUDY / 'mini-transfer/control-verification.json'
    proof = json.loads(transfer.read_text())
    assert proof['all_files_verified'] and proof['source_verified']
    references = {
        'scripted_greedy': CONTROLS / 'serial-scripted_greedy',
        'random': Path('.runs/grid-007/local-evaluation-probe/random'),
    }
    serial = {anchor: panel_evidence(path, anchor)
              for anchor, path in references.items()}
    run = Path('.runs/grid-007/feasibility/results/evaluation-cost-calibration/large/training/run')
    interpreter = str(Path('.venv/bin/python').absolute())
    commands: list[Command] = []
    for anchor in references:
        (ROOT / f'protocol-{anchor}.json').write_bytes(
            (CONTROLS / f'protocol-{anchor}.json').read_bytes()
        )
    for copy in range(2):
        for anchor in references:
            name = f'parallel-{copy}-{anchor}'
            commands.append(Command(name, (
                interpreter, '-m', 'experiments.runners.batching_evaluation',
                '--run', str(run), '--transitions', '0',
                '--protocol', str(ROOT / f'protocol-{anchor}.json'),
                '--out', str(ROOT / name),
            ), 900))
    write_json(ROOT / 'admission.json', {
        'observed_at': datetime.now(UTC).isoformat(),
        'host': platform.node(), 'pid': os.getpid(),
        'purpose': 'cross-host runtime parity and timing; no model selection',
        'amendment_sha256': digest(AMENDMENT),
        'controller_sha256': digest(Path(__file__)),
        'transfer_verification_sha256': digest(transfer),
        'runtime_receipt_sha256': digest(RECEIPT),
        'commands': 4, 'games': 32, 'workers': 4,
        'command_seconds': 900, 'work_seconds': 1200, 'retention_seconds': 240,
        'gpu_cost_usd': 0,
        'serial_controls': {key: asdict(value) for key, value in serial.items()},
    })
    worker = Worker(ROOT, Deadlines(time.time() + 1440))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    error: str | None = None
    started = time.monotonic()
    try:
        code = run_parallel(worker, tuple(commands), workers=4)
        elapsed = time.monotonic() - started
        if code:
            raise RuntimeError(f'Mini control exited {code}; no automatic retry')
        parallel: dict[str, object] = {}
        for command in commands:
            anchor = 'scripted_greedy' if command.name.endswith('scripted_greedy') else 'random'
            evidence = panel_evidence(ROOT / command.name, anchor)
            if evidence.decision_digest != serial[anchor].decision_digest:
                raise ValueError(f'Decisions or identities differ: {command.name}')
            parallel[command.name] = asdict(evidence)
        write_json(ROOT / 'comparison.json', {
            'status': 'completed', 'host': platform.node(),
            'parallel_seconds': elapsed, 'games': 32, 'workers': 4,
            'all_decisions_identities_and_replays_equal': True,
            'gpu_cost_usd': 0, 'panels': parallel,
        })
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        worker.phase('failed' if error else 'completed', error=error,
                     seconds=time.monotonic() - started)
        write_retained_manifest(ROOT)


if __name__ == '__main__':
    main()
