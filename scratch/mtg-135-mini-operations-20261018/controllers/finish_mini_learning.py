"""Wait for the admitted Mini panel, then audit and render its retained evidence.

This one-shot owner never launches games, retries the panel or publishes a
recommendation. It leaves the resulting tables/figures for scientific review.
"""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
import traceback

from experiments.runners.gpu_job import (
    Deadlines, Worker, write_json, write_retained_manifest,
)
from retain_analysis_source import verify_current as verify_source_archive

STUDY = Path('.runs/grid-007/learning-017')
ROOT = STUDY / 'mini-analysis-worker'
LABEL = 'gg.manabot.mtg135.protocol017-evaluation'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    if platform.node().split('.')[0] != 'mini-heart':
        raise RuntimeError('Analysis owner is restricted to the Mini')
    if (ROOT / 'admission.json').exists():
        raise RuntimeError('Analysis owner already admitted; inspect without retry')
    for path in (STUDY / 'analysis', STUDY / 'report'):
        if path.exists():
            raise RuntimeError(f'Retained output already exists: {path}')
    ROOT.mkdir(exist_ok=True)
    verify_source_archive()
    source_receipt = STUDY / 'analysis-source.json'
    source_sha256 = digest(source_receipt)
    admission_path = STUDY / 'evaluation/admission.json'
    admission = json.loads(admission_path.read_text())
    started = datetime.fromisoformat(admission['observed_at']).timestamp()
    wait_until = started + admission['work_seconds'] + admission['retention_seconds'] + 120
    limits = {'analysis_seconds': 900, 'report_seconds': 600, 'retention_seconds': 240}
    worker = Worker(ROOT, Deadlines(wait_until + 1740))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    write_json(ROOT / 'admission.json', {
        'observed_at': datetime.now(UTC).isoformat(), 'pid': os.getpid(),
        'purpose': 'Audit and render after successful full panel; review remains',
        'evaluation_admission_sha256': digest(admission_path),
        'controller_sha256': digest(Path(__file__)), 'wait_until_unix': wait_until,
        'analysis_source_receipt_sha256': source_sha256,
        'new_games': 0, 'new_rentals': 0, 'automatic_retry': False, **limits,
    })
    error: str | None = None
    try:
        worker.phase('waiting_for_full_panel', wait_until=wait_until)
        while True:
            if worker.stopping:
                raise RuntimeError('Analysis owner explicitly stopped')
            if time.time() >= wait_until:
                raise TimeoutError('Panel did not finish inside its retained deadline')
            job = subprocess.run(['launchctl', 'list', LABEL], capture_output=True, text=True, timeout=10)
            if job.returncode:
                raise RuntimeError('Evaluation job is unavailable; no automatic recovery')
            if '"PID"' not in job.stdout:
                if '"LastExitStatus" = 0;' not in job.stdout:
                    raise RuntimeError('Evaluation process failed; partial evidence retained')
                break
            time.sleep(30)
        status = json.loads((STUDY / 'evaluation/worker-status.json').read_text())
        if status['phase'] != 'completed' or status.get('error') is not None:
            raise RuntimeError('Full evaluation did not complete; no partial analysis')
        if digest(source_receipt) != source_sha256:
            raise RuntimeError('Analysis source changed after admission')
        verify_source_archive()
        interpreter = str(Path('.venv/bin/python').absolute())
        for name, script, seconds in (
            ('audit', 'analyze_learning.py', limits['analysis_seconds']),
            ('report', 'render_learning_report.py', limits['report_seconds']),
        ):
            code = worker.command(name, [interpreter, str(Path(__file__).with_name(script))], seconds)
            if code:
                raise RuntimeError(f'{name} exited {code}; no automatic retry')
        write_json(ROOT / 'outputs.json', {
            'status': 'scientific and visual review pending',
            'analysis_sha256': digest(STUDY / 'analysis/results.json'),
            'verification_sha256': digest(STUDY / 'analysis/verification.json'),
            'report_manifest_sha256': digest(STUDY / 'report/manifest.json'),
            'published': False, 'mtg122_handoff': False,
        })
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        raise SystemExit(1) from None
    finally:
        worker.phase('failed' if error else 'review_pending', error=error)
        write_retained_manifest(ROOT)


if __name__ == '__main__':
    main()
