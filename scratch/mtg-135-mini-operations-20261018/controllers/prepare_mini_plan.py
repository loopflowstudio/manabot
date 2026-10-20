"""Freeze Mini deadlines after parity and complete checkpoint-transfer checks.

Scale the previously declared command and cohort work limits by the largest
matched-control slowdown, add 25% engineering headroom and round up to five
minutes. This changes execution placement and bounds, never the scientific grid.
"""

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform

STUDY = Path('.runs/grid-007/learning-017')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    if platform.node().split('.')[0] != 'mini-heart':
        raise RuntimeError('The final operational plan must bind Mini receipts')
    target = STUDY / 'mini-plan.json'
    if target.exists() or (STUDY / 'evaluation/admission.json').exists():
        raise RuntimeError('Mini plan or panel already admitted; inspect without retry')
    proof = json.loads((STUDY / 'mini-transfer/verification.json').read_text())
    assert proof['scope'] == 'full' and proof['files'] == 219
    assert proof['source_verified'] and proof['all_files_verified']
    parity_path = STUDY / 'mini-transfer/calibration-standard-verification.json'
    parity = json.loads(parity_path.read_text())
    assert parity['status'] == 'verified' and parity['games'] == 32
    assert parity['all_decisions_identities_and_replays_equal']
    assert parity['all_manifest_files_verified'] and parity['no_live_control_process']
    original_path = Path('.runs/grid-007/local-evaluation-parallel/comparison.json')
    current_path = STUDY / 'mini-calibration-standard/comparison.json'
    old = json.loads(original_path.read_text())
    new = json.loads(current_path.read_text())
    assert digest(current_path) == parity['comparison_sha256']
    assert old['panels'].keys() == new['panels'].keys()
    ratios = {
        name: panel['evaluator_seconds'] / old['panels'][name]['evaluator_seconds']
        for name, panel in new['panels'].items()
    }
    slowdown = max(1.0, *ratios.values())
    old_limits = json.loads((STUDY / 'allocation.json').read_text())['evaluation']
    assert old_limits['command_seconds'] == 4500 and old_limits['work_seconds'] == 25800
    limits = {
        'workers': 4, 'gpu_cost_usd': 0, 'retention_seconds': 240,
        'command_seconds': math.ceil(4500 * slowdown * 1.25 / 300) * 300,
        'work_seconds': math.ceil(25800 * slowdown * 1.25 / 300) * 300,
    }
    assert limits['command_seconds'] <= limits['work_seconds'] <= 172800
    inputs = [
        STUDY / name for name in (
            'allocation.json', 'verified-results.json', 'evaluation-runtime.json',
            'mini-transfer/verification.json',
            'mini-transfer/calibration-standard-verification.json',
            'mini-transfer/calibration-standard-launch.json',
            'mini-transfer/worker-logging-fix.json',
            'mini-transfer/laptop-cancellation.json',
            'mini-transfer/failed-control-retention.json',
            'mini-calibration/admission.json', 'mini-calibration/operator-stop.json',
            'mini-calibration-standard/admission.json',
            'mini-calibration-standard/manifest.json',
        )
    ]
    inputs.extend((
        original_path, current_path, parity_path,
        Path('experiments/runners/gpu_job.py'),
        Path('experiments/protocols/longer-learning-017.md'),
        Path('scratch/mtg-135-mini-evaluation-20261018.md'),
        Path('scratch/mtg-135-mini-control-repair-20261018.md'),
        Path('.runs/grid-007/evaluate_learning_on_mini.py'),
        Path('.runs/grid-007/calibrate_mini_evaluation.py'),
        Path('.runs/grid-007/verify_mini_transfer.py'), Path(__file__),
    ))
    plan = {
        'status': 'frozen', 'observed_at': datetime.now(UTC).isoformat(),
        'host': platform.node(), 'process_type': 'Standard',
        'purpose': 'Full unchanged protocol017 evaluation after explicit laptop cancellation',
        'games': 18304, 'commands': 44, 'panels': 176,
        'scientific_source': '3c3681d2b26a7fc8167928fa63fe24a6cd3d058a',
        'evaluation': limits,
        'deadline_method': 'Original command/work bounds multiplied by maximum matched-control slowdown and 1.25, rounded up to 300 seconds',
        'matched_control_slowdowns': ratios, 'selected_slowdown': slowdown,
        'timing_scope': 'Engineering limits, not statistical guarantees or predicted completion times',
        'inputs': {str(path): digest(path) for path in inputs},
        'failure_policy': 'Retain partial evidence and stop; no automatic retries or shortened panels',
        'laptop_evaluation': 'prohibited', 'new_rentals': 0, 'new_s3_downloads': 0,
    }
    with target.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'plan': str(target), 'sha256': digest(target),
                      'evaluation': limits, 'selected_slowdown': slowdown}))


if __name__ == '__main__':
    main()
