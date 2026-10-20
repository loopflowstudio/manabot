"""Exercise longer-study integrity checks on retained short-study evidence.

This validates the analysis machinery, not any new protocol017 outcome.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import re

import torch

from analyze_learning import digest, verify_manifest
from evaluation_runtime import verify_current
from experiments.runners.batching_evaluation import select_checkpoints
from manabot.training.monitor_evaluation import MonitorResult, _manifest


def main() -> None:
    torch.set_num_threads(1)
    verify_current()
    root = Path('.runs/grid-007')
    cohorts = ('early-learning-013', 'early-learning-013-added')
    manifests = {cohort:len(verify_manifest(root/cohort)) for cohort in cohorts}
    samples = (
        (cohorts[0], 'small-s4', root/'feasibility/results/small-s4/study/training/run'),
        (cohorts[1], 'large-shallow-s64',
         root/'grouped-feasibility/large-shallow/results/large-shallow-s64/study/training/run'),
    )
    checked: list[str] = []
    for cohort, cell, training in samples:
        selection = select_checkpoints(training, (0, 131072))
        for anchor in ('scripted_greedy', 'random'):
            for checkpoint in selection.points:
                panel = root/cohort/'panels'/f'{cell}-{anchor}'/f'transitions-{checkpoint.coordinates.learner_transitions}'
                observed = MonitorResult.model_validate_json((panel/'monitor.json').read_text())
                expected = _manifest(selection.run, checkpoint.artifact,
                                     checkpoint.coordinates, observed.protocol)
                assert observed.regime_digest == selection.run.regime_digest
                assert observed.key == expected.key and observed.candidate == expected.candidate
                assert observed.opponent == expected.opponent
                checked.append(str(panel))
    inputs = [Path(__file__), root/'analyze_learning.py', root/'learning_selection.py',
              root/'learning_figures.py', root/'test_learning_analysis.py',
              root/'learning-017/analysis-readiness-tests.log']
    test_log = (root/'learning-017/analysis-readiness-tests.log').read_text()
    match = re.search(r'(?m)^(\d+) passed in [\d.]+s$',test_log)
    assert match is not None and 'failed' not in test_log and 'ERROR' not in test_log
    result = {'observed_at':datetime.now(UTC).isoformat(), 'status':'passed',
        'scope':'Analysis readiness using old retained evidence and synthetic selection failures; no protocol017 playing-strength result',
        'unit_test_result':f'{match.group(1)} passed', 'complete_manifest_coverage':manifests,
        'freshly_reconstructed_checkpoint_anchor_registrations':checked,
        'inputs':{str(path):digest(path) for path in inputs}}
    (root/'learning-017/analysis-readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','manifests':manifests,'registrations':len(checked)}))


if __name__ == '__main__':
    main()
