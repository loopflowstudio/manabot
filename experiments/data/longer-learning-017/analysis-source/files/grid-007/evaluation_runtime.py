"""Bind later CPU scoring to the already validated local evaluation runtime."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from pydantic import BaseModel, JsonValue
import torch

from experiments.runners.batching_evaluation import select_checkpoints
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult, _manifest

BASELINE = Path('.runs/grid-007/local-evaluation-parallel/serial-scripted_greedy/transitions-0/monitor.json')
SOURCE = '3c3681d2b26a7fc8167928fa63fe24a6cd3d058a'
RECEIPT = Path('.runs/grid-007/learning-017/evaluation-runtime.json')
SCOPE = ('manabot','managym','content','pyproject.toml','uv.lock',
         'experiments/runners/batching_evaluation.py')


class RuntimeIdentity(BaseModel):
    scientific_source: str
    baseline_monitor_sha256: str
    python: str
    torch: str
    numpy: str
    evaluation_identities: dict[str,JsonValue]


def current() -> RuntimeIdentity:
    """Check tracked runtime bytes and exercise the ordinary admission manifest."""
    result = subprocess.run(['git','diff','--quiet',SOURCE,'--',*SCOPE],check=False)
    if result.returncode:
        raise RuntimeError('Local scientific source differs from the frozen comparison')
    baseline = MonitorResult.model_validate_json(BASELINE.read_text())
    assert baseline.status == 'completed' and baseline.error is None
    selected = select_checkpoints(Path('.runs/grid-007/feasibility/results/small-s4/study/training/run'),(0,))
    point = selected.points[0]
    protocol = MonitorProtocol.model_validate(baseline.protocol.model_dump(mode='json'))
    manifest = _manifest(selected.run,point.artifact,point.coordinates,protocol)
    assert manifest.evaluation_identities == baseline.evaluation_identities
    return RuntimeIdentity(scientific_source=SOURCE,
        baseline_monitor_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        evaluation_identities=manifest.evaluation_identities)


def verify_current() -> RuntimeIdentity:
    expected = RuntimeIdentity.model_validate_json(RECEIPT.read_text())
    assert current() == expected, 'CPU evaluation runtime changed after preparation'
    return expected


def main() -> None:
    identity = current()
    if RECEIPT.exists():
        assert RuntimeIdentity.model_validate_json(RECEIPT.read_text()) == identity
    else:
        RECEIPT.write_text(identity.model_dump_json(indent=2)+'\n')
    paths = sorted(Path('.runs/grid-007/early-learning-013/panels').glob('*/transitions-*/monitor.json'))
    paths += sorted(Path('.runs/grid-007/early-learning-013-added/panels').glob('*/transitions-*/monitor.json'))
    assert len(paths) == 92
    for path in paths:
        panel = MonitorResult.model_validate_json(path.read_text())
        assert panel.status == 'completed' and panel.error is None
        assert panel.evaluation_identities == identity.evaluation_identities,path
    proof = {'status':'verified','panels':len(paths),
        'baseline':str(BASELINE),'baseline_sha256':identity.baseline_monitor_sha256,
        'runtime_receipt_sha256':hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        'all_panel_runtime_identities_equal':True,
        'scope':'Recorded engine binary/source, content, ABIs, matchup, pilot and monitor implementation identities; current Python/Torch/NumPy versions are also frozen for the later evaluation'}
    Path('.runs/grid-007/early-learning-013-runtime-compatibility.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
