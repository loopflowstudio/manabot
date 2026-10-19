"""Audit protocol017 retention before any playing-strength evaluation.

The allocation owns seed, recipe, deadline and checkpoint coordinates. Completed
trajectories admit all four raw checkpoints together; incomplete trajectories
retain their evidence without becoming shorter substitutes for the endpoint.
"""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import torch

from experiments.runners.batching_evaluation import select_checkpoints
from experiments.runners.batching_grid import CAPACITIES, recipe
from experiments.runners.grouped_learning import PlannedCell, verify_completed
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.models import ArtifactReference, TrainingRun

ROOT = Path('.runs/grid-007/learning-017')
PARAMETERS = {'small':188610,'small-deep':288578,'middle':2463170,
              'large-shallow':6169730,'large':16816514}


@dataclass(frozen=True)
class Expected:
    source: str
    seed: int
    size: str
    streams: int
    seconds: int
    transitions: int
    checkpoints: tuple[int, ...]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def retained_module() -> ModuleType:
    path = Path('.runs/grid-007/verify_grouped.py')
    spec = importlib.util.spec_from_file_location('retained_grid_verifier',path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_trajectory(study: Path, expected: Expected) -> TrainingRun:
    run = TrainingRun.model_validate_json((study/'training/run/run.json').read_text())
    assert run.seed == expected.seed and run.parent_run_id is None
    assert len(run.stages) == 1 and run.stages[0].learning_state_origin is None
    capacity = next(cap for cap in CAPACITIES if cap.name == expected.size)
    assert run.regime == recipe(capacity,expected.streams,expected.transitions,
                                seconds=expected.seconds)
    assert json.loads((study/'source-bundle.json').read_text())['commit'] == expected.source
    return run


def verify_artifact(study: Path, artifact: ArtifactReference, size: str) -> None:
    path = study/'training/run'/Path(artifact['path']).name
    manifest = json.loads((study/'manifest.json').read_text())
    assert str(path.relative_to(study)) in {entry['path'] for entry in manifest}
    assert path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256']
    agent,_ = load_checkpoint_agent(str(path))
    assert sum(p.numel() for p in agent.parameters()) == PARAMETERS[size]
    assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())


def main() -> None:
    torch.set_num_threads(1)
    plan = json.loads((ROOT/'allocation.json').read_text())
    launch = json.loads((ROOT/'launch.json').read_text())
    assert digest(ROOT/'allocation.json') == launch['allocation_sha256']
    for name,expected_hash in plan['inputs'].items():
        assert digest(ROOT/'frozen-inputs'/name) == expected_hash,name
    assert json.loads((ROOT/'finished.json').read_text())['provider_absent']
    assert json.loads((ROOT/'rental-settlement.json').read_text())['provider_absent']
    audit_helpers = retained_module()
    auditor = audit_helpers.audit_module()
    wheel = json.loads(Path('.runs/grid-007/linux-wheelhouse/upload-receipt.json').read_text())
    cells: list[dict[str,object]] = []
    incomplete: list[dict[str,object]] = []
    for allocation in plan['groups']:
        seed = allocation['seed']
        group_name = f'seed-{seed}'
        matches = [g for g in launch['groups'] if g['seed'] == seed]
        if not matches:
            incomplete.extend({'name':f'{group_name}-{c["cell"].replace(":","-s")}',
                               'seed':seed,'classification':'unstarted_group'}
                              for c in allocation['cells'])
            continue
        assert len(matches) == 1
        group = matches[0]
        assert group['source'] == plan['source']
        assert group['transitions'] == plan['transitions']
        assert group['checkpoints'] == plan['checkpoint_transitions']
        assert group['cells'] == [c['cell'] for c in allocation['cells']]
        assert group['cell_seconds'] == [c['seconds'] for c in allocation['cells']]
        directory = ROOT/group_name
        output = directory/'results'
        status = json.loads((directory/'status.json').read_text())
        worker_path = output/'worker-status.json'
        completed = json.loads(worker_path.read_text()).get('completed',[]) if worker_path.exists() else []
        declared = {c['cell'].replace(':','-s') for c in allocation['cells']}
        assert len(set(completed)) == len(completed) and set(completed).issubset(declared)
        if status['outcome'] == 'succeeded':
            assert set(completed) == declared
        if completed:
            assert json.loads((output/'cuda-admission.json').read_text())['status'] == 'passed'
            dependency = json.loads((output/'dependency-setup.json').read_text())
            assert dependency['phase'] == 'offline_install_completed' and dependency['wheels'] == 94
            assert dependency['sha256'] == wheel['sha256']
            assert digest(output/'locked-dependency-setup.py') == launch['helper_sha256']
        for item in allocation['cells']:
            size,streams_text = item['cell'].split(':')
            cell = item['cell'].replace(':','-s')
            name = f'{group_name}-{cell}'
            retained = output/cell
            study = retained/'study'
            expected = Expected(plan['source'],seed,size,int(streams_text),item['seconds'],
                                plan['transitions'],tuple(plan['checkpoint_transitions']))
            if cell not in completed:
                record: dict[str,object] = {'name':name,'seed':seed,'cell':cell,
                    'classification':'unstarted_or_no_retained_trajectory','group_outcome':status['outcome']}
                if (study/'training/run/run.json').exists():
                    manifest_count = audit_helpers.verify_manifest(study)
                    run = verify_trajectory(study,expected)
                    points = [(0,run.stages[0].artifacts['initial_raw'])]
                    points.extend((c.learner_transitions,c.artifact) for c in run.monitoring_checkpoints
                                  if c.artifact is not None and c.error is None)
                    for _,artifact in points:
                        verify_artifact(study,artifact,size)
                    record.update(classification='incomplete_retained_trajectory',status=run.status,
                        error=run.error,transitions=run.stages[0].learner_transitions,
                        seconds=run.seconds,manifest_files=manifest_count,
                        finite_checkpoint_transitions=[count for count,_ in points],
                        limitation='Partial evidence only; not admitted to completed endpoint evaluation')
                incomplete.append(record)
                continue
            verify_completed(study,PlannedCell.parse(f'{item["cell"]}:{item["seconds"]}'),
                             seed=seed,transitions=plan['transitions'],
                             checkpoints=tuple(plan['checkpoint_transitions']))
            manifest_count = audit_helpers.verify_manifest(study)
            run = verify_trajectory(study,expected)
            assert run.status == 'completed'
            points = select_checkpoints(study/'training/run',(0,*expected.checkpoints)).points
            for point in points:
                verify_artifact(study,point.artifact,size)
            summary = json.loads((study/'summary.json').read_text())
            assert summary['parameters'] == PARAMETERS[size]
            proof = {'source':plan['source'],'seed':seed,'all_manifest_files_verified':True,
                'manifest_files':manifest_count,'checkpoint_loaded':True,'all_raw_checkpoints_finite':True,
                'exact_recipe_verified':True,'updates':run.updates_through(),
                'checkpoint_transitions':[0,*expected.checkpoints]}
            (retained/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
            numerical = auditor.audit(retained)
            movement,counts = numerical['movement'],numerical['diagnostic_counts']
            assert movement['all_parameters_finite'] and movement['all_moments_finite']
            assert movement['elements_above_bound'] == movement['negative_second_moments'] == 0
            assert counts['nonfinite_count'] == counts['rejected_steps'] == 0
            cells.append({'name':name,'cell':cell,'seed':seed,'size':size,'streams':int(streams_text),
                'directory':str(retained),'summary':summary,'verification':proof,
                'numerical_audit':numerical,'checkpoints':[p.model_dump(mode='json') for p in points]})
            print(json.dumps({'name':name,'verified':True,'seconds':run.seconds}),flush=True)
    assert len(cells)+len(incomplete) == 22
    result = {'cells':cells,'incomplete':incomplete,'provider_absent':True,
        'complete_cohort':len(cells)==22,'allocation_sha256':digest(ROOT/'allocation.json'),
        'verifier_sha256':digest(Path(__file__)),
        'scope':'Exact frozen trajectories and retained numerical health; no historical anomaly clearance'}
    (ROOT/'verified-results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
