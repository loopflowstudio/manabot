"""Verify every completed grouped-grid cell and preserve censored/missing coverage."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

import torch

from experiments.runners.batching_evaluation import select_checkpoints
from experiments.runners.batching_grid import CAPACITIES, recipe
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.models import TrainingRun

ROOT = Path('.runs/grid-007/grouped-feasibility')
PARAMETERS = {'small': 188610, 'middle': 2463170, 'large': 16816514, 'small-deep': 288578, 'large-shallow': 6169730}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_manifest(study: Path) -> int:
    entries = json.loads((study / 'manifest.json').read_text())
    assert isinstance(entries, list) and entries
    seen: set[str] = set()
    for entry in entries:
        assert entry['path'] not in seen
        seen.add(entry['path'])
        path = study / entry['path']
        assert path.resolve().is_relative_to(study.resolve())
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], path
    return len(entries)


def audit_module() -> types.ModuleType:
    """Reuse retained audit definitions without its historical aggregate-writing code."""
    source = Path('.runs/grid-007/audit_completed_numerics.py')
    parsed = ast.parse(source.read_text(), filename=str(source))
    definitions = [node for node in parsed.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    module = types.ModuleType('grouped_retained_numerical_audit')
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), module.__dict__)
    return module


def verify_partial(study: Path, *, source: str, size: str, streams: int) -> dict[str, object]:
    """Retain a deadline censor as partial evidence, never as a shorter endpoint."""
    count = verify_manifest(study)
    run = TrainingRun.model_validate_json((study / 'training/run/run.json').read_text())
    assert run.seed == 13571 and run.parent_run_id is None
    capacity = next(cap for cap in CAPACITIES if cap.name == size)
    assert run.regime == recipe(capacity, streams, 131072)
    assert json.loads((study / 'source-bundle.json').read_text())['commit'] == source
    retained = [(0,run.stages[0].artifacts['initial_raw']), *((row.learner_transitions,row.artifact) for row in run.monitoring_checkpoints if row.artifact is not None and row.error is None)]
    manifested = {row['path'] for row in json.loads((study/'manifest.json').read_text())}
    for _,artifact in retained:
        path = study/'training/run'/Path(artifact['path']).name
        assert str(path.relative_to(study)) in manifested
        assert path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256']
        agent, _ = load_checkpoint_agent(str(path))
        assert sum(p.numel() for p in agent.parameters()) == PARAMETERS[size]
        assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())
    learner_deadline = run.status == 'interrupted' and run.error is not None and 'collector exceeded training deadline' in run.error
    return {'classification':'learner_deadline_censor' if learner_deadline else 'incomplete_other',
            'status':run.status,'error':run.error,'learner_transitions':run.stages[0].learner_transitions,
            'updates':run.updates_through(),'training_seconds':run.seconds,
            'all_manifest_files_verified':True,'manifest_files':count,
            'retained_raw_checkpoints_finite':True,'checkpoint_transitions':[point for point,_ in retained],
            'exact_recipe_verified':True,'source':source,
            'limit':'Partial retained trajectory only; no completed endpoint or full optimizer audit'}


def main() -> None:
    torch.set_num_threads(1)
    finished = json.loads((ROOT / 'finished.json').read_text())
    settlement = json.loads((ROOT / 'rental-settlement.json').read_text())
    assert finished['provider_absent'] and settlement['provider_absent']
    launch = json.loads((ROOT / 'launch.json').read_text())
    wheel = json.loads(Path('.runs/grid-007/linux-wheelhouse/upload-receipt.json').read_text())
    auditor = audit_module()
    cells: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for group in launch['groups']:
        directory = ROOT / group['name']
        status = json.loads((directory / 'status.json').read_text())
        output = directory / 'results'
        worker_path = output / 'worker-status.json'
        completed: list[str] = []
        if worker_path.exists():
            worker = json.loads(worker_path.read_text())
            completed = worker.get('completed', [])
        if completed:
            assert json.loads((output / 'cuda-admission.json').read_text())['status'] == 'passed'
            dependency = json.loads((output / 'dependency-setup.json').read_text())
            assert dependency['phase'] == 'offline_install_completed' and dependency['wheels'] == 94
            assert dependency['sha256'] == wheel['sha256']
            assert digest(output / 'locked-dependency-setup.py') == launch['helper_sha256']
        declared = {value.replace(':', '-s') for value in group['cells']}
        assert set(completed).issubset(declared) and len(completed) == len(set(completed))
        if status['outcome'] == 'succeeded':
            assert set(completed) == declared
        for value in group['cells']:
            size, streams_text = value.split(':')
            streams = int(streams_text)
            name = f'{size}-s{streams}'
            retained = output / name
            study = retained / 'study'
            if name not in completed:
                partial = verify_partial(study, source=group['source'], size=size, streams=streams) if (study / 'training/run/run.json').exists() else None
                missing.append({'name': name, 'group': group['name'], 'group_outcome': status['outcome'], 'study_present': study.exists(), 'partial':partial, 'reason': 'Not a completed target; retained failure/censor remains separate'})
                continue
            count = verify_manifest(study)
            run = TrainingRun.model_validate_json((study / 'training/run/run.json').read_text())
            assert run.status == 'completed' and run.seed == 13571 and run.parent_run_id is None
            assert run.updates_through() == 131072 // (streams * 64)
            assert len(run.stages) == 1 and run.stages[0].learner_transitions == 131072
            capacity = next(cap for cap in CAPACITIES if cap.name == size)
            assert run.regime == recipe(capacity, streams, 131072)
            source = json.loads((study / 'source-bundle.json').read_text())['commit']
            assert source == group['source']
            summary = json.loads((study / 'summary.json').read_text())
            assert summary['status'] == 'completed' and summary['checkpoint_finite'] and summary['parameters'] == PARAMETERS[size]
            selected = select_checkpoints(study / 'training/run', (0, 131072))
            for point in selected.points:
                agent, _ = load_checkpoint_agent(point.artifact['path'])
                assert sum(p.numel() for p in agent.parameters()) == PARAMETERS[size]
                assert all(bool(torch.isfinite(p).all()) for p in agent.parameters())
            proof = {'source': source, 'updates': run.updates_through(), 'all_manifest_files_verified': True, 'manifest_files': count, 'checkpoint_loaded': True, 'both_raw_checkpoints_finite': True, 'exact_recipe_verified': True}
            (retained / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
            numerical = auditor.audit(retained)
            movement = numerical['movement']
            counts = numerical['diagnostic_counts']
            assert movement['all_parameters_finite'] and movement['all_moments_finite'] and movement['elements_above_bound'] == 0 and movement['negative_second_moments'] == 0
            assert counts['nonfinite_count'] == 0 and counts['rejected_steps'] == 0
            cells.append({'name': name, 'source': source, 'group': group['name'], 'directory': str(retained), 'summary': summary, 'verification': proof, 'numerical_audit': numerical})
            print(json.dumps({'cell': name, 'verified': True, 'seconds': summary['run_seconds'], 'transitions_per_second': summary['inclusive_transitions_per_second']}), flush=True)
    result = {'cells': cells, 'missing': missing, 'unstarted_groups': launch['pending'], 'provider_absent': True, 'audit_source_sha256': digest(Path('.runs/grid-007/audit_completed_numerics.py')), 'verifier_sha256': digest(Path(__file__)), 'scope': 'Completed fixed-transition feasibility cells; censored/incomplete outcomes remain separate. Numerical checks do not clear historical anomalies.'}
    (ROOT / 'verified-results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
