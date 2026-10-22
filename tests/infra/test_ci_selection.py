"""CI selection must isolate packages without silently skipping dependencies."""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / '.github/ci/select-tests.sh'
JOBS = {
    'rust', 'python_unit', 'integration', 'academy_integration', 'protocol_v1',
    'semantic_coverage', 'semantic_conformance', 'frontend', 'clean_machine_play',
}


def selected(paths: list[str], *, full: bool = False) -> set[str]:
    result = subprocess.run(
        ['bash', str(SELECTOR), *(['--all'] if full else [])],
        input=''.join(path + '\0' for path in paths), text=True,
        capture_output=True, check=True,
    )
    outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
    assert set(outputs) == JOBS
    assert set(outputs.values()) <= {'true', 'false'}
    return {job for job, value in outputs.items() if value == 'true'}


@pytest.mark.parametrize(('path', 'expected'), [
    ('docs/design.md', set()),
    ('docs/rules/elemental/decks.json', {'academy_integration'}),
    ('academy/README.md', set()),
    ('scratch/notes.md', set()),
    ('website/app.py', {'python_unit'}),
    ('tests/infra/test_ci_selection.py', {'python_unit'}),
    ('academy/ui/src/lib/components/Card.svelte', {'frontend'}),
    ('academy/ui/src/lib/components/a card.svelte', {'frontend'}),
    ('academy/ui/package-lock.json', {'frontend', 'clean_machine_play'}),
    ('manabot/model/train.py', {'integration'}),
    ('manabot/remote/jobs.py', {'integration'}),
    ('experiments/protocols/recipe.md', {'integration'}),
    ('docs/experiment-metrics.md', {'integration'}),
    ('manabot/semantic/compiler.py', {'integration', 'semantic_coverage'}),
    ('academy/server.py', {'academy_integration', 'protocol_v1', 'frontend'}),
    ('academy/study_runtime.py', {'academy_integration', 'protocol_v1'}),
    ('scripts/play.py', {'python_unit', 'protocol_v1', 'clean_machine_play'}),
    ('managym/src/flow/decision.rs', JOBS),
    ('content/semantic/v1/fixtures.json', JOBS),
    ('uv.lock', JOBS),
    ('pyproject.toml', JOBS),
    ('.github/workflows/ci.yml', JOBS),
    ('.github/ci/select-tests.sh', JOBS),
    ('.github/workflows/manabot-integration.yml', {'integration'}),
    ('unknown-new-package/code.py', JOBS),
])
def test_dependency_selection(path: str, expected: set[str]) -> None:
    assert selected([path]) == expected


def test_renames_and_mixed_changes_select_both_consumers() -> None:
    # git diff --no-renames reports deleted and added paths separately.
    assert selected(['manabot/model/old.py', 'academy/ui/src/new.ts']) == {'integration', 'frontend'}
    assert selected(['docs/notes.md', 'website/app.py']) == {'python_unit'}


def test_empty_diff_and_manual_dispatch_run_all_jobs() -> None:
    assert selected([]) == JOBS
    assert selected(['docs/notes.md'], full=True) == JOBS


@pytest.mark.parametrize(('changes', 'choice', 'result', 'passes'), [
    ('success', 'true', 'success', True),
    ('success', 'true', 'failure', False),
    ('success', 'true', 'skipped', False),
    ('success', 'true', 'cancelled', False),
    ('success', 'false', 'skipped', True),
    ('success', 'false', 'failure', False),
    ('success', '', 'skipped', False),
    ('failure', 'false', 'skipped', False),
])
def test_aggregate_rejects_unexpected_skips_and_missing_selection(
    changes: str, choice: str, result: str, passes: bool,
) -> None:
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    body = workflow.split('      - name: Check all jobs\n', 1)[1].split('        run: |\n', 1)[1]
    script = '\n'.join(line[10:] for line in body.splitlines())
    completed = subprocess.run(
        ['bash', '-eo', 'pipefail', '-c', script],
        env={**os.environ, 'CHANGES': changes, 'SELECTED': json.dumps({'frontend': choice}), 'RESULTS': f'frontend={result}'},
        capture_output=True, text=True,
    )
    assert (completed.returncode == 0) == passes, completed.stdout + completed.stderr
