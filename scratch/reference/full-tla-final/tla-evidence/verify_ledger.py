"""Offline checks for the research inventory, evidence pins and batch partition."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path: Path) -> Any:
    # External JSON boundary; required fields and invariants are checked below.
    return json.loads(path.read_text())


def main() -> None:
    manifest = read_json(HERE/'sources.json')
    for source in manifest['sources'] + manifest['repository_sources'] + manifest['research_artifacts']:
        path = ROOT/source['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], source['path']
    rows = [json.loads(line) for line in (HERE.parent/'tla-card-ledger.jsonl').read_text().splitlines()]
    faces = [r for r in rows if r['kind']=='card_face']
    assert len(rows)==305 and len(faces)==294
    assert len({r['key'] for r in rows})==len(rows)
    ids = {r['oracle_id'] for r in faces}
    assert len(ids)==286
    official = read_json(HERE/'official-checklist-rows.json')
    normalize = lambda name: name.replace('’',"'").replace('‘',"'").replace('—','-').strip()
    expected = {(r[0],normalize(r[2])) for r in official if r[1] in ('CARDS MAIN SET','CARDS BASIC LANDS STANDARD')}
    assert expected == {(r['key'].split(':')[1],normalize(r['face_name'])) for r in faces}
    assignments=read_json(HERE/'card-batches.json')
    delivery=read_json(HERE/'delivery-plan.json')
    assert set(assignments)==ids and len(delivery)==8
    for row in faces:
        assert row['planned_pr']==assignments[row['oracle_id']]['planned_pr']
        assert row['prerequisite_prs']==delivery[row['planned_pr']]['depends_on']
        assert row['obligations'] and row['gap'] and row['rule_refs']
        assert row['support_status'] in {'partial','missing','unreviewed','verified_supported_subset'}
    visiting: set[str]=set()
    visited: set[str]=set()
    def visit(key: str) -> None:
        assert key not in visiting, f'dependency cycle at {key}'
        if key in visited:
            return
        visiting.add(key)
        for parent in delivery[key]['depends_on']:
            visit(parent)
        visiting.remove(key)
        visited.add(key)
    for key in delivery:
        visit(key)
    tokens=[r for r in rows if r['kind']!='card_face']
    assert sum(r['in_tla_closure'] for r in tokens)==10
    assert all(r['planned_pr'] is not None for r in tokens if r['in_tla_closure'])
    assert [r['face_name'] for r in tokens if not r['in_tla_closure']]==['Treasure']
    rules=(HERE/'comprehensive-rules.txt').read_text()
    for family in read_json(HERE/'mechanics.json').values():
        for rule in family['rules']:
            assert re.search(r'^'+re.escape(rule)+r'(?:\.|\s)',rules,re.M), rule
    check=read_json(HERE/'inventory-check.json')
    assert check['official_variant_unmatched']==[] and check['printings']==394
    assert check['card_batch_counts']==dict(Counter(a['planned_pr'] for a in assignments.values()))
    validation=read_json(HERE/'validation.json')
    assert sum(x['passed'] for x in validation['native_suites'])==201
    for suite in validation['native_suites']+[validation['compiled_parity']]:
        assert f"test result: ok. {suite['passed']} passed; 0 failed" in (ROOT/suite['log']).read_text()
    print('PASS: 286 cards / 294 faces / 394 printings; token closure; 305 ledger rows; eight acyclic delivery groups; source hashes; 201 rules + 1 parity tests.')


if __name__=='__main__':
    main()
