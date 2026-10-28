"""Rebuild the research ledger from pinned inventory and inspected source.

This is an offline evidence join, never a support certifier. Candidate test
references prove neither execution nor complete card behavior. Manual face
obligations are in card-obligations.tsv; shared prerequisites are separate.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    # Pinned third-party JSON boundary; structure is checked where consumed.
    return json.loads(path.read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(name: str) -> str:
    return name.replace('’', "'").replace('‘', "'").replace('—', '-').strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    pages = [read_json(HERE / f'scryfall-tla-{i}.json') for i in range(1, 4)]
    prints = [c for page in pages for c in page['data']]
    assert len(prints) == pages[0]['total_cards'] == 394
    assert len({c['id'] for c in prints}) == 394
    assert not pages[-1]['has_more']
    assert {int(c['collector_number']) for c in prints} == set(range(1, 395))
    cards = list({c['oracle_id']: c for c in reversed(prints)}.values())
    cards.sort(key=lambda c: int(c['collector_number']))
    assert len(cards) == 286
    official = read_json(HERE / 'official-checklist-rows.json')
    base = [r for r in official if r[1] in ('CARDS MAIN SET', 'CARDS BASIC LANDS STANDARD')]
    expected = {(c['collector_number'] + (chr(97+i) if 'card_faces' in c else ''), normalize(f['name']))
                for c in cards for i, f in enumerate(c.get('card_faces', [c]))}
    observed = {(r[0], normalize(r[2])) for r in base}
    assert observed == expected, (observed - expected, expected - observed)
    # Every alternate printing must also have an official number/name match.
    all_official = {(r[0], normalize(r[2])) for r in official}
    variant_faces = {(c['collector_number'] + (chr(97+i) if 'card_faces' in c else ''), normalize(f['name']))
                     for c in prints for i, f in enumerate(c.get('card_faces', [c]))}
    unmatched = sorted(variant_faces - all_official)
    assert not unmatched, unmatched
    # Kept explicit rather than silently accepting a count-only comparison.
    obligations = dict(line.split('\t', 1) for line in (HERE/'card-obligations.tsv').read_text().splitlines())
    assert set(obligations) == {key for key, _ in expected}
    registry: dict[str, str] = {}
    for path in sorted((ROOT/'managym/src/cardsets').glob('*.rs')):
        source = path.read_text()
        patterns = [r'(?<![\w_])name:\s*"([^"\n]+)"', r'basic_land\("([^"]+)"', r'register_creature\(\s*"([^"]+)"']
        for pattern in patterns:
            for match in re.finditer(pattern, source):
                registry[match[1]] = f'{path.relative_to(ROOT)}:{source[:match.start()].count(chr(10))+1}'
    sources = sorted((ROOT/'content/semantic/v1').glob('*.source.json')) + [ROOT/'content/semantic/w11/elemental_world.source.json']
    compiled: dict[str, list[dict[str, object]]] = {}
    for path in sources:
        source = read_json(path)
        if 'pack_key' not in source:
            continue
        for definition in source['definitions']:
            deck_keys = [d['key'] for d in source['decks']
                         if any(x.get('definition') == definition['key'] for x in d.get('cards', [])+d.get('sideboard', []))]
            compiled.setdefault(definition['registry_name'], []).append({
                'pack_key': source['pack_key'], 'definition_key': definition['key'],
                'source': str(path.relative_to(ROOT)), 'authored_deck_keys': deck_keys,
                'status': 'authored_program_present_not_behavioral_proof'})
    test_refs: dict[str, list[dict[str, object]]] = {}
    names = {f['name'] for c in cards for f in c.get('card_faces', [c])}
    token_prints = read_json(HERE/'scryfall-ttla.json')['data']
    names.update(c['name'] for c in token_prints)
    log = '\n'.join(path.read_text() for path in HERE.glob('native-*-tests.log'))
    for path in sorted((ROOT/'managym/tests/rules').glob('*.rs')):
        source = path.read_text()
        tests = list(re.finditer(r'#\[test\]\s*fn\s+(\w+)\s*\(', source))
        for i, match in enumerate(tests):
            body = source[match.end():tests[i+1].start() if i+1 < len(tests) else len(source)]
            literals = set(re.findall(r'"([^"\n]+)"', body))
            for name in names & literals:
                qualified = f'rules::{path.stem}::{match[1]}'
                test_refs.setdefault(name, []).append({'test': f'{path.relative_to(ROOT)}::{match[1]}',
                    'executed_this_run': f'test {qualified} ... ok' in log,
                    'scope': 'card occurs in test fixture; inspect assertions; not whole-card certification'})
    release = (HERE/'release-notes.txt').read_text().splitlines()
    families = read_json(HERE/'mechanics.json')
    assignments = read_json(HERE/'card-batches.json')
    delivery = read_json(HERE/'delivery-plan.json')
    ledger: list[dict[str, object]] = []
    for c in cards:
        for i, face in enumerate(c.get('card_faces', [c])):
            key = c['collector_number'] + (chr(97+i) if 'card_faces' in c else '')
            name = face['name']
            text = face.get('oracle_text', '')
            searchable = (text + ' ' + face['type_line'] + ' ' + obligations[key]).lower()
            prerequisites = ['core'] + [k for k, f in families.items() if k != 'core' and any(re.search(pattern, searchable) for pattern in f['patterns'])]
            if c['layout'] in ('transform', 'saga') and 'saga_faces' not in prerequisites:
                prerequisites.append('saga_faces')
            native = registry.get(name)
            refs = test_refs.get(name, [])
            status = 'partial' if native and refs else 'unreviewed' if native else 'missing'
            release_lines = [j+1 for j, line in enumerate(release) if normalize(line).startswith(normalize(name)+' {') or normalize(line).startswith(normalize(name)+' Land ')]
            ledger.append({
                'kind': 'card_face', 'key': f'TLA:{key}',
                'planned_pr': assignments[c['oracle_id']]['planned_pr'],
                'prerequisite_prs': delivery[assignments[c['oracle_id']]['planned_pr']]['depends_on'], 'card_name': c['name'], 'face_name': name,
                'face_index': i, 'layout': c['layout'], 'oracle_id': c['oracle_id'],
                'printing_ids': [p['id'] for p in prints if p['oracle_id'] == c['oracle_id']],
                'collector_numbers': [p['collector_number'] for p in prints if p['oracle_id'] == c['oracle_id']],
                'inventory_authority': 'official-card-list.pdf', 'official_inventory_match': True,
                'oracle_transcription_url': c['scryfall_uri'], 'oracle_text_sha256': digest(text.encode()),
                'official_rulings': {'url': 'https://magic.wizards.com/en/news/feature/avatar-the-last-airbender-release-notes',
                                    'local_text_lines': release_lines, 'status': 'card_section_present' if release_lines else 'no_card_specific_section_matched; general_CR_applies'},
                'obligations': obligations[key], 'prerequisite_families': prerequisites,
                'rule_refs': sorted({r for family in prerequisites for r in families[family]['rules']}),
                'support_status': status, 'native_registration': native,
                'compiled_admission': compiled.get(name, []),
                'current_elemental_admission': any(x['pack_key'] == 'elemental-world-v2' for x in compiled.get(name, [])),
                'behavioral_evidence': refs,
                'gap': ('No native card definition or compiled admission found; implement complete face obligations using shared families.' if not native else
                        'Registered subset and listed tests do not establish all clauses or full-file interactions; resolve relevant family gaps and add independent complete-card witnesses.'),
                'evidence_limit': 'Source inspection at pinned revision. No whole-card support claim; test-name/card-name joins are navigation only. Per-clause independent adjudication remains open.',
                'related_token_ids': sorted({x['id'] for x in c.get('all_parts', []) if x['component'] == 'token'}),
            })
    tokens = list({c['oracle_id']:c for c in reversed(token_prints)}.values())
    tokens.sort(key=lambda c:int(c['collector_number']))
    needed_ids = {x['id'] for c in cards for x in c.get('all_parts', []) if x['component']=='token'}
    assert needed_ids <= {c['id'] for c in token_prints}
    for c in tokens:
        aliases = [t for t in token_prints if t['oracle_id']==c['oracle_id']]
        needed = any(t['id'] in needed_ids for t in aliases)
        name = c['name']
        native = registry.get(name)
        ledger.append({'kind':'token_template' if name != 'Copy' else 'copy_marker', 'key':f"TTLA:{c['collector_number']}",
                       'face_name':name, 'oracle_id':c['oracle_id'], 'collector_numbers':[t['collector_number'] for t in aliases],
                       'planned_pr': ('T6' if name == 'Copy' else 'T1') if needed else None,
                       'prerequisite_prs': delivery['T6' if name == 'Copy' else 'T1']['depends_on'] if needed else [],
                       'in_tla_closure':needed, 'native_registration':native, 'compiled_admission':compiled.get(name,[]),
                       'support_status':'partial' if native else 'missing',
                       'characteristics':{k:c.get(k) for k in ['type_line','colors','power','toughness']},
                       'obligations': 'Copy marker is not a fixed game definition: derive copiable values, exceptions and identity from the producing effect.' if name=='Copy' else
                                      f"Create exact {name} token characteristics; execute its abilities and token-zone/SBA lifecycle.",
                       'behavioral_evidence':test_refs.get(name,[]),
                       'prerequisite_families':['core','copy_tokens'],
                       'gap': 'General copying and spell/ability copying absent; narrow Ran and Shaw self-copy exists.' if name=='Copy' else
                              'Complete-card and cross-mechanic evidence required.' if native else 'Missing token definition and independent producer/lifecycle witnesses.',
                       'producers':[c0['name'] for c0 in cards if any(x['id'] in {t['id'] for t in aliases} for x in c0.get('all_parts',[]) if x['component']=='token')],
                       'evidence_limit':'TTLA art variants deduplicated by Oracle ID; Treasure is adjacent-product-only. No TLA emblem-producing text found.'})
    output = ROOT/'scratch/tla-card-ledger.jsonl'
    output.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in ledger))
    stats = {'gameplay_cards':len(cards),'card_faces':len(expected),'printings':len(prints),'printing_faces':len(variant_faces),
             'official_base_mismatches':0,'official_variant_unmatched':unmatched,
             'face_support':dict(Counter(row['support_status'] for row in ledger if row['kind']=='card_face')),
             'native_gameplay_cards':sum(c['name'] in registry for c in cards),
             'current_elemental_cards':sum(any(x['pack_key']=='elemental-world-v2' for x in compiled.get(c['name'],[])) for c in cards),
             'any_compiled_cards':sum(c['name'] in compiled for c in cards),
             'token_templates_including_copy_marker':len(tokens),'required_token_templates_including_copy_marker':sum(bool(r.get('in_tla_closure')) for r in ledger if r['kind']!='card_face'),
             'card_batch_counts':dict(Counter(v['planned_pr'] for v in assignments.values())),
             'ledger_rows':len(ledger),'ledger_sha256':digest(output.read_bytes()),
             'family_face_counts':{k:sum(k in r.get('prerequisite_families',[]) for r in ledger if r['kind']=='card_face') for k in families}}
    write_json(HERE/'inventory-check.json',stats)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
