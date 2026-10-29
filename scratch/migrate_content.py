"""One-time migration from retained sources; not a production content generator."""
import copy
import json
from pathlib import Path

root = Path('content/semantic/current')
root.mkdir(parents=True, exist_ok=True)
paths = [Path('content/semantic/w11/elemental_world.source.json')] + sorted(Path('content/semantic/v1').glob('*.source.json'))
by_name = {}
source_docs = []
for path in paths:
    doc = json.loads(path.read_text())
    names = {d['key']: d['registry_name'] for d in doc['definitions']}
    source_docs.append((path, doc, names))
    for d in doc['definitions']:
        by_name.setdefault(d['registry_name'], (copy.deepcopy(d), names))
name_keys = {name: d['key'] for name, (d, _) in by_name.items()}
def rewrite(value, names):
    if isinstance(value, dict):
        return {k: name_keys[names[v]] if k in ('definition_ref', 'definition') else rewrite(v, names) for k,v in value.items()}
    if isinstance(value, list): return [rewrite(v,names) for v in value]
    return value
fixtures = json.loads(Path('managym/tests/fixtures/scryfall_cards.json').read_text())
texts = {name: row['oracle_text'] for name,row in fixtures.items()}
texts.update({'Ally':'', 'Clue':'{2}, Sacrifice this token: Draw a card.', 'Food':'{2}, {T}, Sacrifice this token: You gain 3 life.', 'Treasure':'{T}, Sacrifice this token: Add one mana of any color.', 'Human Soldier':'', 'Spirit':"This token can't block or be blocked by non-Spirit creatures.", 'Savannah Lions':'', 'Glory Seeker':'', 'Hill Giant':''})
definitions = [rewrite(d, names) | {'text_box': texts[d['registry_name']]} for d,names in by_name.values()]
presets = {}
for path,doc,names in source_docs:
    key = 'optional_payments' if path.parent.name=='w11' else path.stem.removesuffix('.source')
    presets[key] = {'pack_key': doc['pack_key'], 'decks': rewrite(doc['decks'],names)}
Path(root/'definitions.json').write_text(json.dumps(sorted(definitions,key=lambda x:x['key']),indent=2,ensure_ascii=False)+'\n')
Path(root/'setups.json').write_text(json.dumps(presets,indent=2,ensure_ascii=False)+'\n')
Path(root/'presentation.json').write_text(json.dumps(texts,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
