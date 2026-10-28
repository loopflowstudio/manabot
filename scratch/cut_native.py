import json,re,subprocess
from pathlib import Path
cards={c['name'] for c in json.load(open('scratch/tla-evidence/cards.json'))}
tokens={'Ally','Clue','Food','Spirit'}
p=Path('content/semantic/current/definitions.json'); defs=json.loads(p.read_text())
native=[d['key'] for d in defs if d['registry_name'] in cards and 'land' not in d['characteristics']['types'] or d['registry_name'] in cards and 'basic' not in d['characteristics'].get('supertypes',[]) ]
# Every TLA nonbasic definition, including nonbasic lands. Tokens follow references.
Path('content/semantic/current/native_roots.json').write_text(json.dumps(sorted(native),indent=2)+'\n')
for filename in ['tla','elemental','sacrifice','temur']:
 p=Path(f'managym/src/cardsets/{filename}.rs');text=p.read_text()
 matches=list(re.finditer(r'        self\.register_card\(CardDefinition \{.*?^        \}\);',text,re.M|re.S))
 kept=[]
 for match in matches:
  name=re.search(r'name: "([^"]+)"',match[0])[1]
  if name not in cards|tokens:kept.append(match[0])
 if filename=='tla':
  assert len(kept)==1
  a=Path('managym/src/cardsets/alpha.rs');s=a.read_text();s=s.replace('        self.register_tla();\n        self.register_tla_lessons_increment();','        self.register_current_tla();\n        self.register_modal_fixture();')
  start=s.index('    /// Presentation entries for the isolated complexity catalog.')
  end=s.index('    pub fn register_card',start)
  s=s[:start]+s[end:]
  s += '\nimpl ContentPack {\n    fn register_modal_fixture(&mut self) {\n'+kept[0]+'\n    }\n}\n'
  a.write_text(s);p.unlink()
 else:
  # Rebuild each narrow non-TLA module; all TLA-only helpers disappear.
  s='// '+filename+'.rs\n// Non-TLA cards retained alongside the shared typed TLA catalog.\n\n'+text[text.index('use super::alpha'):text.index('/// Enumerate') if filename=='temur' else text.index('impl CardRegistry')]+'impl CardRegistry {\n    pub(super) fn register_'+filename+'(&mut self) {\n'+'\n'.join(kept)+'\n    }\n}\n'
  p.write_text(s)
p=Path('managym/src/cardsets/mod.rs');p.write_text(p.read_text().replace('pub mod tla;',''))
# Preserve the original fixture order and bytes for existing records.
p=Path('managym/tests/fixtures/scryfall_cards.json');new=json.loads(p.read_text());old=json.loads(subprocess.check_output(['git','show','HEAD:managym/tests/fixtures/scryfall_cards.json']))
for name in json.load(open('scratch/new-cards.json')):
 if name not in old:old[name]={key:new[name][key] for key in ['name','mana_cost','type_line','power','toughness','keywords','oracle_text','colors']}
p.write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n')
