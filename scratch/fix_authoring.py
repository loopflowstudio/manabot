import json,subprocess
from pathlib import Path
p=Path('content/semantic/current/definitions.json');d=json.loads(p.read_text())
def visit(x):
 if isinstance(x,dict):
  if x.get('target')=='self':x['target']='source'
  if x.get('mana')=='':x['mana']='0'
  if x.get('op')=='deal_power_damage' and isinstance(x.get('sources'),list):x['sources']=x['sources'][0]
  for v in x.values():visit(v)
 elif isinstance(x,list):
  for v in x:visit(v)
visit(d);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
p=Path('managym/tests/fixtures/scryfall_cards.json');new=json.loads(p.read_text());old=json.loads(subprocess.check_output(['git','show','HEAD:managym/tests/fixtures/scryfall_cards.json']))
for k,v in new.items():old.setdefault(k,v)
p.write_text(json.dumps(dict(sorted(old.items())),indent=2)+'\n')
