import copy,json
from pathlib import Path
p=Path('content/semantic/current/definitions.json');defs=json.loads(p.read_text());cards={c['name']:c for c in json.load(open('scratch/tla-evidence/cards.json'))}
new=[]
def add(name,programs,keywords=()):
 c=cards[name];types=c['type_line'].split(' — ')
 d={'key':'tla.'+name.lower().replace('-','_').replace(' ','_'),'registry_name':name,'text_box':c['oracle_text'],'characteristics':{'types':types[0].lower().split(),'subtypes':types[1].split(),'mana_cost':c['mana_cost'],'power':int(c['power']),'toughness':int(c['toughness']),'keywords':list(keywords)},'programs':programs}
 defs.append(d);new.append(name)
def trigger(event,ops,**kw):return {'key':'ability','kind':'triggered','trigger':{'event':event,'subject':{'kind':'this'},**kw},'ops':ops}
condition={'kind':'graveyard_at_least','count':1,'predicate':{'kind':'subtype','value':'Lesson'}}
add('Walltop Sentries',[trigger('dies',[{'op':'branch','condition':condition,'then':[{'op':'gain_life','amount':2}],'otherwise':[]}],**{'if':condition})],['reach','deathtouch'])
cat=trigger('attacks',[{'op':'untap','target':'permanent'}]);cat['targets']=[{'role':'permanent','selector':{'kind':'permanent','predicate':{'kind':'types_any','values':['artifact','creature']}},'min':1,'max':1}]
add('Cat-Owl',[cat],['flying'])
soldier=trigger('enters_battlefield',[{'op':'target_clause','role':'creatures','body':[{'op':'for_each_target','role':'creatures','body':[{'op':'put_counters','target':'creatures','count':1}]}]}]);soldier['targets']=[{'role':'creatures','selector':{'kind':'creature_you_control'},'min':0,'max':2}]
add('Earth Kingdom Soldier',[soldier],['vigilance'])
add('Wandering Musicians',[trigger('attacks',[{'op':'modify_controlled','predicate':{'kind':'card_type','value':'creature'},'power':1,'toughness':0}])])
tokens={'Ally','Spirit','Clue','Food','Dragon','Monk','Soldier','Bear','Ballistic Boulder'}
raw={c['name']:c for c in json.load(open('scratch/tla-evidence/scryfall-ttla.json'))['data']};known={d['registry_name'] for d in defs}
for name in tokens-known:
 c=raw[name];types=c['type_line'].replace('Token ','').split(' — ')
 d={'key':'token.'+name.lower().replace(' ','_'),'registry_name':name,'text_box':c['oracle_text'],'characteristics':{'types':types[0].lower().split(),'subtypes':types[1].split(),'colors':c['colors'],'power':int(c['power']),'toughness':int(c['toughness']),'token':True,'keywords':['flying'] if name in ['Dragon','Ballistic Boulder'] else []},'programs':[]}
 if name in ['Dragon','Soldier']:d['programs']=[trigger('attacks',[{'op':'add_mana','mana':'RRRR' if name=='Dragon' else 'R','until':'end_of_combat'}])]
 if name=='Monk':d['programs']=[trigger('cast_spell',[{'op':'modify_pt','target':'source','power':1,'toughness':1,'duration':'end_of_turn'}],predicate={'kind':'not_card_types','values':['creature']})]
 defs.append(d)
p.write_text(json.dumps(sorted(defs,key=lambda d:d['key']),ensure_ascii=False,indent=2)+'\n')
Path('content/semantic/current/token_templates.json').write_text(json.dumps(sorted(d['key'] for d in defs if d['registry_name'] in tokens),indent=2)+'\n')
p=Path('content/semantic/current/native_roots.json');roots=json.loads(p.read_text());roots.extend(d['key'] for d in defs if d['registry_name'] in new);p.write_text(json.dumps(sorted(roots),indent=2)+'\n')
p=Path('managym/tests/fixtures/scryfall_cards.json');f=json.loads(p.read_text())
for name in new:f[name]={key:cards[name].get(key) for key in ['name','mana_cost','type_line','power','toughness','keywords','oracle_text','colors']}
p.write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
