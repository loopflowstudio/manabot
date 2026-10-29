import json
from pathlib import Path
p=Path('content/semantic/current/definitions.json')
defs=json.loads(p.read_text()); cards={c['name']:c for c in json.load(open('scratch/tla-evidence/cards.json'))}
keys={d['registry_name']:d['key'] for d in defs}
new=[]
def add(name, programs, keywords=(), **extra):
 c=cards[name]; types=c['type_line'].split(' — '); words=types[0].lower().split()
 char={'types':[w for w in words if w!='legendary'], 'subtypes':types[1].split() if len(types)>1 else [], 'keywords':list(keywords), **extra}
 if c.get('mana_cost'):char['mana_cost']=c['mana_cost']
 if 'power' in c:char.update(power=int(c['power']),toughness=int(c['toughness']))
 key='tla.'+name.lower().replace('-','_').replace(' ','_').replace("'",'')
 defs.append({'key':key,'registry_name':name,'text_box':c['oracle_text'],'characteristics':char,'programs':programs});new.append(name)
def trigger(key,event,ops,**fields):return {'key':key,'kind':'triggered','trigger':{'event':event,'subject':{'kind':'this'},**fields},'ops':ops}
def activated(key,cost,ops,targets=None):
 p={'key':key,'kind':'activated','cost':cost,'ops':ops}
 if targets is not None:p['targets']=targets
 return p
def target(role,kind,**fields):return {'role':role,'selector':{'kind':kind,**fields},'min':1,'max':1}
def spell(name,ops,targets):return {'key':'spell','kind':'spell','cost':{'mana':cards[name]['mana_cost']},'targets':targets,'ops':ops}
add('Avatar Enthusiasts',[{'key':'ally_enters','kind':'triggered','trigger':{'event':'enters_battlefield','subject':{'kind':'another_you_control','predicate':{'kind':'subtype','value':'Ally'}}},'ops':[{'op':'put_counters','target':'self','count':1}]}])
add('Glider Kids',[trigger('scry','enters_battlefield',[{'op':'scry','count':1}])],['flying'])
add('Earth Kingdom Protectors',[activated('protect',{'mana':'','sacrifice_source':True},[{'op':'grant_keywords','target':'ally','keywords':['indestructible'],'duration':'end_of_turn'}],[target('ally','permanent_you_control',predicate={'kind':'subtype','value':'Ally'})])],['vigilance'])
add('Water Tribe Captain',[activated('rally',{'mana':'5'},[{'op':'modify_controlled','predicate':{'kind':'card_type','value':'creature'},'power':1,'toughness':1}])])
add('Iguana Parrot',[trigger('prowess','cast_spell',[{'op':'modify_pt','target':'self','power':1,'toughness':1,'duration':'end_of_turn'}],predicate={'kind':'not_card_types','values':['creature']})],['flying','vigilance'])
add('Boar-q-pine',[trigger('grow','cast_spell',[{'op':'put_counters','target':'self','count':1}],predicate={'kind':'not_card_types','values':['creature']})])
add('Fire Sages',[trigger('firebend','attacks',[{'op':'add_mana','mana':'R','until':'end_of_combat'}]),activated('grow',{'mana':'1RR'},[{'op':'put_counters','target':'self','count':1}])])
add('Treetop Freedom Fighters',[trigger('reinforcements','enters_battlefield',[{'op':'create_token','definition_ref':keys['Ally'],'count':1}])],['haste'])
add('Yuyan Archers',[trigger('rummage','enters_battlefield',[{'op':'optional_discard_draw','count':1}])],['reach'])
add('Cunning Maneuver',[spell('Cunning Maneuver',[{'op':'modify_pt','target':'creature','power':3,'toughness':1,'duration':'end_of_turn'},{'op':'create_token','definition_ref':keys['Clue'],'count':1}],[target('creature','creature')])])
add('Pillar Launch',[spell('Pillar Launch',[{'op':'modify_pt','target':'creature','power':2,'toughness':2,'duration':'end_of_turn'},{'op':'grant_keywords','target':'creature','keywords':['reach'],'duration':'end_of_turn'},{'op':'untap','target':'creature'}],[target('creature','creature')])])
add('Rocky Rebuke',[spell('Rocky Rebuke',[{'op':'deal_power_damage','sources':['source'],'target':'victim'}],[target('source','creature_you_control'),target('victim','creature_opponent_controls')])])
for name, colors in [('Airship Engine Room','UR'),('Boiling Rock Prison','BR'),('Foggy Bottom Swamp','BG'),('Kyoshi Village','GW'),('Meditation Pools','GU'),('Misty Palms Oasis','WB'),('North Pole Gates','WU'),('Omashu City','RG'),("Serpent's Pass",'UB'),('Sun-Blessed Peak','RW')]:
 add(name,[*({'key':'mana_'+c,'kind':'mana','ops':[{'op':'add_mana','mana':c}]} for c in colors),activated('draw',{'mana':'4','tap_source':True,'sacrifice_source':True},[{'op':'draw_cards','count':1}])],enters_tapped=True)
add('Rumble Arena',[trigger('scry','enters_battlefield',[{'op':'scry','count':1}]),{'key':'colorless','kind':'mana','ops':[{'op':'add_mana','mana':'C'}]},activated('filter',{'mana':'1','tap_source':True},[{'op':'modal','modes':[[{'op':'add_mana','mana':c}] for c in 'WUBRG']}])],['vigilance'])
p.write_text(json.dumps(sorted(defs,key=lambda x:x['key']),indent=2,ensure_ascii=False)+'\n')
Path('scratch/new-cards.json').write_text(json.dumps(new,indent=2)+'\n')
# Independent shell snapshot, transcribed before generation.
f=Path('managym/tests/fixtures/scryfall_cards.json'); fixture=json.loads(f.read_text())
for name in new:
 c=cards[name]; fixture[name]={k:c.get(k) for k in ['name','mana_cost','type_line','power','toughness','keywords','oracle_text','colors']}
f.write_text(json.dumps(fixture,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
