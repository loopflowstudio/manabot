"""One-off: rewrite removed continuous-effect variants in card set files as
Effect::ApplyContinuous constructors (T3, continuous effects as records)."""
import re, sys, pathlib

def find_block(s, start):
    # s[start] is '{'; return index after matching '}'
    depth=0
    for i in range(start, len(s)):
        c=s[i]
        if c in '{([': depth+=1
        elif c in '})]':
            depth-=1
            if depth==0: return i+1
    raise ValueError

def fields(body):
    # body inside braces; split top-level "name: expr" by commas
    out={}; depth=0; cur=''
    parts=[]
    for c in body:
        if c in '{([': depth+=1
        if c in '})]': depth-=1
        if c==',' and depth==0:
            parts.append(cur); cur=''
        else: cur+=c
    if cur.strip(): parts.append(cur)
    for p in parts:
        p=p.strip()
        if not p: continue
        k,v=p.split(':',1)
        out[k.strip()]=v.strip()
    return out

def fixed_opt(v):
    v=v.strip()
    if v=='None': return 'None'
    m=re.fullmatch(r'Some\((.*)\)',v,re.S)
    return f'Some(EffectValue::Fixed({m.group(1)}))'

EOT='Effect::until_end_of_turn'
IND='Effect::indefinitely'
def conv(name, f):
    T=lambda: f"AffectedSpec::Target({f['target']})"
    if name=='BuffTarget': return f"{EOT}({T()}, Modification::pump({f['power']}, {f['toughness']}))"
    if name=='BuffTargetValue': return f"{EOT}({T()}, Modification::ModifyPt {{ power: {f['power']}, toughness: EffectValue::Fixed({f['toughness']}) }})"
    if name=='GrantKeywordsToTarget': return f"{EOT}({T()}, Modification::AddKeywords({f['keywords']}))"
    if name=='CantBeBlockedTarget': return f"{EOT}({T()}, Modification::CantBeBlocked)"
    if name=='GrantFirebendingUntilEot': return f"{EOT}(AffectedSpec::EachYouControl(CardPredicate::creature()), Modification::Firebending({f['amount']}))"
    if name=='GrantFirebendingToTarget': return f"{EOT}({T()}, Modification::Firebending({f['amount']}))"
    if name=='NoMaximumHandSize': return f"{IND}(AffectedSpec::You, Modification::NoMaximumHandSize)"
    if name=='BuffTriggeringPermanentValue': return f"{EOT}(AffectedSpec::TriggeringPermanent, Modification::ModifyPt {{ power: {f['power']}, toughness: {f['toughness']} }})"
    if name=='ModifyTriggeringPermanent': return f"Effect::PutCountersOnTriggeringPermanent {{ count: {f['counters']} }}, {EOT}(AffectedSpec::TriggeringPermanent, Modification::AddKeywords({f['keywords']}))"
    if name=='KeepUnspentRedMana': return f"{EOT}(AffectedSpec::You, Modification::KeepUnspentMana(Color::Red))"
    if name=='GrantKeywordsControlled': return f"{EOT}(AffectedSpec::EachYouControl({f['predicate']}), Modification::AddKeywords({f['keywords']}))"
    if name=='ModifyControlled': return f"{EOT}(AffectedSpec::EachYouControl({f['predicate']}), Modification::pump({f['power']}, {f['toughness']}))"
    if name=='SetBasePtControlled': return f"{EOT}(AffectedSpec::EachYouControl({f['predicate']}), Modification::SetBasePt {{ power: Some({f['power']}), toughness: Some({f['toughness']}) }})"
    if name=='AddSubtypesControlled': return f"{EOT}(AffectedSpec::EachYouControl({f['predicate']}), Modification::AddSubtypes({f['subtypes']}))"
    if name=='ModifyUntilEot': return f"{EOT}(AffectedSpec::Source, Modification::pump({f['power_delta']}, {f['toughness_delta']}))"
    if name=='SetBasePtUntilEot': return f"{EOT}(AffectedSpec::Source, Modification::SetBasePt {{ power: {fixed_opt(f['power'])}, toughness: {fixed_opt(f['toughness'])} }})"
    if name=='SetBasePtTarget': return f"{EOT}({T()}, Modification::SetBasePt {{ power: {fixed_opt(f['power'])}, toughness: {fixed_opt(f['toughness'])} }})"
    if name in ('BecomeCreatureSource','BecomeCreatureTarget'):
        who='AffectedSpec::Source' if name.endswith('Source') else T()
        ctor=EOT if f['duration'].endswith('EndOfTurn') else IND
        return f"{ctor}({who}, Modification::BecomeArtifactCreature)"
    if name=='GrantKeywordsToSource': return f"{EOT}(AffectedSpec::Source, Modification::AddKeywords({f['keywords']}))"
    if name=='AddSubtypeToTarget': return f"{EOT}({T()}, Modification::AddSubtypes(vec![{f['subtype']}]))"
    if name=='ExileIfDiesThisTurn': return f"{EOT}({T()}, Modification::ExileIfWouldDie)"
    if name=='CantBeBlockedThisTurnSource': return f"{EOT}(AffectedSpec::Source, Modification::CantBeBlocked)"
    raise KeyError(name)

NAMES="CantBeBlockedTarget GrantFirebendingUntilEot GrantFirebendingToTarget NoMaximumHandSize BuffTargetValue BuffTriggeringPermanentValue KeepUnspentRedMana GrantKeywordsControlled ModifyControlled SetBasePtControlled AddSubtypesControlled ModifyUntilEot SetBasePtUntilEot SetBasePtTarget BecomeCreatureSource BecomeCreatureTarget GrantKeywordsToSource GrantKeywordsToTarget AddSubtypeToTarget ExileIfDiesThisTurn CantBeBlockedThisTurnSource BuffTarget ModifyTriggeringPermanent".split()
pat=re.compile(r'Effect::('+'|'.join(sorted(NAMES,key=len,reverse=True))+r')\b')
for path in sys.argv[1:]:
    p=pathlib.Path(path); s=p.read_text(); n=0
    while True:
        m=pat.search(s)
        if not m: break
        name=m.group(1); j=m.end()
        rest=s[j:]
        if rest.lstrip().startswith('{') :
            b=j+(len(rest)-len(rest.lstrip()))
            e=find_block(s,b)
            f=fields(s[b+1:e-1])
        else:
            e=j; f={}
        s=s[:m.start()]+conv(name,f)+s[e:]; n+=1
    if n:
        used=[x for x in ['AffectedSpec','Modification','EffectValue','CardPredicate','Color'] if x+'::' in s or x+'(' in s]
        imports={'AffectedSpec':'crate::state::ability::AffectedSpec','Modification':'crate::state::effect::Modification','EffectValue':'crate::state::ability::EffectValue','CardPredicate':'crate::state::predicate::CardPredicate','Color':'crate::state::mana::Color'}
        lines=[f"use {imports[x]};" for x in used]
        # insert after first 'use crate::' statement block end
        k=s.index('\n\n', s.index('use crate::'))
        s=s[:k]+'\n'+'\n'.join(lines)+s[k:]
        p.write_text(s)
        print(path,n)
