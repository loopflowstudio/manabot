"""Materialize the inspected whole-card partition; no runtime support inference."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    cards = json.loads((HERE/'cards.json').read_text())
    rows = [json.loads(line) for line in (HERE.parent/'tla-card-ledger.jsonl').read_text().splitlines()]
    native = {row['card_name'] for row in rows if row['kind']=='card_face' and row['native_registration']}
    # Lists are the design review's explicit assignments, not keyword support grades.
    simple = {17,41,56,124,130,136,156,161,189,193,265,267,269,271,272,273,274,275,277,279,280}
    casting = {1,4,5,7,8,12,32,38,39,52,53,65,70,73,81,83,104,110,123,133,148,152,153,154,163,178,191,192,194,195,197,208,224,226,227,238,240,254,270,281}
    characteristics = {2,23,33,47,50,64,68,94,103,108,132,144,147,171,177,179,181,190,200,231,235,247,261,262,263,264,266,268,276,278}
    copying = {105,114,150}
    faces = {27,48,61,102,107,117,126,145,185,186,203,204,207,211,220,255}
    control = {69}
    result: dict[str, dict[str, object]] = {}
    for card in cards:
        number = int(card['collector_number'])
        texts = ' '.join(face.get('oracle_text','') for face in card.get('card_faces',[card]))
        batch = 'T1' if card['name'] in native or number in simple else 'T4'
        if number in casting or 'earthbend' in texts.lower(): batch = 'T2'
        if number in characteristics or any(word in card['type_line'] for word in ('Aura','Equipment','Vehicle')): batch = 'T3'
        if number in copying: batch = 'T5'
        if number in faces: batch = 'T6'
        if number in control: batch = 'T7'
        # Trigger placement/history or copying is the last prerequisite, even when
        # the spell also contains a simple earthbend/payment/characteristic clause.
        if number in {9,10,45,77,87,88,97,99,118,122,137,139,142,151,158,169,173,174,175,209,210,215,218,223,225,228,230,239,242,244,246,253,259}: batch='T4'
        result[card['oracle_id']]={'card_name':card['name'],'collector_number':card['collector_number'],'planned_pr':batch,
            'admission_unit':'all faces and all clauses together; later whole-file interactions are gated in T7'}
    assert len(result)==286
    (HERE/'card-batches.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for batch in ['T1','T2','T3','T4','T5','T6','T7']:
        group=[r for r in result.values() if r['planned_pr']==batch]
        print(batch,len(group),'; '.join(str(r['collector_number'])+' '+str(r['card_name']) for r in group))


if __name__=='__main__':
    main()
