import json, re, sys, time, urllib.request
d = json.load(open('scratch/conversion/before.json'))
def slug(n): return re.sub(r'_+','_',re.sub(r'[^a-z0-9]+','_',n.lower())).strip('_')
TOKEN_SET = {'token.human_soldier': 'tiko', 'token.treasure': 'tafr'}
rows = []
for row in d:
    name = json.loads(re.search(r'name: ("(?:[^"\\]|\\.)*")', row['debug']).group(1))
    key = row['key']
    if key.endswith('.native'):
        key = key.split('.')[0] + '.' + slug(name)
    s = TOKEN_SET.get(key, 'ttla') if key.startswith('token.') else key.split('.')[0]
    rows.append((key, name, s))
out = {}
for i in range(0, len(rows), 70):
    chunk = rows[i:i+70]
    body = json.dumps({'identifiers': [{'name': n, 'set': s} for _, n, s in chunk]}).encode()
    req = urllib.request.Request('https://api.scryfall.com/cards/collection', data=body,
        headers={'User-Agent': 'manabot-research/0.1', 'Accept': 'application/json', 'Content-Type': 'application/json'})
    res = json.load(urllib.request.urlopen(req, timeout=60))
    found = {(c['name'], c['set']): c['id'] for c in res['data']}
    for key, n, s in chunk:
        if (n, s) in found: out[key] = found[(n, s)]
    print('not found:', res['not_found'], file=sys.stderr)
    time.sleep(0.5)
json.dump(out, open('scratch/conversion/ids.json', 'w'), indent=1, sort_keys=True)
print(len(out), len(rows))
