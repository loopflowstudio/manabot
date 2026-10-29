import json, re, sys, time, urllib.parse, urllib.request
d = json.load(open('scratch/conversion/before.json'))
def slug(n): return re.sub(r'_+','_',re.sub(r'[^a-z0-9]+','_',n.lower())).strip('_')
out = {}
for row in d:
    name = json.loads(re.search(r'name: ("(?:[^"\\]|\\.)*")', row['debug']).group(1))
    key = row['key']
    if key.endswith('.native'): key = key.split('.')[0] + '.' + slug(name)
    if key.startswith('token.'): continue
    q = urllib.parse.urlencode({'order': 'released', 'dir': 'asc', 'unique': 'prints', 'q': f'!"{name}" -is:promo'})
    req = urllib.request.Request('https://api.scryfall.com/cards/search?' + q, headers={'User-Agent': 'manabot-research/0.1', 'Accept': 'application/json'})
    for attempt in range(3):
        try:
            res = json.load(urllib.request.urlopen(req, timeout=30)); break
        except Exception as e:
            time.sleep(5)
    first = res['data'][0]
    out[key] = {'name': name, 'first_set': first['set'], 'released': first['released_at'], 'id': first['id']}
    if first['set'] != key.split('.')[0]:
        print('MISMATCH', key, first['set'], first['released_at'], first['set_type'], flush=True)
    time.sleep(0.3)
json.dump(out, open('scratch/conversion/first_printing.json', 'w'), indent=1, sort_keys=True)
print('done', len(out))
