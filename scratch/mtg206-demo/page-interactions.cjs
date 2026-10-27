// Execute the generated page's actual script against a minimal DOM. No visual proof.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children = []; this.value = ''; this.dataset = {}; this.style = {}; this.events = {}; this.classList = { add() {} }; this.textContent = ''; this.innerHTML = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  get options() { return this.children; }
  querySelectorAll() { return this.children.flatMap(node => node.children); }
}
function openPage(path) {
  const html = fs.readFileSync(path, 'utf8');
  const payload = html.match(/<script id="card-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
  const nodes = new Map();
  const get = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  get('card-data').textContent = payload;
  vm.runInNewContext(script, {
    document: { getElementById: get, createElement: () => new Element() },
    Option: function(label, value) { this.textContent = label; this.value = value; },
    location: { hash: '', pathname: path }, history: { replaceState() {} },
    window: { addEventListener() {} }, URLSearchParams
  });
  return { get, data: JSON.parse(payload), input(id, value) { get(id).value = value; get(id).events.input(); } };
}
const mixed = openPage('scratch/mtg206-demo/mixed-cards.html');
const summary = () => mixed.get('summary').innerHTML;
const rowText = () => mixed.get('rows').children[0].children[0].textContent;
assert.match(summary(), /1 replay failed/);
mixed.input('position', 'draw');
assert.match(summary(), /Eligible player-games <strong>0/);
assert.match(summary(), /Win rate <strong>—/);
assert.match(summary(), /Runs <strong>1/);
assert.match(summary(), /1 replay failed/);
assert.equal(rowText(), 'No eligible games. Card rates are unavailable for this selection.');
const excluded = {summary: summary(), message: rowText()};
mixed.input('position', 'play');
mixed.input('deck', 'green');
assert.equal(mixed.get('summary').textContent, 'No games match these filters.');
assert.equal(rowText(), 'No games match these filters.');
mixed.input('deck', 'red');
mixed.input('search', 'no-such-card');
assert.equal(rowText(), 'No cards match.');
mixed.input('search', '');
assert.notEqual(rowText(), 'No cards match.');
const real = openPage('scratch/mtg206-demo/l4-cards.html');
assert.ok(real.data.variants['||||'].games > 0);
real.input('subject', 'verify-one');
real.input('position', 'draw');
assert.match(real.get('summary').innerHTML, /Eligible player-games/);
const evidence = {proof: 'Generated JavaScript with injected DOM; no browser rendering', mixed: excluded, checks: ['excluded provenance and null rates', 'absent combination', 'search hides cards', 'clear search restores rows', 'real L4 subject and position filters'], l4: {variants: Object.keys(real.data.variants).length, base: {...real.data.variants['||||'], cards: real.data.variants['||||'].cards.length}, filteredSummary: real.get('summary').innerHTML}};
fs.writeFileSync('scratch/mtg206-demo/page-interactions.json', JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));
