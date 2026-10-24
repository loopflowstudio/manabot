// Shared query layer for the three prototypes: filter cells, aggregate
// W/D/L into a score with a 95% Wilson interval, build curves, matrices and
// marginals. Presentation lives in each page; this file owns no DOM.
const M = (() => {
  const R = window.REPORT;
  const C = {RUN: 0, EVAL: 1, ANCHOR: 2, DECK: 3, OPP: 4, SEAT: 5, W: 6, D: 7, L: 8};

  function experiment(id) { return R.experiments.find(e => e.id === id) || R.experiments[0]; }

  function agg(rows) {
    let w = 0, d = 0, l = 0;
    for (const c of rows) { w += c[C.W]; d += c[C.D]; l += c[C.L]; }
    const n = w + d + l;
    if (!n) return null;
    const p = (w + d / 2) / n, z = 1.96, z2 = z * z;
    const mid = (p + z2 / (2 * n)) / (1 + z2 / n);
    const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
    return {w, d, l, n, score: p, lo: Math.max(0, mid - half), hi: Math.min(1, mid + half)};
  }

  // f: {anchor, seat: 'all'|0|1, decks?: Set, opps?: Set}
  function keep(c, f) {
    if (c[C.ANCHOR] !== f.anchor) return false;
    if (f.seat !== 'all' && c[C.SEAT] !== f.seat) return false;
    if (f.decks && f.decks.size && !f.decks.has(c[C.DECK])) return false;
    if (f.opps && f.opps.size && !f.opps.has(c[C.OPP])) return false;
    return true;
  }

  // ev may be one evaluation index or an array of them (a pooled window).
  function cellsAt(e, run, ev, f) {
    const evs = new Set(Array.isArray(ev) ? ev : [ev]);
    return e.cells.filter(c => c[C.RUN] === run && evs.has(c[C.EVAL]) && keep(c, f));
  }

  // Seat filter on a run with unsplit records yields null, never a guess.
  function curve(e, run, f) {
    return e.runs[run].evals.map((ev, i) => ({...ev, index: i, ...(agg(cellsAt(e, run, i, f)) || {missing: true})}));
  }

  function matrix(e, run, ev, f) {
    const k = e.decks.length, rows = cellsAt(e, run, ev, {...f, decks: null, opps: null});
    return Array.from({length: k}, (_, d) => Array.from({length: k}, (_, o) =>
      agg(rows.filter(c => c[C.DECK] === d && c[C.OPP] === o))));
  }

  function marginal(e, run, ev, f, by) {
    const col = by === 'opp' ? C.OPP : C.DECK, rows = cellsAt(e, run, ev, f);
    return e.decks.map((_, i) => agg(rows.filter(c => c[col] === i)));
  }

  function x(ev, axis) { return axis === 'hours' ? ev.hours : axis === 'decisions' ? ev.decisions : ev.updates; }

  function nearestEval(e, run, updates) {
    const evals = e.runs[run].evals;
    let best = 0;
    evals.forEach((v, i) => { if (Math.abs(v.updates - updates) < Math.abs(evals[best].updates - updates)) best = i; });
    return best;
  }

  const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
  const pts = v => v == null ? '—' : ((v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(1));
  const num = v => v == null ? '—' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : String(Math.round(v));
  const deckLabel = (e, i) => e.deck_names[i];
  const matchLabel = (e, d, o) => `${e.deck_names[d]} → ${e.deck_names[o]}`;

  // URL hash state: flat key=value pairs.
  function readHash(defaults) {
    const out = {...defaults};
    for (const part of location.hash.slice(1).split('&')) {
      const [k, v] = part.split('=');
      if (k && v !== undefined && k in defaults) out[k] = decodeURIComponent(v);
    }
    return out;
  }
  function writeHash(state) {
    history.replaceState(null, '', '#' + Object.entries(state).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&'));
  }

  return {R, C, experiment, agg, curve, matrix, marginal, cellsAt, x, nearestEval, pct, pts, num, deckLabel, matchLabel, readHash, writeHash};
})();
