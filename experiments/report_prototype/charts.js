// SVG chart primitives shared by the prototypes. Styling comes from CSS custom
// properties on the host page (--ink, --muted, --grid, --axis, --surface), so
// each prototype keeps its own visual language.
const Charts = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };

  let tipNode;
  function tip() {
    if (!tipNode) { tipNode = document.createElement('div'); tipNode.className = 'tip'; document.body.appendChild(tipNode); }
    return tipNode;
  }
  function showTip(evt, rows, title) {
    const t = tip();
    t.replaceChildren();
    if (title) { const h = document.createElement('div'); h.className = 'tip-title'; h.textContent = title; t.appendChild(h); }
    for (const r of rows) {
      const line = document.createElement('div'); line.className = 'tip-row';
      if (r.color) { const k = document.createElement('span'); k.className = 'tip-key'; k.style.background = r.color; if (r.dashed) k.classList.add('dashed'); if (r.dotted) { k.classList.add('dotted'); k.style.color = r.color; } line.appendChild(k); }
      const v = document.createElement('strong'); v.textContent = r.value; line.appendChild(v);
      const l = document.createElement('span'); l.className = 'tip-label'; l.textContent = r.label; line.appendChild(l);
      t.appendChild(line);
    }
    t.style.display = 'block';
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let left = evt.clientX + pad, top = evt.clientY + pad;
    if (left + w > innerWidth - 8) left = evt.clientX - w - pad;
    if (top + h > innerHeight - 8) top = evt.clientY - h - pad;
    t.style.left = left + 'px'; t.style.top = top + 'px';
  }
  function hideTip() { if (tipNode) tipNode.style.display = 'none'; }

  function ticks(lo, hi, count) {
    const span = hi - lo || 1, raw = span / count, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // series: [{label, color, dashed, points: [{x, y, lo, hi, missing, meta}]}]
  // opts: {height, yFormat, xFormat, xLabel, ref, whiskers, band, dots, compact, onPick}
  function line(host, series, opts = {}) {
    host.replaceChildren();
    const width = host.clientWidth || 600, height = opts.height || 220;
    const m = opts.compact ? {t: 8, r: 8, b: 22, l: 34} : {t: 12, r: 16, b: 34, l: 44};
    const svg = el('svg', {viewBox: `0 0 ${width} ${height}`, width, height, class: 'chart-svg', role: 'img'}, host);
    const all = series.flatMap(s => s.points.filter(p => !p.missing && p.x != null));
    if (!all.length) { const t = el('text', {x: width / 2, y: height / 2, 'text-anchor': 'middle', class: 'empty'}, svg); t.textContent = opts.empty || 'No recorded points for this selection'; return; }
    const xs = all.map(p => p.x);
    let ylo = Math.min(...all.map(p => p.lo ?? p.y)), yhi = Math.max(...all.map(p => p.hi ?? p.y));
    if (opts.ref != null) { ylo = Math.min(ylo, opts.ref); yhi = Math.max(yhi, opts.ref); }
    // A near-constant series gets ±5% of its level, not a microscopic range.
    const padY = Math.max((yhi - ylo) * 0.08, Math.abs(yhi) * 0.05, 1e-9);
    ylo = Math.max(opts.yMin ?? -Infinity, ylo - padY); yhi = Math.min(opts.yMax ?? Infinity, yhi + padY);
    const xlo = Math.min(...xs), xhi = Math.max(...xs) || 1;
    const X = v => m.l + (v - xlo) / ((xhi - xlo) || 1) * (width - m.l - m.r);
    const Y = v => m.t + (1 - (v - ylo) / ((yhi - ylo) || 1)) * (height - m.t - m.b);
    const yf = opts.yFormat || (v => v), xf = opts.xFormat || (v => v);

    for (const v of ticks(ylo, yhi, opts.compact ? 3 : 4)) {
      el('line', {x1: m.l, x2: width - m.r, y1: Y(v), y2: Y(v), class: 'grid'}, svg);
      const t = el('text', {x: m.l - 6, y: Y(v) + 3.5, 'text-anchor': 'end', class: 'tick'}, svg); t.textContent = yf(v);
    }
    for (const v of ticks(xlo, xhi, opts.compact ? 3 : 6)) {
      const t = el('text', {x: X(v), y: height - m.b + 14, 'text-anchor': 'middle', class: 'tick'}, svg); t.textContent = xf(v);
    }
    el('line', {x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b, class: 'axis'}, svg);
    if (opts.xLabel && !opts.compact) { const t = el('text', {x: width - m.r, y: height - 4, 'text-anchor': 'end', class: 'axis-label'}, svg); t.textContent = opts.xLabel; }
    if (opts.ref != null) el('line', {x1: m.l, x2: width - m.r, y1: Y(opts.ref), y2: Y(opts.ref), class: 'ref'}, svg);

    for (const s of series) {
      const pts = s.points.filter(p => !p.missing && p.x != null);
      const sparse = opts.whiskers || pts.length <= 5;
      if (opts.band !== false && !sparse && pts.length > 1 && pts[0].lo != null) {
        const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.x)},${Y(p.hi)}`).join('') + pts.slice().reverse().map(p => `L${X(p.x)},${Y(p.lo)}`).join('') + 'Z';
        el('path', {d, fill: s.color, opacity: 0.12, stroke: 'none'}, svg);
      }
      if (sparse && pts[0]?.lo != null) for (const p of pts) el('line', {x1: X(p.x), x2: X(p.x), y1: Y(p.lo), y2: Y(p.hi), stroke: s.color, 'stroke-width': 1.5, opacity: 0.6}, svg);
      // Break the line at recorded gaps; never bridge a missing evaluation.
      let d = '', open = false;
      for (const p of s.points) {
        if (p.missing || p.x == null) { open = false; continue; }
        d += `${open ? 'L' : 'M'}${X(p.x)},${Y(p.y)}`; open = true;
      }
      if (!sparse) el('path', {d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-dasharray': s.dotted ? '1 4' : s.dashed ? '5 4' : 'none', 'stroke-linecap': s.dotted ? 'round' : 'butt', 'stroke-linejoin': 'round', 'stroke-opacity': s.opacity ?? 1, 'stroke-width': s.width ?? 2}, svg);
      if (opts.dots !== false || sparse) for (const p of pts) el('circle', {cx: X(p.x), cy: Y(p.y), r: sparse ? 4 : 2.5, fill: sparse ? s.color : 'var(--surface)', stroke: s.color, 'stroke-width': sparse ? 0 : 1.5}, svg);
    }

    const cross = el('line', {y1: m.t, y2: height - m.b, class: 'cross', visibility: 'hidden'}, svg);
    const hit = el('rect', {x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: 'transparent'}, svg);
    const uniq = [...new Set(xs)].sort((a, b) => a - b);
    const near = px => uniq.reduce((b, v) => Math.abs(X(v) - px) < Math.abs(X(b) - px) ? v : b, uniq[0]);
    hit.addEventListener('pointermove', evt => {
      const box = svg.getBoundingClientRect(), px = (evt.clientX - box.left) * width / box.width;
      const xv = near(px);
      cross.setAttribute('x1', X(xv)); cross.setAttribute('x2', X(xv)); cross.setAttribute('visibility', 'visible');
      const rows = series.map(s => ({s, p: s.points.find(p => p.x === xv)}))
        .map(({s, p}) => ({color: s.color, dashed: s.dashed, dotted: s.dotted, label: s.label + (p?.meta ? ' · ' + p.meta : ''), value: !p ? 'not evaluated' : p.missing ? 'not recorded' : yf(p.y) + (p.lo != null ? `  [${yf(p.lo)}–${yf(p.hi)}]` : ''), y: p && !p.missing ? p.y : -Infinity}))
        .sort((a, b) => b.y - a.y);
      showTip(evt, rows, (opts.xLabel || 'x') + ' ' + xf(xv));
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    if (opts.onPick) hit.addEventListener('click', evt => {
      const box = svg.getBoundingClientRect(); opts.onPick(near((evt.clientX - box.left) * width / box.width));
    });
  }

  // Diverging blue (candidate ahead) ↔ red (candidate behind) around `mid`.
  function diverging(v, lo, mid, hi) {
    const css = getComputedStyle(document.documentElement);
    const neutral = css.getPropertyValue('--div-mid').trim() || '#f0efec';
    const pos = css.getPropertyValue('--div-pos').trim() || '#184f95';
    const neg = css.getPropertyValue('--div-neg').trim() || '#b4302f';
    const t = v >= mid ? Math.min(1, (v - mid) / (hi - mid)) : Math.min(1, (mid - v) / (mid - lo));
    return mix(neutral, v >= mid ? pos : neg, Math.pow(t, 0.85));
  }
  function mix(a, b, t) {
    const pa = hex(a), pb = hex(b);
    return `rgb(${pa.map((c, i) => Math.round(c + (pb[i] - c) * t)).join(',')})`;
  }
  function hex(h) { h = h.replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function inkOn(color) {
    const [r, g, b] = color.match(/\d+/g).map(Number);
    return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? 'var(--cell-ink-dark, #0b0b0b)' : '#ffffff';
  }

  return {el, line, ticks, showTip, hideTip, diverging, inkOn};
})();
