/* Selection and aggregation share one owner in the offline report and Node tests. */
"use strict";
const TrainingCohort = (() => {
  const excludedKinds = new Set(["benchmark", "smoke", "fixture", "test", "evaluation"]);
  const known = x => typeof x === "number" && Number.isFinite(x) && x >= 0;
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  function eligible(run, includeBenchmarks) {
    return includeBenchmarks || !excludedKinds.has(run.kind.toLowerCase());
  }
  function largest(runs, metric, count, includeBenchmarks) {
    return runs.filter(r => eligible(r, includeBenchmarks) && known(r[metric]))
      .sort((a, b) => b[metric] - a[metric] || a.id.localeCompare(b.id))
      .slice(0, count).map(r => r.id);
  }
  function union(intervals) {
    let seconds = 0, end = -Infinity;
    for (const i of [...intervals].sort((a, b) => a.start - b.start)) {
      seconds += Math.max(0, i.end - Math.max(end, i.start));
      end = Math.max(end, i.end);
    }
    return seconds;
  }
  function flamegraph(runs) {
    // Merge only identical hierarchy paths. Parents already include children.
    const root = {name: "Recorded training time", seconds: 0, children: []};
    let covered = 0;
    function merge(target, source) {
      target.seconds += source.seconds;
      for (const child of source.children) {
        let match = target.children.find(n => n.name === child.name);
        if (!match) { match = {name: child.name, seconds: 0, children: []}; target.children.push(match); }
        merge(match, child);
      }
    }
    for (const run of runs) {
      if (!known(run.elapsed_seconds) || !run.profiles.length || run.profiles.some(p => !p.root)) continue;
      covered++;
      for (const profile of run.profiles) merge(root, profile.root);
    }
    const nodes = [];
    function layout(node, left, depth) {
      if (node.seconds <= 0) return;
      nodes.push({name: node.name, seconds: node.seconds, left, depth, width: node.seconds / root.seconds});
      for (const child of node.children) {
        layout(child, left, depth + 1);
        left += child.seconds / root.seconds;
      }
    }
    layout(root, 0, 0);
    return {seconds: root.seconds, covered, nodes};
  }
  function aggregate(runs) {
    const phases = {}, groups = new Map(), calendars = new Map();
    const covered = runs.filter(r => known(r.elapsed_seconds));
    for (const r of covered) {
      for (const name of r.phase_coverage ?? []) phases[name] ??= 0;
      for (const [name, seconds] of Object.entries(r.phases)) phases[name] = (phases[name] || 0) + seconds;
    }
    for (const r of runs) {
      // Unknown hardware/recipe cannot establish cross-run comparability.
      const key = JSON.stringify([r.recipe, r.hardware,
        r.hardware.includes("unknown") || !r.recipe ? r.id : null]);
      if (!groups.has(key)) groups.set(key, {recipe: r.recipe, hardware: r.hardware, runs: []});
      groups.get(key).runs.push(r);
      for (const i of r.intervals) {
        if (!calendars.has(i.basis)) calendars.set(i.basis, []);
        calendars.get(i.basis).push(i);
      }
    }
    const intervals = runs.flatMap(r => r.intervals);
    const values = metric => runs.filter(r => known(r[metric])).map(r => r[metric]);
    const metrics = {};
    for (const metric of ["active_seconds", "elapsed_seconds", "learner_transitions", "updates", "environment_steps", "cost_usd"]) {
      const xs = values(metric).sort((a, b) => a - b), n = xs.length;
      metrics[metric] = {total: n ? sum(xs) : null, count: n,
        min: n ? xs[0] : null, max: n ? xs[n - 1] : null,
        median: n ? (xs[Math.floor((n - 1) / 2)] + xs[Math.floor(n / 2)]) / 2 : null};
    }
    const seconds = metrics.elapsed_seconds.total;
    return {count: runs.length, attempts: sum(runs.map(r => r.members.length)), metrics,
      phases: Object.entries(phases).map(([name, value]) => ({name, seconds: value,
        share: seconds > 0 ? value / seconds : null,
        covered: name === "Unattributed" ? covered.length : covered.filter(r => (r.phase_coverage ?? []).includes(name)).length})),
      rates: [...groups.values()].map(g => {
        const matched = g.runs.filter(r => known(r.learner_transitions) && r.active_seconds > 0);
        return {recipe: g.recipe, hardware: g.hardware, covered: matched.length, count: g.runs.length,
          rate: matched.length ? sum(matched.map(r => r.learner_transitions)) / sum(matched.map(r => r.active_seconds)) : null};
      }),
      calendars: [...calendars].map(([basis, rows]) => ({basis, count: rows.length,
        union: union(rows), additive: sum(rows.map(i => i.end - i.start))})),
      span: intervals.length ? Math.max(...intervals.map(i => i.end)) - Math.min(...intervals.map(i => i.start)) : null};
  }
  function selection(data, ids) {
    return {schema_version: 1, snapshot: data.snapshot, selected: [...ids].sort(),
      sources: data.sources, rule: "Explicit logical-run IDs in this evidence snapshot"};
  }
  function restore(data, saved) {
    if (saved.schema_version !== 1 || saved.snapshot !== data.snapshot || !Array.isArray(saved.selected)
        || saved.selected.some(id => !data.runs.some(r => r.id === id)))
      throw new Error("Selection belongs to different evidence. Open its original report snapshot.");
    return new Set(saved.selected);
  }
  return {known, eligible, largest, union, flamegraph, aggregate, selection, restore};
})();
if (typeof module !== "undefined") module.exports = TrainingCohort;
if (typeof document !== "undefined") {
  const D = JSON.parse(document.getElementById("cohort-data").textContent);
  const $ = id => document.getElementById(id), fmt = n => TrainingCohort.known(n) ? n.toLocaleString(undefined, {maximumFractionDigits: 3}) : "unavailable";
  const labels = {active_seconds: "Active training seconds", elapsed_seconds: "Recorded training seconds", learner_transitions: "Retained learner transitions", updates: "Retained learning iterations", environment_steps: "Recorded environment steps", cost_usd: "Measured cost (USD)"};
  const el = (tag, text) => { const n = document.createElement(tag); if (text != null) n.textContent = text; return n; };
  let selected = new Set(D.selected ?? TrainingCohort.largest(D.runs, "active_seconds", D.top, D.include_benchmarks));
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has("snapshot")) {
    try { selected = TrainingCohort.restore(D, {schema_version: 1, snapshot: hash.get("snapshot"), selected: hash.getAll("run")}); }
    catch (error) { $("selection-message").textContent = error.message; selected = new Set(); }
  }
  $("include-benchmarks").checked = D.include_benchmarks;
  $("top").value = D.top;
  $("stamp").textContent = `Evidence fetched ${D.generated_at}. ${D.runs.length} logical runs discovered. Snapshot ${D.snapshot.slice(0, 12)}.`;
  for (const warning of D.warnings) $("source-warnings").append(el("p", warning));
  function candidates() {
    const query = $("search").value.toLowerCase();
    return D.runs.filter(r => TrainingCohort.eligible(r, $("include-benchmarks").checked)
      && [r.id, r.label, r.experiment, r.hardware, r.model, r.status, r.kind, ...(r.source_uris ?? [])].join(" ").toLowerCase().includes(query));
  }
  function table(headings, rows) {
    const t = el("table"), head = el("tr"); headings.forEach(h => head.append(el("th", h)));
    const thead = el("thead"); thead.append(head); t.append(thead);
    const body = el("tbody"); rows.forEach(cells => { const row = el("tr"); cells.forEach(c => row.append(el("td", c))); body.append(row); });
    t.append(body); return t;
  }
  function drawPicker() {
    const key = $("sort").value;
    const rows = candidates().sort((a, b) => (b[key] ?? -1) - (a[key] ?? -1) || a.id.localeCompare(b.id));
    const body = $("run-rows"); body.replaceChildren();
    rows.forEach(r => {
      const row = el("tr"), cell = el("td"), check = el("input"); check.type = "checkbox";
      check.checked = selected.has(r.id); check.setAttribute("aria-label", `Include ${r.id}`);
      check.onchange = () => { check.checked ? selected.add(r.id) : selected.delete(r.id); draw(); };
      cell.append(check); row.append(cell);
      [r.label, r.id, r.kind, r.model, r.hardware, r.seed ?? "—", r.status,
        r.intervals.length ? new Date(Math.min(...r.intervals.map(i => i.start)) * 1000).toISOString() : "unavailable",
        fmt(r.active_seconds), fmt(r.learner_transitions), fmt(r.updates), fmt(r.cost_usd),
        `${r.intervals.length} lanes; ${r.warnings.length} warnings`].forEach(v => row.append(el("td", v)));
      body.append(row);
    });
    $("shown").textContent = `${rows.length} runs shown. Filters do not remove already selected runs.`;
  }
  function draw() {
    drawPicker();
    const runs = D.runs.filter(r => selected.has(r.id)), summary = TrainingCohort.aggregate(runs);
    const graph = TrainingCohort.flamegraph(runs), chart = $("flamegraph");
    chart.replaceChildren();
    $("flamegraph-coverage").textContent = `${runs.length} selected runs · ${graph.covered} with recorded phase trees · ${fmt(graph.seconds)} accumulated seconds.`;
    if (graph.nodes.length) {
      const svgEl = (tag, attrs) => {
        const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value);
        return n;
      };
      const svg = svgEl("svg", {viewBox: `0 0 1200 ${(1 + Math.max(...graph.nodes.map(n => n.depth))) * 46}`, role: "img", "aria-label": "Selected training time flamegraph"});
      const colors = {Collection: "#53758a", Learning: "#497e71", "Stage exports": "#9b703d", Unattributed: "#77716a"};
      for (const n of graph.nodes) {
        const group = svgEl("g", {}), label = `${n.name}: ${fmt(n.seconds)} s · ${(n.width * 100).toFixed(2)}% of recorded time`;
        const title = svgEl("title", {}); title.textContent = label;
        group.append(title, svgEl("rect", {x: n.left * 1200, y: n.depth * 46, width: n.width * 1200, height: 44, fill: colors[n.name] ?? "#334b5b", stroke: "#faf9f6", "stroke-width": 1}));
        if (n.width * 1200 > 60) {
          const text = svgEl("text", {x: n.left * 1200 + 7, y: n.depth * 46 + 27, fill: "white", "font-size": 15});
          const limit = Math.floor((n.width * 1200 - 14) / 8);
          text.textContent = label.length > limit ? label.slice(0, Math.max(0, limit - 1)) + "…" : label;
          group.append(text);
        }
        svg.append(group);
      }
      chart.append(svg);
    } else {
      chart.append(el("p", runs.length ? "These runs have no recorded phase hierarchy. Select runs with timing records below; missing measurements cannot form a flamegraph." : "No runs selected. Select runs below to display their flamegraph."));
      const ids = TrainingCohort.largest(D.runs.filter(r => r.profiles.some(p => p.root)), "active_seconds", D.top, true);
      if (ids.length) {
        const button = el("button", "Show available timing runs (may include benchmarks)");
        button.onclick = () => { selected = new Set(ids); $("include-benchmarks").checked = true; draw(); };
        chart.append(button);
      }
    }
    $("summary").replaceChildren(el("p", `${runs.length} selected logical runs · ${summary.attempts} recorded attempts/writers. Partial coverage is reported below.`));
    if (!runs.length) $("summary").append(el("p", "No runs selected. Runs without active-time evidence remain available for manual selection or another sort. Labeled benchmarks are excluded by default."));
    $("summary").append(el("p", [...new Set(runs.map(r => r.status))].map(status => `${status}: ${runs.filter(r => r.status === status).length}`).join(" · ")));
    $("summary").append(table(["Quantity", "Observed total", "Covered runs", "Median", "Minimum", "Maximum"],
      Object.entries(summary.metrics).map(([key, value]) => [labels[key], fmt(value.total), `${value.count}/${runs.length}`, fmt(value.median), fmt(value.min), fmt(value.max)])));
    $("summary").append(el("p", `Active training hours: ${fmt(summary.metrics.active_seconds.total == null ? null : summary.metrics.active_seconds.total / 3600)}. Rates below use only runs with both work and active time.`));
    $("phases").replaceChildren(table(["Exclusive phase", "Seconds", "Share of covered recorded time", "Phase coverage"], summary.phases.map(p => [p.name, fmt(p.seconds), p.share == null ? "unavailable" : `${(100 * p.share).toFixed(2)}%`, `${p.covered}/${runs.length} runs`])));
    $("rates").replaceChildren(table(["Recipe / job", "Hardware", "Covered runs", "Retained learner transitions / active second"], summary.rates.map(g => [g.recipe || "unknown", g.hardware, `${g.covered}/${g.count}`, fmt(g.rate)])));
    $("timeline").replaceChildren(el("p", `First-to-last observed calendar span: ${fmt(summary.span)} s. This includes gaps.`));
    $("timeline").append(table(["Interval basis", "Lanes", "Calendar union (s)", "Additive lane duration (s)"], summary.calendars.map(c => [c.basis, c.count, fmt(c.union), fmt(c.additive)])));
    const intervals = runs.flatMap(r => r.intervals), origin = Math.min(...intervals.map(i => i.start));
    if (intervals.length) $("timeline").append(el("p", `Calendar origin: ${new Date(origin * 1000).toISOString()}`));
    intervals.sort((a, b) => a.start - b.start).forEach(i => {
      const label = el("p", `${i.label} · ${i.status} · ${i.basis} · ${fmt(i.end - i.start)} s`);
      const lane = el("div"), segment = el("div"); lane.className = "lane"; segment.className = "segment";
      segment.style.marginLeft = `${summary.span ? 100 * (i.start - origin) / summary.span : 0}%`;
      segment.style.width = `${summary.span ? 100 * (i.end - i.start) / summary.span : 0}%`;
      lane.append(segment); $("timeline").append(label, lane);
    });
    $("warnings").replaceChildren();
    runs.forEach(r => r.warnings.forEach(w => $("warnings").append(el("p", `${r.id}: ${w}`))));
    document.querySelectorAll("[data-detail]").forEach(n => { n.hidden = !selected.has(D.runs[Number(n.dataset.detail)].id); });
    const state = new URLSearchParams({snapshot: D.snapshot}); [...selected].sort().forEach(id => state.append("run", id));
    history.replaceState(null, "", location.pathname + location.search + "#" + state);
  }
  $("select-top").onclick = () => { selected = new Set(TrainingCohort.largest(candidates(), $("sort").value, Math.max(1, Number($("top").value) || D.top), true)); draw(); };
  $("select-visible").onclick = () => { candidates().forEach(r => selected.add(r.id)); draw(); };
  $("clear").onclick = () => { selected.clear(); draw(); };
  ["search", "sort", "include-benchmarks"].forEach(id => { $(id).oninput = drawPicker; });
  $("export").onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(TrainingCohort.selection(D, selected), null, 2)], {type: "application/json"}));
    const a = el("a"); a.href = url; a.download = "training-selection.json"; a.click(); URL.revokeObjectURL(url);
  };
  $("import").onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try { selected = TrainingCohort.restore(D, JSON.parse(await file.text())); $("selection-message").textContent = "Selection restored."; draw(); }
    catch (error) { $("selection-message").textContent = error.message; }
  };
  draw();
}
