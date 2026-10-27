"""A self-contained HTML card data page in the layout of 17Lands' Card Data.

`write_page` computes `card_data` for the base selection and for every
combination of subject, opponent, position, deck and opponent deck that has games, then
embeds them, so the page's filters switch between precomputed selections and no
metric is recomputed in the browser. Python is the only implementation of the
statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from itertools import product
import json
from pathlib import Path

import duckdb

from manabot.eval.card_query import CardData, CardSelection, card_data, selected_games

ALL = ""


def _key(*values: str) -> str:
    return "|".join(values)


def _variant(data: CardData) -> dict[str, object]:
    return {
        "games": data.games,
        "deals": data.deals,
        "runs": len(data.runs),
        "win_rate": data.win_rate,
        "excluded": data.excluded,
        "failed_matches": data.failed_matches,
        "cards": [
            [
                card.name,
                card.gp.count,
                card.gp.rate,
                card.oh.count,
                card.oh.rate,
                card.gd.count,
                card.gd.rate,
                card.gih.count,
                card.gih.rate,
                card.gih_interval[0] if card.gih_interval else None,
                card.gih_interval[1] if card.gih_interval else None,
                card.gns.count,
                card.gns.rate,
                card.iwd,
            ]
            for card in data.cards
        ],
    }


def page_data(
    db: duckdb.DuckDBPyConnection, selection: CardSelection
) -> dict[str, object]:
    """Every non-empty filter combination of `selection`, keyed for the page.

    Subjects and opponents are offered by player name, so one player whose code
    identity changed between runs is still one choice.
    """
    base = card_data(db, selection)
    sql, parameters = selected_games(db, selection)
    names = dict(db.execute("select player, name from players").fetchall())
    present = {
        (names[subject], names[opponent], "play" if on_play else "draw", deck, against)
        for subject, opponent, on_play, deck, against in db.execute(
            "select distinct subject, opponent, on_play, deck, opponent_deck"
            f" from ({sql}) where eligible",
            parameters,
        ).fetchall()
    }
    combos = {
        tuple(value if use else ALL for value, use in zip(row, mask, strict=True))
        for row in present
        for mask in product((False, True), repeat=5)
    }
    variants: dict[str, dict[str, object]] = {}
    for subject, opponent, position, deck, opponent_deck in sorted(combos):
        narrowed = selection.model_copy(
            update={
                "subjects": (subject,) if subject else selection.subjects,
                "opponents": (opponent,) if opponent else selection.opponents,
                "on_play": None if not position else position == "play",
                "deck": deck or selection.deck,
                "opponent_deck": opponent_deck or selection.opponent_deck,
            }
        )
        data = base if narrowed == selection else card_data(db, narrowed)
        variants[_key(subject, opponent, position, deck, opponent_deck)] = _variant(
            data
        )
    return {
        "experiment": selection.experiment,
        "runs": base.runs,
        "subjects": sorted({row[0] for row in present}),
        "opponents": sorted({row[1] for row in present}),
        "positions": sorted({row[2] for row in present}),
        "decks": sorted({row[3] for row in present}),
        "opponent_decks": sorted({row[4] for row in present}),
        "metric_version": base.metric_version,
        "variants": variants,
    }


def write_page(
    db: duckdb.DuckDBPyConnection, selection: CardSelection, path: Path
) -> Path:
    data = page_data(db, selection)
    rendered = datetime.now(timezone.utc).isoformat(timespec="seconds")
    title = f"{selection.experiment} · card data"
    # `</` cannot close the script element from inside the JSON.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (
        PAGE.replace("{{title}}", escape(title))
        .replace("{{experiment}}", escape(selection.experiment))
        .replace("{{rendered}}", escape(rendered))
        .replace("{{data}}", payload)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #ebdfc6; --surface: #f7f0e0; --muted: #ded0af; --field: #fcf9ee;
  --border: #c9b892; --border-strong: #a5926a;
  --text: #3a3122; --text-2: #5f553d; --text-3: #6e6247;
  --accent: #973427; --bronze: #6d5a35;
  --good: 70 130 60; --bad: 151 52 39;
  --mono: ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
  --sans: Lato, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --serif: "Cormorant Garamond", Georgia, serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1d1913; --surface: #26211a; --muted: #322b21; --field: #2d271e;
    --border: #4a3f2f; --border-strong: #685a43;
    --text: #efe6d2; --text-2: #cdbf9f; --text-3: #a99b7c;
    --accent: #d8735f; --bronze: #c8ae7a;
    --good: 120 190 100; --bad: 230 110 90;
  }
}
:root[data-theme="dark"] {
  --bg: #1d1913; --surface: #26211a; --muted: #322b21; --field: #2d271e;
  --border: #4a3f2f; --border-strong: #685a43;
  --text: #efe6d2; --text-2: #cdbf9f; --text-3: #a99b7c;
  --accent: #d8735f; --bronze: #c8ae7a;
  --good: 120 190 100; --bad: 230 110 90;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 var(--sans); }
header { background: var(--surface); border-bottom: 1px solid var(--border); }
.bar { max-width: 1400px; margin: 0 auto; padding: 12px 16px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
.brand { font: 600 22px/1 var(--serif); color: var(--bronze); letter-spacing: .01em; }
.brand b { color: var(--accent); font-weight: 600; }
.crumbs { color: var(--text-2); }
main { max-width: 1400px; margin: 0 auto; padding: 16px; }
.filters { display: flex; flex-wrap: wrap; gap: 12px 16px; align-items: end; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }
label { display: grid; gap: 4px; font-size: 12px; color: var(--text-2); text-transform: uppercase; letter-spacing: .04em; }
select, input { font: 14px var(--sans); color: var(--text); background: var(--field); border: 1px solid var(--border-strong); border-radius: 4px; padding: 6px 8px; min-width: 150px; }
select:focus-visible, input:focus-visible, th button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.summary { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 12px 2px; color: var(--text-2); }
.summary strong { color: var(--text); font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: var(--surface); }
thead th { position: sticky; top: 0; background: var(--muted); z-index: 1; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
thead th:first-child { z-index: 2; background: var(--muted); }
th button { all: unset; cursor: pointer; color: var(--text-2); }
th button[aria-sort] { color: var(--text); }
th button[aria-sort="descending"]::after { content: " ▼"; }
th button[aria-sort="ascending"]::after { content: " ▲"; }
th.group { border-left: 1px solid var(--border-strong); }
td.group { border-left: 1px solid var(--border); }
tbody tr:hover td { background: var(--muted); }
td.name { font-weight: 600; }
td.rate { min-width: 64px; }
td.ci { color: var(--text-3); font-size: 12px; }
td.na { color: var(--text-3); }
.notes { color: var(--text-2); font-size: 13px; margin: 16px 2px; max-width: 900px; }
.notes dt { font-weight: 700; color: var(--text); float: left; width: 72px; }
.notes dd { margin: 0 0 4px 80px; }
.empty { padding: 24px; color: var(--text-2); text-align: center; }
label { min-width: 0; }
select { max-width: 100%; }
.crumbs { overflow-wrap: anywhere; }
@media (max-width: 640px) { select, input { min-width: 0; width: 100%; } label { flex: 1 1 140px; } .filters { padding: 12px; } }
</style>
</head>
<body>
<header><div class="bar">
  <span class="brand"><b>manabot</b> card data</span>
  <span class="crumbs">Experiment <strong>{{experiment}}</strong> · rendered {{rendered}}</span>
</div></header>
<main>
  <section class="filters" aria-label="Filters">
    <label>Subject <select id="subject"></select></label>
    <label>Opponent <select id="opponent"></select></label>
    <label>Position <select id="position"></select></label>
    <label>Deck <select id="deck"></select></label>
    <label>Opponent deck <select id="opponent_deck"></select></label>
    <label>Card <input id="search" type="search" placeholder="Search names"></label>
    <label>Min # GIH <input id="minimum" type="number" min="0" value="0" inputmode="numeric"></label>
  </section>
  <div class="summary" id="summary" aria-live="polite"></div>
  <div class="table-wrap"><table>
    <thead><tr id="head"></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <dl class="notes">
    <dt>GIH WR</dt><dd>Win rate over copies in the opening hand or drawn later. Each copy counts, so two copies in hand count twice. Tutored or returned copies count only if later drawn. The interval resamples whole deals across every run.</dd>
    <dt>OH, GD</dt><dd>The same, split into opening-hand copies and copies drawn after the opening hand.</dd>
    <dt>GP</dt><dd>Games with the card in the deck. GNS: those games in which no copy reached the hand.</dd>
    <dt>IWD</dt><dd>GIH WR minus GNS WR, in percentage points.</dd>
    <dt>Games</dt><dd>Player-games with a winner and a passed replay. With all subjects, each game counts once for each side, so win rates center on 50%; pick a subject to see one player's cards. Draws, crashes, timeouts and failed replays are listed separately and never earn a win. These are associations between cards and wins, not proof that a card causes them.</dd>
  </dl>
</main>
<script id="card-data" type="application/json">{{data}}</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("card-data").textContent);
  const columns = [
    ["Name", 0, "text"], ["# GP", 1, "count", true], ["GP WR", 2, "rate"],
    ["# OH", 3, "count", true], ["OH WR", 4, "rate"], ["# GD", 5, "count", true], ["GD WR", 6, "rate"],
    ["# GIH", 7, "count", true], ["GIH WR", 8, "rate"], ["95% CI", 9, "ci"],
    ["# GNS", 11, "count", true], ["GNS WR", 12, "rate"], ["IWD", 13, "pp"],
  ];
  const filters = ["subject", "opponent", "position", "deck", "opponent_deck"];
  let sortIndex = 8, descending = true;
  const $ = (id) => document.getElementById(id);
  const options = (id, values, labels, allLabel) => {
    const select = $(id);
    select.replaceChildren(new Option(allLabel, ""), ...values.map((v) => new Option(labels ? labels[v] : v, v)));
    if (values.length === 0) select.disabled = true;
  };
  options("subject", data.subjects, null, "All subjects");
  options("opponent", data.opponents, null, "All opponents");
  options("position", data.positions, { play: "On the play", draw: "On the draw" }, "Play and draw");
  options("deck", data.decks, null, "All decks");
  options("opponent_deck", data.opponent_decks, null, "All opponent decks");
  const pct = (v) => v == null ? "—" : (100 * v).toFixed(1) + "%";
  const head = $("head");
  columns.forEach(([label, index, kind, group], position) => {
    const th = document.createElement("th");
    if (group) th.className = "group";
    if (kind === "ci") { th.textContent = label; head.append(th); return; }
    const button = document.createElement("button");
    button.textContent = label;
    button.addEventListener("click", () => {
      descending = sortIndex === index ? !descending : kind !== "text";
      sortIndex = index; render();
    });
    button.dataset.index = index;
    th.append(button); head.append(th);
  });
  const shade = (value, mean) => {
    if (value == null || mean == null) return "";
    const delta = Math.max(-1, Math.min(1, (value - mean) / 0.1));
    return `background: rgb(${delta >= 0 ? "var(--good)" : "var(--bad)"} / ${(Math.abs(delta) * 0.45).toFixed(2)})`;
  };
  function render() {
    const key = filters.map((id) => $(id).value).join("|");
    const variant = data.variants[key];
    const summary = $("summary");
    const rows = $("rows");
    head.querySelectorAll("button").forEach((b) => {
      if (Number(b.dataset.index) === sortIndex) b.setAttribute("aria-sort", descending ? "descending" : "ascending");
      else b.removeAttribute("aria-sort");
    });
    if (!variant) {
      summary.textContent = "No games match these filters.";
      rows.replaceChildren(Object.assign(document.createElement("tr"), { innerHTML: `<td class="empty" colspan="${columns.length}">No games match these filters.</td>` }));
      return;
    }
    const subjects = $("subject").value ? [$("subject").value] : data.subjects;
    const excluded = Object.entries(variant.excluded).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ") || "none";
    summary.innerHTML = [
      ["Subjects", subjects.length > 3 ? `${subjects.length} players` : subjects.join(", ") || "—"],
      ["Opponents", $("opponent").value || "all"],
      ["Games", variant.games.toLocaleString()],
      ["Deals", variant.deals.toLocaleString()],
      ["Win rate", pct(variant.win_rate)],
      ["Runs", variant.runs],
      ["Excluded", excluded + (variant.failed_matches ? `, ${variant.failed_matches} failed matches` : "")],
    ].map(([k, v]) => `<span>${k} <strong>${String(v).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c])}</strong></span>`).join("");
    const query = $("search").value.trim().toLowerCase();
    const minimum = Number($("minimum").value) || 0;
    const list = variant.cards.filter((c) => c[0].toLowerCase().includes(query) && c[7] >= minimum);
    list.sort((a, b) => {
      const x = a[sortIndex], y = b[sortIndex];
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      const order = typeof x === "string" ? x.localeCompare(y) : x - y;
      return descending ? -order : order;
    });
    const mean = variant.win_rate;
    rows.replaceChildren(...list.map((c) => {
      const tr = document.createElement("tr");
      columns.forEach(([, index, kind, group]) => {
        const td = document.createElement("td");
        const value = c[index];
        if (group) td.classList.add("group");
        if (kind === "text") { td.className = "name"; td.textContent = value; }
        else if (kind === "count") td.textContent = value.toLocaleString();
        else if (kind === "rate") { td.classList.add("rate"); td.textContent = pct(value); td.style.cssText = shade(value, mean); if (value == null) td.classList.add("na"); }
        else if (kind === "ci") { td.classList.add("ci"); td.textContent = c[9] == null ? "—" : `${pct(c[9])}–${pct(c[10])}`; }
        else { td.textContent = value == null ? "—" : (value >= 0 ? "+" : "") + value.toFixed(1) + "pp"; if (value == null) td.classList.add("na"); }
        tr.append(td);
      });
      return tr;
    }));
    if (!list.length) rows.replaceChildren(Object.assign(document.createElement("tr"), { innerHTML: `<td class="empty" colspan="${columns.length}">No cards match.</td>` }));
  }
  // Filters live in the URL hash, so a link opens the same view.
  const fromHash = () => {
    const given = new URLSearchParams(location.hash.slice(1));
    filters.forEach((id) => {
      const value = given.get(id) ?? "";
      if ([...$(id).options].some((o) => o.value === value)) $(id).value = value;
    });
  };
  const toHash = () => {
    const given = new URLSearchParams();
    filters.forEach((id) => { if ($(id).value) given.set(id, $(id).value); });
    history.replaceState(null, "", given.toString() ? `#${given}` : location.pathname);
  };
  filters.forEach((id) => $(id).addEventListener("input", () => { toHash(); render(); }));
  ["search", "minimum"].forEach((id) => $(id).addEventListener("input", render));
  window.addEventListener("hashchange", () => { fromHash(); render(); });
  fromHash();
  render();
})();
</script>
</body>
</html>
"""
