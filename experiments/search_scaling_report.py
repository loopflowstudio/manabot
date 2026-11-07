"""Report the test-time compute sweep: win rates, paired gains and search diagnostics.

Reads every host's `games-*.jsonl` and `decisions-*.jsonl.gz` under the given
directories (`experiments.search_scaling` writes them), keeps the first row per
game key, and writes `report.html` (self-contained, inline SVG) and
`summary.json` beside it.

Scores weight every game equally; the plan gives each of the 100 ordered deck
cells the same deals, so this equals the grid's cell-weighted score when a
condition is complete. Intervals are 95% normal intervals on game scores.
Paired gains difference two conditions on the deals both completed, so deal
luck cancels; their intervals use the paired standard error.

    uv run python -m experiments.search_scaling_report DIR [DIR ...] --out OUT
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import gzip
from html import escape
import json
import math
from pathlib import Path
import statistics
import zlib

from experiments.elemental_grid import DECKS
from experiments.search_scaling import BUDGETS, CONDITIONS, DEALS_PER_CELL, plan

OPPONENTS = ("policy", "heuristic")
OPPONENT_TITLE = {"policy": "against the bare policy", "heuristic": "against the heuristic"}
SOURCES = ("uniform", "oracle")
SOURCE_LABEL = {"uniform": "Uniform hands", "oracle": "Oracle hands"}


# ---------------------------------------------------------------------------
# Loading


def _jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _gz_jsonl(path: Path) -> Iterable[dict]:
    """Read complete lines; a worker stopped mid-write leaves a truncated tail."""
    try:
        with gzip.open(path, "rt") as file:
            for line in file:
                if line.endswith("\n"):
                    yield json.loads(line)
    except (EOFError, OSError, zlib.error):
        return


def load(directories: Sequence[Path]) -> tuple[dict[str, dict], list[dict]]:
    games: dict[str, dict] = {}
    for directory in directories:
        for path in sorted(directory.rglob("games-*.jsonl")):
            for row in _jsonl(path):
                games.setdefault(row["key"], row)
    decisions: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for directory in directories:
        for path in sorted(directory.rglob("decisions-*.jsonl.gz")):
            for row in _gz_jsonl(path):
                marker = (row["key"], row["index"])
                if row["key"] in games and marker not in seen:
                    seen.add(marker)
                    decisions.append(row)
    return games, decisions


# ---------------------------------------------------------------------------
# Statistics


@dataclass
class Estimate:
    n: int
    mean: float
    low: float
    high: float

    def to_json(self) -> dict[str, float]:
        return {"n": self.n, "mean": self.mean, "low": self.low, "high": self.high}


def estimate(values: Sequence[float]) -> Estimate | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return Estimate(len(values), mean, mean - 1.96 * se, mean + 1.96 * se)


@dataclass
class Condition:
    key: str
    opponent: str
    hero: str
    sims: int
    source: str | None
    planned: int
    rows: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> list[dict]:
        return [r for r in self.rows if r["score"] is not None]

    @property
    def scores(self) -> dict[int, float]:
        return {r["deal"]: r["score"] for r in self.valid}


def conditions(games: dict[str, dict]) -> dict[str, Condition]:
    planned: dict[str, int] = defaultdict(int)
    for unit in plan(10):
        planned[unit.condition.key] += 1
    found = {
        c.key: Condition(c.key, c.opponent, c.hero, c.sims, c.source, planned[c.key])
        for c in CONDITIONS
    }
    for row in games.values():
        found[row["condition"]].rows.append(row)
    return found


def paired(a: Condition, b: Condition) -> Estimate | None:
    """a − b on deals both completed."""
    sa, sb = a.scores, b.scores
    return estimate([sa[d] - sb[d] for d in sorted(sa.keys() & sb.keys())])


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


@dataclass
class Diagnostics:
    decisions: int
    seconds: float
    override: float
    depth: float
    disagreement: float
    visit_share: float
    multi_action: int


def diagnostics(rows: Sequence[dict]) -> Diagnostics | None:
    """Per searched decision, over decisions with more than one legal action."""
    choice = [r for r in rows if r["actions"] > 1]
    if not choice:
        return None

    def disagree(row: dict) -> float:
        best = {_argmax(v) for v in row["world_visits"] if sum(v)}
        return float(len(best) > 1)

    return Diagnostics(
        decisions=len(rows),
        seconds=statistics.fmean(r["seconds"] for r in rows),
        override=statistics.fmean(float(r["chosen"] != r["prior_argmax"]) for r in choice),
        depth=statistics.fmean(r["max_depth"] for r in choice),
        disagreement=statistics.fmean(disagree(r) for r in choice),
        visit_share=statistics.fmean(max(r["visits"]) / max(1, sum(r["visits"])) for r in choice),
        multi_action=len(choice),
    )


# ---------------------------------------------------------------------------
# SVG charts

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --panel: #ffffff; --border: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #ecebe7; --axis: #b9b8b2;
  --series-1: #2a78d6; --series-2: #eb6834; --reference: #6f6e69;
  --gain-pos: #2a78d6; --gain-neg: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #222221; --border: #363633;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2e2e2b; --axis: #5a5953;
    --series-1: #3987e5; --series-2: #d95926; --reference: #a3a29b;
    --gain-pos: #3987e5; --gain-neg: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --panel: #222221; --border: #363633;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --grid: #2e2e2b; --axis: #5a5953;
  --series-1: #3987e5; --series-2: #d95926; --reference: #a3a29b;
  --gain-pos: #3987e5; --gain-neg: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text-primary);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 6px; }
p.lede { color: var(--text-secondary); margin: 0 0 20px; max-width: 760px; }
p.note { color: var(--text-secondary); margin: 6px 0 14px; max-width: 760px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 18px 0 8px; }
.tile { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .k { color: var(--text-secondary); font-size: 13px; }
.tile .v { font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile .s { color: var(--text-muted); font-size: 12.5px; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 8px; }
.card h3 { font-size: 14.5px; margin: 0 0 2px; }
.card .sub { color: var(--text-secondary); font-size: 12.5px; margin: 0 0 6px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .title { fill: var(--text-primary); }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .hit { fill: transparent; }
svg .hit:hover + .dot, svg .dot:hover { stroke-width: 3; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--text-secondary); margin: 2px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 16px; height: 0; border-top: 2px solid; }
.legend i.dash { border-top-style: dashed; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 500; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--text-secondary); }
code { font-size: 12.5px; }
"""

W, H = 520, 300
PAD = {"l": 46, "r": 16, "t": 14, "b": 40}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


@dataclass
class Axis:
    low: float
    high: float
    log: bool = False

    def __call__(self, value: float, start: float, end: float) -> float:
        a, b, v = self.low, self.high, value
        if self.log:
            a, b, v = math.log2(a), math.log2(b), math.log2(v)
        return start + (v - a) / (b - a) * (end - start)


def _frame(x_ticks: list[tuple[float, str]], y_ticks: list[float], x: Axis, y: Axis,
           x_title: str, y_fmt) -> list[str]:
    out = []
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    for value in y_ticks:
        py = y(value, bottom, top)
        out.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}"/>')
        out.append(f'<text x="{left - 6}" y="{py + 3.5:.1f}" text-anchor="end">{y_fmt(value)}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>')
    for value, label in x_ticks:
        px = x(value, left, right)
        out.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{bottom}" y2="{bottom + 4}"/>')
        out.append(f'<text x="{px:.1f}" y="{bottom + 16}" text-anchor="middle">{escape(label)}</text>')
    out.append(f'<text x="{(left + right) / 2:.1f}" y="{H - 6}" text-anchor="middle">{escape(x_title)}</text>')
    return out


def _svg(body: list[str], label: str) -> str:
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}">'
            + "".join(body) + "</svg>")


def _series(points: list[tuple[float, Estimate, str]], x: Axis, y: Axis, color: str) -> list[str]:
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    out = []
    coords = [(x(px, left, right), est) for px, est, _ in points]
    if len(coords) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{cx:.1f},{y(e.mean, bottom, top):.1f}"
                        for i, (cx, e) in enumerate(coords))
        out.append(f'<path d="{path}" fill="none" stroke="var({color})" stroke-width="2" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
    for (cx, est), (_, _, tip) in zip(coords, points, strict=True):
        y0, y1 = y(est.low, bottom, top), y(est.high, bottom, top)
        out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y0:.1f}" y2="{y1:.1f}" '
                   f'stroke="var({color})" stroke-width="1.5" opacity="0.55"/>')
        cy = y(est.mean, bottom, top)
        out.append(f'<g><title>{escape(tip)}</title>'
                   f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>'
                   f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var({color})" '
                   f'stroke="var(--panel)" stroke-width="2"/></g>')
    return out


def _reference(value: Estimate, label: str, y: Axis, dash: bool) -> list[str]:
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    py = y(value.mean, bottom, top)
    style = 'stroke-dasharray="5 4"' if dash else 'stroke-dasharray="1.5 3.5"'
    tip = f"{label}: {_fmt(value.mean)} [{_fmt(value.low)}, {_fmt(value.high)}], {value.n} games"
    return [f'<g><title>{escape(tip)}</title>'
            f'<line x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}" stroke="var(--reference)" '
            f'stroke-width="1.5" {style}/>'
            f'<line x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}" stroke="transparent" stroke-width="10"/></g>',
            f'<text x="{right - 2}" y="{py - 5:.1f}" text-anchor="end">{escape(label)} {_fmt(value.mean)}</text>']


def _ticks(low: float, high: float, step: float) -> list[float]:
    first = math.ceil(low / step) * step
    return [round(first + i * step, 6) for i in range(int((high - first) / step + 1e-9) + 1)]


def _y_range(values: Iterable[float], pad: float = 0.03, step: float = 0.05) -> Axis:
    values = list(values)
    low = math.floor((min(values) - pad) / step) * step
    high = math.ceil((max(values) + pad) / step) * step
    return Axis(max(0.0, low), min(1.0, high))


def score_chart(found: dict[str, Condition], opponent: str, by_time: bool,
                diag: dict[str, Diagnostics]) -> str:
    series: dict[str, list[tuple[float, Estimate, str]]] = {s: [] for s in SOURCES}
    for source in SOURCES:
        for sims in BUDGETS:
            c = found[f"{opponent}/puct-{sims}-{source}"]
            est = estimate([r["score"] for r in c.valid])
            if est is None:
                continue
            if by_time:
                d = diag.get(c.key)
                if d is None:
                    continue
                px = d.seconds
            else:
                px = float(sims)
            tip = (f"{SOURCE_LABEL[source]}, {sims} sims {OPPONENT_TITLE[opponent]}: "
                   f"{_fmt(est.mean)} [{_fmt(est.low)}, {_fmt(est.high)}], {est.n} of {c.planned} games")
            if by_time:
                tip += f"; {d.seconds:.2f} s per searched decision"
            series[source].append((px, est, tip))
    refs = {name: estimate([r["score"] for r in found[f"{opponent}/{name}"].valid])
            for name in ("policy", "greedy")}
    values = [e.low for pts in series.values() for _, e, _ in pts] + \
             [e.high for pts in series.values() for _, e, _ in pts] + \
             [e.mean for e in refs.values() if e is not None]
    y = _y_range(values or [0.4, 0.6])
    if by_time:
        xs = [px for pts in series.values() for px, _, _ in pts] or [0.01, 1]
        lo, hi = min(xs) / 1.6, max(xs) * 1.6
        x = Axis(lo, hi, log=True)
        nice = [m * 10.0**e for e in range(-3, 3) for m in (1, 3)]
        ticks = [(v, f"{v:g} s" if v >= 1 else f"{v * 1000:g} ms") for v in nice if lo <= v <= hi]
        title = "seconds per searched decision (log scale)"
    else:
        x = Axis(BUDGETS[0] / 1.6, BUDGETS[-1] * 1.6, log=True)
        ticks = [(float(s), str(s)) for s in BUDGETS]
        title = "simulations per decision (log scale)"
    body = _frame(ticks, _ticks(y.low, y.high, 0.05), x, y, title, lambda v: f"{v:.2f}")
    for name, dash in (("policy", True), ("greedy", False)):
        if refs[name] is not None:
            body += _reference(refs[name], "sampled policy" if name == "policy" else "greedy policy", y, dash)
    for index, source in enumerate(SOURCES):
        body += _series(series[source], x, y, f"--series-{index + 1}")
    return _svg(body, f"score {OPPONENT_TITLE[opponent]} by {'time' if by_time else 'simulations'}")


def gain_chart(rows: list[tuple[str, Estimate | None]], title: str) -> str:
    """Horizontal paired-gain bars with intervals around zero."""
    height = 34 + 26 * len(rows)
    left, right = 150, W - 60
    present = [e for _, e in rows if e is not None]
    span = max([abs(e.low) for e in present] + [abs(e.high) for e in present] + [0.05])
    span = math.ceil(span * 20) / 20
    x = Axis(-span, span)
    body = []
    for t in _ticks(-span, span, 0.05 if span <= 0.25 else 0.1):
        px = x(t, left, right)
        body.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="8" y2="{height - 22}"/>')
        body.append(f'<text x="{px:.1f}" y="{height - 8}" text-anchor="middle">{t * 100:+.0f}</text>')
    zero = x(0, left, right)
    body.append(f'<line class="axis" x1="{zero:.1f}" x2="{zero:.1f}" y1="8" y2="{height - 22}"/>')
    for i, (label, est) in enumerate(rows):
        cy = 22 + 26 * i
        body.append(f'<text x="{left - 8}" y="{cy + 4}" text-anchor="end">{escape(label)}</text>')
        if est is None:
            body.append(f'<text x="{zero + 6:.1f}" y="{cy + 4}">pending</text>')
            continue
        px = x(est.mean, left, right)
        color = "--gain-pos" if est.mean >= 0 else "--gain-neg"
        x0, x1 = sorted((zero, px))
        tip = f"{label}: {est.mean * 100:+.1f} points [{est.low * 100:+.1f}, {est.high * 100:+.1f}] over {est.n} paired deals"
        body.append(f'<g><title>{escape(tip)}</title>'
                    f'<rect x="{x0:.1f}" y="{cy - 7}" width="{max(1.0, x1 - x0):.1f}" height="14" rx="3" '
                    f'fill="var({color})" opacity="0.85"/>'
                    f'<line x1="{x(est.low, left, right):.1f}" x2="{x(est.high, left, right):.1f}" '
                    f'y1="{cy}" y2="{cy}" stroke="var(--text-primary)" stroke-width="1.5"/>'
                    f'<rect class="hit" x="{left}" y="{cy - 12}" width="{right - left}" height="24"/></g>')
        anchor_x = max(x(est.high, left, right), px) + 6
        body.append(f'<text x="{anchor_x:.1f}" y="{cy + 4}">{est.mean * 100:+.1f}</text>')
    return (f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">'
            + "".join(body) + "</svg>")


def line_chart(series: dict[str, list[tuple[int, float, str]]], y_max: float, y_fmt, title: str,
               y_step: float) -> str:
    x = Axis(BUDGETS[0] / 1.6, BUDGETS[-1] * 1.6, log=True)
    y = Axis(0.0, y_max)
    body = _frame([(float(s), str(s)) for s in BUDGETS], _ticks(0, y_max, y_step), x, y,
                  "simulations per decision (log scale)", y_fmt)
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    for index, source in enumerate(SOURCES):
        pts = series.get(source, [])
        color = f"--series-{index + 1}"
        if len(pts) > 1:
            path = " ".join(f"{'M' if i == 0 else 'L'}{x(s, left, right):.1f},{y(v, bottom, top):.1f}"
                            for i, (s, v, _) in enumerate(pts))
            body.append(f'<path d="{path}" fill="none" stroke="var({color})" stroke-width="2"/>')
        for s, v, tip in pts:
            cx, cy = x(s, left, right), y(v, bottom, top)
            body.append(f'<g><title>{escape(tip)}</title><circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>'
                        f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="var({color})" '
                        f'stroke="var(--panel)" stroke-width="2"/></g>')
    return _svg(body, title)


# ---------------------------------------------------------------------------
# Page


def _legend(with_refs: bool = True) -> str:
    items = [f'<span><i style="border-color:var(--series-{i + 1})"></i>{SOURCE_LABEL[s]}</span>'
             for i, s in enumerate(SOURCES)]
    if with_refs:
        items += ['<span><i class="dash" style="border-color:var(--reference)"></i>Sampled policy (no search)</span>',
                  '<span><i style="border-color:var(--reference);border-top-style:dotted"></i>Greedy policy (no search)</span>']
    return '<div class="legend">' + "".join(items) + "</div>"


def build(directories: Sequence[Path], out: Path) -> dict:
    games, decisions = load(directories)
    found = conditions(games)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in decisions:
        by_key[games[row["key"]]["condition"]].append(row)
    diag = {k: d for k, rows in by_key.items() if (d := diagnostics(rows)) is not None}

    summary: dict = {"conditions": {}, "paired": {}, "games": len(games), "decisions": len(decisions)}
    for key, c in found.items():
        est = estimate([r["score"] for r in c.valid])
        d = diag.get(key)
        summary["conditions"][key] = {
            "opponent": c.opponent, "hero": c.hero, "sims": c.sims, "source": c.source,
            "planned": c.planned, "played": len(c.rows), "invalid": len(c.rows) - len(c.valid),
            "score": est.to_json() if est else None,
            "hero_seconds_per_game": statistics.fmean(r["hero_seconds"] for r in c.rows) if c.rows else None,
            "diagnostics": d.__dict__ if d else None,
        }

    def gain(a: str, b: str) -> Estimate | None:
        est = paired(found[a], found[b])
        summary["paired"][f"{a} − {b}"] = est.to_json() if est else None
        return est

    # Headline tiles.
    tiles = []
    for opponent in OPPONENTS:
        base = estimate([r["score"] for r in found[f"{opponent}/policy"].valid])
        best = None
        for sims in reversed(BUDGETS):
            e = estimate([r["score"] for r in found[f"{opponent}/puct-{sims}-uniform"].valid])
            if e is not None and e.n >= 100:
                best = (sims, e)
                break
        if base:
            tiles.append(f'<div class="tile"><div class="k">Policy {OPPONENT_TITLE[opponent]}</div>'
                         f'<div class="v">{_pct(base.mean)}</div>'
                         f'<div class="s">[{_pct(base.low)}, {_pct(base.high)}] · {base.n} games</div></div>')
        if best:
            sims, e = best
            g = paired(found[f"{opponent}/puct-{sims}-uniform"], found[f"{opponent}/policy"])
            tiles.append(f'<div class="tile"><div class="k">Search, {sims} sims, uniform hands</div>'
                         f'<div class="v">{_pct(e.mean)}</div>'
                         f'<div class="s">{OPPONENT_TITLE[opponent]} · '
                         + (f'{g.mean * 100:+.1f} pts paired' if g else "") + f' · {e.n} games</div></div>')
    played = sum(len(c.rows) for c in found.values())
    planned = sum(c.planned for c in found.values())
    invalid = sum(len(c.rows) - len(c.valid) for c in found.values())
    tiles.append(f'<div class="tile"><div class="k">Games played</div><div class="v">{played:,}</div>'
                 f'<div class="s">of {planned:,} planned · {invalid} invalid</div></div>')

    sections = []
    sections.append("<h2>Win rate by search budget</h2>"
                    '<p class="note">Each point is one condition on the same fixed deals; bars are 95% intervals. '
                    "Reference lines are the policy with no search. Hover a point for its numbers.</p>"
                    + _legend()
                    + '<div class="row">'
                    + "".join(f'<div class="card"><h3>{OPPONENT_TITLE[o].capitalize()}</h3>'
                              f'<div class="sub">Score of MTG-183 with search</div>{score_chart(found, o, False, diag)}</div>'
                              for o in OPPONENTS)
                    + "</div>")
    sections.append("<h2>Win rate by time spent</h2>"
                    '<p class="note">The same points placed by mean wall seconds per searched decision on one CPU thread, '
                    "the cost a deployed searcher would pay.</p>"
                    + _legend(False)
                    + '<div class="row">'
                    + "".join(f'<div class="card"><h3>{OPPONENT_TITLE[o].capitalize()}</h3>'
                              f'<div class="sub">Score against seconds per decision</div>{score_chart(found, o, True, diag)}</div>'
                              for o in OPPONENTS)
                    + "</div>")

    gain_cards = []
    for opponent in OPPONENTS:
        rows = [(f"{s} sims, {src}", gain(f"{opponent}/puct-{s}-{src}", f"{opponent}/policy"))
                for s in BUDGETS for src in SOURCES]
        gain_cards.append(f'<div class="card"><h3>Search over the sampled policy, {OPPONENT_TITLE[opponent]}</h3>'
                          f'<div class="sub">Paired by deal, percentage points</div>{gain_chart(rows, "gain")}</div>')
    sections.append("<h2>What search adds</h2>"
                    '<p class="note">Paired differences cancel deal luck: each bar compares two conditions on the deals both finished.</p>'
                    '<div class="row">' + "".join(gain_cards) + "</div>")

    oracle_cards = []
    for opponent in OPPONENTS:
        rows = [(f"{s} sims", gain(f"{opponent}/puct-{s}-oracle", f"{opponent}/puct-{s}-uniform")) for s in BUDGETS]
        oracle_cards.append(f'<div class="card"><h3>Oracle over uniform hands, {OPPONENT_TITLE[opponent]}</h3>'
                            f'<div class="sub">The most better beliefs could add to this search</div>{gain_chart(rows, "oracle")}</div>')
    sections.append("<h2>How much beliefs could matter</h2>"
                    '<p class="note">Oracle worlds hold the opponent\'s true hand (libraries still reshuffled). '
                    "The gap is a loose upper bound on what any belief model adds here: it also removes strategy fusion over the hand.</p>"
                    '<div class="row">' + "".join(oracle_cards) + "</div>")

    # Per-deck gains at the largest budget with data.
    deck_cards = []
    for opponent in OPPONENTS:
        chosen = None
        for sims in reversed(BUDGETS):
            if len(found[f"{opponent}/puct-{sims}-uniform"].valid) >= 200:
                chosen = sims
                break
        if chosen is None:
            continue
        a, b = found[f"{opponent}/puct-{chosen}-uniform"], found[f"{opponent}/policy"]
        rows = []
        for deck in DECKS:
            sa = {r["deal"]: r["score"] for r in a.valid if r["model_deck"] == deck}
            sb = {r["deal"]: r["score"] for r in b.valid if r["model_deck"] == deck}
            rows.append((deck.removeprefix("elemental_").upper(),
                         estimate([sa[d] - sb[d] for d in sorted(sa.keys() & sb.keys())])))
        deck_cards.append(f'<div class="card"><h3>By hero deck, {chosen} sims, {OPPONENT_TITLE[opponent]}</h3>'
                          f'<div class="sub">Uniform-hand search minus sampled policy, points</div>{gain_chart(rows, "deck")}</div>')
    if deck_cards:
        sections.append("<h2>Where search helps</h2>"
                        '<p class="note">Paired gain for each deck MTG-183 pilots, across all ten opposing decks. '
                        "About 100 deals per bar at full plan, so intervals are wide.</p>"
                        '<div class="row">' + "".join(deck_cards) + "</div>")

    # Diagnostics.
    def diag_series(attr: str, fmt) -> dict[str, list[tuple[int, float, str]]]:
        out: dict[str, list[tuple[int, float, str]]] = {}
        for source in SOURCES:
            pts = []
            for sims in BUDGETS:
                vals = [diag[k] for k in (f"{o}/puct-{sims}-{source}" for o in OPPONENTS) if k in diag]
                if not vals:
                    continue
                weight = sum(v.multi_action for v in vals)
                value = sum(getattr(v, attr) * v.multi_action for v in vals) / weight
                pts.append((sims, value, f"{SOURCE_LABEL[source]}, {sims} sims: {fmt(value)} over {weight:,} decisions"))
            out[source] = pts
        return out

    if diag:
        override = diag_series("override", _pct)
        disagreement = diag_series("disagreement", _pct)
        depth = diag_series("depth", lambda v: f"{v:.1f} plies")
        seconds = diag_series("seconds", lambda v: f"{v:.3f} s")
        depth_max = max([v for pts in depth.values() for _, v, _ in pts] + [1])
        sec_max = max([v for pts in seconds.values() for _, v, _ in pts] + [0.1])
        depth_top = math.ceil(depth_max / 10) * 10
        sec_top = 10 ** math.ceil(math.log10(sec_max)) if sec_max > 0 else 1
        sec_top = next(t for t in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16) if t >= sec_max)
        cards = [
            ("Search overrides the policy", "Decisions where the chosen action is not the policy's most likely one",
             line_chart(override, 0.5, lambda v: f"{v * 100:.0f}%", "override", 0.1)),
            ("Worlds disagree", "Decisions where sampled worlds' most-visited actions differ",
             line_chart(disagreement, 1.0, lambda v: f"{v * 100:.0f}%", "disagreement", 0.2)),
            ("Tree depth", "Deepest simulation, in decisions below the root",
             line_chart(depth, depth_top, lambda v: f"{v:.0f}", "depth", depth_top / 5)),
            ("Time per decision", "Mean wall seconds per searched decision, one CPU thread",
             line_chart(seconds, sec_top, lambda v: f"{v:g}", "seconds", sec_top / 4)),
        ]
        sections.append("<h2>How search behaves</h2>"
                        '<p class="note">Over searched decisions with more than one legal action, both opponents pooled. '
                        "These are the levers for making search cheaper or sharper.</p>"
                        + _legend(False) + '<div class="row">'
                        + "".join(f'<div class="card"><h3>{t}</h3><div class="sub">{s}</div>{c}</div>' for t, s, c in cards)
                        + "</div>")

    # Table.
    head = "<tr><th>Condition</th><th>Games</th><th>Invalid</th><th>Score</th><th>95% interval</th><th>vs policy (paired)</th><th>s / decision</th><th>Override</th></tr>"
    body = []
    for key, c in sorted(found.items(), key=lambda kv: (kv[1].opponent, kv[1].sims, kv[1].hero, kv[1].source or "")):
        est = estimate([r["score"] for r in c.valid])
        g = paired(c, found[f"{c.opponent}/policy"]) if c.hero != "policy" else None
        d = diag.get(key)
        body.append(
            f"<tr><td>{escape(key)}</td><td>{len(c.rows)} / {c.planned}</td><td>{len(c.rows) - len(c.valid)}</td>"
            f"<td>{_fmt(est.mean) if est else '–'}</td>"
            f"<td>{f'[{_fmt(est.low)}, {_fmt(est.high)}]' if est else '–'}</td>"
            f"<td>{f'{g.mean * 100:+.1f} [{g.low * 100:+.1f}, {g.high * 100:+.1f}]' if g else '–'}</td>"
            f"<td>{f'{d.seconds:.3f}' if d else '–'}</td><td>{_pct(d.override) if d else '–'}</td></tr>")
    sections.append("<h2>Every condition</h2><table>" + head + "".join(body) + "</table>")

    sections.append(
        "<h2>Method</h2><p class=\"note\">MTG-183's final raw policy (update 5,950, w13, SHA-256 "
        "<code>4f2ad656…</code>) plays every Elemental deck against every deck on the first ten deals of each "
        f"10×10 grid cell (the first {DEALS_PER_CELL[max(BUDGETS)]} at {max(BUDGETS)} simulations), seats from the frozen grid schedule. "
        f"Search is determinized PUCT: the budget split over four worlds, one tree each, the network as prior and "
        "leaf value, c<sub>puct</sub> 1.5, most-visited action played. Uniform worlds draw the opponent's unknown hand "
        "uniformly from the cards the hero has not seen; oracle worlds use the true hand. Both reshuffle libraries. "
        "The opponent never searches. One deal is one game; draws score half.</p>")

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Search Scaling Sweep</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>Test-time search on MTG-183</h1>"
        '<p class="lede">How much does determinized PUCT add to the best ten-deck Elemental policy, '
        "and how much of the gain is limited by not knowing the opponent's hand?</p>"
        f'<div class="tiles">{"".join(tiles)}</div>'
        + "".join(sections)
        + "</main></body></html>"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(html)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.directories, args.out)
    print(f"{summary['games']} games, {summary['decisions']} searched decisions -> {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
