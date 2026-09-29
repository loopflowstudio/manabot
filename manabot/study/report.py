"""
report.py
Render measures as one self-contained HTML page a person can read

Building blocks only: a matchup composes its own page from `section`,
`rate_table` and `note`. Every rate is shown with its count and interval so a
reader can see how much evidence is behind it. No scripts or network loads.
"""

from __future__ import annotations

# Standard library
from html import escape
from typing import Iterable, Sequence

# Local imports
from .measures import Rate

STYLE = """
/* Layout: one reading column; each question is a section with a table of rates. */
:root {
  --bg: #fbfbfc; --surface: #ffffff; --fg: #14161a; --muted: #5b616b;
  --rule: #dfe2e7; --track: #e9ebef; --mark: #2a78d6; --range: #9dc1ee;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15171a; --surface: #1c1f23; --fg: #f3f4f6; --muted: #a7adb8;
    --rule: #30343b; --track: #2a2e34; --mark: #3987e5; --range: #2c5287;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #15171a; --surface: #1c1f23; --fg: #f3f4f6; --muted: #a7adb8;
  --rule: #30343b; --track: #2a2e34; --mark: #3987e5; --range: #2c5287;
  color-scheme: dark;
}
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 var(--sans); padding-inline: 20px; padding-block: 32px 64px; }
main { max-width: 880px; margin-inline: auto; display: flex;
  flex-direction: column; gap: 40px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0; text-wrap: balance; }
h2 { font-size: 19px; margin: 0; text-wrap: balance; }
h3 { font-size: 13px; margin: 0; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em; }
p { margin: 0; max-width: 68ch; }
header, section { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.lede { color: var(--muted); }
.answer { font-size: 16px; }
.group { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--rule);
  vertical-align: middle; white-space: nowrap; }
th { font-size: 12px; color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-family: var(--mono); font-size: 13px; }
table.rates td.num, table.rates th.num { width: 104px; }
td.bar, th.bar { width: 40%; min-width: 180px; padding-right: 0; }
.track { position: relative; height: 14px; background: var(--track); border-radius: 3px; }
.track .half { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px;
  background: var(--muted); opacity: .45; }
.track .range { position: absolute; top: 5px; height: 4px; background: var(--range);
  border-radius: 2px; }
.track .mark { position: absolute; top: 2px; width: 10px; height: 10px; margin-left: -5px;
  background: var(--mark); border-radius: 50%; box-shadow: 0 0 0 2px var(--surface); }
.axis { display: flex; justify-content: space-between; font: 11px var(--mono);
  color: var(--muted); }
details { border-top: 1px solid var(--rule); padding-top: 8px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
summary:focus-visible { outline: 2px solid var(--mark); outline-offset: 2px; }
details ul { margin: 8px 0 0; padding-left: 18px; font-size: 13px; }
details code, .mono { font-family: var(--mono); font-size: 12px; }
.note { color: var(--muted); font-size: 13px; }
"""


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _bar(rate: Rate) -> str:
    interval = rate.interval()
    if interval is None or rate.value is None:
        return '<div class="track"></div>'
    low, high = interval
    title = f"{percent(rate.value)}, 95% interval {percent(low)} to {percent(high)}"
    return (
        f'<div class="track" title="{escape(title)}">'
        '<span class="half"></span>'
        f'<span class="range" style="left:{low * 100:.1f}%;'
        f'width:{(high - low) * 100:.1f}%"></span>'
        f'<span class="mark" style="left:{rate.value * 100:.1f}%"></span></div>'
    )


def rate_table(
    rows: Iterable[tuple[str, Rate]],
    *,
    what: str,
    hit: str = "yes",
    of: str = "chances",
) -> str:
    """A row per group: rate, count, 95% interval and a bar on a 0-100% scale."""
    body = []
    for label, rate in rows:
        interval = rate.interval()
        between = (
            f"{percent(interval[0])} to {percent(interval[1])}" if interval else "n/a"
        )
        body.append(
            f"<tr><td>{escape(label)}</td>"
            f'<td class="num">{percent(rate.value)}</td>'
            f'<td class="num">{rate.hits:g} of {rate.total}</td>'
            f'<td class="num">{between}</td>'
            f'<td class="bar">{_bar(rate)}</td></tr>'
        )
    if not body:
        return '<p class="note">No recorded games for this table.</p>'
    return (
        '<div class="scroll"><table class="rates"><thead><tr>'
        f"<th>{escape(what)}</th>"
        f'<th class="num">{escape(hit)}</th>'
        f'<th class="num">{escape(of)}</th>'
        '<th class="num">95% interval</th>'
        '<th class="bar"><div class="axis"><span>0%</span><span>50%</span><span>100%</span></div></th>'
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def count_table(
    rows: Iterable[tuple[str, Sequence[object]]], columns: Sequence[str]
) -> str:
    head = "".join(
        f'<th class="{"num" if index else ""}">{escape(name)}</th>'
        for index, name in enumerate(columns)
    )
    body = "".join(
        "<tr><td>"
        + escape(label)
        + "</td>"
        + "".join(f'<td class="num">{escape(str(value))}</td>' for value in values)
        + "</tr>"
        for label, values in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def misses(title: str, rate: Rate) -> str:
    """The recorded cases where the player did not do the thing."""
    if not rate.misses:
        return ""
    items = "".join(
        f"<li><code>{escape(example.game_id)}</code> turn {example.turn}: "
        f"{escape(example.detail)}</li>"
        for example in rate.misses
    )
    shown = len(rate.misses)
    missed = rate.total - int(rate.hits)
    return (
        f"<details><summary>{escape(title)} ({shown} of {missed} shown)</summary>"
        f"<ul>{items}</ul></details>"
    )


def note(text: str) -> str:
    return f'<p class="note">{escape(text)}</p>'


def group(title: str, *parts: str) -> str:
    return f'<div class="group"><h3>{escape(title)}</h3>{"".join(parts)}</div>'


def section(question: str, answer: str, *parts: str) -> str:
    return (
        f"<section><h2>{escape(question)}</h2>"
        f'<p class="answer">{escape(answer)}</p>{"".join(parts)}</section>'
    )


def page(
    title: str, lede: str, sections: Iterable[str], *, fragment: bool = False
) -> str:
    """The whole page. `fragment` omits the document wrapper for embedding."""
    head = f"<title>{escape(title)}</title><style>{STYLE}</style>"
    body = (
        f'<main><header><h1>{escape(title)}</h1><p class="lede">{escape(lede)}</p>'
        f"</header>{''.join(sections)}</main>"
    )
    if fragment:
        return head + body
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{head}</head><body>{body}</body></html>"
    )
