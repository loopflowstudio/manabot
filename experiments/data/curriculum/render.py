"""Render the curriculum follow-up report from its saved analysis.

usage: uv run python experiments/data/curriculum/render.py

Reads `analysis.json` beside this file and writes `report.html`: the primary
curriculum-minus-scratch contrast with per-seed points and the frozen ±5-point
thresholds, each arm's gain over its frozen initialization, per-cell effects at
the primary milestone, time to +10 points and the boundary check. Saved
evidence only; no evaluation runs here.
"""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHORS = ("heuristic", "scripted_greedy", "random")
ANCHOR_NAMES = {"heuristic": "heuristic", "scripted_greedy": "greedy", "random": "random"}
MILESTONES = (32, 64, 128)
ARMS = ("curriculum", "scratch")
ARM_COLORS = {"curriculum": "#b5602c", "scratch": "#2d6a8a"}
CELL_NAMES = {
    "ur_lessons v ur_lessons": "Lessons v Lessons",
    "ur_lessons v gw_allies": "Lessons v Allies",
    "gw_allies v ur_lessons": "Allies v Lessons",
    "gw_allies v gw_allies": "Allies v Allies",
}


@dataclass(frozen=True)
class Interval:
    """Six-seed mean and 95% t interval, in points."""

    mean: float
    lower: float
    upper: float
    positive_seeds: int

    @classmethod
    def from_json(cls, row: dict[str, float]) -> "Interval":
        return cls(
            100 * row["mean"],
            100 * row["lower"],
            100 * row["upper"],
            int(row["positive_seeds"]),
        )

    def text(self) -> str:
        return f"{self.mean:+.1f} [{self.lower:+.1f}, {self.upper:+.1f}]"


def _contrast(analysis: dict, anchor: str, milestone: int) -> Interval:
    return Interval.from_json(analysis["contrasts"][f"{anchor}/{milestone}"]["summary"])


def _gain(analysis: dict, anchor: str, arm: str, milestone: int) -> Interval:
    return Interval.from_json(analysis["gain_over_initialization"][f"{anchor}/{arm}/{milestone}"])


# Charts -----------------------------------------------------------------------


def _contrast_chart(analysis: dict, anchor: str) -> str:
    """Per-seed A − B dots and the six-seed interval at each milestone."""
    width, height, left, top, bottom = 360, 240, 44, 16, 32
    low, high = -25.0, 15.0

    def y(points: float) -> float:
        return top + (high - points) / (high - low) * (height - top - bottom)

    def x(index: int) -> float:
        return left + (index + 0.5) * (width - left - 12) / len(MILESTONES)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Curriculum minus scratch against {ANCHOR_NAMES[anchor]}">',
        f'<rect x="{left}" y="{y(5):.1f}" width="{width - left - 12}" height="{y(-5) - y(5):.1f}" fill="var(--band)"/>',
    ]
    for tick in range(-25, 16, 5):
        parts.append(
            f'<line x1="{left}" x2="{width - 12}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)" '
            f'stroke-width="{1.4 if tick == 0 else 0.6}"/>'
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" class="tick">{tick:+d}</text>'
        )
    for index, milestone in enumerate(MILESTONES):
        cx = x(index)
        seeds = analysis["contrasts"][f"{anchor}/{milestone}"]["seeds"]
        for offset, row in enumerate(seeds.values()):
            parts.append(
                f'<circle cx="{cx - 25 + offset * 5:.1f}" cy="{y(100 * row["difference"]):.1f}" r="2.6" '
                'fill="var(--muted)" opacity="0.7"/>'
            )
        summary = _contrast(analysis, anchor, milestone)
        parts.append(
            f'<line x1="{cx + 14:.1f}" x2="{cx + 14:.1f}" y1="{y(summary.upper):.1f}" y2="{y(summary.lower):.1f}" '
            'stroke="var(--ink)" stroke-width="2"/>'
            f'<circle cx="{cx + 14:.1f}" cy="{y(summary.mean):.1f}" r="4.5" fill="var(--ink)"/>'
            f'<text x="{cx:.1f}" y="{height - 10}" text-anchor="middle" class="tick'
            f'{" strong" if milestone == 64 else ""}">{milestone}{" primary" if milestone == 64 else ""}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _gain_chart(analysis: dict, anchor: str) -> str:
    """Each arm's gain over initialization at 0, 32, 64 and 128 iterations."""
    width, height, left, top, bottom = 360, 240, 44, 16, 32
    low, high = -5.0, 40.0
    steps = (0, *MILESTONES)

    def y(points: float) -> float:
        return top + (high - points) / (high - low) * (height - top - bottom)

    def x(milestone: int) -> float:
        return left + 10 + milestone / 128 * (width - left - 30)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Gain over initialization against {ANCHOR_NAMES[anchor]}">',
        f'<line x1="{x(32):.1f}" x2="{x(32):.1f}" y1="{top}" y2="{height - bottom}" stroke="var(--muted)" stroke-dasharray="3 3"/>'
        f'<text x="{x(32) + 4:.1f}" y="{top + 10}" class="tick">L1 → L3 switch</text>',
    ]
    for tick in range(0, 41, 10):
        parts.append(
            f'<line x1="{left}" x2="{width - 12}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)" stroke-width="0.6"/>'
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" class="tick">{tick:+d}</text>'
        )
    for step in steps:
        parts.append(f'<text x="{x(step):.1f}" y="{height - 10}" text-anchor="middle" class="tick">{step}</text>')
    for shift, arm in zip((-3, 3), ARMS):
        color = ARM_COLORS[arm]
        points = [(0, Interval(0, 0, 0, 0))] + [(m, _gain(analysis, anchor, arm, m)) for m in MILESTONES]
        path = " ".join(f"{x(m) + shift:.1f},{y(g.mean):.1f}" for m, g in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for milestone, gain in points:
            parts.append(
                f'<line x1="{x(milestone) + shift:.1f}" x2="{x(milestone) + shift:.1f}" y1="{y(gain.upper):.1f}" '
                f'y2="{y(gain.lower):.1f}" stroke="{color}" stroke-width="1.5"/>'
                f'<circle cx="{x(milestone) + shift:.1f}" cy="{y(gain.mean):.1f}" r="3.5" fill="{color}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


# Tables -----------------------------------------------------------------------


def _contrast_table(analysis: dict) -> str:
    rows = []
    for anchor in ANCHORS:
        cells = []
        for milestone in MILESTONES:
            summary = _contrast(analysis, anchor, milestone)
            tag = "strong" if milestone == 64 else "span"
            cells.append(
                f"<td><{tag}>{summary.text()}</{tag}><small>{summary.positive_seeds}/6 seeds positive</small></td>"
            )
        rows.append(f"<tr><th>{ANCHOR_NAMES[anchor]}</th>{''.join(cells)}</tr>")
    return (
        '<div class="table"><table><thead><tr><th>Anchor</th><th>32</th><th>64 (primary)</th><th>128</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _gain_table(analysis: dict) -> str:
    rows = []
    for anchor in ANCHORS:
        for arm in ARMS:
            cells = "".join(f"<td>{_gain(analysis, anchor, arm, m).mean:+.1f}</td>" for m in MILESTONES)
            rows.append(
                f'<tr><th>{ANCHOR_NAMES[anchor]}</th><td><span class="swatch" style="background:{ARM_COLORS[arm]}"></span>'
                f"{arm}</td>{cells}</tr>"
            )
    return (
        '<div class="table"><table><thead><tr><th>Anchor</th><th>Arm</th><th>32</th><th>64</th><th>128</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _cell_table(analysis: dict) -> str:
    rows = []
    for anchor in ANCHORS:
        seeds = analysis["contrasts"][f"{anchor}/64"]["seeds"].values()
        means = {
            name: 100 * sum(row["cells"][name] for row in seeds) / len(seeds) for name in CELL_NAMES
        }
        cells = "".join(f"<td>{means[name]:+.1f}</td>" for name in CELL_NAMES)
        rows.append(f"<tr><th>{ANCHOR_NAMES[anchor]}</th>{cells}</tr>")
    head = "".join(f"<th>{escape(label)}</th>" for label in CELL_NAMES.values())
    return (
        f'<div class="table"><table><thead><tr><th>Anchor</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _progress_table(analysis: dict) -> str:
    rows = []
    for anchor in ANCHORS:
        cells = []
        for arm in ARMS:
            reached = analysis["time_to_plus_10"][f"{anchor}/{arm}"].values()
            early = sum(1 for milestone in reached if milestone == 32)
            cells.append(f"<td>{early}/6 by 32; all by 64</td>")
        rows.append(f"<tr><th>{ANCHOR_NAMES[anchor]}</th>{''.join(cells)}</tr>")
    return (
        '<div class="table"><table><thead><tr><th>Anchor</th><th>curriculum</th><th>scratch</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _boundary_table(analysis: dict) -> str:
    rows = []
    for anchor in ANCHORS:
        for milestone in (64, 128):
            seeds = analysis["scratch_boundary_vs_continuous"][f"{anchor}/{milestone}"]
            cells = "".join(
                f"<td>{100 * row['split']:.1f}% / {100 * row['continuous']:.1f}%</td>" for row in seeds.values()
            )
            rows.append(f"<tr><th>{ANCHOR_NAMES[anchor]}</th><td>{milestone}</td>{cells}</tr>")
    return (
        '<div class="table"><table><thead><tr><th>Anchor</th><th>Iteration</th><th>12251</th><th>12252</th>'
        f"<th>12253</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# Page -------------------------------------------------------------------------

STYLE = """
:root{--ink:#23342d;--muted:#627067;--paper:#fffdf8;--rule:#bbc4b8;--grid:#e0e3d9;--band:#eff2e8;--link:#176b57}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ink:#e4e8df;--muted:#9aa79e;--paper:#18201c;--rule:#45524a;--grid:#2c3630;--band:#232d27;--link:#7cc7ad}}
:root[data-theme="dark"]{--ink:#e4e8df;--muted:#9aa79e;--paper:#18201c;--rule:#45524a;--grid:#2c3630;--band:#232d27;--link:#7cc7ad}
body{font:15px/1.55 'Avenir Next',system-ui,sans-serif;color:var(--ink);background:var(--paper);max-width:1080px;margin:40px auto;padding:0 28px}
h1{font:44px/1.12 Georgia,serif;letter-spacing:-1px;margin:8px 0 12px}h2{font:26px/1.2 Georgia,serif;margin:8px 0}
a{color:var(--link);text-underline-offset:3px}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;color:var(--muted)}
.muted,small,.tick{color:var(--muted);fill:var(--muted)}small{display:block;font-size:11px}.tick{font-size:10px}.tick.strong{font-weight:700;fill:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);border-top:2px solid var(--ink);border-bottom:1px solid var(--rule);margin:24px 0}
.stats>div{padding:14px 12px}.stats span{display:block;font-size:12px;color:var(--muted)}.stats strong{font:26px Georgia,serif}
.reading{background:var(--band);border-left:3px solid #718764;padding:12px 18px;margin:16px 0}.reading p{margin:6px 0}
section{margin-top:40px}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:600}td,th{padding:9px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.charts figure{margin:0}.charts svg{width:100%;height:auto}
figcaption{font-weight:600;font-size:13px}.legend{font-size:12px;color:var(--muted)}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
ul{padding-left:20px}li{margin:6px 0}
@media(max-width:760px){body{margin:24px auto;padding:0 16px}h1{font-size:32px}.stats{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}}
"""


def render(analysis: dict) -> str:
    greedy = _contrast(analysis, "scripted_greedy", 64)
    heuristic = _contrast(analysis, "heuristic", 64)
    verdicts = analysis["verdict_by_decision_anchor"]
    contrast_charts = "".join(
        f"<figure><figcaption>vs {ANCHOR_NAMES[a]}</figcaption>{_contrast_chart(analysis, a)}</figure>" for a in ANCHORS
    )
    gain_charts = "".join(
        f"<figure><figcaption>vs {ANCHOR_NAMES[a]}</figcaption>{_gain_chart(analysis, a)}</figure>" for a in ANCHORS
    )
    legend = " ".join(
        f'<span><span class="swatch" style="background:{ARM_COLORS[arm]}"></span>{arm}</span>' for arm in ARMS
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simple-pool curriculum result</title><style>{STYLE}</style><main>
<div class="eyebrow">MTG-122 FOLLOW-UP · 2026-10-20</div>
<h1>An L1 warm-up does not help Allies versus Lessons</h1>
<p>Spending the first quarter of training (32 of 128 iterations) on the L1 simple pool, then switching to
Allies versus Lessons (L3), starts behind training on L3 throughout and only catches up. Same recipe, same
learner transitions, six paired seeds, one continuous learning state across the switch in both arms.</p>
<div class="stats">
<div><span>Primary, vs greedy at 64</span><strong>{greedy.mean:+.1f}</strong><span>{escape(verdicts["scripted_greedy"])}</span></div>
<div><span>Primary, vs heuristic at 64</span><strong>{heuristic.mean:+.1f}</strong><span>{escape(verdicts["heuristic"])}, leaning negative</span></div>
<div><span>Games scored</span><strong>72,000</strong><span>180 cells × 400, 0 missing</span></div>
<div><span>Cost</span><strong>≈ $6.0</strong><span>GPU $0.37 + rented CPU ≈ $5.59</span></div>
</div>
<div class="reading"><p><strong>Reading.</strong> The frozen rule calls a curriculum useful only if the six-seed
interval for curriculum minus scratch at iteration 64 lies above +5 points. Against greedy the interval
{greedy.text()} lies inside ±5: no useful effect. Against heuristic {heuristic.text()} is inconclusive only
because its lower bound crosses −5. Which anchor decides awaits MTG-153's heuristic-versus-greedy pairing;
neither can yield “useful”.</p></div>

<section><h2>Curriculum minus scratch</h2>
<p class="muted">Points of score, paired by seed and deal, mean of the four deck-role cells. Grey dots are seeds;
black is the six-seed mean with 95% t interval (df 5). The shaded band is the frozen ±5-point region.</p>
<div class="charts">{contrast_charts}</div>
{_contrast_table(analysis)}
</section>

<section><h2>Gain over initialization</h2>
<p class="muted">Each arm minus the same seed's frozen initialization on the same L3 deals. <span class="legend">{legend}</span></p>
<div class="charts">{gain_charts}</div>
{_gain_table(analysis)}
<ul>
<li>At the switch, the L1-trained policy playing L3 for the first time has about half the scratch gain.</li>
<li>By 128 the arms are level against heuristic and greedy. Against random the curriculum endpoint is
{_contrast(analysis, "random", 128).text()}, while scratch dips from 64 to 128; a small secondary signal, not a benefit claim.</li>
</ul>
</section>

<section><h2>Time to +10 points</h2>
<p class="muted">First evaluated milestone at which a seed's gain over initialization reaches +10 points.</p>
{_progress_table(analysis)}
</section>

<section><h2>Cells at the primary milestone</h2>
<p class="muted">Curriculum minus scratch at 64 by candidate deck v opponent deck, six-seed means. Against heuristic and greedy the curriculum is
behind when piloting Allies and roughly level or ahead in Lessons mirrors.</p>
{_cell_table(analysis)}
</section>

<section><h2>Boundary check</h2>
<p class="muted">Scratch with the iteration-32 continuation (split) against the complexity cohort's uninterrupted L3 fits
(continuous), same seeds and deals. The continuation itself costs nothing.</p>
{_boundary_table(analysis)}
</section>

<section><h2>What ran</h2>
<ul>
<li><strong>Training:</strong> Etude run 8cc849175eab, one L4, 12 fits in 5,371 s, source adcf54e3. Recipe 64x2,
64 streams × 64 transitions, 128 iterations = 524,288 learner transitions per fit. Both arms carry weights, EMA,
Adam and the schedule clock across iteration 32; only game streams and RNGs restart.</li>
<li><strong>Scoring:</strong> four rented 8-vCPU CPU pods. Held-out L3 matrix (four deck-role cells, both play
orders, 50 deals) against heuristic, greedy and random at 0, 32, 64 and 128, with frozen-initialization controls.</li>
<li><strong>Parity:</strong> 8 of 9 initialization cells re-scored the Mini's complexity cells exactly: 3,200 games,
0 differing.</li>
<li><strong>Retained failures:</strong> a first scoring attempt failed admission before any game (fixed in 112aff3f);
three EU-SE-1 pods never opened SSH and were deleted; a slow pod's 13 unstarted units moved to other pods. No cell was rerun.</li>
<li><strong>Not tested:</strong> L2 first, other switch points or shares, weights-only transfer, longer horizons, mirror training.</li>
</ul>
<p class="muted">Rendered from <code>experiments/data/curriculum/analysis.json</code>. Full write-up:
<code>experiments/curriculum.md</code>.</p>
</section>
</main></html>
"""


def main() -> None:
    analysis = json.loads((HERE / "analysis.json").read_text())
    (HERE / "report.html").write_text(render(analysis))


if __name__ == "__main__":
    main()
