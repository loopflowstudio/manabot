"""Report the magnetic-mirror study: win rates, paired gains, cost and rollout noise.

Reads `experiments.search_mirror` game and decision logs, the test-time compute
sweep's games (`experiments.search_scaling`, for the sampled policy, greedy policy
and PUCT baselines on the same deals) and `experiments.search_mirror_noise`
positions. Writes a self-contained `report.html` (inline SVG) and `summary.json`.

Every gain is paired by deal: two conditions differ only in the hero's search,
so a difference over the deals both finished cancels deal luck. Intervals are
95% normal intervals on paired game-score differences, with no correction for
comparing many settings.

    uv run python -m experiments.search_mirror_report --mirror DIR \
        --baseline DIR [DIR ...] --noise DIR --out OUT
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from html import escape
import json
import math
from pathlib import Path
import statistics

from experiments.search_mirror import ARMS, Arm
from experiments.search_mirror_noise import BETAS, SUBSAMPLES, NoiseRow, analyze
from experiments.search_scaling import BUDGETS
from experiments.search_scaling_report import (
    CSS,
    OPPONENT_TITLE,
    OPPONENTS,
    PAD,
    W,
    Axis,
    Estimate,
    _frame,
    _gz_jsonl,
    _jsonl,
    _pct,
    _series,
    _svg,
    _ticks,
    estimate,
    gain_chart,
)

H = 300
EXTRA_CSS = """
:root { --series-3: #1baf7a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --series-3: #199e70; } }
:root[data-theme="dark"] { --series-3: #199e70; }
.banner { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--series-2);
  border-radius: 8px; padding: 10px 14px; margin: 14px 0; color: var(--text-secondary); max-width: 860px; }
td.best { font-weight: 600; }
.row.single { grid-template-columns: minmax(0, 560px); }
"""
NOISE_BETAS = (0.01, 0.03, 0.1)


def setting_label(arm: Arm) -> str:
    text = f"depth {arm.depth}, β {arm.beta:g}"
    return text + (f", {arm.worlds} shuffles" if arm.worlds > 1 else "")


def short_label(arm: Arm) -> str:
    return f"d{arm.depth} β{arm.beta:g}" + (f" ×{arm.worlds}" if arm.worlds > 1 else "")


# ---------------------------------------------------------------------------
# Loading and statistics


def _scores(directories: Sequence[Path]) -> dict[str, dict[int, float]]:
    """condition -> deal -> score, first row per game key, valid games only."""
    seen: set[str] = set()
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for directory in directories:
        for path in sorted(directory.rglob("games-*.jsonl")):
            for row in _jsonl(path):
                if row["key"] in seen:
                    continue
                seen.add(row["key"])
                if row["score"] is not None:
                    out[row["condition"]][row["deal"]] = float(row["score"])
    return out


@dataclass(frozen=True)
class Cost:
    decisions: int
    seconds: float
    moves: float
    terminal: float

    def calls(self, arm: Arm) -> float:
        """Network calls per decision if every rollout runs to full depth."""
        return self.moves * arm.depth * arm.worlds


def _costs(directory: Path) -> dict[str, Cost]:
    sums: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for path in sorted(directory.rglob("decisions-*.jsonl.gz")):
        for row in _gz_jsonl(path):
            s = sums[row["key"].rsplit("/", 1)[0]]
            s[0] += 1
            s[1] += row["seconds"]
            s[2] += sum(1 for p in row["base"] if p > 0)
            s[3] += row["terminal_rollouts"]
    return {k: Cost(int(s[0]), s[1] / s[0], s[2] / s[0], s[3] / s[0]) for k, s in sums.items()}


def paired(a: dict[int, float], b: dict[int, float], deals: set[int] | None = None) -> Estimate | None:
    shared = a.keys() & b.keys()
    if deals is not None:
        shared &= deals
    return estimate([a[d] - b[d] for d in sorted(shared)])


# ---------------------------------------------------------------------------
# Charts


def _gain_axis(values: list[float]) -> Axis:
    low = math.floor((min(values + [0.0]) - 0.02) / 0.05) * 0.05
    high = math.ceil((max(values + [0.0]) + 0.02) / 0.05) * 0.05
    return Axis(low, high)


def _zero(y: Axis) -> list[str]:
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    py = y(0.0, bottom, top)
    return [f'<line x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}" stroke="var(--axis)" stroke-width="1.5"/>']


def _points(value: float) -> str:
    return f"{value * 100:+.0f}"


def depth_chart(gains: dict[str, list[tuple[int, Estimate]]]) -> str:
    """Gain over the sampled policy against rollout depth, one line per opponent."""
    depths = sorted({d for pts in gains.values() for d, _ in pts})
    x = Axis(depths[0] / 1.5, depths[-1] * 1.5, log=True)
    values = [v for pts in gains.values() for _, e in pts for v in (e.low, e.high)]
    y = _gain_axis(values)
    body = _frame([(float(d), str(d)) for d in depths], _ticks(y.low, y.high, 0.05), x, y,
                  "rollout depth in decisions (log scale)", _points)
    body += _zero(y)
    for index, opponent in enumerate(OPPONENTS):
        pts = [(float(d), e, f"Depth {d}, β 0.1, 1 shuffle, {OPPONENT_TITLE[opponent]}: "
                f"{e.mean * 100:+.1f} points [{e.low * 100:+.1f}, {e.high * 100:+.1f}], {e.n} paired deals")
               for d, e in gains[opponent]]
        body += _series(pts, x, y, f"--series-{index + 1}")
    return _svg(body, "gain over the sampled policy by rollout depth")


def cost_chart(puct: list[tuple[float, Estimate, str]], mirror: list[tuple[float, Estimate, str, str]],
               label_best: int) -> str:
    """Gain over the sampled policy against seconds per decision."""
    xs = [px for px, _, _ in puct] + [px for px, _, _, _ in mirror]
    lo, hi = min(xs) / 1.6, max(xs) * 1.6
    x = Axis(lo, hi, log=True)
    values = [v for _, e, _ in puct for v in (e.low, e.high)] + [v for _, e, _, _ in mirror for v in (e.low, e.high)]
    y = _gain_axis(values)
    nice = [m * 10.0**e for e in range(-3, 3) for m in (1, 3)]
    ticks = [(v, f"{v:g} s" if v >= 1 else f"{v * 1000:g} ms") for v in nice if lo <= v <= hi]
    body = _frame(ticks, _ticks(y.low, y.high, 0.05), x, y, "seconds per searched decision (log scale)", _points)
    body += _zero(y)
    body += _series(puct, x, y, "--series-2")
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    for i, (px, est, tip, label) in enumerate(mirror):
        body += _series([(px, est, tip)], x, y, "--series-1")
        if i == label_best:
            cx, cy = x(px, left, right), y(est.mean, bottom, top)
            body.append(f'<text x="{cx + 8:.1f}" y="{cy - 8:.1f}">{escape(label)}</text>')
    for px, est, _ in puct[-1:]:
        cx, cy = x(px, left, right), y(est.mean, bottom, top)
        body.append(f'<text x="{cx:.1f}" y="{cy + 22:.1f}" text-anchor="middle">PUCT 1024</text>')
    return _svg(body, "gain over the sampled policy by seconds per decision")


def noise_chart(rows: list[NoiseRow]) -> str:
    """Expected per-decision gain over the policy by rollouts per move, one line per beta."""
    x = Axis(SUBSAMPLES[0] / 1.5, SUBSAMPLES[-1] * 1.5, log=True)
    top_value = max(r.gain_over_policy for r in rows if r.beta in NOISE_BETAS)
    y = Axis(0.0, math.ceil((top_value + 0.002) / 0.005) * 0.005)
    body = _frame([(float(n), str(n)) for n in SUBSAMPLES], _ticks(0, y.high, 0.005), x, y,
                  "rollouts per move (log scale)", lambda v: f"{v:.3f}")
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    for index, beta in enumerate(NOISE_BETAS):
        pts = []
        for r in sorted((r for r in rows if r.beta == beta), key=lambda r: r.rollouts):
            e = Estimate(r.positions, r.gain_over_policy, r.gain_over_policy, r.gain_over_policy)
            pts.append((float(r.rollouts), e, f"β {beta:g}, {r.rollouts} rollouts per move: expected gain "
                        f"{r.gain_over_policy:.4f} (near-exact Q gives {r.gain_with_reference_q:.4f}); "
                        f"best move matches {_pct(r.argmax_agreement, 0)}"))
        body += _series(pts, x, y, f"--series-{index + 1}")
        last = pts[-1]
        cx, cy = x(last[0], left, right), y(last[1].mean, bottom, top)
        body.append(f'<text x="{cx - 6:.1f}" y="{cy - 9:.1f}" text-anchor="end">β {beta:g}</text>')
    return _svg(body, "expected gain by rollouts per move")


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="border-color:var({color})"></i>{escape(label)}</span>' for color, label in items
    ) + "</div>"


def _cell(e: Estimate | None) -> str:
    return "–" if e is None else f"{e.mean * 100:+.1f} <span style='color:var(--text-muted)'>±{(e.high - e.mean) * 100:.1f}</span>"


# ---------------------------------------------------------------------------
# Page


def build(mirror_dir: Path, baselines: Sequence[Path], noise_dir: Path, out: Path,
          baseline_summary: Path) -> dict:
    mirror = _scores([mirror_dir])
    base = _scores(baselines)
    costs = _costs(mirror_dir)
    puct_seconds = {k: v["diagnostics"]["seconds"]
                    for k, v in json.loads(baseline_summary.read_text())["conditions"].items()
                    if v.get("diagnostics")}
    noise_summary, noise_rows = analyze(noise_dir)

    summary: dict = {"arms": {}, "noise": {"summary": noise_summary, "rows": [asdict(r) for r in noise_rows]}}
    gains: dict[tuple[str, str], Estimate | None] = {}
    for arm in ARMS:
        scores = mirror.get(arm.key, {})
        o = arm.opponent
        gains[arm.key, "sampled"] = paired(scores, base[f"{o}/policy"])
        gains[arm.key, "greedy"] = paired(scores, base[f"{o}/greedy"])
        gains[arm.key, "puct"] = paired(scores, base[f"{o}/puct-1024-oracle"])
        cost = costs.get(arm.key)
        summary["arms"][arm.key] = {
            "games": len(scores),
            "score": (e.to_json() if (e := estimate(list(scores.values()))) else None),
            "gain_over_sampled": (g.to_json() if (g := gains[arm.key, "sampled"]) else None),
            "gain_over_greedy": (g.to_json() if (g := gains[arm.key, "greedy"]) else None),
            "gain_over_puct_1024_oracle": (g.to_json() if (g := gains[arm.key, "puct"]) else None),
            "seconds_per_decision": cost.seconds if cost else None,
            "network_calls_per_decision": cost.calls(arm) if cost else None,
            "moves_per_decision": cost.moves if cost else None,
        }

    settings = sorted({(a.depth, a.beta, a.worlds) for a in ARMS}, key=lambda s: (s[2], -s[1], s[0]))
    by_setting = {s: {a.opponent: a for a in ARMS if (a.depth, a.beta, a.worlds) == s} for s in settings}

    def mean_gain(setting: tuple[int, float, int]) -> float:
        found = [gains[a.key, "sampled"] for a in by_setting[setting].values()]
        return statistics.fmean(g.mean for g in found if g) if all(found) else -1.0

    best = max(settings, key=mean_gain)
    best_arms = by_setting[best]
    played = sum(len(v) for k, v in mirror.items())
    planned = len(ARMS) * 500
    deals = {o: set().union(*(mirror.get(a.key, {}).keys() for a in ARMS if a.opponent == o)) for o in OPPONENTS}

    tiles = [f'<div class="tile"><div class="k">Games played</div><div class="v">{played:,}</div>'
             f'<div class="s">of {planned:,} planned · 0 invalid · run still going</div></div>']
    for o in OPPONENTS:
        g, p = gains[best_arms[o].key, "sampled"], gains[best_arms[o].key, "puct"]
        tiles.append(f'<div class="tile"><div class="k">Best setting {OPPONENT_TITLE[o]}</div>'
                     f'<div class="v">{g.mean * 100:+.1f} pts</div>'
                     f'<div class="s">over the sampled policy [{g.low * 100:+.1f}, {g.high * 100:+.1f}] · '
                     f'{p.mean * 100:+.1f} vs PUCT-1024</div></div>')
    best_cost = costs[best_arms["policy"].key]
    tiles.append(f'<div class="tile"><div class="k">Best setting cost</div>'
                 f'<div class="v">{best_cost.seconds:.1f} s</div>'
                 f'<div class="s">per decision · about {best_cost.calls(best_arms["policy"]):.0f} network calls '
                 f'(PUCT-1024: 1,024)</div></div>')

    sections: list[str] = []
    gain_cards = []
    for o in OPPONENTS:
        rows = [(setting_label(by_setting[s][o]), gains[by_setting[s][o].key, "sampled"]) for s in settings]
        gain_cards.append(f'<div class="card"><h3>Mirror over the sampled policy, {OPPONENT_TITLE[o]}</h3>'
                          f'<div class="sub">Paired by deal, percentage points, 95% intervals</div>'
                          f'{gain_chart(rows, "gain")}</div>')
    sections.append(
        "<h2>What mirror search adds</h2>"
        '<p class="note">Mirror search samples its move from the updated policy, so the matching baseline is '
        "the policy's own sampled play. β is the pull back toward the policy: large β barely moves it, "
        "small β trusts the rollout values. Shuffles are independent rollouts per move. β 1 stays within about 0.005 "
        "total variation of the policy, so its bars are a check that pairing is fair: they should straddle zero.</p>"
        '<div class="row pairs">' + "".join(gain_cards) + "</div>")

    depth_gains = {o: [(a.depth, g) for a in sorted(ARMS, key=lambda a: a.depth)
                       if a.opponent == o and a.beta == 0.1 and a.worlds == 1
                       and (g := gains[a.key, "sampled"]) is not None] for o in OPPONENTS}
    sections.append(
        "<h2>How deep to roll out</h2>"
        '<p class="note">At β 0.1 with one shuffle. About ten decisions make a turn, so depth 20 looks about '
        "two turns ahead. Depth 2 adds little; 10 and 20 are close; depth 40 is no better than 20 and costs twice as much.</p>"
        + _legend([("--series-1", "Against the bare policy"), ("--series-2", "Against the heuristic")])
        + f'<div class="row single"><div class="card"><h3>Gain over the sampled policy by depth</h3>'
          f'<div class="sub">Paired by deal, points</div>{depth_chart(depth_gains)}</div></div>')

    cost_cards = []
    for o in OPPONENTS:
        puct = []
        for sims in BUDGETS:
            key = f"{o}/puct-{sims}-oracle"
            g = paired(base[key], base[f"{o}/policy"], deals[o])
            if g is not None and key in puct_seconds:
                puct.append((puct_seconds[key], g, f"PUCT {sims} sims, oracle hands: {g.mean * 100:+.1f} points "
                             f"[{g.low * 100:+.1f}, {g.high * 100:+.1f}] on {g.n} of the same deals; "
                             f"{puct_seconds[key]:.2f} s per decision"))
        points = []
        best_index = 0
        for s in settings:
            arm = by_setting[s][o]
            g, c = gains[arm.key, "sampled"], costs.get(arm.key)
            if g is None or c is None:
                continue
            if s == best:
                best_index = len(points)
            points.append((c.seconds, g, f"Mirror {setting_label(arm)}: {g.mean * 100:+.1f} points "
                           f"[{g.low * 100:+.1f}, {g.high * 100:+.1f}], {g.n} paired deals; {c.seconds:.2f} s per "
                           f"decision, about {c.calls(arm):.0f} network calls", short_label(arm)))
        cost_cards.append(f'<div class="card"><h3>{OPPONENT_TITLE[o].capitalize()}</h3>'
                          f'<div class="sub">Gain over the sampled policy against time per decision</div>'
                          f"{cost_chart(puct, points, best_index)}</div>")
    puct_cards = []
    for o in OPPONENTS:
        rows = [(setting_label(by_setting[s][o]), gains[by_setting[s][o].key, "puct"]) for s in settings]
        puct_cards.append(f'<div class="card"><h3>Mirror minus PUCT-1024, {OPPONENT_TITLE[o]}</h3>'
                          f'<div class="sub">Both with oracle hands, paired by deal, points</div>'
                          f'{gain_chart(rows, "puct")}</div>')
    sections.append(
        "<h2>Mirror against PUCT</h2>"
        '<p class="note">PUCT points are the determinized tree search from the test-time compute sweep with oracle '
        "hands, paired against the sampled policy on the same deals mirror played. Its timings came partly from "
        "rented hosts under different load, so compare costs only roughly. Neither searcher played the other; each "
        "difference is two searchers against the same opponent.</p>"
        + _legend([("--series-1", "Mirror settings (hover each point)"), ("--series-2", "PUCT, 16 to 1,024 sims")])
        + '<div class="row pairs">' + "".join(cost_cards) + "</div>"
        + '<div class="row pairs" style="margin-top:16px">' + "".join(puct_cards) + "</div>")

    ns = noise_summary
    sections.append(
        "<h2>How many rollouts a move needs</h2>"
        f'<p class="note">{ns["positions"]} positions from sampled-policy games, each searched once at depth 20 with '
        "64 oracle-hand rollouts per move. Each point uses n of them as the estimate and the remaining 64 − n as "
        "the reference Q, and reports the reference-Q gain of the resulting move distribution over the policy at "
        "one decision (signed value units). A single rollout's value varies by "
        f'±{ns["single_rollout_sd"]:.2f} while the real gaps between moves are about ±{ns["between_move_sd_of_q"]:.2f}, '
        "so one rollout is mostly noise. Small β gains most, but only once Q averages enough rollouts.</p>"
        + _legend([("--series-1", "β 0.01"), ("--series-2", "β 0.03"), ("--series-3", "β 0.1")])
        + f'<div class="row single"><div class="card"><h3>Expected gain at one decision</h3>'
          f'<div class="sub">Over the policy, by rollouts per move</div>{noise_chart(noise_rows)}</div></div>'
        + "<details><summary>Every step size</summary><table><tr><th>β</th>"
        + "".join(f"<th>{n} rollouts</th>" for n in SUBSAMPLES) + "<th>near-exact Q</th></tr>"
        + "".join(
            f"<tr><td>{beta:g}</td>"
            + "".join(f"<td>{r.gain_over_policy:.4f}</td>" for r in sorted(
                (r for r in noise_rows if r.beta == beta), key=lambda r: r.rollouts))
            + f"<td>{max(r.gain_with_reference_q for r in noise_rows if r.beta == beta):.4f}</td></tr>"
            for beta in BETAS)
        + "</table></details>")

    header = ("<tr><th>Setting</th>"
              + "".join(f"<th>vs sampled ({o})</th><th>vs PUCT-1024 ({o})</th>" for o in OPPONENTS)
              + "<th>s / decision</th><th>network calls</th><th>deals</th></tr>")
    body_rows = []
    for s in settings:
        arms = by_setting[s]
        c = costs.get(arms["policy"].key)
        cls = ' class="best"' if s == best else ""
        body_rows.append(
            f"<tr><td{cls}>{escape(setting_label(arms['policy']))}</td>"
            + "".join(f"<td>{_cell(gains[arms[o].key, 'sampled'])}</td><td>{_cell(gains[arms[o].key, 'puct'])}</td>"
                      for o in OPPONENTS)
            + (f"<td>{c.seconds:.2f}</td><td>{c.calls(arms['policy']):.0f}</td>" if c else "<td>–</td><td>–</td>")
            + f"<td>{min(len(mirror.get(a.key, {})) for a in arms.values())}</td></tr>")
    sections.append("<h2>Every setting</h2>"
                    '<p class="note">Paired gains in points with 95% half-widths. Network calls assume every '
                    "rollout runs its full depth; rollouts that end the game stop early.</p>"
                    f'<div class="card" style="overflow-x:auto"><table>{header}{"".join(body_rows)}</table></div>')

    sections.append(
        "<h2>Method</h2>"
        '<p class="note">Hero: MTG-183\'s final checkpoint (10-deck Elemental, world 13). Each searched decision '
        "tries every policy-supported move from worlds holding the opponent's true hand, with the libraries "
        "reshuffled per world, plays the given depth of policy-sampled decisions, scores the end with the value "
        "network (or the real result if the game ends) and samples from "
        "p ∝ exp((Q + α log uniform + β log policy) / (α + β)), α 0.02. Opponents: the same checkpoint sampling "
        "its policy, and the scripted heuristic. Deals: the first five per ordered deck cell of the test-time "
        "compute sweep (500 per setting when complete), shared with its sampled, greedy and PUCT-1024 arms. "
        "Games ran on one Mac mini, seven workers on one thread each. Oracle hands make every result here an upper "
        "bound on what this search does without seeing the opponent's hand.</p>")

    lede = ("Interim: can the magnetic-mirror local update, given the opponent's true hand, add strength on top of "
            f"MTG-183's policy, and what does it cost next to PUCT? {played:,} of {planned:,} games so far "
            f"(about {min(len(v) for v in mirror.values())} deals per setting).")
    banner = ('<div class="banner">Interim results from a run still in progress. With sixteen settings and no '
              "multiple-comparison correction, a single setting clearing its interval is a lead, not a finding.</div>")
    html = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Mirror search study</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
            f"<h1>Mirror search on MTG-183</h1><p class=\"lede\">{escape(lede)}</p>{banner}"
            f'<div class="tiles">{"".join(tiles)}</div>' + "".join(sections) + "</main></body></html>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(html)
    summary["games"] = played
    summary["best_setting"] = setting_label(best_arms["policy"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.mirror, args.baseline, args.noise, args.out, args.baseline_summary)
    print(json.dumps({"games": summary["games"], "best": summary["best_setting"]}))


if __name__ == "__main__":
    main()
