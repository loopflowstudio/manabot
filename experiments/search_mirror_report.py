"""Report the magnetic-mirror study: win rates, paired gains, cost and rollout noise.

Reads `experiments.search_mirror` game and decision logs, the test-time compute
sweep's games (`experiments.search_scaling`, for the sampled policy, greedy policy
and PUCT baselines on the same deals) and `experiments.search_mirror_noise`
positions. Writes a self-contained `report.html` (inline SVG) and `summary.json`.

Gains pair whole-game scores by starting deal, deck cell and seat. Nominal normal
intervals and cell-stratified joint bootstrap intervals are exploratory: repeated
inspection and selection across settings prevent confirmatory interpretation.

    uv run python -m experiments.search_mirror_report --mirror DIR \
        --baseline DIR [DIR ...] --noise DIR --out OUT
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from experiments.elemental_grid import Game, schedule
from experiments.search_mirror import (
    ARMS,
    DEALS_PER_CELL,
    V2_ARMS,
    Arm,
    MirrorDecision,
    MirrorRow,
    reconcile_extension,
)
from experiments.search_mirror_noise import BETAS, SUBSAMPLES, NoiseRow, analyze
from experiments.search_scaling import BUDGETS, GameRow
from experiments.search_scaling_report import (
    CSS,
    OPPONENT_TITLE,
    OPPONENTS,
    PAD,
    Axis,
    Estimate,
    W,
    _frame,
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
details, .table-scroll { overflow-x: auto; }
"""
NOISE_BETAS = (0.01, 0.03, 0.1)


def setting_label(arm: Arm) -> str:
    text = f"depth {arm.depth}, β {arm.beta:g}"
    return text + (f", {arm.worlds} shuffles" if arm.worlds > 1 else "")


def short_label(arm: Arm) -> str:
    return f"d{arm.depth} β{arm.beta:g}" + (f" ×{arm.worlds}" if arm.worlds > 1 else "")


# ---------------------------------------------------------------------------
# Loading and statistics


def _outcomes(
    directories: Sequence[Path], *, mirror: bool, arms: tuple[Arm, ...] = ARMS,
) -> dict[str, dict[int, MirrorRow | GameRow]]:
    """Read retained records strictly; conflicting copies and identity drift fail closed."""
    expected = {g.deal: g for g in schedule()}
    out: dict[str, dict[int, MirrorRow | GameRow]] = defaultdict(dict)
    for directory in directories:
        for path in sorted(directory.rglob("games-*.jsonl")):
            for raw in _jsonl(path):
                row = MirrorRow(**raw) if mirror else GameRow(**raw)
                game = expected[row.deal]
                if (row.key, row.model_deck, row.opponent_deck, row.model_seat) != (
                    f"{row.condition}/{game.deal}", game.model_deck, game.heuristic_deck, game.model_seat
                ):
                    raise ValueError(f"Schedule identity mismatch: {row.key}")
                if row.score not in (None, 0, 0.5, 1):
                    raise ValueError(f"Invalid score: {row.key}")
                previous = out[row.condition].get(row.deal)
                if previous is not None and (mirror or previous != row):
                    raise ValueError(f"Duplicate/conflicting outcome: {row.key}")
                if mirror:
                    arm = next((a for a in arms if a.key == row.condition), None)
                    if arm is None:
                        raise ValueError(f"Unexpected mirror arm: {row.key}")
                    if game.index >= DEALS_PER_CELL or (row.opponent, row.depth, row.beta) != (
                        arm.opponent, arm.depth, arm.beta
                    ):
                        raise ValueError(f"Arm identity mismatch: {row.key}")
                    if isinstance(row, MirrorRow) and (
                        (row.study, row.sampler) != (arm.study, arm.sampler)
                        or row.worlds not in ((None, arm.worlds) if arm.study == "v1" else (arm.worlds,))
                    ):
                        raise ValueError(f"Sampler/world identity mismatch: {row.key}")
                out[row.condition][row.deal] = row
    return out


def _scores(records: dict[str, dict[int, MirrorRow | GameRow]]) -> dict[str, dict[int, float]]:
    return {key: {d: float(r.score) for d, r in rows.items()
                  if r.failure is None and r.score is not None} for key, rows in records.items()}


@dataclass(frozen=True)
class Cost:
    decisions: int
    seconds: float
    moves: float
    terminal: float
    seconds_median: float
    seconds_p95: float

    def work_proxy(self, arm: Arm) -> float:
        """Supported moves × depth × worlds; neither actual commands nor full-depth work units."""
        return self.moves * arm.depth * arm.worlds


def _costs(directory: Path) -> dict[str, Cost]:
    sums: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    times: dict[str, list[float]] = defaultdict(list)
    for path in sorted(directory.rglob("decisions-*.jsonl.gz")):
        # Unlike the live-snapshot reader, final analysis must reject truncated tails.
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = MirrorDecision(**json.loads(line))
                key = row.key.rsplit("/", 1)[0]
                s = sums[key]
                s[0] += 1
                s[1] += row.seconds
                # Q presence records a rollout even when rounded base probability is zero.
                s[2] += sum(q is not None for q in row.q)
                s[3] += row.terminal_rollouts
                times[key].append(row.seconds)
    return {k: Cost(int(s[0]), s[1] / s[0], s[2] / s[0], s[3] / s[0],
                    float(np.median(times[k])), float(np.quantile(times[k], 0.95)))
            for k, s in sums.items()}


def _sensitivity(a: dict[int, float], b: dict[int, float], planned: set[int]) -> list[float]:
    """Full scheduled denominator; unresolved outcomes independently range over [0, 1]."""
    return [sum(a.get(d, 0.0) - b.get(d, 1.0) for d in planned) / len(planned),
            sum(a.get(d, 1.0) - b.get(d, 0.0) for d in planned) / len(planned)]


def _bootstrap(
    scores: dict[str, dict[int, float]], games: list[Game], comparisons: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], Estimate], set[int]]:
    """Resample shared deal vectors jointly within each fixed deck cell.

    The common-complete subset is explicit; a cell with no complete vector makes
    the full-grid bootstrap unavailable. Missing outcomes still enter sensitivity.
    """
    common = {g.deal for g in games}.intersection(*(set(v) for v in scores.values()))
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for game in games:
        cells[game.model_deck, game.heuristic_deck].append(game.deal)
    rng = np.random.default_rng(26901)
    draws = np.zeros((10000, len(comparisons)))
    for cell in cells.values():
        deals = sorted(set(cell) & common)
        if not deals:
            return {}, common
        differences = np.asarray([[scores[a][d] - scores[b][d] for a, b in comparisons] for d in deals])
        indices = rng.integers(0, len(deals), (10000, len(deals)))
        draws += differences[indices].mean(1) / len(cells)
    result = {}
    for i, pair in enumerate(comparisons):
        # Fixed cells get equal weight, including when coverage is incomplete.
        a, b = pair
        mean = statistics.fmean(statistics.fmean(scores[a][d] - scores[b][d]
                                 for d in cell if d in common) for cell in cells.values())
        low, high = np.quantile(draws[:, i], [0.025, 0.975])
        result[pair] = Estimate(len(common), mean, float(low), float(high))
    return result, common


def paired(a: dict[int, float], b: dict[int, float], deals: set[int] | None = None) -> Estimate | None:
    shared = a.keys() & b.keys()
    if deals is not None:
        shared &= deals
    return estimate([a[d] - b[d] for d in sorted(shared)])


def _coverage(rows: dict[int, MirrorRow | GameRow], games: list[Game]) -> dict[str, Any]:
    """JSON-facing coverage includes absent slots, never just successfully read rows."""
    planned = {g.deal for g in games}
    valid = {d: r for d, r in rows.items() if d in planned and r.score is not None and r.failure is None}
    cells: dict[str, dict[str, int]] = {}
    for game in games:
        cell = cells.setdefault(f"{game.model_deck}/{game.heuristic_deck}", {"planned": 0, "valid": 0})
        cell["planned"] += 1
        cell["valid"] += int(game.deal in valid)
    return {"planned": len(planned), "valid": len(valid), "missing": len(planned - rows.keys()),
            "invalid": len(planned & rows.keys()) - len(valid),
            "wins": sum(r.score == 1 for r in valid.values()),
            "draws": sum(r.score == 0.5 for r in valid.values()),
            "losses": sum(r.score == 0 for r in valid.values()),
            "seats": dict(Counter(str(r.model_seat) for r in valid.values())), "cells": cells,
            "missing_deals": sorted(planned - rows.keys()),
            "failures": [asdict(r) for d, r in rows.items() if d in planned and d not in valid]}


def _reconciliation(mirror_dir: Path) -> dict[str, Any]:
    """Bind the prior full decision audit to the exact bytes read by this report."""
    root = mirror_dir.parent
    inventory = json.loads((root / "reconciliation/remote-inventory.json").read_text())
    for remote, expected in inventory["files"].items():
        prefix, relative = remote.split("/", 2)[1:]
        path = root / ("games" if prefix == "mirror-v1" else "noise") / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Evidence changed since reconciliation: {path}")
    return json.loads((root / "reconciliation/execution.json").read_text())


@dataclass(frozen=True)
class Contrast:
    """A paired linear contrast of game scores, including four-cell interactions."""

    label: str
    terms: dict[str, int]


def _factorial_contrasts() -> list[Contrast]:
    keys = {(a.sampler, a.worlds): a.key for a in V2_ARMS}
    contrasts = [Contrast(f"{a.sampler} {a.worlds} minus sampled policy", {a.key: 1, "policy/policy": -1})
                 for a in V2_ARMS]
    contrasts.extend(Contrast(f"oracle minus uniform, {w} shuffles",
                             {keys["oracle", w]: 1, keys["uniform", w]: -1}) for w in (4, 8, 16))
    for low, high in ((4, 8), (8, 16)):
        contrasts.extend(Contrast(f"{sampler}, {high} minus {low} shuffles",
                                 {keys[sampler, high]: 1, keys[sampler, low]: -1})
                         for sampler in ("oracle", "uniform"))
        contrasts.append(Contrast(f"uniform minus oracle shuffle gain, {high} minus {low}", {
            keys["uniform", high]: 1, keys["uniform", low]: -1,
            keys["oracle", high]: -1, keys["oracle", low]: 1,
        }))
    return contrasts


def _linear_contrast(
    contrast: Contrast, scores: dict[str, dict[int, float]], games: list[Game],
) -> dict[str, Any]:
    """Pair all contributing arms before resampling; missing slots keep full bounds.

    Each contrast uses its own complete deal vectors. Interactions resample the
    four-arm difference, never independent arm means or independent intervals.
    """
    planned = {g.deal for g in games}
    common = planned.intersection(*(scores.get(k, {}).keys() for k in contrast.terms))
    values = {d: sum(weight * scores[key][d] for key, weight in contrast.terms.items())
              for d in sorted(common)}
    nominal = estimate(list(values.values()))
    boot, _ = _bootstrap({"contrast": values, "zero": {d: 0.0 for d in common}},
                         games, [("contrast", "zero")])
    interval = boot.get(("contrast", "zero"))
    bounds = [0.0, 0.0]
    for deal in planned:
        for key, weight in contrast.terms.items():
            value = scores.get(key, {}).get(deal)
            bounds[0] += weight * value if value is not None else min(0, weight)
            bounds[1] += weight * value if value is not None else max(0, weight)
    return {
        "label": contrast.label, "terms": contrast.terms, "planned": len(planned),
        "paired": len(common), "missing_deals": sorted(planned - common),
        "normal": nominal.to_json() if nominal else None,
        "cell_stratified_bootstrap": interval.to_json() if interval else None,
        "failure_sensitivity": [v / len(planned) for v in bounds],
    }


def _extension_report(directory: Path, baselines: dict[str, dict[int, MirrorRow | GameRow]]) -> dict[str, Any]:
    """Read the extension snapshot without treating an absent directory as no games.

    Dated execution receipts must accompany it. Resume reconciliation is shared
    with the runner; this does not certify native admission or replay equivalence.
    """
    receipt_dir = directory / "reconciliation"
    receipts = {name: json.loads((receipt_dir / f"{name}.json").read_text())
                for name in ("remote-inventory", "launcher-inventory", "validation")}
    records = _outcomes([directory / "games"], mirror=True, arms=V2_ARMS)
    reconcile_extension(directory / "games")
    remote = receipts["remote-inventory"]
    validation = receipts["validation"]
    # A prelaunch receipt cannot certify later outcomes or their provenance.
    if records and not remote["v2_directory_exists"]:
        raise ValueError("Extension outcomes require refreshed execution reconciliation")
    games = [g for g in schedule() if g.index < DEALS_PER_CELL]
    scores = _scores(records) | {"policy/policy": _scores(baselines).get("policy/policy", {})}
    costs = _costs(directory / "games")
    arms = {}
    for arm in V2_ARMS:
        rows = records.get(arm.key, {})
        cost = costs.get(arm.key)
        arms[arm.key] = {
            "setting": asdict(arm), "coverage": _coverage(rows, games),
            "score": (e.to_json() if (e := estimate(list(scores.get(arm.key, {}).values()))) else None),
            "search_cost": asdict(cost) if cost else None,
            "hero_seconds": sum(r.hero_seconds for r in rows.values()) if rows else None,
            "game_wall_seconds": [r.game_seconds for r in rows.values() if isinstance(r, MirrorRow)] or None,
        }
    valid = sum(a["coverage"]["valid"] for a in arms.values())
    return {
        "batching": (json.loads((directory / "batching/summary.json").read_text())
                     if (directory / "batching/summary.json").exists() else None),
        "latest_inventory": (json.loads((receipt_dir / "batching-inventory.json").read_text())
                             if (receipt_dir / "batching-inventory.json").exists() else None),
        "protocol": "experiments/protocols/search-mirror-v2.md", "alpha": 0.02,
        "planned_games": len(V2_ARMS) * len(games), "valid_games": valid,
        "missing_games": sum(a["coverage"]["missing"] for a in arms.values()),
        "invalid_games": sum(a["coverage"]["invalid"] for a in arms.values()),
        "execution": remote.get("execution", "not launched" if not remote["v2_directory_exists"] else "retained snapshot; review completeness"),
        "observed_utc": remote["observed_utc"],
        "capacity_reason": receipts["launcher-inventory"].get("status_note") or receipts["launcher-inventory"].get("quote_error") or "no capacity error",
        "arms": arms, "control": _coverage(baselines.get("policy/policy", {}), games),
        "contrasts": [_linear_contrast(c, scores, games) for c in _factorial_contrasts()],
        "context_only_arm": "policy/mirror-d20-b0.03-w4",
        "uncertainty": "exploratory; 10000 whole-deal resamples within fixed cells, seed 26901; complete vectors per contrast",
        "health": validation.get("health", "unassessed: no native smoke or games" if not records else "requires execution evidence review"),
        "native_support": validation.get("native_support", "unresolved; accounting and sampler fixtures do not admit the mini runtime"),
        "replay_identity": validation.get("replay_identity", "absent: no game attempts" if not records else "retained attempts are not replay proof"),
        "learning": "inference-only; no new training, exports or callbacks expected",
        "training_card_exposure": None, "generalization": "untested; reused decks, deals and one training seed",
        "acceptance": "unresolved; missing outcomes establish neither gains nor equivalence",
        "measured_eta_seconds": validation.get("measured_eta_seconds"),
        "timing_note": validation.get("timing_note", "No extension timing retained."),
        "inclusive_worker_hours": validation.get("inclusive_worker_hours"),
        "rental_spend_usd": receipts["launcher-inventory"]["new_rental_spend_usd"],
        "deadline_unix": receipts["validation"]["deadline_unix"],
        "evidence_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(directory.rglob("*")) if p.is_file()},
    }


def _batching_section(extension: dict[str, Any]) -> str:
    """Operational diagnostics never fill scientific game slots or blend costs."""
    batching = extension.get("batching")
    if batching is None:
        return ""
    inventory = extension.get("latest_inventory")
    status = (
        f'<p class="note">Later ledger-only inventory: {inventory["completed"]}/3,000 '
        f'completed rows, {inventory["failed"]} failures at {escape(inventory["observed_utc"])}; '
        f'{len(inventory["workers"])} sequential workers. Private attempt blobs were not '
        'collected in this inventory, so the scientific tables retain their earlier closed-evidence snapshot.</p>'
        if inventory else ""
    )
    rows = ''.join(
        f'<tr><td>{cell["step"]}</td><td>{escape(cell["sampler"])}</td><td>{cell["worlds"]}</td>'
        f'<td>{cell["sequential_seconds"]:.3f}</td><td>{cell["batched_seconds"]:.3f}</td>'
        f'<td>{cell["speedup"]:.2f}×</td><td>{"pass" if cell["passed"] else "FAIL"}</td></tr>'
        for cell in batching["cells"]
    )
    return (
        '<h3>W13 batching admission</h3>'
        f'<p><strong>{escape(batching["decision"])}.</strong> {escape(batching["reason"])}</p>'
        f'<p class="note">{escape(batching["correctness"])} {escape(batching["limits"])}</p>'
        '<details><summary>All fixed-position timings, seconds (one pair per cell)</summary>'
        '<div class="table-scroll" role="region" aria-label="Batching diagnostics" tabindex="0">'
        '<table><tr><th>Root step</th><th>Hands</th><th>Shuffles</th><th>Sequential s</th>'
        '<th>Batched s</th><th>Ratio</th><th>Receipt parity</th></tr>' + rows + '</table></div>'
        '<p class="note">These are paired fixed searches on the laptop, separate from mini game cost. '
        'Two harness API errors are retained; six opening cells completed before the first error, '
        'and the corrected invocation measured only the other roots. No failed comparison was replaced. '
        'Source, complete private receipts, process logs and errors are stored under '
        'search-mirror-v2/batching/.</p></details>' + status
    )


def _extension_section(extension: dict[str, Any]) -> str:
    rows = []
    costs = []
    for arm in extension["arms"].values():
        setting, coverage = arm["setting"], arm["coverage"]
        rows.append(f'<tr><td>{setting["sampler"]}</td><td>{setting["worlds"]}</td>'
                    f'<td>{coverage["valid"]}/{coverage["planned"]}</td>'
                    f'<td>{coverage["missing"]}</td><td>{coverage["invalid"]}</td>'
                    f'<td>{_interval(arm["score"])}</td></tr>')
        cost, wall = arm["search_cost"], arm["game_wall_seconds"]
        timing = (
            f'<td>{cost["decisions"]:,}</td><td>{cost["seconds"]:.3f}</td>'
            f'<td>{cost["seconds_median"]:.3f} / {cost["seconds_p95"]:.3f}</td>'
            if cost else '<td colspan="3">unavailable</td>'
        )
        game_timing = f'{statistics.mean(wall):.1f}' if wall else 'unavailable'
        hero_time = f'{arm["hero_seconds"]:.1f}' if arm["hero_seconds"] is not None else 'unavailable'
        costs.append(f'<tr><td>{setting["sampler"]}</td><td>{setting["worlds"]}</td>'
                     f'{timing}<td>{game_timing}</td><td>{hero_time}</td></tr>')
    contrasts = []
    for c in extension["contrasts"]:
        lo, hi = c["failure_sensitivity"]
        contrasts.append(f'<tr><td>{escape(c["label"])}</td><td>{c["paired"]}/{c["planned"]}</td>'
                         f'<td>{_interval(c["normal"])}</td><td>{_interval(c["cell_stratified_bootstrap"])}</td>'
                         f'<td>[{100*lo:+.1f}, {100*hi:+.1f}]</td></tr>')
    return (
        _batching_section(extension) + '<h2>Extension: oracle versus uniform, 4 / 8 / 16 shuffles</h2>'
        f'<p><strong>{extension["valid_games"]:,}/{extension["planned_games"]:,} valid games; '
        f'{escape(extension["execution"])}.</strong> Snapshot: {escape(extension["observed_utc"])}. '
        f'Etude receipt: {escape(extension["capacity_reason"])}.</p>'
        '<p class="note">All six cells fix depth 20, α 0.02 and β 0.01 against the raw sampled MTG-183 '
        'checkpoint policy. The opponent has no search, mirror update, greedy or EMA substitution. '
        'Uniform samples physical unseen copies subject to public information; oracle retains the true hand. '
        'These are comparisons against a common opponent, not oracle-versus-uniform head-to-head games. '
        'The historical β 0.03/four-shuffle result is context only.</p>'
        '<div class="table-scroll" role="region" aria-label="Extension coverage" tabindex="0">'
        '<table><tr><th>Hands</th><th>Shuffles</th><th>Valid/planned</th><th>Missing</th><th>Invalid</th>'
        '<th>Score %, nominal 95%</th></tr>' + ''.join(rows) + '</table></div>'
        '<p class="note">Partial snapshots follow the fixed execution order; completed deals need not '
        'represent the full grid. Sparse nominal intervals cannot establish strength or shuffle effects. '
        'A one-deal nominal interval is degenerate by convention, not an uncertainty estimate; zero-width '
        'intervals from a few identical observed differences do not demonstrate equivalence.</p>'
        '<details><summary>All planned paired contrasts and missing-evidence bounds</summary>'
        '<table><tr><th>Contrast</th><th>Paired/planned</th><th>Nominal 95%, points</th>'
        '<th>Stratified 95%, points</th><th>Sensitivity, points</th></tr>' + ''.join(contrasts) + '</table></details>'
        '<p class="note">Unavailable is not zero. Each contrast pairs all contributing scores on the same deal; '
        'the interaction is (uniform higher − uniform lower) − (oracle higher − oracle lower). '
        'Bootstrap resamples those complete vectors within fixed deck cells, 10,000 times with seed 26901; '
        'an empty cell makes that interval unavailable. Full-denominator sensitivity lets every missing score '
        'range from 0 to 1; these bounds are not confidence intervals. No direction of interaction is assumed.</p>'
        '<details><summary>Measured campaign cost for retained outcomes</summary>'
        '<table><tr><th>Hands</th><th>Shuffles</th><th>Search decisions</th>'
        '<th>Mean s/decision</th><th>Median / p95 s/decision</th><th>Mean wall s/game</th>'
        '<th>Total hero search s</th></tr>' + ''.join(costs) + '</table>'
        '<p class="note">Only closed campaign outcomes contribute. In-flight work, smoke, setup and '
        'interrupted attempts are excluded. Game wall time includes opponent play; hero search time '
        'does not. Game length and concurrent load confound these timings; this is not an equal-compute '
        'strength comparison or inclusive worker-hour accounting.</p></details>'
        f'<p class="note">Health: {escape(extension["health"])}. '
        f'Native support: {escape(extension["native_support"])}. '
        f'Replay identity: {escape(extension["replay_identity"])}. '
        'Smoke outcomes are separate from the 3,000 planned games and establish no strength result. '
        'Policy-only timing stays in its separate historical table. No new rental spend is recorded. '
        f'{escape(extension["timing_note"])} '
        'Inclusive setup/worker time and a reliable full-grid ETA require execution accounting. '
        'The shared execution deadline remains November 16, 2026, 18:16:29 UTC. Missing games are neither '
        'failed games nor evidence of no benefit. Training card '
        'exposure remains absent independently of evaluation coverage; no new learning or generalization is '
        'established. Evidence review must retain these limits. This analysis starts no execution.</p>'
    )


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
    top_value = max(r.gain_high for r in rows if r.beta in NOISE_BETAS)
    low_value = min(0.0, min(r.gain_low for r in rows if r.beta in NOISE_BETAS))
    y = Axis(math.floor(low_value / 0.005) * 0.005, math.ceil((top_value + 0.002) / 0.005) * 0.005)
    body = _frame([(float(n), str(n)) for n in SUBSAMPLES], _ticks(y.low, y.high, 0.005), x, y,
                  "rollouts per move (log scale)", lambda v: f"{v:.3f}")
    left, right, top, bottom = PAD["l"], W - PAD["r"], PAD["t"], H - PAD["b"]
    for index, beta in enumerate(NOISE_BETAS):
        pts = []
        for r in sorted((r for r in rows if r.beta == beta), key=lambda r: r.rollouts):
            e = Estimate(r.positions, r.gain_over_policy, r.gain_low, r.gain_high)
            pts.append((float(r.rollouts), e, f"β {beta:g}, {r.rollouts} rollouts per move: expected gain "
                        f"{r.gain_over_policy:.4f} (finite-reference optimized Q gives {r.gain_with_reference_q:.4f}); "
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


def _interval(value: dict[str, float] | None) -> str:
    if value is None:
        return "unavailable"
    return f'{100 * value["mean"]:+.1f} [{100 * value["low"]:+.1f}, {100 * value["high"]:+.1f}]'


def _evidence_sections(summary: dict[str, Any]) -> str:
    """Render coverage and limits from the same structured report, not a second store."""
    rows = []
    costs = []
    for key, arm in summary["arms"].items():
        coverage = arm["coverage"]
        for control, comparison in arm["comparisons"].items():
            valid = comparison["valid_only"]
            sensitivity = comparison["failure_sensitivity"]
            rows.append(f'<tr><td>{escape(key)}</td><td>{control}</td>'
                        f'<td>{valid["n"] if valid else 0}/{comparison["planned"]}</td>'
                        f'<td>{_interval(valid)}</td><td>{_interval(comparison["cell_stratified_bootstrap"])}</td>'
                        f'<td>[{100*sensitivity[0]:+.1f}, {100*sensitivity[1]:+.1f}]</td></tr>')
        c, g = arm["cost"], arm["game_hero_seconds"]
        costs.append(f'<tr><td>{escape(key)}</td><td>{coverage["wins"]}/{coverage["draws"]}/{coverage["losses"]}</td>'
                     f'<td>{c["seconds"]:.3f} / {c["seconds_median"]:.3f} / {c["seconds_p95"]:.3f}</td>'
                     f'<td>{g["mean"]:.1f} / {g["median"]:.1f} / {g["p95"]:.1f}</td>'
                     f'<td>{c["moves"]:.2f}</td><td>{c["terminal"] * c["decisions"]:.0f}</td></tr>')
    controls = []
    for key, c in summary["controls"].items():
        controls.append(f'<tr><td>{escape(key)}</td><td>{c["valid"]}/{c["planned"]}</td>'
                        f'<td>{c["missing"]}</td><td>{c["invalid"]}</td>'
                        f'<td>{c["wins"]}/{c["draws"]}/{c["losses"]}</td></tr>')
    contrasts = []
    for opponent, items in summary["setting_contrasts"].items():
        for row in items:
            contrasts.append(f'<tr><td>{escape(row["a"])} minus {escape(row["b"])}</td>'
                             f'<td>{_interval(row["normal"])}</td><td>{_interval(row["bootstrap"])}</td></tr>')
    policy_costs = ''.join(
        f'<tr><td>{key}</td><td>{c["seconds_per_decision"]:.6f}</td>'
        f'<td>{c["seconds_per_game"]:.3f} / {c["game_seconds_median"]:.3f} / {c["game_seconds_p95"]:.3f}</td>'
        f'<td>{c["hero_seconds"]:.3f}</td></tr>' for key, c in summary["policy_only_cost"].items()
    )
    health = summary["reconciliation"]
    d = health["decisions"]
    warnings = health["worker_log_messages"]
    return (
        '<h2>Coverage and uncertainty</h2><p class="note">All 16 mirror arms retain 500/500 valid outcomes, '
        'five per ordered deck cell across 100 cells, with 246 seat-0 and 254 seat-1 games per arm. '
        'There are no missing, invalid, duplicate or unexpected mirror outcomes. Sampled and greedy controls '
        'cover all 500 deals. PUCT controls have unresolved slots shown below. Full cell and seat coverage, '
        'missing deal IDs and retained failure records are in summary.json. Scores are wins + half draws; '
        'the controls below are restricted to the same first five deals per cell.</p>'
        '<details><summary>Control coverage</summary><table><tr><th>Control</th><th>Valid/planned</th>'
        '<th>Missing</th><th>Invalid</th><th>W/D/L</th></tr>' + ''.join(controls) + '</table></details>'
        '<p class="note">Intervals are exploratory. Nominal = mean ± 1.96 sample SD / √n. Bootstrap = '
        '10,000 joint whole-deal resamples within each fixed cell, seed 26901, equal cell weight. '
        'Each comparison family uses the common-complete subset across all eight mirror settings and its control; '
        'the PUCT-1024 policy families use 479 deals, the other displayed families 500. Pairing fixes starting '
        'deals, decks and seats, not subsequent draws. Sensitivity assigns each unresolved score 0 or 1 on the '
        'full 500-deal denominator; it is a missing-evidence range, not a confidence interval. Repeated interim '
        'inspection and winner selection are not corrected by either interval.</p>'
        '<details><summary>Every paired comparison, including greedy and uniform PUCT separately</summary>'
        '<table><tr><th>Arm</th><th>Control</th><th>Paired/planned</th><th>Nominal 95%, points</th>'
        '<th>Stratified 95%, points</th><th>Failure sensitivity, points</th></tr>' + ''.join(rows) + '</table></details>'
        '<details><summary>Direct depth, shuffle and beta contrasts</summary><table><tr><th>Contrast</th>'
        '<th>Nominal 95%, points</th><th>Stratified 95%, points</th></tr>' + ''.join(contrasts) + '</table></details>'
        '<h2>Measured cost</h2><p class="note">Hero search time only: decision and game means, medians and p95s. '
        'Policy-only controls perform no mirror search; their retained hero-action time is tabulated separately below. '
        'Supported actions count non-null Q entries, since rounded base probabilities lose support. '
        'Actual rollout commands, network batch calls and evaluated network rows were not captured. '
        'Moves × depth × worlds is an approximate unbatched work proxy, not a network-call count.</p>'
        '<details><summary>All arms: outcome and timing distributions</summary><table><tr><th>Arm</th><th>W/D/L</th>'
        '<th>Decision seconds: mean / median / p95</th><th>Game seconds: mean / median / p95</th>'
        '<th>Supported actions, mean</th><th>Terminal rollouts</th></tr>' + ''.join(costs) + '</table></details>'
        '<details><summary>Policy-only timing (historical host/load, no controlled speed ratio)</summary>'
        '<table><tr><th>Control</th><th>Seconds/decision, mean</th><th>Game seconds: mean / median / p95</th>'
        '<th>Total hero seconds</th></tr>' + policy_costs + '</table></details>'
        f'<p class="note">Retained mirror hero searches total {d["hero_seconds"]/3600:.4f} hours; noise searches '
        f'total {health["noise"]["seconds"]:.3f} seconds. These exclude setup, opponent decisions, interrupted '
        'attempts and overhead. Inclusive worker-hours, energy/storage costs and historical PUCT rental costs '
        'remain unavailable. The mirror campaign used the owned mini with $0 rental spend. No execution was '
        'added by analysis. The retained SDK rejects the current launcher schema. Later extension reconciliation '
        'used a separate compatible reader without changing the frozen environment; no remaining rental budget is inferred.</p>'
        '<h2>Health, learning and acceptance limits</h2>'
        f'<p class="note">Execution: {d["rows"]:,} decision records reconcile with all game counts and consecutive '
        'indices, with no orphans, duplicate decisions, malformed tails or nonfinite values. '
        f'Health remains qualified: worker logs retain {warnings["invalid_focus_object"]} invalid-focus warnings '
        f'and {warnings["event_list_truncated"]} event truncations to capacity 32. '
        f'{d["rounded_zero_chosen"]["base"]} chosen base probabilities round to zero. The unconstrained scalar '
        f'value head has maximum |Q| {d["max_abs_q"]:.4f}; {d["decisions_with_q_outside_unit_interval"]:,} '
        'decisions contain |Q| &gt; 1. These values are retained, not clipped or treated as probabilities.</p>'
        '<p class="note">Historical recovery preserved 89 completed games during a seven-worker restart. '
        'Interrupted attempts lack receipts and cannot be counted exactly. Deleted smoke evidence survives '
        'only as printed results/timing (12 mirror games, 228.62 seconds; two noise positions, 94.75 seconds), '
        'with preparation failures. Full Commands, root/RNG state and replay identities were not captured: '
        'schedule/decision reconciliation is not a replay proof. Observed disk hashes are not retrospective '
        'loaded-source admission. Native support and clean observation encoding remain unresolved.</p>'
        '<p class="note">Learning: one frozen raw MTG-183 checkpoint at update 5,950, training seed 18301; '
        'zero new training or weight updates, so no new training exports or callbacks are expected. This measures inference-time search, not learned improvement. '
        'Training card exposure is absent and remains unresolved independently of evaluation draws/actions, '
        'deck membership and schedule completeness. Fixed deck cells and rules permit paired intervention '
        'comparisons; absolute scores also reflect deck matchups. No deck-strength or rules change is credited '
        'as learned skill. Generalization to held-out cards, combinations, mechanics or independent training '
        'seeds is untested. Oracle gains do not establish playable uniform-hand strength or a mathematical '
        'upper bound. No confirmatory winner, controlled speedup or equal-wall-budget claim is supported.</p>'
        '<p class="note">Evidence review must retain these unresolved claims. The authorized beta-0.01 '
        'oracle/uniform extension has a separate frozen protocol and denominator; this analysis starts no '
        'new games or parameter changes.</p>'
    )


# ---------------------------------------------------------------------------
# Page


def build(mirror_dir: Path, baselines: Sequence[Path], noise_dir: Path, out: Path,
          baseline_summary: Path, extension: Path | None = None) -> dict[str, Any]:
    reconciliation = _reconciliation(mirror_dir)
    mirror_records = _outcomes([mirror_dir], mirror=True)
    baseline_records = _outcomes(baselines, mirror=False)
    mirror = _scores(mirror_records)
    base = _scores(baseline_records)
    games = [g for g in schedule() if g.index < DEALS_PER_CELL]
    planned_deals = {g.deal for g in games}
    costs = _costs(mirror_dir)
    puct_seconds = {k: v["diagnostics"]["seconds"]
                    for k, v in json.loads(baseline_summary.read_text())["conditions"].items()
                    if v.get("diagnostics")}
    noise_summary, noise_rows = analyze(noise_dir)

    summary: dict[str, Any] = {"arms": {}, "noise": {"summary": noise_summary, "rows": [asdict(r) for r in noise_rows]}}
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
            "full_depth_work_proxy": cost.work_proxy(arm) if cost else None,
            "actual_rollout_commands": None,
            "network_batch_calls": None,
            "evaluated_network_rows": None,
            "depth": arm.depth, "worlds": arm.worlds,
            "cost": asdict(cost) if cost else None,
            "game_hero_seconds": {
                "mean": statistics.fmean(r.hero_seconds for r in mirror_records[arm.key].values()),
                "median": float(np.median([r.hero_seconds for r in mirror_records[arm.key].values()])),
                "p95": float(np.quantile([r.hero_seconds for r in mirror_records[arm.key].values()], 0.95)),
                "total": sum(r.hero_seconds for r in mirror_records[arm.key].values()),
            },
            "coverage": _coverage(mirror_records[arm.key], games),
            "moves_per_decision": cost.moves if cost else None,
        }

    summary["reconciliation"] = reconciliation
    summary["controls"] = {k: _coverage(v, games) for k, v in baseline_records.items()}
    summary["policy_only_cost"] = {}
    for key, records in baseline_records.items():
        if key.rsplit("/", 1)[-1] not in ("policy", "greedy"):
            continue
        rows = [r for d, r in records.items() if d in planned_deals]
        seconds = sum(r.hero_seconds for r in rows)
        decisions = sum(r.hero_decisions for r in rows)
        summary["policy_only_cost"][key] = {
            "hero_seconds": seconds, "hero_decisions": decisions,
            "seconds_per_decision": seconds / decisions,
            "seconds_per_game": seconds / len(rows),
            "game_seconds_median": float(np.median([r.hero_seconds for r in rows])),
            "game_seconds_p95": float(np.quantile([r.hero_seconds for r in rows], 0.95)),
            "hosts": sorted({r.host for r in rows}),
            "controlled_comparison": False,
        }
    # Bind the external baseline snapshot and analysis source without changing evidence.
    summary["analysis_inputs"] = {
        "baseline_games": {f"{i}/{p.relative_to(directory)}": hashlib.sha256(p.read_bytes()).hexdigest()
                           for i, directory in enumerate(baselines)
                           for p in sorted(directory.rglob("games-*.jsonl"))},
        "baseline_summary_sha256": hashlib.sha256(baseline_summary.read_bytes()).hexdigest(),
        "source": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "experiments/search_mirror_report.py", "experiments/search_mirror_noise.py")},
    }
    summary["planned_games"] = len(ARMS) * len(games)
    summary["uncertainty"] = {
        "normal": "nominal 95% descriptive, paired whole-deal score differences",
        "bootstrap": "exploratory 95% percentile; jointly resample deals within each fixed cell",
        "resamples": 10000, "seed": 26901, "selection_adjusted": False,
        "common_complete_by_opponent_control": {},
    }
    summary["training"] = {
        "seed": 18301, "raw_checkpoint_update": 5950,
        "checkpoint_sha256": "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4",
        "new_training_updates": 0, "card_exposure": None,
        "new_training_exports_expected": 0, "training_callbacks": "not applicable: inference-only campaign",
        "exposure_status": "not captured; evaluation actions, noise draws and deck membership are not training exposure",
    }
    for opponent in OPPONENTS:
        arms = [a for a in ARMS if a.opponent == opponent]
        for label, control in (("sampled", "policy"), ("greedy", "greedy"),
                               ("puct_1024_oracle", "puct-1024-oracle"),
                               ("puct_1024_uniform", "puct-1024-uniform")):
            key = f"{opponent}/{control}"
            cohort = {a.key: mirror.get(a.key, {}) for a in arms} | {key: base.get(key, {})}
            intervals, common = _bootstrap(cohort, games, [(a.key, key) for a in arms])
            summary["uncertainty"]["common_complete_by_opponent_control"][key] = len(common)
            for arm in arms:
                estimate_common = paired(mirror.get(arm.key, {}), base.get(key, {}), common)
                boot = intervals.get((arm.key, key))
                summary["arms"][arm.key].setdefault("comparisons", {})[label] = {
                    "planned": len(games),
                    "valid_only": (g.to_json() if (g := paired(mirror.get(arm.key, {}), base.get(key, {}), planned_deals)) else None),
                    "common_complete": estimate_common.to_json() if estimate_common else None,
                    "cell_stratified_bootstrap": boot.to_json() if boot else None,
                    "failure_sensitivity": _sensitivity(mirror.get(arm.key, {}), base.get(key, {}), planned_deals),
                    "paired_coverage": _coverage({d: r for d, r in mirror_records[arm.key].items()
                        if d in base.get(key, {}) and d in mirror.get(arm.key, {})}, games),
                }
    # Within-sweep contrasts answer depth and shuffle questions directly.
    summary["setting_contrasts"] = {}
    for opponent in OPPONENTS:
        cohort = {a.key: mirror.get(a.key, {}) for a in ARMS if a.opponent == opponent}
        pairs = [(f"{opponent}/mirror-{a}", f"{opponent}/mirror-{b}") for a, b in (
            ("d40-b0.1", "d20-b0.1"), ("d20-b0.1", "d10-b0.1"),
            ("d20-b0.1-w4", "d20-b0.1"), ("d20-b0.03-w4", "d20-b0.03"),
            ("d20-b0.03-w4", "d20-b0.1-w4"))]
        intervals, _ = _bootstrap(cohort, games, pairs)
        summary["setting_contrasts"][opponent] = [{
            "a": a, "b": b, "normal": paired(cohort[a], cohort[b]).to_json(),
            "bootstrap": intervals[a, b].to_json() if (a, b) in intervals else None,
        } for a, b in pairs]

    settings = sorted({(a.depth, a.beta, a.worlds) for a in ARMS}, key=lambda s: (s[2], -s[1], s[0]))
    by_setting = {s: {a.opponent: a for a in ARMS if (a.depth, a.beta, a.worlds) == s} for s in settings}

    def mean_gain(setting: tuple[int, float, int]) -> float:
        found = [gains[a.key, "sampled"] for a in by_setting[setting].values()]
        return statistics.fmean(g.mean for g in found if g) if all(found) else -1.0

    best = max(settings, key=mean_gain)
    best_arms = by_setting[best]
    summary["interpretation"] = {
        "execution": "complete retained mirror schedule; PUCT policy controls lack 21/500 outcomes",
        "health": "finite numerical evidence; invalid-focus warnings and event truncation retained",
        "learning": "inference-time intervention on one frozen trained policy; no new learning",
        "training_exposure": "absent, unresolved",
        "generalization": "untested; familiar decks, one training seed, oracle hands",
        "native_support": "loaded-source admission and replay proof unavailable",
        "selection": "exploratory selected maximum mean gain over sampled policy across both opponents",
        "negative_findings": "depth 40 vs 20 and four vs one shuffle at beta 0.1 span zero for both opponents",
        "method_sensitive": "selected arm vs heuristic PUCT-1024, and four vs one shuffle at beta 0.03 against policy: nominal spans zero, bootstrap does not",
    }
    played = sum(len(v) for k, v in mirror.items())
    planned = len(ARMS) * 500
    deals = {o: set().union(*(mirror.get(a.key, {}).keys() for a in ARMS if a.opponent == o)) for o in OPPONENTS}

    tiles = [f'<div class="tile"><div class="k">Games played</div><div class="v">{played:,}</div>'
             f'<div class="s">of {planned:,} planned · reconciled retained execution</div></div>']
    for o in OPPONENTS:
        g, p = gains[best_arms[o].key, "sampled"], gains[best_arms[o].key, "puct"]
        tiles.append(f'<div class="tile"><div class="k">Best setting {OPPONENT_TITLE[o]}</div>'
                     f'<div class="v">{g.mean * 100:+.1f} pts</div>'
                     f'<div class="s">over the sampled policy [{g.low * 100:+.1f}, {g.high * 100:+.1f}] · '
                     f'{p.mean * 100:+.1f} vs PUCT-1024</div></div>')
    best_cost = costs[best_arms["policy"].key]
    tiles.append(f'<div class="tile"><div class="k">Best setting cost</div>'
                 f'<div class="v">{best_cost.seconds:.1f} s</div>'
                 f'<div class="s">per decision · about {best_cost.work_proxy(best_arms["policy"]):.0f} full-depth work units '
                 f'(moves × depth × worlds)</div></div>')

    sections: list[str] = []
    sections.append(
        '<h2>What the completed sweep supports</h2><p class="note">'
        f'The selected setting is {escape(setting_label(best_arms["policy"]))}, chosen by mean sampled-policy '
        'gain across both opponents. It wins 438/500 games against heuristic and 328/500 against policy. '
        'Sampled-policy gains are +9.4 points (nominal 95% [5.8, 13.0], stratified [6.2, 12.8]) and '
        '+14.7 points ([9.8, 19.6], stratified [10.4, 19.1]), respectively. These are exploratory oracle-hand '
        'results on one trained policy, not confirmation or playable strength.</p>'
        '<p class="note">Depth 40 versus 20 adds −1.4 points against heuristic and −3.2 against policy; '
        'both interval methods span zero, despite higher cost. Four versus one shuffle at β 0.1 also spans '
        'zero against both opponents. At β 0.03 the four-shuffle gain is +6.8 against heuristic and +4.0 '
        'against policy; the latter spans zero nominally but not under the stratified bootstrap. '
        'These method-sensitive comparisons remain inconclusive under the protocol.</p>'
        '<p class="note">Against common opponents, the selected setting exceeds oracle PUCT-1024 by '
        '+3.2 points against heuristic (500 deals; nominal interval includes zero, stratified does not) '
        'and +7.93 against policy (479/500 paired deals; full-denominator missing-outcome range +6.4 to +10.6). '
        'This is not head-to-head evidence. The 300-position noise screen favors averaging over a single '
        'rollout, but β 0.01 at one or two rollouts has a whole-deal interval spanning zero; only 21 source '
        'deals limit precision. No game-strength claim for β 0.01, 8 or 16 shuffles follows from that screen.</p>'
    )
    gain_cards = []
    for o in OPPONENTS:
        rows = [(setting_label(by_setting[s][o]), gains[by_setting[s][o].key, "sampled"]) for s in settings]
        gain_cards.append(f'<div class="card"><h3>Mirror over the sampled policy, {OPPONENT_TITLE[o]}</h3>'
                          f'<div class="sub">Paired by deal, percentage points, nominal 95% intervals</div>'
                          f'{gain_chart(rows, "gain")}</div>')
    sections.append(
        "<h2>What mirror search adds</h2>"
        '<p class="note">Mirror search samples its move from the updated policy, so the matching baseline is '
        "the policy's own sampled play. β is the pull back toward the policy: large β barely moves it, "
        "small β trusts the rollout values. Worlds share their root shuffle across actions. A large β is a "
        "near-policy reference, not a guarantee of identical trajectories or a test that pairing is fair.</p>"
        '<div class="row pairs">' + "".join(gain_cards) + "</div>")

    depth_gains = {o: [(a.depth, g) for a in sorted(ARMS, key=lambda a: a.depth)
                       if a.opponent == o and a.beta == 0.1 and a.worlds == 1
                       and (g := gains[a.key, "sampled"]) is not None] for o in OPPONENTS}
    sections.append(
        "<h2>How deep to roll out</h2>"
        '<p class="note">At β 0.1 with one shuffle. Depth counts canonical decisions including the root, '
        "not a fixed number of turns. Direct paired depth contrasts below distinguish observed differences from uncertainty.</p>"
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
                           f"decision, about {c.work_proxy(arm):.0f} full-depth work units", short_label(arm)))
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
        "rented hosts under different load; these timings cannot establish a speedup or equal-compute strength. Neither searcher played the other; each "
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
        f'±{ns["single_rollout_sd"]:.2f} while between-action estimated mean-Q SD is about ±{ns["between_move_sd_of_q"]:.2f}, '
        "The finite remainder is a noisy reference, not ground truth; optimizing on that reference is optimistic. "
        "Intervals resample 21 whole source deals (both opponents together), not 6,000 independent draws. "
        "These one-decision values are not game-win percentages or evidence of training exposure.</p>"
        + _legend([("--series-1", "β 0.01"), ("--series-2", "β 0.03"), ("--series-3", "β 0.1")])
        + f'<div class="row single"><div class="card"><h3>Expected gain at one decision</h3>'
          f'<div class="sub">Over policy; whole-deal bootstrap 95% intervals</div>{noise_chart(noise_rows)}</div></div>'
        + "<details><summary>Every step size</summary><table><tr><th>β</th>"
        + "".join(f"<th>{n} rollouts</th>" for n in SUBSAMPLES) + "</tr>"
        + "".join(
            f"<tr><td>{beta:g}</td>"
            + "".join(f"<td>{r.gain_over_policy:.4f} [{r.gain_low:.4f}, {r.gain_high:.4f}]</td>" for r in sorted(
                (r for r in noise_rows if r.beta == beta), key=lambda r: r.rollouts))
            + "</tr>"
            for beta in BETAS)
        + "</table></details>")

    header = ("<tr><th>Setting</th>"
              + "".join(f"<th>vs sampled ({o})</th><th>vs PUCT-1024 ({o})</th>" for o in OPPONENTS)
              + "<th>s / decision</th><th>full-depth work units</th><th>deals</th></tr>")
    body_rows = []
    for s in settings:
        arms = by_setting[s]
        c = costs.get(arms["policy"].key)
        cls = ' class="best"' if s == best else ""
        body_rows.append(
            f"<tr><td{cls}>{escape(setting_label(arms['policy']))}</td>"
            + "".join(f"<td>{_cell(gains[arms[o].key, 'sampled'])}</td><td>{_cell(gains[arms[o].key, 'puct'])}</td>"
                      for o in OPPONENTS)
            + (f"<td>{c.seconds:.2f}</td><td>{c.work_proxy(arms['policy']):.0f}</td>" if c else "<td>–</td><td>–</td>")
            + f"<td>{min(len(mirror.get(a.key, {})) for a in arms.values())}</td></tr>")
    sections.append("<h2>Every setting</h2>"
                    '<p class="note">Paired gains in points with 95% half-widths. Work units approximate every '
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
        "Games ran on one Mac mini, seven workers on one thread each. Oracle hands supply an information "
        "advantage benchmark, not a proven mathematical upper bound on this approximate search with hidden hands.</p>")

    sections.append(_evidence_sections(summary))
    extension_html = ""
    if extension is not None:
        summary["extension"] = _extension_report(extension, baseline_records)
        extension_html = _extension_section(summary["extension"])

    lede = ("Can the magnetic-mirror local update, given the opponent's true hand, add strength on top of "
            f"MTG-183's policy, and what does it cost next to PUCT? {played:,} of {planned:,} retained games "
            f"(about {min(len(v) for v in mirror.values())} deals per setting).")
    banner = ('<div class="banner">Original 8,000-game execution is complete. Eight settings against two opponents were inspected repeatedly. With no '
              "multiple-comparison correction, a selected setting clearing its interval remains exploratory; playable strength is unresolved.</div>")
    html = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Mirror search study</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
            f"<h1>Mirror search on MTG-183</h1><p class=\"lede\">{escape(lede)}</p>{banner}"
            f'<div class="tiles">{"".join(tiles)}</div>' + extension_html + "".join(sections) + "</main></body></html>")
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
    parser.add_argument("--extension", type=Path, help="V2 snapshot root, containing games/ and reconciliation/")
    args = parser.parse_args()
    summary = build(args.mirror, args.baseline, args.noise, args.out, args.baseline_summary, args.extension)
    print(json.dumps({"games": summary["games"], "best": summary["best_setting"]}))


if __name__ == "__main__":
    main()
