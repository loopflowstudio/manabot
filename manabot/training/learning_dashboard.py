"""The learning story projected from retained runs and paired monitoring games.

CheckpointPanel keeps opponent/protocol/world identities separate. Deck summaries
route through the candidate's actual seat, then resample whole deals so the two
seat legs stay paired. These are conditional monitoring intervals, not seed or
post-selection significance. Rendering owns no training or evaluation state.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import StringIO
import json
from typing import TYPE_CHECKING, Literal

from matplotlib.figure import Figure
import numpy as np

from manabot.arena.models import canonical_sha256
from manabot.training.monitor_evaluation import ArenaRow, MonitorResult, RateInterval

if TYPE_CHECKING:
    from manabot.training.experiment_report import ReportEvidence


@dataclass(frozen=True)
class DeckResult:
    deck: str
    wins: int
    draws: int
    games: int
    interval: RateInterval


@dataclass(frozen=True)
class CheckpointPanel:
    identity: str
    run_id: str
    results: tuple[MonitorResult, ...]


def candidate_deck(row: ArenaRow) -> str | None:
    """Old rows without seat metadata remain unavailable, never guessed from leg."""
    extra = row.model_extra or {}
    seat = extra.get("player_a_seat")
    decks = extra.get("seat_decks")
    if type(seat) is not int or seat not in (0, 1):
        return None
    if not isinstance(decks, list) or len(decks) != 2:
        return None
    return decks[seat] if isinstance(decks[seat], str) else None


def deck_results(result: MonitorResult) -> tuple[DeckResult, ...]:
    """Require the complete frozen four-leg cohort before computing any deck rate."""
    expected = {(s, leg) for s in result.protocol.deal_seeds for leg in range(4)}
    if result.status != "completed":
        return ()
    if (
        len(result.rows) != len(expected)
        or {(r.deal_seed, r.leg) for r in result.rows} != expected
    ):
        raise ValueError(
            "completed monitoring cohort has missing or duplicate deal legs"
        )
    for row in result.rows:
        if (
            not row.valid
            or row.arena_key != result.key
            or row.player_a_registration_sha256 != result.candidate.identity_sha256
            or row.player_b_registration_sha256 != result.opponent.identity_sha256
        ):
            raise ValueError("monitoring row differs from completed frozen cohort")
    decks = [candidate_deck(r) for r in result.rows]
    if None in decks:
        return ()
    rng = np.random.default_rng(result.protocol.bootstrap_seed)
    indices = rng.integers(
        0,
        len(result.protocol.deal_seeds),
        (result.protocol.bootstrap_replicates, len(result.protocol.deal_seeds)),
    )
    summaries: list[DeckResult] = []
    for deck in sorted(set(d for d in decks if d is not None)):
        selected = [r for r in result.rows if candidate_deck(r) == deck]
        blocks: list[float] = []
        for seed in result.protocol.deal_seeds:
            pair = [r for r in selected if r.deal_seed == seed]
            if len(pair) != 2 or {r.model_extra.get("player_a_seat") for r in pair} != {
                0,
                1,
            }:
                raise ValueError(
                    "deck cohort must contain both candidate seats per deal"
                )
            blocks.append(sum(r.score_a == 1 for r in pair) / 2)
        draws = np.asarray(blocks)[indices].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        summaries.append(
            DeckResult(
                deck,
                sum(r.score_a == 1 for r in selected),
                sum(r.score_a == 0.5 for r in selected),
                len(selected),
                RateInterval(
                    mean=float(np.mean(blocks)), lower=float(lo), upper=float(hi)
                ),
            )
        )
    if result.win is not None:
        observed = sum(r.score_a == 1 for r in result.rows) / len(result.rows)
        if abs(observed - result.win.mean) > 1e-10:
            raise ValueError("saved win mean differs from terminal rows")
    return tuple(summaries)


def checkpoint_panels(evidence: "ReportEvidence") -> tuple[CheckpointPanel, ...]:
    groups: dict[str, list[MonitorResult]] = {}
    for result in evidence.monitors:
        identity = canonical_sha256(
            {
                "run": result.run_id,
                "candidate_spec": result.candidate.player_spec,
                "observation_abi": result.candidate.observation_abi_sha256,
                "action_abi": result.candidate.action_abi_sha256,
                "protocol": result.protocol.model_dump(mode="json"),
                "opponent": result.opponent.model_dump(mode="json"),
                "arena": result.key.model_dump(mode="json"),
                "purpose": result.purpose,
            }
        )
        groups.setdefault(identity, []).append(result)
    return tuple(
        CheckpointPanel(
            key,
            values[0].run_id,
            tuple(
                sorted(
                    values,
                    key=lambda r: (
                        r.coordinates.updates,
                        r.coordinates.training_seconds,
                    ),
                )
            ),
        )
        for key, values in sorted(
            groups.items(),
            key=lambda item: (
                item[1][0].protocol.opponent != "scripted_greedy",
                item[1][0].training_seed,
                item[0],
            ),
        )
    )


def _deck_name(deck: str) -> str:
    return {"gw_allies": "Allies", "ur_lessons": "Lessons"}.get(deck, deck)


def _number(value: object, unit: str = "") -> str:
    return f"{value:,.1f}{unit}" if isinstance(value, (float, int)) else "unavailable"


def _hours(value: object) -> str:
    return (
        _number(value / 3600, " h")
        if isinstance(value, (float, int))
        else "unavailable"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(
                f'<td data-label="{escape(h, quote=True)}">{escape(c)}</td>'
                for h, c in zip(headers, row, strict=True)
            )
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _svg(figure: Figure) -> str:
    stream = StringIO()
    figure.savefig(stream, format="svg", metadata={"Date": None})
    return stream.getvalue()[stream.getvalue().index("<svg") :]


def panel_figures(
    panel: CheckpointPanel, axis: Literal["updates", "training_seconds"] = "updates"
) -> list[Figure]:
    """One small chart per deck, one seed per panel; no absent-seed averages."""
    measured = [(r, deck_results(r)) for r in panel.results]
    decks = sorted({d.deck for _, values in measured for d in values})
    figures: list[Figure] = []
    for deck in decks:
        points = [(r, d) for r, values in measured for d in values if d.deck == deck]
        fig = Figure(figsize=(5.1, 2.7), layout="constrained", facecolor="#fffdf8")
        ax = fig.subplots(subplot_kw={"facecolor": "#fffdf8"})
        x = [
            getattr(r.coordinates, axis) / (3600 if axis == "training_seconds" else 1)
            for r, _ in points
        ]
        y = [d.interval.mean * 100 for _, d in points]
        lo = [d.interval.lower * 100 for _, d in points]
        hi = [d.interval.upper * 100 for _, d in points]
        color = "#176b57" if deck == "gw_allies" else "#ad562b"
        ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        ax.errorbar(
            x,
            y,
            yerr=[np.maximum(0, np.array(y) - lo), np.maximum(0, np.array(hi) - y)],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2,
        )
        ax.set(
            title=f"Model playing {_deck_name(deck)}",
            xlabel="Updates" if axis == "updates" else "Recorded training time (h)",
            ylabel="Wins / games (%)",
            ylim=(-3, 103),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_yticks([0, 25, 50, 75, 100])
        if len(points) == 1:
            ax.annotate(
                "One checkpoint; trend unavailable",
                (0.04, 0.92),
                xycoords="axes fraction",
                fontsize=8,
            )
        figures.append(fig)
    return figures


def learning_story(evidence: "ReportEvidence", docs: str) -> str:
    """Build a compact overview followed by inspectable per-seed checkpoint panels."""
    parts = [
        '<div class="eyebrow">MANABOT / LEARNING OBSERVATORY</div><h1>Is training improving play?</h1>'
    ]
    snapshot = evidence.root / "snapshot.json"
    captured = (
        json.loads(snapshot.read_text()).get("completed_utc")
        if snapshot.exists()
        else None
    )
    as_of = [
        e.last_seen_unix for e in evidence.executions if e.last_seen_unix is not None
    ]
    heartbeat = (
        datetime.fromtimestamp(max(as_of), timezone.utc).isoformat(timespec="seconds")
        if as_of
        else "unavailable"
    )
    parts.append(
        f'<p class="stamp">Saved snapshot · captured {escape(str(captured or "unavailable"))}<br>Coordinator heartbeat {escape(heartbeat)} · no live refresh</p>'
    )
    parts.append(
        '<nav><a href="#strength">Strength curves</a><a href="#checkpoints">Checkpoint table</a><a href="#details">Diagnostics &amp; evidence</a></nav>'
    )
    attempts = [a for e in evidence.executions for a in e.attempts]
    planned = {(a.case, a.seed) for a in attempts}
    done = {(a.case, a.seed) for a in attempts if a.status == "completed"}
    count = (
        f"{len(done)} / {len(planned)}"
        if planned
        else f"{sum(r.status == 'completed' for r in evidence.runs)} / unknown"
    )
    elapsed = (
        sum(e.elapsed_seconds for e in evidence.executions)
        if evidence.executions
        else None
    )
    dollars = (
        sum(e.host_dollars for e in evidence.executions if e.host_dollars is not None)
        if evidence.executions
        and all(e.host_dollars is not None for e in evidence.executions)
        else None
    )
    parts.append(
        '<div class="stats">'
        + "".join(
            f"<div><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
            for label, value in [
                ("Arm × seed runs complete", count),
                ("Recorded elapsed wall", _hours(elapsed)),
                (
                    "Recorded host cost",
                    _number(dollars, " USD") if dollars is not None else "Unknown",
                ),
            ]
        )
        + "</div>"
    )
    rows: list[list[str]] = []
    losses: list[str] = []
    for run in evidence.runs:
        metrics = evidence.metrics(run)
        latest = metrics[-1] if metrics else {}
        updates = latest.get("progress/updates")
        target = sum(
            getattr(s, "updates", getattr(s, "epochs", 0)) for s in run.regime.stages
        )
        target_label = f"{target:,}" if target else "unavailable"
        if any(s.operation == "train_supervised" for s in run.regime.stages):
            target_label += " (includes epochs)"
        active_target = sum(
            getattr(s, "active_seconds", None) or 0 for s in run.regime.stages
        )
        if active_target:
            target_label += f" ceiling; {_hours(active_target)} active target"
        active = latest.get("progress/active_training_seconds")
        # Older receipts have no active clock. Collection + learning counters are
        # only usable when recorded; zeros in a running stage can be placeholders.
        recorded = latest.get("progress/training_seconds")
        monitors = [
            r
            for r in evidence.monitors
            if r.run_id == run.id and r.status == "completed"
        ]
        last = (
            max(monitors, key=lambda r: r.coordinates.training_seconds)
            if monitors
            else None
        )
        lag = (
            f"{int(updates) - last.coordinates.updates:,} updates"
            if last and isinstance(updates, (int, float))
            else "unavailable"
        )
        loss_key = next(
            (k for k in ("rl/loss", "distillation/train_cross_entropy") if k in latest),
            None,
        )
        if loss_key:
            semantics = (
                "RL minibatch objective (not log loss)"
                if loss_key == "rl/loss"
                else "teacher cross-entropy (nats)"
            )
            losses.append(
                f"{evidence.label(run.id)}: {semantics} {_number(latest[loss_key])}"
            )
        else:
            losses.append(f"{evidence.label(run.id)}: latest loss unavailable")
        rows.append(
            [
                evidence.label(run.id),
                f"{int(updates):,} / {target_label}"
                if isinstance(updates, (int, float))
                else f"unavailable / {target_label}",
                run.status,
                _hours(active),
                _hours(recorded),
                _number(latest.get("throughput/learner_transitions_per_second"), " /s"),
                f"{last.coordinates.updates:,} · lag {lag}"
                if last
                else "pending / unavailable",
            ]
        )
    parts.append(
        '<div class="progress">'
        + _table(
            [
                "Arm / seed",
                "Updates / plan",
                "Saved state",
                "Active training",
                "Recorded training",
                "Learner transitions",
                "Last evaluated update",
            ],
            rows,
        )
        + "</div>"
    )
    if attempts:
        pending = [
            f"{a.case} / seed {a.seed}: {a.status}"
            for a in attempts
            if a.run_id not in {r.id for r in evidence.runs}
            and (a.case, a.seed) not in {(r.regime.id, r.seed) for r in evidence.runs}
        ]
        if pending:
            parts.append(
                '<p class="muted">Not yet measured: '
                + escape("; ".join(pending))
                + "</p>"
            )
    parts.append(
        f'<p class="muted"><a href="{escape(docs)}#learning">Updates, clocks &amp; throughput</a> · Active = collection + optimization when recorded; unavailable is not zero. Recorded training can include overhead. Host cost is a receipt, not an invoice; external charges are unknown.</p>'
    )
    parts.append(
        f'<p class="muted"><a href="{escape(docs)}#learning">Latest loss</a> · {escape("; ".join(losses))}</p>'
    )
    panels = checkpoint_panels(evidence)
    parts.append('<div class="latest-grid">')
    for panel in panels:
        completed = [r for r in panel.results if r.status == "completed"]
        if not completed:
            continue
        last = completed[-1]
        parts.append(
            f'<div class="latest"><b>{escape(evidence.label(panel.run_id))} · vs {escape(last.opponent.display_name)}</b><div>'
        )
        for d in deck_results(last):
            parts.append(
                f"<span>{escape(_deck_name(d.deck))} <strong>{d.interval.mean:.0%}</strong> <small>{d.wins}/{d.games} wins<br>95%: {d.interval.lower:.0%}–{d.interval.upper:.0%}</small></span>"
            )
        finished = getattr(last, "finished_unix", None)
        observed = (
            datetime.fromtimestamp(finished, timezone.utc).isoformat(timespec="seconds")
            if finished
            else "time unavailable"
        )
        parts.append(
            f"</div><small>Update {last.coordinates.updates:,} · evaluated {escape(observed)}</small></div>"
        )
    parts.append("</div>")
    parts.append(
        '<section id="strength"><div class="section-label">01 / PLAYING STRENGTH</div><h2>Separate the decks. Keep the opponent fixed.</h2>'
    )
    parts.append(
        f'<p class="muted">Each panel is one arm and training seed. Bands: 95% whole-deal bootstrap, retaining both seats. <a href="{escape(docs)}#evaluation">What uncertainty means</a>. Monitoring is development evidence, not held-out final scoring.</p>'
    )
    if not panels:
        parts.append(
            "<p>No monitoring evaluations retained. Learning direction is unavailable.</p>"
        )
    checkpoint_rows: list[list[str]] = []
    table_decks = sorted(
        {d.deck for panel in panels for r in panel.results for d in deck_results(r)}
    )
    for index, panel in enumerate(panels):
        result = panel.results[-1]
        complete = [r for r in panel.results if r.status == "completed"]
        heading = f"{evidence.label(panel.run_id)} · vs {result.opponent.display_name}"
        purpose = (
            "Held-out frozen study"
            if result.purpose == "frozen-study-evaluation"
            else "Development monitoring"
        )
        parts.append(
            f'<details class="panel"{" open" if index == 0 else ""}><summary>{escape(heading)} · {len(complete)} checkpoints</summary><p class="muted">{purpose} · opponent {result.opponent.identity_sha256[:12]} · panel {panel.identity[:12]}</p>'
        )
        figures = panel_figures(panel)
        parts.append(
            '<div class="charts">' + "".join(_svg(f) for f in figures) + "</div>"
            if figures
            else "<p>Deck curves unavailable: incomplete cohort or missing candidate-seat metadata.</p>"
        )
        if complete:
            first, last = complete[0], complete[-1]
            initial = {d.deck: d for d in deck_results(first)}
            latest = deck_results(last)
            observations = []
            for d in latest:
                baseline = initial.get(d.deck)
                delta = (
                    100 * (d.interval.mean - baseline.interval.mean)
                    if baseline
                    else None
                )
                change = (
                    f"{delta:+.1f} points since update {first.coordinates.updates:,}"
                    if first != last and delta is not None
                    else "baseline or trend unavailable"
                )
                values = [
                    v.interval.mean
                    for r in complete
                    for v in deck_results(r)
                    if v.deck == d.deck
                ]
                observations.append(
                    f"{_deck_name(d.deck)} {d.wins}/{d.games} wins ({d.interval.mean:.0%}, 95% interval {d.interval.lower:.0%}–{d.interval.upper:.0%}); {change}; observed range {min(values):.0%}–{max(values):.0%}."
                )
            parts.append(
                '<div class="reading"><b>Reading this evidence</b><p>'
                + escape(" ".join(observations))
                + "</p><p>"
                + (
                    "Initialization retained."
                    if first.coordinates.updates == 0
                    else f"Initialization unavailable; first measured update is {first.coordinates.updates:,}."
                )
                + " These are descriptive changes from repeatedly inspected deals, not a significance test or a multi-seed method result.</p></div>"
            )
        parts.append(
            '<details><summary>View against recorded training hours</summary><div class="charts">'
            + "".join(_svg(f) for f in panel_figures(panel, "training_seconds"))
            + "</div></details></details>"
        )
        for r in reversed(panel.results):
            values = {d.deck: d for d in deck_results(r)}
            row = [
                evidence.label(r.run_id),
                r.opponent.display_name,
                f"{r.coordinates.updates:,}",
                _hours(r.coordinates.training_seconds),
            ]
            for deck in table_decks:
                d = values.get(deck)
                row.append(
                    f"{d.wins}/{d.games} · {d.interval.mean:.0%} [{d.interval.lower:.0%}–{d.interval.upper:.0%}]"
                    + (f" · {d.draws} draws" if d.draws else "")
                    if d
                    else f"unavailable · {r.status}; {sum(row.valid for row in r.rows)}/{r.expected_games} valid games"
                )
            checkpoint_rows.append(row)
    parts.append(
        f'<p class="muted"><a href="{escape(docs)}#opponent-asymmetry">Opponent asymmetry hypothesis</a>: greedy may pilot Allies creature/attack choices better than Lessons targeting. A model playing Lessons faces Allies. Scores alone do not establish this strategic cause.</p></section>'
    )
    parts.append(
        '<section id="checkpoints"><div class="section-label">02 / CHECKPOINTS</div><h2>The scores behind the curves</h2><p class="muted">Wins / games · win rate [95% paired-deal interval]. Draws are not wins. Each row keeps its original checkpoint and opponent.</p>'
        + _table(
            [
                "Arm / seed",
                "Opponent",
                "Update",
                "Training time",
                *[f"Model playing {_deck_name(deck)}" for deck in table_decks],
            ],
            checkpoint_rows,
        )
        + "</section>"
    )
    return "".join(parts)


DASHBOARD_CSS = """
body{font:15px/1.5 'Avenir Next',system-ui,sans-serif;color:#23342d;background:#fffdf8;max-width:1120px;margin:40px auto;padding:0 28px}h1{font:46px/1.12 Georgia,serif;letter-spacing:-1px;margin:12px 0}h2{font:28px/1.2 Georgia,serif;margin:12px 0}a{color:#176b57;text-underline-offset:3px}.eyebrow,.section-label{font-size:11px;letter-spacing:2px;font-weight:700;color:#66776d}.stamp,.muted{font-size:12px;color:#627067}nav{display:flex;gap:24px;margin:22px 0;font-weight:600}.stats{display:grid;grid-template-columns:repeat(3,1fr);border-top:2px solid #284b3b;border-bottom:1px solid #bbc4b8;margin:22px 0}.stats>div{padding:16px 12px}.stats span{display:block;font-size:12px;color:#627067}.stats strong{font:28px Georgia,serif}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th{text-align:left;color:#627067;font-weight:600}td,th{padding:10px 8px;border-bottom:1px solid #e0e3d9;vertical-align:top}section{margin-top:36px}summary{cursor:pointer;font-weight:600;padding:12px 0}details.panel{border-top:1px solid #bbc4b8}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.charts svg{width:100%;height:auto}.reading{background:#eff2e8;border-left:3px solid #718764;padding:12px 18px;font-size:13px;margin:12px 0}.reading p{margin:6px 0}.figure svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}#details{margin-top:36px;border-top:2px solid #bbc4b8}#details h1{font-size:26px}@media(max-width:650px){body{margin:24px auto;padding:0 16px}h1{font-size:34px}h2{font-size:24px}nav{gap:14px;font-size:12px}.stats strong{font-size:22px}.stats>div{padding:12px 6px}.charts{grid-template-columns:1fr}.table{max-width:100%}td,th{min-width:78px}summary{font-size:13px}}@media print{details{display:block}section{break-inside:avoid}}
"""

DASHBOARD_CSS += """
.latest-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:20px 0}.latest{border:1px solid #d7dfce;padding:12px 16px;font-size:12px}.latest>div{display:flex;gap:28px;margin:8px 0}.latest strong{font:25px Georgia,serif}.latest small{color:#627067}.progress td{white-space:nowrap}@media(max-width:650px){.latest-grid{grid-template-columns:1fr}.progress thead{display:none}.progress tr{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #bbc4b8}.progress td{display:block;white-space:normal;border:0;padding:5px 8px}.progress td:before{content:attr(data-label);display:block;font-size:11px;color:#627067}.progress td:first-child{grid-column:1/-1}}
"""
