"""Build the experiment-report PROTOTYPE: page, scripts and data in one folder.

This is a design prototype for MTG-180, not the production report. It renders
three experiments through one data model, laid out by `definitions.py`:

- `seat-comparison`: REAL. The from-scratch study (commit 0ecd955), read from
  the artifact store at `experiments/data/seat-comparison-v1/` (ledger and
  native arena rows; fetch with `manabot.data fetch`). Two jobs x four seeds,
  scored at learner-decision milestones, so seeds align on the milestone.
  Only total training seconds were retained, not the collection/learning split.
- `two-seat`: REAL but optional. MTG-165's retained October 22 projection under
  `.runs/report-defaults/` and `.runs/two-seat-report-prototype/`, which exist
  only in the MTG-180 working copy; skipped when absent. Its one-seat job is
  the earlier v1 ladder, so it is not a matched comparison.
- `elemental`: SYNTHETIC. Roster from `experiments/protocols/elemental-174.md`
  and the MTG-181 cohort design (model deck x heuristic deck, 50 games per cell).
  Its timing tree uses the real `collection_timings`/`learning_timings` names
  with invented values to preview default instrumentation.

A matchup is (model deck, opponent deck): two decks give a 2x2 grid, ten a
10x10 one. Runs are grouped into jobs by declared name; the page averages seeds.

Each experiment carries `timing`: one time tree per run and resource, as deep
as recorded timers go. Node status is `recorded` (a timer), `derived` (parent
minus recorded children), `uninstrumented` (known work with no timer; drawn at
the parent's width, hatched) or `hypothetical` (synthetic sub-timers showing
what instrumentation would add).

Each experiment also carries `training` (per-run metric series under the
training code's own keys, plus a phase time split) and `context` (status,
freshness, device). Two-seat series are what the Mini snapshot retained; keys
the trainer logs but the snapshot dropped are absent, never synthesized.

    uv run --extra artifacts python -m manabot.data fetch experiments/data/seat-comparison-v1
    uv run python -m experiments.report_prototype.build --out .runs/report-prototype

then open `.runs/report-prototype/report.html` (or serve the folder). The folder
gets `report.html`, `model.js`, `charts.js` and `data.js` (`window.REPORT`). Cell rows are
[run, eval index, anchor, deck, opponent, seat, wins, draws, losses] where
seat is 0 play, 1 draw, -1 recorded without a seat split.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
from typing import TypedDict

from experiments.report_prototype.definitions import REPORTS

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
RETAINED = REPOSITORY / ".runs/report-defaults/data.json"
SNAPSHOT = REPOSITORY / ".runs/two-seat-report-prototype/mini-snapshot.json"
ASSETS = ("report.html", "model.js", "charts.js")


class Eval(TypedDict, total=False):
    updates: int
    align: float  # what seeds of a job align on (defaults to updates)
    decisions: float | None
    hours: float | None
    usd: float | None
    sha: str | None


class Run(TypedDict, total=False):
    id: str
    label: str
    job_name: str
    seed: int
    evals: list[Eval]


class Experiment(TypedDict):
    id: str
    title: str
    question: str
    synthetic: bool
    decks: list[str]
    deck_names: list[str]
    anchors: list[str]
    runs: list[Run]
    cells: list[list[int]]
    direct: list[dict[str, object]]
    training: dict[str, object]
    context: dict[str, object]
    timing: list[dict[str, object]]


# ---------------------------------------------------------------- two-seat (real)


class TimeNode(TypedDict, total=False):
    name: str
    seconds: float
    category: str
    status: str
    source: str
    note: str
    children: list["TimeNode"]


def _node(
    name: str,
    seconds: float,
    category: str,
    source: str,
    status: str = "recorded",
    children: list[TimeNode] | None = None,
    note: str | None = None,
) -> TimeNode:
    node: TimeNode = {
        "name": name,
        "seconds": seconds,
        "category": category,
        "status": status,
        "source": source,
    }
    if children:
        node["children"] = children
    if note:
        node["note"] = note
    return node


def _with_remainder(node: TimeNode, name: str, source: str) -> TimeNode:
    """Append the parent's unaccounted seconds as a derived child."""
    rest = node["seconds"] - sum(c["seconds"] for c in node.get("children", []))
    if rest > 1e-6:
        node.setdefault("children", []).append(
            _node(name, rest, "unattributed", source, "derived")
        )
    return node


def _uninstrumented(name: str, seconds: float, category: str) -> list[TimeNode]:
    return [_node(name, seconds, category, "no timer", "uninstrumented")]


def _timed(
    timings: dict[str, dict[str, float]], category: str, source: str
) -> list[TimeNode]:
    """Nodes from path-keyed PhaseTiming records ("A", "A/B"), with remainders."""
    tops = sorted(
        {path.split("/", 1)[0] for path in timings},
        key=lambda t: -timings.get(t, {"seconds": 0})["seconds"],
    )
    out: list[TimeNode] = []
    for top in tops:
        below = {
            p.split("/", 1)[1]: v for p, v in timings.items() if p.startswith(f"{top}/")
        }
        node = _node(
            top,
            timings[top]["seconds"],
            category,
            f"{source}[{top!r}]",
            children=_timed(below, category, source) if below else None,
        )
        if below:
            _with_remainder(node, "Unattributed", f"{top} − sub-timers")
        out.append(node)
    return out


def _stage_tree(stage: dict[str, float], prefix: str, total: float | None) -> TimeNode:
    parts = [
        ("Learning", "learning_seconds", "learning", "forward · backward · optimizer"),
        (
            "Collection",
            "collection_seconds",
            "collection",
            "inference · environment steps · transfer",
        ),
        ("Export", "export_seconds", "export", None),
        ("Diagnostics", "diagnostic_seconds", "export", None),
        ("Admission wait", "admission_wait_seconds", "wait", None),
    ]
    recorded = {
        "collection_seconds": stage.get("collection_timings"),
        "learning_seconds": stage.get("learning_timings"),
    }
    children = [
        _node(
            label,
            stage[key],
            cat,
            f"{prefix}.{key}",
            children=(
                _timed(recorded[key], cat, f"{prefix}.{key[:-8]}_timings")
                if recorded.get(key)
                else _uninstrumented(inside, stage[key], cat)
                if inside
                else None
            ),
        )
        for label, key, cat, inside in parts
        if stage.get(key)
    ]
    for child in children:
        if child.get("children") and child["children"][0]["status"] == "recorded":
            _with_remainder(child, "Unattributed", "timer − sub-timers")
    recorded = total is not None and total > 0
    seconds = total if recorded else sum(c["seconds"] for c in children)
    note = (
        f"CPU {stage['cpu_seconds']:,.0f} s on {stage.get('actual_threads', 1)} thread(s)"
        if stage.get("cpu_seconds")
        else None
    )
    node = _node(
        "Self-play stage",
        seconds,
        "stage",
        f"{prefix}.seconds" if recorded else "sum of stage timers",
        "recorded" if recorded else "derived",
        children,
        note
        or (
            None
            if recorded
            else "stage total not finalized while running; derived from its timers"
        ),
    )
    return _with_remainder(
        node, "Unattributed stage time", f"{prefix}.seconds − timers"
    )


def _counts(point: dict[str, float]) -> tuple[int, int, int]:
    """W/D/L; control points retain only score x games, so half points become draws."""
    if point.get("wins") is not None:
        return int(point["wins"]), int(point["draws"]), int(point["losses"])
    games = int(point["games"])
    halves = round(point["score"] * games * 2)
    wins, draws = halves // 2, halves % 2
    return wins, draws, games - wins - draws


def two_seat() -> Experiment:
    data = json.loads(RETAINED.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    control_hours = {
        p["updates"]: p["training_seconds"] / 3600
        for p in snapshot["control_run"]["monitoring_checkpoints"]
        if p.get("training_seconds") is not None
    }
    decks = ["A", "L"]
    runs: list[Run] = []
    cells: list[list[int]] = []
    for r, (arm, label) in enumerate(
        (("two", "Two-seat"), ("one", "One-seat control"))
    ):
        points = [p for p in data["points"] if p["arm"] == arm]
        updates = sorted({p["updates"] for p in points})
        evals: list[Eval] = []
        for u, update in enumerate(updates):
            at = [p for p in points if p["updates"] == update]
            head = next(p for p in at if p["matchup"] == "all" and p["seat"] == "all")
            hours = head.get("hours") if arm == "two" else control_hours.get(update)
            evals.append(
                {
                    "updates": update,
                    "decisions": head.get("decisions"),
                    "hours": hours,
                    "usd": None,
                    "sha": head.get("sha256"),
                }
            )
            split = any(p["seat"] != "all" for p in at)
            for p in at:
                if p["matchup"] == "all" or (split and p["seat"] == "all"):
                    continue
                seat = {"play": 0, "draw": 1, "all": -1}[p["seat"]]
                deck, opponent = (
                    decks.index(p["matchup"][0]),
                    decks.index(p["matchup"][2]),
                )
                cells.append([r, u, 0, deck, opponent, seat, *_counts(p)])
        runs.append({"id": arm, "label": label, "evals": evals})
    direct = [
        {
            "updates": d["updates"],
            **{k: d["groups"]["all"][k] for k in ("wins", "draws", "losses")},
        }
        for d in (data["duels"] if isinstance(data["duels"], list) else [data["duels"]])
    ]
    rows = snapshot["training"]["rows"]
    keep = [
        r
        for i, r in enumerate(rows)
        if i % 4 == 0
        or r.get("numerical/nonfinite_count")
        or r.get("numerical/rejected_steps")
    ]
    keys = sorted(
        {k for r in keep for k in r if not k.startswith("progress/observation")}
    )
    series = {k: [r.get(k) for r in keep] for k in keys}
    seconds = [r["progress/training_seconds"] for r in keep]
    ups = [r["progress/updates"] for r in keep]
    series["derived/seconds_per_update"] = [None] + [
        (seconds[i] - seconds[i - 1]) / (ups[i] - ups[i - 1])
        for i in range(1, len(keep))
    ]
    stage = snapshot["training"]["summary"]["stages"][0]
    control_stage = snapshot["control_run"]["stages"][0]
    phases = [
        {
            "run": 0,
            "collection": stage["collection_seconds"],
            "learning": stage["learning_seconds"],
            "export": stage["export_seconds"] + stage["diagnostic_seconds"],
            "other": stage["seconds"]
            - stage["collection_seconds"]
            - stage["learning_seconds"]
            - stage["export_seconds"]
            - stage["diagnostic_seconds"],
        },
        {
            "run": 1,
            "collection": control_stage["collection_seconds"],
            "learning": control_stage["learning_seconds"],
            "export": control_stage["export_seconds"]
            + control_stage["diagnostic_seconds"],
            "other": None,
        },
    ]
    summary = snapshot["training"]["summary"]
    control = snapshot["control_run"]
    two_run = _with_remainder(
        _node(
            "Training run",
            summary["elapsed_seconds"],
            "run",
            "training.summary.elapsed_seconds",
            children=[
                _stage_tree(stage, "stage", stage["seconds"]),
                _node(
                    "Monitoring export",
                    summary["monitoring_export_seconds"],
                    "export",
                    "summary.monitoring_export_seconds",
                ),
            ],
        ),
        "Unattributed run time",
        "elapsed − recorded",
    )
    control_run = _with_remainder(
        _node(
            "Training run",
            control["calendar_seconds"],
            "run",
            "control_run.calendar_seconds",
            children=[
                _node(
                    "Setup",
                    control["setup_seconds"],
                    "setup",
                    "control_run.setup_seconds",
                ),
                _stage_tree(
                    control_stage, "control_run.stage", control_stage["seconds"]
                ),
                _node(
                    "Monitoring export",
                    control["monitoring_export_seconds"],
                    "export",
                    "control_run.monitoring_export_seconds",
                ),
            ],
        ),
        "Unattributed run time",
        "calendar − recorded",
    )
    attempts = sorted(
        snapshot["ledger"]["attempts"].values(),
        key=lambda a: (a["updates"], a["started_at"]),
    )
    done = [a for a in attempts if a["finished_at"]]
    open_attempts = [a for a in attempts if not a["finished_at"]]
    evaluation = _node(
        "Evaluation follower",
        sum(a["finished_at"] - a["started_at"] for a in done),
        "eval",
        "ledger.attempts",
        children=[
            _node(
                f"update {a['updates']:,}",
                a["finished_at"] - a["started_at"],
                "eval",
                f"attempt {a['digest'][:12]}",
                note=f"{a['projection']} projection",
            )
            for a in done
        ],
        note=f"{len(open_attempts)} interrupted attempt(s) with no end time are not drawn"
        if open_attempts
        else None,
    )
    timing = [
        {
            "run": 0,
            "resource": "Training · L4",
            "tree": two_run,
            "decisions": stage["learner_transitions"],
        },
        {
            "run": 1,
            "resource": "Training · L4",
            "tree": control_run,
            "decisions": control_stage["learner_transitions"],
        },
        {
            "run": 0,
            "resource": "Evaluation · Mini",
            "tree": evaluation,
            "decisions": stage["learner_transitions"],
        },
    ]
    context = {
        "status": summary["status"],
        "detail": summary["error"],
        "controller": f"{snapshot['controller']['phase']} ({snapshot['controller']['error']})",
        "captured_at": snapshot["captured_at"],
        "heartbeat_at": snapshot["controller"]["heartbeat_at"],
        "device": "NVIDIA L4",
        "seeds": [snapshot["control_run"]["seed"]],
        "world": "w6 · Allies v Lessons",
        "caveat": "Not a matched comparison: the one-seat job is the earlier v1 ladder, with a different learning path and host, "
        "scheduled by updates. Seat comparison v1 is the matched study.",
        "budget": None,
    }
    return {
        "training": {
            "series": [{"run": 0, "updates": ups, "metrics": series}],
            "phases": phases,
        },
        "timing": timing,
        "context": context,
        "id": "two-seat",
        "title": "MTG-165 · two-seat ladder (confounded)",
        "synthetic": False,
        "source": "Retained snapshot · Oct 22",
        "question": "Does learning from both seats make a stronger bot?",
        "decks": decks,
        "deck_names": ["Allies", "Lessons"],
        "anchors": ["heuristic"],
        "runs": runs,
        "cells": cells,
        "direct": direct,
    }


# ---------------------------------------------------------------- seat comparison (real, store)

SEAT_DATA = REPOSITORY / "experiments/data/seat-comparison-v1"
SEAT_DECKS = ["gw_allies", "ur_lessons"]


def seat_comparison() -> Experiment:
    ledger = json.loads((SEAT_DATA / "ledger.json").read_text())
    attempts = {
        k: v for k, v in ledger["attempts"].items() if v["phase"] == "completed"
    }
    job_ids = sorted(
        {v["job"] for k, v in attempts.items() if not k.endswith("-initial")},
        key=lambda j: (j.endswith("-one"), j),
    )  # both-seat jobs first
    runs: list[Run] = []
    cells: list[list[int]] = []
    timing: list[dict[str, object]] = []
    for r, job in enumerate(job_ids):
        seed = int(job.split("-")[-2])
        # The shared initial model is each run's 0 point.
        points = [
            (
                f"{seed}-initial",
                {
                    **attempts[f"{seed}-initial"],
                    "learner_decisions": 0,
                    "label": "initial",
                },
            )
        ]
        points += sorted(
            (
                (k, v)
                for k, v in attempts.items()
                if v["job"] == job and not k.endswith("-initial")
            ),
            key=lambda kv: kv[1]["learner_decisions"],
        )
        evals: list[Eval] = []
        for u, (key, a) in enumerate(points):
            label = a["label"]
            align = 0.0 if label == "initial" else float(label.rstrip("M")) * 1e6
            evals.append(
                {
                    "updates": a["updates"],
                    "decisions": a["learner_decisions"],
                    "align": align,
                    "hours": a["training_seconds"] / 3600,
                    "usd": None,
                    "sha": a["sha256"],
                }
            )
            for rows in (SEAT_DATA / "evaluations" / key).glob("arena/*/rows.json"):
                for row in json.loads(rows.read_text()):
                    mine_a = row["player_a"] == "monitor-checkpoint"
                    seat = row["player_a_seat"] if mine_a else 1 - row["player_a_seat"]
                    score = row["score_a"] if mine_a else 1 - row["score_a"]
                    deck = SEAT_DECKS.index(row["seat_decks"][seat])
                    opponent = SEAT_DECKS.index(row["seat_decks"][1 - seat])
                    w, d = (1, 0) if score == 1 else (0, 1) if score == 0.5 else (0, 0)
                    cells.append([r, u, 0, deck, opponent, seat, w, d, 1 - w - d])
        runs.append({"id": job.removeprefix("seat-comparison-v1-"), "evals": evals})
        last = points[-1][1]
        timing.append(
            {
                "run": r,
                "resource": "Training · L4",
                "decisions": last["learner_decisions"],
                "tree": _node(
                    "Training to last scored checkpoint",
                    last["training_seconds"],
                    "run",
                    "ledger.attempts[].training_seconds",
                    children=_uninstrumented(
                        "collection · learning split not retained for this run",
                        last["training_seconds"],
                        "unattributed",
                    ),
                    note="calibration (seed 64000, 60 updates) measured learner decisions/hour, collection / learning: "
                    + ("19.7M / 13.9M" if job.endswith("both") else "10.6M / 13.8M"),
                ),
            }
        )
        done = [
            v for k, v in points if v.get("finished_at") and v["label"] != "initial"
        ]
        timing.append(
            {
                "run": r,
                "resource": "Evaluation · Mini",
                "decisions": last["learner_decisions"],
                "tree": _node(
                    "Evaluation follower",
                    sum(v["finished_at"] - v["started_at"] for v in done),
                    "eval",
                    "ledger.attempts",
                    children=[
                        _node(
                            v["label"],
                            v["finished_at"] - v["started_at"],
                            "eval",
                            "attempt",
                        )
                        for v in done
                    ],
                ),
            }
        )
    heartbeat = max(j["heartbeat_at"] for j in ledger["jobs"].values())
    return {
        "id": "seat-comparison",
        "title": "Seat comparison v1 · from scratch",
        "synthetic": False,
        "source": "Artifact store · Oct 23",
        "question": "Does learning from both seats match one seat per learner decision?",
        "decks": SEAT_DECKS,
        "deck_names": ["Allies", "Lessons"],
        "anchors": ["heuristic"],
        "runs": runs,
        "cells": cells,
        "direct": [],
        "training": {"series": [], "phases": []},
        "timing": timing,
        "context": {
            "status": "completed",
            "detail": "one-seat jobs paused at their rental deadline after 2,390–2,589 updates",
            "controller": "completed",
            "captured_at": max(v.get("finished_at") or 0 for v in attempts.values()),
            "heartbeat_at": heartbeat,
            "device": "NVIDIA L4",
            "world": "w13 · Allies v Lessons",
            "seeds": [64001, 64002, 64003, 64004],
            "budget": {"spent": 54.40, "ceiling": 100.0},
        },
    }


# ---------------------------------------------------------------- elemental (synthetic)

ELEMENTAL_DECKS = ["BG", "GR", "GW", "RB", "UB", "UG", "UR", "UW", "WB", "WR"]
ELEMENTAL_UPDATES = list(range(0, 5001, 500))
ELEMENTAL_RUNS = (
    ("seed-17402", "Uniform · seed 17402", 1.0, 2.6e6),
    ("seed-17403", "Uniform · seed 17403", 0.85, 2.5e6),
    ("curriculum", "Curriculum · seed 17402", 0.6, 2.6e6),
)


def _p_win(
    rate: float, deck: int, opponent: int, anchor: int, updates: int, seat: int
) -> float:
    deck_edge = 0.45 * math.sin(1.7 * deck + 0.4) - 0.45 * math.sin(
        1.7 * opponent + 0.4
    )
    learn = rate * (0.65 + 0.45 * math.cos(2.3 * deck))
    progress = learn * (1 - math.exp(-updates / 1600))
    logit = (
        -1.3
        + 2.8 * progress
        + deck_edge
        - 1.0 * anchor
        + (0.25 if seat == 0 else -0.25)
    )
    return 1 / (1 + math.exp(-logit))


def elemental() -> Experiment:
    rng = random.Random(17402)
    runs: list[Run] = []
    cells: list[list[int]] = []
    for r, (run_id, label, rate, per_hour) in enumerate(ELEMENTAL_RUNS):
        evals: list[Eval] = []
        for u, update in enumerate(ELEMENTAL_UPDATES):
            hours = update * 26.0 / 3600
            evals.append(
                {
                    "updates": update,
                    "decisions": hours * per_hour,
                    "hours": hours,
                    "usd": hours * 0.62,
                    "sha": f"{rng.getrandbits(64):016x}",
                }
            )
            # MTG-181 cohort: every model deck x heuristic deck, 50 games per
            # cell, play/draw drawn per game; one row per (cell, seat) with counts.
            for d in range(10):
                for o in range(10):
                    counts = [[0, 0, 0], [0, 0, 0]]
                    for _ in range(50):
                        seat = rng.randrange(2)
                        x = rng.random()
                        p = _p_win(rate, d, o, 1, update, seat)
                        counts[seat][
                            0
                            if x >= 0.03 and x < 0.03 + 0.97 * p
                            else 1
                            if x < 0.03
                            else 2
                        ] += 1
                    for seat, (w, dr, lost) in enumerate(counts):
                        if w + dr + lost:
                            cells.append([r, u, 0, d, o, seat, w, dr, lost])
        runs.append({"id": run_id, "label": label, "evals": evals})
    return {
        "training": _elemental_training(rng),
        "timing": _elemental_timing(),
        "context": {
            "status": "running",
            "detail": None,
            "controller": "observing",
            "captured_at": 1792800000.0,
            "heartbeat_at": 1792799940.0,
            "device": "NVIDIA L4",
            "seeds": [17402, 17403],
            "world": "w10 · Elemental ten-deck",
            "budget": {"spent": 3 * 22.39, "ceiling": 75.0},
        },
        "id": "elemental",
        "title": "Elemental · ten decks (synthetic)",
        "synthetic": True,
        "source": "Synthetic data",
        "question": "Does uniform ten-deck self-play improve every deck?",
        "decks": ELEMENTAL_DECKS,
        "deck_names": ELEMENTAL_DECKS,
        "anchors": ["heuristic"],
        "runs": runs,
        "cells": cells,
        "direct": [],
    }


def _elemental_training(rng: random.Random) -> dict[str, object]:
    """Synthetic per-update telemetry under the trainer's real metric keys."""
    series = []
    ups = list(range(10, 5001, 20))
    for r, (_, _, rate, per_hour) in enumerate(ELEMENTAL_RUNS):

        def noise(scale: float) -> float:
            return rng.gauss(0, scale)

        m: dict[str, list[float | None]] = {
            k: []
            for k in (
                "rl/loss",
                "rl/policy_loss",
                "rl/value_loss",
                "rl/entropy",
                "rl/collection_kl",
                "rl/learning_rate",
                "rl/retained_fraction",
                "numerical/gradient_norm_before",
                "numerical/gradient_norm_after",
                "numerical/nonfinite_count",
                "numerical/rejected_steps",
                "numerical/legal_underflow_count",
                "throughput/learner_transitions_per_second",
                "throughput/games_per_second",
                "progress/rss_bytes",
                "progress/host_load_1m",
                "derived/seconds_per_update",
                "progress/training_seconds",
                "progress/learner_transitions",
            )
        }
        for u in ups:
            t = u / 5000
            value = 0.55 * math.exp(-3 * t * rate) + 0.18 + noise(0.015)
            policy = -0.02 - 0.03 * t + noise(0.006)
            m["rl/value_loss"].append(value)
            m["rl/policy_loss"].append(policy)
            m["rl/entropy"].append(1.3 * math.exp(-1.4 * t * rate) + 0.35 + noise(0.02))
            m["rl/loss"].append(value * 0.5 + policy - 0.01 * m["rl/entropy"][-1])
            spike = 6.0 if (r == 2 and 3100 < u < 3300) else 0.0
            m["rl/collection_kl"].append(
                abs(0.008 + 0.004 * t + noise(0.0015)) + spike * 0.004
            )
            m["rl/learning_rate"].append(3e-4 * (1 - 0.8 * t))
            m["rl/retained_fraction"].append(0.25)
            g = abs(1.4 + 0.6 * math.exp(-4 * t) + noise(0.15)) + spike
            m["numerical/gradient_norm_before"].append(g)
            m["numerical/gradient_norm_after"].append(min(g, 1.0))
            m["numerical/nonfinite_count"].append(0)
            m["numerical/rejected_steps"].append(
                1 if spike and rng.random() < 0.5 else 0
            )
            m["numerical/legal_underflow_count"].append(rng.choice([0] * 30 + [1]))
            m["throughput/learner_transitions_per_second"].append(
                per_hour / 3600 * (1 + noise(0.04))
            )
            m["throughput/games_per_second"].append(
                per_hour / 3600 / 105 * (1 + noise(0.04))
            )
            m["progress/rss_bytes"].append((5.4 + 0.5 * t + noise(0.03)) * 1e9)
            m["progress/host_load_1m"].append(9.5 + noise(0.6))
            m["derived/seconds_per_update"].append(
                26.0 * (1 + noise(0.05)) + (40 if u % 500 == 0 else 0)
            )
            m["progress/training_seconds"].append(u * 26.0)
            m["progress/learner_transitions"].append(u * 26.0 / 3600 * per_hour)
        series.append({"run": r, "updates": ups, "metrics": m})
    total = 5000 * 26.0
    phases = [
        {
            "run": r,
            "collection": total * 0.31,
            "learning": total * 0.63,
            "export": total * 0.02,
            "other": total * 0.04,
        }
        for r in range(len(ELEMENTAL_RUNS))
    ]
    return {"series": series, "phases": phases}


def _elemental_timing() -> list[dict[str, object]]:
    """Synthetic trees using the exact timer names training now records.

    Collection and learning subphases are the keys of
    `StageRecord.collection_timings` / `learning_timings`; values are invented
    (status "illustrative") to preview the default instrumentation.
    """
    out: list[dict[str, object]] = []
    for r, (_, _, rate, per_hour) in enumerate(ELEMENTAL_RUNS):
        k = 1.0 + 0.04 * r

        def fake(
            name: str, sec: float, cat: str, children: list[TimeNode] | None = None
        ) -> TimeNode:
            return _node(
                name,
                sec,
                cat,
                "illustrative · StageRecord timings",
                "hypothetical",
                children,
            )

        collection = _node(
            "Collection",
            40300 * k,
            "collection",
            "stage.collection_seconds",
            children=[
                fake("Environment step and encoding", 17800 * k, "collection"),
                fake("Policy inference and action selection", 13900 * k, "collection"),
                fake("Observation preparation", 2600 * k, "collection"),
                fake("Rollout recording", 3100 * k, "collection"),
                fake("Batch assembly", 1500 * k, "collection"),
            ],
        )
        minibatches = fake(
            "Minibatches",
            72400,
            "learning",
            [
                fake("Forward and loss", 21800, "learning"),
                fake("Backward and gradient clipping", 34600, "learning"),
                fake("Optimizer step", 9300, "learning"),
                fake("Minibatch diagnostics", 5200, "learning"),
            ],
        )
        learning = _node(
            "Learning",
            81900,
            "learning",
            "stage.learning_seconds",
            children=[
                fake("Batch transfer and behavior checks", 2900, "learning"),
                fake("Targets", 3100, "learning"),
                fake("Selection and health diagnostics", 1700, "learning"),
                minibatches,
                fake("EMA update", 900, "learning"),
            ],
        )
        _with_remainder(minibatches, "Unattributed", "Minibatches − sub-timers")
        export = _node("Export", 2600, "export", "stage.export_seconds")
        for n in (collection, learning):
            _with_remainder(n, "Unattributed", "timer − sub-timers")
        stage = _with_remainder(
            _node(
                "Self-play stage",
                130000 + 1612 * r,
                "stage",
                "stage.seconds",
                children=[learning, collection, export],
            ),
            "Unattributed stage time",
            "stage.seconds − timers",
        )
        run = _with_remainder(
            _node(
                "Training run",
                stage["seconds"] + 520,
                "run",
                "run.calendar_seconds",
                children=[
                    _node("Setup", 120, "setup", "run.setup_seconds"),
                    stage,
                    _node(
                        "Monitoring export",
                        400,
                        "export",
                        "run.monitoring_export_seconds",
                    ),
                ],
            ),
            "Unattributed run time",
            "calendar − recorded",
        )
        decisions = 5000 * 26.0 / 3600 * per_hour
        out.append(
            {"run": r, "resource": "Training · L4", "tree": run, "decisions": decisions}
        )
        out.append(
            {
                "run": r,
                "resource": "Evaluation · Mini",
                "decisions": decisions,
                "tree": _node(
                    "Evaluation follower",
                    11 * 880 * k,
                    "eval",
                    "ledger.attempts",
                    children=[
                        _node(f"update {u:,}", 880 * k, "eval", "attempt")
                        for u in ELEMENTAL_UPDATES
                    ],
                ),
            }
        )
    return out


# What each run's launch declared. Real runs read `RunDeclaration.job_name`
# and `seed`; the MTG-165 runs predate typed jobs, so their names are supplied
# here as stand-ins. Elemental is synthetic throughout.
DECLARED = {
    # seat-comparison: `experiments/seat_comparison.py` Case labels; "one" declares none.
    **{f"{s}-both": ("learn from both seats", s) for s in (64001, 64002, 64003, 64004)},
    **{f"{s}-one": ("one", s) for s in (64001, 64002, 64003, 64004)},
    "two": ("Both seats learn", 15902),
    "one": ("One seat learns", 15902),
    "seed-17402": ("Uniform decks", 17402),
    "seed-17403": ("Uniform decks", 17403),
    "curriculum": ("Curriculum", 17402),
}


# Each experiment's declared baseline job. seat-comparison: `Experiment(baseline=
# Baseline.capture("seat-comparison-one", ...))` in experiments/seat_comparison.py.
DECLARED_BASELINE = {
    "seat-comparison": "one",
    "two-seat": "One seat learns",
    "elemental": "Uniform decks",
}


def _label_runs(runs: list[Run]) -> None:
    """Label = job name, plus the seed when several runs share the job."""
    for run in runs:
        run["job_name"], run["seed"] = DECLARED[run["id"]]
    for run in runs:
        siblings = sum(r["job_name"] == run["job_name"] for r in runs)
        run["label"] = run["job_name"] + (
            f" · seed {run['seed']}" if siblings > 1 else ""
        )


def write(out: Path) -> Path:
    """Write the page, its scripts and `data.js` to `out`; return the page."""
    if not (SEAT_DATA / "ledger.json").exists():
        raise SystemExit(
            "seat-comparison data missing; run "
            "`uv run --extra artifacts python -m manabot.data fetch "
            "experiments/data/seat-comparison-v1`"
        )
    experiments = [seat_comparison()]
    if RETAINED.exists() and SNAPSHOT.exists():
        experiments.append(two_seat())
    else:
        print("skipping MTG-165: its retained snapshot is not in this checkout")
    experiments.append(elemental())
    for experiment in experiments:
        experiment["layout"] = asdict(REPORTS[experiment["id"]])
        _label_runs(experiment["runs"])
        # The report's choice wins; otherwise the experiment's declared baseline.
        baseline = (
            experiment["layout"]["baseline"] or DECLARED_BASELINE[experiment["id"]]
        )
        if not any(r["job_name"] == baseline for r in experiment["runs"]):
            raise ValueError(f"{experiment['id']}: baseline {baseline!r} is not a job")
        experiment["baseline"] = baseline
    out.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        shutil.copyfile(ROOT / asset, out / asset)
    report = {"experiments": experiments}
    (out / "data.js").write_text(
        "window.REPORT=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
    return out / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY / ".runs/report-prototype"
    )
    print(write(parser.parse_args().out))


if __name__ == "__main__":
    main()
