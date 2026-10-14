"""Read retained ETU-118 attempts and generate a notebook-owned HTML report.

Validation is at the saved JSON boundary. No function trains, plays games,
mutates evidence or accesses a tracker. Bootstrap intervals resample training
seeds; identical observed seed gains do not establish population certainty.
"""

import argparse
import base64
from html import escape
from io import BytesIO
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

from matplotlib.figure import Figure
import nbformat
import numpy as np
from pydantic import BaseModel

from experiments.runners.current_baseline_evidence import Result
from manabot.arena.models import file_sha256
from manabot.training.execution import atomic_json
from manabot.training.experiment_report import load_evidence, metric_figure


class SeedResult(BaseModel):
    seed: int
    initial: float
    trained: float
    frozen: float
    gain: float


class Summary(BaseModel):
    status: str
    phase: str
    seconds: float
    source_sha256: str
    seeds: list[SeedResult] = []
    passed: bool = False
    mean_gain: float | None = None
    seed_interval: tuple[float, float] | None = None


def summarize(path: Path) -> Summary:
    result = Result.model_validate_json(path.read_text())
    summary = Summary(
        status=result.status,
        phase="full-game" if result.full_game else "target-task",
        seconds=result.seconds,
        source_sha256=file_sha256(path),
    )
    expected = (11821, 11822, 11823) if result.full_game else (11801, 11802, 11803)
    size = 48 if result.full_game else 128
    if result.status != "completed" or len(result.attempts) != 6:
        return summary
    for seed in expected:
        paired = [a for a in result.attempts if a.seed == seed]
        if len(paired) != 2 or {a.frozen for a in paired} != {False, True}:
            raise ValueError("cohort is not the declared seed/control cross")
        learned = next(a for a in paired if not a.frozen)
        frozen = next(a for a in paired if a.frozen)
        cohorts = (learned.initial, learned.rows, frozen.rows)
        if any(a.status != "completed" for a in paired) or any(
            len(rows) != size or not all(row.replay for row in rows) for rows in cohorts
        ):
            raise ValueError("incomplete or non-replayed cohort")
        start = 1_911_181_000 if result.full_game else 1_911_180_000
        expected_games = [
            (deal, leg)
            for deal in range(start, start + (12 if result.full_game else 64))
            for leg in range(4 if result.full_game else 2)
        ]
        if any(
            [(r.deal, r.leg if result.full_game else r.seat) for r in rows]
            != expected_games
            for rows in cohorts
        ):
            raise ValueError("evaluation differs from the frozen paired cohort")
        if learned.initial != frozen.rows:
            raise ValueError("frozen outputs differ from initialization")
        initial, trained, control = [
            sum(row.win for row in rows) / size for rows in cohorts
        ]
        summary.seeds.append(
            SeedResult(
                seed=seed,
                initial=initial,
                trained=trained,
                frozen=control,
                gain=trained - initial,
            )
        )
    gains = np.array([s.gain for s in summary.seeds])
    bootstrap = np.random.default_rng(118).choice(gains, (10000, 3)).mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    summary.seed_interval = float(lower), float(upper)
    summary.mean_gain = float(gains.mean())
    summary.passed = bool(
        (summary.mean_gain >= 0.10 and gains.min() > 0 and lower > 0)
        if result.full_game
        else all(s.trained >= 0.85 and s.gain >= 0.25 for s in summary.seeds)
    )
    return summary


def _image(figure: Figure) -> str:
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=120, bbox_inches="tight")
    return (
        '<img alt="Saved evidence chart" src="data:image/png;base64,'
        + base64.b64encode(stream.getvalue()).decode()
        + '">'
    )


def render(root: Path, output: Path) -> Path:
    """Only retained originals are read. Output is derived, never measurement authority."""
    paths = sorted(root.glob("etu118-*/result.json"))
    summaries = [(path, summarize(path)) for path in paths]
    if not summaries:
        raise ValueError("no ETU-118 evidence found")
    sections: list[str] = []
    for path, summary in summaries:
        title = f"{path.parent.name}: {summary.status}; {'DEBUGGING PASS' if summary.passed else 'DEBUGGING CRITERION NOT MET'}"
        rows = "".join(
            f"<tr><td>{s.seed}</td><td>{s.initial:.1%}</td><td>{s.trained:.1%}</td><td>{s.frozen:.1%}</td><td>{s.gain:+.1%}</td></tr>"
            for s in summary.seeds
        )
        result = Result.model_validate_json(path.read_text())
        error = result.error or "None"
        sections.append(
            f"<h2>{escape(title)}</h2><p>Elapsed: {summary.seconds:.2f} s. Failure: {escape(error)}.</p><table><tr><th>Seed</th><th>Initial</th><th>Trained</th><th>Frozen</th><th>Gain</th></tr>{rows}</table>"
        )
        if summary.seed_interval:
            sections.append(
                f"<p>Mean gain {summary.mean_gain:+.1%}; paired training-seed bootstrap 95% interval [{summary.seed_interval[0]:+.1%}, {summary.seed_interval[1]:+.1%}]. Three seeds remain exploratory.</p>"
            )
        attempts = "".join(
            f"<li>Seed {a.seed}, {'frozen' if a.frozen else 'learning'}: {a.status}; "
            f"{len(a.rows)} scored games; {a.seconds:.2f} s"
            + (
                f"; score {sum(row.win for row in a.rows) / len(a.rows):.1%}"
                if a.rows
                else ""
            )
            + (f"; {escape(a.error)}" if a.error else "")
            + "</li>"
            for a in result.attempts
        )
        sections.append(
            "<details><summary>Every attempt, including partial results</summary><ul>"
            + attempts
            + "</ul></details>"
        )
        if summary.seeds:
            figure = Figure(figsize=(7, 3))
            axes = figure.subplots()
            x = np.arange(3)
            for offset, field, label in (
                (-0.24, "initial", "Initial"),
                (0, "trained", "Trained"),
                (0.24, "frozen", "Frozen"),
            ):
                axes.bar(
                    x + offset,
                    [getattr(s, field) for s in summary.seeds],
                    width=0.24,
                    label=label,
                )
            axes.set(
                xticks=x,
                xticklabels=[s.seed for s in summary.seeds],
                ylim=(0, 1),
                ylabel="Terminal score",
            )
            axes.legend()
            sections.append(_image(figure))
        sections.append(
            f"<details><summary>Learning diagnostics and provenance</summary>{_image(metric_figure(load_evidence(path.parent), 'rl/loss'))}<p>{escape(str(path.relative_to(root)))}<br>SHA256 {summary.source_sha256}</p></details>"
        )
    total = sum(summary.seconds for _, summary in summaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '<!doctype html><meta charset="utf-8"><title>ETU-118 current baseline</title><style>body{font:16px system-ui;max-width:900px;margin:3rem auto;padding:0 1rem;color:#20322b}table{border-collapse:collapse}td,th{padding:.5rem 1rem;border-bottom:1px solid #ddd}img{max-width:100%}details{margin:1rem 0}h2{margin-top:2rem}</style><h1>Baseline debugging evidence</h1><p><strong>Sustained baseline and predictive daily test remain unestablished.</strong></p><p>Read-only retained evidence. No training or evaluation runs from this notebook.</p><p>Total recorded attempt time: '
        + f"{total:.2f} seconds. Failed attempts included once. Setup, tests and report generation are additional software work.</p><p>The target task has scripted preparation and continuation. Full-game results use fixed random during training and scripted greedy for evaluation. Neither establishes self-play strength, best architecture or chapter acceptance.</p>"
        + "".join(sections)
    )
    atomic_json(
        output.with_suffix(".summary.json"),
        [summary.model_dump(mode="json") for _, summary in summaries],
    )
    return output


def render_saved(source: Path, output: Path) -> Path:
    """Render originals or the byte-preserving published JSON bundle."""
    if source.is_dir():
        return render(source, output)
    with tempfile.TemporaryDirectory() as temporary, ZipFile(source) as archive:
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in archive.namelist()
        ):
            raise ValueError("evidence archive contains an invalid relative path")
        archive.extractall(temporary)
        return render(Path(temporary), output)


def notebook(root: Path, output: Path) -> None:
    """Create once; subsequent calls preserve every user edit."""
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    book = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Current baseline evidence\n\nRun All reads retained JSON and writes a read-only HTML dashboard. Edit paths or the reporting function call; no cell trains or plays games. Preserve failed attempts alongside completed cohorts."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nfrom experiments.runners.current_baseline_report import render_saved\n\n"
                + f'DATA_ROOT = Path({os.path.relpath(root.resolve(), output.parent.resolve())!r})\nOUTPUT = Path("current-baseline.html")\nrender_saved(DATA_ROOT, OUTPUT)'
            ),
            nbformat.v4.new_markdown_cell(
                "## Interpretation\n\nTarget-task learning is a narrow positive control. Short full-game criteria are debugging evidence only. Sustained multi-seed improvement must precede baseline promotion and daily-test validation. Three seeds and one scripted evaluation opponent do not establish general strength. Add observations here; edits survive refresh."
            ),
        ],
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        },
    )
    with output.open("x") as stream:
        nbformat.write(book, stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook(args.root, args.output.with_suffix(".ipynb"))
    render_saved(args.root, args.output)


if __name__ == "__main__":
    main()
