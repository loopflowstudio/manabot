"""Atomic incremental evidence and concise human/Discord summaries."""

from collections import defaultdict
from pathlib import Path

from manabot.verify.learnability.schema import Result, Score


def markdown(result: Result) -> str:
    lines = [
        f"# manabot learnability: {result.status}",
        f"Source `{result.source}` · world `{result.world or 'unavailable'}`",
        f"Seeds {result.config.seeds}; {result.config.updates} capacity updates/seed; "
        f"threshold {result.config.probability:.2f}; CPU only, one Torch thread.",
        f"Wall {result.wall_seconds:.1f}s; CPU {result.cpu_seconds:.1f}s (setup/build separate).",
        f"{result.fit_cases} fit / {result.held_out_cases} held-out cases "
        f"(`{result.case_set or 'unversioned'}`); "
        f"{result.fit_cases * result.config.updates} supervised sample exposures/seed. "
        "Initialization is the frozen no-update control. Each row gates every preference/variant.",
        "",
        "| Seed | Family | Split | Min P | Init / no update | Ablated | Result |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    groups: dict[tuple[int, str, str], list[Score]] = defaultdict(list)
    for score in result.scores:
        groups[score.seed, score.family, score.split].append(score)
    for (seed, family, split), rows in sorted(groups.items()):
        lines.append(
            f"| {seed} | {family} | {split} | {min(s.trained for s in rows):.3f} | "
            f"{min(s.initial for s in rows):.3f} | {max(s.identity_removed for s in rows):.3f} | "
            f"{'PASS' if all(s.passed for s in rows) else 'FAIL'} |"
        )
    if result.ppo is not None:
        lines += ["", f"Production PPO control: `{result.ppo}`"]
    lines += ["", *[f"Error: {e}" for e in result.errors], "", *result.skips]
    lines += [
        "",
        "Evidence (relative to this report):",
        "[Machine report](result.json)",
    ]
    lines += [
        f"- [{name}]({name}) SHA256 `{digest}`"
        for name, digest in result.artifacts.items()
    ]
    return "\n".join(lines) + "\n"


def save(out: Path, result: Result) -> None:
    for name, content in (
        ("result.json", result.model_dump_json(indent=2)),
        ("report.md", markdown(result)),
    ):
        temp = out / (name + ".tmp")
        temp.write_text(content)
        temp.replace(out / name)
