"""Report panels for the comparisons a manabot training study asks.

An experiment's `report()` composes these into an Etude `Report`, beside
`panels.PERF` and `panels.ML`, and adds only what is specific to it. Each helper
returns an ordinary Etude record, so a report mixes them with its own panels:

- `job_labels`: `Report.labels`, a display name per job from its regime's model;
- `against_earlier_self`, `against_fixed`: eval `Curve`s with a line per job;
- `head_to_head`: an eval `Curve` of named pairs, one line for the first seat;


Jobs are keyed by their Etude name, which for a `TrainingJob` is its regime id.
Matches are selected by the tags or names `eval_jobs(state)` gives its seatings.
"""

from collections.abc import Mapping, Sequence

from etude.report import Curve, Matches

from manabot.training.regime import TrainingRegime

# -- Names ---------------------------------------------------------------------


def job_labels(jobs: Mapping[str, tuple[str, TrainingRegime]]) -> dict[str, str]:
    """`Report.labels`: each job's size word, then its model's width and depth.

    `jobs` maps a job's Etude name to the word a reader knows it by and its
    regime. Attention layers are named only for a job whose depth differs from
    the first job's, so "Small (width 64)" and "Large (width 160, 8 layers)".
    """
    if not jobs:
        return {}
    first = next(iter(jobs.values()))[1].agent.attention_layers
    labels: dict[str, str] = {}
    for name, (word, regime) in jobs.items():
        model = regime.agent
        parts = [f"width {model.hidden_dim}"]
        if model.attention_layers != first:
            layers = model.attention_layers
            parts.append(f"{layers} layer{'' if layers == 1 else 's'}")
        labels[name] = f"{word} ({', '.join(parts)})"
    return labels


# -- Comparisons ---------------------------------------------------------------


def against_earlier_self(title: str, tags: Sequence[str]) -> Curve:
    """Each checkpoint against an earlier checkpoint of its own job, a line per job.

    `tags` name the matchups that pair a checkpoint with its earlier self,
    typically one tag per job. Each checkpoint's point counts every such match
    it sat in.
    """
    return Curve(title=title, matches=tuple(Matches(tag=tag) for tag in tags))


def against_fixed(title: str, tags: Sequence[str]) -> Curve:
    """Each checkpoint against the fixed player selected by the tags, a line per job."""
    return Curve(title=title, matches=tuple(Matches(tag=tag) for tag in tags))


def head_to_head(title: str, seatings: Sequence[str]) -> Curve:
    """One line: the first seat's score where its checkpoint stood.

    `seatings` are the names `eval_jobs(state)` gave the pairs. An empty sequence
    raises `ValueError`.
    """
    if not seatings:
        raise ValueError("head-to-head needs at least one seating")
    return Curve(title=title, matches=tuple(Matches(seating=name) for name in seatings))
