"""Etude report panels over the metrics a self-play learner logs.

An experiment file's `report()` passes these as `Report(perf=PERF, ml=ML)`, so
its Perf and ML details draw manabot's own keys (`update_metrics.metric_row`)
rather than Etude's reinforcement-learning defaults. Keys are unprefixed:
`entropy`, `collection_kl`, `numerical/...`, and the run's coordinates under
`coordinates/`. Phase seconds per update are under `time/`
(`update_metrics.phase_seconds`), drawn by `Flame`.
"""

from etude.report import Attempts, Counters, Elapsed, Flame, MetricPanel, Pace, Series

PERF: tuple[MetricPanel, ...] = (
    Pace(),
    Elapsed(),
    Attempts(),
    Flame(),
    Series(title="Games played", keys=("coordinates/games",), unit="games"),
    Series(title="Process memory", keys=("coordinates/rss_bytes",), unit="bytes"),
    Series(title="Host load", keys=("coordinates/host_load_1m",), unit="1-min load"),
)

ML: tuple[MetricPanel, ...] = (
    # Expected to stay at zero; a nonzero value is a numerical incident.
    Counters(
        title="Numerical stability",
        keys=(
            "numerical/nonfinite_count",
            "numerical/rejected_steps",
            "numerical/skipped_steps",
            "numerical/legal_underflow_count",
            "numerical/empty_actor_batch",
        ),
    ),
    Series(title="Losses", keys=("loss", "policy_loss", "value_loss")),
    Series(title="Policy entropy", keys=("entropy",), unit="nats"),
    Series(title="KL divergence", keys=("collection_kl", "reference_kl"), scale="log"),
    Series(
        title="Gradient norm",
        keys=("numerical/gradient_norm_before", "numerical/gradient_norm_after"),
        unit="L2",
    ),
    Series(
        title="Relative parameter change",
        keys=("numerical/parameter_relative_delta",),
        scale="log",
    ),
    Series(title="Learning rate", keys=("learning_rate",)),
    Series(title="Policy damping", keys=("tau",)),
    # Rows the update kept of those collected (Ataraxos-style filtering).
    Series(title="Retained rows", keys=("retained", "rows"), unit="rows"),
    Series(title="Clipped steps", keys=("numerical/clip_fraction",), unit="fraction"),
)
