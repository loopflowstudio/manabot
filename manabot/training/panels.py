"""Etude report panels over the metrics a self-play learner logs.

An experiment file's `report()` passes these as `Report(perf=PERF, ml=ML)`, or
calls `self_play_report()` when it has no evaluation panels of its own, so
its Perf and ML details draw manabot's own keys (`update_metrics.metric_row`
and `combat/...`) rather than Etude's reinforcement-learning defaults. Keys are
unprefixed: `entropy`, `collection_kl`, `numerical/...`, `combat/...`, and the
run's coordinates under `coordinates/`. Phase seconds per update are under
`time/` (`update_metrics.phase_seconds`, read from the live collector and
learner profilers, plus `time/combat_probe` on probed updates), drawn by `Flame`
and, for the costs that decide a training machine, by the "Update time" series. Rows from runs before a phase
was timed leave that series blank; retained rows and the Python vector carry only
the environment step's parent key, so a new run compares to them on it. Series
overlay, so the parent beside its children double counts nothing.
"""

from etude.report import (
    Attempts,
    Counters,
    Elapsed,
    Flame,
    MetricPanel,
    Pace,
    Report,
    Series,
)

PERF: tuple[MetricPanel, ...] = (
    Pace(),
    Elapsed(),
    Attempts(),
    Flame(),
    Series(
        title="Update time",
        keys=(
            "time/collection/environment_step_and_encoding",
            "time/collection/environment_step_and_encoding/native_stepping",
            "time/collection/environment_step_and_encoding/observation_encoding",
            "time/collection/observation_preparation",
            "time/collection/policy_inference_and_action_selection",
            "time/collection/stream_bookkeeping",
            "time/collection/batch_assembly",
            "time/learning",
            "time/combat_probe",
        ),
        unit="seconds",
    ),
    Series(title="Games played", keys=("coordinates/games",), unit="games"),
    Series(
        title="Idle stream steps",
        keys=("coordinates/idle_stream_steps",),
        unit="stream slots",
    ),
    Series(title="Process memory", keys=("coordinates/rss_bytes",), unit="bytes"),
    Series(title="Host load", keys=("coordinates/host_load_1m",), unit="1-min load"),
)

ML: tuple[MetricPanel, ...] = (
    Series(title="Learning rate", keys=("learning_rate",)),
    Series(title="Policy damping", keys=("tau",)),
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
    # Rows the update kept of those collected (Ataraxos-style filtering).
    Series(title="Retained rows", keys=("retained", "rows"), unit="rows"),
    Series(title="Clipped steps", keys=("numerical/clip_fraction",), unit="fraction"),
    Series(
        title="Attack and block log-odds",
        keys=("combat/attack/gap", "combat/block/gap"),
        unit="log-odds",
    ),
    *(
        Series(
            title=f"{kind.title()} gap shift under board edits",
            keys=tuple(
                f"combat/{kind}/shift/{edit}"
                for edit in (
                    "my_creatures_0_1",
                    "my_creatures_9_9",
                    "their_creatures_0_1_tapped",
                    "their_creatures_9_9_untapped",
                )
            ),
            unit="log-odds",
        )
        for kind in ("attack", "block")
    ),
)


def self_play_report(*, other_metrics: bool = False) -> Report:
    return Report(perf=PERF, ml=ML, other_metrics=other_metrics)
