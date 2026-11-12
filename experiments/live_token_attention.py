"""MTG-319: live-token attention against bucketed shapes on one rented L4.

Exploratory pilot of `ActorCritic(..., live_tokens=True)` (`manabot.model.live_tokens`).
It reuses MTG-301's profile (`experiments/l4_step_profile.py`): one in-situ w160l8
update keeps collection observations and learner minibatches, and those inputs replay
through four arms in separate processes, interleaved with baseline repeats:
`baseline`, `bucketed`, `live` (packed attention over the full tables) and
`bucketed-live` (packed attention over bucket-trimmed tables), at w160l8 and w384l8.
Each arm also writes fixed-weight numerics compared against the baseline.

Decision rule (report: experiments/live_token_attention.md): build live tokens out only
if `bucketed-live` takes at most 1/1.5 of `bucketed`'s learner step at w384l8 with
outputs equal to within float tolerance.

    uv run etude launch experiments/live_token_attention.py --rent l4-etude \\
        --revision COMMIT
"""

from __future__ import annotations

from typing import Any

from experiments import l4_step_profile


def runs(revision: str, hours: str = "2.0", usd: str = "3.00") -> Any:
    """One L4 job of the `live` candidate at `revision`."""
    return l4_step_profile.runs(
        revision,
        candidate="live",
        hours=hours,
        usd=usd,
        name="mtg319-live-token-attention",
    )
