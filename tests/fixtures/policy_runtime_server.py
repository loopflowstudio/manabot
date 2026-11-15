"""Isolated browser proof for direct objects and provider-backed Watch seats."""

import json
import os
from pathlib import Path

from managym.academy.launch import serve
from managym.academy.opponent import policy_registry
from managym.policy import PolicySpec
from tests.academy.test_play_modes import _write_tiny_checkpoint
from tests.academy.test_policy_runtime import IndependentFactory


def main() -> None:
    output = Path(os.environ["ACADEMY_POLICY_PROOF"])
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / "baseline.pt"
    _write_tiny_checkpoint(checkpoint)
    refused = IndependentFactory("Refused configuration")
    with serve(
        port=int(os.environ["ACADEMY_POLICY_PORT"]),
        traces_dir=output.parent / "traces",
        policies={
            "independent": IndependentFactory(
                "Independent/long-run/tracker/source/snapshot"
            ),
            "refused-config": refused,
            "baseline": PolicySpec("manabot", str(checkpoint)),
        },
    ) as academy:
        refused_url = academy.new_game(opponent="refused-config", seed=7).url
        refused.runner = "changed-after-launch"
        output.write_text(
            json.dumps(
                {
                    "refused": refused_url,
                    "play": academy.new_game(opponent="independent", seed=7).url,
                    "watch": academy.watch(players=("independent", "baseline")).url,
                    "ids": {
                        name: policy_registry().descriptor(name).id
                        for name in ("independent", "baseline", "refused-config")
                    },
                }
            )
        )
        print("ACADEMY_POLICY_READY", flush=True)
        academy.wait()


if __name__ == "__main__":
    main()
