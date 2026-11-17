"""Record independent Python heuristic decisions on the current engine.

Run with ``uv run --extra dev python scripts/record_heuristic_parity.py``.
The reference bytes come unchanged from their original commit. The adapter
restores its pre-v11 card-list view using the native legacy projection; source
payments, which did not exist in that contract, explicitly choose PayCost.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import managym
from tests.sim.test_heuristic import _players

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "757bb6257ef07cf2536ad7f7ec24f0f7309628c6:manabot/sim/heuristic.py"


class _ObservationAdapter:
    def __init__(self, raw: managym.Observation) -> None:
        self._raw = raw
        self.agent_cards, self.opponent_cards = raw.legacy_card_lists()

    def __getattr__(self, name: str) -> Any:
        # The frozen Python reference consumes the historical native interface.
        return getattr(self._raw, name)


def main() -> None:
    source = subprocess.check_output(["git", "show", REFERENCE], cwd=ROOT)
    original = json.loads(
        (ROOT / "tests/fixtures/native_heuristic_parity.json").read_text()
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "reference.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("retained_heuristic", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        receipt = {
            "reference": REFERENCE,
            "reference_sha256": hashlib.sha256(source).hexdigest(),
            "engine_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "observation_adapter": "Observation.legacy_card_lists restores pre-v11 card lists",
            "payment_policy": "PayCost at source-only payments; unchanged Python reference elsewhere",
            "world": managym.world_identity(),
            "max_decisions": original["max_decisions"],
            "cases": [],
        }
        for case in original["cases"]:
            env = managym.Env(seed=case["seed"])
            raw, _ = env.reset(_players(case["decks"]))
            players = [
                module.HeuristicPlayer(max_actions=case["max_actions"])
                for _ in range(2)
            ]
            actions: list[int] = []
            done = truncated = False
            for _ in range(original["max_decisions"]):
                offered = raw.action_space.actions
                sources_only = (
                    raw.action_space.action_space_type
                    == managym.ActionSpaceEnum.PAY_OR_NOT
                    and any(
                        a.action_type == managym.ActionEnum.TAP_FOR_COST
                        for a in offered
                    )
                    and all(
                        a.action_type
                        in (
                            managym.ActionEnum.TAP_FOR_COST,
                            managym.ActionEnum.PAY_COST,
                        )
                        for a in offered
                    )
                )
                chosen = (
                    next(
                        i
                        for i, a in enumerate(offered)
                        if a.action_type == managym.ActionEnum.PAY_COST
                    )
                    if sources_only
                    else players[raw.agent.player_index].act(
                        SimpleNamespace(last_raw_obs=_ObservationAdapter(raw)), {}
                    )
                )
                actions.append(chosen)
                raw, _, done, truncated, _ = env.step(chosen)
                if done or truncated:
                    break
            receipt["cases"].append(
                dict(case, actions=actions, terminated=done, truncated=truncated)
            )
    (ROOT / "tests/fixtures/native_heuristic_smart_payment_parity.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
