"""Capture native search positions for presentation-only browser inspection.

The existing constructed-position helper supplies both seats and durations.
Academy's frame builder produces the browser payload without hand-written UI
ownership data. These frames do not claim a WebSocket search-recovery proof.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "semantic"))

from test_player_control import _choose, _controlled_root

from academy.server import GameSession
import managym


def main() -> None:
    cases: list[dict[str, object]] = []
    for caster in (0, 1):
        for paid in (False, True):
            env = _controlled_root(paid, caster=caster, controlled_hand="Giant Koi")
            session = GameSession(trace_dir=ROOT / ".cache/mtg226-demo/capture")
            session.env = env
            frames: dict[str, object] = {}

            def capture(stage: str) -> None:
                session.revision += 1
                raw = env.observation_for_player(caster)
                frames[stage] = session._build_decision_context(raw, viewer=caster).frame

            capture("delegated")
            _choose(env, managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY)
            for _ in range(20):
                raw = env.observation_for_player(caster)
                frame = session._build_decision_context(raw, viewer=caster).frame
                context = frame["projection"]["decision_context"]
                if context["stage"] == "private_search":
                    capture("search")
                    break
                if "chooses for" not in context["instruction"]:
                    capture("own_response")
                _choose(env, managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            else:
                raise AssertionError("No search reached")
            _choose(env, managym.ActionEnum.SELECT_CARD)
            capture("after_search")
            assert set(frames) == {"delegated", "own_response", "search", "after_search"}
            cases.append({"caster": caster, "paid": paid, "frames": frames})
    target = ROOT / ".cache/mtg226-demo/search-frames.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cases, indent=2))
    print(f"Captured {len(cases)} native seat/duration cases: {target}")


if __name__ == "__main__":
    main()
