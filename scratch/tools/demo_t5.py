"""Headless T5 demo: play Joo Dee's copy ability through the built extension.

Run `uv run python scratch/tools/demo_t5.py`. Every choice after the scenario
root is an ordinary native Command chosen from the text decision; the root
keeps sole-offer decisions visible so each priority pass is shown.
"""

from managym.decision_capture import DecisionInput

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root


def _step(env, prefer: list[str]) -> bool:
    decision = capture_text_decision(env, history_capacity=32)
    source = DecisionInput.model_validate_json(decision.capture_json)
    if source.observation.decision is None:
        return False
    pairs = list(zip(source.observation.decision.offers, decision.actions, strict=True))
    print("--- decision ---")
    for offer, action in pairs:
        print(f"  [{offer['verb']}] {action.description}")
    for want in prefer:
        for offer, action in pairs:
            if want in action.description or want == offer["verb"]:
                print(f"  -> choose: {action.description}")
                submit_response(env, decision, action.reference)
                return True
    offer, action = pairs[0]
    print(f"  -> choose (first): {action.description}")
    submit_response(env, decision, action.reference)
    return True


def _show(env, title: str) -> None:
    text = capture_text_decision(env, history_capacity=32).text
    print(f"=== {title} ===")
    for line in text.splitlines():
        if any(k in line for k in ("Joo Dee", "This turn", "copy of", "graveyard")):
            print(line)


def main() -> None:
    env = priority_root((), ("Swamp", "Joo Dee, One of Many"))
    _step(env, ["Activate ability of C2"])
    # Resolve: pass priority, then sacrifice the original (not the token copy).
    for _ in range(8):
        _show(env, "state")
        if not _step(env, ["C2 (Joo Dee", "Pass priority"]):
            break
        text = capture_text_decision(env, history_capacity=32).text
        if "copy of" in text and "sacrificed" in text:
            break
    _show(env, "after the ability")


if __name__ == "__main__":
    main()
