"""Headless T5 demo: Ran and Shaw's cast-only copy and Phoenix Fleet Airship's
end-step copy through the text decision path.

Run `uv run python scratch/tools/demo_t5_more.py`. Every choice is an ordinary
native Command chosen from the rendered text decision.
"""

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root


def _drive(env, prefer: list[str], *, until, limit: int = 40) -> str:
    done = until if callable(until) else (lambda text: until in text)
    for _ in range(limit):
        decision = capture_text_decision(env, history_capacity=32)
        if done(decision.text):
            return decision.text
        stage = next(l for l in decision.text.splitlines() if l.startswith("Stage:"))
        choice = next(
            (a for want in prefer for a in decision.actions if want in a.description),
            decision.actions[0],
        )
        print(f"{stage[:70]:70} -> {choice.description[:90]}")
        submit_response(env, decision, choice.reference)
    raise AssertionError(f"never reached {until!r}")


def _lines(text: str, *keys: str) -> None:
    for line in text.splitlines():
        if any(k in line for k in keys) and "zone hand" not in line:
            print("   ", line)


def ran_and_shaw() -> None:
    print("=== Ran and Shaw ===")
    env = priority_root(("Ran and Shaw",), ("Mountain",) * 5)
    for _ in range(4):
        env.scenario_force_battlefield(0, "Ran and Shaw", ready=True)
    text = _drive(
        env,
        ["Keep", "Cast C", "Pass priority"],
        until="token yes",
    )
    _lines(text, "Ran and Shaw) [D2]: controller", "You: life")


def airship() -> None:
    print("=== Phoenix Fleet Airship ===")
    env = priority_root(
        (), ("Swamp", "Joo Dee, One of Many", "Phoenix Fleet Airship")
    )
    text = _drive(
        env,
        ["Activate ability of", "Keep on top", "Joo Dee, One of Many). Native description: Sacrifice", "Pass priority"],
        until="This turn: sacrificed 1",
    )
    text = _drive(
        env,
        ["Pass priority"],
        until=lambda t: any(
            "Phoenix Fleet Airship) [" in l and "token yes" in l for l in t.splitlines()
        ),
        limit=60,
    )
    _lines(text, "Stage:", "Phoenix Fleet Airship) [D2]: controller", "This turn", "Turn")


if __name__ == "__main__":
    ran_and_shaw()
    airship()
