"""A native payment root for the release proof; no gameplay injection endpoint."""

import managym

DECK: dict[str, int] = {
    "Island": 16,
    "Boomerang Basics": 8,
    "Prophetic Prism": 4,
    "Tarnished Citadel": 4,
    "White Lotus Hideout": 4,
    "Spectral Sailor": 4,
}


def configure_root(env: managym.Env) -> managym.Observation:
    for seat in range(2):
        env.scenario_clear_hand(seat)
    env.scenario_force_card_in_hand(0, "Boomerang Basics")
    for name in (
        "Island",
        "Island",
        "Prophetic Prism",
        "Spectral Sailor",
        "Tarnished Citadel",
        "White Lotus Hideout",
    ):
        env.scenario_force_battlefield(0, name, ready=True)
    return env.scenario_refresh()
