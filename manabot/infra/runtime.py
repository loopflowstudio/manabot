"""Labels in retained training records; new recipes track directly with Etude."""


def regime_tag(regime: str) -> str:
    """The checkpoint tag naming a regime.

    A launched run is stored under its launcher id, and an experiment's
    `observe` state keeps no job name, so this tag is what tells a stored
    checkpoint's regime apart. Etude keeps tags out of checkpoint keys, so it
    changes no match identity.
    """
    return f"regime:{regime}"
