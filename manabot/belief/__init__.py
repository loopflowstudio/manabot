"""Viewer-safe beliefs over canonical managym possible worlds.

Import names from their defining modules. The lazy map below serves only the
belief test helpers that still import from the package."""

from importlib import import_module

_EXPORTS = {
    "BeliefEncodingSchema": "manabot.belief.encoding",
    "BeliefError": "manabot.belief.range",
    "BeliefRow": "manabot.belief.encoding",
    "BeliefState": "manabot.belief.range",
    "CompatibleDealBeliefModel": "manabot.belief.state",
    "EmptyBeliefSupport": "manabot.belief.state",
    "HAND_ZONE_ID": "manabot.belief.encoding",
    "LIBRARY_ZONE_ID": "manabot.belief.encoding",
    "Manabot": "manabot.belief.agent",
    "OPPONENT_OWNER_ROLE_ID": "manabot.belief.encoding",
    "ViewerDecision": "manabot.belief.agent",
    "ViewerHistory": "manabot.belief.state",
    "condition_belief": "manabot.belief.state",
    "encode_belief": "manabot.belief.encoding",
    "query_mass": "manabot.belief.state",
}


def __getattr__(name: str):
    """Load optional model/runtime dependencies only when requested."""

    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
