"""Viewer-safe beliefs over canonical managym possible worlds."""

from importlib import import_module

_EXPORTS = {
    "AgentMemory": "manabot.belief.agent",
    "AgentStep": "manabot.belief.agent",
    "BeliefEncodingSchema": "manabot.belief.encoding",
    "BeliefCheckpointBinding": "manabot.belief.encoding",
    "BeliefError": "manabot.belief.range",
    "BeliefModel": "manabot.belief.state",
    "BeliefRow": "manabot.belief.encoding",
    "BeliefState": "manabot.belief.range",
    "BeliefTensorView": "manabot.belief.encoding",
    "BeliefUpdate": "manabot.belief.state",
    "BeliefUpdateReceipt": "manabot.belief.state",
    "CompatibleDealBeliefModel": "manabot.belief.state",
    "EmptyBeliefSupport": "manabot.belief.state",
    "HAND_ZONE_ID": "manabot.belief.encoding",
    "LIBRARY_ZONE_ID": "manabot.belief.encoding",
    "Manabot": "manabot.belief.agent",
    "ManabotPlayer": "manabot.belief.runtime",
    "OPPONENT_OWNER_ROLE_ID": "manabot.belief.encoding",
    "PolicyValueResult": "manabot.belief.agent",
    "ViewerDecision": "manabot.belief.agent",
    "ViewerHistory": "manabot.belief.state",
    "belief_schema_from_engine": "manabot.belief.encoding",
    "condition_belief": "manabot.belief.state",
    "encode_belief": "manabot.belief.encoding",
    "query_mass": "manabot.belief.state",
    "viewer_decision_from_engine": "manabot.belief.runtime",
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
