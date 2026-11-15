"""Deprecated Search API spellings; removed in 0.4.0. Use managym.search."""

from __future__ import annotations

import warnings

from managym import search

_ALIASES = {
    "MirrorSearch": "SearchConfig",
    "MirrorSearchError": "SearchError",
    "mirror_unavailable": "search_unavailable",
    "descriptor_mirror_unavailable": "descriptor_search_unavailable",
    "mirror_descriptor": "search_descriptor",
    "mirror_source_identity": "search_source_identity",
    "mirror_decision": "search_decision",
    "NATIVE_PHYSICAL_BELIEF": "NATIVE_PHYSICAL_BELIEF",
    "MIRROR_METHOD": "MIRROR_METHOD",
}


def __getattr__(name: str) -> object:
    if name not in _ALIASES:
        raise AttributeError(name)
    replacement = _ALIASES[name]
    warnings.warn(
        f"managym.mirror_search.{name} is deprecated; use managym.search.{replacement} (removed in 0.4.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(search, replacement)
