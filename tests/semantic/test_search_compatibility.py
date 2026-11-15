"""Published search spellings retain one implementation during deprecation."""

import importlib

import pytest

from managym.bots import SearchSummary
from managym.search import SearchConfig, SearchError, search_decision
from tests.semantic.test_policy_search import IndependentSession, _runner


def test_published_module_names_warn_and_resolve_to_search() -> None:
    old = importlib.import_module("managym.mirror_search")
    for name, replacement in (
        ("MirrorSearch", SearchConfig),
        ("MirrorSearchError", SearchError),
        ("mirror_decision", search_decision),
    ):
        with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
            assert getattr(old, name) is replacement
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
        assert (
            getattr(importlib.import_module("managym.bots"), "MirrorSummary")
            is SearchSummary
        )


def test_runner_legacy_configuration_keeps_search_identity() -> None:
    runner = _runner(IndependentSession(), 0)
    spec = SearchConfig()
    try:
        with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
            runner.configure_mirror(spec, seed=7)
        legacy_identity = runner.descriptor.id
        assert runner.search == spec
        assert "+ search 1×1" in runner.descriptor.name
        runner.configure_search(spec, seed=7)
        assert runner.descriptor.id == legacy_identity
        with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
            assert runner.mirror == spec
        with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
            assert runner.mirror_runtime == runner.search_runtime
    finally:
        runner.close()
