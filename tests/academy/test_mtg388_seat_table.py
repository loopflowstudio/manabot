"""Tests for MTG-388: Policy rename, callable/file policies, top-level CLI, MTG-399 fit, and partial kept-deck imports."""

from __future__ import annotations

from pathlib import Path
import warnings

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import cli, launch, server
from managym.academy.opponent import set_policy_registry
from managym.policy import (
    Policy,
    PolicyDescriptor,
    PolicyRegistry,
    as_policy,
    is_python_policy_ref,
    load_python_policy,
)
from managym.settings_fit import (
    compare_settings,
    distributions_from_counts,
)


def test_policy_factory_alias_warns_targeting_0_3_0() -> None:
    import managym.policy as policy_mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = getattr(policy_mod, "PolicyFactory")
    assert alias is Policy
    assert any(
        issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
        for w in caught
    )


def test_callable_policy_and_file_loader_play_legal_moves(tmp_path: Path) -> None:
    bot_file = tmp_path / "my_bot.py"
    bot_file.write_text(
        "def choose(obs):\n"
        "    return 0\n"
        "\n"
        "def prob_choose(obs):\n"
        "    n = len(obs.action_space.actions)\n"
        "    return [1.0 / n] * n\n",
        encoding="utf-8",
    )
    assert is_python_policy_ref(f"{bot_file}:choose")
    assert is_python_policy_ref(str(bot_file))

    first_policy = load_python_policy(f"{bot_file}:choose")
    prob_policy = load_python_policy(f"{bot_file}:prob_choose")
    default_policy = load_python_policy(str(bot_file))
    inline_policy = as_policy(lambda obs: 0, name="inline_zero")

    reg = PolicyRegistry(providers={})
    id1 = reg.register(first_policy)
    id2 = reg.register(prob_policy)
    id3 = reg.register(inline_policy)
    assert len({id1, id2, id3}) == 3
    assert default_policy.describe().name == "my_bot"

    setup = [
        managym.authored_deck_setup("two_deck", "ur_lessons"),
        managym.authored_deck_setup("two_deck", "gw_allies"),
    ]
    runner = reg.load(id2, setup, seat=0, seed=1)
    try:
        env = managym.Env(skip_trivial=False)
        obs, _ = env.reset(managym.GameSetup(setup, decklists="open"))
        action = runner(env, obs)
        assert 0 <= action < len(obs.action_space.actions)
        assert runner.last_decision is not None
        assert len(runner.last_decision.probabilities) == len(obs.action_space.actions)
    finally:
        runner.close()


def test_settings_fit_informs_never_refuses_and_marks_unknown() -> None:
    trained = distributions_from_counts(
        {"decklists:closed": 266321, "skip_trivial:true": 266321}
    )
    open_report = compare_settings(
        trained, {"decklists": "open", "mana_payment": "manual"}
    )
    assert open_report.status == "mismatch"
    assert len(open_report.warnings) == 1
    assert "closed" in open_report.warnings[0].phrase()
    assert len(open_report.unknown) == 1
    assert open_report.unknown[0].setting == "mana_payment"

    closed_report = compare_settings(trained, {"decklists": "closed"})
    assert closed_report.status == "match"
    assert closed_report.warnings == ()

    empty_report = compare_settings({}, {"decklists": "closed"})
    assert empty_report.status == "unknown"


def test_policies_check_returns_informational_fit_and_partial_deck_import_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TaggedPolicy:
        def describe(self) -> PolicyDescriptor:
            return PolicyDescriptor(
                name="TLA Closed",
                provider="test",
                runner="test-runner-v1",
                artifacts={"actor": "a" * 64},
                alias="tla-closed",
                recommended=True,
                summary="78.2% vs Heuristic",
                trained_settings={"decklists:closed": 1000},
            )

        def admit(self, setup: list[managym.PlayerConfig]) -> None:
            del setup

        def load(self, *, seat: int, seed: int):  # type: ignore[no-untyped-def]
            return as_policy(lambda obs: 0).load(seat=seat, seed=seed)

    monkeypatch.setattr(server.trace_store, "TRACES_DIR", tmp_path)
    reg = PolicyRegistry(providers={})
    key = reg.register(_TaggedPolicy())
    set_policy_registry(reg)
    try:
        client = TestClient(server.app)
        cat = client.get("/api/opponents").json()
        assert cat["default_opponent"] == "tla-closed"
        assert any(b["id"] == "search-64" for b in cat["builtins"])

        resp = client.post(
            "/api/policies/check",
            json={
                "hero_deck": "ur_lessons",
                "villain_deck": "gw_allies",
                "decklists": "open",
                "villain_type": "policy",
                "opponent_sha256": key,
            },
        ).json()
        assert resp["compatible"] is True
        assert resp["fit"]["status"] == "mismatch"
        assert "closed" in resp["fit"]["summary"]

        # Partial one-side kept deck vs bundled deck on the other side resolves cleanly.
        ur_setup = managym.authored_deck_setup("two_deck", "ur_lessons")
        src = "\n".join(f"{c} {n}" for n, c in ur_setup.decklist.items())
        expanded = server.resolve_pool_decks(
            {
                "hero_deck": "kept:my-ur",
                "villain_deck": "gw_allies",
                "deck_import": {
                    "hero": {"source": src, "filename": "my-ur.txt", "name": "My UR"},
                },
            },
            seed=7,
        )
        assert expanded["deck_import"]["hero"]["name"] == "My UR"
        assert expanded["deck_import"]["villain"]["name"] == "GW Allies"
    finally:
        set_policy_registry(None)


def test_top_level_managym_opponents_cli_and_serve_discover_false(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["opponents", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"default_opponent"' in out
    assert '"search-64"' in out

    # serve(discover=False) ignores ambient providers and serves only explicit policies.
    with launch.serve(
        port=0,
        traces_dir=tmp_path,
        policies={"zero": lambda obs: 0},
        discover=False,
    ) as academy:
        opps = academy.opponents()
        assert [o["name"] for o in opps] == ["zero"]
        launched = academy.new_game(opponent="zero", order="draw")
        assert "/?launch=" in launched.url
