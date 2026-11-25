"""Python API tests for managym.group and deprecated managym.combat wrapper."""

import warnings

import pytest

import managym
from managym.combat import CombatPlan, preview_combat
from managym.decision import DecisionFrame
from managym.group import (
    DeclareAttackersGroup,
    DeclareAttackersPlan,
    OrderLibraryGroup,
    OrderLibraryPlan,
    ScryPlan,
    SurveilGroup,
    SurveilPlan,
    apply_group,
    group,
    preview_group,
)


def _setup_env(deck_p0: dict[str, int], deck_p1: dict[str, int], seed: int = 7) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True)
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("P0", deck_p0),
            managym.PlayerConfig("P1", deck_p1),
        ]
    )
    env.reset(setup)
    return env


def _advance_to_group(
    env: managym.Env,
    target_kind: str,
    *,
    play_lands_and_cast: tuple[str, ...] = (),
    max_steps: int = 400,
) -> tuple[int, object]:
    for _ in range(max_steps):
        if env.is_game_over():
            break
        viewer = env.current_agent_index()
        assert viewer is not None
        active = group(env, viewer=viewer)
        if active is not None and active.kind == target_kind:
            return viewer, active
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        chosen = 0
        if viewer == 0:
            # Keep opening hand, play lands, cast named spells, otherwise pay/pass/first legal.
            for idx, offer in enumerate(frame.offers):
                if offer.get("kind") == "play_land":
                    chosen = idx
                    break
            else:
                for idx, offer in enumerate(frame.offers):
                    label = str(offer.get("label", ""))
                    if offer.get("kind") == "cast" and any(
                        name in label for name in play_lands_and_cast
                    ):
                        chosen = idx
                        break
                else:
                    for idx, offer in enumerate(frame.offers):
                        kind = offer.get("kind")
                        label = str(offer.get("label", ""))
                        if kind in ("pay_cost", "pass_priority") or "Keep" in label:
                            chosen = idx
                            break
        else:
            for idx, offer in enumerate(frame.offers):
                if offer.get("kind") == "pass_priority" or "Keep" in str(
                    offer.get("label", "")
                ):
                    chosen = idx
                    break
        env.step(chosen)
    raise AssertionError(f"did not reach group kind {target_kind!r}")


def test_group_combat_apply_matches_manual_and_combat_preview_warns() -> None:
    env = _setup_env(
        {"Mountain": 20, "Gray Ogre": 20},
        {"Forest": 40},
        seed=19,
    )
    viewer, active = _advance_to_group(
        env, "declare_attackers", play_lands_and_cast=("Gray Ogre",)
    )
    assert isinstance(active, DeclareAttackersGroup)
    assert len(active.declarable) >= 1
    assert group(env, viewer=1 - viewer) is None

    # Deprecated preview_combat emits DeprecationWarning targeting 0.3.0.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = preview_combat(
            env,
            viewer=viewer,
            revision=active.revision,
            fingerprint=active.fingerprint,
            plan=CombatPlan(attackers=[active.declarable[0].item]),
        )
    assert any(
        issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
        for w in caught
    )
    assert legacy.refused is None

    # Wrong plan kind and stale revision error without mutating.
    digest_before = env.state_digest()
    with pytest.raises(managym.AgentError):
        preview_group(
            env,
            viewer=viewer,
            revision=active.revision,
            fingerprint=active.fingerprint,
            plan=ScryPlan(),
        )
    with pytest.raises(managym.AgentError):
        apply_group(
            env,
            viewer=viewer,
            revision=active.revision + 1,
            fingerprint=active.fingerprint,
            plan=DeclareAttackersPlan(attackers=[active.declarable[0].item]),
        )
    assert env.state_digest() == digest_before

    # Preview is read-only; apply_group matches stepping preview.steps manually.
    plan = DeclareAttackersPlan(attackers=[c.item for c in active.declarable])
    preview = preview_group(
        env,
        viewer=viewer,
        revision=active.revision,
        fingerprint=active.fingerprint,
        plan=plan,
    )
    assert env.state_digest() == digest_before
    assert preview.complete is True
    assert preview.refused is None

    manual = env.clone_env()
    for step in preview.steps:
        manual.step(step.action_index)

    applied = apply_group(
        env,
        viewer=viewer,
        revision=active.revision,
        fingerprint=active.fingerprint,
        plan=plan,
    )
    assert applied == preview
    assert env.state_digest() == manual.state_digest()


def test_group_surveil_to_order_library_python_round_trip() -> None:
    env = _setup_env(
        {"Forest": 24, "Esper Origins": 16},
        {"Island": 40},
        seed=7,
    )
    viewer, active = _advance_to_group(
        env, "surveil", play_lands_and_cast=("Esper Origins",)
    )
    assert isinstance(active, SurveilGroup)
    assert len(active.items) == 2

    # Keep both on top -> opens OrderLibrary Top with 2 cards.
    manual = env.clone_env()
    keep_plan = SurveilPlan(graveyard=[])
    surveil_preview = preview_group(
        env,
        viewer=viewer,
        revision=active.revision,
        fingerprint=active.fingerprint,
        plan=keep_plan,
    )
    assert surveil_preview.complete is True
    assert len(surveil_preview.steps) == 1
    assert surveil_preview.steps[0].item is None
    for step in surveil_preview.steps:
        manual.step(step.action_index)

    apply_group(
        env,
        viewer=viewer,
        revision=active.revision,
        fingerprint=active.fingerprint,
        plan=keep_plan,
    )
    assert env.state_digest() == manual.state_digest()

    order_active = group(env, viewer=viewer)
    assert isinstance(order_active, OrderLibraryGroup)
    assert order_active.end == "top"
    assert len(order_active.items) == 2

    # Incomplete OrderLibrary plan returns complete=False and apply_group refuses without mutating.
    digest_before = env.state_digest()
    incomplete = preview_group(
        env,
        viewer=viewer,
        revision=order_active.revision,
        fingerprint=order_active.fingerprint,
        plan=OrderLibraryPlan(order=[order_active.items[1].item]),
    )
    assert incomplete.complete is False
    assert incomplete.refused is None
    assert incomplete.steps == []
    with pytest.raises(managym.AgentError):
        apply_group(
            env,
            viewer=viewer,
            revision=order_active.revision,
            fingerprint=order_active.fingerprint,
            plan=OrderLibraryPlan(order=[order_active.items[1].item]),
        )
    assert env.state_digest() == digest_before

    # Complete OrderLibrary plan matches manual stepping.
    full_order = OrderLibraryPlan(
        order=[order_active.items[1].item, order_active.items[0].item]
    )
    order_preview = preview_group(
        env,
        viewer=viewer,
        revision=order_active.revision,
        fingerprint=order_active.fingerprint,
        plan=full_order,
    )
    assert order_preview.complete is True
    for step in order_preview.steps:
        manual.step(step.action_index)
    apply_group(
        env,
        viewer=viewer,
        revision=order_active.revision,
        fingerprint=order_active.fingerprint,
        plan=full_order,
    )
    assert env.state_digest() == manual.state_digest()
