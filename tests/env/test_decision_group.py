"""Python API tests for the unified Env.decision / preview / submit surface
and 0.3.0 deprecation warnings on legacy structured/combat wrappers."""

import json
import warnings

import pytest

import managym
from managym.combat import CombatPlan, preview_combat
from managym.decision_group import (
    Decision,
    DecisionAddress,
    DeclareAttackersChoice,
    DeclareAttackersDecision,
    OrderLibraryChoice,
    OrderLibraryDecision,
    PartitionPilesChoice,
    PartitionPilesDecision,
    PartitionPilesOutcome,
    ScryChoice,
    SingleDecision,
    SingleOutcome,
    SurveilChoice,
    SurveilDecision,
)


def _setup_env(
    deck_p0: dict[str, int],
    deck_p1: dict[str, int],
    *,
    seed: int = 7,
    player_policies: tuple[str, str] | None = None,
) -> managym.Env:
    env = managym.Env(seed=seed, skip_trivial=True, player_policies=player_policies)
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("P0", deck_p0),
            managym.PlayerConfig("P1", deck_p1),
        ]
    )
    env.reset(setup)
    return env


def _keep_opening_hands(env: managym.Env) -> None:
    while True:
        d = env.decision()
        assert d is not None
        keep = next((o for o in d.offers if "Keep" in o.label), None)
        if keep is None:
            break
        env.submit(keep.id)


def _advance_to_decision(
    env: managym.Env,
    target_kind: str,
    *,
    max_steps: int = 400,
) -> Decision:
    for _ in range(max_steps):
        if env.is_game_over():
            break
        active = env.decision()
        assert active is not None
        if active.kind == target_kind:
            return active
        chosen = 0
        if active.address.viewer == 0:
            for offer in active.offers:
                if "Keep" in offer.label:
                    chosen = offer.id
                    break
            else:
                for offer in active.offers:
                    if offer.verb in ("play_land", "cast", "pay_cost"):
                        chosen = offer.id
                        break
                else:
                    for offer in active.offers:
                        if offer.verb == "pass_priority":
                            chosen = offer.id
                            break
        else:
            for offer in active.offers:
                if offer.verb == "pass_priority" or "Keep" in offer.label:
                    chosen = offer.id
                    break
        env.submit(chosen)
    raise AssertionError(f"did not reach decision kind {target_kind!r}")


def test_unified_decision_single_and_combat_submit_match_manual_step() -> None:
    env = _setup_env(
        {"Mountain": 20, "Gray Ogre": 20},
        {"Forest": 40},
        seed=19,
    )
    # Initial decision is a SingleDecision with offers and bound address.
    first = env.decision()
    assert isinstance(first, SingleDecision)
    assert first.address.viewer == env.current_agent_index()
    assert env.decision(viewer=1 - first.address.viewer) is None
    assert len(first.offers) >= 1

    # Single offer preview and submit match env.step(offer_id).
    single_preview = env.preview(first.offers[0].id)
    assert single_preview.complete is True
    assert single_preview.refused is None
    assert single_preview.outcome == SingleOutcome(offer_id=first.offers[0].id)

    # Grouped choice on a SingleDecision raises AgentError without mutating.
    digest_before = env.state_digest()
    with pytest.raises(managym.AgentError):
        env.preview(DeclareAttackersChoice())
    assert env.state_digest() == digest_before

    active = _advance_to_decision(env, "declare_attackers")
    assert isinstance(active, DeclareAttackersDecision)
    assert len(active.declarable) >= 1
    assert len(active.offers) >= 1
    assert env.decision(viewer=1 - active.address.viewer) is None

    # Wrong choice kind and stale explicit `at` error without mutating.
    digest_before = env.state_digest()
    with pytest.raises(managym.AgentError):
        env.preview(ScryChoice(), at=active.address)
    stale_at = DecisionAddress(
        viewer=active.address.viewer,
        revision=active.address.revision + 1,
        fingerprint=active.address.fingerprint,
    )
    with pytest.raises(managym.AgentError):
        env.submit(
            DeclareAttackersChoice(attackers=[active.declarable[0].item]),
            at=stale_at,
        )
    assert env.state_digest() == digest_before

    # Preview is read-only; default at=None matches explicit at=active.address.
    choice = DeclareAttackersChoice(attackers=[c.item for c in active.declarable])
    preview = env.preview(choice)
    assert preview == env.preview(choice, at=active.address)
    assert env.state_digest() == digest_before
    assert preview.complete is True
    assert preview.refused is None

    manual = env.clone_env()
    manual_result = None
    for step in preview.steps:
        manual_result = manual.step(step.action_index)
    assert manual_result is not None

    submitted_obs, reward, done, trunc, _info = env.submit(choice, at=active.address)
    assert env.state_digest() == manual.state_digest()
    assert submitted_obs.agent.player_index == manual_result[0].agent.player_index
    assert (reward, done, trunc) == manual_result[1:4]


def test_mixed_policy_submit_settles_identically_to_step() -> None:
    env = _setup_env(
        {"Mountain": 20, "Gray Ogre": 20},
        {"Forest": 20, "Llanowar Elves": 20},
        seed=19,
        player_policies=("agent", "heuristic"),
    )
    _keep_opening_hands(env)
    env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    env.scenario_force_battlefield(1, "Llanowar Elves", ready=True)
    env.scenario_force_battlefield(1, "Llanowar Elves", ready=True)
    env.scenario_refresh()

    active = _advance_to_decision(env, "declare_attackers")
    assert isinstance(active, DeclareAttackersDecision)
    assert active.address.viewer == 0
    assert len(active.declarable) == 2

    choice = DeclareAttackersChoice(attackers=[c.item for c in active.declarable])
    preview = env.preview(choice)
    assert preview.complete is True
    assert len(preview.steps) == 2

    manual = env.clone_env()
    manual_out = None
    for step in preview.steps:
        manual_out = manual.step(step.action_index)
    assert manual_out is not None

    sub_obs, sub_reward, sub_done, sub_trunc, _ = env.submit(choice)
    assert env.state_digest() == manual.state_digest()
    assert sub_obs.agent.player_index == manual_out[0].agent.player_index
    assert (sub_reward, sub_done, sub_trunc) == manual_out[1:4]


def test_surveil_to_order_library_round_trip() -> None:
    env = _setup_env(
        {"Forest": 24, "Esper Origins": 16},
        {"Island": 40},
        seed=7,
    )
    active = _advance_to_decision(env, "surveil")
    assert isinstance(active, SurveilDecision)
    assert len(active.items) == 2

    # Keep both on top -> opens OrderLibrary Top with 2 cards.
    manual = env.clone_env()
    keep_choice = SurveilChoice(graveyard=[])
    surveil_preview = env.preview(keep_choice)
    assert surveil_preview.complete is True
    assert len(surveil_preview.steps) == 1
    assert surveil_preview.steps[0].item is None
    for step in surveil_preview.steps:
        manual.step(step.action_index)

    env.submit(keep_choice, at=active.address)
    assert env.state_digest() == manual.state_digest()

    order_active = env.decision()
    assert isinstance(order_active, OrderLibraryDecision)
    assert order_active.end == "top"
    assert len(order_active.items) == 2
    assert order_active.committed == []

    # Incomplete OrderLibrary choice returns complete=False and submit refuses without mutating.
    digest_before = env.state_digest()
    incomplete = env.preview(OrderLibraryChoice(order=[order_active.items[1].item]))
    assert incomplete.complete is False
    assert incomplete.refused is None
    assert incomplete.steps == []
    with pytest.raises(managym.AgentError):
        env.submit(OrderLibraryChoice(order=[order_active.items[1].item]))
    assert env.state_digest() == digest_before

    # Complete OrderLibrary choice matches manual stepping.
    full_order = OrderLibraryChoice(
        order=[order_active.items[1].item, order_active.items[0].item]
    )
    order_preview = env.preview(full_order)
    assert order_preview.complete is True
    for step in order_preview.steps:
        manual.step(step.action_index)
    env.submit(full_order, at=order_active.address)
    assert env.state_digest() == manual.state_digest()


def test_deprecated_structured_and_combat_preview_apis_warn_for_0_3_0() -> None:
    env = _setup_env(
        {"Mountain": 20, "Gray Ogre": 20},
        {"Forest": 40},
        seed=19,
    )
    # Advance to priority so structured_offers / compound_offers / step_structured work.
    _keep_opening_hands(env)

    for label, fn in (
        ("structured_offers", lambda: env.structured_offers()),
        ("compound_offers", lambda: env.compound_offers()),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            offers = fn()
        assert any(
            issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
            for w in caught
        ), f"{label} did not emit 0.3.0 DeprecationWarning"
        projection = json.loads(offers.projection_json())
        pass_offer = next(
            o for o in projection["offers"] if o["verb"] == "pass_priority"
        )
        submission = json.dumps({"offer_id": pass_offer["id"], "answers": []})
        if label == "compound_offers":
            with warnings.catch_warnings(record=True) as caught_cmd:
                warnings.simplefilter("always")
                env.compound_commands_json(offers, submission)
            assert any(
                issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
                for w in caught_cmd
            )
        else:
            clone = env.clone_env()
            with warnings.catch_warnings(record=True) as caught_step:
                warnings.simplefilter("always")
                clone.step_structured(offers, submission)
            assert any(
                issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
                for w in caught_step
            )

    # Deprecated preview_combat emits DeprecationWarning targeting 0.3.0.
    active = _advance_to_decision(env, "declare_attackers")
    assert isinstance(active, DeclareAttackersDecision)
    with warnings.catch_warnings(record=True) as caught_combat:
        warnings.simplefilter("always")
        legacy = preview_combat(
            env,
            viewer=active.address.viewer,
            revision=active.address.revision,
            fingerprint=active.address.fingerprint,
            plan=CombatPlan(attackers=[active.declarable[0].item]),
        )
    assert any(
        issubclass(w.category, DeprecationWarning) and "0.3.0" in str(w.message)
        for w in caught_combat
    )
    assert legacy.refused is None


def test_single_decision_offers_have_readable_labels_sources_and_stable_fingerprint() -> (
    None
):
    env = _setup_env(
        {"Mountain": 10, "Island": 10, "Gray Ogre": 10, "Hill Giant": 10},
        {"Forest": 40},
        seed=7,
    )
    _keep_opening_hands(env)
    for _ in range(4):
        env.scenario_force_battlefield(0, "Mountain", ready=True)
    env.scenario_clear_hand(0)
    for name in ("Mountain", "Island", "Gray Ogre", "Hill Giant"):
        env.scenario_force_card_in_hand(0, name)
    env.scenario_refresh()

    raw_frame = json.loads(env.semantic_decision_frame_json())
    decision = env.decision()
    assert isinstance(decision, SingleDecision)
    assert decision.address.fingerprint == raw_frame["fingerprint"]
    assert any(
        o["label"] == "PriorityPlayLand" and o["source"] is None
        for o in raw_frame["offers"]
    )

    by_label = {offer.label: offer for offer in decision.offers}
    for label in ("Play Mountain", "Play Island", "Cast Gray Ogre", "Cast Hill Giant"):
        assert label in by_label, sorted(by_label)
        src = by_label[label].source
        assert src is not None and src.kind == "object"
    assert by_label["Play Mountain"].source != by_label["Play Island"].source
    assert by_label["Cast Gray Ogre"].source != by_label["Cast Hill Giant"].source
    assert by_label["Pass priority"].source is None


def test_partition_piles_round_trip_and_choose_pile_transition() -> None:
    env = _setup_env(
        {"Island": 24, "Curator of Destinies": 16},
        {"Forest": 40},
        seed=13,
    )
    _keep_opening_hands(env)
    for _ in range(6):
        env.scenario_force_battlefield(0, "Island", ready=True)
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Curator of Destinies")
    env.scenario_refresh()

    active = _advance_to_decision(env, "partition_piles", max_steps=60)
    assert isinstance(active, PartitionPilesDecision)
    assert active.address.viewer == 0
    assert len(active.items) == 5
    assert active.committed_face_up == []
    assert active.committed_face_down == []
    assert env.decision(viewer=1) is None

    # Duplicate card refuses without mutating.
    digest_before = env.state_digest()
    dup = env.preview(
        PartitionPilesChoice(face_up=[active.items[0].item, active.items[0].item]),
        at=active.address,
    )
    assert dup.complete is False
    assert dup.refused is not None
    with pytest.raises(managym.AgentError):
        env.submit(
            PartitionPilesChoice(face_up=[active.items[0].item, active.items[0].item]),
            at=active.address,
        )
    assert env.state_digest() == digest_before

    # Full 5-card split matches manual step-by-step execution and hands off to ChoosePile.
    choice = PartitionPilesChoice(face_up=[active.items[0].item, active.items[2].item])
    preview = env.preview(choice, at=active.address)
    assert preview.complete is True
    assert preview.refused is None
    assert len(preview.steps) == 5
    assert preview.outcome == PartitionPilesOutcome(
        face_up=[active.items[0].item, active.items[2].item],
        face_down=[active.items[1].item, active.items[3].item, active.items[4].item],
    )
    assert env.state_digest() == digest_before

    manual = env.clone_env()
    for step in preview.steps:
        manual.step(step.action_index)
    env.submit(choice, at=active.address)
    assert env.state_digest() == manual.state_digest()

    next_decision = env.decision(viewer=1)
    assert isinstance(next_decision, SingleDecision)
    assert any("face-up pile" in o.label for o in next_decision.offers)


def test_decision_surface_and_deprecated_structured_search_offers_alias() -> None:
    env = _setup_env({"Mountain": 40}, {"Forest": 40}, seed=5)
    _keep_opening_hands(env)

    assert managym.StructuredOfferSet is managym.DecisionSurface
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surface = env.decision_surface()
    assert isinstance(surface, managym.DecisionSurface)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    with pytest.warns(DeprecationWarning, match="0.3.0"):
        legacy = env.structured_search_offers()
    assert legacy.projection_json() == surface.projection_json()
