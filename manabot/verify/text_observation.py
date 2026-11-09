"""Local representation proof with native Commands and a response stub.

Run `uv run python -m manabot.verify.text_observation --out .runs/text-observation`.
Scenario helpers establish declared priority roots only. Every subsequent choice
uses ordinary native Commands. Retained JSON/text is evidence, not model memory.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from managym.decision import Observation, SemanticContractError, apply_semantic_command
from managym.decision_capture import CardReference, DecisionInput

from manabot.semantic.decision_contract import SemanticDecisionContract
from manabot.semantic.text_observation import (
    ResponseProtocolError,
    TextDecision,
    capture_text_decision,
    render_decision,
    resolve_action,
    submit_response,
)
import managym


def priority_root(
    hand: tuple[str, ...],
    battlefield: tuple[str, ...],
    *,
    opponent: tuple[str, ...] = (),
) -> managym.Env:
    """Seed 23, 20 of each named card per deck; inject at first-main priority."""
    # Diagnostic exception: the collision proof needs sole-offer continuations
    # (equal state, distinct pending history) to stay visible. No game is
    # played for a model or policy from these roots.
    env = managym.Env(seed=23, skip_trivial=False)
    deck = {card: 20 for card in sorted(set(hand + battlefield + opponent))}
    env.reset([managym.PlayerConfig(name, deck) for name in ("a", "b")])
    for _ in range(20):
        contract = SemanticDecisionContract.from_env(env)
        observation = Observation.from_json(
            env.semantic_observation_json(contract.frame.actor)
        )
        if observation.viewer_state["turn"]["phase"] == 1:
            break
        contract.apply(env, contract.frame.find_verb("pass_priority")["id"])
    else:
        raise AssertionError("main phase not reached")
    actor = contract.frame.actor
    env.scenario_clear_hand(actor)
    for card in hand:
        env.scenario_force_card_in_hand(actor, card)
    for card in battlefield:
        env.scenario_force_battlefield(actor, card, ready=True)
    for card in opponent:
        env.scenario_force_battlefield(1 - actor, card, ready=True)
    env.scenario_refresh()
    return env


def _choose(env: managym.Env, verb: str, *, focus_name: str | None = None) -> None:
    decision = capture_text_decision(env, history_capacity=32)
    source = DecisionInput.model_validate_json(decision.capture_json)
    assert source.observation.decision is not None
    for offer, text_action in zip(
        source.observation.decision.offers,
        decision.actions,
        strict=True,
    ):
        if offer["verb"] == verb and (
            focus_name is None or focus_name in text_action.description
        ):
            submit_response(env, decision, text_action.reference)
            return
    raise AssertionError(f"missing {verb} {focus_name}: {decision.text[-1600:]}")


def collision_branches() -> tuple[managym.Env, managym.Env]:
    env = priority_root(
        ("Deadly Precision",), ("Swamp", "Gray Ogre"), opponent=("Gray Ogre",)
    )
    _choose(env, "cast")
    target = capture_text_decision(env, history_capacity=32)
    branches: list[managym.Env] = []
    assert len(target.actions) == 2
    for action in target.actions:
        branch = env.clone_env()
        branch_decision = capture_text_decision(branch, history_capacity=32)
        submit_response(branch, branch_decision, action.reference)
        branches.append(branch)
    return branches[0], branches[1]


def native_fixtures() -> dict[str, managym.Env]:
    priority = priority_root(
        ("Lightning Bolt", "Lightning Bolt"),
        ("Mountain", "Mountain", "Gray Ogre", "Gray Ogre"),
        opponent=("Gray Ogre",),
    )
    target = priority.clone_env()
    _choose(target, "cast")
    combat = priority.clone_env()
    for _ in range(20):
        capture = capture_text_decision(combat, history_capacity=32)
        if (
            DecisionInput.model_validate_json(
                capture.capture_json
            ).observation.viewer_state["action_space"]["type"]
            == 2
        ):
            break
        _choose(combat, "pass_priority")
    else:
        raise AssertionError("combat not reached")
    blockers = combat.clone_env()
    for _ in range(20):
        decision = capture_text_decision(blockers, history_capacity=32)
        state = DecisionInput.model_validate_json(
            decision.capture_json
        ).observation.viewer_state
        if state["action_space"]["type"] == 3:
            break
        if state["action_space"]["type"] == 2:
            _choose(blockers, "declare_attacker", focus_name="Attack with ")
        else:
            _choose(blockers, "pass_priority")
    else:
        raise AssertionError("blocker assignment not reached")
    selection = priority_root(("Accumulate Wisdom",), ("Island", "Island"))
    _choose(selection, "cast")
    for _ in range(20):
        capture = capture_text_decision(selection, history_capacity=32)
        source = DecisionInput.model_validate_json(capture.capture_json)
        if source.observation.viewer_state["action_space"]["type"] == 6:
            break
        assert source.observation.decision is not None
        verbs = [offer["verb"] for offer in source.observation.decision.offers]
        _choose(selection, "pay_cost" if "pay_cost" in verbs else "pass_priority")
    else:
        raise AssertionError("selection not reached")
    mana_payment = priority_root(
        ("Lightning Bolt",), ("Island", "Prophetic Prism", "Prophetic Prism")
    )
    _choose(mana_payment, "cast")
    _choose(mana_payment, "choose_target")
    mana_color = mana_payment.clone_env()
    _choose(mana_color, "activate_ability")
    alternative = priority_root(
        ("Deadly Precision",), ("Swamp",) * 5 + ("Gray Ogre", "Gray Ogre")
    )
    _choose(alternative, "cast")
    _choose(alternative, "choose_target")
    mode = priority_root(
        ("Azula Always Lies",), ("Swamp", "Swamp", "Gray Ogre"), opponent=("Gray Ogre",)
    )
    _choose(mode, "cast")
    optional_payment = priority_root(
        ("Firebending Lesson",), ("Mountain",) * 5, opponent=("Gray Ogre",)
    )
    _choose(optional_payment, "cast")
    left, right = collision_branches()
    return {
        "priority": priority,
        "target": target,
        "combat": combat,
        "blockers": blockers,
        "selection": selection,
        "payment_left": left,
        "payment_right": right,
        "mana_payment": mana_payment,
        "mana_color": mana_color,
        "alternative_cost": alternative,
        "mode": mode,
        "optional_payment": optional_payment,
    }


def history_branches() -> tuple[managym.Env, managym.Env]:
    """Swap the order of two public targets; identical final state, distinct history."""
    root = priority_root(("Lightning Bolt", "Lightning Bolt"), ("Mountain", "Mountain"))
    branches: list[managym.Env] = []
    for order in (("You", "Opponent"), ("Opponent", "You")):
        env = root.clone_env()
        for target in order:
            _choose(env, "cast")
            _choose(env, "choose_target", focus_name=f"Target {target}")
            for _ in range(12):
                current = DecisionInput.model_validate_json(
                    capture_text_decision(env, history_capacity=32).capture_json
                )
                if (
                    not current.observation.viewer_state["stack_objects"]
                    and current.continuation.stage == "priority"
                ):
                    break
                verbs = [offer["verb"] for offer in current.observation.decision.offers]
                _choose(env, "pay_cost" if "pay_cost" in verbs else "pass_priority")
            else:
                raise AssertionError("spell did not resolve")
        branches.append(env)
    return branches[0], branches[1]


def _retain(path: Path, decision: TextDecision) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "input.json").write_text(decision.capture_json + "\n")
    (path / "prompt.txt").write_text(decision.text)
    (path / "bindings.json").write_text(
        json.dumps(
            {
                "representation_version": decision.representation_version,
                "capture_digest": decision.capture_digest,
                "actions": [asdict(action) for action in decision.actions],
                "origin": "live native handle; deliberately not reconstructible from retained JSON",
                "memory": "public policy history v1, capacity 32; no writable model memory",
                "selection_source": "response stub; no model loaded",
            },
            indent=2,
        )
        + "\n"
    )


def prove_fixture(env: managym.Env, path: Path) -> int:
    decision = capture_text_decision(env, history_capacity=32)
    assert render_decision(decision.capture_json).text == decision.text
    _retain(path, decision)
    checks: list[dict[str, str]] = []
    for response in ("", "A", "A0 A1", "A999999", "A0 because", "```A0```", "A00"):
        before = env.state_digest()
        try:
            submit_response(env, decision, response)
        except ResponseProtocolError as error:
            checks.append({"response": response, "error": str(error)})
        else:
            raise AssertionError("malformed answer accepted")
        assert env.state_digest() == before
    foreign = env.clone_env()
    before = foreign.state_digest()
    try:
        submit_response(foreign, decision, decision.actions[0].reference)
    except SemanticContractError as error:
        assert error.code == "wrong_origin"
        checks.append({"check": "cross-branch", "error": str(error)})
    else:
        raise AssertionError("foreign capture accepted")
    assert foreign.state_digest() == before
    receipts: list[dict[str, object]] = []
    for action in decision.actions:
        branch, replay = env.clone_env(), env.clone_env()
        local = capture_text_decision(branch, history_capacity=32)
        response = f"  {action.reference}\n"
        result = submit_response(branch, local, response)
        replayed = apply_semantic_command(replay, resolve_action(local, response))
        assert replayed == result
        assert branch.state_digest() == replay.state_digest()
        before = branch.state_digest()
        try:
            submit_response(branch, local, response)
        except SemanticContractError as error:
            assert error.code == "stale_capture"
        else:
            raise AssertionError("stale answer accepted")
        assert branch.state_digest() == before
        receipts.append(
            {
                "response": response,
                "origin": "initial",
                "producer": "response_stub",
                "command": json.loads(action.command.to_json()),
                "transition": asdict(result),
                "replay_matches": True,
                "stale_rejected_without_mutation": True,
            }
        )
    (path / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (path / "executions.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return len(receipts)


def run_demo(out: Path) -> None:
    fixtures = native_fixtures()
    totals = {name: prove_fixture(env, out / name) for name, env in fixtures.items()}
    left = capture_text_decision(fixtures["payment_left"], history_capacity=32)
    right = capture_text_decision(fixtures["payment_right"], history_capacity=32)
    a, b = [
        DecisionInput.model_validate_json(decision.capture_json)
        for decision in (left, right)
    ]
    assert a.observation == b.observation
    assert a.continuation.targets != b.continuation.targets
    assert left.capture_digest != right.capture_digest and left.text != right.text
    first, second = history_branches()
    histories = [
        capture_text_decision(env, history_capacity=32) for env in (first, second)
    ]
    x, y = [
        DecisionInput.model_validate_json(decision.capture_json)
        for decision in histories
    ]
    assert x.observation == y.observation
    assert x.history != y.history and histories[0].text != histories[1].text
    assert capture_text_decision(first).text == capture_text_decision(second).text
    assert any(
        isinstance(ref, CardReference) and ref.context_id is None
        for row in x.history.rows
        for ref in (row.source, row.target)
    )
    for name, decision in zip(
        ("history_left", "history_right"), histories, strict=True
    ):
        _retain(out / name, decision)
    summary = {
        "legal_alternatives_executed_and_replayed": totals,
        "schema_6_collision_preserved": True,
        "captured_continuations_and_text_distinct": True,
        "equal_current_state_distinct_history": True,
        "expired_history_links_explicit": True,
        "history": "public v1 / 32",
        "model": "response stub",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".runs/text-observation"))
    args = parser.parse_args()
    run_demo(args.out)


if __name__ == "__main__":
    main()
