"""Native effect fixtures shared by observation and model checks.

These roots resolve effects through viewer-bound Commands, retaining the native
source, duration and activation order needed by transport and model assertions.
"""

from managym.decision_capture import DecisionInput

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym


def source_effect_root() -> managym.Env:
    """Reach Ty Lee's source-dependent effect through a cast and target Command."""
    env = priority_root(
        ("Ty Lee, Chi Blocker",), ("Island",) * 3, opponent=("Gray Ogre",)
    )
    for step in range(12):
        obs = env.observation_for_player(env.current_agent_index())
        if any(p.effects for p in (*obs.agent_permanents, *obs.opponent_permanents)):
            return env
        decision = capture_text_decision(env)
        source = DecisionInput.model_validate_json(decision.capture_json)
        assert source.observation.decision is not None
        action = next(
            action
            for offer, action in zip(
                source.observation.decision.offers, decision.actions, strict=True
            )
            if (step == 0 and offer["verb"] == "cast")
            or (
                step > 0
                and (
                    offer["verb"] in ("pay_cost", "pass_priority")
                    or (
                        offer["verb"] == "choose_target"
                        and "Gray Ogre" in action.description
                    )
                )
            )
        )
        submit_response(env, decision, action.reference)
    raise AssertionError("Ty Lee did not resolve within the declared prefix")


def shared_effect_root(activations: int) -> managym.Env:
    """Resolve Drill activations through Commands in a bounded constructed root.

    Six activations fit v5; seven reproduce the 140-fact capacity failure.
    This is a new directed fixture, not a replay of the pilot's retained seed.
    """
    if not 1 <= activations <= 7:
        raise ValueError(
            "the shared-effect fixture admits one through seven activations"
        )
    env = priority_root(
        (),
        ("The Fire Nation Drill",) + ("Mountain",) * 7,
        opponent=("Gray Ogre",) * 10,
    )
    for activation in range(activations):
        decision = capture_text_decision(env)
        source = DecisionInput.model_validate_json(decision.capture_json)
        assert source.observation.decision is not None
        action = next(
            action
            for offer, action in zip(
                source.observation.decision.offers, decision.actions, strict=True
            )
            if offer["verb"] == "activate_ability"
            and "The Fire Nation Drill" in action.description
        )
        submit_response(env, decision, action.reference)
        for _ in range(8):
            obs = env.observation_for_player(env.current_agent_index())
            records = sum(
                len(p.effects)
                for p in (*obs.agent_permanents, *obs.opponent_permanents)
            )
            if records == 10 * (activation + 1):
                break
            decision = capture_text_decision(env)
            source = DecisionInput.model_validate_json(decision.capture_json)
            assert source.observation.decision is not None
            action = next(
                action
                for offer, action in zip(
                    source.observation.decision.offers, decision.actions, strict=True
                )
                if offer["verb"] in ("pay_cost", "pass_priority")
            )
            submit_response(env, decision, action.reference)
        else:
            raise AssertionError("Drill did not resolve within the declared prefix")
    return env


def kellan_upgrade_root(hand: tuple[str, ...] = ()) -> managym.Env:
    """Resolve Kellan's first upgrade through Commands in a bounded fixture."""
    env = priority_root(hand, ("Kellan, Planar Trailblazer",) + ("Mountain",) * 5)
    for step in range(16):
        obs = env.observation_for_player(env.current_agent_index())
        if any(r.effect.program for r in obs.shared_effects or []):
            return env
        decision = capture_text_decision(env)
        actions = obs.action_space.actions
        index = next(
            i
            for i, action in enumerate(actions)
            if (step == 0 and action.ability_index == 0)
            or (
                step > 0
                and (
                    "pass" in decision.actions[i].description.lower()
                    or "pay" in decision.actions[i].description.lower()
                )
            )
        )
        submit_response(env, decision, decision.actions[index].reference)
    raise AssertionError("Kellan upgrade did not resolve")


def kellan_trigger_root(hand: tuple[str, ...] = ()) -> managym.Env:
    """Reach the upgraded Kellan's captured combat trigger on the stack."""
    env = kellan_upgrade_root(hand)
    for _ in range(24):
        obs = env.observation_for_player(env.current_agent_index())
        if any(frame.program for frame in obs.continuation):
            return env
        decision = capture_text_decision(env)
        action = next(
            a
            for a in decision.actions
            if a.description.startswith(("Pass priority", "Attack with"))
        )
        submit_response(env, decision, action.reference)
    raise AssertionError("Kellan's combat trigger did not reach the stack")

