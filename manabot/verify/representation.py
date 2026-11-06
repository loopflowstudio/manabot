"""Native effect fixtures shared by observation and model checks.

These roots resolve effects through viewer-bound Commands, retaining the native
source, duration and activation order needed by transport and model assertions.
"""

from collections.abc import Callable
from typing import Literal

from managym.decision_capture import DecisionInput

from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import priority_root
import managym


def _submit_offer(env: managym.Env, wanted: Callable[[str, str], bool]) -> None:
    """Submit the first offer whose verb and description satisfy ``wanted``."""
    decision = capture_text_decision(env)
    source = DecisionInput.model_validate_json(decision.capture_json)
    assert source.observation.decision is not None
    action = next(
        action
        for offer, action in zip(
            source.observation.decision.offers, decision.actions, strict=True
        )
        if wanted(offer["verb"], action.description)
    )
    submit_response(env, decision, action.reference)


def source_effect_root() -> managym.Env:
    """Reach Ty Lee's source-dependent effect through a cast and target Command."""
    env = priority_root(
        ("Ty Lee, Chi Blocker",), ("Island",) * 3, opponent=("Gray Ogre",)
    )
    for step in range(12):
        obs = env.observation_for_player(env.current_agent_index())
        if any(p.effects for p in (*obs.agent_permanents, *obs.opponent_permanents)):
            return env
        _submit_offer(
            env,
            lambda verb, description: (
                verb == "cast"
                if step == 0
                else verb in ("pay_cost", "pass_priority")
                or (verb == "choose_target" and "Gray Ogre" in description)
            ),
        )
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
        _submit_offer(
            env,
            lambda verb, description: (
                verb == "activate_ability" and "The Fire Nation Drill" in description
            ),
        )
        for _ in range(8):
            obs = env.observation_for_player(env.current_agent_index())
            records = sum(
                len(p.effects)
                for p in (*obs.agent_permanents, *obs.opponent_permanents)
            )
            if records == 10 * (activation + 1):
                break
            _submit_offer(env, lambda verb, _: verb in ("pay_cost", "pass_priority"))
        else:
            raise AssertionError("Drill did not resolve within the declared prefix")
    return env


def _resolve_stack(
    env: managym.Env, target: Callable[[str], bool] = lambda _: False
) -> None:
    """Answer Commands until the announced object has left the stack."""
    for _ in range(12):
        if not env.observation_for_player(env.current_agent_index()).continuation:
            return
        _submit_offer(
            env,
            lambda verb, description: (
                verb in ("pay_cost", "pass_priority")
                or (verb == "choose_target" and target(description))
            ),
        )
    raise AssertionError("the stack did not empty within the declared prefix")


def aged_grant_root(
    duration: Literal["EndOfTurn", "Indefinite"], *, fill: bool = True
) -> managym.Env:
    """Age one grant past the public history window using only real Commands.

    An end-of-turn root casts Yip Yip! on Gray Ogre. An indefinite root activates
    Invasion Submersible, whose payload v7 can represent, and casts Lightning
    Bolt, which leaves the same hand because an activation records no history
    row. Eight Raging Goblins then add four rows each (stack arrival, cast,
    battlefield arrival, resolved) within the turn, so the end-of-turn record is
    still active when the 32-row window holds only their rows. ``fill=False``
    stops before them.
    """
    spell = "Yip Yip!" if duration == "EndOfTurn" else "Lightning Bolt"
    env = priority_root(
        (spell,) + ("Raging Goblin",) * 8,
        ("Invasion Submersible", "Gray Ogre", "Plains") + ("Mountain",) * 12,
    )
    if duration == "Indefinite":
        _submit_offer(
            env,
            lambda verb, description: (
                verb == "activate_ability" and "Invasion Submersible" in description
            ),
        )
        _resolve_stack(env)
    _submit_offer(env, lambda verb, text: verb == "cast" and spell in text)
    # Yip Yip! names the Ogre; the Bolt must leave both creatures in place.
    _resolve_stack(
        env,
        lambda text: (
            "Gray Ogre" in text
            if duration == "EndOfTurn"
            else "Ogre" not in text and "Submersible" not in text
        ),
    )
    for _ in range(8 if fill else 0):
        _submit_offer(env, lambda verb, text: verb == "cast" and "Goblin" in text)
        _resolve_stack(env)
    return env


def advance_to_next_turn(env: managym.Env) -> None:
    """Pass through cleanup without attacking, with no other choice made."""
    start = env.observation_for_player(0).turn.turn_number
    for _ in range(40):
        if env.observation_for_player(0).turn.turn_number != start:
            return
        _submit_offer(
            env,
            lambda verb, description: (
                verb == "pass_priority" or description.startswith("Do not attack")
            ),
        )
    raise AssertionError("the next turn did not begin within the declared prefix")


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
