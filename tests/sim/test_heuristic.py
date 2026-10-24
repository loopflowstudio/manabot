"""Rule-of-thumb heuristic anchor over synthetic viewer observations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from managym.enums import ActionEnum, ActionSpaceEnum, EventTypeEnum, StepEnum, ZoneEnum
import pytest

from manabot.env import Env
from manabot.eval.monitor import MonitorProtocol, _opponent_source_sha256
from manabot.sim.heuristic import (
    HeuristicPlayer,
    TargetedSpell,
    heuristic_source_paths,
    targeted_spells,
)
from manabot.sim.players import ScriptedGreedyPlayer

OWN, OPPONENT = 10, 20


def keywords(**on: bool) -> SimpleNamespace:
    names = (
        "flying",
        "reach",
        "first_strike",
        "double_strike",
        "deathtouch",
        "menace",
        "haste",
    )
    return SimpleNamespace(**{name: on.get(name, False) for name in names})


def creature(
    pid: int, controller: int, power: int, toughness: int, *, mv: int = 2, **flags: Any
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """A battlefield permanent and its visible card (registry key = id)."""
    permanent = SimpleNamespace(
        id=pid,
        registry_key=pid,
        controller_id=controller,
        tapped=flags.pop("tapped", False),
        damage=0,
        is_summoning_sick=flags.pop("sick", False),
        cant_be_blocked_this_turn=False,
        power=power,
        toughness=toughness,
        is_animated=False,
        keywords=keywords(**flags),
    )
    card = SimpleNamespace(
        id=1000 + pid,
        registry_key=pid,
        owner_id=controller,
        name=f"Creature {pid}",
        zone=ZoneEnum.BATTLEFIELD,
        kicker_cost=0,
        card_types=SimpleNamespace(is_creature=True, is_land=False),
        mana_cost=SimpleNamespace(mana_value=mv),
    )
    return permanent, card


def lands(
    count: int, *, tapped: bool = False
) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
    """Own lands with their visible cards."""
    out = []
    for offset in range(count):
        permanent, card = creature(80 + offset, OWN, 0, 0, tapped=tapped)
        card.card_types = SimpleNamespace(is_creature=False, is_land=True)
        out.append((permanent, card))
    return out


def hand(cid: int, name: str, mv: int, *, kicker: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        registry_key=cid,
        owner_id=OWN,
        name=name,
        zone=ZoneEnum.HAND,
        kicker_cost=kicker,
        card_types=SimpleNamespace(is_creature=False, is_land=False),
        mana_cost=SimpleNamespace(mana_value=mv),
    )


def blocked(attacker: int, blocker: int | None = None) -> SimpleNamespace:
    """An attacker-declared event, or a blocker-declared one when blocked."""
    if blocker is None:
        return SimpleNamespace(
            event_type=EventTypeEnum.COMBAT_ATTACKERS_DECLARED,
            source_id=attacker,
            target_id=0,
        )
    return SimpleNamespace(
        event_type=EventTypeEnum.BLOCKERS_DECLARED,
        source_id=attacker,
        target_id=blocker,
    )


def act(
    kind: ActionSpaceEnum,
    actions: list[SimpleNamespace],
    *,
    mine: list[tuple[SimpleNamespace, SimpleNamespace]] = (),
    theirs: list[tuple[SimpleNamespace, SimpleNamespace]] = (),
    cards: list[SimpleNamespace] = (),
    focus: list[int] = (),
    life: tuple[int, int] = (20, 20),
    step: StepEnum = StepEnum.PRECOMBAT_MAIN_STEP,
    stack: list[SimpleNamespace] = (),
    events: list[SimpleNamespace] = (),
    turn: int = 3,
    player: ScriptedGreedyPlayer | None = None,
) -> int:
    raw = SimpleNamespace(
        agent=SimpleNamespace(id=OWN, life=life[0]),
        opponent=SimpleNamespace(id=OPPONENT, life=life[1]),
        agent_cards=[c for _, c in mine] + list(cards),
        opponent_cards=[c for _, c in theirs],
        agent_permanents=[p for p, _ in mine],
        opponent_permanents=[p for p, _ in theirs],
        stack_objects=list(stack),
        recent_events=list(events),
        turn=SimpleNamespace(step=step, turn_number=turn),
        action_space=SimpleNamespace(
            action_space_type=kind, focus=list(focus), actions=actions
        ),
    )
    env = cast(Env, SimpleNamespace(last_raw_obs=raw))
    return (player or HeuristicPlayer()).act(env, {})


def offer(
    action: ActionEnum, *focus: int, declared: bool | None = None
) -> SimpleNamespace:
    return SimpleNamespace(action_type=action, focus=list(focus), declared=declared)


# Card knowledge -------------------------------------------------------------


def test_polarity_is_derived_for_every_targeted_spell_in_the_pool() -> None:
    spells = targeted_spells()
    assert {name: s.polarity for name, s in spells.items()} == {
        "Firebending Lesson": "harmful",
        "Igneous Inspiration": "harmful",
        "Divide by Zero": "harmful",
        "It'll Quench Ya!": "harmful",
        "Allies at Last": "harmful",
        "Yip Yip!": "beneficial",
        "Fancy Footwork": "beneficial",
        "Lightning Bolt": "harmful",
    }
    assert spells["Lightning Bolt"].damage == 3
    fire = spells["Firebending Lesson"]
    assert (fire.damage, fire.kicked_damage) == (2, 5)
    assert spells["Fancy Footwork"].pump == (2, 2)
    assert spells["Allies at Last"].side_restricted
    assert not spells["Fancy Footwork"].side_restricted


def test_unknown_or_mixed_target_ops_have_no_polarity() -> None:
    from manabot.sim.heuristic import derive_targeted_spells

    def source(*ops: dict[str, Any]) -> dict[str, Any]:
        program = {
            "kind": "spell",
            "targets": [{"role": "t", "selector": {"kind": "creature"}}],
            "ops": [dict(op, target="t") for op in ops],
        }
        return {"definitions": [{"registry_name": "X", "programs": [program]}]}

    assert derive_targeted_spells(source({"op": "tap"}))["X"].polarity is None
    mixed = source({"op": "deal_damage", "amount": 1}, {"op": "untap"})
    assert derive_targeted_spells(mixed)["X"].polarity is None
    shrink = source({"op": "modify_pt", "power": -2, "toughness": -2})
    assert derive_targeted_spells(shrink)["X"] == TargetedSpell(
        "harmful", "creature", False, pump=(-2, -2)
    )


# Priority -------------------------------------------------------------------


def test_casts_highest_mana_value_and_holds_removal_without_a_kill() -> None:
    bolt, bear, giant = (
        hand(1, "Firebending Lesson", 1),
        hand(2, "Bear", 2),
        hand(3, "Giant", 4),
    )
    actions = [
        offer(ActionEnum.PRIORITY_CAST_SPELL, 1),
        offer(ActionEnum.PRIORITY_CAST_SPELL, 2),
        offer(ActionEnum.PRIORITY_CAST_SPELL, 3),
        offer(ActionEnum.PRIORITY_PASS_PRIORITY),
    ]
    assert act(ActionSpaceEnum.PRIORITY, actions, cards=[bolt, bear, giant]) == 2
    only_bolt = [actions[0], actions[3]]
    wall = creature(50, OPPONENT, 0, 8)
    assert act(ActionSpaceEnum.PRIORITY, only_bolt, cards=[bolt], theirs=[wall]) == 1
    bear_body = creature(51, OPPONENT, 2, 2)
    assert (
        act(ActionSpaceEnum.PRIORITY, only_bolt, cards=[bolt], theirs=[bear_body]) == 0
    )


def test_counter_waits_for_an_opponent_spell() -> None:
    quench = hand(1, "It'll Quench Ya!", 2)
    actions = [
        offer(ActionEnum.PRIORITY_CAST_SPELL, 1),
        offer(ActionEnum.PRIORITY_PASS_PRIORITY),
    ]
    assert act(ActionSpaceEnum.PRIORITY, actions, cards=[quench]) == 1
    theirs = SimpleNamespace(kind=0, controller_id=OPPONENT)
    assert act(ActionSpaceEnum.PRIORITY, actions, cards=[quench], stack=[theirs]) == 0


# Targets --------------------------------------------------------------------


def test_burn_kills_the_biggest_creature_it_can_else_goes_face() -> None:
    source = hand(1, "Igneous Inspiration", 3)
    small, big, huge = (
        creature(30, OPPONENT, 1, 1),
        creature(31, OPPONENT, 3, 3),
        creature(32, OPPONENT, 6, 6),
    )
    own = creature(33, OWN, 5, 3)
    targets = [OWN, OPPONENT, 33, 30, 31, 32]
    actions = [offer(ActionEnum.CHOOSE_TARGET, t) for t in targets]
    board = dict(mine=[own], theirs=[small, big, huge], cards=[source], focus=[1])
    assert targets[act(ActionSpaceEnum.CHOOSE_TARGET, actions, **board)] == 31
    face = dict(board, theirs=[huge])
    remaining = [a for a in actions if a.focus[0] in (OWN, OPPONENT, 33, 32)]
    picked = remaining[act(ActionSpaceEnum.CHOOSE_TARGET, remaining, **face)]
    assert picked.focus == [OPPONENT]


def test_pump_targets_own_side_and_side_restricted_spells_use_v1() -> None:
    footwork = hand(1, "Fancy Footwork", 3)
    mine, theirs = creature(30, OWN, 2, 2, tapped=True), creature(31, OPPONENT, 4, 4)
    actions = [offer(ActionEnum.CHOOSE_TARGET, 31), offer(ActionEnum.CHOOSE_TARGET, 30)]
    board = dict(mine=[mine], theirs=[theirs], cards=[footwork], focus=[1])
    assert act(ActionSpaceEnum.CHOOSE_TARGET, actions, **board) == 1
    allies = dict(board, cards=[hand(1, "Allies at Last", 3)])
    assert act(ActionSpaceEnum.CHOOSE_TARGET, actions, **allies) == 0


# Combat ---------------------------------------------------------------------


def attack_prompt(pid: int) -> list[SimpleNamespace]:
    return [
        offer(ActionEnum.DECLARE_ATTACKER, pid, declared=True),
        offer(ActionEnum.DECLARE_ATTACKER, pid, declared=False),
    ]


@pytest.mark.parametrize(
    ("blocker", "attacks"),
    [
        (creature(40, OPPONENT, 3, 4), False),  # kills and survives
        (creature(40, OPPONENT, 3, 3), True),  # trade only
        (creature(40, OPPONENT, 3, 4, tapped=True), True),  # cannot block
        (creature(40, OPPONENT, 1, 4, deathtouch=True), False),
        (creature(40, OPPONENT, 3, 1, first_strike=True), False),
    ],
)
def test_attacks_unless_a_blocker_dominates(blocker: Any, attacks: bool) -> None:
    attacker = creature(30, OWN, 3, 3)
    choice = act(
        ActionSpaceEnum.DECLARE_ATTACKER,
        attack_prompt(30),
        mine=[attacker],
        theirs=[blocker],
    )
    assert choice == (0 if attacks else 1)


def test_flying_evades_and_lethal_attacks_into_dominating_blockers() -> None:
    flyer = creature(30, OWN, 2, 2, flying=True)
    wall = creature(40, OPPONENT, 0, 5)
    ogre = creature(41, OPPONENT, 5, 5)
    prompt = attack_prompt(31)
    grunt = creature(31, OWN, 3, 3)
    board = dict(mine=[flyer, grunt], theirs=[wall, ogre])
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, prompt, **board) == 1
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, prompt, life=(20, 2), **board) == 0


def block_prompt(blocker: int, *attackers: int) -> list[SimpleNamespace]:
    return [
        offer(ActionEnum.DECLARE_BLOCKER, blocker, a, declared=True) for a in attackers
    ] + [offer(ActionEnum.DECLARE_BLOCKER, blocker, declared=False)]


def test_blocks_good_then_trade_and_chumps_only_against_lethal() -> None:
    blocker = creature(30, OWN, 2, 2, mv=2)
    small, big = creature(40, OPPONENT, 1, 1), creature(41, OPPONENT, 2, 2, mv=3)
    board = dict(mine=[blocker], theirs=[small, big])
    assert act(ActionSpaceEnum.DECLARE_BLOCKER, block_prompt(30, 41, 40), **board) == 1
    assert act(ActionSpaceEnum.DECLARE_BLOCKER, block_prompt(30, 41), **board) == 0
    ogre = creature(42, OPPONENT, 5, 5)
    chump = dict(mine=[blocker], theirs=[ogre])
    assert act(ActionSpaceEnum.DECLARE_BLOCKER, block_prompt(30, 42), **chump) == 1
    assert (
        act(
            ActionSpaceEnum.DECLARE_BLOCKER, block_prompt(30, 42), life=(4, 20), **chump
        )
        == 0
    )


def test_pays_optional_costs() -> None:
    actions = [offer(ActionEnum.DECLINE_CHOICE), offer(ActionEnum.PAY_COST)]
    assert act(ActionSpaceEnum.PAY_OR_NOT, actions) == 1
    assert act(ActionSpaceEnum.PAY_OR_NOT, actions, player=ScriptedGreedyPlayer()) == 1


# Identity -------------------------------------------------------------------


def test_v1_monitor_identity_is_unchanged_and_heuristic_has_its_own() -> None:
    from pathlib import Path

    from manabot.infra.identity import file_sha256
    from manabot.sim import players as players

    assert (
        MonitorProtocol().model_dump()
        == MonitorProtocol(opponent="scripted_greedy").model_dump()
    )
    assert "opponent" not in MonitorProtocol().model_dump()
    assert _opponent_source_sha256("scripted_greedy") == file_sha256(
        Path(players.__file__)
    )
    assert _opponent_source_sha256("heuristic") != _opponent_source_sha256(
        "scripted_greedy"
    )
    assert all(path.exists() for path in heuristic_source_paths())


# Kicker, attack pool and own tricks ------------------------------------------


def test_kicker_decides_removal_reach_and_target() -> None:
    fire = hand(1, "Firebending Lesson", 1, kicker=4)
    cast_it = [
        offer(ActionEnum.PRIORITY_CAST_SPELL, 1),
        offer(ActionEnum.PRIORITY_PASS_PRIORITY),
    ]
    ogre = creature(50, OPPONENT, 4, 4)
    poor = dict(cards=[fire], theirs=[ogre], mine=lands(2))
    assert act(ActionSpaceEnum.PRIORITY, cast_it, **poor) == 1
    rich = dict(cards=[fire], theirs=[ogre], mine=lands(5))
    assert act(ActionSpaceEnum.PRIORITY, cast_it, **rich) == 0

    pay = [offer(ActionEnum.PAY_COST), offer(ActionEnum.DECLINE_CHOICE)]
    targets = [offer(ActionEnum.CHOOSE_TARGET, 51), offer(ActionEnum.CHOOSE_TARGET, 50)]
    board = dict(cards=[fire], theirs=[ogre, creature(51, OPPONENT, 2, 2)], focus=[1])
    kicked = HeuristicPlayer()
    assert (
        act(ActionSpaceEnum.PAY_OR_NOT, pay, mine=lands(5), player=kicked, **board) == 0
    )
    spent = lands(5, tapped=True)
    assert (
        act(ActionSpaceEnum.CHOOSE_TARGET, targets, mine=spent, player=kicked, **board)
        == 1
    )
    unkicked = HeuristicPlayer()
    assert (
        act(ActionSpaceEnum.PAY_OR_NOT, pay, mine=lands(2), player=unkicked, **board)
        == 1
    )
    assert (
        act(
            ActionSpaceEnum.CHOOSE_TARGET,
            targets,
            mine=lands(2),
            player=unkicked,
            **board,
        )
        == 0
    )


def test_lethal_counts_attackers_already_declared_and_tapped() -> None:
    a, b = creature(30, OWN, 3, 3), creature(31, OWN, 3, 3)
    flyer = creature(40, OPPONENT, 5, 5, flying=True)
    guarded = dict(mine=[a, b], theirs=[flyer], life=(20, 6))
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, attack_prompt(30), **guarded) == 1
    player = HeuristicPlayer()
    open_board = dict(theirs=[creature(40, OPPONENT, 5, 5, tapped=True)], life=(20, 6))
    assert (
        act(
            ActionSpaceEnum.DECLARE_ATTACKER,
            attack_prompt(30),
            mine=[a, b],
            player=player,
            **open_board,
        )
        == 0
    )
    # The first attacker is now tapped; the pool still counts it toward lethal.
    a_tapped = creature(30, OWN, 3, 3, tapped=True)
    blocker = creature(41, OPPONENT, 5, 5, tapped=True)
    later = dict(open_board, theirs=[blocker])
    assert (
        act(
            ActionSpaceEnum.DECLARE_ATTACKER,
            attack_prompt(31),
            mine=[a_tapped, b],
            player=player,
            **later,
        )
        == 0
    )


def test_pump_in_hand_lets_one_dominated_creature_attack() -> None:
    yip = hand(1, "Yip Yip!", 1)
    grunt, other = creature(30, OWN, 3, 3), creature(31, OWN, 3, 3)
    blocker = creature(40, OPPONENT, 3, 4)
    player = HeuristicPlayer()
    board = dict(
        mine=[grunt, other, *lands(1)], theirs=[blocker], cards=[yip], player=player
    )
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, attack_prompt(30), **board) == 0
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, attack_prompt(31), **board) == 1
    no_mana = dict(board, mine=[grunt, other], player=HeuristicPlayer())
    assert act(ActionSpaceEnum.DECLARE_ATTACKER, attack_prompt(30), **no_mana) == 1


def test_pump_after_blocks_only_where_it_changes_the_result() -> None:
    yip = hand(1, "Yip Yip!", 1)
    cast_it = [
        offer(ActionEnum.PRIORITY_CAST_SPELL, 1),
        offer(ActionEnum.PRIORITY_PASS_PRIORITY),
    ]
    attacker = creature(30, OWN, 3, 3, tapped=True)
    step = StepEnum.COMBAT_DECLARE_BLOCKERS
    flips = creature(40, OPPONENT, 3, 4)
    board = dict(mine=[attacker], cards=[yip], step=step, events=[blocked(30, 40)])
    assert act(ActionSpaceEnum.PRIORITY, cast_it, theirs=[flips], **board) == 0
    hopeless = creature(40, OPPONENT, 9, 9)
    assert act(ActionSpaceEnum.PRIORITY, cast_it, theirs=[hopeless], **board) == 1
    early = dict(board, step=StepEnum.PRECOMBAT_MAIN_STEP)
    assert act(ActionSpaceEnum.PRIORITY, cast_it, theirs=[flips], **early) == 1
    lethal = dict(board, events=[blocked(30)], life=(20, 5))
    assert act(ActionSpaceEnum.PRIORITY, cast_it, theirs=[hopeless], **lethal) == 0
