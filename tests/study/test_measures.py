from manabot.study import measures as m
from manabot.study.record import (
    ATTACK,
    BLOCK,
    CAST_SPELL,
    PASS_PRIORITY,
    PLAY_LAND,
    Decision,
    GameRecord,
    Permanent,
)


def creature(name="Bear", *, tapped=False, keywords=()):
    return Permanent(name, 2, 2, tapped, False, True, False, tuple(keywords))


def decision(turn, actor, chosen, *, active=None, offered=(), kind="PRIORITY", **more):
    fields = dict(
        turn=turn,
        active=actor if active is None else active,
        actor=actor,
        phase="PRECOMBAT_MAIN",
        step="PRECOMBAT_MAIN_STEP",
        kind=kind,
        offered=tuple(sorted({chosen, *offered})),
        offer_labels=("Cast Bear",) if CAST_SPELL in (chosen, *offered) else (),
        chosen=chosen,
        chosen_index=0,
        label=chosen,
        declared=None,
        subject=None,
        life=(20, 20),
        hand=5,
        mine=(),
        theirs=(),
    )
    return Decision(**{**fields, **more})


def game(decisions, *, labels=("bot", "other"), winner=0, first=0, end="terminal"):
    return GameRecord(
        game_id="g",
        labels=labels,
        decks=("ur_lessons", "gw_allies"),
        seed=1,
        first_player=first,
        winner=winner,
        end=end,
        turns=max((d.turn for d in decisions), default=0),
        decisions=list(decisions),
    )


def test_land_drop_counts_each_own_turn_once():
    record = game(
        [
            decision(1, 0, PASS_PRIORITY, offered=[PLAY_LAND]),
            decision(1, 0, PLAY_LAND),
            decision(3, 0, PASS_PRIORITY, offered=[PLAY_LAND]),
            decision(5, 0, PASS_PRIORITY),
            decision(2, 0, PASS_PRIORITY, active=1, offered=[PLAY_LAND]),
        ]
    )
    rate = m.land_drops([record], "bot")
    assert (rate.hits, rate.total) == (1, 2)
    assert [miss.turn for miss in rate.misses] == [3]


def test_open_attack_requires_no_untapped_blocker():
    attack = dict(kind=ATTACK, subject="Bear", mine=(creature(),))
    record = game(
        [
            decision(3, 0, ATTACK, declared=True, theirs=(), **attack),
            decision(
                5, 0, ATTACK, declared=False, theirs=(creature(tapped=True),), **attack
            ),
            decision(7, 0, ATTACK, declared=False, theirs=(creature(),), **attack),
        ]
    )
    assert (
        m.open_attacks([record], "bot").hits,
        m.open_attacks([record], "bot").total,
    ) == (1, 2)
    assert m.contested_attacks([record], "bot").total == 1
    assert m.attacks([record], "bot").total == 3


def test_ground_creature_does_not_block_a_flyer():
    flyer = creature("Hawk", keywords=["flying"])
    record = game(
        [
            decision(
                3,
                0,
                ATTACK,
                kind=ATTACK,
                declared=True,
                subject="Hawk",
                mine=(flyer,),
                theirs=(creature(),),
            ),
            decision(
                5,
                0,
                ATTACK,
                kind=ATTACK,
                declared=True,
                subject="Hawk",
                mine=(flyer,),
                theirs=(creature(keywords=["reach"]),),
            ),
        ]
    )
    assert m.open_attacks([record], "bot").total == 1
    assert m.contested_attacks([record], "bot").total == 1


def test_blocks_and_casting():
    record = game(
        [
            decision(2, 0, BLOCK, active=1, kind=BLOCK, declared=True),
            decision(2, 0, BLOCK, active=1, kind=BLOCK, declared=False),
            decision(3, 0, CAST_SPELL),
            decision(5, 0, PASS_PRIORITY, offered=[CAST_SPELL]),
        ]
    )
    blocks = m.blocks([record], "bot")
    assert (blocks.hits, blocks.total) == (1, 2)
    casting = m.casting([record], "bot")
    assert (casting.hits, casting.total) == (1, 2)
    assert casting.misses[0].detail == "Cast Bear"


def test_win_rates_split_a_mirror_by_deck_and_position():
    games = [
        game([], labels=("bot", "bot"), winner=0, first=0),
        game([], labels=("bot", "bot"), winner=1, first=0),
        game([], labels=("bot", "bot"), winner=None, first=1),
        game([], labels=("bot", "bot"), winner=0, first=0, end="crash"),
    ]
    rates = m.win_rates(games, "bot")
    assert rates[("ur_lessons", "play")].hits == 1
    assert rates[("ur_lessons", "play")].total == 2
    assert rates[("gw_allies", "play")].hits == 0.5
    assert sum(rate.total for rate in rates.values()) == 6
    assert m.outcomes(games) == {"terminal": 3, "crash": 1}


def test_only_the_labelled_player_is_measured():
    record = game([decision(1, 1, PASS_PRIORITY, offered=[PLAY_LAND])])
    assert m.land_drops([record], "bot").total == 0
    assert m.land_drops([record], "other").total == 1


def test_wilson_interval_stays_inside_the_unit_range():
    assert m.Rate(0, 0).interval() is None
    low, high = m.Rate(10, 10).interval()
    assert 0.65 < low < 0.75 and high == 1.0
    low, high = m.Rate(5, 10).interval()
    assert round(low, 2) == 0.24 and round(high, 2) == 0.76


def test_breakdowns_by_lands_in_play_and_by_creature():
    land = Permanent("Island", 0, 0, False, False, False, True)
    record = game(
        [
            decision(1, 0, PLAY_LAND),
            decision(3, 0, PASS_PRIORITY, offered=[PLAY_LAND], mine=(land,) * 3),
            decision(
                5,
                0,
                ATTACK,
                kind=ATTACK,
                declared=False,
                subject="Bear",
                mine=(creature(),),
            ),
        ]
    )
    rates = m.land_drops_by_lands([record], "bot")
    assert (
        rates["0 to 2 lands in play"].hits,
        rates["0 to 2 lands in play"].total,
    ) == (1, 1)
    assert (rates["3 to 4"].hits, rates["3 to 4"].total) == (0, 1)
    assert m.held_back([record], "bot")["Bear"].total == 1
