//! Tests for cards filed in `src/m10.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, force_new_card_in_hand, until_end_of_turn},
    inspect::counters_of,
    play::{cast_card, pass_until, settle, tap_for_mana},
};
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        effect::Modification,
        game_object::{PlayerId, Target},
        permanent::CounterKind,
        player::ManaPayment,
    },
};

/// Elvish Archdruid's mana ability counts the Elves its controller controls
/// as it is activated, itself included, and resolves at once (CR 605.3a).
/// Losing its other abilities keeps that mana ability (CR 605.1a) and ends
/// the lord bonus; losing the Elf type leaves nothing to count.
#[test]
fn elvish_archdruid_adds_green_for_each_elf_off_the_stack() {
    const GREEN: usize = 4;
    for (case, expected_green, token_pt) in [
        ("lord", 3, (2, 2)),
        ("nonmana abilities lost", 3, (1, 1)),
        ("no elf type", 2, (2, 2)),
    ] {
        let mut s = five_color_setup(&["Elvish Archdruid"]);
        s.allow_priority_floats();
        let druid = s.force_permanent_on_battlefield(0, "Elvish Archdruid");
        clear_summoning_sickness(&mut s, druid);
        let elves: Vec<_> = (0..2)
            .map(|_| {
                s.game_mut()
                    .create_token("token.green_elf_warrior", PlayerId(0), false)
                    .unwrap()
            })
            .collect();
        match case {
            "nonmana abilities lost" => {
                until_end_of_turn(&mut s, druid, Modification::LoseNonManaAbilities)
            }
            "no elf type" => until_end_of_turn(
                &mut s,
                druid,
                Modification::SetCreatureTypes(vec!["Druid".into()]),
            ),
            _ => {}
        }
        assert_eq!(
            s.game().characteristics(elves[0]).pt(),
            token_pt,
            "{case}: lord bonus"
        );
        assert_eq!(
            s.game().characteristics(druid).activated_abilities.len(),
            1,
            "{case}: the mana ability stays"
        );
        s.set_player_mana_pool(0, "");
        s.game_mut().scenario_refresh_priority().unwrap();
        tap_for_mana(&mut s, druid);
        assert!(
            s.game().state.stack_objects.is_empty(),
            "{case}: off the stack"
        );
        assert_eq!(
            s.game().state.players[0].mana_pool.mana[GREEN],
            expected_green,
            "{case}: green added"
        );
    }
}

/// Mold Adder may grow when an opponent casts a blue or black spell: the
/// controller accepts or declines, a blue-black spell triggers it once, and a
/// red spell or its controller's own blue spell not at all.
#[test]
fn mold_adder_may_grow_when_an_opponent_casts_a_blue_or_black_spell() {
    let mut s = five_color_setup(&[
        "Mold Adder",
        "Consider",
        "Foggy Swamp Spirit Keeper",
        "Lightning Bolt",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let adder = s.force_permanent_on_battlefield(0, "Mold Adder");
    let own = force_new_card_in_hand(&mut s, 0, "Consider");
    cast_card(&mut s, own);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    let consider = force_new_card_in_hand(&mut s, 1, "Consider");
    cast_card(&mut s, consider);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    let pending = s.game().clone();
    for accept in [false, true] {
        *s.game_mut() = pending.clone();
        assert!(s.take_action_by_type(if accept {
            ActionType::PayCost
        } else {
            ActionType::DeclineChoice
        }));
        settle(&mut s);
        assert_eq!(
            counters_of(&s, adder, CounterKind::PlusOnePlusOne),
            i32::from(accept)
        );
    }
    let keeper = force_new_card_in_hand(&mut s, 1, "Foggy Swamp Spirit Keeper");
    cast_card(&mut s, keeper);
    assert_eq!(s.game().state.stack_objects.len(), 2, "one trigger");
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    settle(&mut s);
    assert_eq!(counters_of(&s, adder, CounterKind::PlusOnePlusOne), 2);
    let bolt = force_new_card_in_hand(&mut s, 1, "Lightning Bolt");
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}
