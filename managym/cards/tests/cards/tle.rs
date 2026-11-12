//! TLE card behavior through ordinary choices and exact Command forks.
use crate::support::{
    game::{clear_summoning_sickness, five_color_unfunded_setup, move_to_hand, Scenario},
    inspect::{assert_choice_contract, can_activate, can_cast, card_of, offered_cards, tapped},
    play::{activate, cast_owned, choose, choose_mode, pass_until, select_card, settle},
};
use managym::{
    agent::{action::{Action, ActionSpaceKind}, observation::Observation, observation_encoder::encode},
    flow::turn::StepKind,
    state::game_object::{PermanentId, PlayerId},
};

fn merchant(foods: usize) -> (Scenario, PermanentId, Vec<PermanentId>) {
    let mut s =
        five_color_unfunded_setup(&["The Cabbage Merchant", "Prophetic Prism", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "The Cabbage Merchant");
    let foods = (0..foods)
        .map(|_| {
            s.game_mut()
                .create_token("Food", PlayerId(0), false)
                .unwrap()
        })
        .collect();
    s.game_mut().scenario_refresh_priority().unwrap();
    (s, source, foods)
}

#[test]
fn merchant_repeats_with_four_foods_and_preserves_exact_tap_commands() {
    let (mut s, source, foods) = merchant(4);
    // This ability has no source {T} cost; sickness and a tapped Merchant do not prevent it.
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .tapped = true;
    let prism = move_to_hand(&mut s, "Prophetic Prism");
    assert!(
        can_cast(&s, prism),
        "the payment needs two uses of the same Merchant"
    );
    for (activation, pair) in foods.chunks_exact(2).enumerate() {
        assert!(can_activate(&s, source));
        activate(&mut s, source, 0);
        assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
        for food in pair {
            assert_choice_contract(&s);
            let card = card_of(&s, *food);
            assert!(offered_cards(&s).contains(&card));
            select_card(&mut s, card);
            assert!(tapped(&s, *food));
            assert!(!offered_cards(&s).contains(&card));
        }
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_choice_contract(&s);
        choose_mode(&mut s, 4);
        assert_eq!(
            s.game().state.players[0].mana_pool.total(),
            (activation + 1) as u16
        );
    }
    assert!(!can_activate(&s, source));
    assert!(can_cast(&s, prism));
}

#[test]
fn merchant_three_foods_fund_only_one_activation() {
    let (mut s, source, foods) = merchant(3);
    let prism = move_to_hand(&mut s, "Prophetic Prism");
    assert!(!can_cast(&s, prism));
    activate(&mut s, source, 0);
    for food in &foods[..2] {
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    choose_mode(&mut s, 4);
    assert!(!can_activate(&s, source));
    assert!(!can_cast(&s, prism));
    assert!(!tapped(&s, foods[2]));
}

#[test]
fn merchant_cannot_reuse_the_food_reserved_for_its_own_activation() {
    let (s, _, foods) = merchant(4);
    assert!(
        !can_activate(&s, foods[0]),
        "its own tap leaves only three Foods for mana"
    );

    let (mut s, source, foods) = merchant(5);
    let payer = foods[0];
    let payer_card = card_of(&s, payer);
    assert!(can_activate(&s, payer));
    activate(&mut s, payer, 0);
    for pair in foods[1..].chunks_exact(2) {
        activate(&mut s, source, 0);
        for food in pair {
            assert!(!offered_cards(&s).contains(&payer_card));
            assert_choice_contract(&s);
            let card = card_of(&s, *food);
            select_card(&mut s, card);
        }
        choose_mode(&mut s, 4);
    }
    choose(&mut s, |action| matches!(action, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 23);
    assert!(s.game().state.permanents[payer].is_none());
    assert!(foods[1..].iter().all(|food| tapped(&s, *food)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn merchant_tap_count_reaches_the_acting_viewer_and_policy() {
    let (mut s, source, foods) = merchant(2);
    activate(&mut s, source, 0);
    for (index, food) in foods.iter().enumerate() {
        for viewer in [PlayerId(0), PlayerId(1)] {
            let observation = Observation::for_player(s.game(), viewer);
            if viewer == PlayerId(0) {
                let pending = observation.continuation.last().unwrap();
                assert_eq!(pending.mode, Some(2 - index));
            } else {
                assert!(observation.continuation.is_empty());
                assert!(observation.action_space.actions.is_empty());
            }
            encode(&observation, &Default::default()).unwrap();
        }
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
}

#[test]
fn merchant_creates_food_only_for_an_opponents_noncreature_spell() {
    let (mut s, _, _) = merchant(0);
    s.force_permanent_on_battlefield(1, "The Cabbage Merchant");
    s.set_player_mana_pool(0, "10RRR");
    cast_owned(&mut s, "Gray Ogre");
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert!(s.battlefield_permanents_named(1, "Food").is_empty());
    cast_owned(&mut s, "Prophetic Prism");
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert_eq!(s.battlefield_permanents_named(1, "Food").len(), 1);
}

#[test]
fn merchant_sacrifices_a_food_for_each_creature_dealing_combat_damage_to_you() {
    let (mut s, _, foods) = merchant(2);
    let enemy_merchant = s.force_permanent_on_battlefield(1, "The Cabbage Merchant");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, enemy_merchant);
    clear_summoning_sickness(&mut s, ogre);
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |action| {
            matches!(action, Action::DeclareAttacker { attack: true, .. })
        });
    }
    s.advance_to_active_step(1, StepKind::DeclareBlockers);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        s.decline_block();
    }
    for food in foods {
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert_choice_contract(&s);
        let card = card_of(&s, food);
        select_card(&mut s, card);
    }
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert_eq!(s.game().state.players[0].life, 16);
}

#[test]
fn merchant_paid_foods_are_not_offered_again_for_waterbending() {
    let mut s = five_color_unfunded_setup(&["The Cabbage Merchant", "Waterbending Lesson"]);
    let source = s.force_permanent_on_battlefield(0, "The Cabbage Merchant");
    let foods = (0..3)
        .map(|_| {
            s.game_mut()
                .create_token("Food", PlayerId(0), false)
                .unwrap()
        })
        .collect::<Vec<_>>();
    s.set_player_mana_pool(0, "3U");
    cast_owned(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    s.set_player_mana_pool(0, "0");
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    activate(&mut s, source, 0);
    for food in &foods[..2] {
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    choose_mode(&mut s, 4);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s.action_space().actions.iter().any(
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if foods[..2].contains(permanent))
    ));
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == foods[2]),
    );
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert!(foods.iter().all(|food| tapped(&s, *food)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}
