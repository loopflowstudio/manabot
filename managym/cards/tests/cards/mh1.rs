//! Tests for cards filed in `src/mh1.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{find_owned_card, has_action, offered_cards, offered_targets, zone_of},
    play::{activate, cast, choose, choose_mode, settle, target_opponent},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        mana::Mana,
        zone::ZoneType,
    },
};

fn casts(card: managym::state::game_object::CardId) -> impl Fn(&Action) -> bool {
    move |action| matches!(action, Action::CastSpell { card: c, .. } if *c == card)
}

/// Outside its caster's turn and without mana, Force of Negation is cast by
/// exiling another blue card; the countered noncreature spell is exiled.
/// A creature spell is never a legal target.
#[test]
fn force_of_negation_exiles_a_blue_card_and_the_countered_spell() {
    let mut s = five_color_setup(&[
        "Force of Negation",
        "Counterspell",
        "Lightning Bolt",
        "Gray Ogre",
    ]);
    s.set_player_mana_pool(1, "");
    s.force_card_in_hand(1, "Force of Negation");
    s.force_card_in_hand(1, "Counterspell");
    s.force_card_in_hand(1, "Lightning Bolt");
    let force = find_owned_card(&s, 1, "Force of Negation");
    let counterspell = find_owned_card(&s, 1, "Counterspell");
    let bolt = cast(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    s.pass_priority();
    choose(&mut s, casts(force));
    // Unpayable mana: the alternative is used without asking.
    assert_eq!(offered_targets(&s), vec![Target::StackSpell(bolt)]);
    assert!(s.choose_target(Target::StackSpell(bolt)));
    // Lightning Bolt is red: only Counterspell can be exiled, so the single
    // choice is skipped.
    settle(&mut s);
    assert_eq!(zone_of(&s, counterspell), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, force), Some(ZoneType::Graveyard));
    s.assert_life(1, 20);

    // A creature spell can't be its target.
    let mut s = five_color_setup(&["Force of Negation", "Counterspell", "Gray Ogre"]);
    s.force_card_in_hand(1, "Force of Negation");
    let force = find_owned_card(&s, 1, "Force of Negation");
    cast(&mut s, "Gray Ogre");
    assert!(!has_action(&s, casts(force)));
}

/// With both costs payable the caster chooses; paying the cost uses the
/// alternative, and the exiled card is chosen among blue cards in hand. On
/// the caster's own turn only the mana cost is offered.
#[test]
fn force_of_negation_offers_its_alternative_only_outside_your_turn() {
    let mut s = five_color_setup(&[
        "Force of Negation",
        "Counterspell",
        "Island",
        "Visions of Beyond",
    ]);
    s.force_card_in_hand(1, "Force of Negation");
    s.force_card_in_hand(1, "Counterspell");
    s.force_card_in_hand(1, "Visions of Beyond");
    s.force_card_in_hand(1, "Island");
    let force = find_owned_card(&s, 1, "Force of Negation");
    let visions = find_owned_card(&s, 1, "Visions of Beyond");
    let theirs = cast(&mut s, "Visions of Beyond");
    s.pass_priority();
    choose(&mut s, casts(force));
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    choose(&mut s, |action| matches!(action, Action::PayCost { .. }));
    assert!(s.choose_target(Target::StackSpell(theirs)));
    let mut exiled = offered_cards(&s);
    exiled.sort();
    let mut blue = vec![find_owned_card(&s, 1, "Counterspell"), visions];
    blue.sort();
    assert_eq!(exiled, blue);
    choose(
        &mut s,
        |action| matches!(action, Action::SelectCard { card, .. } if *card == visions),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, visions), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, theirs), Some(ZoneType::Exile));
    assert_eq!(s.game().state.players[1].mana_pool.total(), 40);

    // Its own turn, no mana: not castable even with a blue card in hand.
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "");
    s.force_card_in_hand(1, "Force of Negation");
    let force = find_owned_card(&s, 1, "Force of Negation");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert!(!has_action(&s, casts(force)));
}

#[test]
fn sunbaked_canyon_pays_life_for_red_or_white_and_cycles_itself() {
    let mut s = five_color_setup(&["Sunbaked Canyon"]);
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    s.set_player_mana_pool(0, "");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, canyon, 0);
    choose_mode(&mut s, 1);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("W"));
    s.assert_life(0, 19);

    let mut s = five_color_setup(&["Sunbaked Canyon"]);
    let canyon = s.force_permanent_on_battlefield(0, "Sunbaked Canyon");
    s.game_mut().scenario_refresh_priority().unwrap();
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, canyon, 1);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    assert!(s.game().state.permanents[canyon].is_none());
    s.assert_life(0, 20);
}
