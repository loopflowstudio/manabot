//! Tests for cards filed in `src/isd.rs`, in card order.

use crate::support::game::move_card;
use crate::support::{
    game::five_color_unfunded_setup,
    inspect::{can_cast, owned_card},
    play::{cast_card, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn blasphemous_act_costs_one_less_per_creature_on_the_battlefield() {
    let mut s = five_color_unfunded_setup(&["Blasphemous Act", "Gray Ogre"]);
    let act = owned_card(&s, 0, "Blasphemous Act");
    let act = move_card(&mut s, act, ZoneType::Hand);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    // Four creatures: {4}{R} against five mana's worth less one.
    s.set_player_mana_pool(0, "3R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, act));
    // Either player's creature reduces the cost.
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, act));
    cast_card(&mut s, act);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Gray Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
    s.assert_zone_size(1, ZoneType::Graveyard, 3);
}

#[test]
fn mentor_of_the_meek_offers_a_draw_for_each_small_creature_entering() {
    use crate::support::{
        game::five_color_setup,
        inspect::library_card,
        play::{choose, pay_rest},
    };
    use managym::agent::action::{Action, ActionSpaceKind};

    let mut s = five_color_setup(&[
        "Mentor of the Meek",
        "Gray Ogre",
        "Craw Wurm",
        "Fire Nation's Conquest",
    ]);
    s.force_permanent_on_battlefield(0, "Mentor of the Meek");
    let enter = |s: &mut crate::support::game::Scenario, name: &str| {
        let card = library_card(s, 0, name);
        s.game_mut().move_object(card, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
        !s.game().state.pending_triggers.is_empty() || !s.game().state.stack_objects.is_empty()
    };
    let hand = s.zone_size(0, ZoneType::Hand);
    assert!(enter(&mut s, "Gray Ogre"));
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::PayOrNot,
        "Mentor offers its payment".into(),
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        pay_rest(&mut s);
    }
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);

    assert!(!enter(&mut s, "Craw Wurm"), "power 6 is too great");
    // Layer 7c: an anthem applies as the creature enters, so a 2/2 enters as
    // a 3/2 and doesn't trigger Mentor.
    s.force_permanent_on_battlefield(0, "Fire Nation's Conquest");
    assert!(!enter(&mut s, "Gray Ogre"));
}
