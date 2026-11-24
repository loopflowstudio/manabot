//! Tests for cards filed in `src/isd.rs`, in card order.

use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, move_card},
    inspect::{can_cast, owned_card, permanent_of, tapped},
    play::{cast_card, cast_owned, resolve_top, settle},
};
use managym::agent::action::ActionSpaceKind;
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
fn diregraf_ghoul_enters_tapped() {
    let mut s = five_color_setup(&["Diregraf Ghoul", "Typhoid Rats"]);
    for (name, enters_tapped) in [("Diregraf Ghoul", true), ("Typhoid Rats", false)] {
        let spell = cast_owned(&mut s, name);
        resolve_top(&mut s);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
        let creature = permanent_of(&s, spell).expect("creature spell resolved");
        assert_eq!(tapped(&s, creature), enters_tapped, "{name}");
        assert!(s.game().state.stack_objects.is_empty());
    }
}
