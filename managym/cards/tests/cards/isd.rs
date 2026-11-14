//! Tests for cards filed in `src/isd.rs`, in card order.

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
    s.game_mut().move_object(act, ZoneType::Hand);
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
