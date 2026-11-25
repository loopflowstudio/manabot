//! Tests for cards filed in `src/m13.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{activate, settle},
};

/// Krenko counts every Goblin its controller controls, itself included, as
/// the ability resolves: with one other Goblin it makes two, then four.
#[test]
fn krenko_makes_a_goblin_for_each_goblin_you_control() {
    let mut s = five_color_setup(&["Krenko, Mob Boss", "Fanatical Firebrand"]);
    let krenko = s.force_permanent_on_battlefield(0, "Krenko, Mob Boss");
    s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    s.force_permanent_on_battlefield(1, "Fanatical Firebrand");
    clear_summoning_sickness(&mut s, krenko);
    activate(&mut s, krenko, 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Goblin").len(), 2);
    s.game_mut().state.permanents[krenko]
        .as_mut()
        .unwrap()
        .tapped = false;
    activate(&mut s, krenko, 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Goblin").len(), 6);
}

/// Crusader of Odric's power and toughness are each the number of creatures
/// its controller controls (CR 604.3): on the battlefield, counting itself,
/// and in hand, read by the policy input's card row as well.
#[test]
fn crusader_of_odric_counts_creatures_on_and_off_the_battlefield() {
    use crate::support::game::force_new_card_in_hand;
    use managym::{agent::observation::Observation, state::game_object::PlayerId};
    let mut s = five_color_setup(&["Crusader of Odric", "Gray Ogre"]);
    let crusader = s.force_permanent_on_battlefield(0, "Crusader of Odric");
    assert_eq!(s.game().characteristics(crusader).pt(), (1, 1));
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(s.game().characteristics(crusader).pt(), (3, 3));
    let in_hand = force_new_card_in_hand(&mut s, 0, "Crusader of Odric");
    assert_eq!(s.game().object_pt(in_hand), (3, 3));
    let view = Observation::for_player(s.game(), PlayerId(0));
    let row = view
        .agent_cards
        .iter()
        .find(|row| row.id == s.game().state[in_hand].id.0 as i32)
        .unwrap();
    assert_eq!(
        (row.characteristics.power, row.characteristics.toughness),
        (3, 3)
    );
}
