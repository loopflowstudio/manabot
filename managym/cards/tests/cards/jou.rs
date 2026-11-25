//! Tests for cards filed in `src/jou.rs`, in card order.

use crate::support::play::temple_scries_one_and_pays_either_color;

/// Journey into Nyx's two Temples enter tapped, scry 1 as they enter (the
/// top card stays or goes to the bottom, as chosen) and pay either of their
/// two colors, and no other.
#[test]
fn temples_enter_tapped_scry_one_and_supply_either_color() {
    for (name, colors) in [("Temple of Epiphany", "UR"), ("Temple of Malady", "BG")] {
        temple_scries_one_and_pays_either_color(name, colors);
    }
}

/// Dictate of Kruphix has flash, and each player draws an additional card in
/// their own draw step, its controller included.
#[test]
fn dictate_of_kruphix_adds_a_draw_to_each_players_draw_step() {
    use crate::support::{
        game::{five_color_setup, force_new_card_in_hand},
        play::{cast_card, castable},
    };
    use managym::{
        flow::turn::StepKind,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Dictate of Kruphix", "Gray Ogre"]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let dictate = force_new_card_in_hand(&mut s, 0, "Dictate of Kruphix");
    cast_card(&mut s, ogre);
    assert!(castable(&mut s, dictate), "flash");
    let mut s = five_color_setup(&["Dictate of Kruphix"]);
    s.force_permanent_on_battlefield(0, "Dictate of Kruphix");
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.zone_size(1, ZoneType::Hand), 2);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(s.zone_size(0, ZoneType::Hand), 2);
}
