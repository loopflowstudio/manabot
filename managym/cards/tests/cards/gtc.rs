//! Tests for cards filed in `src/gtc.rs`, in card order.

use crate::support::play::tapped_dual_land_pays_either_color;
use managym::agent::action::ActionSpaceKind;

/// Gatecrash's five Guildgates enter tapped, do nothing else as they
/// enter, are Gates and pay either of their two colors, and no other.
#[test]
fn guildgates_enter_tapped_and_supply_either_color() {
    for (name, colors) in [
        ("Boros Guildgate", "RW"),
        ("Dimir Guildgate", "UB"),
        ("Gruul Guildgate", "RG"),
        ("Orzhov Guildgate", "WB"),
        ("Simic Guildgate", "GU"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |s, land| {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(s.game().characteristics(land).subtypes, ["Gate"]);
        });
    }
}

/// Consuming Aberration is as large as its opponents' graveyards, in every
/// zone. Each spell its controller casts has each opponent reveal until a
/// land and put every revealed card into their graveyard at once.
#[test]
fn consuming_aberration_mills_opponents_until_a_land_and_grows() {
    use crate::support::{
        game::{five_color_setup, force_new_card_in_hand, pin_top, plant_in_graveyard},
        inspect::zone_of,
        play::{cast_card, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{game_object::PlayerId, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Consuming Aberration", "Gray Ogre"]);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "3R");
    let aberration = s.force_permanent_on_battlefield(1, "Consuming Aberration");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 1, "Gray Ogre");
    assert_eq!(s.game().characteristics(aberration).pt(), (1, 1));
    let revealed = pin_top(&mut s, &["Gray Ogre", "Gray Ogre", "Plains", "Island"]);
    let graveyard = s.game().state.zones.size(ZoneType::Graveyard, PlayerId(0));
    let spell = force_new_card_in_hand(&mut s, 1, "Gray Ogre");
    cast_card(&mut s, spell);
    settle(&mut s);
    for object in &revealed[..3] {
        assert_eq!(zone_of(&s, *object), Some(ZoneType::Graveyard));
    }
    assert_eq!(zone_of(&s, revealed[3]), Some(ZoneType::Library));
    let graveyard = graveyard + 3;
    assert_eq!(
        s.game().state.zones.size(ZoneType::Graveyard, PlayerId(0)),
        graveyard
    );
    let side = graveyard as i32;
    assert_eq!(s.game().characteristics(aberration).pt(), (side, side));
    let in_hand = force_new_card_in_hand(&mut s, 1, "Consuming Aberration");
    assert_eq!(s.game().object_pt(in_hand), (side, side));
}
