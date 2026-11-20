//! Tests for cards filed in `src/atq.rs`, in card order.

use crate::support::{
    game::{five_color_unfunded_setup, plant_in_graveyard},
    inspect::{card_of, top_activation, zone_of},
    play::{activate, settle},
};
use managym::state::{
    payment::{How, Toward},
    zone::ZoneType,
};

/// Feldon's Cane is exiled, not sacrificed, as its ability is activated: the
/// receipt says so, it is not in the graveyard its ability then shuffles into
/// the library, and it stays exiled.
#[test]
fn feldons_cane_is_exiled_as_a_cost_and_shuffles_the_graveyard_into_the_library() {
    let mut s = five_color_unfunded_setup(&["Feldon's Cane", "Gray Ogre", "Hill Giant"]);
    let cane = s.force_permanent_on_battlefield(0, "Feldon's Cane");
    let cane_card = card_of(&s, cane);
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 0, "Hill Giant");
    let library = s.zone_size(0, ZoneType::Library);
    activate(&mut s, cane, 0);
    assert_eq!(zone_of(&s, cane_card), Some(ZoneType::Exile));
    let paid = &top_activation(&s).paid;
    assert_eq!(paid.contributed(How::Exiled, Toward::Printed).count(), 1);
    assert_eq!(paid.contributed(How::Tapped, Toward::Printed).count(), 1);
    assert_eq!(paid.sacrificed().count(), 0);
    s.assert_zone_size(0, ZoneType::Graveyard, 2);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Graveyard, 0);
    s.assert_zone_size(0, ZoneType::Library, library + 2);
    assert_eq!(zone_of(&s, cane_card), Some(ZoneType::Exile));
}
