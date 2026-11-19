//! Tests for cards filed in `src/ktk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle, spend_colorless, spend_named, tapped_dual_land_pays_either_color},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn prowess_resolves_above_the_noncreature_spell() {
    let mut s = five_color_setup(&["Monastery Swiftspear", "Abandon Attachments"]);
    let swift = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    cast(&mut s, "Abandon Attachments");
    spend_named(&mut s, |label| label == "Spend {U} (expires at step end)");
    spend_colorless(&mut s);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().characteristics(swift).power, 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}

/// The ten gain-lands enter tapped, gain 1 life as they enter and pay either
/// of their two colors, and no other.
#[test]
fn gain_lands_enter_tapped_gain_one_life_and_supply_either_color() {
    let lands = [
        ("Bloodfell Caves", "BR"),
        ("Blossoming Sands", "GW"),
        ("Dismal Backwater", "UB"),
        ("Jungle Hollow", "BG"),
        ("Rugged Highlands", "RG"),
        ("Scoured Barrens", "WB"),
        ("Swiftwater Cliffs", "UR"),
        ("Thornwood Falls", "GU"),
        ("Tranquil Cove", "WU"),
        ("Wind-Scarred Crag", "RW"),
    ];
    for (name, colors) in lands {
        tapped_dual_land_pays_either_color(name, colors, |s, _| {
            settle(s);
            s.assert_life(0, 21);
        });
    }
}
