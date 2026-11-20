//! Tests for cards filed in `src/rtr.rs`, in card order.

use crate::support::play::tapped_dual_land_pays_either_color;
use managym::agent::action::ActionSpaceKind;

/// Return to Ravnica's five Guildgates enter tapped, do nothing else as they
/// enter, are Gates and pay either of their two colors, and no other.
#[test]
fn guildgates_enter_tapped_and_supply_either_color() {
    for (name, colors) in [
        ("Azorius Guildgate", "WU"),
        ("Golgari Guildgate", "BG"),
        ("Izzet Guildgate", "UR"),
        ("Rakdos Guildgate", "BR"),
        ("Selesnya Guildgate", "GW"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |s, land| {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(s.game().characteristics(land).subtypes, ["Gate"]);
        });
    }
}
