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
